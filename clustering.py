"""K-Means / DBSCAN flight-pattern clustering with plain-language descriptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import WINDOW_FEATURE_NAMES


def embed_windows(win: pd.DataFrame, feats: list[str], variance: float = 0.90):
    """Standardise window features and project with PCA. Returns (scores, pca, scaler)."""
    Z = StandardScaler().fit(win[feats])
    Zs = Z.transform(win[feats])
    full = PCA().fit(Zs)
    k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), variance) + 1)
    k = max(2, min(k, len(feats), len(win) - 1))
    pca = PCA(n_components=k).fit(Zs)
    return pca.transform(Zs), pca, Z


def _quality(X, labels):
    mask = labels >= 0
    if mask.sum() < 3 or len(set(labels[mask])) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    return {"silhouette": float(silhouette_score(X[mask], labels[mask])),
            "davies_bouldin": float(davies_bouldin_score(X[mask], labels[mask])),
            "calinski_harabasz": float(calinski_harabasz_score(X[mask], labels[mask]))}


def run_kmeans(X, k: int | None = None, k_max: int = 8, seed: int = 42):
    """If k is None, choose k in 2..k_max by best silhouette (computed, not assumed)."""
    n = len(X)
    k_max = min(k_max, n - 1)
    sil = {}
    for kk in range(2, k_max + 1):
        lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(X)
        sil[kk] = float(silhouette_score(X, lab)) if len(set(lab)) > 1 else -1
    if k is None:
        k = max(sil, key=sil.get)
    model = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(X)
    return {"labels": model.labels_, "k": int(k), "silhouette_by_k": sil, "inertia": float(model.inertia_),
            "centers": model.cluster_centers_, "quality": _quality(X, model.labels_), "auto_k": k is None}


def auto_eps(X, min_samples: int, quantile: float = 0.90) -> float:
    nn = NearestNeighbors(n_neighbors=min(min_samples, len(X) - 1) + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.quantile(d[:, -1], quantile))


def run_dbscan(X, eps: float | None = None, min_samples: int = 8):
    used = eps if eps else auto_eps(X, min_samples)
    lab = DBSCAN(eps=used, min_samples=min_samples).fit_predict(X)
    return {"labels": lab, "eps": float(used), "min_samples": min_samples,
            "n_clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),
            "n_noise": int((lab == -1).sum()), "quality": _quality(X, lab)}


def describe_clusters(win: pd.DataFrame, labels, feats: list[str]) -> pd.DataFrame:
    """Plain-language description of every cluster, relative to the whole fleet."""
    w = win.copy()
    w["_c"] = labels
    mu, sd = w[feats].mean(), w[feats].std(ddof=0).replace(0, 1)
    rows = []
    for c, g in w.groupby("_c"):
        z = (g[feats].mean() - mu) / sd
        notable = z[z.abs() >= 0.6].sort_values(key=np.abs, ascending=False).head(3)
        bits = []
        for f, v in notable.items():
            name = WINDOW_FEATURE_NAMES.get(f, f)
            if f == "straightness":
                bits.append("follows a noticeably straighter path" if v > 0 else "follows a more winding path")
            else:
                bits.append(f"{'higher' if v > 0 else 'lower'} {name}")
        if c == -1:
            title = "Noise / outlier windows"
            text = ("DBSCAN could not place these windows in any dense group, so they behave differently "
                    "from the rest of the fleet" + (": " + ", ".join(bits) if bits else "") + ".")
        else:
            title = f"Pattern {c + 1}"
            text = ("Typical fleet behaviour - nothing stands out." if not bits else
                    "Compared with the fleet average this pattern shows " + ", ".join(bits) + ".")
        share = g["drone_id"].value_counts(normalize=True)
        rows.append({"cluster": int(c), "pattern": title, "windows": len(g),
                     "share_pct": 100 * len(g) / len(w),
                     "main_drones": ", ".join(f"{d} ({100 * s:.0f}%)" for d, s in share.head(4).items()),
                     "explanation": text, **{f: g[f].mean() for f in feats}})
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
