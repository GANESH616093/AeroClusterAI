"""PCA compression with measured storage and reconstruction metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import ROW_FEATURE_NAMES


def _compressed_bytes(n, p, k, itemsize):
    """scores (n*k) + components (k*p) + mean (p) + scale (p)."""
    return (n * k + k * p + 2 * p) * itemsize


def compress(X: np.ndarray, features: list[str], n_components: int | None = None,
             variance_target: float = 0.95, store_dtype: str = "float32") -> dict:
    """Fit PCA on standardised X and measure compression honestly.

    Reconstruction is computed from the values *as stored* (cast to store_dtype),
    so quantisation error is included in every reported error metric.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    if n < 3 or p < 2:
        raise ValueError("Need at least 3 rows and 2 features for PCA.")
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    Z = (X - mean) / scale

    full = PCA(n_components=min(n, p), svd_solver="full").fit(Z)
    cum = np.cumsum(full.explained_variance_ratio_)
    k = int(n_components) if n_components else int(np.searchsorted(cum, variance_target) + 1)
    k = max(1, min(k, p))
    comps = full.components_[:k]
    dt = np.dtype(store_dtype)

    scores = Z @ comps.T
    s_st, c_st, m_st, sc_st = (a.astype(dt) for a in (scores, comps, mean, scale))
    Zr = s_st.astype(float) @ c_st.astype(float)
    Xr = Zr * sc_st.astype(float) + m_st.astype(float)

    err = Xr - X
    orig_bytes = n * p * 8
    comp_bytes = _compressed_bytes(n, p, k, dt.itemsize)
    rmse_feat = np.sqrt((err ** 2).mean(axis=0))
    feat_df = pd.DataFrame({
        "feature": [ROW_FEATURE_NAMES.get(f, f) for f in features], "key": features,
        "rmse": rmse_feat, "nrmse_pct_of_std": 100 * rmse_feat / scale,
        "mae": np.abs(err).mean(axis=0)})
    metrics = {
        "n_rows": n, "n_features": p, "n_components": k, "store_dtype": store_dtype,
        "retained_variance": float(cum[k - 1]),
        "original_bytes": orig_bytes, "compressed_bytes": int(comp_bytes),
        "compression_ratio": orig_bytes / comp_bytes,
        "storage_saving_pct": 100 * (1 - comp_bytes / orig_bytes),
        "mse_standardised": float(((Zr - Z) ** 2).mean()),
        "rmse_standardised": float(np.sqrt(((Zr - Z) ** 2).mean())),
        "mean_abs_error_standardised": float(np.abs(Zr - Z).mean()),
    }
    if "east_m" in features and "north_m" in features:
        ie, inn = features.index("east_m"), features.index("north_m")
        d = np.hypot(err[:, ie], err[:, inn])
        metrics.update(pos_err_mean_m=float(d.mean()), pos_err_p95_m=float(np.percentile(d, 95)),
                       pos_err_max_m=float(d.max()))

    # sweep over k with real measurements (float64/float32 as chosen)
    sweep = []
    for kk in range(1, p + 1):
        cc = full.components_[:kk]
        sc = (Z @ cc.T).astype(dt).astype(float) @ cc.astype(dt).astype(float)
        sweep.append({"components": kk, "cumulative_variance": float(cum[kk - 1]),
                      "compression_ratio": orig_bytes / _compressed_bytes(n, p, kk, dt.itemsize),
                      "rmse_standardised": float(np.sqrt(((sc - Z) ** 2).mean()))})
    loadings = pd.DataFrame(comps.T, index=[ROW_FEATURE_NAMES.get(f, f) for f in features],
                            columns=[f"PC{i + 1}" for i in range(k)])
    return {
        "metrics": metrics, "scores": scores, "components": comps, "mean": mean, "scale": scale,
        "recon": Xr, "explained_ratio": full.explained_variance_ratio_, "cumulative": cum,
        "feature_errors": feat_df, "sweep": pd.DataFrame(sweep), "loadings": loadings,
        "stored": {"scores": s_st, "components": c_st, "mean": m_st, "scale": sc_st},
        "features": features,
    }


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.2f} {unit}"
        n /= 1024
