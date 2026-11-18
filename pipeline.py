"""End-to-end analysis pipeline (UI-independent, fully testable)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, precision_recall_fscore_support

from . import anomaly as an
from . import clustering as cl
from .compression import compress
from .config import LABEL_COLUMNS, ROW_FEATURES, WINDOW_FEATURES, EARTH_R
from .data_io import clean_telemetry
from .features import active_features, add_kinematics, build_windows
from .formation import analyse_formation, row_deviation


@dataclass
class Params:
    variance_target: float = 0.95
    n_components: int | None = None          # overrides variance_target when set
    store_dtype: str = "float32"
    window_size: int = 20
    window_step: int = 5
    cluster_variance: float = 0.90
    kmeans_k: int | None = None              # None = auto by silhouette
    dbscan_eps: float | None = None          # None = auto from k-distance
    dbscan_min_samples: int = 8
    anomaly_method: str = "Isolation Forest"
    contamination: float = 0.08
    z_min: float = 3.0
    formation_tolerance_m: float | None = None
    seed: int = 42


def run(raw: pd.DataFrame, params: Params | None = None, origin: str = "unknown") -> dict:
    p = params or Params()
    df, rep = clean_telemetry(raw)
    df = add_kinematics(df)
    labels = [c for c in LABEL_COLUMNS if c in df.columns]

    # ---- row-level PCA compression (labels never enter) ----
    row_feats = active_features(df, ROW_FEATURES)
    comp = compress(df[row_feats].to_numpy(float), row_feats, p.n_components, p.variance_target, p.store_dtype)
    for i in range(comp["scores"].shape[1]):
        df[f"PC{i + 1}"] = comp["scores"][:, i]
    for j, f in enumerate(row_feats):
        df[f"{f}_recon"] = comp["recon"][:, j]

    # ---- formation ----
    form = analyse_formation(df, p.formation_tolerance_m)
    df["formation_dev_m"] = row_deviation(df, form)

    # ---- window behaviour ----
    win = build_windows(df, p.window_size, p.window_step)
    win["formation_dev"] = [df["formation_dev_m"].to_numpy()[ids].mean() for ids in win["row_ids"]]
    wfeats = active_features(win, WINDOW_FEATURES)
    if len(wfeats) < 2:
        raise ValueError("Not enough informative behaviour features after cleaning.")
    Xw, wpca, _ = cl.embed_windows(win, wfeats, p.cluster_variance)
    km = cl.run_kmeans(Xw, p.kmeans_k, seed=p.seed)
    db = cl.run_dbscan(Xw, p.dbscan_eps, p.dbscan_min_samples)
    win["kmeans_cluster"] = km["labels"]
    win["dbscan_cluster"] = db["labels"]
    for i in range(min(3, Xw.shape[1])):
        win[f"WPC{i + 1}"] = Xw[:, i]

    # ---- anomaly detection ----
    afeats = wfeats + (["formation_dev"] if win["formation_dev"].std() > 1e-9 else [])
    members = form.get("members") if form.get("available") else None
    det = an.detect(win, afeats, p.anomaly_method, p.contamination, p.z_min, p.seed, members)
    wa = det["windows"]
    events = an.build_events(det)
    rscore, rflag = an.rows_from_windows(len(df), wa)
    df["anomaly_score"], df["is_anomaly"] = rscore, rflag

    # per-row cluster = cluster of window centred nearest to the row
    for col in ("kmeans_cluster", "dbscan_cluster"):
        df[col] = _row_cluster(df, wa, col)

    result = {
        "origin": origin, "params": asdict(p), "df": df, "windows": wa, "clean_report": rep,
        "compression": comp, "kmeans": km, "dbscan": db, "window_features": wfeats,
        "anomaly_features": afeats, "anomaly": det, "events": events, "formation": form,
        "window_pca": {"scores": Xw, "explained": wpca.explained_variance_ratio_,
                       "loadings": pd.DataFrame(wpca.components_.T, index=wfeats,
                                                columns=[f"WPC{i + 1}" for i in range(wpca.n_components_)])},
        "cluster_desc": {"kmeans": cl.describe_clusters(wa, km["labels"], wfeats),
                         "dbscan": cl.describe_clusters(wa, db["labels"], wfeats)},
        "input_columns_used": {"row_pca": row_feats, "window_clustering": wfeats,
                               "anomaly": afeats, "excluded_labels": labels},
    }
    result["drone_summary"] = drone_summary(df, wa, form)
    result["mission"] = mission_stats(df, wa, events, form)
    result["evaluation"] = evaluate_against_labels(df, wa, p) if labels else None
    return result


def _row_cluster(df, win, col):
    out = np.full(len(df), -2, dtype=int)
    best = np.full(len(df), np.inf)
    for ids, tmid, c, did in zip(win["row_ids"], win["t_mid"], win[col], win["drone_id"]):
        d = np.abs(df["timestamp_s"].to_numpy()[ids] - tmid)
        upd = d < best[ids]
        out[ids[upd]] = c
        best[ids[upd]] = d[upd]
    return out


def path_length_m(g: pd.DataFrame) -> float:
    return float(np.hypot(np.diff(g["east_m"]), np.diff(g["north_m"])).sum())


def drone_summary(df, win, form) -> pd.DataFrame:
    rows = []
    members = set(form.get("members", [])) if form.get("available") else set()
    for did, g in df.groupby("drone_id"):
        w = win[win["drone_id"] == did]
        km_main = w["kmeans_cluster"].mode().iat[0]
        role = ("Formation member" if did in members else
                "Independent / off-formation" if form.get("available") else "n/a")
        rows.append({
            "drone_id": did, "samples": len(g), "duration_s": g["timestamp_s"].max() - g["timestamp_s"].min(),
            "distance_m": path_length_m(g), "mean_speed_mps": g["ground_speed_mps"].mean(),
            "max_speed_mps": g["ground_speed_mps"].max(), "mean_alt_m": g["altitude_m"].mean(),
            "min_alt_m": g["altitude_m"].min(), "max_alt_m": g["altitude_m"].max(),
            "max_abs_vspeed_mps": g["vertical_speed_mps"].abs().max(),
            "battery_used_pct": (g["battery_pct"].iloc[0] - g["battery_pct"].iloc[-1]) if g["battery_pct"].notna().all() else np.nan,
            "anomalous_windows_pct": 100 * w["is_anomaly"].mean(), "anomalous_rows": int(g["is_anomaly"].sum()),
            "dominant_kmeans_cluster": int(km_main), "role": role})
    return pd.DataFrame(rows)


def mission_stats(df, win, events, form) -> dict:
    return {
        "drones": df["drone_id"].nunique(), "samples": len(df),
        "duration_s": float(df["timestamp_s"].max() - df["timestamp_s"].min()),
        "total_distance_km": sum(path_length_m(g) for _, g in df.groupby("drone_id")) / 1000,
        "mean_speed_mps": float(df["ground_speed_mps"].mean()),
        "max_speed_mps": float(df["ground_speed_mps"].max()),
        "mean_altitude_m": float(df["altitude_m"].mean()),
        "max_altitude_m": float(df["altitude_m"].max()),
        "anomaly_events": int(len(events)),
        "high_severity_events": int((events["severity"] == "High").sum()) if len(events) else 0,
        "anomalous_rows_pct": float(100 * df["is_anomaly"].mean()),
        "formation_members": len(form.get("members", [])) if form.get("available") else 0,
        "formation_cohesion_m": form.get("cohesion_mean_m"),
    }


def evaluate_against_labels(df, win, p) -> dict | None:
    """EVALUATION ONLY - labels are compared to outputs, never used for fitting."""
    out = {"caveat": "Labels describe programmed scenarios in SIMULATED data. These numbers are a software "
                     "sanity check, not real-world accuracy."}
    if "is_injected_anomaly" in df:
        y = df["is_injected_anomaly"].astype(int).to_numpy()
        pr, rc, f1, _ = precision_recall_fscore_support(y, df["is_anomaly"].astype(int), average="binary", zero_division=0)
        out["anomaly_rows"] = {"precision": float(pr), "recall": float(rc), "f1": float(f1),
                               "true_anomalous_rows": int(y.sum()), "flagged_rows": int(df["is_anomaly"].sum())}
        per = []
        for did, g in df.groupby("drone_id"):
            if g["is_injected_anomaly"].sum() or g["is_anomaly"].sum():
                per.append({"drone_id": did, "labelled_rows": int(g["is_injected_anomaly"].sum()),
                            "flagged_rows": int(g["is_anomaly"].sum()),
                            "overlap_rows": int(((g["is_injected_anomaly"] == 1) & g["is_anomaly"]).sum())})
        out["per_drone"] = pd.DataFrame(per)
    if "scenario_label" in df:
        lab = df.groupby("drone_id")["scenario_label"].first()
        wl = win["drone_id"].map(lab)
        out["ari_kmeans_vs_scenario"] = float(adjusted_rand_score(wl, win["kmeans_cluster"]))
        out["ari_dbscan_vs_scenario"] = float(adjusted_rand_score(wl, win["dbscan_cluster"]))
    return out
