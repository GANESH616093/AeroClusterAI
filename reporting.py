"""Exports: processed data, cluster results, compressed payload and mission report."""
from __future__ import annotations

import io
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .compression import fmt_bytes
from .config import ROW_FEATURES


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def processed_frame(res) -> pd.DataFrame:
    df = res["df"]
    drop = [c for c in df.columns if c.endswith("_recon")]
    return df.drop(columns=drop)


def cluster_frame(res) -> pd.DataFrame:
    w = res["windows"].drop(columns=["row_ids"])
    return w.rename(columns={"is_anomaly": "window_is_anomaly"})


def reconstructed_frame(res) -> pd.DataFrame:
    df, comp = res["df"], res["compression"]
    cols = ["drone_id", "timestamp_s"] + [f"{f}_recon" for f in comp["features"]]
    return df[cols].rename(columns={f"{f}_recon": f for f in comp["features"]})


def compressed_npz(res) -> bytes:
    """The actual compressed representation: scores + components + mean + scale (+ ids/times)."""
    c, df = res["compression"], res["df"]
    buf = io.BytesIO()
    np.savez_compressed(buf, scores=c["stored"]["scores"], components=c["stored"]["components"],
                        mean=c["stored"]["mean"], scale=c["stored"]["scale"],
                        features=np.array(c["features"]), drone_id=df["drone_id"].to_numpy().astype(str),
                        timestamp_s=df["timestamp_s"].to_numpy(float))
    return buf.getvalue()


def summary_json(res) -> bytes:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    c = res["compression"]["metrics"]
    out = {"data_origin": res["origin"], "parameters": res["params"], "compression": c, "mission": res["mission"],
           "kmeans": {"k": res["kmeans"]["k"], **res["kmeans"]["quality"]},
           "dbscan": {k: res["dbscan"][k] for k in ("eps", "min_samples", "n_clusters", "n_noise")} | res["dbscan"]["quality"],
           "features_used": res["input_columns_used"], "evaluation": {k: v for k, v in (res["evaluation"] or {}).items() if k != "per_drone"}}
    return json.dumps(out, indent=2, default=conv).encode("utf-8")


def mission_report_md(res) -> str:
    m, c, km, db = res["mission"], res["compression"]["metrics"], res["kmeans"], res["dbscan"]
    ev, ds, form = res["events"], res["drone_summary"], res["formation"]
    sim = "SIMULATED" in res["origin"].upper()
    L = [f"# AeroCluster AI - Mission Report", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
         f"> **Data origin:** {res['origin']}. " + ("This is **simulated** data, not real drone telemetry." if sim else
           "Uploaded data - provenance not verified by this tool."), "",
         "## 1. Mission overview",
         f"- Drones: **{m['drones']}**, samples: **{m['samples']:,}**, duration: **{m['duration_s']:.0f} s**",
         f"- Total path length: **{m['total_distance_km']:.2f} km**; mean speed **{m['mean_speed_mps']:.2f} m/s** (max {m['max_speed_mps']:.2f})",
         f"- Mean altitude **{m['mean_altitude_m']:.1f} m** (max {m['max_altitude_m']:.1f})",
         f"- Anomaly events: **{m['anomaly_events']}** ({m['high_severity_events']} high severity); {m['anomalous_rows_pct']:.1f}% of samples flagged", "",
         "## 2. PCA compression (measured)",
         f"- Features compressed: {c['n_features']} -> **{c['n_components']}** components, retained variance **{c['retained_variance'] * 100:.2f}%**",
         f"- Original: {fmt_bytes(c['original_bytes'])} (float64) -> compressed: {fmt_bytes(c['compressed_bytes'])} ({c['store_dtype']}, incl. basis, mean, scale)",
         f"- Compression ratio **{c['compression_ratio']:.2f}x**, storage saving **{c['storage_saving_pct']:.1f}%**",
         f"- Reconstruction RMSE (standardised) **{c['rmse_standardised']:.4f}**" +
         (f"; position error mean {c['pos_err_mean_m']:.2f} m, p95 {c['pos_err_p95_m']:.2f} m, max {c['pos_err_max_m']:.2f} m" if "pos_err_mean_m" in c else ""), "",
         "## 3. Flight-pattern clusters",
         f"- K-Means: k = {km['k']}" + (f", silhouette {km['quality']['silhouette']:.3f}" if km["quality"]["silhouette"] is not None else ""),
         f"- DBSCAN: eps = {db['eps']:.3f}, min_samples = {db['min_samples']}, clusters = {db['n_clusters']}, noise windows = {db['n_noise']}", ""]
    for _, r in res["cluster_desc"]["kmeans"].iterrows():
        L.append(f"- **{r['pattern']}** ({r['windows']} windows; {r['main_drones']}): {r['explanation']}")
    L += ["", "## 4. Formation analysis"]
    if form.get("available") and form.get("members"):
        L += [f"- Coordinated group detected from the data: **{', '.join(form['members'])}** (tolerance {form['tolerance_m']:.1f} m)",
              f"- Mean offset from group reference path: {form['cohesion_mean_m']:.2f} m"]
        others = [d for d in form["ids"] if d not in form["members"]]
        if others:
            L.append(f"- Flying independently / off-formation: {', '.join(others)}")
    else:
        L.append(f"- {form.get('note') or 'No formation detected.'}")
    L += ["", "## 5. Anomaly alerts"]
    if len(ev):
        for _, e in ev.iterrows():
            L.append(f"- **[{e['severity']}] {e['event_id']} {e['drone_id']}** t={e['t_start_s']:.0f}-{e['t_end_s']:.0f}s - {e['explanation']}")
    else:
        L.append("- No anomalies flagged with the current settings.")
    L += ["", "## 6. Per-drone summary", ds.round(2).to_markdown(index=False) if _has_tabulate() else ds.round(2).to_string(index=False), ""]
    if res.get("evaluation"):
        e = res["evaluation"]
        L += ["## 7. Check against simulated labels (evaluation only)", f"_{e['caveat']}_"]
        if "anomaly_rows" in e:
            a = e["anomaly_rows"]
            L.append(f"- Row-level precision {a['precision']:.2f}, recall {a['recall']:.2f}, F1 {a['f1']:.2f} "
                     f"({a['flagged_rows']} flagged vs {a['true_anomalous_rows']} labelled)")
    L += ["", "## Methods", "PCA on standardised kinematic features; K-Means and DBSCAN on PCA-projected 20-sample behaviour windows; "
          "Isolation Forest / LOF on per-drone robust z-scores. Labels, IDs and timestamps are never model inputs.",
          f"Row PCA features: {', '.join(res['input_columns_used']['row_pca'])}."]
    return "\n".join(L)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
