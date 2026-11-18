"""AeroCluster AI - dashboard entry point.  Run:  streamlit run app.py"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import streamlit as st

from aerocluster import reporting as R
from aerocluster import viz
from aerocluster.compression import fmt_bytes
from aerocluster.config import ROW_FEATURE_NAMES, WINDOW_FEATURE_NAMES
from aerocluster.data_io import DataValidationError, load_bundled, read_uploaded, simulate_fleet
from aerocluster.pipeline import Params, run

st.set_page_config(page_title="AeroCluster AI", page_icon="🛰️", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.2rem}
.hero{background:linear-gradient(120deg,#0b1d3a 0%,#0f172a 55%,#1e1b4b 100%);border:1px solid #1e3a5f;
 border-radius:14px;padding:18px 24px;margin-bottom:10px}
.hero h1{margin:0;font-size:1.9rem;letter-spacing:.5px}.hero p{margin:.2rem 0 0;color:#94a3b8}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.78rem;font-weight:600;margin-top:8px}
.sim{background:#78350f;color:#fde68a;border:1px solid #f59e0b}.up{background:#164e63;color:#a5f3fc;border:1px solid #22d3ee}
[data-testid=stMetric]{background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:10px 14px}
.alert-High{border-left:5px solid #ef4444}.alert-Medium{border-left:5px solid #f59e0b}.alert-Low{border-left:5px solid #facc15}
.alert{background:#0f172a;border-radius:8px;padding:8px 14px;margin:6px 0}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar: data source
st.sidebar.title("🛰️ AeroCluster AI")
source = st.sidebar.radio("Data source", ["Provided synthetic dataset", "Demo mode (generate simulated fleet)", "Upload CSV"])
raw, origin, meta = None, "", {}

try:
    if source == "Provided synthetic dataset":
        raw, meta = load_bundled()
        origin = "SIMULATED - bundled AeroCluster synthetic dataset (SIH26057)"
    elif source.startswith("Demo"):
        st.sidebar.caption("Normal formation flights, independent routes, speed/altitude changes and route deviations.")
        if "seed" not in st.session_state:
            st.session_state.seed = 42
        if st.sidebar.button("🎲 New random seed"):
            st.session_state.seed = int(np.random.randint(0, 100000))
        c1, c2 = st.sidebar.columns(2)
        nc = c1.number_input("Formation drones", 0, 12, 6)
        ni = c2.number_input("Independent", 0, 6, 2)
        ns = c1.number_input("Speed/alt dev.", 0, 4, 1)
        nr = c2.number_input("Route dev.", 0, 4, 1)
        dur = st.sidebar.slider("Duration (s)", 60, 600, 180, 10)
        noise = st.sidebar.slider("Sensor noise ×", 0.2, 4.0, 1.0, 0.1)
        st.session_state.seed = st.sidebar.number_input("Seed", 0, 10**6, st.session_state.seed)
        raw = simulate_fleet(nc, ni, ns, nr, dur, st.session_state.seed, noise)
        origin = f"SIMULATED - generated in demo mode (seed {st.session_state.seed})"
    else:
        tpl = simulate_fleet(2, 0, 0, 0, 60, 1).drop(columns=["scenario_label", "expected_behavior", "is_injected_anomaly"])
        st.sidebar.download_button("⬇ CSV template", tpl.to_csv(index=False).encode(), "aerocluster_template.csv", "text/csv")
        up = st.sidebar.file_uploader("Telemetry CSV", type=["csv", "txt"])
        st.sidebar.caption("Needs: timestamp, latitude, longitude, altitude. Optional: drone_id, speed, heading, vertical speed, battery.")
        if up is None:
            st.markdown('<div class="hero"><h1>AeroCluster AI</h1><p>Upload a telemetry CSV in the sidebar, or switch to the provided / demo data.</p></div>', unsafe_allow_html=True)
            st.stop()
        raw = read_uploaded(up)
        origin = f"UPLOADED file '{up.name}' (provenance not verified - not assumed to be real telemetry)"
except DataValidationError as exc:
    st.error(f"Input problem: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load data: {exc}")
    st.stop()

# ------------------------------------------------------------------ sidebar: model controls
with st.sidebar.expander("⚙️ PCA & compression", expanded=False):
    mode = st.radio("Components", ["By retained variance", "Fixed number"], horizontal=True)
    vt = st.slider("Variance to keep", 0.70, 0.999, 0.95, 0.005)
    nfix = st.slider("Components", 1, 10, 6)
    dtype = st.selectbox("Stored precision", ["float32", "float64", "float16"])
with st.sidebar.expander("🧩 Clustering", expanded=False):
    win = st.slider("Window (samples)", 8, 60, 20)
    stp = st.slider("Window step", 1, 30, 5)
    auto_k = st.checkbox("Auto-pick K-Means k (silhouette)", True)
    kk = st.slider("k", 2, 8, 4, disabled=auto_k)
    auto_eps = st.checkbox("Auto DBSCAN eps", True)
    eps = st.slider("eps", 0.1, 6.0, 1.5, 0.05, disabled=auto_eps)
    ms = st.slider("DBSCAN min_samples", 3, 30, 8)
with st.sidebar.expander("🚨 Anomaly detection", expanded=False):
    method = st.selectbox("Method", ["Isolation Forest", "Local Outlier Factor"])
    cont = st.slider("Expected outlier share", 0.01, 0.30, 0.08, 0.01)
    zmin = st.slider("Min. deviation (robust z)", 2.0, 8.0, 3.0, 0.5)
with st.sidebar.expander("🛩️ Formation", expanded=False):
    auto_tol = st.checkbox("Auto tolerance", True)
    tol = st.slider("Route-shape tolerance (m)", 1.0, 100.0, 5.0, disabled=auto_tol)

params = Params(variance_target=vt, n_components=(nfix if mode == "Fixed number" else None), store_dtype=dtype,
                window_size=win, window_step=stp, kmeans_k=None if auto_k else kk,
                dbscan_eps=None if auto_eps else eps, dbscan_min_samples=ms, anomaly_method=method,
                contamination=cont, z_min=zmin, formation_tolerance_m=None if auto_tol else tol)


@st.cache_data(show_spinner="Cleaning, compressing, clustering…")
def analyse(raw_df: pd.DataFrame, p: Params, origin_: str):
    return run(raw_df, p, origin_)


try:
    res = analyse(raw, params, origin)
except DataValidationError as exc:
    st.error(f"Data validation failed: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Analysis failed: {exc}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
    st.stop()

df, wins, comp, m = res["df"], res["windows"], res["compression"], res["mission"]
cm = comp["metrics"]
all_ids = sorted(df["drone_id"].unique())
shown = st.sidebar.multiselect("Drones shown", all_ids, default=all_ids) or all_ids

# ------------------------------------------------------------------ header
is_sim = origin.startswith("SIMULATED")
st.markdown(f"""<div class="hero"><h1>🛰️ AeroCluster AI</h1>
<p>PCA-based compression of multi-drone telemetry for flight-pattern cluster visualisation · SIH26057</p>
<span class="badge {'sim' if is_sim else 'up'}">{'⚠ SIMULATED DATA — not real drone telemetry' if is_sim else 'USER-UPLOADED DATA'}</span>
<span style="color:#94a3b8;font-size:.8rem;margin-left:8px">{origin}</span></div>""", unsafe_allow_html=True)

tabs = st.tabs(["📊 Overview", "🗜️ PCA compression", "🧩 Clusters", "🚨 Anomalies", "🗺️ Map & paths",
                "🤝 Formation & comparison", "🧹 Data quality", "⬇ Export"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    a = st.columns(6)
    a[0].metric("Drones", m["drones"]); a[1].metric("Samples", f"{m['samples']:,}")
    a[2].metric("Duration", f"{m['duration_s']:.0f} s"); a[3].metric("Distance", f"{m['total_distance_km']:.2f} km")
    a[4].metric("Mean speed", f"{m['mean_speed_mps']:.2f} m/s"); a[5].metric("Max altitude", f"{m['max_altitude_m']:.1f} m")
    b = st.columns(6)
    b[0].metric("PCA components", f"{cm['n_components']} / {cm['n_features']}")
    b[1].metric("Variance kept", f"{cm['retained_variance'] * 100:.1f}%")
    b[2].metric("Compression", f"{cm['compression_ratio']:.2f}×")
    b[3].metric("Storage saved", f"{cm['storage_saving_pct']:.1f}%")
    b[4].metric("Anomaly alerts", m["anomaly_events"], f"{m['high_severity_events']} high", delta_color="inverse")
    b[5].metric("Formation", f"{m['formation_members']} drones")
    st.plotly_chart(viz.timeline(df[df["drone_id"].isin(shown)], res["events"][res["events"]["drone_id"].isin(shown)]), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.line_chart(df, "ground_speed_mps", "Ground speed", "m/s", shown), use_container_width=True)
    c2.plotly_chart(viz.line_chart(df, "altitude_m", "Altitude", "m", shown), use_container_width=True)
    st.subheader("Mission alerts")
    ev = res["events"][res["events"]["drone_id"].isin(shown)]
    if ev.empty:
        st.success("No anomalies flagged with the current settings.")
    for _, e in ev.sort_values("t_start_s").iterrows():
        st.markdown(f'<div class="alert alert-{e["severity"]}"><b>{e["severity"].upper()} · {e["event_id"]} · {e["drone_id"]}</b> '
                    f'· t = {e["t_start_s"]:.0f}–{e["t_end_s"]:.0f} s<br>{e["explanation"]}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ PCA
with tabs[1]:
    st.caption(f"PCA input: {', '.join(ROW_FEATURE_NAMES[f] for f in comp['features'])}. IDs, timestamps and labels are excluded.")
    k1 = st.columns(5)
    k1[0].metric("Original size", fmt_bytes(cm["original_bytes"]), "float64 matrix", delta_color="off")
    k1[1].metric("Compressed size", fmt_bytes(cm["compressed_bytes"]), f"{cm['store_dtype']} incl. basis", delta_color="off")
    k1[2].metric("Compression ratio", f"{cm['compression_ratio']:.2f}×")
    k1[3].metric("Reconstruction RMSE", f"{cm['rmse_standardised']:.4f}", "standardised units", delta_color="off")
    k1[4].metric("Position error (mean)", f"{cm.get('pos_err_mean_m', float('nan')):.2f} m", f"p95 {cm.get('pos_err_p95_m', float('nan')):.1f} m", delta_color="off")
    st.info("Compressed size = scores (rows × k) + component matrix + mean + scale, all stored in the chosen precision. "
            "Errors are measured from values after that storage cast. More components → lower error, lower ratio.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.scree(comp), use_container_width=True)
    c2.plotly_chart(viz.tradeoff(comp["sweep"], cm["n_components"]), use_container_width=True)
    st.subheader("Original vs PCA-compressed telemetry")
    s1, s2 = st.columns(2)
    d_sel = s1.selectbox("Drone", all_ids)
    f_sel = s2.selectbox("Feature", comp["features"], format_func=lambda f: ROW_FEATURE_NAMES.get(f, f))
    c3, c4 = st.columns(2)
    c3.plotly_chart(viz.original_vs_recon(df, comp, f_sel, d_sel), use_container_width=True)
    c4.plotly_chart(viz.path_original_vs_recon(df, d_sel) if "east_m_recon" in df else viz.go.Figure(), use_container_width=True)
    c5, c6 = st.columns(2)
    color = c5.radio("Colour PCA plot by", ["drone_id", "anomaly"], horizontal=True)
    dims = c6.radio("Dimensions", [2, 3], horizontal=True)
    if dims == 3 and cm["n_components"] < 3:
        st.warning("Keep ≥ 3 components to view the 3-D plot."); dims = 2
    st.plotly_chart(viz.pca_scatter(df[df["drone_id"].isin(shown)], color, dims), use_container_width=True)
    c7, c8 = st.columns(2)
    c7.plotly_chart(viz.loadings_heatmap(comp["loadings"]), use_container_width=True)
    c8.plotly_chart(viz.feature_error_bar(comp["feature_errors"]), use_container_width=True)
    with st.expander("Per-feature error table"):
        st.dataframe(comp["feature_errors"].drop(columns="key"), use_container_width=True)

# ------------------------------------------------------------------ Clusters
with tabs[2]:
    st.caption(f"Clustering runs on {len(wins)} behaviour windows ({params.window_size} samples each) projected with PCA. "
               f"Features: {', '.join(WINDOW_FEATURE_NAMES[f] for f in res['window_features'])}.")
    which = st.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True)
    col, key = ("kmeans_cluster", "kmeans") if which == "K-Means" else ("dbscan_cluster", "dbscan")
    q = res[key]["quality"]
    q1 = st.columns(4)
    if key == "kmeans":
        q1[0].metric("k", res["kmeans"]["k"]); q1[1].metric("Inertia", f"{res['kmeans']['inertia']:.1f}")
    else:
        q1[0].metric("Clusters", res["dbscan"]["n_clusters"]); q1[1].metric("Noise windows", res["dbscan"]["n_noise"], f"eps {res['dbscan']['eps']:.2f}", delta_color="off")
    q1[2].metric("Silhouette", "n/a" if q["silhouette"] is None else f"{q['silhouette']:.3f}")
    q1[3].metric("Davies-Bouldin", "n/a" if q["davies_bouldin"] is None else f"{q['davies_bouldin']:.3f}")
    if q["silhouette"] is None:
        st.warning("Fewer than 2 non-noise clusters - quality scores are undefined. Adjust eps / min_samples in the sidebar.")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(viz.cluster_scatter(wins, col, f"{which} patterns (PC1 vs PC2 of behaviour windows)"), use_container_width=True)
    c2.plotly_chart(viz.cluster_by_drone(wins, col), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(viz.cluster_scatter3d(wins, col), use_container_width=True)
    if key == "kmeans":
        c4.plotly_chart(viz.silhouette_curve(res["kmeans"]["silhouette_by_k"], res["kmeans"]["k"]), use_container_width=True)
    desc = res["cluster_desc"][key]
    st.plotly_chart(viz.cluster_profile(desc, res["window_features"]), use_container_width=True)
    st.subheader("What each pattern means")
    for _, r in desc.iterrows():
        st.markdown(f"**{r['pattern']}** · {r['windows']} windows ({r['share_pct']:.0f}%) · mostly {r['main_drones']}  \n{r['explanation']}")

# ------------------------------------------------------------------ Anomalies
with tabs[3]:
    st.caption("A window is flagged when the model calls it an outlier **and** a feature is far from that drone's normal "
               "(robust z ≥ threshold). Formation drones are also compared with their peers, so shared manoeuvres are not alarms.")
    ev = res["events"]
    st.dataframe(ev.drop(columns=["first_window"]).round(2), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.anomaly_heat(df[df["drone_id"].isin(shown)]), use_container_width=True)
    c2.plotly_chart(viz.line_chart(df, "vertical_speed_mps", "Vertical speed", "m/s", shown), use_container_width=True)
    st.subheader("Explained in plain language")
    for _, e in ev.iterrows():
        with st.expander(f"[{e['severity']}] {e['event_id']} · {e['drone_id']} · t={e['t_start_s']:.0f}–{e['t_end_s']:.0f}s"):
            st.write(e["explanation"])
            st.caption(f"Main drivers: {e['drivers']} · peak deviation {e['peak_z']:.1f}σ-equivalent · {e['windows']} windows")
    if not len(ev):
        st.success("Nothing flagged. Lower the thresholds in the sidebar to be more sensitive.")
    if res["evaluation"]:
        with st.expander("🧪 Check against simulated labels (evaluation only - labels never used for fitting)"):
            e = res["evaluation"]; st.warning(e["caveat"])
            if "anomaly_rows" in e:
                a = e["anomaly_rows"]; x = st.columns(4)
                x[0].metric("Precision", f"{a['precision']:.2f}"); x[1].metric("Recall", f"{a['recall']:.2f}")
                x[2].metric("F1", f"{a['f1']:.2f}"); x[3].metric("Flagged / labelled", f"{a['flagged_rows']} / {a['true_anomalous_rows']}")
                st.dataframe(e["per_drone"], hide_index=True)
            if "ari_kmeans_vs_scenario" in e:
                st.write(f"Adjusted Rand index vs scenario labels - K-Means {e['ari_kmeans_vs_scenario']:.2f}, DBSCAN {e['ari_dbscan_vs_scenario']:.2f}. "
                         "Low values are expected: clusters group *behaviour windows*, not whole drones.")

# ------------------------------------------------------------------ Map
with tabs[4]:
    c1, c2, c3 = st.columns([2, 2, 3])
    color_by = c1.radio("Colour paths by", ["Drone", "K-Means pattern", "DBSCAN pattern"], horizontal=False)
    flags = c2.checkbox("Show flagged points", True)
    t_lo, t_hi = float(df["timestamp_s"].min()), float(df["timestamp_s"].max())
    t_now = c3.slider("Mission time (s)", t_lo, t_hi, t_hi, 1.0)
    cc = {"Drone": "drone", "K-Means pattern": "cluster", "DBSCAN pattern": "cluster"}[color_by]
    ccol = "dbscan_cluster" if color_by.startswith("DBSCAN") else "kmeans_cluster"
    try:
        st.pydeck_chart(viz.build_map(df, shown, cc, t_now, flags, ccol), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Map could not be drawn ({exc}). Showing 2-D tracks instead.")
        st.plotly_chart(viz.trajectory_2d(df, shown), use_container_width=True)
    st.caption("Coordinates come from the telemetry as given; for simulated data the location is illustrative only.")
    d1, d2 = st.columns(2)
    d1.plotly_chart(viz.trajectory_2d(df, shown), use_container_width=True)
    d2.plotly_chart(viz.trajectory_3d(df, shown), use_container_width=True)

# ------------------------------------------------------------------ Formation / comparison
with tabs[5]:
    form = res["formation"]
    st.subheader("Coordinated formation analysis")
    if not form["available"]:
        st.info(form["note"])
    else:
        if form.get("members"):
            st.success(f"Drones **{', '.join(form['members'])}** fly the same route shape (tolerance {form['tolerance_m']:.1f} m, derived from the data). "
                       f"They stay on average {form['cohesion_mean_m']:.2f} m from the group's reference path.")
            others = [d for d in form["ids"] if d not in form["members"]]
            if others:
                st.write(f"Flying their own routes: **{', '.join(others)}**.")
        else:
            st.info(form["note"])
        f1, f2 = st.columns(2)
        f1.plotly_chart(viz.distance_heatmap(form["distance"]), use_container_width=True)
        if form.get("members"):
            f2.plotly_chart(viz.cohesion_chart(form), use_container_width=True)
            st.plotly_chart(viz.deviation_chart(form, [d for d in shown if d in form["members"]]), use_container_width=True)
    st.subheader("Compare drones")
    pick = st.multiselect("Drones to compare", all_ids, default=all_ids[:3], max_selections=6) or all_ids[:2]
    r1, r2 = st.columns(2)
    r1.plotly_chart(viz.radar(wins, res["window_features"], WINDOW_FEATURE_NAMES, pick), use_container_width=True)
    r2.plotly_chart(viz.line_chart(df, "ground_speed_mps", "Speed comparison", "m/s", pick), use_container_width=True)
    st.plotly_chart(viz.line_chart(df, "altitude_m", "Altitude comparison", "m", pick), use_container_width=True)
    st.dataframe(res["drone_summary"].round(2), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ Data quality
with tabs[6]:
    rep = res["clean_report"]
    q = st.columns(3)
    q[0].metric("Rows in", f"{rep.rows_in:,}"); q[1].metric("Rows kept", f"{rep.rows_out:,}"); q[2].metric("Rows removed", f"{rep.rows_in - rep.rows_out:,}")
    if rep.column_map:
        st.write("Column mapping applied:", rep.column_map)
    if rep.derived_columns:
        st.info("Derived from GPS/altitude: " + ", ".join(rep.derived_columns))
    for w_ in rep.warnings:
        st.warning(w_)
    st.dataframe(rep.as_frame(), use_container_width=True, hide_index=True) if rep.steps else st.success("No cleaning actions were needed.")
    st.write("**Columns used as model inputs**")
    st.json(res["input_columns_used"])
    st.dataframe(df.head(200), use_container_width=True)

# ------------------------------------------------------------------ Export
with tabs[7]:
    st.caption("Everything below is generated from the current analysis.")
    ex = [("Processed telemetry (CSV)", R.to_csv_bytes(R.processed_frame(res)), "aerocluster_processed.csv", "text/csv"),
          ("Cluster results per window (CSV)", R.to_csv_bytes(R.cluster_frame(res)), "aerocluster_clusters.csv", "text/csv"),
          ("Anomaly events (CSV)", R.to_csv_bytes(res["events"]), "aerocluster_anomalies.csv", "text/csv"),
          ("Drone summary (CSV)", R.to_csv_bytes(res["drone_summary"]), "aerocluster_drone_summary.csv", "text/csv"),
          ("Reconstructed telemetry (CSV)", R.to_csv_bytes(R.reconstructed_frame(res)), "aerocluster_reconstructed.csv", "text/csv"),
          ("Compressed payload (NPZ)", R.compressed_npz(res), "aerocluster_compressed.npz", "application/octet-stream"),
          ("Run summary (JSON)", R.summary_json(res), "aerocluster_summary.json", "application/json"),
          ("Mission report (Markdown)", R.mission_report_md(res).encode("utf-8"), "aerocluster_mission_report.md", "text/markdown")]
    cols = st.columns(2)
    for i, (label, data, name, mime) in enumerate(ex):
        cols[i % 2].download_button(f"⬇ {label}", data, name, mime, use_container_width=True)
    with st.expander("Preview mission report"):
        st.markdown(R.mission_report_md(res))
