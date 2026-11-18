"""Plotly charts and PyDeck map for the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#38bdf8", "#f472b6", "#a3e635", "#fbbf24", "#a78bfa", "#2dd4bf", "#fb7185", "#60a5fa",
           "#f97316", "#34d399", "#e879f9", "#facc15"]
BG = "rgba(0,0,0,0)"
GRID = "rgba(148,163,184,0.15)"


def _style(fig, title=None, height=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
                      title=dict(text=title, font=dict(size=15)) if title else None,
                      margin=dict(l=10, r=10, t=45 if title else 15, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", color="#e2e8f0"),
                      legend=dict(bgcolor="rgba(15,23,42,0.6)"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def drone_colors(ids):
    return {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(sorted(ids))}


def cluster_label(c):
    return "Noise" if c == -1 else f"Pattern {int(c) + 1}"


# ---------------------------------------------------------------- PCA / compression
def scree(comp):
    ev, cum = comp["explained_ratio"] * 100, comp["cumulative"] * 100
    k = comp["metrics"]["n_components"]
    x = [f"PC{i + 1}" for i in range(len(ev))]
    fig = go.Figure()
    fig.add_bar(x=x, y=ev, name="Variance per component (%)",
                marker_color=["#38bdf8" if i < k else "#334155" for i in range(len(ev))])
    fig.add_scatter(x=x, y=cum, name="Cumulative (%)", mode="lines+markers", line=dict(color="#fbbf24", width=3))
    fig.add_hline(y=comp["metrics"]["retained_variance"] * 100, line_dash="dot", line_color="#a3e635",
                  annotation_text=f"kept: {comp['metrics']['retained_variance'] * 100:.1f}%")
    fig.update_yaxes(title="Explained variance (%)", range=[0, 102])
    return _style(fig, "Explained variance - blue bars are kept")


def tradeoff(sweep, k):
    fig = go.Figure()
    fig.add_scatter(x=sweep["components"], y=sweep["compression_ratio"], name="Compression ratio (x)",
                    mode="lines+markers", line=dict(color="#38bdf8", width=3))
    fig.add_scatter(x=sweep["components"], y=sweep["rmse_standardised"], name="Reconstruction RMSE (std. units)",
                    mode="lines+markers", line=dict(color="#f472b6", width=3), yaxis="y2")
    fig.add_vline(x=k, line_dash="dash", line_color="#a3e635", annotation_text="selected")
    fig.update_layout(yaxis=dict(title="Compression ratio (x)"),
                      yaxis2=dict(title="RMSE", overlaying="y", side="right", gridcolor=BG),
                      xaxis=dict(title="Components kept", dtick=1))
    return _style(fig, "Compression vs. accuracy trade-off (measured for every k)")


def pca_scatter(df, color="drone_id", dims=2):
    d = df.copy()
    d["drone_id"] = d["drone_id"].astype(str)
    if color == "anomaly":
        d["state"] = np.where(d["is_anomaly"], "Flagged", "Normal")
        color = "state"
        cmap = {"Normal": "#38bdf8", "Flagged": "#ef4444"}
    else:
        cmap = drone_colors(d["drone_id"].unique())
    if dims == 3 and "PC3" in d:
        fig = px.scatter_3d(d, x="PC1", y="PC2", z="PC3", color=color, color_discrete_map=cmap,
                            hover_data=["drone_id", "timestamp_s"], opacity=0.8)
        fig.update_traces(marker_size=3)
        return _style(fig, "Telemetry in PCA space (3-D)", 560)
    fig = px.scatter(d, x="PC1", y="PC2", color=color, color_discrete_map=cmap,
                     hover_data=["drone_id", "timestamp_s"], opacity=0.75)
    fig.update_traces(marker_size=5)
    return _style(fig, "Telemetry in PCA space (PC1 vs PC2)", 480)


def loadings_heatmap(load):
    fig = px.imshow(load, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", text_auto=".2f")
    return _style(fig, "What each principal component is made of", 420)


def original_vs_recon(df, comp, feature_key, drone):
    feats = comp["features"]
    if feature_key not in feats:
        return go.Figure()
    g = df[df["drone_id"] == drone]
    fig = go.Figure()
    fig.add_scatter(x=g["timestamp_s"], y=g[feature_key], name="Original", mode="lines", line=dict(color="#38bdf8", width=3))
    fig.add_scatter(x=g["timestamp_s"], y=g[f"{feature_key}_recon"], name="PCA-reconstructed", mode="lines",
                    line=dict(color="#f472b6", width=2, dash="dot"))
    err = (g[f"{feature_key}_recon"] - g[feature_key]).abs()
    fig.add_scatter(x=g["timestamp_s"], y=err, name="|error|", mode="lines", line=dict(color="#fbbf24", width=1),
                    fill="tozeroy", opacity=0.35)
    fig.update_xaxes(title="Time (s)")
    return _style(fig, f"{drone}: original vs reconstructed", 380)


def path_original_vs_recon(df, drone):
    g = df[df["drone_id"] == drone]
    fig = go.Figure()
    fig.add_scatter(x=g["east_m"], y=g["north_m"], name="Original path", mode="lines", line=dict(color="#38bdf8", width=3))
    fig.add_scatter(x=g["east_m_recon"], y=g["north_m_recon"], name="Reconstructed path", mode="lines",
                    line=dict(color="#f472b6", width=2, dash="dot"))
    fig.update_xaxes(title="East (m)")
    fig.update_yaxes(title="North (m)", scaleanchor="x")
    return _style(fig, f"{drone}: path before / after compression", 380)


def feature_error_bar(fe):
    fig = px.bar(fe, x="feature", y="nrmse_pct_of_std", color="nrmse_pct_of_std", color_continuous_scale="Turbo")
    fig.update_yaxes(title="RMSE as % of feature std. dev.")
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, "Reconstruction error by feature", 360)


# ---------------------------------------------------------------- clustering
def cluster_scatter(win, col, title):
    d = win.copy()
    d["cluster"] = d[col].map(cluster_label)
    order = sorted(d["cluster"].unique())
    fig = px.scatter(d, x="WPC1", y="WPC2", color="cluster", symbol="drone_id", category_orders={"cluster": order},
                     color_discrete_sequence=PALETTE, hover_data=["window_id", "t_start", "t_end"], opacity=0.85)
    fig.update_traces(marker_size=9, marker_line_width=0.5, marker_line_color="#0f172a")
    return _style(fig, title, 480)


def cluster_scatter3d(win, col):
    if "WPC3" not in win:
        return go.Figure()
    d = win.copy()
    d["cluster"] = d[col].map(cluster_label)
    fig = px.scatter_3d(d, x="WPC1", y="WPC2", z="WPC3", color="cluster", color_discrete_sequence=PALETTE,
                        hover_data=["window_id"])
    fig.update_traces(marker_size=4)
    return _style(fig, "Behaviour windows in 3-D PCA space", 520)


def silhouette_curve(sil, chosen):
    fig = go.Figure(go.Scatter(x=list(sil), y=list(sil.values()), mode="lines+markers", line=dict(color="#a3e635", width=3)))
    fig.add_vline(x=chosen, line_dash="dash", line_color="#fbbf24")
    fig.update_xaxes(title="k", dtick=1)
    fig.update_yaxes(title="Silhouette")
    return _style(fig, "K-Means: silhouette score by k (computed)", 300)


def cluster_by_drone(win, col):
    ct = pd.crosstab(win["drone_id"], win[col].map(cluster_label), normalize="index") * 100
    fig = px.imshow(ct, color_continuous_scale="Viridis", aspect="auto", text_auto=".0f")
    fig.update_layout(coloraxis_colorbar_title="% windows")
    return _style(fig, "Share of each drone's flight spent in each pattern (%)", 380)


def cluster_profile(desc, feats):
    z = desc.set_index("pattern")[feats]
    z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    fig = px.imshow(z, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto", text_auto=".1f")
    return _style(fig, "Cluster fingerprints (z-score vs. other clusters)", 360)


# ---------------------------------------------------------------- flight charts
def line_chart(df, y, title, ylabel, drones, flag=True):
    cmap = drone_colors(df["drone_id"].unique())
    fig = go.Figure()
    for d in drones:
        g = df[df["drone_id"] == d]
        fig.add_scatter(x=g["timestamp_s"], y=g[y], name=d, mode="lines", line=dict(color=cmap[d], width=2))
        if flag:
            f = g[g["is_anomaly"]]
            if len(f):
                fig.add_scatter(x=f["timestamp_s"], y=f[y], mode="markers", name=f"{d} flagged", showlegend=False,
                                marker=dict(color="#ef4444", size=6, line=dict(color="white", width=0.5)))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title=ylabel)
    return _style(fig, title, 380)


def anomaly_heat(df):
    p = df.assign(t=(df["timestamp_s"] // 5 * 5)).pivot_table(index="drone_id", columns="t", values="anomaly_score", aggfunc="mean")
    fig = px.imshow(p, color_continuous_scale="Inferno", aspect="auto", labels=dict(x="Time (s)", y="Drone", color="Score"))
    return _style(fig, "Anomaly score by drone and time", 360)


def timeline(df, events):
    cmap = drone_colors(df["drone_id"].unique())
    fig = go.Figure()
    for d, g in df.groupby("drone_id"):
        fig.add_bar(y=[d], x=[g["timestamp_s"].max() - g["timestamp_s"].min()], base=[g["timestamp_s"].min()], orientation="h",
                    marker_color=cmap[d], opacity=0.35, name=d, showlegend=False, hoverinfo="skip")
    sev_col = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#facc15"}
    for _, e in events.iterrows():
        fig.add_bar(y=[e["drone_id"]], x=[max(e["duration_s"], 1)], base=[e["t_start_s"]], orientation="h",
                    marker_color=sev_col[e["severity"]], name=e["severity"], showlegend=False,
                    hovertext=f"{e['event_id']} ({e['severity']}): {e['explanation']}", hoverinfo="text")
    fig.update_layout(barmode="overlay", yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title="Mission time (s)")
    return _style(fig, "Mission timeline - red/amber = anomaly alerts", 120 + 38 * df["drone_id"].nunique())


# ---------------------------------------------------------------- comparison / formation
def distance_heatmap(D):
    fig = px.imshow(D, color_continuous_scale="Viridis_r", aspect="auto", text_auto=".0f")
    fig.update_layout(coloraxis_colorbar_title="metres")
    return _style(fig, "Route-shape distance between drones (low = same pattern)", 460)


def cohesion_chart(form):
    fig = go.Figure()
    fig.add_scatter(x=form["cohesion_by_time"].index, y=form["cohesion_by_time"], name="Median offset from reference path (m)",
                    line=dict(color="#a3e635", width=3))
    fig.add_scatter(x=form["mean_spacing_by_time"].index, y=form["mean_spacing_by_time"], name="Mean spacing between members (m)",
                    line=dict(color="#38bdf8", width=2))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="metres")
    return _style(fig, "Formation cohesion over time", 360)


def deviation_chart(form, drones):
    dev = form["deviation_by_time"]
    cmap = drone_colors(form["ids"])
    fig = go.Figure()
    for d in drones:
        if d in dev:
            fig.add_scatter(x=dev.index, y=dev[d], name=d, line=dict(color=cmap[d], width=2))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Offset from formation path (m)")
    return _style(fig, "Deviation from the formation reference path", 360)


def radar(win, feats, names, drones):
    z = win.groupby("drone_id")[feats].mean()
    z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    cmap = drone_colors(win["drone_id"].unique())
    fig = go.Figure()
    for d in drones:
        r = z.loc[d].tolist()
        fig.add_scatterpolar(r=r + r[:1], theta=[names.get(f, f) for f in feats] + [names.get(feats[0], feats[0])],
                             name=d, line=dict(color=cmap[d]), fill="toself", opacity=0.45)
    fig.update_layout(polar=dict(bgcolor=BG, radialaxis=dict(gridcolor=GRID), angularaxis=dict(gridcolor=GRID)))
    return _style(fig, "Behaviour fingerprint (z-score vs. fleet)", 480)


def trajectory_2d(df, drones):
    cmap = drone_colors(df["drone_id"].unique())
    fig = go.Figure()
    for d in drones:
        g = df[df["drone_id"] == d]
        fig.add_scatter(x=g["east_m"], y=g["north_m"], name=d, mode="lines", line=dict(color=cmap[d], width=2))
        f = g[g["is_anomaly"]]
        if len(f):
            fig.add_scatter(x=f["east_m"], y=f["north_m"], mode="markers", showlegend=False,
                            marker=dict(color="#ef4444", size=6))
    fig.update_xaxes(title="East (m)")
    fig.update_yaxes(title="North (m)", scaleanchor="x")
    return _style(fig, "Ground tracks (red = flagged)", 480)


def trajectory_3d(df, drones):
    cmap = drone_colors(df["drone_id"].unique())
    fig = go.Figure()
    for d in drones:
        g = df[df["drone_id"] == d]
        fig.add_scatter3d(x=g["east_m"], y=g["north_m"], z=g["altitude_m"], mode="lines", name=d, line=dict(color=cmap[d], width=4))
    fig.update_layout(scene=dict(xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Altitude (m)"))
    return _style(fig, "3-D flight paths", 560)


# ---------------------------------------------------------------- map (PyDeck)
def hex_rgb(h, a=220):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def build_map(df, drones, color_by="drone", t_now=None, show_flags=True, cluster_col="kmeans_cluster"):
    import pydeck as pdk
    d = df[df["drone_id"].isin(drones)]
    cmap = drone_colors(df["drone_id"].unique())
    paths = []
    for did, g in d.groupby("drone_id"):
        if color_by == "drone":
            segs = [(g, hex_rgb(cmap[did]))]
        else:  # colour by cluster: break the path where the cluster changes
            segs, run = [], (g[cluster_col] != g[cluster_col].shift()).cumsum()
            for _, s in g.groupby(run):
                c = int(s[cluster_col].iloc[0])
                col = [148, 163, 184, 200] if c < 0 else hex_rgb(PALETTE[c % len(PALETTE)])
                s = pd.concat([s, g.loc[[min(s.index.max() + 1, g.index.max())]]]).drop_duplicates()
                segs.append((s, col))
        for s, col in segs:
            paths.append({"drone": did, "path": s[["longitude", "latitude"]].values.tolist(), "color": col})
    layers = [pdk.Layer("PathLayer", data=paths, get_path="path", get_color="color", width_min_pixels=3,
                        get_width=1.2, pickable=True)]
    if show_flags and d["is_anomaly"].any():
        f = d[d["is_anomaly"]][["drone_id", "longitude", "latitude", "altitude_m", "timestamp_s"]].copy()
        layers.append(pdk.Layer("ScatterplotLayer", data=f, get_position=["longitude", "latitude"],
                                get_fill_color=[239, 68, 68, 230], get_radius=1.3, radius_min_pixels=4, pickable=True))
    if t_now is not None:
        cur = (d.assign(dt=(d["timestamp_s"] - t_now).abs()).sort_values("dt").groupby("drone_id").head(1)
               [["drone_id", "longitude", "latitude", "altitude_m", "ground_speed_mps", "timestamp_s"]].copy())
        cur["color"] = cur["drone_id"].map(lambda x: hex_rgb(cmap[x], 255))
        layers.append(pdk.Layer("ScatterplotLayer", data=cur, get_position=["longitude", "latitude"], get_fill_color="color",
                                get_line_color=[255, 255, 255], line_width_min_pixels=2, stroked=True,
                                get_radius=2.5, radius_min_pixels=8, pickable=True))
        layers.append(pdk.Layer("TextLayer", data=cur, get_position=["longitude", "latitude"], get_text="drone_id",
                                get_size=14, get_color=[255, 255, 255, 255], get_pixel_offset=[0, -18]))
    lat, lon = d["latitude"].mean(), d["longitude"].mean()
    span_m = max(np.ptp(d["east_m"]), np.ptp(d["north_m"]), 20)
    zoom = float(np.clip(16.5 - np.log2(span_m / 400), 3, 19))
    return pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=35),
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    tooltip={"text": "{drone}{drone_id}\n{timestamp_s}"})
