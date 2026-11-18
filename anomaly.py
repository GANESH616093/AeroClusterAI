"""Explainable anomaly detection on behavioural windows.

Each window feature is first converted to a robust z-score against *that drone's own typical
behaviour* (median / MAD over its windows). An IsolationForest or LocalOutlierFactor then scores the
windows, and a window is flagged only when the model calls it an outlier AND at least one feature
is far from the drone's norm (|z| >= z_min) - so every flag has a human-readable reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .config import WINDOW_FEATURE_NAMES

EXTRA_NAMES = {"formation_dev": "distance from the formation's reference path"}
UNITS = {"mean_speed": "m/s", "std_speed": "m/s", "mean_alt": "m", "alt_range": "m", "mean_vspeed": "m/s",
         "max_abs_vspeed": "m/s", "heading_spread": "deg", "mean_abs_turn": "deg/s",
         "straightness": "", "battery_drain": "%/min", "formation_dev": "m"}


def own_baseline_z(win: pd.DataFrame, feats: list[str], members=None):
    """Robust z per drone. Returns (z_frame, baseline_median_frame).

    For drones in a coordinated formation (`members`), a window is also compared with what its
    peers did at the same time; if the whole group manoeuvred together (e.g. a survey-leg turn)
    the z-score is shrunk, because that is routine group behaviour, not an anomaly.
    """
    fleet_scale = {f: 1.4826 * np.median(np.abs(win[f] - win[f].median())) for f in feats}
    z = pd.DataFrame(index=win.index, columns=feats, dtype=float)
    med = pd.DataFrame(index=win.index, columns=feats, dtype=float)
    for _, g in win.groupby("drone_id"):
        for f in feats:
            m = g[f].median()
            s = 1.4826 * np.median(np.abs(g[f] - m))
            floor = max(0.25 * fleet_scale[f], 0.05 * (win[f].std(ddof=0) or 1.0), 1e-6)
            z.loc[g.index, f] = (g[f] - m) / max(s, floor)
            med.loc[g.index, f] = m
    if members:
        mem = win[win["drone_id"].isin(members)]
        key = mem["t_start"].round().astype(int)
        for f in feats:
            peer_med = mem.groupby(key)[f].transform("median")
            floor = max(0.25 * fleet_scale[f], 0.05 * (win[f].std(ddof=0) or 1.0), 1e-6)
            peer_scale = max(1.4826 * np.median(np.abs(mem[f] - peer_med)), floor)
            peer_z = (mem[f] - peer_med) / peer_scale
            own = z.loc[mem.index, f]
            shrunk = np.sign(own) * np.minimum(own.abs(), peer_z.abs())
            z.loc[mem.index, f] = shrunk
    return z, med


def detect(win: pd.DataFrame, feats: list[str], method: str = "Isolation Forest",
           contamination: float = 0.08, z_min: float = 3.0, seed: int = 42, members=None) -> dict:
    z, med = own_baseline_z(win, feats, members)
    Zc = z.clip(-12, 12).to_numpy(float)
    contamination = float(np.clip(contamination, 0.005, 0.40))
    if method == "Local Outlier Factor":
        lof = LocalOutlierFactor(n_neighbors=min(20, len(win) - 1), contamination=contamination)
        pred = lof.fit_predict(Zc)
        raw = -lof.negative_outlier_factor_
    else:
        iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(Zc)
        pred = iso.predict(Zc)
        raw = -iso.score_samples(Zc)
    score = (raw - raw.min()) / (np.ptp(raw) or 1.0)
    zmax = z.abs().max(axis=1).to_numpy()
    flagged = (pred == -1) & (zmax >= z_min)
    out = win.copy()
    out["anomaly_score"] = score
    out["max_abs_z"] = zmax
    out["is_anomaly"] = flagged
    out["top_feature"] = z.abs().idxmax(axis=1)
    return {"windows": out, "z": z, "baseline": med, "method": method,
            "contamination": contamination, "z_min": z_min}


def _reason(feat, value, base, z):
    name = {**WINDOW_FEATURE_NAMES, **EXTRA_NAMES}.get(feat, feat)
    u = UNITS.get(feat, "")
    fmt = lambda v: f"{v:.2f}{(' ' + u) if u else ''}"
    up = z > 0
    if feat == "mean_speed":
        return f"flew {'much faster' if up else 'much slower'} than its own normal pace ({fmt(value)} vs typical {fmt(base)})"
    if feat == "std_speed":
        return f"its speed fluctuated more than usual (variation {fmt(value)} vs typical {fmt(base)})"
    if feat == "mean_alt":
        return f"{'climbed well above' if up else 'dropped well below'} its normal altitude ({fmt(value)} vs typical {fmt(base)})"
    if feat == "alt_range":
        return f"altitude swung by {fmt(value)} within a short span (typical {fmt(base)})"
    if feat == "mean_vspeed":
        return f"{'sustained climb' if up else 'sustained descent'} of {fmt(abs(value))} on average"
    if feat == "max_abs_vspeed":
        return f"sudden vertical-speed spike of {fmt(value)} (typical peak {fmt(base)})"
    if feat == "heading_spread":
        return f"heading was {'more erratic' if up else 'unusually steady'} than normal ({fmt(value)} vs typical {fmt(base)})"
    if feat == "mean_abs_turn":
        return f"{'turned much more' if up else 'turned much less'} than usual ({fmt(value)} vs typical {fmt(base)})"
    if feat == "straightness":
        return (f"path became {'unusually straight' if up else 'curved / non-linear'} "
                f"(straightness {value:.2f} vs typical {base:.2f})")
    if feat == "battery_drain":
        return f"battery drained {'faster' if up else 'slower'} than usual ({fmt(value)} vs typical {fmt(base)})"
    if feat == "formation_dev":
        return f"strayed {fmt(value)} from the formation's reference path while the group stayed together"
    return f"{name} was unusual"


def explain_window(row: pd.Series, z_row: pd.Series, base_row: pd.Series, max_reasons: int = 3, z_min: float = 3.0):
    order = z_row.abs().sort_values(ascending=False)
    reasons = [(f, _reason(f, row[f], base_row[f], z_row[f]), float(z_row[f]))
               for f in order.index if abs(z_row[f]) >= z_min][:max_reasons]
    return reasons


def severity(zmax: float) -> str:
    return "High" if zmax >= 8 else "Medium" if zmax >= 5 else "Low"


def build_events(res: dict, gap_s: float = 10.0) -> pd.DataFrame:
    """Merge consecutive flagged windows (per drone) into alert events with explanations."""
    w, z, base, z_min = res["windows"], res["z"], res["baseline"], res["z_min"]
    ev = []
    for did, g in w[w["is_anomaly"]].sort_values("t_start").groupby("drone_id"):
        cur = None
        for i, r in g.iterrows():
            if cur is not None and r["t_start"] <= cur["t_end"] + gap_s:
                cur["t_end"] = max(cur["t_end"], r["t_end"])
                cur["idx"].append(i)
            else:
                if cur:
                    ev.append(cur)
                cur = {"drone_id": did, "t_start": r["t_start"], "t_end": r["t_end"], "idx": [i]}
        if cur:
            ev.append(cur)
    rows = []
    for e in ev:
        sub = w.loc[e["idx"]]
        zsub = z.loc[e["idx"]]
        peak_i = sub["max_abs_z"].idxmax()
        feat_peak = zsub.abs().max().sort_values(ascending=False)
        reasons = []
        for f in feat_peak.index:
            if feat_peak[f] < z_min or len(reasons) >= 3:
                break
            wi = zsub[f].abs().idxmax()
            reasons.append(_reason(f, w.loc[wi, f], base.loc[wi, f], z.loc[wi, f]))
        zmax = float(sub["max_abs_z"].max())
        sev = severity(zmax)
        drivers = [f for f in feat_peak.index if feat_peak[f] >= z_min][:3]
        text = (f"{e['drone_id']} behaved unusually from t={e['t_start']:.0f}s to t={e['t_end']:.0f}s: "
                + "; ".join(reasons) + ".")
        rows.append({"event_id": f"A{len(rows) + 1:02d}", "drone_id": e["drone_id"],
                     "t_start_s": e["t_start"], "t_end_s": e["t_end"],
                     "duration_s": e["t_end"] - e["t_start"], "severity": sev,
                     "peak_z": zmax, "peak_score": float(sub["anomaly_score"].max()),
                     "windows": len(sub), "drivers": ", ".join(drivers), "explanation": text,
                     "first_window": w.loc[peak_i, "window_id"]})
    cols = ["event_id", "drone_id", "t_start_s", "t_end_s", "duration_s", "severity", "peak_z",
            "peak_score", "windows", "drivers", "explanation", "first_window"]
    return pd.DataFrame(rows, columns=cols)


def rows_from_windows(n_rows: int, win: pd.DataFrame, min_fraction: float = 0.5):
    """Per-row anomaly score and flag. A row is flagged if >= min_fraction of the windows
    that contain it were flagged (trims fuzzy window edges)."""
    cover = np.zeros(n_rows)
    hit = np.zeros(n_rows)
    ssum = np.zeros(n_rows)
    for ids, f, s in zip(win["row_ids"], win["is_anomaly"], win["anomaly_score"]):
        cover[ids] += 1
        hit[ids] += float(f)
        ssum[ids] += s
    cover[cover == 0] = 1
    return ssum / cover, (hit / cover) >= min_fraction
