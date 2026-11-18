"""Kinematic features (row level) and behavioural window features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EARTH_R, WINDOW_FEATURES


def circ_diff_deg(a: pd.Series | np.ndarray) -> np.ndarray:
    d = np.diff(np.asarray(a, float), prepend=np.nan)
    return (d + 180) % 360 - 180


def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Add local-metre coordinates, heading sin/cos, turn rate and acceleration."""
    out = df.copy()
    lat0, lon0 = out["latitude"].median(), out["longitude"].median()
    out["east_m"] = (out["longitude"] - lon0) * np.deg2rad(1) * EARTH_R * np.cos(np.deg2rad(lat0))
    out["north_m"] = (out["latitude"] - lat0) * np.deg2rad(1) * EARTH_R
    out["heading_sin"] = np.sin(np.deg2rad(out["heading_deg"]))
    out["heading_cos"] = np.cos(np.deg2rad(out["heading_deg"]))
    turn = np.zeros(len(out))
    acc = np.zeros(len(out))
    for _, idx in out.groupby("drone_id").indices.items():
        sub = out.iloc[idx]
        dt = np.diff(sub["timestamp_s"].to_numpy(float), prepend=np.nan)
        dt[~(dt > 0)] = np.nan
        tr = circ_diff_deg(sub["heading_deg"]) / dt
        ac = np.diff(sub["ground_speed_mps"].to_numpy(float), prepend=np.nan) / dt
        turn[idx] = np.nan_to_num(tr, nan=0.0)
        acc[idx] = np.nan_to_num(ac, nan=0.0)
    out["turn_rate_dps"] = turn
    out["accel_mps2"] = acc
    return out


def _window_stats(w: pd.DataFrame) -> dict:
    hs = np.deg2rad(w["heading_deg"].to_numpy(float))
    R = np.hypot(np.mean(np.sin(hs)), np.mean(np.cos(hs)))
    spread = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9))))
    e, n = w["east_m"].to_numpy(float), w["north_m"].to_numpy(float)
    path = np.hypot(np.diff(e), np.diff(n)).sum()
    chord = np.hypot(e[-1] - e[0], n[-1] - n[0])
    dur = max(w["timestamp_s"].iloc[-1] - w["timestamp_s"].iloc[0], 1e-9)
    batt = w["battery_pct"].to_numpy(float)
    drain = (batt[0] - batt[-1]) / dur * 60 if np.isfinite(batt).all() else 0.0
    vs = w["vertical_speed_mps"].to_numpy(float)
    return {
        "mean_speed": w["ground_speed_mps"].mean(), "std_speed": w["ground_speed_mps"].std(ddof=0),
        "mean_alt": w["altitude_m"].mean(), "alt_range": np.ptp(w["altitude_m"].to_numpy(float)),
        "mean_vspeed": vs.mean(), "max_abs_vspeed": np.abs(vs).max(),
        "heading_spread": spread, "mean_abs_turn": np.abs(w["turn_rate_dps"]).mean(),
        "straightness": chord / path if path > 1e-9 else 1.0, "battery_drain": drain,
    }


def build_windows(df: pd.DataFrame, size: int = 20, step: int = 5) -> pd.DataFrame:
    """Sliding windows per drone. `size`/`step` are in samples."""
    if size < 5:
        raise ValueError("Window size must be at least 5 samples.")
    recs = []
    for did, idx in df.groupby("drone_id").indices.items():
        sub = df.iloc[idx].reset_index(drop=False)  # 'index' keeps df row positions
        n = len(sub)
        if n < size:
            continue
        starts = list(range(0, n - size + 1, step))
        if starts[-1] != n - size:
            starts.append(n - size)
        for k, s in enumerate(starts):
            w = sub.iloc[s:s + size]
            rec = {"drone_id": did, "window_id": f"{did}-W{k:03d}",
                   "t_start": w["timestamp_s"].iloc[0], "t_end": w["timestamp_s"].iloc[-1],
                   "row_first": int(w["index"].iloc[0]), "row_last": int(w["index"].iloc[-1]),
                   "row_ids": w["index"].to_numpy()}
            rec.update(_window_stats(w))
            recs.append(rec)
    if not recs:
        raise ValueError(f"No drone has at least {size} samples; reduce the window size.")
    win = pd.DataFrame(recs)
    win["t_mid"] = (win["t_start"] + win["t_end"]) / 2
    return win


def active_features(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Drop features that are missing or constant (they carry no information)."""
    keep = []
    for c in candidates:
        s = frame[c] if c in frame else None
        if s is not None and s.notna().all() and np.nanstd(s.to_numpy(float)) > 1e-9:
            keep.append(c)
    return keep
