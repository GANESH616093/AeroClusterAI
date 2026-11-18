"""Loading, simulating, validating and cleaning drone telemetry."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (BUNDLED_CSV, BUNDLED_META, COLUMN_ALIASES, EARTH_R,
                     LABEL_COLUMNS, REQUIRED_COLUMNS)

NUMERIC = ["latitude", "longitude", "altitude_m", "ground_speed_mps",
           "heading_deg", "vertical_speed_mps", "battery_pct"]


class DataValidationError(ValueError):
    """Raised when the input cannot be turned into usable telemetry."""


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    steps: list = field(default_factory=list)       # (action, rows_affected, detail)
    warnings: list = field(default_factory=list)
    derived_columns: list = field(default_factory=list)
    column_map: dict = field(default_factory=dict)

    def add(self, action, n, detail=""):
        self.steps.append({"action": action, "rows_affected": int(n), "detail": detail})

    def as_frame(self):
        return pd.DataFrame(self.steps) if self.steps else pd.DataFrame(
            columns=["action", "rows_affected", "detail"])


# --------------------------------------------------------------------------- loading
def load_bundled() -> tuple[pd.DataFrame, dict]:
    """The synthetic dataset shipped with the project."""
    if not BUNDLED_CSV.exists():
        raise FileNotFoundError(f"Bundled dataset missing: {BUNDLED_CSV}")
    meta = json.loads(BUNDLED_META.read_text(encoding="utf-8")) if BUNDLED_META.exists() else {}
    return pd.read_csv(BUNDLED_CSV), meta


def read_uploaded(file_or_bytes) -> pd.DataFrame:
    """Read an uploaded CSV (file-like or bytes) with friendly errors."""
    try:
        raw = file_or_bytes.getvalue() if hasattr(file_or_bytes, "getvalue") else (
            file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes)
        if not raw:
            raise DataValidationError("The uploaded file is empty.")
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding_errors="replace")
    except DataValidationError:
        raise
    except pd.errors.EmptyDataError:
        raise DataValidationError("The uploaded file contains no data.")
    except Exception as exc:  # noqa: BLE001
        raise DataValidationError(f"Could not parse the file as CSV: {exc}")
    if df.empty:
        raise DataValidationError("The CSV has headers but no rows.")
    return df


def standardise_columns(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    lower = {c: str(c).strip().lower() for c in df.columns}
    rename = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for col, low in lower.items():
            if low in aliases and col not in rename and canon not in rename.values():
                rename[col] = canon
                break
    report.column_map = {k: v for k, v in rename.items() if k != v}
    out = df.rename(columns=rename)
    # keep evaluation labels if present under their canonical name
    return out.loc[:, ~out.columns.duplicated()].copy()


# --------------------------------------------------------------------------- simulation
def simulate_fleet(n_coordinated=6, n_independent=2, n_speed_alt=1, n_route_dev=1,
                   duration_s=180, seed=42, noise=1.0) -> pd.DataFrame:
    """Generate a SIMULATED fleet. Labels are returned for evaluation only."""
    rng = np.random.default_rng(seed)
    lat0, lon0 = 31.25, 75.70
    t = np.arange(duration_s, dtype=float)
    rows = []

    def survey_heading(tt):
        leg = 45.0
        k = (tt // leg).astype(int)
        h = np.where(k % 2 == 0, 90.0, 270.0)
        phase = tt % leg
        h = np.where((phase > leg - 6), h + 90.0 * ((phase - (leg - 6)) / 6.0) * (np.where(k % 2 == 0, 1, -1)), h)
        return h

    def build(did, scenario, behaviour, hd, speed, alt, e0, n0, anomaly_mask=None):
        h = np.deg2rad(hd)
        v = speed
        e = e0 + np.cumsum(v * np.sin(h))
        n = n0 + np.cumsum(v * np.cos(h))
        vs = np.gradient(alt)
        batt = 100 - 0.035 * t * (1 + 0.1 * rng.normal()) - np.cumsum(np.abs(rng.normal(0, 0.002, len(t))))
        lat = lat0 + n / EARTH_R * 180 / np.pi + rng.normal(0, 2e-7 * noise, len(t))
        lon = lon0 + e / (EARTH_R * np.cos(np.deg2rad(lat0))) * 180 / np.pi + rng.normal(0, 2e-7 * noise, len(t))
        df = pd.DataFrame({
            "timestamp_s": t, "drone_id": did, "latitude": lat, "longitude": lon,
            "altitude_m": alt, "ground_speed_mps": v + rng.normal(0, 0.15 * noise, len(t)),
            "heading_deg": np.mod(hd + rng.normal(0, 1.0 * noise, len(t)), 360),
            "vertical_speed_mps": vs, "battery_pct": batt,
            "scenario_label": scenario, "expected_behavior": behaviour,
            "is_injected_anomaly": (anomaly_mask.astype(int) if anomaly_mask is not None else 0)})
        rows.append(df)

    idx = 1
    base_h = survey_heading(t)
    for i in range(n_coordinated):
        alt = 60 + rng.normal(0, 0.3 * noise, len(t)).cumsum() * 0.05 + 1.0 * np.sin(t / 30 + i * 0.1)
        sp = 5.8 + 0.3 * np.sin(t / 20) + rng.normal(0, 0.1 * noise, len(t))
        build(f"D{idx:02d}", "coordinated_survey", "coordinated", base_h + rng.normal(0, 0.5), sp, alt, 0, 12.0 * i)
        idx += 1
    for i in range(n_independent):
        hd = rng.uniform(0, 360) + (40 + 25 * i) * np.sin(t / (40 + 15 * i))
        hd = hd + (t * (0.8 + 0.4 * i)) * (1 if i % 2 == 0 else -1)
        alt = 50 + 6 * np.sin(t / (35 + 10 * i) + i)
        sp = 4.5 + 0.8 * i + 0.4 * np.sin(t / 15 + i)
        build(f"D{idx:02d}", "independent_route", "independent", hd, sp, alt, rng.uniform(-60, 60), rng.uniform(-60, 60))
        idx += 1
    lo_a, hi_a = int(0.36 * duration_s), int(0.58 * duration_s)
    for i in range(n_speed_alt):
        m = (t >= lo_a) & (t <= hi_a)
        ramp = np.where(m, np.sin(np.pi * (t - lo_a) / max(hi_a - lo_a, 1)), 0)
        alt = 60 + 14 * ramp + np.where(m, 3 * np.sin(t / 2.5), 0)
        sp = 5.8 + 3.0 * ramp + rng.normal(0, 0.1, len(t)) + np.where(m, 0.8 * np.sin(t / 3), 0)
        build(f"D{idx:02d}", "speed_altitude_deviation", "anomalous_simulated", base_h, sp, alt, 0, 12.0 * (n_coordinated + 1), m)
        idx += 1
    lo_r, hi_r = int(0.44 * duration_s), int(0.70 * duration_s)
    for i in range(n_route_dev):
        m = (t >= lo_r) & (t <= hi_r)
        ramp = np.where(m, np.sin(np.pi * (t - lo_r) / max(hi_r - lo_r, 1)), 0)
        hd = base_h + 70 * ramp
        alt = 60 + np.sin(t / 30)
        sp = 5.8 + 0.3 * np.sin(t / 20)
        build(f"D{idx:02d}", "route_deviation", "anomalous_simulated", hd, sp, alt, 0, 12.0 * (n_coordinated + 2), m)
        idx += 1
    if not rows:
        raise DataValidationError("Choose at least one drone to simulate.")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- cleaning
def _heading_from_xy(east, north):
    return np.mod(np.degrees(np.arctan2(east, north)), 360)


def clean_telemetry(raw: pd.DataFrame, min_samples: int = 20) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and clean telemetry. Returns (clean_df, CleaningReport)."""
    rep = CleaningReport(rows_in=len(raw))
    df = standardise_columns(raw, rep)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Found columns: {', '.join(map(str, raw.columns))}. "
            "Required: timestamp, latitude, longitude, altitude (drone_id, speed, heading are optional).")

    if "drone_id" not in df.columns:
        df["drone_id"] = "D01"
        rep.warnings.append("No drone_id column - treating the file as a single drone (D01).")
    df["drone_id"] = df["drone_id"].astype(str).str.strip()
    blank = df["drone_id"].isin(["", "nan", "None"])
    if blank.any():
        df = df.loc[~blank].copy()
        rep.add("Dropped rows with blank drone_id", blank.sum())

    # timestamps -> seconds on a shared clock
    ts = df["timestamp_s"]
    num = pd.to_numeric(ts, errors="coerce")
    if num.notna().mean() >= 0.9:
        df["timestamp_s"] = num
        if num.max() > 1e9:  # unix epoch
            df["timestamp_s"] = num - num.min()
            rep.add("Converted epoch timestamps to seconds since first sample", len(df))
    else:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
        if dt.notna().mean() < 0.9:
            raise DataValidationError("Could not interpret the timestamp column as numbers or dates.")
        df["timestamp_s"] = (dt - dt.min()).dt.total_seconds()
        rep.add("Parsed datetime timestamps to seconds since first sample", len(df))

    # numeric coercion
    for c in NUMERIC:
        if c in df.columns:
            before = df[c].isna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            bad = df[c].isna().sum() - before
            if bad:
                rep.add(f"Non-numeric values in '{c}' set to missing", bad)

    bad_core = df[["timestamp_s", "latitude", "longitude", "altitude_m"]].isna().any(axis=1)
    # core columns may be gap-filled later only for lat/lon/alt; timestamp must exist
    no_ts = df["timestamp_s"].isna()
    if no_ts.any():
        df = df.loc[~no_ts].copy()
        rep.add("Dropped rows without a timestamp", no_ts.sum())

    # impossible values
    rules = {
        "latitude": (-90, 90), "longitude": (-180, 180),
        "altitude_m": (-500, 20000), "ground_speed_mps": (0, 120),
        "vertical_speed_mps": (-60, 60),
    }
    for c, (lo, hi) in rules.items():
        if c in df.columns:
            bad = (df[c] < lo) | (df[c] > hi)
            if bad.any():
                df.loc[bad, c] = np.nan
                rep.add(f"Out-of-range '{c}' ({lo}..{hi}) set to missing", bad.sum())
    null_island = (df["latitude"].abs() < 1e-9) & (df["longitude"].abs() < 1e-9)
    if null_island.any():
        df.loc[null_island, ["latitude", "longitude"]] = np.nan
        rep.add("GPS (0,0) fixes treated as invalid", null_island.sum())
    if "battery_pct" in df.columns:
        over = (df["battery_pct"] > 100) | (df["battery_pct"] < 0)
        if over.any():
            df["battery_pct"] = df["battery_pct"].clip(0, 100)
            rep.add("Battery clipped to 0-100 %", over.sum())
    if "heading_deg" in df.columns:
        wrap = (df["heading_deg"] < 0) | (df["heading_deg"] >= 360)
        if wrap.any():
            df["heading_deg"] = np.mod(df["heading_deg"], 360)
            rep.add("Heading wrapped to 0-360 deg", wrap.sum())

    # duplicates and ordering
    dup = df.duplicated(subset=["drone_id", "timestamp_s"], keep="first")
    if dup.any():
        df = df.loc[~dup].copy()
        rep.add("Removed duplicate (drone, timestamp) rows", dup.sum())
    df = df.sort_values(["drone_id", "timestamp_s"]).reset_index(drop=True)

    # gap filling: short gaps only
    fill_cols = [c for c in NUMERIC if c in df.columns]
    n_missing = int(df[fill_cols].isna().sum().sum())
    if n_missing:
        filled = df.groupby("drone_id", group_keys=False)[fill_cols].apply(
            lambda g: g.interpolate(method="linear", limit=5, limit_area="inside"))
        df[fill_cols] = filled
        rep.add("Linearly interpolated short gaps (<=5 samples)", n_missing - int(df[fill_cols].isna().sum().sum()))

    # derive missing optional channels from GPS
    lat0, lon0 = df["latitude"].median(), df["longitude"].median()
    g = df.groupby("drone_id")
    cosl = np.cos(np.deg2rad(lat0))
    east = (df["longitude"] - lon0) * np.deg2rad(1) * EARTH_R * cosl
    north = (df["latitude"] - lat0) * np.deg2rad(1) * EARTH_R
    dt = g["timestamp_s"].diff()
    de, dn = east.groupby(df["drone_id"]).diff(), north.groupby(df["drone_id"]).diff()
    dalt = g["altitude_m"].diff()
    if "ground_speed_mps" not in df.columns or df["ground_speed_mps"].isna().all():
        df["ground_speed_mps"] = np.hypot(de, dn) / dt.replace(0, np.nan)
        rep.derived_columns.append("ground_speed_mps (from GPS)")
    if "heading_deg" not in df.columns or df["heading_deg"].isna().all():
        df["heading_deg"] = _heading_from_xy(de, dn)
        rep.derived_columns.append("heading_deg (from GPS)")
    if "vertical_speed_mps" not in df.columns or df["vertical_speed_mps"].isna().all():
        df["vertical_speed_mps"] = dalt / dt.replace(0, np.nan)
        rep.derived_columns.append("vertical_speed_mps (from altitude)")
    if "battery_pct" not in df.columns:
        df["battery_pct"] = np.nan
        rep.warnings.append("No battery column - battery is excluded from the model features.")
    for c in ["ground_speed_mps", "heading_deg", "vertical_speed_mps"]:
        df[c] = df.groupby("drone_id")[c].transform(lambda s: s.bfill().ffill())

    # drop rows that still miss essential channels
    essential = ["latitude", "longitude", "altitude_m", "ground_speed_mps", "heading_deg", "vertical_speed_mps"]
    bad = df[essential].isna().any(axis=1)
    if bad.any():
        df = df.loc[~bad].copy()
        rep.add("Dropped rows still incomplete after gap filling", bad.sum())

    # drones with too few samples
    counts = df["drone_id"].value_counts()
    small = counts[counts < min_samples].index.tolist()
    if small:
        df = df[~df["drone_id"].isin(small)].copy()
        rep.add(f"Dropped drones with < {min_samples} samples ({', '.join(small)})", int(counts[small].sum()))
    if df.empty:
        raise DataValidationError(f"No usable data left after cleaning (each drone needs >= {min_samples} valid samples).")

    # timing sanity
    steps = df.groupby("drone_id")["timestamp_s"].diff().dropna()
    if len(steps) and steps.median() > 0 and (steps > 5 * steps.median()).any():
        rep.warnings.append("Irregular sampling detected (large time gaps); window features assume roughly even sampling.")

    keep_labels = [c for c in LABEL_COLUMNS if c in df.columns]
    cols = ["drone_id", "timestamp_s"] + [c for c in NUMERIC if c in df.columns] + keep_labels
    df = df[cols].sort_values(["drone_id", "timestamp_s"]).reset_index(drop=True)
    rep.rows_out = len(df)
    return df, rep
