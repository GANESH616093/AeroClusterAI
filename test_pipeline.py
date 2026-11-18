"""Run with:  python -m pytest -q   (or plain: python tests/test_pipeline.py)"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aerocluster import reporting as R  # noqa: E402
from aerocluster.config import LABEL_COLUMNS  # noqa: E402
from aerocluster.data_io import (DataValidationError, clean_telemetry, load_bundled,  # noqa: E402
                                 read_uploaded, simulate_fleet)
from aerocluster.pipeline import Params, run  # noqa: E402


def test_bundled_pipeline_and_no_label_leakage():
    raw, _ = load_bundled()
    r = run(raw, origin="SIMULATED")
    used = sum([r["input_columns_used"][k] for k in ("row_pca", "window_clustering", "anomaly")], [])
    assert not set(used) & set(LABEL_COLUMNS) and "drone_id" not in used and "timestamp_s" not in used
    assert r["formation"]["members"] == [f"D0{i}" for i in range(1, 7)]
    assert {"D09", "D10"} <= set(r["events"]["drone_id"])


def test_compression_numbers_are_consistent():
    r = run(load_bundled()[0])
    c = r["compression"]; m = c["metrics"]; n, p, k = m["n_rows"], m["n_features"], m["n_components"]
    assert m["compressed_bytes"] == (n * k + k * p + 2 * p) * 4
    assert abs(m["compression_ratio"] - n * p * 8 / m["compressed_bytes"]) < 1e-9
    X = r["df"][c["features"]].to_numpy()
    assert abs(m["rmse_standardised"] - np.sqrt((((c["recon"] - X) / c["scale"]) ** 2).mean())) < 1e-6
    full = run(load_bundled()[0], Params(n_components=10, store_dtype="float64"))["compression"]["metrics"]
    assert full["rmse_standardised"] < 1e-9  # all components, full precision -> lossless
    assert len(c["stored"]["scores"]) == n and R.compressed_npz(r)


def test_variance_target_monotonic():
    raw = load_bundled()[0]
    k = [run(raw, Params(variance_target=v))["compression"]["metrics"]["n_components"] for v in (0.8, 0.95, 0.99)]
    assert k == sorted(k) and k[0] < k[-1]


def test_dbscan_and_lof_run():
    r = run(load_bundled()[0], Params(anomaly_method="Local Outlier Factor", dbscan_eps=1.0, kmeans_k=3))
    assert r["kmeans"]["k"] == 3 and len(r["windows"]) > 0


def test_demo_generator_and_exports():
    r = run(simulate_fleet(4, 1, 1, 1, 120, seed=3), origin="SIMULATED demo")
    assert r["df"]["drone_id"].nunique() == 7
    for f in (R.mission_report_md(r), R.summary_json(r), R.to_csv_bytes(R.processed_frame(r)), R.to_csv_bytes(R.cluster_frame(r))):
        assert len(f) > 100


def test_upload_with_odd_columns_missing_channels_and_dirt():
    s = simulate_fleet(3, 1, 0, 0, 100, seed=5)
    u = s[["timestamp_s", "drone_id", "latitude", "longitude", "altitude_m"]].rename(
        columns={"timestamp_s": "Time", "drone_id": "UAV", "latitude": "Lat", "longitude": "Lon", "altitude_m": "Alt"})
    u["Lon"] = u["Lon"].astype(object); u.loc[5, "Lat"] = np.nan; u.loc[9, "Alt"] = 99999; u.loc[12, "Lon"] = "bad"
    u = pd.concat([u, u.iloc[:3]])  # duplicates
    r = run(read_uploaded(u.to_csv(index=False).encode()), origin="UPLOADED")
    assert "ground_speed_mps (from GPS)" in r["clean_report"].derived_columns
    assert r["evaluation"] is None and r["clean_report"].rows_out > 0


def test_datetime_timestamps_and_single_drone():
    s = simulate_fleet(1, 0, 0, 0, 80, seed=2)[["timestamp_s", "latitude", "longitude", "altitude_m"]].copy()
    s["timestamp_s"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(s["timestamp_s"], unit="s")
    r = run(s.rename(columns={"timestamp_s": "timestamp"}), origin="UPLOADED")
    assert r["formation"]["available"] is False and r["df"]["drone_id"].iloc[0] == "D01"


def test_bad_inputs_raise_friendly_errors():
    for bad in (b"", b"a,b\n1,2\n", b"timestamp,latitude\n1,2\n"):
        try:
            clean_telemetry(read_uploaded(bad)); assert False, "should fail"
        except DataValidationError as e:
            assert str(e)
    try:
        clean_telemetry(pd.DataFrame({"timestamp_s": [1, 2], "latitude": [1, 2], "longitude": [1, 2], "altitude_m": [1, 2]}))
        assert False
    except DataValidationError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
