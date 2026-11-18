"""Constants shared across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLED_CSV = ROOT / "data" / "aerocluster_synthetic_telemetry.csv"
BUNDLED_META = ROOT / "data" / "dataset_metadata.json"

# Accepted column-name aliases for uploaded CSVs (lower-case, stripped).
COLUMN_ALIASES = {
    "timestamp_s": ["timestamp_s", "timestamp", "time", "time_s", "t", "datetime", "epoch", "ts"],
    "drone_id": ["drone_id", "drone", "uav_id", "uav", "id", "vehicle_id", "callsign"],
    "latitude": ["latitude", "lat", "gps_lat", "gps_latitude"],
    "longitude": ["longitude", "lon", "lng", "long", "gps_lon", "gps_longitude"],
    "altitude_m": ["altitude_m", "altitude", "alt", "alt_m", "height", "height_m", "relative_altitude"],
    "ground_speed_mps": ["ground_speed_mps", "speed", "speed_mps", "ground_speed", "groundspeed", "velocity", "speed_ms"],
    "heading_deg": ["heading_deg", "heading", "yaw", "course", "bearing", "yaw_deg"],
    "vertical_speed_mps": ["vertical_speed_mps", "vertical_speed", "vspeed", "climb_rate", "vz"],
    "battery_pct": ["battery_pct", "battery", "battery_percent", "batt", "soc"],
}
REQUIRED_COLUMNS = ["latitude", "longitude", "altitude_m", "timestamp_s"]  # others are derived/optional

# Evaluation-only columns. NEVER used as model inputs.
LABEL_COLUMNS = ["scenario_label", "expected_behavior", "is_injected_anomaly"]

# Row-level features used for PCA compression (no IDs, no timestamps, no labels).
ROW_FEATURES = [
    "east_m", "north_m", "altitude_m", "ground_speed_mps",
    "heading_sin", "heading_cos", "vertical_speed_mps", "battery_pct",
    "turn_rate_dps", "accel_mps2",
]
ROW_FEATURE_NAMES = {
    "east_m": "East offset (m)", "north_m": "North offset (m)", "altitude_m": "Altitude (m)",
    "ground_speed_mps": "Ground speed (m/s)", "heading_sin": "sin(heading)", "heading_cos": "cos(heading)",
    "vertical_speed_mps": "Vertical speed (m/s)", "battery_pct": "Battery (%)",
    "turn_rate_dps": "Turn rate (deg/s)", "accel_mps2": "Acceleration (m/s²)",
}

# Window-level behavioural features used for clustering + anomaly scoring.
WINDOW_FEATURES = [
    "mean_speed", "std_speed", "mean_alt", "alt_range", "mean_vspeed",
    "max_abs_vspeed", "heading_spread", "mean_abs_turn", "straightness", "battery_drain",
]
WINDOW_FEATURE_NAMES = {
    "mean_speed": "average speed", "std_speed": "speed variability", "mean_alt": "average altitude",
    "alt_range": "altitude swing", "mean_vspeed": "net climb/descent rate",
    "max_abs_vspeed": "peak vertical speed", "heading_spread": "heading variability",
    "mean_abs_turn": "turning rate", "straightness": "path straightness",
    "battery_drain": "battery drain rate",
}
EARTH_R = 6371000.0
