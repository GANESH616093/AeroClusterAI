AEROCLUSTER SYNTHETIC MULTI-DRONE TELEMETRY DATASET
Project: SIH26057

Files
- aerocluster_synthetic_telemetry.csv: 1,800 rows; 10 simulated drones; 180 samples per drone.
- dataset_metadata.json: dataset description, label meanings, and caveats.

Columns
- timestamp_s: seconds since the start of each simulated drone log
- drone_id: drone identifier
- latitude, longitude: simulated GPS coordinates
- altitude_m: simulated altitude in metres (reference is illustrative, not a certified AGL/AMSL measurement)
- ground_speed_mps: simulated ground speed in metres/second
- heading_deg: simulated heading in degrees
- vertical_speed_mps: altitude change per second (approximate)
- battery_pct: simulated battery percentage
- scenario_label: broad expected scenario group
- expected_behavior: coordinated / independent / anomalous_simulated
- is_injected_anomaly: 1 only during deliberately injected abnormal intervals for D09 and D10

Suggested use
1. Load CSV with pandas.
2. Exclude IDs, timestamps, scenario labels, and anomaly labels from PCA input features.
3. Clean and standardize numeric features.
4. Fit PCA; report explained variance and reconstruction error.
5. Cluster the PCA representations using K-Means and/or DBSCAN.
6. Evaluate clustering with silhouette score where appropriate, and compare cluster assignments against scenario labels cautiously.
7. Build trajectory plots and a PCA scatter plot.
8. Use is_injected_anomaly only for evaluation, not as an input feature.

Important
This is SYNTHETIC data generated for development and demonstration. It is not real drone telemetry and must not be described as a real-world dataset. A simulated anomaly label indicates a deliberately programmed scenario, not proof of an actual unsafe flight.
