# 🛰️ AeroCluster AI
**PCA-Based Compression of Multi-Drone Telemetry Data for Flight-Pattern Cluster Visualization (SIH26057)**

A Streamlit dashboard that cleans multi-drone telemetry, compresses it with PCA (with measured storage and
reconstruction numbers), groups flight behaviour with K-Means and DBSCAN, detects and explains unusual
movements, detects coordinated formations, and plots everything on dark-themed interactive charts and a map.

> **Data honesty.** The bundled dataset and demo mode are **SIMULATED**. Every screen shows a banner saying so.
> Uploaded files are never assumed to be real telemetry. Labels in the simulated data
> (`scenario_label`, `expected_behavior`, `is_injected_anomaly`) are **never** model inputs; they are only used in
> an optional "check against simulated labels" panel, which is a software sanity check, not real-world accuracy.

## Install and run (Windows 10/11, PowerShell or CMD)
1. Install Python 3.10–3.12 from python.org (tick **Add python.exe to PATH**).
2. Unzip the project, open a terminal in the folder, then:
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python tests\test_pipeline.py
streamlit run app.py
```
(Or just double-click `setup_windows.bat`, then `run_windows.bat`.) The app opens at http://localhost:8501.
PowerShell blocking activation? Run `Set-ExecutionPolicy -Scope Process Bypass` first, or use CMD.
The map uses a free CARTO basemap, so it needs internet; all analysis works offline.

## Using it
* **Data source**: provided synthetic dataset · demo mode (generate your own simulated fleet: formation, independent,
  speed/altitude deviation, route deviation) · upload CSV (a template can be downloaded).
* **Upload format**: required `timestamp` (seconds, epoch or datetime), `latitude`, `longitude`, `altitude`; optional
  `drone_id`, `speed`, `heading`, `vertical_speed`, `battery`. Common column aliases are recognised. Missing speed/heading/
  vertical speed are derived from GPS/altitude and reported.
* Sidebar controls: PCA variance/components/precision, window size, K-Means k (auto by silhouette), DBSCAN eps/min_samples,
  anomaly method (Isolation Forest / LOF), sensitivity, formation tolerance.
* Tabs: Overview · PCA compression · Clusters · Anomalies · Map & paths · Formation & comparison · Data quality · Export.
* Exports: processed CSV, cluster CSV, anomalies CSV, drone summary, reconstructed CSV, compressed `.npz`, JSON summary, Markdown report.

## How it works
| Stage | Method |
|---|---|
| Cleaning | alias mapping, type coercion, range checks, duplicate removal, short-gap interpolation, derived channels, per-step log |
| Compression | Row-level PCA on 10 standardised kinematic features (east/north m, altitude, speed, sin/cos heading, vertical speed, battery, turn rate, acceleration). Compressed size = scores + components + mean + scale in the chosen dtype; errors are measured on the *stored* values |
| Clustering | Sliding windows per drone → 10 behaviour features → standardise → PCA → K-Means (k by silhouette) and DBSCAN (eps from k-distance) |
| Formation | Offset-invariant, outlier-robust route-shape distance between drones; DBSCAN on the distance matrix finds the group |
| Anomalies | Per-drone robust z-scores (peer-aware for formation drones) → Isolation Forest / LOF; flagged only if a feature also deviates ≥ threshold, so every alert has a reason in plain language |

## Limitations (please read)
* With 10 features, PCA gives modest compression (≈2–3× at 95–97 % variance on the bundled data). Position features
  are the hardest to compress; the dashboard shows the resulting position error in metres rather than hiding it.
* Windows are defined in samples and assume roughly even sampling.
* Anomaly alerts are statistical flags, not proof of unsafe flight. Window-based detection blurs event edges, so
  precision against simulated labels is only moderate (see the evaluation panel).
* Formation analysis assumes drones share a common time axis.

## Layout
```
app.py                 Streamlit dashboard
aerocluster/           data_io · features · compression · clustering · formation · anomaly · pipeline · viz · reporting
data/                  provided synthetic dataset + metadata
tests/test_pipeline.py end-to-end tests (python -m pytest -q)
```

## Deploy as a web app
**A. Streamlit Community Cloud (free, no server to manage)**
1. Create a GitHub repository and push this folder (`app.py`, `aerocluster/`, `data/`, `requirements.txt`, `.streamlit/`).
2. Go to https://share.streamlit.io, sign in with GitHub, choose **New app**, select the repo, branch and `app.py`, then **Deploy**.
3. You get a public URL. Uploaded files are processed in memory for that session; for sensitive flight data, deploy privately instead (option B or C).

**B. Docker (any VPS, Azure, AWS, Google Cloud Run)**
```bat
docker build -t aerocluster-ai .
docker run -p 8501:8501 aerocluster-ai
```
Then open http://localhost:8501 (or put the container behind your cloud's HTTPS load balancer).

**C. Local network / on-site use**: `streamlit run app.py --server.address 0.0.0.0` and share `http://<your-PC-IP>:8501`.

## Using real drone telemetry
Export a flight log to CSV (e.g. from Mission Planner, QGroundControl or PX4/ArduPilot log tools) with time, lat, lon and
altitude columns - one file with a `drone_id` column for several drones, with timestamps on a shared clock so formation
analysis works. The app has not been validated on real flights: treat alerts as prompts for review, not safety decisions.

## Browser simulator (no install)
`web_simulation/index.html` is a single self-contained web page: double-click it to open in any modern browser, or upload it
to any static host (GitHub Pages, Netlify, an institute web server). It has a form for the scenario (drones, noise, seed,
PCA variance, clustering, alert threshold), a live play/pause flight simulation, PCA compression with measured size and error,
K-Means/DBSCAN pattern clusters, plain-language alerts, and export (Copy/Download report and CSVs).
It runs the same ideas as the Python app in JavaScript, entirely in the browser, on **simulated** drones only - it does not
read uploaded files; use the Streamlit app (`app.py`) for CSV upload and the full feature set (Isolation Forest/LOF,
formation analysis, PyDeck map). The browser version uses a simpler robust z-score detector, so gentle route deviations
are caught less reliably there.
