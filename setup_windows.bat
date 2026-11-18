@echo off
echo === AeroCluster AI setup ===
python --version || (echo Python 3.10+ not found. Install from https://www.python.org/downloads/ and tick "Add python.exe to PATH" & exit /b 1)
python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1
python tests\test_pipeline.py
echo.
echo Setup finished. Start the dashboard with run_windows.bat
