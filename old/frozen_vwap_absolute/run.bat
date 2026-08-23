@echo off
setlocal
cd /d "%~dp0\..\.."
set "RTHARB_PY=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=python"
"%RTHARB_PY%" old\frozen_vwap_absolute\research.py
if errorlevel 1 pause & exit /b 1
"%RTHARB_PY%" old\frozen_vwap_absolute\audit.py
if errorlevel 1 pause & exit /b 1
start "" "old\frozen_vwap_absolute\output\results\REPORT.html"
endlocal
