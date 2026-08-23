@echo off
setlocal
cd /d "%~dp0"
set "RTHARB_PY=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=python"
"%RTHARB_PY%" research_vwap_absolute_brackets.py
if errorlevel 1 pause & exit /b 1
"%RTHARB_PY%" audit_vwap_absolute_brackets.py
if errorlevel 1 pause & exit /b 1
start "" "research_output\vwap_absolute_brackets\REPORT.html"
endlocal
