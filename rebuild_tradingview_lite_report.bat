@echo off
set "RTHARB_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=python"
"%RTHARB_PY%" "%~dp0build_tradingview_lite_report.py"
if errorlevel 1 pause & exit /b 1
"%RTHARB_PY%" "%~dp0publish_completed_research.py"
if errorlevel 1 pause & exit /b 1
"%RTHARB_PY%" "%~dp0audit_tradingview_lite_report.py"
pause
