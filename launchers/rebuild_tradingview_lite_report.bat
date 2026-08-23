@echo off
cd /d "%~dp0\.."
set "RTHARB_PY=.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=python"
"%RTHARB_PY%" -m rtharb.reporting.tradingview_lite
if errorlevel 1 pause & exit /b 1
"%RTHARB_PY%" -m rtharb.reporting.publish_completed
if errorlevel 1 pause & exit /b 1
"%RTHARB_PY%" -m rtharb.audit.tradingview_lite
pause
