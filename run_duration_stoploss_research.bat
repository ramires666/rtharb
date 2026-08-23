@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "RTHARB_PY=.venv\Scripts\python.exe"
) else (
  set "RTHARB_PY=python"
)
%RTHARB_PY% research_duration_stoploss.py
if errorlevel 1 exit /b %errorlevel%
%RTHARB_PY% build_tradingview_lite_report.py
if errorlevel 1 exit /b %errorlevel%
%RTHARB_PY% publish_completed_research.py
if errorlevel 1 exit /b %errorlevel%
%RTHARB_PY% audit_tradingview_lite_report.py
endlocal
