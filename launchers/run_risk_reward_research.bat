@echo off
setlocal
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
  set "RTHARB_PY=.venv\Scripts\python.exe"
) else (
  set "RTHARB_PY=python"
)
%RTHARB_PY% -m rtharb.research.risk_reward
if errorlevel 1 exit /b %errorlevel%
%RTHARB_PY% -m rtharb.reporting.tradingview_lite
if errorlevel 1 exit /b %errorlevel%
%RTHARB_PY% -m rtharb.reporting.publish_completed
if errorlevel 1 exit /b %errorlevel%
%RTHARB_PY% -m rtharb.audit.tradingview_lite
endlocal
