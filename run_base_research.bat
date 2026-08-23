@echo off
setlocal
cd /d "%~dp0"
set "RTHARB_PY=.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=..\rtharb\.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" (
  echo Python environment not found.
  pause
  exit /b 1
)
"%RTHARB_PY%" research_base_strategy.py
if errorlevel 1 goto :fail
"%RTHARB_PY%" verify_research_selected.py
if errorlevel 1 goto :fail
"%RTHARB_PY%" audit_base_research.py
if errorlevel 1 goto :fail
start "" "research_output\BASE_STRATEGY_REPORT.html"
exit /b 0
:fail
echo Base research failed.
pause
exit /b 1
