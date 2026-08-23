@echo off
setlocal
cd /d "%~dp0\.."
set "RTHARB_PY=.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=..\rtharb\.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" (
  echo Python environment not found. Install Python 3.11 and run: pip install -e .
  pause
  exit /b 1
)
"%RTHARB_PY%" -m rtharb.audit.recalculate
if errorlevel 1 goto :fail
"%RTHARB_PY%" -m rtharb.audit.integrity
if errorlevel 1 goto :fail
start "" "audit_output\REPORT.html"
exit /b 0
:fail
echo Audit failed.
pause
exit /b 1
