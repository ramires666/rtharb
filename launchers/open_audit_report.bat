@echo off
cd /d "%~dp0\.."
if not exist "audit_output\REPORT.html" (
  echo Report not found. Run launchers\run_full_audit.bat first.
  pause
  exit /b 1
)
start "" "audit_output\REPORT.html"
