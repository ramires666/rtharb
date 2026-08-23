@echo off
cd /d "%~dp0"
if not exist "audit_output\REPORT.html" (
  echo Report not found. Run recalculate_audit.py first.
  pause
  exit /b 1
)
start "" "audit_output\REPORT.html"
