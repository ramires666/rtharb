@echo off
setlocal
chcp 65001 >nul
set "PYTHON=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%~dp0research_synthetic_index.py"
if errorlevel 1 pause
"%PYTHON%" "%~dp0build_synthetic_tradingview_report.py"
if errorlevel 1 pause
"%PYTHON%" "%~dp0audit_synthetic_tradingview_report.py"
if errorlevel 1 pause
endlocal
