@echo off
setlocal
chcp 65001 >nul
set "PYTHON=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
cd /d "%~dp0\.."
"%PYTHON%" -m rtharb.research.synthetic_index
if errorlevel 1 pause
"%PYTHON%" -m rtharb.reporting.synthetic_tradingview
if errorlevel 1 pause
"%PYTHON%" -m rtharb.audit.synthetic_tradingview
if errorlevel 1 pause
endlocal
