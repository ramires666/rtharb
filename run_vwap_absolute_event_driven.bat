@echo off
chcp 65001 >nul
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" research_vwap_absolute_event_driven.py
if errorlevel 1 (
  echo.
  echo ОШИБКА: независимый event-driven расчёт не завершён.
  pause
  exit /b 1
)
"%PYTHON_EXE%" build_vwap_absolute_trading_report.py
if errorlevel 1 (
  echo.
  echo ОШИБКА: подробный интерактивный отчёт не собран.
  pause
  exit /b 1
)
"%PYTHON_EXE%" audit_vwap_absolute_trading_report.py
if errorlevel 1 (
  echo.
  echo ОШИБКА: аудит подробного отчёта не пройден.
  pause
  exit /b 1
)
echo.
echo Расчёт и аудит завершены.
start "" "%~dp0tradingview_vwap_absolute\index.html"
