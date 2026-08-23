@echo off
chcp 65001 >nul
setlocal
set "PROJECT_DIR=%~dp0\.."
set "PYTHON_EXE=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" -m rtharb.research.vwap_absolute_event_driven
if errorlevel 1 (
  echo.
  echo ОШИБКА: независимый event-driven расчёт не завершён.
  pause
  exit /b 1
)
"%PYTHON_EXE%" -m rtharb.reporting.vwap_absolute_trading
if errorlevel 1 (
  echo.
  echo ОШИБКА: подробный интерактивный отчёт не собран.
  pause
  exit /b 1
)
"%PYTHON_EXE%" -m rtharb.audit.vwap_absolute_trading
if errorlevel 1 (
  echo.
  echo ОШИБКА: аудит подробного отчёта не пройден.
  pause
  exit /b 1
)
echo.
echo Расчёт и аудит завершены.
start "" "%PROJECT_DIR%\tradingview_vwap_absolute\index.html"
