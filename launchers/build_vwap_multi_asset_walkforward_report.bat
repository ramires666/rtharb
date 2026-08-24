@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m rtharb.reporting.vwap_all_assets_robust_selection
if errorlevel 1 (
  echo.
  echo ОШИБКА: отчёт не построен. Research обязан иметь status COMPLETE 9/9.
  pause
  exit /b 1
)
echo.
echo Готово: tradingview_vwap_multi_asset_walkforward\index.html
pause
