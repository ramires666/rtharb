@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m rtharb.reporting.vwap_robust_portfolio
if errorlevel 1 (echo ОШИБКА: portfolio source должен быть COMPLETE/PASS.& pause & exit /b 1)
echo Готово: tradingview_vwap_robust_portfolio\index.html
pause
