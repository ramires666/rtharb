@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m rtharb.research.vwap_robust_portfolio
if errorlevel 1 (echo ОШИБКА: source robust-selection должен быть COMPLETE/PASS 9/9.& pause & exit /b 1)
"%PYTHON_EXE%" -m rtharb.audit.vwap_robust_portfolio
if errorlevel 1 (echo ОШИБКА независимого portfolio-аудита.& pause & exit /b 1)
echo Готово: research_output\vwap_robust_portfolio
pause
