@echo off
chcp 65001 >nul
setlocal
set "PROJECT_DIR=%~dp0\.."
set "PYTHON_EXE=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" -m rtharb.research.vwap_all_assets_robust_selection %*
if errorlevel 1 (echo ОШИБКА: robust selection не завершён.& pause & exit /b 1)
echo Готово: research_output\vwap_all_assets_robust_selection
pause
