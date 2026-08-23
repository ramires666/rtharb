@echo off
chcp 65001 >nul
setlocal
set "PROJECT_DIR=%~dp0\.."
set "PYTHON_EXE=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" -m rtharb.research.vwap_absolute_multi_asset %*
if errorlevel 1 (
  echo.
  echo ОШИБКА: мультиактивный VWAP-bracket расчёт не завершён.
  pause
  exit /b 1
)
echo.
echo Мультиактивный VWAP-bracket расчёт и внутренний аудит завершены.
echo Результаты: research_output\vwap_absolute_multi_asset
pause
