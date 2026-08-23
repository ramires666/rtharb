@echo off
chcp 65001 >nul
setlocal
set "PROJECT_DIR=%~dp0\.."
set "PYTHON_EXE=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" -m rtharb.research.duration_stoploss_combined %*
if errorlevel 1 (
  echo.
  echo ОШИБКА: combined q95 time-stop + stop-loss расчёт не завершён.
  pause
  exit /b 1
)
echo.
echo Combined q95 time-stop + stop-loss расчёт и аудит завершены.
echo Результаты: research_output\duration_stoploss_combined
pause
