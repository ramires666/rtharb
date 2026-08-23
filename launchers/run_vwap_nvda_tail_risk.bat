@echo off
chcp 65001 >nul
setlocal
set "PROJECT_DIR=%~dp0\.."
set "PYTHON_EXE=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" -m rtharb.research.vwap_nvda_tail_risk %*
if errorlevel 1 (
  echo.
  echo ОШИБКА: расчёт NVDA VWAP tail-risk не завершён.
  pause
  exit /b 1
)
echo.
echo Расчёт NVDA VWAP tail-risk завершён.
echo Результаты: research_output\vwap_nvda_tail_risk
pause
