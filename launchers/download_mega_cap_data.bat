@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."
set "RTHARB_PY=.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe"
if not exist "%RTHARB_PY%" set "RTHARB_PY=python"
"%RTHARB_PY%" -m rtharb.data.download_mega_cap --download-missing
if errorlevel 1 (
  echo ОШИБКА: котировки не загружены или аудит покрытия не пройден.
  pause
  exit /b 1
)
echo Котировки и manifest проверены.
pause
