@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0.."
set "RTHARB_PORT=8765"
set "RTHARB_URL=http://127.0.0.1:%RTHARB_PORT%/tradingview_vwap_nvda_tail_risk/index.html"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port=%RTHARB_PORT%; if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) { $python = if (Test-Path '.venv\Scripts\python.exe') { (Resolve-Path '.venv\Scripts\python.exe').Path } elseif (Test-Path 'C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe') { 'C:\Users\wafer\Documents\rtharb\.venv\Scripts\python.exe' } else { 'python' }; Start-Process -FilePath $python -ArgumentList '-m','http.server',$port,'--bind','127.0.0.1' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden; Start-Sleep -Milliseconds 900 }; Start-Process '%RTHARB_URL%'"
endlocal
