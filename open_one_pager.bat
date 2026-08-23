@echo off
title RTH Arbitrage - On-Demand Parquet Report
echo ======================================================================
echo Starting Real-Time Parquet On-Demand Report Server...
echo ======================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe server.py
) else (
    python server.py
)

pause
