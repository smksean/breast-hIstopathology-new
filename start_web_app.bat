@echo off
echo ============================================================
echo Starting Breast Histopathology AI System (Lightweight)
echo ============================================================
echo.

echo Step 1: Starting API Server...
start "API Server" cmd /k "cd /d %~dp0 && python api.py"
timeout /t 3 /nobreak >nul

echo Step 2: Opening Web Interface...
start "" "http://localhost:8000"
cd web
start "" "index.html"

echo.
echo ============================================================
echo System Started!
echo ============================================================
echo.
echo Web Interface: file:///web/index.html
echo API Server: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to exit...
pause >nul

