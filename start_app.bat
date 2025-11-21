@echo off
echo ============================================================
echo Starting Breast Histopathology Classification System
echo ============================================================
echo.

echo Step 1: Installing dependencies...
pip install -r requirements.txt
echo.

echo ============================================================
echo Step 2: Starting FastAPI Backend...
echo ============================================================
start "FastAPI Backend" cmd /k "python api.py"
timeout /t 3 /nobreak >nul
echo.

echo ============================================================
echo Step 3: Starting Streamlit UI...
echo ============================================================
echo Opening web browser...
streamlit run streamlit_app.py

