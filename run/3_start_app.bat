@echo off
REM Starts the backend and the web interface in two windows, then opens a browser.
REM Close both windows to stop the app.
setlocal
cd /d "%~dp0.."

set PY=.venv\Scripts\python.exe
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

if not exist "data\indexes\faiss.index" (
    echo.
    echo   No search index yet.
    echo   Run 1_ingest.bat and 2_build_index.bat first.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   Starting Ask My Docs
echo.
echo   Two windows will open. Keep both running.
echo   The browser opens automatically once the backend is ready.
echo ============================================================
echo.

start "Ask My Docs - backend" cmd /k "%PY% -m uvicorn askmydocs.api.main:app --host 127.0.0.1 --port 8000"

echo Waiting for the backend to load its models (about 20 seconds)...
REM The backend deliberately loads both models before accepting traffic, so the
REM first question is fast instead of taking 13 seconds.
timeout /t 22 /nobreak >nul

start "Ask My Docs - web interface" cmd /k "%PY% -m streamlit run src\askmydocs\ui\streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"

timeout /t 6 /nobreak >nul
start http://localhost:8501

echo.
echo   Open in your browser:  http://localhost:8501
echo.
echo   To stop: close the two windows that opened.
echo.
pause
