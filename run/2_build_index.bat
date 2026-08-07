@echo off
REM Turns the chunks into the two search indexes.
REM The first run downloads the embedding model (about 440MB) - that is normal.
setlocal
cd /d "%~dp0.."

set PY=.venv\Scripts\python.exe
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

if not exist "data\processed\chunks.jsonl" (
    echo.
    echo   No chunks found. Run 1_ingest.bat first.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   Building the search indexes
echo.
echo   The first run downloads the embedding model (~440MB).
echo   That happens once. Later runs take seconds.
echo ============================================================
echo.

"%PY%" scripts\build_index.py --log-level WARNING

echo.
echo Next: run 3_start_app.bat  (or ask.bat for a quick question)
echo.
pause
