@echo off
REM Reads every PDF in data\raw_pdfs and splits them into searchable chunks.
setlocal
cd /d "%~dp0.."

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo ERROR: no virtual environment. Run 0_check_setup.bat first.
    pause
    exit /b 1
)

dir /b "data\raw_pdfs\*.pdf" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   No PDFs found in:
    echo   %CD%\data\raw_pdfs
    echo.
    echo   Copy some PDFs into that folder, then run this again.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   Reading your PDFs
echo ============================================================
echo.

"%PY%" scripts\ingest.py --log-level WARNING

echo.
echo Next: run 2_build_index.bat
echo.
pause
