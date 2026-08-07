@echo off
REM Diagnoses the environment and reports what is ready and what is not.
setlocal
cd /d "%~dp0.."

echo ============================================================
echo   Ask My Docs - setup check
echo ============================================================
echo.

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [X] Virtual environment MISSING at .venv
    echo     Nothing will run until this exists.
    goto :end
)
echo [ok] Virtual environment found
"%PY%" --version

echo.
echo Checking installed packages...
"%PY%" -c "import fitz, faiss, sentence_transformers, groq, fastapi, streamlit; print('[ok] All packages installed')" 2>nul
if errorlevel 1 (
    echo [X] Some packages are missing.
    echo     Fix: .venv\Scripts\python.exe -m pip install -e ".[dev,ml,llm,serve]"
    goto :end
)

echo.
if exist ".env" (
    echo [ok] .env file found
    "%PY%" -c "import sys; sys.path.insert(0,'src'); from askmydocs.config import load_config; k=load_config().groq_api_key; print('[ok] GROQ_API_KEY loaded' if k else '[X] GROQ_API_KEY is empty in .env')"
) else (
    echo [X] No .env file - answering questions will not work
)

echo.
echo Checking your documents...
dir /b "data\raw_pdfs\*.pdf" >nul 2>&1
if errorlevel 1 (
    echo [!] No PDFs in data\raw_pdfs
    echo     Put some PDFs there, then run 1_ingest.bat
) else (
    for /f %%N in ('dir /b "data\raw_pdfs\*.pdf" ^| find /c /v ""') do echo [ok] %%N PDF^(s^) ready to ingest
)

echo.
if exist "data\indexes\faiss.index" (
    echo [ok] Search index is built - you can run 3_start_app.bat
) else (
    echo [!] No search index yet - run 1_ingest.bat then 2_build_index.bat
)

:end
echo.
echo ============================================================
pause
