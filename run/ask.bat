@echo off
REM Ask one question and print the cited answer. No servers, no browser.
REM The quickest way to check the system works.
setlocal enabledelayedexpansion
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
echo   Ask your documents
echo ============================================================
echo.

set "QUESTION="
set /p QUESTION="Your question: "

if "!QUESTION!"=="" (
    echo.
    echo   No question entered.
    pause
    exit /b 1
)

echo.
echo Thinking...
"%PY%" scripts\ask.py "!QUESTION!" --show-sources

echo.
pause
