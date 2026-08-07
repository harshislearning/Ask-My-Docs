# Double-click launchers

For running the project without VS Code or a terminal. Run them in order the
first time; after that, only `3_start_app.bat` (or `ask.bat`) is needed.

| File | What it does |
|---|---|
| `0_check_setup.bat` | Reports what is ready and what is missing. Start here if anything breaks. |
| `1_ingest.bat` | Reads every PDF in `data\raw_pdfs` and splits them into chunks |
| `2_build_index.bat` | Builds the two search indexes. First run downloads ~440MB. |
| `3_start_app.bat` | Starts the backend and web interface, opens the browser |
| `ask.bat` | Asks one question in the console. No servers, fastest way to test. |

## First time

1. Copy your PDFs into `data\raw_pdfs`
2. Double-click `1_ingest.bat`
3. Double-click `2_build_index.bat`
4. Double-click `3_start_app.bat`

## After that

Just `3_start_app.bat`. Re-run steps 1 and 2 only when you add or change PDFs.

## If a window closes instantly

Every script ends with `pause`, so a window that vanishes means it never
started. Run `0_check_setup.bat` — it prints what is wrong instead of exiting.
