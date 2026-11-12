@echo off
REM Convenience launcher for OpenFormula on Windows.
REM Double-click this file, or run it from a command prompt.
REM
REM Uses a local .venv next to this script if one exists (e.g. created via
REM   python -m venv .venv
REM   .venv\Scripts\pip install -r requirements.txt
REM otherwise falls back to whatever "python" is on your PATH.

cd /d "%~dp0"

echo ============================================
echo   Starting OpenFormula...
echo   A browser tab should open automatically. If it
echo   doesn't, use the "Local URL" printed below.
echo.
echo   To stop the app, close this window (or press Ctrl+C).
echo ============================================
echo.

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    python -m streamlit run app.py
)

echo.
echo OpenFormula has stopped.
pause
