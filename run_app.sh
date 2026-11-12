#!/bin/bash
# Convenience launcher for OpenFormula on Linux/macOS.
# Double-click (if your file manager runs .sh files) or run from a
# terminal: ./run_app.sh
#
# Uses a local .venv next to this script if one exists (e.g. created via
# `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`),
# otherwise falls back to whatever `python3`/streamlit is on your PATH.

cd "$(dirname "$(readlink -f "$0")")" || {
    echo "Could not find the app folder."
    read -r -p "Press Enter to close..."
    exit 1
}

if [ -x ".venv/bin/streamlit" ]; then
    STREAMLIT=".venv/bin/streamlit"
else
    STREAMLIT="python3 -m streamlit"
fi

echo "============================================"
echo "  Starting OpenFormula..."
echo "  A browser tab should open automatically. If it"
echo "  doesn't, use the 'Local URL' printed below."
echo ""
echo "  To stop the app, close this window (or press Ctrl+C)."
echo "============================================"
echo ""

$STREAMLIT run app.py

echo ""
echo "OpenFormula has stopped."
read -r -p "Press Enter to close..."
