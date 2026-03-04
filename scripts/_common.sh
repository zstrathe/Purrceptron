#!/usr/bin/env bash
# Shared preamble for project scripts. Source this, don't execute it.
#
# Provides:
#   SCRIPT_DIR, PROJECT_ROOT — path variables
#   .env loading              — PI_USER, PI_HOST, etc.
#   ensure_venv()             — auto-creates/activates .venv for Python scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Auto-create and activate .venv (the scripts-only venv).
# Call this from any script that needs Python.
ensure_venv() {
    local venv="$PROJECT_ROOT/.venv"
    if [ ! -d "$venv" ]; then
        echo "[INFO] Creating .venv for scripts..."
        python3 -m venv "$venv"
        "$venv/bin/pip" install -q --upgrade pip
        "$venv/bin/pip" install -q -r "$PROJECT_ROOT/requirements-scripts.txt"
    fi
    source "$venv/bin/activate"
}
