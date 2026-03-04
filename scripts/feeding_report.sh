#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Feeding report — auto-activates .venv, then runs the Python script.
# Usage: ./scripts/feeding_report.sh [--days 7] [--cat Cat3] [--format table]
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
ensure_venv

python "$PROJECT_ROOT/scripts/feeding_report.py" "$@"
