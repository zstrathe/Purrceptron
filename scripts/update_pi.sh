#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Push code changes to a Pi that is already set up, and load them.
# Runs on the PC, over SSH. For a new Pi use install_pi.sh.
#
# Usage: ./scripts/update_pi.sh [--deps] [--yes] [user@host]
#
#   1. copies the code from this checkout (like sync_to_pi.sh)
#   2. restarts the dashboard service, so the new code runs (sudo asks for
#      the Pi password if it needs one); without the service, says how to
#      restart by hand
#   3. waits until the dashboard answers and prints its address
#
#   --deps   also run setup_pi.sh (Python packages, hailo-apps, system
#            settings). Needed when requirements-pi.txt changed; without it,
#            the script says so.
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
source "$SCRIPT_DIR/_pi_remote.sh"

YES=0
FORCE=0
DEPS=0
TARGET=""
while [ $# -gt 0 ]; do
    case "$1" in
        --deps) DEPS=1; shift ;;
        --yes|-y) YES=1; shift ;;
        -h|--help) sed -n '10,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "Unknown option $1 (see --help)" ;;
        *) TARGET="$1"; shift ;;
    esac
done
TARGET="${TARGET:-${PI_USER:?Set PI_USER in .env or pass user@host}@${PI_HOST:?Set PI_HOST in .env or pass user@host}}"
REMOTE_DIR="${PI_REMOTE_DIR:-~/Repos/cat_detection}"
REMOTE_DIR="${REMOTE_DIR/#$HOME\//\~/}"

connect
rsh "test -f $(q "$REMOTE_DIR")/pi_dashboard.py" \
    || die "The project isn't on $TARGET:$REMOTE_DIR yet; set the Pi up with ./scripts/install_pi.sh"

step_code
if [ "$DEPS" = 1 ]; then
    step_setup
elif [ "$DEPS_CHANGED" = 1 ]; then
    echo "[WARN] requirements-pi.txt changed: install the new Python packages with"
    echo "         ./scripts/update_pi.sh --deps"
fi

if service_installed; then
    echo "[INFO] Restarting the dashboard with the new code..."
    rsh_tty "cd $(q "$REMOTE_DIR") && bash scripts/install_service.sh --restart"
    SERVICE_ON=1
    echo ""
    dashboard_howto
else
    echo ""
    echo "[INFO] The dashboard isn't a service on this Pi, so restart it by hand to load the new code"
    echo "       (or run ./scripts/install_pi.sh --steps service to start it at boot)."
    dashboard_howto
fi
