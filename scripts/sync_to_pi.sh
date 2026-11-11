#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Sync project code to Raspberry Pi via rsync over SSH
# Usage: ./scripts/sync_to_pi.sh [pi_user@pi_host] [remote_dir]
#
# Reads defaults from .env (PI_USER, PI_HOST, PI_REMOTE_DIR).
# CLI arguments override .env values.
#
# Examples:
#   ./scripts/sync_to_pi.sh                              # uses .env
#   ./scripts/sync_to_pi.sh pi@raspberrypi.local
#   ./scripts/sync_to_pi.sh pi@192.168.1.50 ~/projects/cat_detection
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

PI_TARGET="${1:-${PI_USER:?Set PI_USER in .env}@${PI_HOST:?Set PI_HOST in .env}}"
REMOTE_DIR="${2:-${PI_REMOTE_DIR:-~/Repos/cat_detection}}"

echo "[INFO] Syncing project to $PI_TARGET:$REMOTE_DIR ..."

rsync -avz --delete \
    --exclude-from="$PROJECT_ROOT/.syncignore" \
    "$PROJECT_ROOT/" \
    "$PI_TARGET:$REMOTE_DIR/"

echo ""
echo "[OK] Sync complete!"
echo ""
echo "On the Pi:"
echo "  cd $REMOTE_DIR"
echo "  pip install -r requirements-pi.txt   # first time only"
