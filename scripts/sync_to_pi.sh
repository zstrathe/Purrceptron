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

pi_ssh_target "$PI_TARGET"
RSH=(ssh "${PI_SSH_ALIAS[@]}")

echo "[INFO] Syncing project to $PI_TARGET:$REMOTE_DIR ..."

# go-w: files on /mnt/c look world-writable to rsync.
rsync -avz --delete --chmod=go-w \
    --exclude-from="$PROJECT_ROOT/.syncignore" \
    -e "${RSH[*]}" \
    "$PROJECT_ROOT/" \
    "$PI_SSH_DEST:$REMOTE_DIR/"

echo ""
echo "[OK] Sync complete!"
if "${RSH[@]}" "$PI_SSH_DEST" test -f /etc/systemd/system/cat-dashboard.service 2>/dev/null; then
    echo "The dashboard service still runs the old code. To load the new code:"
    echo "  ssh $PI_SSH_DEST sudo systemctl restart cat-dashboard"
else
    echo "Restart the dashboard on the Pi to load the new code."
fi
