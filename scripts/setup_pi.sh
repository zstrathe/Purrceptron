#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Setup Raspberry Pi for cat detection data collection
# Run from the project root: ~/cat-detector/
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[INFO] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-opencv python3-yaml python3-pip python3-venv

# ── Install hailo-apps (Hailo AI detection pipeline) ──────────────────
# Prerequisites: hailort driver + packages must already be installed.
# See: https://github.com/hailo-ai/hailo-apps#prerequisites
HAILO_APPS_DIR="$PROJECT_ROOT/hailo-apps"
# The hailo-apps commit this project runs on, with HailoRT 5.1.1 (hailo-h10-all
# from the Raspberry Pi apt repository). Newer commits may expect a newer
# HailoRT; override with HAILO_APPS_COMMIT=<commit> if you upgrade both.
HAILO_APPS_COMMIT="${HAILO_APPS_COMMIT:-29acd11a8e641b2b61c3df45c95c97a775878e9f}"
# Marks a finished install, so an install that failed part way (for example
# with no Hailo driver loaded) is run again instead of skipped.
HAILO_APPS_DONE="$HAILO_APPS_DIR/.cat_detection_installed"
if [ -f "$HAILO_APPS_DONE" ]; then
    echo "[INFO] hailo-apps already installed at $HAILO_APPS_DIR"
else
    if [ ! -d "$HAILO_APPS_DIR" ]; then
        echo "[INFO] Installing hailo-apps (commit ${HAILO_APPS_COMMIT:0:7})..."
        git clone https://github.com/hailo-ai/hailo-apps.git "$HAILO_APPS_DIR"
        git -C "$HAILO_APPS_DIR" checkout -q "$HAILO_APPS_COMMIT"
    else
        echo "[INFO] Running the hailo-apps installer again (no finished install recorded)..."
    fi
    if [ ! -e /dev/hailo0 ]; then
        echo "[ERROR] No Hailo device (/dev/hailo0); hailo-apps' installer needs it." >&2
        echo "        Install the driver first: sudo apt install dkms hailo-h10-all, then reboot." >&2
        exit 1
    fi
    cd "$HAILO_APPS_DIR"
    sudo ./install.sh
    cd "$PROJECT_ROOT"
    touch "$HAILO_APPS_DONE"
fi

# ── Install project dependencies into hailo-apps venv ─────────────────
# setup_env.sh activates venv_hailo_apps and sets GStreamer/Hailo env vars.
# We use this single venv for everything (dashboard + detection + classifier).
SAVE_PROJECT_ROOT="$PROJECT_ROOT"
echo "[INFO] Installing project dependencies in hailo-apps venv..."
cd "$HAILO_APPS_DIR"
set +u  # setup_env.sh uses variables that may be unbound (e.g. ZSH_VERSION)
source setup_env.sh
set -u
pip install --upgrade pip
pip install -r "$SAVE_PROJECT_ROOT/requirements-pi.txt"
# ExecuTorch runtime for the classifier (see the note in requirements-pi.txt).
# Must match executorch in requirements-training.txt.
pip install --no-deps "executorch==1.1.0"
deactivate 2>/dev/null || true
PROJECT_ROOT="$SAVE_PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ── Fewer SD card writes ───────────────────────────────────────────────
# SD cards wear out from writes. The app keeps its frequently rewritten files
# in RAM (/dev/shm); these system settings cover swap, logs and /tmp. Each
# file is only written when its content changes, so re-running is safe.
write_if_changed() {    # write_if_changed PATH CONTENT -> 0 if written, 1 if unchanged
    local path="$1" content="$2"
    if sudo test -f "$path" && [ "$(sudo cat "$path")" = "$content" ]; then
        return 1
    fi
    sudo mkdir -p "$(dirname "$path")"
    printf '%s\n' "$content" | sudo tee "$path" > /dev/null
    return 0
}
NEED_REBOOT=0

# Swap: Raspberry Pi OS (trixie) defaults to compressed RAM swap plus a swap
# file on the card that idle memory is written out to. Keep only the RAM part.
if dpkg -s rpi-swap >/dev/null 2>&1; then
    if write_if_changed /etc/rpi/swap.conf.d/80-cat-detection.conf \
"# cat-detection: compressed RAM swap only, no swap file on the SD card
[Main]
Mechanism=zram"; then
        echo "[INFO] Swap set to RAM only (zram); takes effect after a reboot"
        NEED_REBOOT=1
    else
        echo "[OK] Swap already set to RAM only (zram)"
    fi
    # Remove the old swap file once nothing uses it: not active swap, and not
    # the backing file of a zram device (the zram+file mode).
    if [ -f /var/swap ] && ! grep -q "^/var/swap " /proc/swaps \
            && ! grep -qs "/var/swap" /sys/block/zram*/backing_dev; then
        sudo rm -f /var/swap
        echo "[INFO] Removed the unused swap file /var/swap"
    fi
else
    echo "[WARN] rpi-swap is not installed; check swap with: swapon --show"
fi

# System journal: kept across reboots for reference, capped at 200 MB.
if write_if_changed /etc/systemd/journald.conf.d/80-cat-detection.conf \
"# cat-detection: keep system logs across reboots, capped
[Journal]
Storage=persistent
SystemMaxUse=200M"; then
    sudo systemctl restart systemd-journald
    echo "[INFO] System journal: persistent, at most 200 MB"
else
    echo "[OK] System journal already persistent, at most 200 MB"
fi

# /tmp should be a RAM disk (trixie's default).
if [ "$(findmnt -n -o FSTYPE /tmp 2>/dev/null)" = "tmpfs" ]; then
    echo "[OK] /tmp is in RAM (tmpfs)"
else
    echo "[WARN] /tmp is on the SD card. Enable the RAM disk with:"
    echo "       sudo systemctl enable tmp.mount && sudo reboot"
fi

# Dashboard output log (the cat-dashboard service, or ... > logs/pi_dashboard.out):
# rotated at 10 MB, 5 compressed copies kept. copytruncate lets the running
# dashboard keep writing to the same open file.
if write_if_changed /etc/logrotate.d/cat-detection \
"$PROJECT_ROOT/logs/pi_dashboard.out {
    su $(id -un) $(id -gn)
    size 10M
    rotate 5
    compress
    copytruncate
    missingok
    notifempty
}"; then
    echo "[INFO] Installed log rotation for logs/pi_dashboard.out"
else
    echo "[OK] Log rotation for logs/pi_dashboard.out already installed"
fi
sudo logrotate --debug /etc/logrotate.d/cat-detection > /dev/null 2>&1 \
    || echo "[WARN] logrotate rejected /etc/logrotate.d/cat-detection; check it with: sudo logrotate --debug /etc/logrotate.d/cat-detection"

echo "[INFO] Creating data directories..."
mkdir -p data/raw data/trash data/export models logs

echo "[INFO] Testing camera..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f'[OK] Camera working: {w}x{h}')
    else:
        print('[WARN] Camera opened but cannot read frames')
    cap.release()
else:
    print('[WARN] Cannot open camera at index 0')
    print('       Check: ls /dev/video*')
    print('       Try: --device /dev/video0 or --device /dev/video2')
"

echo ""
echo "[OK] Pi setup complete!"
echo ""
echo "To start the dashboard (includes live detection + review):"
echo "  cd $PROJECT_ROOT/hailo-apps && source setup_env.sh && cd .."
echo "  python pi_dashboard.py"
echo ""
echo "Use --no-camera for review-only mode without the detection pipeline."
echo "To start it at every boot instead: bash scripts/install_service.sh"
if [ "$NEED_REBOOT" = 1 ]; then
    echo ""
    echo "[INFO] Reboot once to apply the swap setting: sudo reboot"
    echo "       Then run this script again to remove the old swap file."
fi
