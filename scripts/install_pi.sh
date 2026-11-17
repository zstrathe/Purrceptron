#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Set up a freshly flashed Pi for this project. Runs on the PC, over SSH.
# (To move an old Pi's images and settings onto a new card, use restore_pi.sh.)
#
# Usage: ./scripts/install_pi.sh [--steps a,b] [--yes] [--force] [user@host]
#
# The target defaults to PI_USER@PI_HOST from .env. The code, including
# configs/project.yaml, comes from this checkout.
#
# Steps, in this order (all by default; every step can be run again):
#   check    SSH works; it's a Pi 5 with 64-bit Raspberry Pi OS and room to spare
#   system   apt full-upgrade and the Hailo runtime (hailo-h10-all); reboots if needed
#   code     copy the code from this PC (like sync_to_pi.sh)
#   setup    scripts/setup_pi.sh: hailo-apps, Python packages, SD card settings;
#            reboots once for the swap setting, then runs it again
#   service  start the dashboard at every boot (asks first)
#   verify   check the Hailo device, the Python packages, the SD card settings
#            and that the dashboard answers
#
# Before running: flash the card with Raspberry Pi Imager (Raspberry Pi OS
# 64-bit). In its settings set a hostname and user name, Wi-Fi and the Wi-Fi
# country, and turn on SSH. Either paste this PC's key (~/.ssh/id_ed25519.pub)
# for public-key login, or use a password and this script offers ssh-copy-id.
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
source "$SCRIPT_DIR/_pi_remote.sh"

ALL_STEPS="check system code setup service verify"
STEPS="$ALL_STEPS"
YES=0
FORCE=0
TARGET=""
ENV_HAS_HOST="${PI_HOST:+yes}"

usage() { sed -n '10,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --steps) STEPS="${2:?--steps needs a list}"; STEPS="${STEPS//,/ }"; shift 2 ;;
        --yes|-y) YES=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "[ERROR] Unknown option $1 (see --help)" >&2; exit 1 ;;
        *) TARGET="$1"; shift ;;
    esac
done
TARGET="${TARGET:-${PI_USER:?Set PI_USER in .env or pass user@host}@${PI_HOST:?Set PI_HOST in .env or pass user@host}}"
[[ "$TARGET" == *@* ]] || { echo "[ERROR] Give the Pi as user@host" >&2; exit 1; }
REMOTE_DIR="${PI_REMOTE_DIR:-~/Repos/cat_detection}"
# An unquoted ~ in .env is expanded to the PC's home when .env is loaded; mean
# the Pi's home instead (connect() fills it in).
REMOTE_DIR="${REMOTE_DIR/#$HOME\//\~/}"


for s in $STEPS; do
    [[ " $ALL_STEPS " == *" $s "* ]] || die "Unknown step '$s'. Steps: $ALL_STEPS"
done

# The Pi gets this PC's project.yaml, so it has to exist and name the cats.
if [[ " $STEPS " == *" code "* ]] && [ ! -f "$PROJECT_ROOT/configs/project.yaml" ]; then
    cp "$PROJECT_ROOT/configs/project.yaml.example" "$PROJECT_ROOT/configs/project.yaml"
    echo "[INFO] Created configs/project.yaml from the example."
    echo "       Edit it first: list your cats under \"cats:\" and set camera.device."
    echo "       Then run this again."
    exit 1
fi

# ── Main ─────────────────────────────────────────────────────────────
connect
echo "[INFO] Pi: $TARGET:$REMOTE_DIR"
rc=0
for s in $ALL_STEPS; do
    [[ " $STEPS " == *" $s "* ]] || continue
    echo ""
    echo "══ $s ══"
    # Called plainly so set -e stops at the first failing command inside the
    # step (a function called as `f || ...` would run on past failures).
    if [ "$s" = verify ]; then
        verify_pi
        if [ "$fails" -gt 0 ]; then
            echo "[WARN] $fails check(s) failed"
            rc=1
        else
            echo "[OK] Everything checks out"
        fi
    else
        "step_$s"
    fi
done

# Let sync_to_pi.sh and the other scripts find the Pi next time.
if [ -z "$ENV_HAS_HOST" ] && [ -t 0 ] \
   && ask "Save the Pi's address in .env (PI_USER, PI_HOST, PI_REMOTE_DIR) for the other scripts?"; then
    {
        echo ""
        echo "PI_USER=${TARGET%@*}"
        echo "PI_HOST=$PI_NAME"
        # Quoted so the ~ means the Pi's home, not this PC's.
        [ -n "${PI_REMOTE_DIR:-}" ] || echo "PI_REMOTE_DIR='~/Repos/cat_detection'"
    } >> "$PROJECT_ROOT/.env"
    echo "[OK] Added to .env"
fi

if [[ " $STEPS " == *" setup "* ]] && [ "$rc" = 0 ]; then
    echo ""
    echo "[OK] The Pi is set up."
    dashboard_howto
    cat <<'EOF'
Next, in the dashboard (the README's Quick start has the details):
  - Capture tab: when a cat is eating, tap its name under "Who is this?" (or turn
    on "Save images automatically") until each cat has 300 or more images.
  - Label tab: check which cat each image shows, and give images without a
    bounding box one.
  - Training tab: set up where training runs (Local PC: run scripts/worker.sh on
    this PC; or Hugging Face Jobs), Start training, then "Deploy newest model".
  - Backup tab: choose where nightly backups go.
EOF
elif [ "$rc" = 0 ]; then
    echo ""
    dashboard_howto
fi
exit "$rc"
