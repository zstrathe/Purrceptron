#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Start the dashboard at boot, at http://<web_ui.local_name>.local (default
# http://cats.local) as well as http://<hostname>.local:<port>.
# Runs on the Pi, after scripts/setup_pi.sh. Safe to run again.
#
# Usage: bash scripts/install_service.sh            install, enable and start
#        bash scripts/install_service.sh --restart  same, and restart a running dashboard
#        bash scripts/install_service.sh --remove   stop, disable and remove
#
# Installs three systemd services:
#   cat-dashboard        runs pi_dashboard.py in the hailo-apps venv as the
#                        user who ran this script; restarts it if it crashes;
#                        output to logs/pi_dashboard.out (rotated by setup_pi.sh)
#   cat-dashboard-port80 forwards port 80 to the dashboard's port (nftables)
#   cat-dashboard-name   announces web_ui.local_name on the network (Avahi);
#                        not installed when local_name is blank
#
#   systemctl status cat-dashboard          is it running?
#   sudo systemctl restart cat-dashboard    load new code after sync_to_pi.sh
#   sudo systemctl stop cat-dashboard       stop it until the next boot
# ═══════════════════════════════════════════════════════════════

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME=cat-dashboard
UNIT_DIR=/etc/systemd/system
NFT_FILE=/etc/cat-detection/port80.nft
LOG="$PROJECT_ROOT/logs/pi_dashboard.out"

remove_unit() {    # remove_unit SERVICE
    sudo systemctl disable --now "$1" 2>/dev/null || true
    sudo rm -f "$UNIT_DIR/$1.service"
}

if [ "${1:-}" = --remove ]; then
    remove_unit "$NAME-name"
    remove_unit "$NAME-port80"
    remove_unit "$NAME"
    sudo rm -f "$NFT_FILE"
    sudo systemctl daemon-reload
    echo "[OK] Removed the $NAME services. Start the dashboard by hand from now on."
    exit 0
fi
RESTART=0
case "${1:-}" in
    "") ;;
    --restart) RESTART=1 ;;     # restart a running dashboard (new code)
    *) sed -n '10,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

[ "$(id -u)" != 0 ] || { echo "[ERROR] Run this as your normal user, not with sudo" >&2; exit 1; }
[ -f "$PROJECT_ROOT/hailo-apps/setup_env.sh" ] \
    || { echo "[ERROR] hailo-apps is not installed; run scripts/setup_pi.sh first" >&2; exit 1; }

# web_ui.port and web_ui.local_name from project.yaml ("-" when blank).
read -r PORT LOCAL_NAME < <(python3 - "$PROJECT_ROOT/configs/project.yaml" <<'EOF' 2>/dev/null || echo "5003 cats"
import sys, yaml
web = (yaml.safe_load(open(sys.argv[1])) or {}).get("web_ui") or {}
name = str(web.get("local_name", "cats") or "").strip().removesuffix(".local")
print(int(web.get("port", 5003)), name or "-")
EOF
)
[ "$LOCAL_NAME" = - ] && LOCAL_NAME=""

UNIT_CHANGED=0
install_unit() {    # install_unit SERVICE TEXT -> writes, enables; sets UNIT_CHANGED=1 if it changed
    local unit="$UNIT_DIR/$1.service"
    UNIT_CHANGED=0
    if sudo test -f "$unit" && [ "$(sudo cat "$unit")" = "$2" ]; then
        echo "[OK] $1 service already installed"
    else
        printf '%s\n' "$2" | sudo tee "$unit" > /dev/null
        sudo systemctl daemon-reload
        echo "[INFO] Installed the $1 service"
        UNIT_CHANGED=1
    fi
    sudo systemctl enable "$1" 2>/dev/null
}

# ── Port 80 -> the dashboard's port ─────────────────────────────────
# A redirect instead of a web server: nothing else to run or update, and the
# dashboard's own port keeps working.
if ! command -v nft >/dev/null && [ ! -x /usr/sbin/nft ]; then
    echo "[WARN] nftables is not installed; the dashboard stays on port $PORT (sudo apt install nftables, then run this again)."
    remove_unit "$NAME-port80"
elif ss -Hltn "sport = :80" 2>/dev/null | grep -q .; then
    echo "[WARN] Something already listens on port 80; not forwarding it to the dashboard."
    remove_unit "$NAME-port80"
else
    NFT_TEXT="# Written by scripts/install_service.sh: port 80 -> the dashboard (port $PORT).
table inet cat_dashboard
delete table inet cat_dashboard
table inet cat_dashboard {
    chain prerouting {
        type nat hook prerouting priority dstnat; policy accept;
        fib daddr type local tcp dport 80 redirect to :$PORT
    }
    chain output {
        type nat hook output priority dstnat; policy accept;
        fib daddr type local tcp dport 80 redirect to :$PORT
    }
}"
    sudo mkdir -p "$(dirname "$NFT_FILE")"
    printf '%s\n' "$NFT_TEXT" | sudo tee "$NFT_FILE" > /dev/null
    install_unit "$NAME-port80" "[Unit]
Description=Forward port 80 to the cat dashboard (port $PORT)
After=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/sbin/nft -f $NFT_FILE
ExecStop=/usr/sbin/nft delete table inet cat_dashboard

[Install]
WantedBy=multi-user.target"
    if sudo systemctl restart "$NAME-port80"; then
        echo "[OK] Port 80 goes to the dashboard"
    else
        echo "[WARN] Could not forward port 80 (see: journalctl -u $NAME-port80); the dashboard stays on port $PORT."
        remove_unit "$NAME-port80"
    fi
fi

# ── Second network name ─────────────────────────────────────────────
if [ -n "$LOCAL_NAME" ]; then
    if ! command -v avahi-publish >/dev/null; then
        echo "[INFO] Installing avahi-utils (announces $LOCAL_NAME.local)..."
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq avahi-utils >/dev/null
    fi
    install_unit "$NAME-name" "[Unit]
Description=Announce $LOCAL_NAME.local for the cat dashboard
Requires=avahi-daemon.service
After=avahi-daemon.service network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
ExecStart=/bin/bash $PROJECT_ROOT/scripts/publish_local_name.sh $LOCAL_NAME.local
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target"
    if sudo systemctl restart "$NAME-name"; then
        echo "[OK] This Pi also answers to $LOCAL_NAME.local"
    else
        echo "[WARN] Could not announce $LOCAL_NAME.local (see: journalctl -u $NAME-name)."
    fi
    URL="http://$LOCAL_NAME.local"
else
    remove_unit "$NAME-name"
    URL="http://$(hostname).local"
fi
sudo systemctl daemon-reload
systemctl is-active --quiet "$NAME-port80" 2>/dev/null || URL="$URL:$PORT"

# ── The dashboard ───────────────────────────────────────────────────
# Create the log file as this user. systemd would create it as root, and then
# logrotate (which runs as this user for it) couldn't truncate it.
mkdir -p "$(dirname "$LOG")"
touch "$LOG"

# GStreamer warns without a runtime dir. It exists while the user is logged
# in (or has lingering on); the desktop image logs in at boot.
XDG_LINE=""
[ -d "/run/user/$(id -u)" ] && XDG_LINE="Environment=XDG_RUNTIME_DIR=/run/user/$(id -u)"

install_unit "$NAME" "[Unit]
Description=Cat detection dashboard (pi_dashboard.py)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$(id -un)
Group=$(id -gn)
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONUNBUFFERED=1
$XDG_LINE
# setup_env.sh finds the venv relative to the folder it's sourced from.
ExecStart=/bin/bash -c 'cd hailo-apps && source setup_env.sh && cd .. && exec python pi_dashboard.py'
StandardOutput=append:$LOG
StandardError=inherit
# Ctrl+C (SIGINT) makes pi_dashboard.py stop the detection pipeline cleanly.
KillSignal=SIGINT
TimeoutStopSec=20
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target"
echo "[OK] The dashboard starts at boot"

if systemctl is-active --quiet "$NAME"; then
    if [ "$RESTART" = 1 ] || [ "$UNIT_CHANGED" = 1 ]; then
        echo "[INFO] Restarting the dashboard to load the new code and settings..."
        sudo systemctl stop "$NAME"
    else
        echo "[OK] $NAME is running (restart it with: sudo systemctl restart $NAME)"
        echo "[OK] Dashboard: $URL"
        exit 0
    fi
fi

# A dashboard started by hand would hold the port, and the service would fail.
holder=$(ss -Hltnp "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)
if [ -n "$holder" ] || ss -Hltn "sport = :$PORT" 2>/dev/null | grep -q .; then
    echo "[WARN] Port $PORT is already in use${holder:+ by process $holder ($(ps -o args= -p "$holder" | cut -c1-60))}."
    echo "       If that's a dashboard you started by hand, stop it${holder:+ (kill $holder)}, then:"
    echo "         sudo systemctl start $NAME"
    echo "       Otherwise the service starts at the next boot."
    exit 0
fi

sudo systemctl start "$NAME"
echo "[INFO] Starting the dashboard (loading takes a little while)..."
start=$SECONDS
for _ in $(seq 1 24); do
    if python3 -c "import urllib.request,sys; urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/api/status', timeout=5)" "$PORT" 2>/dev/null; then
        echo "[OK] Dashboard (up after $((SECONDS - start)) s): $URL"
        exit 0
    fi
    case "$(systemctl is-active "$NAME" || true)" in
        active|activating) sleep 5 ;;
        *) break ;;    # it exited; Restart= tries again in 10 s, but say so now
    esac
done
echo "[WARN] The dashboard isn't answering. Check:"
echo "         systemctl status $NAME"
echo "         tail -50 $LOG"
exit 1
