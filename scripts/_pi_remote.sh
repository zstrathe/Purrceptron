#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

# Shared by install_pi.sh and restore_pi.sh, which run on the PC and set up a
# Pi over SSH. Source it after _common.sh, once these are set:
#   TARGET      user@host of the Pi
#   REMOTE_DIR  project folder on the Pi (a leading ~ means the Pi's home)
#   YES, FORCE  1 for --yes / --force
#
# Provides the SSH helpers, the steps both scripts share (check, system, code,
# setup, service) and verify_pi, the checks of the Pi itself.

die() { echo "[ERROR] $*" >&2; exit 1; }
q() { printf '%q' "$1"; }

# ── SSH helpers ──────────────────────────────────────────────────────
# accept-new: trust a host key the first time (a new card has a new one) but
# still refuse a key that changed, which is handled with a clear message below.
SSH_OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
          -o StrictHostKeyChecking=accept-new)
[ -n "${PI_SSH_PORT:-}" ] && SSH_OPTS+=(-p "$PI_SSH_PORT")
PI_NAME=""          # the host name as given, for messages and the dashboard URL
SERVICE_ON=0        # set by step_service
rsh() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }
# With a terminal: shows progress as it happens and lets sudo ask for a password.
rsh_tty() { ssh -t "${SSH_OPTS[@]}" "$TARGET" "$@"; }

ask() {    # ask "question" -> 0 for yes
    [ "$YES" = 1 ] && return 0
    local a
    read -rp "[ASK] $1 [Y/n] " a
    [[ ! "$a" =~ ^[Nn] ]]
}

has_ssh_key() {
    local k
    for k in ~/.ssh/id_ed25519.pub ~/.ssh/id_ecdsa.pub ~/.ssh/id_rsa.pub; do
        [ -f "$k" ] && return 0
    done
    ssh-add -l >/dev/null 2>&1
}

pub_key() {
    local k
    for k in ~/.ssh/id_ed25519.pub ~/.ssh/id_ecdsa.pub ~/.ssh/id_rsa.pub; do
        [ -f "$k" ] && { cat "$k"; return 0; }
    done
}

try_ssh() { ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$TARGET" true 2>&1; }

connect() {
    local err
    PI_NAME="${TARGET#*@}"
    # Look the name up once and use the address from then on.
    pi_ssh_target "$TARGET"
    TARGET="$PI_SSH_DEST"
    SSH_OPTS+=("${PI_SSH_ALIAS[@]}")
    RSYNC_SSH="ssh ${SSH_OPTS[*]}"

    if ! has_ssh_key; then
        echo "[INFO] This PC has no SSH key yet; the Pi lets this PC in with one."
        if [ -t 0 ] && ask "Create one now (ssh-keygen -t ed25519)?"; then
            ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
        else
            die "Create a key with: ssh-keygen -t ed25519"
        fi
    fi

    if ! err=$(try_ssh); then
        if grep -q "REMOTE HOST IDENTIFICATION HAS CHANGED" <<<"$err"; then
            echo "[ERROR] $PI_NAME has a new SSH host key (expected after flashing a new card)." >&2
            echo "        Forget the old one, then run this again:" >&2
            echo "          ssh-keygen -R $PI_NAME" >&2
            exit 1
        elif grep -qi "permission denied" <<<"$err"; then
            echo "[INFO] The Pi doesn't accept this PC's SSH key yet."
            if [ -t 0 ] && ask "Add it now? (asks for the Pi user's password once)"; then
                ssh-copy-id "${SSH_OPTS[@]}" "$TARGET" || die "ssh-copy-id failed"
                err=$(try_ssh) || die "The Pi still refuses the key: $err"
            else
                echo "[ERROR] Either add it with:  ssh-copy-id $TARGET" >&2
                echo "        or, in Raspberry Pi Imager's settings, turn on SSH with public-key" >&2
                echo "        login, paste this PC's key and flash again:" >&2
                echo "          $(pub_key)" >&2
                exit 1
            fi
        else
            echo "[ERROR] Cannot reach $TARGET over SSH:" >&2
            sed 's/^/        /' <<<"$err" >&2
            echo "        Is the Pi on and on the network? Is the hostname right?" >&2
            echo "        (You can also give its address: user@192.168.x.y)" >&2
            exit 1
        fi
    fi
    local remote_home
    remote_home=$(rsh 'printf %s "$HOME"')
    REMOTE_DIR="${REMOTE_DIR/#\~/$remote_home}"
}

reboot_and_wait() {    # reboot_and_wait "reason"
    local old new
    ask "Reboot the Pi now ($1)?" \
        || die "Reboot the Pi yourself, then run this again with --steps for the remaining steps"
    old=$(rsh cat /proc/sys/kernel/random/boot_id)
    rsh_tty 'sudo systemctl reboot' >/dev/null 2>&1 || true
    echo "[INFO] Waiting for the Pi to come back..."
    for _ in $(seq 1 72); do
        sleep 5
        new=$(ssh "${SSH_OPTS[@]}" -o BatchMode=yes -o ConnectTimeout=5 "$TARGET" \
              cat /proc/sys/kernel/random/boot_id 2>/dev/null) || continue
        if [ -n "$new" ] && [ "$new" != "$old" ]; then
            echo "[OK] The Pi is back"
            return 0
        fi
    done
    die "The Pi did not come back within 6 minutes"
}

# hailortcli fw-control identify exits 0 even when there is no device (it just
# prints nothing), so look for the device node and the architecture it reports.
hailo_ok() {
    rsh 'test -e /dev/hailo0 && hailortcli fw-control identify 2>/dev/null | grep -q "Device Architecture"'
}

# A kernel newer than the running one is installed (Raspberry Pi OS doesn't
# always flag /run/reboot-required after a kernel update). Compares kernels of
# the same flavor (rpi-2712 on a Pi 5).
newer_kernel_installed() {
    rsh 'test -f /run/reboot-required ||
         { k=$(uname -r); newest=$(ls /lib/modules | grep -- "-${k##*-}\$" | sort -V | tail -1);
           [ -n "$newest" ] && [ "$newest" != "$k" ]; }'
}

# web_ui settings from the PC's project.yaml (what the Pi gets).
web_ui_setting() {    # web_ui_setting KEY DEFAULT
    local v
    v=$(sed -n "/^web_ui:/,/^[^ #]/s/^ \{1,\}$1: *\([^ #]*\).*/\1/p" \
        "$PROJECT_ROOT/configs/project.yaml" 2>/dev/null | head -1)
    if grep -qE "^ +$1:" <(sed -n '/^web_ui:/,/^[^ #]/p' "$PROJECT_ROOT/configs/project.yaml" 2>/dev/null); then
        echo "$v"           # set (maybe blank)
    else
        echo "$2"
    fi
}
dashboard_port() { local p; p=$(web_ui_setting port 5003); echo "${p:-5003}"; }

# ── Shared steps ─────────────────────────────────────────────────────
step_check() {
    local info os arch model hailo sudo_ok free_gb disk has_rsync
    info=$(rsh 'bash -s' <<'EOF'
. /etc/os-release
echo "os=${VERSION_CODENAME:-?}"
echo "arch=$(uname -m)"
echo "model=$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"
echo "hailo=$( (lspci -d 1e60: 2>/dev/null || true) | head -1)"
echo "sudo=$(sudo -n true 2>/dev/null && echo ok || echo password)"
echo "free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)"
src=$(findmnt -no SOURCE /)
echo "disk=$(lsblk -dno SIZE,MODEL "$(sed -E 's/p?[0-9]+$//' <<<"$src")" 2>/dev/null | xargs)"
echo "rsync=$(command -v rsync >/dev/null && echo yes || echo no)"
EOF
)
    os=$(sed -n 's/^os=//p' <<<"$info");       arch=$(sed -n 's/^arch=//p' <<<"$info")
    model=$(sed -n 's/^model=//p' <<<"$info"); hailo=$(sed -n 's/^hailo=//p' <<<"$info")
    sudo_ok=$(sed -n 's/^sudo=//p' <<<"$info"); free_gb=$(sed -n 's/^free_gb=//p' <<<"$info")
    disk=$(sed -n 's/^disk=//p' <<<"$info");   has_rsync=$(sed -n 's/^rsync=//p' <<<"$info")
    echo "[INFO] $TARGET: ${model:-unknown model}, Raspberry Pi OS $os ($arch), card $disk, ${free_gb} GB free"
    local bad=0
    [ "$arch" = aarch64 ] || { echo "[ERROR] Needs 64-bit Raspberry Pi OS (found $arch)"; bad=1; }
    [[ "$model" == *"Raspberry Pi 5"* ]] || { echo "[ERROR] Expected a Raspberry Pi 5"; bad=1; }
    [ "$os" = trixie ] || { echo "[ERROR] Expected Raspberry Pi OS trixie (found $os);" \
        "the Hailo packages and hailo-apps commit were tested on trixie"; bad=1; }
    [ "${free_gb:-0}" -ge 10 ] || { echo "[ERROR] Only ${free_gb} GB free; needs 10 GB"; bad=1; }
    if [ "$bad" = 1 ]; then
        [ "$FORCE" = 1 ] && echo "[WARN] Continuing anyway (--force)" || die "Use --force to continue anyway"
    fi
    if [ -n "$hailo" ]; then
        echo "[OK] Hailo device on PCIe: $hailo"
    else
        echo "[WARN] No Hailo device seen on PCIe yet (normal before the system step if lspci" \
             "is missing; otherwise check the AI HAT+ 2 ribbon cable)"
    fi
    [ "$sudo_ok" = ok ] || echo "[INFO] sudo on the Pi asks for a password; you'll be asked for it"
    [ "$has_rsync" = yes ] || echo "[INFO] rsync is not on the Pi yet; the system step installs it"
}

step_system() {
    echo "[INFO] Updating Raspberry Pi OS..."
    rsh_tty 'set -e
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y'
    # The Hailo driver package compiles its kernel module for the running
    # kernel only. So boot the updated kernel first, and install dkms before
    # it: with dkms the module is also rebuilt after every later kernel update.
    if newer_kernel_installed; then
        reboot_and_wait "to boot the updated kernel before building the Hailo driver"
    fi
    echo "[INFO] Installing the Hailo runtime..."
    rsh_tty 'set -e
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y dkms git rsync sqlite3 python3-venv python3-pip
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y hailo-h10-all
        # Installed earlier without dkms: build it again so dkms tracks it.
        if ! /usr/sbin/dkms status hailo1x_pci 2>/dev/null | grep -q "$(uname -r).*installed"; then
            sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --reinstall h10-hailort-pcie-driver
        fi'
    if newer_kernel_installed || ! hailo_ok; then
        reboot_and_wait "to load the Hailo driver"
    fi
    if ! hailo_ok; then
        die "The Hailo device doesn't answer (no /dev/hailo0, or hailortcli fw-control identify" \
            "shows no device). Check the AI HAT+ 2 and its cable, and: sudo dkms status"
    fi
    local ver
    ver=$(rsh 'hailortcli --version 2>/dev/null' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
    echo "[OK] Hailo-10H answers; HailoRT ${ver:-unknown}; driver kept up to date by dkms"
    if [ "$ver" != "5.1.1" ]; then
        echo "[WARN] This project's hailo-apps commit was tested with HailoRT 5.1.1, not ${ver:-unknown}." \
             "If detection fails, set HAILO_APPS_COMMIT to a hailo-apps commit for that version."
    fi
}

DEPS_CHANGED=0
step_code() {
    echo "[INFO] Copying the code from this PC to $TARGET:$REMOTE_DIR ..."
    rsh "mkdir -p $(q "$REMOTE_DIR")"
    # Python packages are installed by the setup step, not by copying code:
    # note when the Pi's requirements are about to change.
    local mine theirs
    mine=$(sha256sum "$PROJECT_ROOT/requirements-pi.txt" | cut -d' ' -f1)
    theirs=$(rsh "sha256sum $(q "$REMOTE_DIR")/requirements-pi.txt 2>/dev/null" | cut -d' ' -f1 || true)
    if [ -n "$theirs" ] && [ "$mine" != "$theirs" ]; then
        # shellcheck disable=SC2034  # read by update_pi.sh
        DEPS_CHANGED=1
    fi
    # Same as sync_to_pi.sh. go-w: files on /mnt/c look world-writable to rsync.
    rsync -az --delete --chmod=go-w --exclude-from="$PROJECT_ROOT/.syncignore" \
        -e "$RSYNC_SSH" "$PROJECT_ROOT/" "$TARGET:$REMOTE_DIR/"
    echo "[OK] Code copied"
}

step_setup() {
    echo "[INFO] Running scripts/setup_pi.sh on the Pi (hailo-apps, Python packages, SD card settings)..."
    local log
    log=$(mktemp)
    rsh_tty "cd $(q "$REMOTE_DIR") && bash scripts/setup_pi.sh" | tee "$log"
    if grep -q "Reboot once to apply the swap setting" "$log"; then
        reboot_and_wait "to switch swap to RAM only"
        echo "[INFO] Running setup_pi.sh again to finish (removes the old swap file)..."
        rsh_tty "cd $(q "$REMOTE_DIR") && bash scripts/setup_pi.sh" | tee "$log"
    fi
    rm -f "$log"
}

service_installed() { rsh 'test -f /etc/systemd/system/cat-dashboard.service' 2>/dev/null; }

step_service() {
    # Already set up: just update it and load the new code, no question.
    if service_installed; then
        echo "[INFO] Updating the dashboard service and restarting it with the new code..."
        rsh_tty "cd $(q "$REMOTE_DIR") && bash scripts/install_service.sh --restart"
        SERVICE_ON=1
        return 0
    fi
    if ! ask "Start the dashboard automatically when the Pi boots (systemd service cat-dashboard)?"; then
        echo "[INFO] Skipped. Turn it on later with: bash scripts/install_service.sh (on the Pi)"
        return 0
    fi
    # --restart: the code was usually just copied over, so load it.
    rsh_tty "cd $(q "$REMOTE_DIR") && bash scripts/install_service.sh --restart"
    SERVICE_ON=1
}

# ── Checks of the Pi itself ──────────────────────────────────────────
# Callers count failures in $fails; pass/fail print one line each.
fails=0
pass() { echo "  [OK]   $*"; }
fail() { echo "  [FAIL] $*"; fails=$((fails + 1)); }

verify_pi() {
    echo "[INFO] Checking the Pi..."
    local report
    report=$(rsh "bash -s -- $(q "$REMOTE_DIR") $(dashboard_port)" <<'EOF'
cd "$1"
port="$2"
ok() { echo "OK|$*"; }
no() { echo "FAIL|$*"; }
info() { echo "INFO|$*"; }
for db in data/*.db; do
    [ -f "$db" ] || continue
    r=$(python3 -c "import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute('PRAGMA integrity_check').fetchone()[0])" "$db" 2>&1)
    [ "$r" = ok ] && ok "$db passes integrity_check" || no "$db: $r"
done
info "$(find data/raw -type f -name '*.jpg' 2>/dev/null | wc -l) images in data/raw"
out=$(hailortcli fw-control identify 2>&1 || true)
if [ -e /dev/hailo0 ] && grep -q "Device Architecture" <<<"$out"; then
    ok "Hailo device answers ($(grep -m1 -i 'firmware version' <<<"$out" | xargs))"
else
    no "Hailo device doesn't answer (no driver for kernel $(uname -r)? see: sudo dkms status)"
fi
if /usr/sbin/dkms status hailo1x_pci 2>/dev/null | grep -q "$(uname -r).*installed"; then
    ok "Hailo driver is built by dkms (rebuilt after kernel updates)"
else
    no "Hailo driver isn't managed by dkms; the next kernel update would break it (run the system step)"
fi
if [ -f hailo-apps/setup_env.sh ]; then
    r=$(cd hailo-apps && set +u && source setup_env.sh >/dev/null 2>&1 && python -c "
import importlib.metadata as m
import cv2, flask, hailo, hailo_apps, huggingface_hub, numpy, torch, yaml
from executorch.runtime import Runtime
print('torch', torch.__version__, '| executorch', m.version('executorch'), '| hailo-apps venv ok')
" 2>&1 | tail -1)
    [[ "$r" == *"venv ok"* ]] && ok "Python packages: $r" || no "Python packages: $r"
    [[ "$r" == *"executorch 1.1.0"* ]] || no "ExecuTorch on the Pi should be 1.1.0 (the version models are exported with)"
    c=$(git -C hailo-apps rev-parse --short HEAD 2>/dev/null)
    ok "hailo-apps at commit $c"
else
    no "hailo-apps is not installed (run the setup step)"
fi
[ "$(findmnt -no FSTYPE /tmp)" = tmpfs ] && ok "/tmp is in RAM" || no "/tmp is not a RAM disk"
sw=$(awk 'NR > 1 {print $1}' /proc/swaps | xargs)
if ! tr ' ' '\n' <<<"$sw" | grep -v '^$' | grep -qv '^/dev/zram'; then   # nothing but zram
    ok "swap is RAM only (${sw:-none})"
else
    no "swap still uses the card: $sw (reboot, then run the setup step again)"
fi
for f in /etc/rpi/swap.conf.d/80-cat-detection.conf /etc/systemd/journald.conf.d/80-cat-detection.conf \
         /etc/logrotate.d/cat-detection; do
    [ -f "$f" ] && ok "$f installed" || no "$f missing (run the setup step)"
done
if [ -f /etc/systemd/system/cat-dashboard.service ]; then
    [ "$(systemctl is-enabled cat-dashboard 2>/dev/null)" = enabled ] \
        && ok "cat-dashboard service starts at boot" || no "cat-dashboard service is not enabled"
    [ "$(systemctl is-active cat-dashboard 2>/dev/null)" = active ] \
        && ok "cat-dashboard service is running" \
        || no "cat-dashboard service is not running (see: journalctl -u cat-dashboard; tail logs/pi_dashboard.out)"
    for unit in cat-dashboard-port80 cat-dashboard-name; do
        [ -f "/etc/systemd/system/$unit.service" ] || continue
        [ "$(systemctl is-active "$unit" 2>/dev/null)" = active ] \
            && ok "$unit service is running" || no "$unit service is not running (systemctl status $unit)"
    done
else
    info "No cat-dashboard service (the dashboard is started by hand)"
fi
answers() {
    python3 -c "import urllib.request,sys; urllib.request.urlopen(f'http://127.0.0.1:{sys.argv[1]}/api/status', timeout=5)" "$port" 2>/dev/null
}
if [ "$(systemctl is-active cat-dashboard 2>/dev/null)" = active ]; then
    # It may still be starting (loading PyTorch takes a while on a Pi).
    for _ in $(seq 1 12); do answers && break; sleep 5; done
    answers && ok "dashboard answers on port $port" || no "dashboard doesn't answer on port $port"
    if [ "$(systemctl is-active cat-dashboard-port80 2>/dev/null)" = active ]; then
        port=80
        answers && ok "dashboard answers on port 80" || no "port 80 doesn't reach the dashboard"
    fi
elif answers; then
    ok "dashboard answers on port $port"
else
    info "dashboard is not running"
fi
EOF
)
    local status msg
    while IFS='|' read -r status msg; do
        case "$status" in
            OK) pass "$msg" ;;
            FAIL) fail "$msg" ;;
            INFO) echo "  [INFO] $msg" ;;
        esac
    done <<<"$report"
}

# How to reach the dashboard and keep it running, for the scripts' last words.
dashboard_howto() {
    local port url name
    port=$(dashboard_port)
    url="http://$PI_NAME:$port"
    [ "$PI_NAME" = "${TARGET#*@}" ] || url="$url  (or http://${TARGET#*@}:$port)"
    if [ "$SERVICE_ON" = 1 ] || rsh 'test -f /etc/systemd/system/cat-dashboard.service' 2>/dev/null; then
        name=$(web_ui_setting local_name "")
        name="${name%.local}"
        rsh 'systemctl is-active --quiet cat-dashboard-port80' 2>/dev/null && [ -n "$name" ] \
            && url="http://$name.local  (or $url)"
        cat <<EOF
[OK] Dashboard: $url
It runs as a service and starts at every boot.
  Logs:      ssh $TARGET tail -f $REMOTE_DIR/logs/pi_dashboard.out
  Restart:   ssh $TARGET sudo systemctl restart cat-dashboard   (after sync_to_pi.sh)
EOF
    else
        cat <<EOF
Start the dashboard on the Pi:
  ssh $TARGET
  cd $REMOTE_DIR/hailo-apps && source setup_env.sh && cd ..
  nohup python pi_dashboard.py > logs/pi_dashboard.out 2>&1 &
Then open $url
EOF
    fi
}
