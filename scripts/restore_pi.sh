#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
# Set up a freshly flashed Pi and restore the project onto it. Runs on the PC.
#
# Usage: ./scripts/restore_pi.sh --from DIR [--steps a,b] [--yes] [--force] [user@host]
#
# DIR is either
#   - a folder written by scripts/restore_backup.py (one backup snapshot), or
#   - a manual copy of the Pi with repo/, home/ and snapshots/ in it.
# The code comes from this checkout (like sync_to_pi.sh). Data, models, logs
# and the Pi-only settings (configs/runtime.yaml, configs/secrets.yaml) come
# from DIR. The target defaults to PI_USER@PI_HOST from .env.
#
# Steps, in this order (all by default; every step can be run again):
#   check   SSH works; it's a Pi 5 with 64-bit Raspberry Pi OS and room to spare
#   system  apt full-upgrade and the Hailo runtime (hailo-h10-all); reboots if needed
#   code    copy the code from this PC
#   data    copy data/, models/, logs/ and the Pi-only settings from DIR
#   home    shell files and SSH keys from DIR/home (manual copies only)
#   setup   scripts/setup_pi.sh: hailo-apps, Python packages, SD card settings;
#           reboots once for the swap setting, then runs it again
#   verify  compare the restored files with DIR byte for byte; check the database,
#           the Hailo device, the Python packages and the SD card settings
#
# Before running: flash the card with Raspberry Pi Imager (Raspberry Pi OS 64-bit),
# and in its settings use the old hostname and user name, turn on SSH with
# public-key login and paste this PC's key (~/.ssh/id_ed25519.pub), and set Wi-Fi.
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ALL_STEPS="check system code data home setup verify"
FROM=""
STEPS="$ALL_STEPS"
YES=0
FORCE=0
TARGET=""

usage() { sed -n '10,34p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "[ERROR] $*" >&2; exit 1; }
q() { printf '%q' "$1"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="${2:?--from needs a folder}"; shift 2 ;;
        --steps) STEPS="${2:?--steps needs a list}"; STEPS="${STEPS//,/ }"; shift 2 ;;
        --yes|-y) YES=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) die "Unknown option $1 (see --help)" ;;
        *) TARGET="$1"; shift ;;
    esac
done
for s in $STEPS; do
    [[ " $ALL_STEPS " == *" $s "* ]] || die "Unknown step '$s'. Steps: $ALL_STEPS"
done
TARGET="${TARGET:-${PI_USER:?Set PI_USER in .env or pass user@host}@${PI_HOST:?Set PI_HOST in .env or pass user@host}}"
REMOTE_DIR="${PI_REMOTE_DIR:-~/Repos/cat_detection}"
# An unquoted ~ in .env is expanded to the PC's home when .env is loaded; mean
# the Pi's home instead (connect() fills it in).
REMOTE_DIR="${REMOTE_DIR/#$HOME\//\~/}"

# ── Backup layout ────────────────────────────────────────────────────
# SRC: project-shaped folder (data/, models/, logs/, configs/)
# HOME_SRC: the old home folder (manual copies only)
# DB_COPIES: consistent database copies that replace the ones under SRC/data
needs_from() { [ -n "$FROM" ] || die "--from DIR is required for the '$1' step"; }
SRC="" HOME_SRC="" DB_COPIES=()
if [ -n "$FROM" ]; then
    FROM="$(cd "$FROM" && pwd)" || die "No folder $FROM"
    if [ -d "$FROM/repo/data" ]; then
        SRC="$FROM/repo"
        [ -d "$FROM/home" ] && HOME_SRC="$FROM/home"
        for db in "$FROM"/snapshots/*.db; do
            [ -f "$db" ] && DB_COPIES+=("$db")
        done
    elif [ -d "$FROM/data" ]; then
        SRC="$FROM"
    else
        die "$FROM has neither data/ nor repo/data/; point --from at a backup"
    fi
    [ -d "$SRC/data/raw" ] || die "$SRC/data/raw is missing; is this the right backup?"
fi

# ── SSH helpers ──────────────────────────────────────────────────────
# accept-new: trust a host key the first time (a new card has a new one) but
# still refuse a key that changed, which is handled with a clear message below.
SSH_OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4
          -o StrictHostKeyChecking=accept-new)
[ -n "${PI_SSH_PORT:-}" ] && SSH_OPTS+=(-p "$PI_SSH_PORT")
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
rsh() { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }
# With a terminal: shows progress as it happens and lets sudo ask for a password.
rsh_tty() { ssh -t "${SSH_OPTS[@]}" "$TARGET" "$@"; }

connect() {
    local err host="${TARGET#*@}"
    if ! err=$(ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$TARGET" true 2>&1); then
        if grep -q "REMOTE HOST IDENTIFICATION HAS CHANGED" <<<"$err"; then
            echo "[ERROR] $host has a new SSH host key (expected after flashing a new card)." >&2
            echo "        Forget the old one, then run this again:" >&2
            echo "          ssh-keygen -R $host" >&2
        elif grep -qi "permission denied" <<<"$err"; then
            echo "[ERROR] The Pi refused this PC's SSH key." >&2
            echo "        In Raspberry Pi Imager's settings, turn on SSH with public-key login" >&2
            echo "        and paste this PC's key, then flash again:" >&2
            for k in ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub; do
                [ -f "$k" ] && { echo "          $(cat "$k")" >&2; break; }
            done
            echo "        (or add it now with: ssh-copy-id $TARGET)" >&2
        else
            echo "[ERROR] Cannot reach $TARGET over SSH:" >&2
            sed 's/^/        /' <<<"$err" >&2
            echo "        Is the Pi on and on the network? Is the hostname right?" >&2
        fi
        exit 1
    fi
    local remote_home
    remote_home=$(rsh 'printf %s "$HOME"')
    REMOTE_DIR="${REMOTE_DIR/#\~/$remote_home}"
}

ask() {    # ask "question" -> 0 for yes
    [ "$YES" = 1 ] && return 0
    local a
    read -rp "[ASK] $1 [Y/n] " a
    [[ ! "$a" =~ ^[Nn] ]]
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

hailo_ok() { rsh 'hailortcli fw-control identify' >/dev/null 2>&1; }

# Excludes for the data copy: database copies taken separately, and SQLite's
# side files, which belong to a live database.
data_excludes() {
    local db
    printf -- '--exclude=*.db-wal\n--exclude=*.db-shm\n--exclude=*.db-journal\n'
    for db in "${DB_COPIES[@]}"; do
        printf -- '--exclude=/data/%s\n' "$(basename "$db")"
    done
}

# ── Steps ────────────────────────────────────────────────────────────
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
    echo "[INFO] Updating Raspberry Pi OS and installing the Hailo runtime..."
    rsh_tty 'set -e
        sudo apt-get update
        sudo DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
            hailo-h10-all git rsync sqlite3 python3-venv python3-pip'
    # Raspberry Pi OS doesn't always flag /run/reboot-required after a kernel
    # update, so also compare the running kernel with the newest installed one
    # of the same flavor (rpi-2712 on a Pi 5).
    if rsh 'test -f /run/reboot-required ||
            { k=$(uname -r); newest=$(ls /lib/modules | grep -- "-${k##*-}\$" | sort -V | tail -1);
              [ -n "$newest" ] && [ "$newest" != "$k" ]; }' || ! hailo_ok; then
        reboot_and_wait "to load the updated kernel and the Hailo driver"
    fi
    if ! hailo_ok; then
        die "The Hailo device doesn't answer (hailortcli fw-control identify). Check the AI HAT+ 2 and its cable."
    fi
    local ver
    ver=$(rsh 'hailortcli --version 2>/dev/null' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
    echo "[OK] Hailo-10H answers; HailoRT ${ver:-unknown}"
    if [ "$ver" != "5.1.1" ]; then
        echo "[WARN] This project's hailo-apps commit was tested with HailoRT 5.1.1, not ${ver:-unknown}." \
             "If detection fails, set HAILO_APPS_COMMIT to a hailo-apps commit for that version."
    fi
}

step_code() {
    echo "[INFO] Copying the code from this PC to $TARGET:$REMOTE_DIR ..."
    rsh "mkdir -p $(q "$REMOTE_DIR")"
    # Same as sync_to_pi.sh. go-w: files on /mnt/c look world-writable to rsync.
    rsync -az --delete --chmod=go-w --exclude-from="$PROJECT_ROOT/.syncignore" \
        -e "$RSYNC_SSH" "$PROJECT_ROOT/" "$TARGET:$REMOTE_DIR/"
    echo "[OK] Code copied"
}

step_data() {
    needs_from data
    # Don't mix the backup into a Pi that already collected new images, unless
    # it's this script's own earlier (maybe interrupted) restore.
    local state
    state=$(rsh "bash -s -- $(q "$REMOTE_DIR")" <<'EOF'
if [ -f "$HOME/.cat_detection_restored" ]; then echo restored
elif find "$1/data/raw" -type f -name '*.jpg' 2>/dev/null | grep -q .; then echo has_images
else echo empty; fi
EOF
)
    if [ "$state" = has_images ] && [ "$FORCE" != 1 ]; then
        die "The Pi already has images in data/raw that this script didn't restore." \
            "Back them up first; --force adds the backup's files to them."
    fi
    local n
    n=$(find "$SRC/data/raw" -type f -name '*.jpg' | wc -l)
    echo "[INFO] Copying data, models and logs ($(du -sh "$SRC/data" | cut -f1) of data, $n images)..."
    rsh "mkdir -p $(q "$REMOTE_DIR")/configs $(q "$REMOTE_DIR")/data"
    local -a excl
    mapfile -t excl < <(data_excludes)
    local -a parts=()
    for d in data models logs; do [ -d "$SRC/$d" ] && parts+=("$SRC/$d"); done
    # Not --delete: never removes anything already on the Pi. Times are kept
    # (the crop cache and the backups use them to spot changed files).
    rsync -a --info=progress2 --chmod=D755,F644 "${excl[@]}" -e "$RSYNC_SSH" \
        "${parts[@]}" "$TARGET:$REMOTE_DIR/"
    local db
    for db in "${DB_COPIES[@]}"; do
        rsync -a --chmod=F644 -e "$RSYNC_SSH" "$db" "$TARGET:$REMOTE_DIR/data/$(basename "$db")"
        echo "[OK] Database $(basename "$db") restored from its consistent copy"
    done
    local f
    for f in runtime.yaml secrets.yaml; do
        if [ -f "$SRC/configs/$f" ]; then
            local mode=F644
            [ "$f" = secrets.yaml ] && mode=F600
            rsync -a --chmod="$mode" -e "$RSYNC_SSH" "$SRC/configs/$f" "$TARGET:$REMOTE_DIR/configs/$f"
            echo "[OK] configs/$f restored"
        elif [ "$f" = secrets.yaml ]; then
            echo "[INFO] This backup has no configs/secrets.yaml (cloud backups never do):" \
                 "enter the tokens again on the dashboard's Training and Backup tabs"
        fi
    done
    # project.yaml normally comes with the code; take the backup's only if the
    # Pi has none, and say so when the two differ.
    if [ -f "$SRC/configs/project.yaml" ]; then
        rsync -a --ignore-existing --chmod=F644 -e "$RSYNC_SSH" \
            "$SRC/configs/project.yaml" "$TARGET:$REMOTE_DIR/configs/project.yaml"
        if ! rsh "cat $(q "$REMOTE_DIR")/configs/project.yaml" | cmp -s - <(tr -d '\r' <"$SRC/configs/project.yaml"); then
            echo "[WARN] configs/project.yaml on the Pi (from this PC) differs from the backup's;" \
                 "the PC's is kept. Compare them if settings look wrong."
        fi
    fi
    rsh 'date -Is > ~/.cat_detection_restored'
    echo "[OK] Data restored"
}

step_home() {
    needs_from home
    if [ -z "$HOME_SRC" ]; then
        echo "[INFO] No home folder in this backup; skipping"
        return 0
    fi
    local -a items=()
    local f
    for f in .bashrc .profile .bash_history frigate; do
        [ -e "$HOME_SRC/$f" ] && items+=("$HOME_SRC/$f")
    done
    if [ ${#items[@]} -gt 0 ]; then
        # Files the new card already has are kept as <name>.before-restore.
        rsync -a --chmod=D755,F644 --backup --suffix=.before-restore -e "$RSYNC_SSH" \
            "${items[@]}" "$TARGET:"
        rsh 'chmod 600 ~/.bash_history 2>/dev/null || true'
        echo "[OK] Restored: $(for i in "${items[@]}"; do printf '%s ' "$(basename "$i")"; done)"
    fi
    if [ -f "$HOME_SRC/.ssh/authorized_keys" ]; then
        rsh 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
        rsync -a --chmod=F600 -e "$RSYNC_SSH" "$HOME_SRC/.ssh/authorized_keys" \
            "$TARGET:.ssh/authorized_keys.from-backup"
        rsh 'cd ~/.ssh && touch authorized_keys && chmod 600 authorized_keys &&
             awk "NF && !seen[\$0]++" authorized_keys authorized_keys.from-backup > authorized_keys.new &&
             mv authorized_keys.new authorized_keys && rm -f authorized_keys.from-backup'
        echo "[OK] SSH keys that could log in before can log in again"
    fi
    echo "[INFO] Left in the backup on purpose: desktop settings (.config), the Cloudflare" \
         "tunnel for Frigate (.cloudflared, snapshots/etc_cloudflared.tgz) and Wi-Fi profiles" \
         "(Imager sets Wi-Fi). See docs/frigate.md to bring Frigate back."
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

step_verify() {
    needs_from verify
    local fails=0
    pass() { echo "  [OK]   $*"; }
    fail() { echo "  [FAIL] $*"; fails=$((fails + 1)); }

    echo "[INFO] Comparing the Pi's data and models with the backup (reads every file)..."
    local -a excl parts=()
    mapfile -t excl < <(data_excludes)
    for d in data models; do [ -d "$SRC/$d" ] && parts+=("$SRC/$d"); done
    local diff
    diff=$(rsync -a --dry-run --checksum --itemize-changes "${excl[@]}" -e "$RSYNC_SSH" \
           "${parts[@]}" "$TARGET:$REMOTE_DIR/" | grep -E '^[<>c]f' || true)
    if [ -z "$diff" ]; then
        pass "every file in data/ and models/ matches the backup"
    else
        fail "$(wc -l <<<"$diff") file(s) differ from the backup or are missing, e.g.:"
        head -5 <<<"$diff" | sed 's/^/           /'
    fi
    local f
    for f in runtime.yaml secrets.yaml; do
        [ -f "$SRC/configs/$f" ] || continue
        if rsh "cat $(q "$REMOTE_DIR")/configs/$f" | cmp -s - "$SRC/configs/$f"; then
            pass "configs/$f matches"
        else
            fail "configs/$f differs (fine if you changed settings since)"
        fi
    done
    local db name local_sha remote_sha
    for db in "${DB_COPIES[@]}"; do
        name=$(basename "$db")
        local_sha=$(sha256sum "$db" | cut -d' ' -f1)
        remote_sha=$(rsh "sha256sum $(q "$REMOTE_DIR")/data/$name" | cut -d' ' -f1)
        [ "$local_sha" = "$remote_sha" ] && pass "data/$name matches its consistent copy" \
            || fail "data/$name differs from snapshots/$name (fine if the dashboard already ran)"
    done

    echo "[INFO] Checking the Pi..."
    local report
    report=$(rsh "bash -s -- $(q "$REMOTE_DIR")" <<'EOF'
cd "$1"
ok() { echo "OK|$*"; }
no() { echo "FAIL|$*"; }
for db in data/*.db; do
    [ -f "$db" ] || continue
    r=$(python3 -c "import sqlite3,sys; print(sqlite3.connect(sys.argv[1]).execute('PRAGMA integrity_check').fetchone()[0])" "$db" 2>&1)
    [ "$r" = ok ] && ok "$db passes integrity_check" || no "$db: $r"
done
echo "INFO|$(find data/raw -type f -name '*.jpg' | wc -l) images in data/raw"
if out=$(hailortcli fw-control identify 2>&1); then
    ok "Hailo device answers ($(grep -m1 -i 'firmware version' <<<"$out" | xargs))"
else
    no "Hailo device doesn't answer: $(tail -1 <<<"$out")"
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
    local n
    n=$(find "$SRC/data/raw" -type f -name '*.jpg' | wc -l)
    echo "  [INFO] $n images in the backup's data/raw"
    if [ "$fails" -gt 0 ]; then
        echo "[WARN] $fails check(s) failed"
        return 1
    fi
    echo "[OK] Everything checks out"
}

# ── Main ─────────────────────────────────────────────────────────────
connect
echo "[INFO] Pi: $TARGET:$REMOTE_DIR"
[ -n "$FROM" ] && echo "[INFO] Backup: $FROM"
rc=0
for s in $ALL_STEPS; do
    [[ " $STEPS " == *" $s "* ]] || continue
    echo ""
    echo "══ $s ══"
    # Called plainly so set -e stops at the first failing command inside the
    # step (a function called as `f || ...` would run on past failures).
    # verify reports each failed check itself and returns 1 at the end.
    if [ "$s" = verify ]; then
        step_verify || rc=$?
    else
        "step_$s"
    fi
done

if [[ " $STEPS " == *" setup "* ]] && [ "$rc" = 0 ]; then
    cat <<EOF

[OK] Restore finished. Start the dashboard on the Pi:
  ssh $TARGET
  cd $REMOTE_DIR && source hailo-apps/setup_env.sh
  nohup python pi_dashboard.py > logs/pi_dashboard.out 2>&1 &
Then open http://${TARGET#*@}:5003 and:
  - Backup tab: choose Local PC (or Hugging Face), Save, then "Back up now"
  - Training tab: check the backend's Test connection
EOF
fi
exit "$rc"
