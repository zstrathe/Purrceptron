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
#   check    SSH works; it's a Pi 5 with 64-bit Raspberry Pi OS and room to spare
#   system   apt full-upgrade and the Hailo runtime (hailo-h10-all); reboots if needed
#   code     copy the code from this PC
#   data     copy data/, models/, logs/ and the Pi-only settings from DIR
#   home     shell files and SSH keys from DIR/home (manual copies only)
#   setup    scripts/setup_pi.sh: hailo-apps, Python packages, SD card settings;
#            reboots once for the swap setting, then runs it again
#   service  start the dashboard at every boot (asks first)
#   verify   compare the restored files with DIR byte for byte; check the database,
#            the Hailo device, the Python packages, the SD card settings and the
#            dashboard
#
# Before running: flash the card with Raspberry Pi Imager (Raspberry Pi OS 64-bit),
# and in its settings use the old hostname and user name, set Wi-Fi and the Wi-Fi
# country, and turn on SSH. Either paste this PC's key (~/.ssh/id_ed25519.pub)
# for public-key login, or use a password and this script offers ssh-copy-id.
# For a new Pi without a backup, use install_pi.sh instead.
# ═══════════════════════════════════════════════════════════════

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
source "$SCRIPT_DIR/_pi_remote.sh"

ALL_STEPS="check system code data home setup service verify"
FROM=""
STEPS="$ALL_STEPS"
YES=0
FORCE=0
TARGET=""

usage() { sed -n '10,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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

step_verify() {
    needs_from verify
    fails=0
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

    verify_pi
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
    echo ""
    echo "[OK] Restore finished."
    dashboard_howto
    cat <<'EOF'
In the dashboard:
  - Backup tab: choose Local PC (or Hugging Face), Save, then "Back up now"
  - Training tab: check the backend's Test connection
EOF
elif [ "$rc" = 0 ]; then
    echo ""
    dashboard_howto
fi
exit "$rc"
