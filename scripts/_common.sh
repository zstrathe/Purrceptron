#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

# Shared preamble for project scripts. Source this, don't execute it.
#
# Provides:
#   SCRIPT_DIR, PROJECT_ROOT — path variables
#   .env loading              — PI_USER, PI_HOST, etc.
#   ensure_venv()             — auto-creates/activates .venv for Python scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

# Auto-create and activate .venv (the scripts-only venv).
# Call this from any script that needs Python.
ensure_venv() {
    local venv="$PROJECT_ROOT/.venv"
    if [ ! -d "$venv" ]; then
        echo "[INFO] Creating .venv for scripts..."
        python3 -m venv "$venv"
        "$venv/bin/pip" install -q --upgrade pip
        "$venv/bin/pip" install -q -r "$PROJECT_ROOT/requirements-scripts.txt"
    fi
    source "$venv/bin/activate"
}

# Print an IPv4 address for a host name; non-zero if it can't be found.
# WSL often can't resolve mDNS names like raspberrypi.local (it works only
# some of the time), so: retry, ask Windows, and as a last resort use the
# address that worked last time. That's safe for SSH: pi_ssh_target sets
# HostKeyAlias, so ssh still checks that the machine there is the same Pi.
PI_ADDR_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/cat_detection/addresses"
pi_resolve() {
    local host="$1" ip="" _
    if [[ "$host" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
        echo "$host"
        return 0
    fi
    for _ in 1 2 3; do
        ip=$(getent ahostsv4 "$host" 2>/dev/null | awk 'NR == 1 {print $1}') || true
        [ -n "$ip" ] && break
        sleep 1
    done
    if [ -z "$ip" ] && command -v avahi-resolve-host-name >/dev/null 2>&1; then
        ip=$(timeout 5 avahi-resolve-host-name -4 "$host" 2>/dev/null | awk '{print $2}') || true
    fi
    # Windows' ping (WSL). Not under timeout: that breaks running .exe files.
    if [ -z "$ip" ] && [ -x /mnt/c/Windows/System32/ping.exe ]; then
        ip=$(/mnt/c/Windows/System32/ping.exe -4 -n 1 -w 2000 "$host" 2>/dev/null \
             | tr -d '\r' | grep -oE '\[[0-9]+(\.[0-9]+){3}\]' | head -1 | tr -d '[]') || true
    fi
    if [ -n "$ip" ]; then
        mkdir -p "$(dirname "$PI_ADDR_CACHE")"
        { grep -v "^$host " "$PI_ADDR_CACHE" 2>/dev/null || true; echo "$host $ip"; } > "$PI_ADDR_CACHE.tmp" \
            && mv "$PI_ADDR_CACHE.tmp" "$PI_ADDR_CACHE"
    else
        ip=$(awk -v h="$host" '$1 == h {print $2}' "$PI_ADDR_CACHE" 2>/dev/null | tail -1)
        [ -n "$ip" ] && echo "[INFO] Can't look up $host now; trying its last known address $ip" >&2
    fi
    [ -n "$ip" ] && echo "$ip"
}

# pi_ssh_target USER@HOST: sets PI_SSH_DEST (what to pass to ssh/rsync) and
# PI_SSH_ALIAS (extra ssh options). The name is looked up once and the address
# used from then on, because mDNS lookups from WSL work only some of the time.
# HostKeyAlias keeps the Pi's key filed under its name in known_hosts.
# shellcheck disable=SC2034  # PI_SSH_DEST and PI_SSH_ALIAS are for the caller
pi_ssh_target() {
    local target="$1" host user="" ip
    host="${target#*@}"
    [[ "$target" == *@* ]] && user="${target%@*}@"
    PI_SSH_DEST="$target"
    PI_SSH_ALIAS=()
    if ip=$(pi_resolve "$host") && [ "$ip" != "$host" ]; then
        echo "[INFO] $host is $ip"
        PI_SSH_DEST="$user$ip"
        PI_SSH_ALIAS=(-o "HostKeyAlias=$host")
    fi
}
