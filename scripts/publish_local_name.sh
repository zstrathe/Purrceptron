#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

# Announce a second .local name for this Pi (web_ui.local_name) with Avahi,
# alongside its hostname. Run by the cat-dashboard-name service, which
# install_service.sh sets up. Announces the address again when it changes
# (a new DHCP lease, or a switch between Wi-Fi and Ethernet).
#
# Usage: publish_local_name.sh NAME.local

set -uo pipefail
name="${1:?usage: publish_local_name.sh NAME.local}"
pid=""
current=""

stop() {
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
    exit 0
}
trap stop TERM INT

# The address this Pi uses on the local network (the one with the default route).
address() {
    ip -4 route get 1.1.1.1 2>/dev/null \
        | awk '{for (i = 1; i < NF; i++) if ($i == "src") {print $(i + 1); exit}}'
}

while true; do
    ip=$(address)
    if [ -n "$ip" ] && { [ "$ip" != "$current" ] || ! kill -0 "$pid" 2>/dev/null; }; then
        if [ -n "$pid" ]; then
            kill "$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
        fi
        # -R: name -> address only; looking up the address still gives the
        # hostname.
        avahi-publish -a -R "$name" "$ip" &
        pid=$!
        current="$ip"
        echo "Announcing $name as $ip"
    fi
    sleep 30 &
    wait $!
done
