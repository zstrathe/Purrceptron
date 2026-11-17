#!/usr/bin/env bash
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

# Start, watch and stop the training worker that runs in Docker (compose.worker.yaml).
#
#   scripts/worker.sh            start it if needed, then show its output live
#                                (Ctrl+C stops watching; the worker keeps running)
#   scripts/worker.sh run        run it in this terminal; Ctrl+C stops the worker
#   scripts/worker.sh status     container state, GPU and versions, recent jobs
#   scripts/worker.sh logs       show the output live
#   scripts/worker.sh stop       stop it (a running job is cancelled)
#   scripts/worker.sh restart    restart it (picks up code changes)
#   scripts/worker.sh rebuild    rebuild the image (after editing requirements-training.txt)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMPOSE=(docker compose -f compose.worker.yaml)
SERVICE=worker

running() {
    [ -n "$("${COMPOSE[@]}" ps -q --status running "$SERVICE" 2>/dev/null)" ]
}

ensure_token() {
    if ! grep -q '^WORKER_TOKEN=.' .env 2>/dev/null; then
        echo "[ERROR] No WORKER_TOKEN in .env. Add one, for example:" >&2
        echo "  echo \"WORKER_TOKEN=\$(openssl rand -hex 24)\" >> .env" >&2
        echo "and enter the same token on the Pi dashboard's Training tab." >&2
        exit 1
    fi
}

start() {
    ensure_token
    if running; then
        echo "[OK] Worker already running."
    else
        echo "[INFO] Starting the worker (the first run builds the image, which takes a while)..."
        "${COMPOSE[@]}" up -d --build
    fi
}

status() {
    if ! running; then
        echo "Worker: not running (start it with: scripts/worker.sh)"
        return 0
    fi
    "${COMPOSE[@]}" ps --format 'Container: {{.Name}}  {{.Status}}  {{.Ports}}'
    # Ask the worker itself (inside the container, so no token handling here).
    # The container runs in UTC; pass this machine's offset so times read locally.
    "${COMPOSE[@]}" exec -T -e HOST_UTC_OFFSET="$(date +%z)" "$SERVICE" python - <<'EOF'
import json, os, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

off = os.environ.get("HOST_UTC_OFFSET", "+0000")
sign = -1 if off.startswith("-") else 1
local_tz = timezone(sign * timedelta(hours=int(off[1:3]), minutes=int(off[3:5])))

req = urllib.request.Request("http://127.0.0.1:5004/api/worker/health",
                             headers={"Authorization": "Bearer " + os.environ.get("WORKER_TOKEN", "")})
try:
    h = json.load(urllib.request.urlopen(req, timeout=60))
    print(f"Worker:    torch {h.get('torch_version')}, ExecuTorch {h.get('executorch_version')}, "
          f"GPU: {h.get('gpu') or 'none (CPU)'}")
    print(f"Now:       {('busy with ' + h['busy']) if h.get('busy') else 'idle, waiting for jobs from the Pi'}")
except Exception as exc:
    print(f"Worker:    not answering yet ({exc})")

req = urllib.request.Request("http://127.0.0.1:5004/api/worker/backup/snapshots",
                             headers={"Authorization": "Bearer " + os.environ.get("WORKER_TOKEN", "")})
try:
    st = json.load(urllib.request.urlopen(req, timeout=60)).get("stats", {})
    newest = st.get("newest")
    if newest:
        when = datetime.strptime(newest, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).astimezone(local_tz)
        newest = when.strftime("%Y-%m-%d %H:%M")
    print(f"Backups:   {st.get('snapshots', 0)} snapshot(s) from the Pi, newest {newest or '-'}, "
          f"{st.get('stored_bytes', 0) / 1e6:.0f} MB in data/backups")
except Exception as exc:
    print(f"Backups:   unknown ({exc})")

jobs = sorted((d for d in Path("/jobs").iterdir() if d.is_dir()),
              key=lambda d: d.stat().st_mtime, reverse=True)[:5]
print("Recent jobs:" if jobs else "Recent jobs: none yet")
for d in jobs:
    task = "?"
    try:
        task = json.loads((d / "job" / "manifest.json").read_text()).get("task", "?")
    except Exception:
        pass
    when = datetime.fromtimestamp(d.stat().st_mtime, local_tz).strftime("%Y-%m-%d %H:%M")
    status_file = d / "output" / "status.json"
    if status_file.is_file():
        st = json.loads(status_file.read_text())
        result = f"{'ok' if st.get('ok') else 'failed'}, {st.get('duration_seconds', 0) / 60:.1f} min"
    elif (d / "results.tar.gz").is_file():
        result = "ok"
    else:
        result = "running or stopped early"
    print(f"  {d.name}  {task:8s}  {result:24s}  {when}")
EOF
}

case "${1:-}" in
    "")
        start
        echo "[INFO] Showing the worker's output. Ctrl+C stops watching; the worker keeps running."
        "${COMPOSE[@]}" logs -f --tail 20 "$SERVICE"
        ;;
    run)
        ensure_token
        if running; then
            echo "[INFO] Stopping the background worker first..."
            "${COMPOSE[@]}" stop "$SERVICE"
        fi
        echo "[INFO] Running in this terminal. Ctrl+C stops the worker (a running job is cancelled)."
        "${COMPOSE[@]}" up --build --abort-on-container-exit "$SERVICE"
        ;;
    status)  status ;;
    logs)    "${COMPOSE[@]}" logs -f --tail 50 "$SERVICE" ;;
    stop)    "${COMPOSE[@]}" stop "$SERVICE" ;;
    restart) ensure_token; "${COMPOSE[@]}" restart "$SERVICE" ;;
    rebuild) ensure_token; "${COMPOSE[@]}" up -d --build --force-recreate "$SERVICE" ;;
    -h|--help|help)
        sed -n '7,16p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        echo "Unknown command: $1 (try: scripts/worker.sh help)" >&2
        exit 2
        ;;
esac
