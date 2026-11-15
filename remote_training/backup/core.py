# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""What the Pi backs up, and one backup run (scan, upload, snapshot, prune)."""
from __future__ import annotations

import hashlib
import os
import socket
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from remote_training.backup.destinations import BackupError
from remote_training.backup.format import (
    FORMAT, VERSION, new_snapshot_id, sha256_file, snapshot_summary,
)
from utils.config import resolve_path
from utils.runtime_paths import ram_dir_with_room

# Rehash every file at least this often, instead of trusting an unchanged
# size and modification time (catches silent corruption on the card).
FULL_HASH_DAYS = 7
# Files up to this size are read into memory once for hashing and upload, so
# the uploaded bytes are exactly the hashed bytes even if the file changes.
_IN_MEMORY_MAX = 256 * 1024 * 1024

_SKIP_DIRS = {".thumbnails", "__pycache__", "crop_cache"}
_SKIP_SUFFIXES = (".tmp", "-wal", "-shm", "-journal")


@dataclass
class Entry:
    rel: str               # path inside the backup (relative to the project)
    path: Path             # file to read
    mtime_path: Path       # file whose modification time is recorded
    always_hash: bool = False


def _rel(path: Path, project_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project_root.resolve()).as_posix()
    except ValueError:      # configured outside the project
        return "_external/" + path.as_posix().lstrip("/")


def _walk(root: Path, skip: set[Path]):
    """Regular files under root, not following symlinks."""
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if name.endswith(_SKIP_SUFFIXES) or p.is_symlink() or p.resolve() in skip:
                continue
            yield p


def database_paths(project_root: Path, config: dict) -> list[Path]:
    paths = []
    for section in ("classifier", "feeding"):
        db = (config.get(section) or {}).get("db_path")
        if db:
            p = resolve_path(db, project_root)
            if p.is_file():
                paths.append(p)
    return paths


def collect(project_root: Path, config: dict, config_path: Path,
            include_secrets: bool, db_copies: dict[Path, Path]) -> list[Entry]:
    """Everything to back up. db_copies maps each SQLite database to a
    consistent copy of it (made with snapshot_databases)."""
    entries: list[Entry] = []
    seen: set[str] = set()
    skip = {p.resolve() for p in db_copies}

    def add(path: Path, read: Path | None = None, always_hash: bool = False) -> None:
        rel = _rel(path, project_root)
        if rel not in seen:
            seen.add(rel)
            entries.append(Entry(rel, read or path, path, always_hash))

    raw = resolve_path((config.get("collection") or {}).get("output_dir") or "data/raw",
                       project_root)
    trash = resolve_path((config.get("label_review") or {}).get("trash_dir") or "data/trash",
                         project_root)
    for root in (raw, trash, project_root / "data" / "training_jobs",
                 project_root / "data" / "export", project_root / "models",
                 project_root / "logs"):
        for p in _walk(root, skip):
            add(p)
    configs = config_path.parent
    for p in (config_path, configs / "runtime.yaml"):
        if p.is_file():
            add(p)
    if include_secrets:
        for p in (configs / "secrets.yaml", project_root / ".env"):
            if p.is_file():
                add(p)
    for db, copy in db_copies.items():
        add(db, read=copy, always_hash=True)
    return entries


def snapshot_databases(dbs: list[Path], tmp_dir: Path) -> dict[Path, Path]:
    """Consistent copies of SQLite databases (SQLite's backup API), even while
    the classifier or dashboard is writing to them."""
    copies = {}
    for i, db in enumerate(dbs):
        dest = tmp_dir / f"db{i}.sqlite"
        # A normal connection: read-only mode can fail on WAL databases whose
        # -shm file doesn't exist. The backup API only reads the source.
        src = sqlite3.connect(str(db), timeout=30)
        try:
            out = sqlite3.connect(dest)
            try:
                src.backup(out)
            finally:
                out.close()
        finally:
            src.close()
        copies[db] = dest
    return copies


def _read_for_upload(path: Path) -> tuple[bytes | Path, str, int]:
    """(content or the path itself for very large files, sha256, size)."""
    size = path.stat().st_size
    if size <= _IN_MEMORY_MAX:
        data = path.read_bytes()
        return data, hashlib.sha256(data).hexdigest(), len(data)
    return path, sha256_file(path), size


def run_backup(dest, *, project_root: Path, config: dict, config_path: Path,
               keep_days: float, log: Callable[[str], None],
               should_stop: Callable[[], bool] = lambda: False) -> dict:
    """One backup run to a destination (see destinations.py). Returns a summary."""
    now = datetime.now(timezone.utc)
    snapshot_id = new_snapshot_id(now)
    previous = dest.list_snapshots()
    prev = dest.get_manifest(previous[-1]["id"]) if previous else None
    prev_files = (prev or {}).get("files", {})
    full_hash_at = (prev or {}).get("full_hash_at")
    full = (prev is None or not full_hash_at
            or datetime.fromisoformat(full_hash_at) < now - timedelta(days=FULL_HASH_DAYS))
    log(f"Snapshot {snapshot_id} to {dest.label}"
        f"{' (checking every file)' if full else ''}")

    dbs = database_paths(project_root, config)
    db_bytes = sum(p.stat().st_size for p in dbs)
    tmp_parent = ram_dir_with_room(db_bytes + 10_000_000)
    with tempfile.TemporaryDirectory(dir=tmp_parent, prefix="backup-") as tmp:
        db_copies = snapshot_databases(dbs, Path(tmp))
        entries = collect(project_root, config, config_path, dest.allows_secrets, db_copies)
        log(f"Scanning {len(entries)} files")

        files: dict[str, list] = {}
        for i, e in enumerate(entries, 1):
            if should_stop():
                raise BackupError("Cancelled")
            try:
                st = e.path.stat()
                mtime = e.mtime_path.stat().st_mtime_ns
            except FileNotFoundError:
                continue                       # deleted while scanning
            old = prev_files.get(e.rel)
            if (not full and not e.always_hash and old
                    and old[1] == st.st_size and old[2] == mtime):
                sha = old[0]
            else:
                sha = sha256_file(e.path)
            files[e.rel] = [sha, st.st_size, mtime]
            if i % 2000 == 0:
                log(f"  scanned {i} / {len(entries)}")

        by_rel = {e.rel: e for e in entries}
        missing = set(dest.missing(sorted({f[0] for f in files.values()})))
        to_send = [rel for rel, f in files.items() if f[0] in missing]
        send_bytes = sum(files[rel][1] for rel in to_send)
        log(f"Uploading {len(to_send)} new or changed files ({send_bytes / 1e6:.1f} MB); "
            f"{len(files) - len(to_send)} unchanged")
        sent: set[str] = set()
        for i, rel in enumerate(to_send, 1):
            if should_stop():
                raise BackupError("Cancelled")
            e = by_rel[rel]
            try:
                content, sha, size = _read_for_upload(e.path)
                mtime = e.mtime_path.stat().st_mtime_ns
            except FileNotFoundError:
                del files[rel]                 # deleted since the scan
                continue
            # Record exactly what is uploaded, even if the file changed since the scan
            files[rel] = [sha, size, mtime]
            if sha not in sent:
                dest.put_blob(sha, content)
                sent.add(sha)
            if i % 500 == 0:
                log(f"  uploaded {i} / {len(to_send)}")
        dest.flush()

    manifest = {
        "format": FORMAT, "version": VERSION, "id": snapshot_id,
        "created": now.astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "full_hash_at": now.isoformat(timespec="seconds") if full else full_hash_at,
        "secrets": dest.allows_secrets,
        "files": files,
    }
    dest.put_manifest(manifest)
    summary = snapshot_summary(manifest)
    summary["uploaded_files"] = len(sent)
    summary["uploaded_bytes"] = sum(files[r][1] for r in to_send if r in files)
    log(f"Saved snapshot {snapshot_id}: {summary['file_count']} files, "
        f"{summary['total_bytes'] / 1e6:.1f} MB")
    pruned = dest.prune(keep_days)
    if pruned.get("deleted_snapshots") or pruned.get("deleted_blobs"):
        log(f"Removed {len(pruned.get('deleted_snapshots', []))} snapshot(s) older than "
            f"{keep_days:g} days and {pruned.get('deleted_blobs', 0)} unused file(s)")
    summary["pruned"] = pruned
    return summary
