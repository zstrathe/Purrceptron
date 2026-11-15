# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""A backup folder on disk. The PC worker keeps the Pi's backups in one (see
the backup routes in remote_training/worker.py), and scripts/restore_backup.py
reads one directly."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO

from remote_training.backup.format import (
    SHA256_RE, SNAPSHOT_ID_RE, blob_path, decode_manifest, encode_manifest, manifest_hashes,
    manifest_path, select_expired, snapshot_summary,
)

# Blobs younger than this are never garbage-collected, so a prune can't remove
# blobs a backup has just uploaded but not referenced in a manifest yet.
GC_GRACE_SECONDS = 3600


class StoreError(RuntimeError):
    pass


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()     # serializes manifest writes and pruning

    # ── Blobs ────────────────────────────────────────────────────
    def _blob(self, sha: str) -> Path:
        if not SHA256_RE.match(sha):
            raise StoreError(f"Bad SHA-256: {sha!r}")
        return self.root / blob_path(sha)

    def missing(self, hashes) -> list[str]:
        return [h for h in hashes if not self._blob(h).is_file()]

    def put_blob(self, sha: str, stream: BinaryIO, max_bytes: int) -> int:
        """Store a blob read from a stream. Its SHA-256 must equal sha; the
        file only appears (atomically) once it is complete and verified."""
        dest = self._blob(sha)
        tmp_dir = self.root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        size = 0
        fd, tmp = tempfile.mkstemp(dir=tmp_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                while chunk := stream.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        raise StoreError(f"Blob larger than {max_bytes} bytes")
                    h.update(chunk)
                    f.write(chunk)
            if h.hexdigest() != sha:
                raise StoreError("Content does not match its SHA-256 (changed or damaged "
                                 "in transit)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            # mkstemp makes the file owner-only; the worker often runs as root
            # in Docker, and the restore script runs as you.
            os.chmod(tmp, 0o644)
            os.replace(tmp, dest)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return size

    def blob_file(self, sha: str) -> Path:
        path = self._blob(sha)
        if not path.is_file():
            raise StoreError(f"Missing blob {sha}")
        return path

    # ── Snapshots ────────────────────────────────────────────────
    def snapshot_ids(self) -> list[str]:
        d = self.root / "snapshots"
        if not d.is_dir():
            return []
        return sorted(p.name[:-len(".json.gz")] for p in d.glob("*.json.gz")
                      if SNAPSHOT_ID_RE.match(p.name[:-len(".json.gz")]))

    def get_manifest(self, snapshot_id: str) -> dict:
        if not SNAPSHOT_ID_RE.match(snapshot_id):
            raise StoreError(f"Bad snapshot id: {snapshot_id!r}")
        path = self.root / manifest_path(snapshot_id)
        if not path.is_file():
            raise StoreError(f"No snapshot {snapshot_id}")
        return decode_manifest(path.read_bytes())

    def list_snapshots(self) -> list[dict]:
        return [snapshot_summary(self.get_manifest(i)) for i in self.snapshot_ids()]

    def put_manifest(self, manifest: dict) -> None:
        """Save a snapshot, after checking that every blob it lists is here."""
        data = encode_manifest(manifest)
        manifest = decode_manifest(data)          # validates it
        with self._lock:
            missing = self.missing(sorted(manifest_hashes(manifest)))
            if missing:
                raise StoreError(f"{len(missing)} file(s) of the snapshot were not uploaded")
            path = self.root / manifest_path(manifest["id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)

    # ── Retention ────────────────────────────────────────────────
    def prune(self, keep_days: float) -> dict:
        """Delete expired snapshots, then blobs no remaining snapshot uses."""
        with self._lock:
            expired = select_expired(self.snapshot_ids(), keep_days)
            for sid in expired:
                (self.root / manifest_path(sid)).unlink(missing_ok=True)
            referenced: set[str] = set()
            for sid in self.snapshot_ids():
                referenced |= manifest_hashes(self.get_manifest(sid))
            deleted = freed = 0
            cutoff = time.time() - GC_GRACE_SECONDS
            blobs = self.root / "blobs"
            if blobs.is_dir():
                for f in blobs.glob("*/*/*"):
                    if f.name in referenced or not SHA256_RE.match(f.name):
                        continue
                    st = f.stat()
                    if st.st_mtime < cutoff:
                        f.unlink()
                        deleted += 1
                        freed += st.st_size
            # Leftovers of uploads that were cut off
            for f in (self.root / "tmp").glob("*") if (self.root / "tmp").is_dir() else []:
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
        return {"deleted_snapshots": expired, "deleted_blobs": deleted, "freed_bytes": freed}

    def stats(self) -> dict:
        ids = self.snapshot_ids()
        total = count = 0
        blobs = self.root / "blobs"
        if blobs.is_dir():
            for f in blobs.glob("*/*/*"):
                count += 1
                total += f.stat().st_size
        return {"snapshots": len(ids), "newest": ids[-1] if ids else None,
                "blobs": count, "stored_bytes": total}
