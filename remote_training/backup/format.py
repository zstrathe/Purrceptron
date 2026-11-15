# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Backup format shared by the Pi, the PC worker and the restore script:
snapshot ids, manifests, blob paths and the retention rule."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

FORMAT = "cat-detection-backup"
VERSION = 1

SNAPSHOT_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")      # UTC, e.g. 20261115T030000Z
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_FMT = "%Y%m%dT%H%M%SZ"


def new_snapshot_id(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(_ID_FMT)


def snapshot_time(snapshot_id: str) -> datetime:
    return datetime.strptime(snapshot_id, _ID_FMT).replace(tzinfo=timezone.utc)


def blob_path(sha: str) -> str:
    """Path of a blob inside the backup, e.g. blobs/ab/cd/abcd...."""
    return f"blobs/{sha[:2]}/{sha[2:4]}/{sha}"


def manifest_path(snapshot_id: str) -> str:
    return f"snapshots/{snapshot_id}.json.gz"


def encode_manifest(manifest: dict) -> bytes:
    return gzip.compress(json.dumps(manifest, separators=(",", ":")).encode(), mtime=0)


def decode_manifest(data: bytes) -> dict:
    manifest = json.loads(gzip.decompress(data))
    check_manifest(manifest)
    return manifest


def check_manifest(manifest: dict) -> None:
    """Raise ValueError unless this is a well-formed manifest."""
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError("Not a cat-detection backup manifest (or a newer version)")
    if not SNAPSHOT_ID_RE.match(str(manifest.get("id", ""))):
        raise ValueError("Manifest has a bad snapshot id")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Manifest has no file list")
    for rel, entry in files.items():
        if (not isinstance(entry, list) or len(entry) != 3 or not SHA256_RE.match(str(entry[0]))
                or not safe_relpath(rel)):
            raise ValueError(f"Bad manifest entry: {rel!r}")


def safe_relpath(rel: str) -> bool:
    """A relative path that stays inside the folder it is restored into."""
    p = Path(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts and "\\" not in rel


def manifest_hashes(manifest: dict) -> set[str]:
    return {entry[0] for entry in manifest["files"].values()}


def snapshot_summary(manifest: dict) -> dict:
    return {"id": manifest["id"], "created": manifest.get("created"),
            "file_count": len(manifest["files"]),
            "total_bytes": sum(entry[1] for entry in manifest["files"].values())}


def select_expired(snapshot_ids: list[str], keep_days: float,
                   now: datetime | None = None) -> list[str]:
    """Snapshots older than keep_days. The newest snapshot is always kept,
    so a backup that hasn't run for a while still has one to restore."""
    ids = sorted(i for i in snapshot_ids if SNAPSHOT_ID_RE.match(i))
    if not ids:
        return []
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=max(0.0, float(keep_days)))
    return [i for i in ids[:-1] if snapshot_time(i) < cutoff]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
