#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Restore a snapshot of the Pi's backups into a new folder.

From the PC's backup folder (Local PC destination):
    python scripts/restore_backup.py --source pc --dir data/backups --list
    python scripts/restore_backup.py --source pc --dir data/backups --dest ~/restored

From a Hugging Face bucket (token in the HF_TOKEN environment variable):
    HF_TOKEN=hf_... python scripts/restore_backup.py --source hf --bucket you/cat-backups --list
    HF_TOKEN=hf_... python scripts/restore_backup.py --source hf --bucket you/cat-backups \\
        --snapshot 20261115T030000Z --dest ~/restored

The folder you restore into must be new or empty: the script never
overwrites anything. Files keep their original modification times, and every
file's SHA-256 is checked. Paths are relative to the Pi's project folder, so
copy what you need from the restored folder back to the Pi, e.g.
    rsync -a ~/restored/data/ pi:~/Repos/cat_detection/data/
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from remote_training.backup.format import (  # noqa: E402
    SNAPSHOT_ID_RE, blob_path, decode_manifest, manifest_path, safe_relpath, snapshot_summary,
)
from remote_training.backup.store import LocalStore  # noqa: E402


class PCSource:
    def __init__(self, folder: Path):
        if not (folder / "snapshots").is_dir():
            sys.exit(f"No backups in {folder} (expected a snapshots/ folder there)")
        self.store = LocalStore(folder)

    def snapshot_ids(self) -> list[str]:
        return self.store.snapshot_ids()

    def manifest(self, sid: str) -> dict:
        return self.store.get_manifest(sid)

    def fetch(self, hashes: list[str], into: Path) -> None:
        for sha in hashes:
            shutil.copyfile(self.store.blob_file(sha), into / sha)


class HFSource:
    def __init__(self, bucket: str, token: str | None):
        from huggingface_hub import HfApi
        self.api = HfApi(token=token)
        self.bucket = bucket

    def snapshot_ids(self) -> list[str]:
        ids = []
        for f in self.api.list_bucket_tree(self.bucket, prefix="snapshots/", recursive=True):
            name = f.path.rsplit("/", 1)[-1]
            if name.endswith(".json.gz") and SNAPSHOT_ID_RE.match(name[:-8]):
                ids.append(name[:-8])
        return sorted(ids)

    def manifest(self, sid: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "m.json.gz"
            self.api.download_bucket_files(self.bucket, [(manifest_path(sid), str(local))],
                                           raise_on_missing_files=True)
            return decode_manifest(local.read_bytes())

    def fetch(self, hashes: list[str], into: Path) -> None:
        for i in range(0, len(hashes), 500):
            batch = hashes[i:i + 500]
            self.api.download_bucket_files(
                self.bucket, [(blob_path(h), str(into / h)) for h in batch],
                raise_on_missing_files=True)
            print(f"  downloaded {min(i + 500, len(hashes))} / {len(hashes)}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", choices=("pc", "hf"), required=True)
    p.add_argument("--dir", type=Path, help="Backup folder on the PC (--source pc)")
    p.add_argument("--bucket", help="Hugging Face bucket, e.g. you/cat-backups (--source hf)")
    p.add_argument("--list", action="store_true", help="List snapshots and exit")
    p.add_argument("--snapshot", default="latest", help="Snapshot id, or latest (default)")
    p.add_argument("--dest", type=Path, help="New or empty folder to restore into")
    args = p.parse_args()

    if args.source == "pc":
        if not args.dir:
            p.error("--source pc needs --dir")
        source = PCSource(args.dir.expanduser().resolve())
    else:
        if not args.bucket:
            p.error("--source hf needs --bucket")
        source = HFSource(args.bucket, os.environ.get("HF_TOKEN"))

    ids = source.snapshot_ids()
    if args.list:
        if not ids:
            print("No snapshots.")
        for sid in ids:
            print(sid, end="", flush=True)
            s = snapshot_summary(source.manifest(sid))
            print(f"  {s['created']}  {s['file_count']} files  {s['total_bytes'] / 1e6:.1f} MB")
        return
    if not args.dest:
        p.error("give --dest (a new or empty folder), or --list")
    if not ids:
        sys.exit("No snapshots to restore.")
    sid = ids[-1] if args.snapshot == "latest" else args.snapshot
    if sid not in ids:
        sys.exit(f"No snapshot {sid}. Available: {', '.join(ids)}")
    dest = args.dest.expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"{dest} is not empty. Restore into a new or empty folder.")

    manifest = source.manifest(sid)
    files = manifest["files"]
    for rel in files:
        if not safe_relpath(rel):
            sys.exit(f"Refusing unsafe path in the snapshot: {rel!r}")
    hashes = sorted({e[0] for e in files.values()})
    print(f"Restoring snapshot {sid} ({len(files)} files, {len(hashes)} unique) to {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest, prefix=".restore-") as tmp:
        blobs = Path(tmp)
        source.fetch(hashes, blobs)
        bad = [h for h in hashes if _sha256(blobs / h) != h]
        if bad:
            sys.exit(f"{len(bad)} file(s) are damaged in the backup (SHA-256 mismatch), "
                     f"e.g. {bad[0]}. Nothing was restored.")
        for i, (rel, (sha, size, mtime_ns)) in enumerate(sorted(files.items()), 1):
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blobs / sha, out)
            os.utime(out, ns=(mtime_ns, mtime_ns))
            if i % 2000 == 0:
                print(f"  restored {i} / {len(files)}", flush=True)
    print(f"Done: {len(files)} files restored to {dest}")
    if not manifest.get("secrets"):
        print("This snapshot has no configs/secrets.yaml or .env (they are never uploaded "
              "to the cloud); set tokens again on the dashboard.")


if __name__ == "__main__":
    main()
