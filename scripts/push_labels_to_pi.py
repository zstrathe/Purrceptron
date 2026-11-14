#!/usr/bin/env python3
# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""One-time: copy labels and bounding boxes made on the PC back to the Pi.

The old PC training dashboard labeled and annotated images inside a version
folder (versions/<v>/raw_images/) and never sent that work back to the Pi.
Now the Pi's data/raw/ is the only copy that matters, so this script brings
the Pi up to date with one PC version:

  - images the PC labeled that the Pi still has in unlabeled/ are moved into
    the cat's folder on the Pi
  - bounding boxes edited or drawn on the PC replace the Pi's sidecars
  - an image the Pi holds under a different cat is a conflict: reported and
    skipped, unless --prefer-pc
  - images the PC has but the Pi doesn't: skipped when they are in the Pi's
    trash (you deleted them there); otherwise copied only with
    --include-missing

Images are matched by file name, or by capture timestamp when the Pi's Review
tab renamed one (<cat>_<timestamp>.jpg).

It is a dry run unless --apply is given. Stop the Pi dashboard first (or
press Rescan in Review afterwards).

Usage:
    python scripts/push_labels_to_pi.py --version versions/v11
    python scripts/push_labels_to_pi.py --version versions/v11 --apply
Reads PI_USER, PI_HOST and PI_REMOTE_DIR from .env (as sync_to_pi.sh does).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EDIT_SOURCES = {"manual", "auto", "auto-detect"}


def read_env() -> dict:
    env = {}
    path = PROJECT_ROOT / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


def stamp(name: str) -> str:
    """Capture timestamp part of <label>_<timestamp>.jpg."""
    return Path(name).stem.split("_", 1)[-1]


def edited_on_pc(sidecar: Path) -> bool:
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return ("original_detections" in meta or meta.get("created_by") == "annotate"
            or any(d.get("source") in _EDIT_SOURCES for d in meta.get("detections", [])))


class Pi:
    """Runs commands on the Pi over ssh, or on a local folder (--local-pi-dir,
    for testing)."""

    def __init__(self, target: str | None, root: str):
        self.target = target
        self.root = root.rstrip("/")

    def run(self, script: str) -> str:
        if self.target:
            cmd = ["ssh", self.target, "bash -s"]
        else:
            cmd = ["bash", "-s"]
        res = subprocess.run(cmd, input=script, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"Command on the Pi failed:\n{res.stderr.strip()}")
        return res.stdout

    def upload(self, staging: Path, dest: str) -> None:
        target = f"{self.target}:{dest}/" if self.target else f"{dest}/"
        subprocess.run(["rsync", "-a", f"{staging}/", target], check=True)

    def md5s(self, rel_paths: list[str]) -> dict[str, str]:
        """md5 of each existing file under data/raw (missing ones left out)."""
        if not rel_paths:
            return {}
        quoted = " ".join(shlex.quote(p) for p in rel_paths)
        out = self.run(f"cd {shlex.quote(self.root + '/data/raw')} && "
                       f"for f in {quoted}; do [ -f \"$f\" ] && md5sum \"$f\"; done; true\n")
        return {line.split(None, 1)[1]: line.split(None, 1)[0]
                for line in out.splitlines() if line.strip()}

    def list_jpgs(self, rel_dir: str) -> list[str]:
        """Relative paths (<folder>/<file>.jpg) one level below rel_dir."""
        d = shlex.quote(f"{self.root}/{rel_dir}")
        out = self.run(f"[ -d {d} ] && cd {d} && find . -mindepth 2 -maxdepth 2 "
                       "-name '*.jpg' -printf '%P\\n' || true\n")
        return [line for line in out.splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", type=Path, required=True,
                        help="PC version folder, e.g. versions/v11")
    parser.add_argument("--apply", action="store_true", help="Make the changes (default: dry run)")
    parser.add_argument("--prefer-pc", action="store_true",
                        help="Resolve conflicts with the PC's label")
    parser.add_argument("--include-missing", action="store_true",
                        help="Copy images the Pi doesn't have (and hasn't trashed)")
    parser.add_argument("--local-pi-dir", type=Path, default=None,
                        help="Use a local folder as the Pi project (testing)")
    args = parser.parse_args()

    raw = (args.version / "raw_images").resolve()
    if not raw.is_dir():
        sys.exit(f"No raw_images/ in {args.version}")
    cats = sorted(d.name for d in raw.iterdir() if d.is_dir() and d.name != "unlabeled")

    if args.local_pi_dir:
        pi = Pi(None, str(args.local_pi_dir.resolve()))
        where = str(args.local_pi_dir)
    else:
        env = read_env()
        if not env.get("PI_HOST") or not env.get("PI_USER"):
            sys.exit("Set PI_USER and PI_HOST in .env, or pass --local-pi-dir")
        target = f"{env['PI_USER']}@{env['PI_HOST']}"
        remote = env.get("PI_REMOTE_DIR") or "~/Repos/cat_detection"
        remote = pi_home(target, remote)
        pi = Pi(target, remote)
        where = f"{target}:{remote}"

    print(f"PC version: {args.version} ({', '.join(cats)})")
    print(f"Pi project: {where}")
    on_pi = pi.list_jpgs("data/raw")
    trashed = {Path(p).name for p in pi.list_jpgs("data/trash")}
    trashed_stamps = {stamp(n) for n in trashed}
    by_name = {Path(p).name: p for p in on_pi}
    by_stamp: dict[str, str] = {}
    for p in on_pi:
        by_stamp.setdefault(stamp(Path(p).name), p)

    moves: list[tuple[str, str]] = []       # (from, to) relative to data/raw
    uploads: list[tuple[Path, str]] = []    # (local file, to) relative to data/raw
    conflicts, deleted, missing = [], [], []
    already = 0

    for cat in cats:
        for img in sorted((raw / cat).glob("*.jpg")):
            pi_rel = by_name.get(img.name) or by_stamp.get(stamp(img.name))
            sidecar = img.with_suffix(".json")
            edited = sidecar.is_file() and edited_on_pc(sidecar)
            if pi_rel is None:
                if img.name in trashed or stamp(img.name) in trashed_stamps:
                    deleted.append(f"{cat}/{img.name}")
                    continue
                missing.append(f"{cat}/{img.name}")
                if args.include_missing:
                    uploads.append((img, f"{cat}/{img.name}"))
                    if sidecar.is_file():
                        uploads.append((sidecar, f"{cat}/{sidecar.name}"))
                continue
            pi_folder = pi_rel.split("/", 1)[0]
            dest = pi_rel
            if pi_folder != cat:
                if pi_folder != "unlabeled":
                    conflicts.append(f"{img.name}: PC says {cat}, Pi says {pi_folder}")
                    if not args.prefer_pc:
                        continue
                dest = f"{cat}/{Path(pi_rel).name}"
                moves.append((pi_rel, dest))
            elif not edited:
                already += 1
            if edited:
                uploads.append((sidecar, str(Path(dest).with_suffix(".json"))))

    # Skip bounding boxes the Pi already has (a second run sends nothing).
    remote_md5 = pi.md5s([rel for f, rel in uploads if f.suffix == ".json"])
    uploads = [(f, rel) for f, rel in uploads
               if f.suffix != ".json" or remote_md5.get(rel) != hashlib.md5(f.read_bytes()).hexdigest()]
    n_sidecars = sum(1 for f, _ in uploads if f.suffix == ".json")
    n_images = sum(1 for f, _ in uploads if f.suffix == ".jpg")
    print()
    print(f"Already in sync:                    {already}")
    print(f"Move into a cat folder on the Pi:   {len(moves)}")
    print(f"Bounding boxes to send:             {n_sidecars}")
    print(f"Conflicts (different cat on Pi):    {len(conflicts)}"
          + ("  -> using the PC's label" if args.prefer_pc and conflicts else ""))
    print(f"Deleted on the Pi (skipped):        {len(deleted)}")
    print(f"Not on the Pi:                      {len(missing)}"
          + (f"  -> copying {n_images}" if args.include_missing else "  (skipped; see --include-missing)"))
    for c in conflicts[:20]:
        print(f"  conflict: {c}")
    if len(conflicts) > 20:
        print(f"  ... and {len(conflicts) - 20} more")

    if not args.apply:
        print("\nDry run. Run again with --apply to make these changes.")
        return
    if not moves and not uploads:
        print("\nNothing to do.")
        return

    raw_root = f"{pi.root}/data/raw"
    if moves:
        lines = ["set -e", f"cd {shlex.quote(raw_root)}"]
        for src, dst in moves:
            s, d = shlex.quote(src), shlex.quote(dst)
            sj, dj = (shlex.quote(str(Path(x).with_suffix(".json"))) for x in (src, dst))
            lines.append(f"mkdir -p {shlex.quote(str(Path(dst).parent))}")
            lines.append(f"[ -e {d} ] || mv {s} {d}")
            lines.append(f"[ ! -e {sj} ] || [ -e {dj} ] || mv {sj} {dj}")
        pi.run("\n".join(lines) + "\n")
        print(f"Moved {len(moves)} images on the Pi")
    if uploads:
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for src, rel in uploads:
                (staging / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, staging / rel)
            pi.upload(staging, raw_root)
        print(f"Sent {n_sidecars} bounding boxes and {n_images} images")
    print("Done. Restart the Pi dashboard (or press Rescan in Review) to see the changes.")


def pi_home(target: str, remote: str) -> str:
    """Expand ~ in the remote dir on the Pi, since the paths get quoted."""
    if not remote.startswith("~"):
        return remote
    home = subprocess.run(["ssh", target, "echo $HOME"], capture_output=True,
                          text=True, check=True).stdout.strip()
    return home + remote[1:]


if __name__ == "__main__":
    main()
