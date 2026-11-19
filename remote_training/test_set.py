# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""The permanent test set: labeled images no model is ever trained on.

Every train job leaves these images out, so any two models can be compared
fairly on them; only the ones a person confirmed are scored.

Images are assigned by capture session (images closer together than
session_gap_minutes), so near-identical frames of one visit never end up on
both sides. A session is assigned once, by a hash of its first image, and
stays where it is: the choice is saved in data/raw/.test_set.json, keyed by
the timestamp part of the file name, which survives relabeling (that only
renames the <cat>_ prefix and moves the file to another folder).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

STATE_NAME = ".test_set.json"
REVIEW_STATE_NAME = ".review_state.json"
_IMG_EXT = {".jpg", ".jpeg"}
DEFAULT_TEST_FRAC = 0.15


def image_key(path: Path | str) -> str:
    """The capture time part of an image's name: <cat>_<YYYYmmdd>_<HHMMSS>_<ms>
    gives "<YYYYmmdd>_<HHMMSS>_<ms>"; any other name is used whole."""
    stem = Path(path).stem
    parts = stem.rsplit("_", 3)
    if len(parts) == 4 and all(p.isdigit() for p in parts[1:]):
        return "_".join(parts[1:])
    return stem


def _key_ms(key: str) -> int:
    """Milliseconds-ish sort value of a key (only differences matter)."""
    parts = key.split("_")
    try:
        if len(parts) == 3:
            t = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
            return int(t.timestamp()) * 1000 + int(parts[2])
        return int(key)
    except ValueError:
        return 0


def _in_test(first_key: str, frac: float) -> bool:
    h = int(hashlib.sha1(first_key.encode()).hexdigest()[:8], 16)
    return h / 0x1_0000_0000 < frac


def _images(d: Path) -> list[Path]:
    return [p for p in d.iterdir() if p.suffix.lower() in _IMG_EXT] if d.is_dir() else []


def confirmed_relpaths(raw_dir: Path) -> set[str]:
    """Images whose label a person confirmed ("cat/file.jpg"), from the
    Label tab's review state. Skipped images don't count."""
    try:
        state = json.loads((raw_dir / REVIEW_STATE_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    return {rel.replace(os.sep, "/") for rel, v in state.items() if v is True}


class TestSet:
    __test__ = False        # not a pytest test class
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir
        self.path = raw_dir / STATE_NAME
        self.test: set[str] = set()
        self.train: set[str] = set()
        try:
            data = json.loads(self.path.read_text())
            self.test = set(data.get("test") or [])
            self.train = set(data.get("train") or [])
        except (OSError, json.JSONDecodeError):
            pass

    def is_test(self, path: Path | str) -> bool:
        return image_key(path) in self.test

    def assign(self, cats: list[str], gap_minutes: float = 5.0,
               frac: float = DEFAULT_TEST_FRAC) -> int:
        """Give every image not assigned yet a side. A new image in a session
        that already has assigned images joins their side (test if any of
        them is test); a session of only new images is decided by the hash of
        its first image. Returns how many images were newly assigned."""
        gap_ms = gap_minutes * 60_000
        added = 0
        for cat in cats:
            keys = sorted({image_key(p) for p in _images(self.raw_dir / cat)}, key=_key_ms)
            session: list[str] = []
            for key in keys:
                if session and _key_ms(key) - _key_ms(session[-1]) >= gap_ms:
                    added += self._assign_session(session, frac)
                    session = []
                session.append(key)
            if session:
                added += self._assign_session(session, frac)
        return added

    def _assign_session(self, session: list[str], frac: float) -> int:
        new = [k for k in session if k not in self.test and k not in self.train]
        if not new:
            return 0
        if any(k in self.test for k in session):
            side = self.test
        elif any(k in self.train for k in session):
            side = self.train
        else:
            side = self.test if _in_test(session[0], frac) else self.train
        side.update(new)
        return len(new)

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"test": sorted(self.test), "train": sorted(self.train)}))
        os.replace(tmp, self.path)


def load_assigned(raw_dir: Path, cats: list[str], gap_minutes: float = 5.0,
                  frac: float = DEFAULT_TEST_FRAC, save: bool = True) -> TestSet:
    """The test set with every current image assigned (saved when anything
    changed and save is true)."""
    ts = TestSet(raw_dir)
    if ts.assign(cats, gap_minutes, frac) and save:
        ts.save()
    return ts


def settings_from(config: dict) -> tuple[float, float]:
    """(session_gap_minutes, test_frac) from the project config."""
    tc = config.get("classifier_training") or {}
    return (float(tc.get("session_gap_minutes", 5) or 5),
            float(tc.get("test_frac", DEFAULT_TEST_FRAC)))
