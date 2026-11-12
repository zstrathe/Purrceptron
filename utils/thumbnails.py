# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Disk-backed JPEG thumbnail cache shared by the Pi and training dashboards.

Thumbnails mirror the source tree under ``cache_root/<max_side>/`` so that a
size change never serves stale thumbnails of the old size.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2


class ThumbnailCache:
    """Generates and caches aspect-preserving thumbnails for image browsers."""

    def __init__(self, source_root: Path, cache_root: Path,
                 max_side: int = 240, quality: int = 70):
        self.source_root = Path(source_root)
        self.max_side = int(max_side)
        self.quality = int(quality)
        self.thumb_dir = Path(cache_root) / str(self.max_side)

    def _thumb_path(self, image_path: Path) -> Path | None:
        try:
            rel = Path(image_path).relative_to(self.source_root)
        except ValueError:
            return None
        return self.thumb_dir / rel

    def get_or_create(self, image_path: Path) -> Path | None:
        """Return a cached thumbnail path, generating it if missing or stale."""
        image_path = Path(image_path)
        thumb_path = self._thumb_path(image_path)
        if thumb_path is None or not image_path.is_file():
            return None

        try:
            if thumb_path.stat().st_mtime >= image_path.stat().st_mtime:
                return thumb_path
        except OSError:
            pass

        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return None

        # Write atomically: concurrent requests for the same thumbnail must
        # never serve a half-written file.
        try:
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=thumb_path.parent, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, thumb_path)
        except OSError:
            return None
        return thumb_path

    def remove(self, image_path: Path) -> None:
        thumb_path = self._thumb_path(Path(image_path))
        if thumb_path is None:
            return
        try:
            thumb_path.unlink(missing_ok=True)
        except OSError:
            pass
