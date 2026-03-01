#!/usr/bin/env python3
"""Label review classes for iterating through captured images and assigning cat labels.

Provides LabelReviewApp class for use by the unified dashboard (dashboard.py).
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.display import print_info


class LabelReviewApp:
    """Manages image iteration and label assignment."""

    EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, data_dir: Path, trash_dir: Path, cats: list[str],
                 scope: str, scope_cat: str | None = None):
        self.data_dir = data_dir
        self.trash_dir = trash_dir
        self.cats = cats
        self.scope = scope
        self.scope_cat = scope_cat

        self.images: list[Path] = []
        self.index = 0

        # Stats
        self.labeled_count = 0
        self.deleted_count = 0
        self.skipped_count = 0

        self._scan_images()

    def _scan_images(self) -> None:
        """Scan directories for images based on scope."""
        self.images = []

        if self.scope == "unlabeled":
            dirs = ["unlabeled"]
        elif self.scope == "cat":
            dirs = [self.scope_cat]
        else:  # "all" — unlabeled first so reviewers see what needs attention
            dirs = ["unlabeled"] + self.cats

        for dirname in dirs:
            dir_path = self.data_dir / dirname
            if not dir_path.is_dir():
                continue
            for ext in self.EXTENSIONS:
                self.images.extend(sorted(dir_path.glob(f"*{ext}")))

        print_info(f"Scanned {len(self.images)} images")

    def total(self) -> int:
        return len(self.images)

    def remaining(self) -> int:
        return max(0, len(self.images) - self.index)

    def get_current(self) -> dict | None:
        """Return info about the current image, or None if done."""
        if self.index < 0 or self.index >= len(self.images):
            return None

        path = self.images[self.index]

        # Check file still exists (may have been moved externally)
        if not path.exists():
            self.images.pop(self.index)
            if self.index >= len(self.images):
                self.index = max(0, len(self.images) - 1)
            return self.get_current() if self.images else None

        label = path.parent.name
        return {
            "path": str(path),
            "filename": path.name,
            "label": label,
            "index": self.index,
            "total": len(self.images),
            "remaining": self.remaining(),
        }

    def assign_label(self, new_label: str) -> dict:
        """Move current image to the new label directory."""
        if self.index >= len(self.images):
            return {"status": "error", "message": "No image to label"}

        path = self.images[self.index]
        if not path.exists():
            self.images.pop(self.index)
            return {"status": "error", "message": "Image no longer exists"}

        current_label = path.parent.name
        if new_label == current_label:
            # Already correctly labeled, just advance
            self.skipped_count += 1
            self.index += 1
            return {"status": "ok", "action": "already_labeled"}

        # Build destination path
        dest_dir = self.data_dir / new_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Rename to match label convention: {label}_{timestamp}.jpg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ext = path.suffix
        dest_name = f"{new_label}_{ts}{ext}"
        dest_path = dest_dir / dest_name

        # Avoid collision
        counter = 1
        while dest_path.exists():
            dest_name = f"{new_label}_{ts}_{counter}{ext}"
            dest_path = dest_dir / dest_name
            counter += 1

        shutil.move(str(path), str(dest_path))
        self.labeled_count += 1
        self.images.pop(self.index)
        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        return {"status": "ok", "action": "labeled", "new_label": new_label,
                "filename": dest_name}

    def delete_current(self) -> dict:
        """Move current image to trash."""
        if self.index >= len(self.images):
            return {"status": "error", "message": "No image to delete"}

        path = self.images[self.index]
        if not path.exists():
            self.images.pop(self.index)
            return {"status": "error", "message": "Image no longer exists"}

        label = path.parent.name
        trash_cat_dir = self.trash_dir / label
        trash_cat_dir.mkdir(parents=True, exist_ok=True)

        dest = trash_cat_dir / path.name
        counter = 1
        while dest.exists():
            dest = trash_cat_dir / f"{path.stem}_{counter}{path.suffix}"
            counter += 1

        shutil.move(str(path), str(dest))
        self.deleted_count += 1
        self.images.pop(self.index)
        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        return {"status": "ok", "action": "deleted"}

    def skip(self) -> dict:
        """Advance to next image without action."""
        self.skipped_count += 1
        self.index = min(self.index + 1, len(self.images))
        return {"status": "ok"}

    def prev(self) -> dict:
        """Go back one image."""
        self.index = max(0, self.index - 1)
        return {"status": "ok"}

    def get_stats(self) -> dict:
        return {
            "labeled": self.labeled_count,
            "deleted": self.deleted_count,
            "skipped": self.skipped_count,
            "remaining": self.remaining(),
            "total": self.total(),
            "index": self.index,
        }


