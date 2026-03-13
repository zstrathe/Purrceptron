#!/usr/bin/env python3
"""Label review classes for iterating through captured images and assigning cat labels.

Provides LabelReviewApp class for use by the unified dashboard (pi_dashboard.py).
Includes review state tracking and thumbnail caching.
"""
from __future__ import annotations

import json as json_module
import os
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.display import print_info


# ═══════════════════════════════════════════════════════════════════════════
# Review State Tracker
# ═══════════════════════════════════════════════════════════════════════════

class ReviewStateTracker:
    """Tracks reviewed/unreviewed state per image in a JSON file.

    State file: data/raw/.review_state.json
    Format: {"relative/path/to/image.jpg": true, ...}
    Only reviewed images are stored (absence = unreviewed).
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.state_path = data_dir / ".review_state.json"
        self._state: dict[str, bool] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    self._state = json_module.load(f)
            except (json_module.JSONDecodeError, IOError):
                self._state = {}

    def _save(self) -> None:
        try:
            tmp_path = str(self.state_path) + ".tmp"
            with open(tmp_path, "w") as f:
                json_module.dump(self._state, f)
            os.replace(tmp_path, str(self.state_path))
        except IOError as exc:
            print_info(f"Warning: could not save review state: {exc}")

    def is_reviewed(self, path: Path) -> bool:
        rel = str(path.relative_to(self.data_dir))
        with self._lock:
            return self._state.get(rel, False)

    def mark_reviewed(self, path: Path) -> None:
        rel = str(path.relative_to(self.data_dir))
        with self._lock:
            self._state[rel] = True
            self._save()

    def remove_entry(self, path: Path) -> None:
        rel = str(path.relative_to(self.data_dir))
        with self._lock:
            self._state.pop(rel, None)
            self._save()

    def rename_entry(self, old_path: Path, new_path: Path) -> None:
        old_rel = str(old_path.relative_to(self.data_dir))
        new_rel = str(new_path.relative_to(self.data_dir))
        with self._lock:
            if old_rel in self._state:
                self._state[new_rel] = self._state.pop(old_rel)
                self._save()


# ═══════════════════════════════════════════════════════════════════════════
# Thumbnail Cache
# ═══════════════════════════════════════════════════════════════════════════

class ThumbnailCache:
    """Generates and caches thumbnail images for the review strip.

    Thumbnails are stored in data/raw/.thumbnails/ mirroring the directory
    structure of the source images.
    """

    THUMB_SIZE = (150, 100)
    JPEG_QUALITY = 60

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.thumb_dir = data_dir / ".thumbnails"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, image_path: Path) -> Path | None:
        try:
            rel = image_path.relative_to(self.data_dir)
        except ValueError:
            return None

        thumb_path = self.thumb_dir / rel

        # Return cached thumbnail if it exists and is newer than source
        if thumb_path.exists():
            try:
                if thumb_path.stat().st_mtime >= image_path.stat().st_mtime:
                    return thumb_path
            except OSError:
                pass

        # Generate thumbnail
        try:
            thumb_path.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            thumb = cv2.resize(img, self.THUMB_SIZE)
            cv2.imwrite(str(thumb_path), thumb,
                        [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
            return thumb_path
        except Exception:
            return None

    def remove(self, image_path: Path) -> None:
        try:
            rel = image_path.relative_to(self.data_dir)
            thumb_path = self.thumb_dir / rel
            if thumb_path.exists():
                thumb_path.unlink()
        except (ValueError, OSError):
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Label Review App
# ═══════════════════════════════════════════════════════════════════════════

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

        self.images: list[Path] = []         # unreviewed images (review queue)
        self.all_images: list[Path] = []     # ALL images (for thumbnail strip)
        self.index = 0

        # Stats
        self.labeled_count = 0
        self.deleted_count = 0
        self.skipped_count = 0

        # Review state tracker
        self._review_state = ReviewStateTracker(data_dir)

        # Thumbnail cache
        self._thumbnails = ThumbnailCache(data_dir)

        self._scan_images()

    def _scan_images(self) -> None:
        """Scan directories for images based on scope.

        Populates both self.all_images (everything) and
        self.images (unreviewed only).
        """
        self.all_images = []

        if self.scope == "unlabeled":
            dirs = ["unlabeled"]
        elif self.scope == "cat":
            dirs = [self.scope_cat]
        else:  # "all" — unlabeled first so reviewers see what needs attention
            dirs = ["unlabeled"] + self.cats + ["background"]

        for dirname in dirs:
            dir_path = self.data_dir / dirname
            if not dir_path.is_dir():
                continue
            for ext in self.EXTENSIONS:
                self.all_images.extend(dir_path.glob(f"*{ext}"))

        # Sort newest first by modification time
        self.all_images.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Filter to unreviewed for the review queue (preserves newest-first order)
        self.images = [
            img for img in self.all_images
            if not self._review_state.is_reviewed(img)
        ]

        print_info(f"Scanned {len(self.all_images)} images, {len(self.images)} unreviewed")

    def total(self) -> int:
        return len(self.all_images)

    def remaining(self) -> int:
        return max(0, len(self.images) - self.index)

    def unreviewed_count(self) -> int:
        return len(self.images)

    def get_current(self) -> dict | None:
        """Return info about the current image, or None if no unreviewed images remain."""
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

        # Find position in all_images for thumbnail strip sync
        try:
            all_index = self.all_images.index(path)
        except ValueError:
            all_index = -1

        return {
            "path": str(path),
            "filename": path.name,
            "label": label,
            "index": self.index,
            "total": len(self.images),
            "remaining": self.remaining(),
            "all_index": all_index,
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

        # Mark as reviewed
        self._review_state.mark_reviewed(path)

        if new_label == current_label:
            # Already correctly labeled, just remove from unreviewed queue
            self.skipped_count += 1
            self.images.pop(self.index)
            if self.index >= len(self.images):
                self.index = max(0, len(self.images) - 1)
            return {"status": "ok", "action": "already_labeled"}

        # Build destination path
        dest_dir = self.data_dir / new_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Preserve original timestamp from filename: {label}_{timestamp}.jpg
        ext = path.suffix
        original_ts = path.stem.split('_', 1)[1] if '_' in path.stem else datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        dest_name = f"{new_label}_{original_ts}{ext}"
        dest_path = dest_dir / dest_name

        # Avoid collision
        counter = 1
        while dest_path.exists():
            dest_name = f"{new_label}_{original_ts}_{counter}{ext}"
            dest_path = dest_dir / dest_name
            counter += 1

        # Update review state for old → new path
        self._review_state.remove_entry(path)

        orig_mtime = path.stat().st_mtime
        shutil.move(str(path), str(dest_path))
        os.utime(str(dest_path), (orig_mtime, orig_mtime))

        self._review_state.mark_reviewed(dest_path)

        # Update thumbnail cache
        self._thumbnails.remove(path)

        self.labeled_count += 1
        self.images.pop(self.index)
        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        # Update all_images list too
        try:
            all_idx = self.all_images.index(path)
            self.all_images[all_idx] = dest_path
        except ValueError:
            pass

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

        # Mark as reviewed and clean up
        self._review_state.remove_entry(path)
        self._thumbnails.remove(path)

        shutil.move(str(path), str(dest))
        self.deleted_count += 1
        self.images.pop(self.index)
        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        # Remove from all_images too
        try:
            self.all_images.remove(path)
        except ValueError:
            pass

        return {"status": "ok", "action": "deleted"}

    def skip(self) -> dict:
        """Advance to next image. Marks current as reviewed."""
        if self.index < len(self.images):
            self._review_state.mark_reviewed(self.images[self.index])
            self.images.pop(self.index)
            if self.index >= len(self.images):
                self.index = max(0, len(self.images) - 1)
        self.skipped_count += 1
        return {"status": "ok"}

    def prev(self) -> dict:
        """Go back one image."""
        self.index = max(0, self.index - 1)
        return {"status": "ok"}

    # ── Batch operations ──

    def _label_by_path(self, path: Path, new_label: str) -> dict:
        """Label a specific image by its path. Used by batch operations."""
        if not path.exists():
            return {"status": "error", "message": "Image no longer exists"}

        current_label = path.parent.name
        self._review_state.mark_reviewed(path)

        if new_label == current_label:
            self.skipped_count += 1
            try:
                idx = self.images.index(path)
                self.images.pop(idx)
                if self.index >= idx and self.index > 0:
                    self.index -= 1
            except ValueError:
                pass
            return {"status": "ok", "action": "already_labeled"}

        dest_dir = self.data_dir / new_label
        dest_dir.mkdir(parents=True, exist_ok=True)

        ext = path.suffix
        original_ts = path.stem.split('_', 1)[1] if '_' in path.stem else datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        dest_name = f"{new_label}_{original_ts}{ext}"
        dest_path = dest_dir / dest_name

        counter = 1
        while dest_path.exists():
            dest_name = f"{new_label}_{original_ts}_{counter}{ext}"
            dest_path = dest_dir / dest_name
            counter += 1

        self._review_state.remove_entry(path)
        orig_mtime = path.stat().st_mtime
        shutil.move(str(path), str(dest_path))
        os.utime(str(dest_path), (orig_mtime, orig_mtime))
        self._review_state.mark_reviewed(dest_path)
        self._thumbnails.remove(path)
        self.labeled_count += 1

        try:
            idx = self.images.index(path)
            self.images.pop(idx)
            if self.index >= idx and self.index > 0:
                self.index -= 1
        except ValueError:
            pass

        try:
            all_idx = self.all_images.index(path)
            self.all_images[all_idx] = dest_path
        except ValueError:
            pass

        return {"status": "ok", "action": "labeled", "new_label": new_label}

    def _delete_by_path(self, path: Path) -> dict:
        """Delete a specific image by its path. Used by batch operations."""
        if not path.exists():
            return {"status": "error", "message": "Image no longer exists"}

        label = path.parent.name
        trash_cat_dir = self.trash_dir / label
        trash_cat_dir.mkdir(parents=True, exist_ok=True)

        dest = trash_cat_dir / path.name
        counter = 1
        while dest.exists():
            dest = trash_cat_dir / f"{path.stem}_{counter}{path.suffix}"
            counter += 1

        self._review_state.remove_entry(path)
        self._thumbnails.remove(path)
        shutil.move(str(path), str(dest))
        self.deleted_count += 1

        try:
            idx = self.images.index(path)
            self.images.pop(idx)
            if self.index >= idx and self.index > 0:
                self.index -= 1
        except ValueError:
            pass

        try:
            self.all_images.remove(path)
        except ValueError:
            pass

        return {"status": "ok", "action": "deleted"}

    def batch_label(self, indices: list[int], label: str) -> dict:
        """Label multiple images by their all_images indices."""
        paths = []
        for idx in indices:
            if 0 <= idx < len(self.all_images):
                paths.append(self.all_images[idx])

        labeled = 0
        errors = 0
        for path in paths:
            result = self._label_by_path(path, label)
            if result.get("status") == "ok":
                labeled += 1
            else:
                errors += 1

        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        return {"status": "ok", "labeled": labeled, "errors": errors}

    def batch_delete(self, indices: list[int]) -> dict:
        """Delete multiple images by their all_images indices."""
        paths = []
        for idx in indices:
            if 0 <= idx < len(self.all_images):
                paths.append(self.all_images[idx])

        deleted = 0
        errors = 0
        for path in paths:
            result = self._delete_by_path(path)
            if result.get("status") == "ok":
                deleted += 1
            else:
                errors += 1

        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

        return {"status": "ok", "deleted": deleted, "errors": errors}

    def get_stats(self) -> dict:
        return {
            "labeled": self.labeled_count,
            "deleted": self.deleted_count,
            "skipped": self.skipped_count,
            "remaining": self.remaining(),
            "total": self.total(),
            "index": self.index,
        }

    # ── Thumbnail and review state methods ──

    def get_all_images(self) -> list[dict]:
        """Return info for ALL images (for thumbnail strip)."""
        result = []
        for img in self.all_images:
            result.append({
                "filename": img.name,
                "label": img.parent.name,
                "relative_path": str(img.relative_to(self.data_dir)),
                "reviewed": self._review_state.is_reviewed(img),
            })
        return result

    def get_or_create_thumbnail(self, relative_path: str) -> Path | None:
        """Get or create a thumbnail for the image at the given relative path."""
        image_path = self.data_dir / relative_path
        if not image_path.exists():
            return None
        return self._thumbnails.get_or_create(image_path)

    def goto(self, index: int) -> dict:
        """Jump to a specific index in the all_images list.

        Used when clicking a thumbnail. Finds the corresponding position
        in the unreviewed images list, or temporarily inserts the image
        if it's already reviewed.
        """
        if index < 0 or index >= len(self.all_images):
            return {"status": "error", "message": "Index out of range"}

        target_path = self.all_images[index]

        # Try to find it in the unreviewed queue
        try:
            queue_index = self.images.index(target_path)
            self.index = queue_index
        except ValueError:
            # Image is already reviewed but user wants to see it.
            # Temporarily insert it at current position.
            self.images.insert(self.index, target_path)

        return {"status": "ok"}
