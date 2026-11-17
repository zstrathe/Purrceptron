# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""The Capture tab's footer: one row per animal in view, and naming visits.

Pure Python (no Flask), so it is unit-tested on the PC.

- ``footer_rows()`` decides what each row shows, and whether it asks
  "Who is this?": only when no model is working, or the model has given up on
  that animal (Unknown, or its name is held by another animal in view).
- ``VisitNamer`` keeps the names a person gave, per visit. While a visit is
  named, an image is saved under the name every ``interval`` seconds, images
  saved as Unlabeled during it move to the name, and all of them count as
  checked on the Label tab. A name stays with the animal through a short
  dropout: a new visit that is the only one in view, starting within
  ``END_AFTER`` seconds of a named visit going out of view, takes its name.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

_NOT_LOADED = re.compile(r"not found|not an ExecuTorch")


def visit_record(status: dict | None, visit_id) -> dict | None:
    if not status or visit_id is None:
        return None
    return next((r for r in status.get("visits") or []
                 if isinstance(r, dict) and r.get("visit_id") == visit_id), None)


def stable_visits(preview_dets: list | None) -> list[float]:
    """The visit ids of the stable animals in the region, oldest first."""
    ids = {d["visit_id"] for d in preview_dets or []
           if isinstance(d, dict) and d.get("visit_id") is not None and d.get("stable")}
    return sorted(ids)


def _votes_text(rec: dict) -> str:
    votes = rec.get("votes") or {}
    if not votes:
        return ""
    n = rec.get("vote_window") or 10
    return " · ".join(f"{c} {k}" for c, k in sorted(votes.items(), key=lambda x: -x[1])) + \
        f" of the last {n} frames"


def footer_rows(preview_dets: list | None, clf_status: dict | None, named: dict, *,
                enabled: bool, model_exists: bool, worker_error: str | None,
                running: bool, words: dict) -> dict:
    """What the Capture tab's footer shows.

    named: VisitNamer.by_visit() ({visit_id: {cat, saved, moved, in_view}}).
    words: utils.config.subject_words() (animal, an_animal, ...).
    Returns {"running", "model_working", "rows": [...], "idle": {...} | None}.
    """
    animal = words.get("animal", "animal")
    loading = bool(worker_error and _NOT_LOADED.search(worker_error) and model_exists)
    model_working = bool(enabled and model_exists and not worker_error)
    out = {"running": running, "model_working": model_working, "rows": [], "idle": None}
    if not running:
        out["idle"] = {"kind": "not_running", "label": "Detection is not running.", "detail": ""}
        return out

    ids = stable_visits(preview_dets)
    for vid, n in named.items():                 # named, out of view for a moment
        if vid not in ids:
            ids.append(vid)
    for vid in sorted(ids):
        row = {"visit_id": vid, "label": "", "detail": "", "conf": "", "is_name": False,
               "name": None, "named_by": None, "tone": "info", "prompt": False, "stop": False,
               "pending": False}
        n = named.get(vid)
        rec = visit_record(clf_status, vid) if model_working else None
        if n:
            k = n["saved"] + n["moved"]
            row.update(label=n["cat"], name=n["cat"], named_by="you", is_name=True, tone="name",
                       prompt=True, stop=True,
                       detail=f"named by you · {k} image{'' if k == 1 else 's'} saved · " +
                       (f"stops when the {animal} leaves" if n["in_view"]
                        else f"stopping (no {animal} in view)"))
        elif not model_working:
            row.update(label="Who is this?", prompt=True,
                       detail=f"Tap the name: images are saved under it until the {animal} leaves.")
        elif rec is None:
            row.update(label="Identifying...", tone="muted", pending=True)
        elif rec.get("visit_cat"):
            row.update(label=rec["visit_cat"], name=rec["visit_cat"], named_by="model",
                       is_name=True, tone="name", detail=_votes_text(rec),
                       conf=f"{round((rec.get('confidence') or 0) * 100)}%"
                       if rec.get("prediction") else "")
            if rec.get("ood_rejected"):
                row.update(detail="unclear right now", conf="")
        elif rec.get("name_taken"):
            t = rec["name_taken"]
            row.update(label=f"Looks like {t}, but {t} is already here. Who is this?",
                       tone="warn", prompt=True)
        elif rec.get("unknown"):
            label = (f"Unknown {animal}. Who is this?" if rec.get("ood_rejected")
                     else f"Who is this? The model thinks {rec['prediction']}, but is not sure."
                     if rec.get("prediction") else "Who is this?")
            row.update(label=label, tone="warn", prompt=True)
        elif rec.get("prediction"):
            row.update(label=f"{rec['prediction']}?", name=rec["prediction"], is_name=True,
                       tone="name", pending=True, detail=_votes_text(rec),
                       conf=f"{round((rec.get('confidence') or 0) * 100)}%")
        else:
            row.update(label="Identifying...", tone="muted", pending=True)
        if row["prompt"] and not n:
            row["detail"] = row["detail"] or \
                f"Tap the name: images are saved under it until the {animal} leaves."
        out["rows"].append(row)

    if not out["rows"]:
        waiting = f"Waiting for {words.get('an_animal', 'an animal')}..."
        if not enabled:
            out["idle"] = {"kind": "ident_off", "label": waiting,
                           "detail": "Identification is off. When one comes, tap its name here "
                                     "to save its images under that name."}
        elif not model_exists:
            out["idle"] = {"kind": "no_model", "label": waiting,
                           "detail": "No model yet, so tap its name here when one comes: its "
                                     "images are saved under that name."}
        elif loading:
            out["idle"] = {"kind": "loading", "label": "Loading the new model...", "detail": ""}
        elif worker_error:
            out["idle"] = {"kind": "error", "label": f"Identification failed: {worker_error}",
                           "detail": ""}
        else:
            out["idle"] = {"kind": "waiting", "label": waiting, "detail": ""}
    return out


class _Named:
    def __init__(self, cat: str, visit_id: float, now: float):
        self.cat = cat
        self.ids: set[float] = {visit_id}   # every visit this name has covered
        self.current = visit_id             # the latest of them
        self.named_at = now
        self.last_seen = now
        self.last_capture = 0.0
        self.saved: list[str] = []          # filenames saved under the name
        self.moved: list[str] = []          # Unlabeled images moved to it (new filenames)


class VisitNamer:
    """Names given to visits on the dashboard (see the module docstring)."""

    END_AFTER = 10.0        # seconds out of view = the named visit is over
    FRESH = 2.0             # preview detections older than this are stale

    def __init__(self, cats: list[str], interval: float, reviewer):
        self.cats = cats
        self.interval = interval
        self.reviewer = reviewer
        self._lock = threading.Lock()
        self._named: list[_Named] = []
        self._in_view: list[float] = []         # stable visits in view, oldest first
        self._seen: set[float] = set()          # visits already looked at for inheritance
        self._pending: tuple[str, float] | None = None   # (name, when) given with nothing in view
        self.last_result: dict | None = None    # the previous named visit, for the UI

    # ── called from the loop ──
    def observe(self, in_view: list[float], now: float) -> None:
        with self._lock:
            self._in_view = list(in_view)
            for n in self._named:
                if n.ids & set(in_view):
                    n.last_seen = now
            new = [v for v in in_view if v not in self._seen and not self._covered(v)]
            for v in new:
                self._seen.add(v)
                if self._pending and now - self._pending[1] <= self.END_AFTER:
                    self._named.append(_Named(self._pending[0], v, now))
                    self._pending = None
                    continue
                # The same animal back after a dropout: the only one in view, and
                # exactly one named visit is out of view.
                away = [n for n in self._named if not (n.ids & set(in_view))
                        and now - n.last_seen <= self.END_AFTER]
                if len(in_view) == 1 and len(away) == 1:
                    away[0].ids.add(v)
                    away[0].current = v
                    away[0].last_seen = now
            if len(self._seen) > 500:
                self._seen = set(sorted(self._seen)[-100:])
            if self._pending and now - self._pending[1] > self.END_AFTER:
                self._pending = None

    def _covered(self, visit_id: float) -> _Named | None:
        return next((n for n in self._named if visit_id in n.ids), None)

    def due_captures(self, now: float) -> list[tuple[float, str]]:
        """(visit_id, name) to save an image of now."""
        out = []
        with self._lock:
            for n in self._named:
                vids = [v for v in self._in_view if v in n.ids]
                if vids and now - n.last_capture >= self.interval:
                    n.last_capture = now
                    out.append((vids[-1], n.cat))
        return out

    def finish_if_ended(self, now: float) -> None:
        with self._lock:
            ended = [n for n in self._named if now - n.last_seen > self.END_AFTER]
        for n in ended:
            self._finish(n)

    def on_output_line(self, line: str) -> None:
        """detect.py prints a JSON line for every manual capture."""
        if '"capture_ok"' not in line:
            return
        try:
            msg = json.loads(line)
        except ValueError:
            return
        if not msg.get("filename"):
            return
        with self._lock:
            n = self._covered(msg.get("visit_id"))
            if n is None:
                same = [x for x in self._named if x.cat == msg.get("cat")]
                n = same[0] if len(same) == 1 else None
            if n is not None and msg.get("cat") == n.cat:
                n.saved.append(msg["filename"])

    def names(self) -> dict[str, str]:
        """{visit_id: name} for detect.py's live view (set_visit_names)."""
        with self._lock:
            return {str(v): n.cat for n in self._named for v in n.ids}

    # ── called from the API ──
    def name(self, cat: str, visit_id: float | None = None, now: float | None = None) -> dict:
        if cat not in self.cats:
            return {"status": "error", "message": f"Unknown name: {cat}"}
        now = time.time() if now is None else now
        with self._lock:
            if visit_id is None:
                if len(self._in_view) > 1:
                    return {"status": "error",
                            "message": "Several are in view: tap the name on the right row."}
                if not self._in_view:
                    self._pending = (cat, now)       # named before it was seen: wait for it
                    return {"status": "ok", **self._state()}
                visit_id = self._in_view[0]
            n = self._covered(visit_id)
            if n is None:
                n = _Named(cat, visit_id, now)
                self._named.append(n)
                self._seen.add(visit_id)
                old, files = None, []
            else:
                old, files = n.cat, n.saved + n.moved
        if old and old != cat and files:
            # A correction: move what this visit saved under the old name.
            results = self.reviewer.label_paths([self.reviewer.data_dir / old / f for f in files], cat)
            renamed = [r["filename"] for r in results if r.get("status") == "ok" and r.get("filename")]
            with self._lock:
                n.saved, n.moved = renamed, []
        with self._lock:
            n.cat = cat
            n.named_at = now
            n.last_seen = now
            n.last_capture = 0.0
        self._claim_unlabeled(n)
        with self._lock:
            return {"status": "ok", **self._state()}

    def stop(self, visit_id: float | None = None) -> dict:
        with self._lock:
            todo = [n for n in self._named if visit_id is None or visit_id in n.ids]
        for n in todo:
            self._finish(n)
        with self._lock:
            return {"status": "ok", **self._state()}

    # ── internals ──
    def _claim_unlabeled(self, n: _Named) -> None:
        """Move images saved as Unlabeled during this named visit to its name:
        the ones whose sidecar has one of its visit ids."""
        with self._lock:
            cat, ids, since = n.cat, set(n.ids), min(n.ids)
        folder = self.reviewer.data_dir / "unlabeled"
        if not folder.is_dir():
            return
        paths = []
        for p in folder.glob("*.jpg"):
            try:
                if p.stat().st_mtime < since - 1:
                    continue
                meta = json.loads(p.with_suffix(".json").read_text())
            except (OSError, ValueError):
                continue
            if meta.get("visit_id") in ids:
                paths.append(p)
        if not paths:
            return
        results = self.reviewer.label_paths(sorted(paths), cat)
        with self._lock:
            n.moved += [r["filename"] for r in results if r.get("status") == "ok" and r.get("filename")]

    def _finish(self, n: _Named) -> None:
        self._claim_unlabeled(n)
        with self._lock:
            if n not in self._named:
                return
            self._named.remove(n)
            saved, moved = list(n.saved), len(n.moved)
        self.reviewer.mark_reviewed_paths([self.reviewer.data_dir / n.cat / f for f in saved])
        with self._lock:
            self.last_result = {"cat": n.cat, "saved": len(saved), "moved": moved,
                                "ended_at": time.time()}

    def by_visit(self) -> dict[float, dict]:
        """{visit_id: {cat, saved, moved, in_view}} for the footer: each named
        visit under the visit id it has now."""
        with self._lock:
            out = {}
            for n in self._named:
                vids = [v for v in self._in_view if v in n.ids]
                out[vids[-1] if vids else n.current] = {
                    "cat": n.cat, "saved": len(n.saved), "moved": len(n.moved),
                    "in_view": bool(vids)}
            return out

    def _state(self) -> dict:
        names = [n.cat for n in self._named]
        return {
            "in_view": bool(self._in_view),
            "named": ", ".join(names) if names else None,
            "saved": sum(len(n.saved) for n in self._named),
            "moved": sum(len(n.moved) for n in self._named),
            "interval": self.interval,
            "end_after": self.END_AFTER,
            "last": self.last_result,
        }

    def state(self) -> dict:
        with self._lock:
            return self._state()


def read_preview_detections(path: Path, fresh: float = VisitNamer.FRESH) -> list:
    """detect.py's preview detections, or [] if missing or older than fresh seconds."""
    try:
        if time.time() - path.stat().st_mtime > fresh:
            return []
        dets = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return dets if isinstance(dets, list) else []
