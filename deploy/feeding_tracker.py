#!/usr/bin/env python3
"""Feeding session tracker — aggregates per-frame detections into feeding sessions."""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.display import print_info, print_success, print_warning


class CatState(Enum):
    IDLE = auto()
    DETECTING = auto()
    EATING = auto()
    COOLDOWN = auto()


@dataclass
class CatSession:
    state: CatState = CatState.IDLE
    detection_streak: int = 0
    start_time: float | None = None
    last_seen: float = 0.0
    detection_count: int = 0


class FeedingTracker:
    """Track feeding sessions per cat using a state machine.

    Call update() once per frame with the set of detected cat names.
    Completed sessions are automatically written to SQLite.
    Call shutdown() on exit to finalize any active sessions.
    """

    def __init__(
        self,
        db_path: Path,
        cats: list[str],
        cooldown_seconds: float = 30.0,
        min_detection_frames: int = 3,
        log_interval_seconds: float = 60.0,
    ):
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds
        self.min_detection_frames = min_detection_frames
        self.log_interval_seconds = log_interval_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, CatSession] = {cat: CatSession() for cat in cats}
        self._last_log_time = 0.0
        self._total_logged = 0
        self._init_db()

    def _init_db(self) -> None:
        """Create the feeding_sessions table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feeding_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cat_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    detection_count INTEGER NOT NULL
                )
                """
            )

    def update(self, detected_cats: set[str]) -> None:
        """Called once per frame with the set of cat names currently detected."""
        now = time.time()
        with self._lock:
            for cat, session in self._sessions.items():
                if cat in detected_cats:
                    self._handle_detection(cat, session, now)
                else:
                    self._handle_no_detection(cat, session, now)

            if now - self._last_log_time > self.log_interval_seconds:
                self._print_status(now)
                self._last_log_time = now

    def _handle_detection(self, cat: str, s: CatSession, now: float) -> None:
        if s.state == CatState.IDLE:
            s.state = CatState.DETECTING
            s.detection_streak = 1
            s.last_seen = now
        elif s.state == CatState.DETECTING:
            s.detection_streak += 1
            s.last_seen = now
            if s.detection_streak >= self.min_detection_frames:
                s.state = CatState.EATING
                # Approximate start time by backtracking streak frames (~30fps)
                s.start_time = now - (s.detection_streak / 30.0)
                s.detection_count = s.detection_streak
                print_info(f"Feeding started: {cat}")
        elif s.state == CatState.EATING:
            s.last_seen = now
            s.detection_count += 1
        elif s.state == CatState.COOLDOWN:
            s.state = CatState.EATING
            s.last_seen = now
            s.detection_count += 1
            print_info(f"Feeding resumed: {cat}")

    def _handle_no_detection(self, cat: str, s: CatSession, now: float) -> None:
        if s.state == CatState.DETECTING:
            s.state = CatState.IDLE
            s.detection_streak = 0
        elif s.state == CatState.EATING:
            s.state = CatState.COOLDOWN
        elif s.state == CatState.COOLDOWN:
            if now - s.last_seen > self.cooldown_seconds:
                self._end_session(cat, s)

    def _end_session(self, cat: str, s: CatSession) -> None:
        """Log completed session to SQLite and reset state."""
        end_time = s.last_seen
        duration = end_time - s.start_time

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO feeding_sessions "
                "(cat_name, start_time, end_time, duration_seconds, detection_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cat,
                    datetime.fromtimestamp(s.start_time).isoformat(),
                    datetime.fromtimestamp(end_time).isoformat(),
                    duration,
                    s.detection_count,
                ),
            )

        self._total_logged += 1
        print_success(f"Feeding ended: {cat} — {duration:.0f}s ({s.detection_count} frames)")

        # Reset
        s.state = CatState.IDLE
        s.detection_streak = 0
        s.start_time = None
        s.detection_count = 0

    def _print_status(self, now: float) -> None:
        """Print periodic status of active sessions."""
        active = [
            (cat, s)
            for cat, s in self._sessions.items()
            if s.state in (CatState.EATING, CatState.COOLDOWN)
        ]
        if active:
            for cat, s in active:
                elapsed = now - s.start_time
                print_info(
                    f"Active: {cat} eating for {elapsed:.0f}s "
                    f"({s.detection_count} frames)"
                )

    def get_cat_state(self, cat_name: str) -> CatState:
        """Return the current state for the given cat (thread-safe)."""
        with self._lock:
            session = self._sessions.get(cat_name)
            if session is None:
                return CatState.IDLE
            return session.state

    def get_session_id(self, cat_name: str) -> str | None:
        """Return a unique identifier for the cat's current feeding session, or None if idle."""
        with self._lock:
            session = self._sessions.get(cat_name)
            if session is None or session.start_time is None:
                return None
            return f"{cat_name}_{datetime.fromtimestamp(session.start_time).strftime('%Y%m%d_%H%M%S')}"

    def shutdown(self) -> None:
        """End any active sessions (call on pipeline shutdown)."""
        with self._lock:
            for cat, s in self._sessions.items():
                if s.state in (CatState.EATING, CatState.COOLDOWN):
                    self._end_session(cat, s)
