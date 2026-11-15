# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

"""Snapshot backups of the Pi's data, to the PC worker or a Hugging Face bucket.

Every run makes a snapshot: a complete picture of the backed-up files at that
moment, deletions included, that can be restored on its own. File contents
are stored once by their SHA-256 ("blobs"), so files that didn't change cost
nothing in later snapshots. Snapshots older than the retention period are
deleted (the newest is always kept), and so are blobs no remaining snapshot
uses.

Layout at a destination:
    blobs/<aa>/<bb>/<sha256>          file contents
    snapshots/<YYYYMMDDTHHMMSSZ>.json.gz   one manifest per snapshot

    remote_training/backup/format.py        manifest, ids, retention (shared)
    remote_training/backup/store.py         a backup folder on disk (PC worker, restore)
    remote_training/backup/core.py          what to back up, and one backup run (Pi)
    remote_training/backup/destinations.py  Local PC and Hugging Face destinations
    remote_training/backup/manager.py       nightly schedule and status (Pi dashboard)
    scripts/restore_backup.py               restore a snapshot into a new folder
"""
