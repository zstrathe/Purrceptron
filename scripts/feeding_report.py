#!/usr/bin/env python3
"""Query and display feeding session history."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PROJECT_ROOT, add_common_args, get_config_with_args, resolve_path
from utils.display import print_error, print_header, print_info


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display feeding session history"
    )
    add_common_args(parser)
    parser.add_argument("--db", type=Path, default=None,
                        help="Path to feeding_log.db (default: from config)")
    parser.add_argument("--days", type=int, default=7,
                        help="Show last N days (default: 7)")
    parser.add_argument("--cat", type=str, default=None,
                        help="Filter by cat name")
    parser.add_argument("--format", choices=["table", "csv", "json"],
                        default="table", help="Output format (default: table)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config_with_args(args)

    feeding_cfg = cfg.get("feeding", {})
    db_path = resolve_path(
        args.db or feeding_cfg.get("db_path", "data/feeding_log.db"),
        PROJECT_ROOT,
    )

    if not db_path.exists():
        print_error(f"Feeding database not found: {db_path}")
        print_info("Run detect.py with --feeding-log first to create feeding data.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    query = (
        "SELECT cat_name, start_time, end_time, duration_seconds, detection_count "
        "FROM feeding_sessions WHERE start_time > ?"
    )
    params: list = [cutoff]

    if args.cat:
        query += " AND cat_name = ?"
        params.append(args.cat)

    query += " ORDER BY start_time DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        print_info("No feeding sessions found in the specified period.")
        return

    if args.format == "csv":
        print("cat_name,start_time,end_time,duration_seconds,detection_count")
        for row in rows:
            print(",".join(str(v) for v in row))

    elif args.format == "json":
        data = [
            {
                "cat": r[0], "start": r[1], "end": r[2],
                "duration_s": r[3], "frames": r[4],
            }
            for r in rows
        ]
        print(json.dumps(data, indent=2))

    else:
        # Table format
        print_header(f"Feeding Report — Last {args.days} Days")
        print(f"  {'Cat':<8s}  {'Start':>19s}  {'End':>19s}  {'Duration':>10s}  {'Frames':>6s}")
        print(f"  {'─' * 70}")
        for cat, start, end, duration, frames in rows:
            dur_str = format_duration(duration)
            print(f"  {cat:<8s}  {start:>19s}  {end:>19s}  {dur_str:>10s}  {frames:>6d}")

        # Per-cat summary
        print(f"\n  {'─' * 70}")
        print_header("Summary by Cat")
        for cat in sorted(set(r[0] for r in rows)):
            cat_rows = [r for r in rows if r[0] == cat]
            total_dur = sum(r[3] for r in cat_rows)
            avg_dur = total_dur / len(cat_rows)
            print(
                f"  {cat}: {len(cat_rows)} sessions, "
                f"total {format_duration(total_dur)}, "
                f"avg {format_duration(avg_dur)}"
            )


if __name__ == "__main__":
    main()
