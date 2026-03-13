"""
Prepare a session-aware train/val split for the MobileNetV2 cat classifier.

Source:  data/raw/<cat>/<timestamp_ms>.jpg
Output:  data/classifier_dataset/{train,val}/<cat>/

Session inference: images captured within `session_gap_minutes` of each other
belong to the same session.  Val set = 20% of *sessions* (not images), ensuring
no visually-identical crops appear in both splits.
"""

import argparse
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root and config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None  # handled below


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("pyyaml is required: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Session inference
# ---------------------------------------------------------------------------

def infer_sessions(
    image_paths: list[Path], gap_minutes: float
) -> dict[str, list[Path]]:
    """Group images into sessions by timestamp gap.

    Returns dict: session_id -> list[Path], ordered by timestamp.
    """
    if not image_paths:
        return {}

    gap_ms = gap_minutes * 60 * 1000

    # Parse timestamps from stem (e.g. "1741608123456" -> int)
    def ts(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            pass
        # Handle CatName_YYYYMMDD_HHMMSS_mmm format
        parts = p.stem.rsplit("_", 3)
        if len(parts) >= 4:
            try:
                date_s, time_s, ms_s = parts[-3], parts[-2], parts[-1]
                return int(date_s + time_s) * 1000 + int(ms_s)
            except ValueError:
                pass
        return 0

    sorted_paths = sorted(image_paths, key=ts)

    sessions: dict[str, list[Path]] = {}
    session_idx = 0
    session_key = f"session_{session_idx:03d}"
    sessions[session_key] = [sorted_paths[0]]

    for prev, curr in zip(sorted_paths, sorted_paths[1:]):
        if ts(curr) - ts(prev) >= gap_ms:
            session_idx += 1
            session_key = f"session_{session_idx:03d}"
            sessions[session_key] = []
        sessions[session_key].append(curr)

    return sessions


# ---------------------------------------------------------------------------
# Link or copy helpers
# ---------------------------------------------------------------------------

def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare classifier dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.yaml",
        help="Path to project.yaml",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Fraction of sessions to assign to val (default 0.20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for session shuffle",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Override labeled crops directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: data/classifier_dataset)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cats: list[str] = cfg.get("cats", [])
    if not cats:
        print("ERROR: no cats defined in config", file=sys.stderr)
        sys.exit(1)

    session_gap = cfg.get("classifier_training", {}).get("session_gap_minutes", 5)

    source_root = args.source_dir or (PROJECT_ROOT / "data" / "raw")
    output_root = args.output_dir or (PROJECT_ROOT / "data" / "classifier_dataset")

    rng = random.Random(args.seed)

    # Collect per-cat sessions
    per_cat: dict[str, dict[str, list[Path]]] = {}
    for cat in cats:
        cat_dir = source_root / cat
        if not cat_dir.is_dir():
            print(f"  [warn] No directory for '{cat}' at {cat_dir}")
            per_cat[cat] = {}
            continue
        images = sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.jpeg"))
        if not images:
            print(f"  [warn] No images found for '{cat}' in {cat_dir}")
            per_cat[cat] = {}
            continue
        per_cat[cat] = infer_sessions(images, gap_minutes=session_gap)

    print("\n=== Session summary ===")
    for cat in cats:
        sessions = per_cat.get(cat, {})
        total_imgs = sum(len(v) for v in sessions.values())
        print(f"  {cat}: {total_imgs} images, {len(sessions)} sessions")
    print()

    # Assign sessions to train/val per cat
    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}

    for cat in cats:
        sessions = per_cat.get(cat, {})
        if not sessions:
            train_counts[cat] = 0
            val_counts[cat] = 0
            continue

        session_ids = list(sessions.keys())
        rng.shuffle(session_ids)

        n_val = max(1, round(len(session_ids) * args.val_frac))
        val_ids = set(session_ids[:n_val])
        train_ids = set(session_ids[n_val:])

        train_imgs: list[Path] = []
        val_imgs: list[Path] = []
        for sid, paths in sessions.items():
            if sid in val_ids:
                val_imgs.extend(paths)
            else:
                train_imgs.extend(paths)

        # Write outputs (clear stale files from previous runs first)
        for split, img_list in [("train", train_imgs), ("val", val_imgs)]:
            dst_dir = output_root / split / cat
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img_path in img_list:
                link_or_copy(img_path, dst_dir / img_path.name)

        train_counts[cat] = len(train_imgs)
        val_counts[cat] = len(val_imgs)
        print(
            f"  {cat}: {len(train_imgs)} train ({len(train_ids)} sessions), "
            f"{len(val_imgs)} val ({len(val_ids)} sessions)"
        )

    # Verify no session appears in both splits (sanity check via filenames)
    print("\n=== Split verification ===")
    all_ok = True
    for cat in cats:
        train_dir = output_root / "train" / cat
        val_dir = output_root / "val" / cat
        if not train_dir.exists() or not val_dir.exists():
            continue
        train_names = {p.name for p in train_dir.iterdir()}
        val_names = {p.name for p in val_dir.iterdir()}
        overlap = train_names & val_names
        if overlap:
            print(f"  [ERROR] {cat}: {len(overlap)} files in both splits! {list(overlap)[:3]}")
            all_ok = False
        else:
            print(f"  {cat}: OK (no overlap)")

    # Class imbalance warning
    total_per_cat = {cat: train_counts[cat] + val_counts[cat] for cat in cats}
    max_count = max(total_per_cat.values()) if total_per_cat else 0
    if max_count > 0:
        print("\n=== Class balance ===")
        for cat in cats:
            ratio = total_per_cat[cat] / max_count
            flag = " [IMBALANCED]" if ratio < 0.65 else ""
            print(f"  {cat}: {total_per_cat[cat]} images ({ratio:.0%} of max){flag}")

    print(f"\nOutput: {output_root}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
