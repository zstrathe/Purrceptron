"""
Import captured images from the staging directory into a version's raw_images/.

Copies new JPEGs (and their .json sidecars if present) from data/raw/<cat>/
into versions/{version}/raw_images/<cat>/, skipping files that already exist.

Usage:
    python train/import_captures.py \\
        --source-dir data/raw \\
        --output-dir versions/v1/raw_images \\
        [--config configs/project.yaml]
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Path) -> dict:
    if yaml is None:
        raise ImportError("pyyaml is required: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def import_captures(source_dir: Path, output_dir: Path, cats: list[str]) -> None:
    subdirs = list(cats) + ["unlabeled"]
    total_imported = 0

    for subdir in subdirs:
        src = source_dir / subdir
        dst = output_dir / subdir
        if not src.is_dir():
            continue
        dst.mkdir(parents=True, exist_ok=True)

        existing = {p.name for p in dst.glob("*.jpg")}
        imported = 0

        for jpg in sorted(src.glob("*.jpg")):
            if jpg.name in existing:
                continue
            shutil.copy2(jpg, dst / jpg.name)
            sidecar = jpg.with_suffix(".json")
            if sidecar.exists():
                shutil.copy2(sidecar, dst / sidecar.name)
            imported += 1

        if imported:
            print(f"  {subdir}: imported {imported} new image(s)")
        total_imported += imported

    print(f"Import complete: {total_imported} new image(s) total")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import captures from data/raw/ staging into a version's raw_images/"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Staging directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination raw_images directory (e.g. versions/v1/raw_images)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.yaml",
        help="Path to project.yaml (default: configs/project.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cats: list[str] = cfg.get("cats", [])
    if not cats:
        print("ERROR: no cats defined in config", file=sys.stderr)
        sys.exit(1)

    source = args.source_dir if args.source_dir.is_absolute() else PROJECT_ROOT / args.source_dir
    output = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    print(f"Source: {source}")
    print(f"Output: {output}")
    import_captures(source, output, cats)


if __name__ == "__main__":
    main()
