#!/usr/bin/env python3
"""
scripts/update_showdown_data.py
Synchronizes Gen 9 Random Battle datasets and mechanics tables from upstream
smogon/pokemon-showdown into Chronos.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def update_datasets(showdown_dir: Path) -> bool:
    src_sets = showdown_dir / "data" / "random-battles" / "gen9" / "sets.json"
    if not src_sets.exists():
        print(f"Error: Could not find sets.json at {src_sets}", file=sys.stderr)
        return False

    with open(src_sets, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    target_dir = Path(__file__).resolve().parent.parent / "engine" / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_sets = target_dir / "randbats_sets.json"

    changed = True
    if target_sets.exists():
        try:
            with open(target_sets, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if old_data == new_data:
                changed = False
        except Exception:
            pass

    if changed:
        print(f"Updating {target_sets} with {len(new_data)} species from {src_sets}...")
        with open(target_sets, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2)
        print("Updated randbats_sets.json successfully.")
    else:
        print("randbats_sets.json is already up-to-date with upstream.")

    # Re-run data_extractor.py to ensure mechanics tables and mappings are in sync
    extractor_path = (
        Path(__file__).resolve().parent.parent / "engine" / "data_extractor.py"
    )
    if extractor_path.exists():
        print("Synchronizing mechanics tables and mappings via data_extractor.py...")
        subprocess.run([sys.executable, str(extractor_path)], check=True)

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Update Chronos datasets from smogon/pokemon-showdown"
    )
    parser.add_argument(
        "--showdown-dir",
        type=Path,
        default=Path("smogon-showdown"),
        help="Path to smogon/pokemon-showdown clone",
    )
    args = parser.parse_args()

    showdown_path = args.showdown_dir
    if not showdown_path.exists():
        local_fallback = Path("pokemon-showdown")
        if local_fallback.exists():
            showdown_path = local_fallback

    update_datasets(showdown_path)


if __name__ == "__main__":
    main()
