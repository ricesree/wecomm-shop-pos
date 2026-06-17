"""
Merge GCS feedback (local staging in feedback_data/) into dataset_overall/.

Does not rebuild from train/val/test — dataset_overall/ is the single image store.
Run split_dataset.py later when you need train/val/test for training.

Usage:
    python pull_gcs_feedback.py
    python merge_dataset.py
    python merge_dataset.py --delete-gcs   # also remove feedback images from GCS
    python merge_dataset.py --keep-staging # keep feedback_data/ after merge
"""

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dataset_overall"
FEEDBACK = ROOT / "feedback_data"
THRESHOLDS = ROOT / "results" / "class_thresholds.csv"
BUCKET = "vegdetect-pos-models"

NEW_CLASS_NAMES = {
    "carrot": "Carrot",
    "cucumber": "Cucumber",
    "garlic": "Garlic",
    "sweet": "Sweetpotato",
    "sweetpotato": "Sweetpotato",
}


def to_feedback_key(class_name: str) -> str:
    return re.sub(r"\s+", "_", class_name.strip().lower())


def load_class_map() -> dict[str, str]:
    """Map GCS folder key -> canonical class folder name."""
    mapping = dict(NEW_CLASS_NAMES)
    if THRESHOLDS.is_file():
        with open(THRESHOLDS) as f:
            for row in csv.DictReader(f):
                name = row["class"]
                mapping[to_feedback_key(name)] = name
    return mapping


    
def copy_images(src_dir: Path, dst_class_dir: Path) -> int:
    if not src_dir.is_dir():
        return 0
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src_dir.iterdir():
        if f.suffix.lower() not in IMG_EXT:
            continue
        dest = dst_class_dir / f.name
        if dest.exists():
            dest = dst_class_dir / f"{f.stem}_dup{count}{f.suffix.lower()}"
        shutil.copy2(f, dest)
        count += 1
    return count


def merge_feedback(class_map: dict[str, str]) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not FEEDBACK.is_dir():
        print(f"No staging folder at {FEEDBACK} — run pull_gcs_feedback.py first.")
        return stats

    for bucket_type in ("confirmations", "corrections", "new"):
        base = FEEDBACK / bucket_type
        if not base.is_dir():
            continue
        for fb_dir in base.iterdir():
            if not fb_dir.is_dir():
                continue
            key = fb_dir.name.lower()
            class_name = class_map.get(key)
            if not class_name:
                class_name = fb_dir.name.replace("_", " ").title()
                print(f"  Warning: unmapped folder '{fb_dir.name}' -> '{class_name}'")
            n = copy_images(fb_dir, OUT / class_name)
            if n:
                stats[class_name] = stats.get(class_name, 0) + n
    return stats


def _gsutil_cmd() -> list[str]:
    for name in ("gsutil", "gsutil.cmd"):
        path = shutil.which(name)
        if path:
            return [path]
    win = Path.home() / "AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gsutil.cmd"
    if win.is_file():
        return [str(win)]
    return ["gsutil"]


def delete_gcs_feedback() -> None:
    """Remove feedback image trees from GCS (keeps models/ folder)."""
    gsutil = _gsutil_cmd()
    for prefix in ("confirmations", "corrections", "new"):
        uri = f"gs://{BUCKET}/{prefix}"
        print(f"Deleting {uri} ...")
        result = subprocess.run(gsutil + ["-m", "rm", "-r", uri])
        if result.returncode != 0:
            print(f"  (nothing to delete at {uri} — skipped)")


def normalize_dataset_names(root: Path, classes: list[str] | None = None) -> None:
    from rename_dataset_images import rename_class_folder

    print()
    print("Normalizing image names ...")
    if classes:
        targets = [root / c for c in classes if (root / c).is_dir()]
    else:
        targets = sorted([d for d in root.iterdir() if d.is_dir()])

    total = 0
    for class_dir in targets:
        total += rename_class_folder(class_dir)
    print(f"  Renamed {total} images across {len(targets)} class folders")


def count_images(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not root.is_dir():
        return counts
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        if n:
            counts[class_dir.name] = n
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Merge feedback_data/ into dataset_overall/ (append only)"
    )
    parser.add_argument(
        "--delete-gcs",
        action="store_true",
        help="Delete confirmations/corrections/new from GCS after merge",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep feedback_data/ after merge (default: remove staging folder)",
    )
    args = parser.parse_args()

    class_map = load_class_map()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Merging GCS feedback from feedback_data/ into dataset_overall/ ...")
    fb_stats = merge_feedback(class_map)
    fb_total = sum(fb_stats.values())

    if fb_total == 0:
        print("  No new images to merge.")
    else:
        print(f"  Added {fb_total} images:")
        for cls, n in sorted(fb_stats.items()):
            print(f"    {cls}: +{n}")
        normalize_dataset_names(OUT, list(fb_stats.keys()))

    counts = count_images(OUT)
    total = sum(counts.values())

    print()
    print("=" * 55)
    print(f"DATASET_OVERALL: {OUT}")
    print(f"Classes: {len(counts)} | Total images: {total}")
    print("=" * 55)
    print("Per-class counts:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")

    if not args.keep_staging and FEEDBACK.is_dir():
        shutil.rmtree(FEEDBACK)
        print(f"\nRemoved staging folder {FEEDBACK}")

    if args.delete_gcs:
        print()
        print("Deleting feedback images from GCS (models/ kept) ...")
        delete_gcs_feedback()
        print("GCS feedback folders removed.")


if __name__ == "__main__":
    main()
