"""
Pull GCS feedback images (confirmations, corrections, new) to local folders.

Usage:
    pip install google-cloud-storage   # optional; uses gsutil if not installed
    python pull_gcs_feedback.py

    python pull_gcs_feedback.py --bucket vegdetect-pos-models --dest ./feedback_data
    python pull_gcs_feedback.py --analyze-only
"""

import argparse
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Original test-set weak classes (low F1 from classification_report.txt)
MODEL_WEAK_CLASSES = [
    "Chilli",      # F1 0.67
    "Okra",        # F1 0.62
    "Tindora",     # F1 0.75
    "Potato",      # F1 0.82, precision 0.69
    "Pumpkin",     # F1 0.80
    "Coconut",     # recall 0.75
    "Dasakai",     # recall 0.75
    "Onion",       # precision 0.78
    "Tomato",      # user-reported live issues
    "Fruits",      # broad class, confusion hub
]


def _gsutil_cmd() -> list[str]:
    for name in ("gsutil", "gsutil.cmd"):
        path = shutil.which(name)
        if path:
            return [path]
    win = Path.home() / "AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gsutil.cmd"
    if win.is_file():
        return [str(win)]
    return ["gsutil"]


def pull_with_gsutil(bucket: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    gsutil = _gsutil_cmd()
    for prefix in ("confirmations", "corrections", "new"):
        src = f"gs://{bucket}/{prefix}"
        print(f"Downloading {src} ...")
        result = subprocess.run(gsutil + ["-m", "cp", "-r", src, str(dest)])
        if result.returncode != 0:
            print(f"  (no images at {src} — skipped)")


def pull_with_python(bucket: str, dest: Path) -> None:
    from google.cloud import storage

    client = storage.Client()
    b = client.bucket(bucket)
    for prefix in ("confirmations/", "corrections/", "new/"):
        print(f"Downloading gs://{bucket}/{prefix} ...")
        for blob in b.list_blobs(prefix=prefix):
            if blob.name.endswith("/"):
                continue
            ext = Path(blob.name).suffix.lower()
            if ext not in IMG_EXT:
                continue
            local = dest / blob.name
            local.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local))


def count_images(folder: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not folder.is_dir():
        return counts
    for cls_dir in folder.iterdir():
        if cls_dir.is_dir():
            n = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT)
            if n:
                counts[cls_dir.name] = n
    return counts


def normalize_key(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def analyze(dest: Path) -> None:
    conf = count_images(dest / "confirmations")
    corr = count_images(dest / "corrections")
    new = count_images(dest / "new")

    all_names = sorted(set(conf) | set(corr) | set(new))
    print("\n" + "=" * 60)
    print("FEEDBACK SUMMARY")
    print("=" * 60)
    print(f"Confirmations : {sum(conf.values())} images")
    print(f"Corrections   : {sum(corr.values())} images")
    print(f"New produce   : {sum(new.values())} images")
    print(f"Local path    : {dest.resolve()}")
    print()
    print(f"{'Class':<22} {'Confirm':>8} {'Correct':>8} {'New':>6} {'Wrong%':>8}")
    print("-" * 60)

    rows = []
    for c in all_names:
        a, b, d = conf.get(c, 0), corr.get(c, 0), new.get(c, 0)
        total = a + b
        wrong_pct = (b / total * 100) if total else 0.0
        rows.append((c, a, b, d, wrong_pct))
        print(f"{c:<22} {a:8} {b:8} {d:6} {wrong_pct:7.1f}%")

    print("\n" + "=" * 60)
    print("WEAK CLASSES (from live feedback — high corrections)")
    print("=" * 60)
    for c, a, b, d, w in sorted(rows, key=lambda x: -x[2]):
        if b >= 3:
            print(f"  {c}: {b} corrections vs {a} confirmations ({w:.0f}% wrong among feedback)")

    print("\n" + "=" * 60)
    print("WEAK CLASSES (from original model test — low F1)")
    print("=" * 60)
    key_map = {normalize_key(c): c for c in all_names}
    for model_cls in MODEL_WEAK_CLASSES:
        k = normalize_key(model_cls)
        fb = key_map.get(k, k)
        a, b = conf.get(fb, 0), corr.get(fb, 0)
        print(f"  {model_cls}: test F1 weak | feedback: {b} corrections, {a} confirmations")

    print("\n" + "=" * 60)
    print("NEW PRODUCE (not in original 27 classes)")
    print("=" * 60)
    for c, n in sorted(new.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n} images")


def main():
    parser = argparse.ArgumentParser(description="Pull GCS feedback and analyze weak classes")
    parser.add_argument("--bucket", default="vegdetect-pos-models")
    parser.add_argument(
        "--dest",
        default=str(Path(__file__).resolve().parent / "feedback_data"),
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip download; only analyze existing local folder",
    )
    parser.add_argument("--use-python", action="store_true", help="Use google-cloud-storage instead of gsutil")
    args = parser.parse_args()

    dest = Path(args.dest)

    if not args.analyze_only:
        try:
            if args.use_python:
                pull_with_python(args.bucket, dest)
            else:
                pull_with_gsutil(args.bucket, dest)
        except FileNotFoundError:
            print("gsutil not found — trying google-cloud-storage ...")
            pull_with_python(args.bucket, dest)
        except subprocess.CalledProcessError as e:
            print(f"Download failed: {e}")
            sys.exit(1)
        print("Download complete.\n")

    if not dest.exists():
        print(f"No data at {dest}. Run without --analyze-only first.")
        sys.exit(1)

    analyze(dest)


if __name__ == "__main__":
    main()
