"""
download_corrections.py
=======================
Downloads all saved corrections from GCS and merges them into your
local DATASET + MANUAL_LABELS folders, ready for fine-tuning.

Run:
  python download_corrections.py

Then:
  python augment.py          (adds augmented copies of new images)
  Upload to Drive + Colab fine-tune from best.pt
"""

import os
import json
from google.cloud import storage

BUCKET_NAME  = "vegdetect-feedback-1076778092661"
DATASET_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET - Copy"
LABELS_DIR   = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MANUAL_LABELS"
DOWNLOAD_DIR = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\CORRECTIONS_DOWNLOAD"

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs  = list(bucket.list_blobs())

    print(f"Found {len(blobs)} files in gs://{BUCKET_NAME}\n")

    stats = {"corrections": {}, "new_classes": {}}
    saved = 0

    for blob in blobs:
        if not blob.name.endswith(".jpg"):
            continue

        parts  = blob.name.split("/")
        if len(parts) < 3:
            continue
        folder, cls_name, filename = parts[0], parts[1], parts[2]

        # Create destination directories
        img_dest = os.path.join(DATASET_DIR, cls_name)
        lbl_dest = os.path.join(LABELS_DIR,  cls_name)
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)

        out_img = os.path.join(img_dest, f"fb_{filename}")
        if os.path.exists(out_img):
            continue   # already downloaded

        # Download image
        blob.download_to_filename(out_img)

        # Create a full-frame YOLO label (bbox covers whole image)
        # class_id will be fixed during retrain when data.yaml is rebuilt
        # We use 0 as placeholder — the retrain script handles class ordering
        lbl_file = os.path.join(lbl_dest, f"fb_{os.path.splitext(filename)[0]}.txt")
        with open(lbl_file, "w") as f:
            f.write("0 0.500000 0.500000 0.900000 0.900000\n")

        count_key = "corrections" if folder == "corrections" else "new_classes"
        stats[count_key][cls_name] = stats[count_key].get(cls_name, 0) + 1
        saved += 1

    print("=== Corrections downloaded ===")
    if stats["corrections"]:
        print("\nCorrections (existing classes):")
        for cls, n in sorted(stats["corrections"].items()):
            print(f"  {cls:<30} +{n} images")

    if stats["new_classes"]:
        print("\nNew classes:")
        for cls, n in sorted(stats["new_classes"].items()):
            print(f"  {cls:<30} +{n} images  << NEW CLASS")

    print(f"\nTotal downloaded: {saved} images")
    print(f"Saved to: {DATASET_DIR}")
    print(f"\nNEXT STEPS:")
    print(f"  1. python augment.py          (generate augmented copies)")
    print(f"  2. Zip DATASET + MANUAL_LABELS")
    print(f"  3. Upload to Google Drive")
    print(f"  4. Run Colab fine-tune from best.pt (30 epochs)")
    print(f"  5. Download new best.pt")
    print(f"  6. Copy to api/best.pt and redeploy")

if __name__ == "__main__":
    main()
