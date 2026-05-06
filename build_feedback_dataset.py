"""
GCS Feedback → YOLO Dataset Builder
-------------------------------------
Downloads confirmed + corrected images from GCS feedback bucket
and merges them into YOLO_CATEGORIES/ (on top of existing dataset).

Run this on Colab AFTER unzipping YOLO_CATEGORIES.zip, BEFORE training:
    python build_feedback_dataset.py

Requirements: pip install google-cloud-storage
"""

import os, json, random
from google.cloud import storage

BUCKET_NAME = "vegdetect-feedback-1076778092661"
DATA_DIR    = "/content/YOLO_CATEGORIES"   # where YOLO_CATEGORIES was extracted
TRAIN_SPLIT = 0.85
random.seed(42)

INCLUDE_FOLDERS = ["confirmations", "corrections"]  # skip no_bbox, new_classes

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    added = 0
    skipped = 0

    for blob in bucket.list_blobs():
        if not blob.name.endswith(".jpg"):
            continue

        parts = blob.name.split("/")
        if len(parts) < 3:
            continue

        folder, cls_name, filename = parts[0], parts[1], parts[2]

        if folder not in INCLUDE_FOLDERS:
            continue

        stem     = os.path.splitext(filename)[0]
        lbl_name = f"{stem}.txt"
        lbl_blob = bucket.blob(f"{folder}/{cls_name}/{lbl_name}")

        if not lbl_blob.exists():
            skipped += 1
            continue

        lbl_content = lbl_blob.download_as_text().strip()
        if not lbl_content:
            skipped += 1
            continue

        split   = "train" if random.random() < TRAIN_SPLIT else "val"
        out_img = f"{DATA_DIR}/images/{split}/{folder}_{cls_name}_{filename}"
        out_lbl = f"{DATA_DIR}/labels/{split}/{folder}_{cls_name}_{lbl_name}"

        if os.path.exists(out_img):
            skipped += 1
            continue

        img_bytes = blob.download_as_bytes()
        with open(out_img, "wb") as f:
            f.write(img_bytes)
        with open(out_lbl, "w") as f:
            f.write(lbl_content + "\n")

        added += 1

    print(f"Done — {added} feedback images added to dataset  ({skipped} skipped / no label)")
    print(f"Train: {len(os.listdir(DATA_DIR+'/images/train'))}  Val: {len(os.listdir(DATA_DIR+'/images/val'))}")

if __name__ == "__main__":
    main()
