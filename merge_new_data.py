"""
Merges 'all new/' images into DATASET_FULL and their .txt labels into MANUAL_LABELS.
Existing folders: images added alongside originals.
New folders (Chilli, Edo, etc.): created fresh.
Run once, then delete this script.
"""

import os, shutil

DATASET_FULL  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET_FULL"
MANUAL_LABELS = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MANUAL_LABELS"
NEW_DATA_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\all new"

# new folder -> target product folder in DATASET_FULL / MANUAL_LABELS
MERGE_MAP = {
    "banana new":              "Banana",
    "boxed sweets new":        "Boxed Sweets",
    "chayote new":             "Chayote",
    "chikku new":              "FRESH CHIKKU",
    "chilli new":              "Thai Chilli",
    "coconut green new":       "Coconut",
    "coconut new":             "Coconut",
    "dosakai new wintermelon": "Dasakai",
    "edo new":                 "Edo",               # new product folder
    "flat velor new":          "FLAT VELOR",
    "garlic new":              "Garlic",
    "graphiti eggplant new":   "Graphiti Eggplant", # new product folder
    "green egg plant new":     "Chinese Green Eggplant",
    "lemon new":               "Lemon",
    "okra new":                "Okra",
    "papaya new":              "Papaya",
    "pearl new":               "Pearl",
    "potato new":              "Potato",
    "squash new":              "Squah",
    "sweet potato new":        "Sweet Potato",
    "tindora new":             "Tindora",
    "WHITE ONION NEW":         "White Onions",
}

imgs_copied = lbls_copied = skipped = 0

for src_folder, target_name in MERGE_MAP.items():
    src_path = os.path.join(NEW_DATA_DIR, src_folder)
    if not os.path.isdir(src_path):
        print(f"  MISSING  {src_folder}")
        continue

    dst_img = os.path.join(DATASET_FULL, target_name)
    dst_lbl = os.path.join(MANUAL_LABELS, target_name)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    for fname in os.listdir(src_path):
        src_file = os.path.join(src_path, fname)
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            dst = os.path.join(dst_img, fname)
            if os.path.exists(dst):
                skipped += 1
            else:
                shutil.copy2(src_file, dst)
                imgs_copied += 1
        elif fname.endswith(".txt"):
            dst = os.path.join(dst_lbl, fname)
            if os.path.exists(dst):
                skipped += 1
            else:
                shutil.copy2(src_file, dst)
                lbls_copied += 1

    print(f"  {src_folder:<30} -> {target_name}")

print(f"\nDone. Images copied: {imgs_copied}  Labels copied: {lbls_copied}  Skipped (already exist): {skipped}")
print("\nNew product folders added to DATASET_FULL:")
for name in ["Edo", "Graphiti Eggplant"]:
    n = len([f for f in os.listdir(os.path.join(DATASET_FULL, name)) if not f.endswith(".txt")]) if os.path.isdir(os.path.join(DATASET_FULL, name)) else 0
    print(f"  {name}: {n} images")
