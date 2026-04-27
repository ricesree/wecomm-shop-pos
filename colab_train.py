# =============================================================================
#  PASTE THIS ENTIRE FILE INTO A SINGLE COLAB CELL AND RUN IT
#  OR paste each STEP block into separate cells one by one.
#
#  BEFORE RUNNING:
#    1. Upload your two folders to Google Drive exactly like this:
#       MyDrive/TUNE-DATAPOS/DATASET/          (the folder with 44 class folders)
#       MyDrive/TUNE-DATAPOS/MANUAL_LABELS/    (the folder with 44 label folders)
#    2. Make sure Runtime -> Change runtime type -> T4 GPU is selected
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  Install + Mount Drive
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "ultralytics", "-q"], check=True)

from google.colab import drive
drive.mount("/content/drive")
print("Drive mounted.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  Copy data from Drive → /content/  (much faster I/O during training)
# ─────────────────────────────────────────────────────────────────────────────
import shutil, os

DRIVE_ROOT    = "/content/drive/MyDrive/TUNE-DATAPOS"
DATASET_SRC   = os.path.join(DRIVE_ROOT, "DATASET")
LABELS_SRC    = os.path.join(DRIVE_ROOT, "MANUAL_LABELS")

DATASET_DIR   = "/content/DATASET"
LABELS_DIR    = "/content/MANUAL_LABELS"
OUTPUT_DIR    = "/content/YOLO_MANUAL"
MODEL_DIR     = "/content/drive/MyDrive/MODEL_MANUAL"   # save directly to Drive

os.makedirs(MODEL_DIR, exist_ok=True)

print("Copying DATASET  to /content/ ...")
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
shutil.copytree(DATASET_SRC, DATASET_DIR)
print("Copying MANUAL_LABELS to /content/ ...")
if os.path.exists(LABELS_DIR):
    shutil.rmtree(LABELS_DIR)
shutil.copytree(LABELS_SRC, LABELS_DIR)
print("Data copied.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  Build YOLO dataset (train / val split)
# ─────────────────────────────────────────────────────────────────────────────
import random, yaml

TRAIN_SPLIT  = 0.85
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)

classes = sorted([
    d for d in os.listdir(LABELS_DIR)
    if os.path.isdir(os.path.join(LABELS_DIR, d))
])
print(f"Found {len(classes)} classes.")

for split in ("train", "val"):
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

cfg = {
    "path":  OUTPUT_DIR,
    "train": "images/train",
    "val":   "images/val",
    "nc":    len(classes),
    "names": classes,
}
with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

total = 0
for cls in classes:
    lbl_cls = os.path.join(LABELS_DIR, cls)
    img_cls = os.path.join(DATASET_DIR, cls)

    txts = [f for f in os.listdir(lbl_cls) if f.endswith(".txt")]
    random.shuffle(txts)
    cut = max(1, int(len(txts) * TRAIN_SPLIT))
    splits_map = {"train": txts[:cut], "val": txts[cut:]}

    for split, files in splits_map.items():
        for txt in files:
            stem    = os.path.splitext(txt)[0]
            lbl_src = os.path.join(lbl_cls, txt)
            img_src = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = os.path.join(img_cls, stem + ext)
                if os.path.exists(cand):
                    img_src = cand
                    break
            if img_src is None:
                continue
            fname = f"{cls.replace(' ', '_')}_{os.path.basename(img_src)}"
            shutil.copy2(img_src, os.path.join(OUTPUT_DIR, "images", split, fname))
            shutil.copy2(lbl_src, os.path.join(OUTPUT_DIR, "labels", split,
                         os.path.splitext(fname)[0] + ".txt"))
            total += 1

    print(f"  {cls}: {len(txts)} images")

print(f"\nDataset ready — {total} total images  ({int(total*0.85)} train / {int(total*0.15)} val)\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  Train  —  YOLOv8-SMALL  (upgraded from nano)
#
#  Model comparison:
#    yolov8n.pt  (nano)   3M params   8 GFLOPs   <-- what you used before
#    yolov8s.pt  (small) 11M params  28 GFLOPs   <-- upgraded (this run)
#    yolov8m.pt  (medium)25M params  79 GFLOPs   <-- even better, ~2x slower
#
#  Changes vs last run:
#    - Model:    yolov8s.pt  (was yolov8n.pt)
#    - Epochs:   75          (was 50)
#    - degrees:  15          (was 0  — rotation now enabled)
#    - flipud:   0.3         (was 0  — vertical flip enabled)
#    - shear:    5.0         (was 0  — shear enabled)
#    - Dataset:  ~5x larger  (augmented images added offline)
# ─────────────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")

print("Loading YOLOv8-Small model...")
model = YOLO("yolov8s.pt")

print("Starting training...\n")
model.train(
    data        = yaml_path,
    epochs      = 75,
    imgsz       = 640,
    batch       = 16,
    project     = MODEL_DIR,
    name        = "small-aug",
    patience    = 20,
    save        = True,
    plots       = True,
    workers     = 4,
    device      = 0,
    # online augmentation (on top of the offline augmented images)
    degrees     = 15,
    flipud      = 0.3,
    shear       = 5.0,
    hsv_h       = 0.02,
    hsv_s       = 0.8,
    hsv_v       = 0.5,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  Validate best model + print per-class scores
# ─────────────────────────────────────────────────────────────────────────────
import os

best_pt = os.path.join(MODEL_DIR, "small-aug", "weights", "best.pt")
print(f"\nLoading best model: {best_pt}")

val_model = YOLO(best_pt)
metrics   = val_model.val(
    data    = yaml_path,
    imgsz   = 640,
    device  = 0,
    split   = "val",
    plots   = True,
    save_json = True,
)

print("\n========== PER-CLASS RESULTS ==========")
print(f"{'Class':<30} {'Precision':>10} {'Recall':>8} {'mAP50':>8} {'mAP50-95':>10}")
print("-" * 70)

names = val_model.names
ap_per_class = metrics.box.ap_class_index
maps          = metrics.box.maps         # mAP50-95 per class
map50s        = metrics.box.ap50         # mAP50 per class (if available)

try:
    for i, cls_idx in enumerate(ap_per_class):
        cls_name = names[cls_idx]
        p  = metrics.box.p[i]   if hasattr(metrics.box, 'p')   else 0
        r  = metrics.box.r[i]   if hasattr(metrics.box, 'r')   else 0
        m50  = map50s[i] if map50s is not None else 0
        m5095 = maps[i]
        grade = "EXCELLENT" if m50 >= 0.95 else "GOOD" if m50 >= 0.85 else "OK" if m50 >= 0.70 else "!! WEAK"
        print(f"  {cls_name:<28} {p:>9.1%} {r:>7.1%} {m50:>7.1%} {m5095:>9.1%}   {grade}")
except Exception:
    print("(Run the validation cell above to see per-class breakdown)")

print("-" * 70)
print(f"\n  Overall mAP50    : {metrics.box.map50:.1%}")
print(f"  Overall mAP50-95 : {metrics.box.map:.1%}")
print(f"\nBest model saved to: {best_pt}")
print("Download it from Google Drive: MODEL_MANUAL/small-aug/weights/best.pt")
