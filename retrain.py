"""
RETRAIN on manual labels
Builds a fresh YOLO dataset from MANUAL_LABELS + original images, then trains.
"""

import os, shutil, random, yaml
from ultralytics import YOLO

DATASET_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET - Copy"
LABELS_DIR   = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MANUAL_LABELS"
OUTPUT_DIR   = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\YOLO_MANUAL"
MODEL_DIR    = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MODEL_MANUAL"
TRAIN_SPLIT  = 0.85
RANDOM_SEED  = 42

def build_dataset():
    random.seed(RANDOM_SEED)
    classes = sorted([d for d in os.listdir(LABELS_DIR)
                      if os.path.isdir(os.path.join(LABELS_DIR, d))])
    print(f"Found {len(classes)} classes with manual labels.")

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    cfg = {
        "path":  OUTPUT_DIR.replace("\\", "/"),
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
        splits = {"train": txts[:cut], "val": txts[cut:]}

        for split, files in splits.items():
            for txt in files:
                stem     = os.path.splitext(txt)[0]
                lbl_src  = os.path.join(lbl_cls, txt)

                img_src  = None
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = os.path.join(img_cls, stem + ext)
                    if os.path.exists(candidate):
                        img_src = candidate; break
                if img_src is None:
                    continue

                fname = f"{cls.replace(' ','_')}_{os.path.basename(img_src)}"
                shutil.copy2(img_src, os.path.join(OUTPUT_DIR,"images",split,fname))
                shutil.copy2(lbl_src, os.path.join(OUTPUT_DIR,"labels",split,
                             os.path.splitext(fname)[0]+".txt"))
                total += 1

        print(f"  {cls}: {len(txts)} images")

    print(f"\nDataset ready — {total} images total.\n")

def train():
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    print("Loading YOLOv8-nano base model...")
    model = YOLO("yolov8n.pt")
    print("Training on your manual labels — this takes ~60-90 min on CPU.\n")
    model.train(
        data     = yaml_path,
        epochs   = 50,
        imgsz    = 640,
        batch    = 16,
        project  = MODEL_DIR,
        name     = "manual",
        patience = 15,
        save     = True,
        plots    = True,
        workers  = 0,
        degrees  = 15,      # rotation augmentation (was 0)
        flipud   = 0.3,     # vertical flip
        shear    = 5.0,     # slight shear
    )
    best = os.path.join(MODEL_DIR, "manual", "weights", "best.pt")
    print(f"\nTraining complete! Best model: {best}")
    print("Run:  python detect.py   to see results.")

if __name__ == "__main__":
    build_dataset()
    train()
