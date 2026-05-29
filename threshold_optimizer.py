"""
Per-Class Confidence Threshold Optimizer for YOLOv8
=====================================================
Run this in Colab AFTER training to find the optimal confidence
threshold for each vegetable class individually.

Why per-class thresholds help:
- Each class has different background-confusion rates.
  ladyfinger and beans are often confused with background at low
  confidence, while banana and tomato are reliably detected even
  at 0.20. A single global threshold forces a compromise.
- Per-class thresholds maximize F1 for every class independently,
  reducing false positives on hard classes without hurting recall
  on easy ones.

How F1 optimization works:
- We run inference ONCE at a very low conf to collect all raw
  predictions, then sweep thresholds analytically (no re-inference).
- For each threshold t: compute TP / FP / FN → Precision / Recall / F1
- The threshold with the highest F1 is selected.

Output: thresholds.json, thresholds_detailed.json, thresholds.csv, plots/
Copy thresholds.json into api/ and push to deploy.
"""

import os, json, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
MODEL_PATH  = "/content/best.pt"
VAL_IMG_DIR = "/content/YOLO_CATEGORIES/images/val"
VAL_LBL_DIR = "/content/YOLO_CATEGORIES/labels/val"
OUT_DIR     = "/content/thresholds"
DRIVE_DIR   = "/content/drive/MyDrive/TUNE-DATAPOS/MODEL_CATEGORIES/thresholds"

IOU_THRESH  = 0.5    # IoU required to count a prediction as TP
MIN_CONF    = 0.05   # lowest threshold to consider (floor for sweep)
STEP        = 0.02   # sweep step size: 0.05, 0.07, 0.09, ... 0.95

CLASS_NAMES = [
    "banana", "beans", "chilli", "coconut", "dasakai",
    "eggplant", "fruit", "gourd", "ladyfinger", "leafy",
    "onion", "root", "special", "tomato"
]

# =========================
# LOAD MODEL
# =========================
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")
img_files = sorted(Path(VAL_IMG_DIR).glob("*.jpg")) + \
            sorted(Path(VAL_IMG_DIR).glob("*.png"))
print(f"Val images: {len(img_files)}\n")


# =========================
# HELPER: IoU between two boxes in (x1,y1,x2,y2) pixel format
# =========================
def box_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 1e-6 else 0.0


# =========================
# STEP 1 — COLLECT PREDICTIONS VS GROUND TRUTH
# Run model ONCE at MIN_CONF. For each prediction record (conf, is_tp).
# Greedy TP matching: highest-confidence prediction gets priority.
# =========================
nc         = len(CLASS_NAMES)
all_preds  = [[] for _ in range(nc)]   # per class: [(conf, is_tp), ...]
gt_counts  = [0] * nc

print(f"Running inference on {len(img_files)} val images at conf={MIN_CONF}...")

for img_path in tqdm(img_files):
    lbl_path = Path(VAL_LBL_DIR) / (img_path.stem + ".txt")

    result  = model.predict(str(img_path), conf=MIN_CONF, verbose=False, imgsz=640)[0]
    img_h, img_w = result.orig_shape

    # Parse GT labels (YOLO normalized format → absolute pixels)
    gt_boxes = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * img_w;  y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w;  y2 = (cy + bh / 2) * img_h
                gt_boxes.append([cls_id, x1, y1, x2, y2])
                if cls_id < nc:
                    gt_counts[cls_id] += 1

    if len(result.boxes) == 0:
        continue

    confs   = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    xyxy    = result.boxes.xyxy.cpu().numpy()

    # Sort by confidence descending — highest conf prediction gets first pick of GT boxes
    order      = np.argsort(-confs)
    matched_gt = set()

    for idx in order:
        cls_id   = int(cls_ids[idx])
        conf_val = float(confs[idx])
        pred_box = xyxy[idx].tolist()

        # Find best unmatched GT box of same class
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gt[0] != cls_id or gi in matched_gt:
                continue
            iou = box_iou(pred_box, gt[1:])
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        is_tp = 1 if best_iou >= IOU_THRESH else 0
        if is_tp:
            matched_gt.add(best_gi)

        if cls_id < nc:
            all_preds[cls_id].append((conf_val, is_tp))

print("\nCollection complete. GT counts per class:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<15} GT={gt_counts[i]:4d}  Preds={len(all_preds[i]):4d}")


# =========================
# STEP 2 — SWEEP THRESHOLDS, FIND BEST F1 PER CLASS
# No re-inference needed — we threshold the already-collected predictions.
# =========================
print("\nOptimizing per-class thresholds...")

sweep      = np.arange(MIN_CONF, 0.96, STEP)
results    = {}

for cls_id, cls_name in enumerate(CLASS_NAMES):
    preds    = np.array(all_preds[cls_id]) if all_preds[cls_id] else np.empty((0, 2))
    total_gt = gt_counts[cls_id]

    if len(preds) == 0 or total_gt == 0:
        results[cls_name] = {"threshold": 0.25, "precision": 0.0,
                             "recall": 0.0, "f1": 0.0, "note": "no data"}
        print(f"  {cls_name:<15} → no data, default 0.25")
        continue

    best_f1, best_t, best_p, best_r = -1.0, 0.25, 0.0, 0.0
    curve = []

    for t in sweep:
        kept = preds[preds[:, 0] >= t]
        tp   = float(kept[:, 1].sum()) if len(kept) > 0 else 0.0
        fp   = float(len(kept) - tp)
        fn   = float(total_gt - tp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        curve.append((round(float(t), 3), round(prec, 4), round(rec, 4), round(f1, 4)))

        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, float(t), prec, rec

    results[cls_name] = {
        "threshold": round(best_t, 2),
        "precision": round(best_p, 4),
        "recall":    round(best_r, 4),
        "f1":        round(best_f1, 4),
        "curve":     curve
    }
    print(f"  {cls_name:<15} → threshold={best_t:.2f}  "
          f"P={best_p:.3f}  R={best_r:.3f}  F1={best_f1:.3f}")


# =========================
# STEP 3 — SAVE RESULTS
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

# thresholds.json — minimal, goes into api/ for production
simple = {k: v["threshold"] for k, v in results.items()}
with open(f"{OUT_DIR}/thresholds.json", "w") as f:
    json.dump(simple, f, indent=2)

# thresholds_detailed.json — includes P/R/F1 per class
detailed = {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
            for k, v in results.items()}
with open(f"{OUT_DIR}/thresholds_detailed.json", "w") as f:
    json.dump(detailed, f, indent=2)

# thresholds.csv — for spreadsheet analysis
with open(f"{OUT_DIR}/thresholds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "threshold", "precision", "recall", "f1"])
    for cls_name, info in results.items():
        w.writerow([cls_name, info["threshold"], info.get("precision", 0),
                    info.get("recall", 0), info.get("f1", 0)])

print(f"\nSaved → {OUT_DIR}/thresholds.json")
print(f"Saved → {OUT_DIR}/thresholds_detailed.json")
print(f"Saved → {OUT_DIR}/thresholds.csv")


# =========================
# STEP 4 — GENERATE PLOTS (F1 / Precision / Recall vs threshold per class)
# =========================
plots_dir = f"{OUT_DIR}/plots"
os.makedirs(plots_dir, exist_ok=True)

for cls_name, info in results.items():
    curve = info.get("curve", [])
    if not curve:
        continue

    ts   = [r[0] for r in curve]
    prec = [r[1] for r in curve]
    rec  = [r[2] for r in curve]
    f1   = [r[3] for r in curve]
    best_t = info["threshold"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ts, prec, label="Precision",  color="steelblue",   linewidth=1.5)
    ax.plot(ts, rec,  label="Recall",     color="forestgreen", linewidth=1.5)
    ax.plot(ts, f1,   label="F1 Score",   color="crimson",     linewidth=2.5)
    ax.axvline(best_t, color="gray", linestyle="--", alpha=0.7,
               label=f"Best threshold = {best_t}")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{cls_name}  |  Best F1 = {info['f1']:.3f}  @  t = {best_t}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(MIN_CONF, 0.95)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{cls_name}.png", dpi=100, bbox_inches="tight")
    plt.close()

print(f"Plots saved → {plots_dir}/")


# =========================
# STEP 5 — BACKUP TO DRIVE
# =========================
import shutil
try:
    os.makedirs(DRIVE_DIR, exist_ok=True)
    for fname in ["thresholds.json", "thresholds_detailed.json", "thresholds.csv"]:
        shutil.copy(f"{OUT_DIR}/{fname}", f"{DRIVE_DIR}/{fname}")
    drive_plots = f"{DRIVE_DIR}/plots"
    os.makedirs(drive_plots, exist_ok=True)
    for p in Path(plots_dir).glob("*.png"):
        shutil.copy(str(p), f"{drive_plots}/{p.name}")
    print(f"Backed up to Drive: {DRIVE_DIR}")
except Exception as e:
    print(f"Drive backup skipped: {e}")


# =========================
# FINAL SUMMARY
# =========================
print("\n" + "=" * 52)
print("OPTIMAL PER-CLASS THRESHOLDS")
print("=" * 52)
print(json.dumps(simple, indent=2))
print("\nNext step: copy thresholds.json into api/ folder and push to deploy.")
