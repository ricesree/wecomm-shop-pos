# =============================================================================
# TRAIN 14-CLASS POS VEGETABLE DETECTOR - COLAB HIGH-RAM / GPU
# =============================================================================
# Upload the cleaned dataset zip to:
#   MyDrive/TUNE-DATAPOS/YOLO_CATEGORIES.zip
#
# Recommended runtime:
#   A100 GPU + High-RAM. High-RAM helps with cache="ram"; GPU VRAM controls batch.
#
# Important:
#   This script trains a clean 14-class model from yolo11l.pt.
#   Do not fine-tune from an old 15-class .pt that still contains ladystickers.
# =============================================================================

import csv
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import yaml


# -------------------------
# User settings
# -------------------------
DRIVE_ROOT = "/content/drive/MyDrive/TUNE-DATAPOS"
DATA_DIR = "/content/YOLO_CATEGORIES"
MODEL_DIR = f"{DRIVE_ROOT}/MODEL_CATEGORIES"
CKPT_DIR = f"{MODEL_DIR}/checkpoints"
THRESH_DIR = f"{MODEL_DIR}/thresholds"
EXPORT_DIR = f"{MODEL_DIR}/exports"

MODEL_NAME = "yolo11l.pt"  # best upgrade path if yolo11l was already fast enough
RUN_NAME = "yolo11l-14class-img640"
EPOCHS = 100
IMGSZ = 640               # production Cloud Run CPU target; keep this fast
BATCH = 256               # A100 target. Lower only if Colab OOMs.
WORKERS = 16
CACHE = "ram"             # change to False if Colab RAM fills up
DEVICE = 0
SAVE_PERIOD = 100         # saves epoch100.pt; best.pt and last.pt are always saved

CLASS_NAMES = [
    "banana", "beans", "chilli", "coconut", "dasakai",
    "eggplant", "fruit", "gourd", "ladyfinger", "leafy",
    "onion", "root", "special", "tomato",
]


def install_and_mount():
    subprocess.run(["pip", "install", "ultralytics", "onnx", "onnxruntime", "-q"], check=True)
    from google.colab import drive
    drive.mount("/content/drive")

    import torch
    assert torch.cuda.is_available(), "GPU runtime is required for this training script."
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f"GPU: {gpu_name} ({gpu_mem} GB)")


def unzip_dataset():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(THRESH_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    zip_path = f"{DRIVE_ROOT}/YOLO_CATEGORIES_new.zip"
    if not os.path.exists(zip_path):
        zip_path = f"{DRIVE_ROOT}/YOLO_CATEGORIES.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    print(f"Unzipping {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            member.filename = member.filename.replace("\\", "/")
            z.extract(member, "/content")

    yaml_path = f"{DATA_DIR}/data.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update({
        "path": DATA_DIR,
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    })
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"Classes ({cfg['nc']}): {cfg['names']}")
    for split in ("train", "val"):
        img_dir = Path(DATA_DIR) / "images" / split
        count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        print(f"{split}: {count} images")
    return yaml_path


def train_model(yaml_path):
    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        cache=CACHE,
        amp=True,

        patience=10,
        save_period=SAVE_PERIOD,

        cos_lr=True,
        lr0=0.006,
        lrf=0.01,
        warmup_epochs=4,
        warmup_momentum=0.8,
        weight_decay=0.0005,

        degrees=8,
        translate=0.08,
        scale=0.45,
        shear=2.0,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.012,
        hsv_s=0.6,
        hsv_v=0.35,
        mosaic=1.0,
        mixup=0.12,
        close_mosaic=20,

        project=MODEL_DIR,
        name=RUN_NAME,
        save=True,
        plots=True,
    )
    return results


def copy_checkpoints(results):
    weights_dir = Path(results.save_dir) / "weights"
    print("\nCopying checkpoints...")
    copied = set()
    for ckpt in [
        "best.pt", "last.pt", "epoch100.pt",
    ]:
        src = weights_dir / ckpt
        if src.exists():
            dest_name = "best_14class.pt" if ckpt == "best.pt" else ckpt
            shutil.copy(src, Path(CKPT_DIR) / dest_name)
            copied.add(ckpt)
            print(f"Saved: {dest_name}")
            if ckpt == "best.pt":
                shutil.copy(src, Path(CKPT_DIR) / "best_new.pt")
                print("Saved: best_new.pt")
    if "epoch100.pt" not in copied:
        print("WARNING: epoch100.pt was not found. Training may have stopped before epoch 100.")
    return str(weights_dir / "best.pt")


def validate_model(best_path, yaml_path):
    from ultralytics import YOLO

    print("\nValidating...")
    model = YOLO(best_path)
    metrics = model.val(data=yaml_path, imgsz=IMGSZ, device=DEVICE, plots=True)

    print("\n========== RESULTS ==========")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print("\nPer-class mAP50:")
    for name, ap in zip(CLASS_NAMES, metrics.box.ap50):
        print(f"  {name:<15} {ap:.4f}")
    return model


def iou_xyxy(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (
        (b1[2] - b1[0]) * (b1[3] - b1[1]) +
        (b2[2] - b2[0]) * (b2[3] - b2[1]) -
        inter
    )
    return inter / union if union > 1e-6 else 0.0


def optimize_thresholds(model):
    val_img = Path(DATA_DIR) / "images" / "val"
    val_lbl = Path(DATA_DIR) / "labels" / "val"
    min_conf = 0.05
    iou_thr = 0.5
    step = 0.02

    nc = len(CLASS_NAMES)
    all_preds = [[] for _ in range(nc)]
    gt_counts = [0] * nc
    img_files = sorted(val_img.glob("*.jpg")) + sorted(val_img.glob("*.png"))

    print(f"\nOptimizing thresholds on {len(img_files)} val images...")
    for img_path in img_files:
        lbl_path = val_lbl / f"{img_path.stem}.txt"
        result = model.predict(str(img_path), conf=min_conf, verbose=False, imgsz=IMGSZ)[0]
        h, w = result.orig_shape

        gt = []
        if lbl_path.exists():
            for line in open(lbl_path):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                c = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                gt.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                if c < nc:
                    gt_counts[c] += 1

        if not len(result.boxes):
            continue
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        xyxy = result.boxes.xyxy.cpu().numpy()
        matched = set()

        for idx in np.argsort(-confs):
            cid = int(cls_ids[idx])
            if cid >= nc:
                continue
            pred_box = xyxy[idx].tolist()
            best_iou, best_gt = 0.0, -1
            for gi, g in enumerate(gt):
                if g[0] != cid or gi in matched:
                    continue
                score = iou_xyxy(pred_box, g[1:])
                if score > best_iou:
                    best_iou, best_gt = score, gi
            is_tp = 1 if best_iou >= iou_thr else 0
            if is_tp:
                matched.add(best_gt)
            all_preds[cid].append((float(confs[idx]), is_tp))

    sweep = np.arange(min_conf, 0.96, step)
    opt = {}
    for cid, cname in enumerate(CLASS_NAMES):
        preds = np.array(all_preds[cid]) if all_preds[cid] else np.empty((0, 2))
        total_gt = gt_counts[cid]
        if not len(preds) or not total_gt:
            opt[cname] = {"threshold": 0.25, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        best = (-1.0, 0.25, 0.0, 0.0)
        for t in sweep:
            kept = preds[preds[:, 0] >= t]
            tp = float(kept[:, 1].sum()) if len(kept) else 0.0
            fp = float(len(kept) - tp)
            fn = float(total_gt - tp)
            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
            if f1 > best[0]:
                best = (f1, float(t), p, r)
        f1, t, p, r = best
        opt[cname] = {
            "threshold": round(t, 2),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        }
        print(f"{cname:<15} t={t:.2f} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    simple = {k: v["threshold"] for k, v in opt.items()}
    with open(f"{THRESH_DIR}/thresholds.json", "w") as f:
        json.dump(simple, f, indent=2)
    with open(f"{THRESH_DIR}/thresholds_detailed.json", "w") as f:
        json.dump(opt, f, indent=2)
    with open(f"{THRESH_DIR}/thresholds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "threshold", "precision", "recall", "f1"])
        for name, values in opt.items():
            writer.writerow([name, values["threshold"], values["precision"], values["recall"], values["f1"]])
    print(f"\nSaved thresholds -> {THRESH_DIR}")


def export_onnx(best_path):
    from ultralytics import YOLO

    print("\nExporting ONNX...")
    model = YOLO(best_path)
    onnx_path = model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False)
    dest = Path(EXPORT_DIR) / "best_14class_img640.onnx"
    shutil.copy(onnx_path, dest)
    print(f"Saved ONNX -> {dest}")


def main():
    install_and_mount()
    yaml_path = unzip_dataset()
    results = train_model(yaml_path)
    best_path = copy_checkpoints(results)
    model = validate_model(best_path, yaml_path)
    optimize_thresholds(model)
    export_onnx(best_path)

    print("\nDone.")
    print(f"Best PyTorch model: {CKPT_DIR}/best_14class.pt")
    print(f"ONNX export:        {EXPORT_DIR}/best_14class_img640.onnx")
    print(f"Thresholds:         {THRESH_DIR}/thresholds.json")


if __name__ == "__main__":
    main()
