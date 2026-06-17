"""
train.py — EfficientNet-B3 produce classification (single-file training pipeline)

Run:
    python train.py

Expects class-wise folders under DATASET_DIR. Creates train/val/test splits,
trains with on-the-fly Albumentations (no augmented image files saved).
"""

from __future__ import annotations

import copy
import json
import os
import platform
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  PATHS — edit these for your machine
# ═══════════════════════════════════════════════════════════════════════════════
PROJECT_DIR = Path(r"D:\DATASET\DATASET_NEW_efficient")
DATASET_DIR = PROJECT_DIR / "dataset_overall"
SPLIT_ROOT = PROJECT_DIR  # train/, val/, test/ created here
RESULTS_DIR = PROJECT_DIR / "results_new"

# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
EPOCHS = 45
PATIENCE = 5
IMG_SIZE = 300
LR = 1e-4
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Shape-sensitive / large produce — mild augmentation only
SOFT_AUG_CLASSES = {
    "pumpkin",
    "squash",
    "squah",
    "snake guard",
    "turai",
    "cucumber",
    "dasakai",
    "karela",
    "eggplant",
    "banana flower",
}

NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
TO_TENSOR = ToTensorV2()

REQUIREMENTS = """
torch
torchvision
timm
albumentations
opencv-python
scikit-learn
pandas
numpy
matplotlib
tqdm
onnx
onnxruntime
onnxscript
psutil
pillow
"""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_soft_class(class_name: str) -> bool:
    return class_name.strip().lower() in SOFT_AUG_CLASSES


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )


def split_counts(n: int, train_r: float, val_r: float, test_r: float) -> tuple[int, int, int]:
    if n == 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    train_n = max(1, int(round(n * train_r)))
    val_n = max(1, int(round(n * val_r)))
    test_n = max(1, int(round(n * test_r)))
    while train_n + val_n + test_n > n:
        if test_n > 1:
            test_n -= 1
        elif val_n > 1:
            val_n -= 1
        else:
            train_n -= 1
    while train_n + val_n + test_n < n:
        train_n += 1
    return train_n, val_n, test_n


def choose_batch_size(device: torch.device) -> int:
    try:
        import psutil
    except ImportError:
        psutil = None

    if device.type == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb >= 10:
                return 32
            if vram_gb >= 6:
                return 16
            return 8
        except Exception:
            return 8

    ram_gb = psutil.virtual_memory().total / (1024 ** 3) if psutil else 16.0
    print(f"CPU mode — system RAM ~{ram_gb:.1f} GB")
    if ram_gb >= 14:
        return 12
    if ram_gb >= 8:
        return 8
    return 4


def get_num_workers() -> int:
    if platform.system() == "Windows":
        return 0
    return min(2, os.cpu_count() or 1)


def build_soft_train_transform() -> A.Compose:
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.4),
        A.Rotate(limit=8, p=0.35),
        A.Affine(
            translate_percent={"x": 0.04, "y": 0.04},
            scale=(0.95, 1.05),
            rotate=0,
            shear=0,
            p=0.35,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=6,
            sat_shift_limit=12,
            val_shift_limit=12,
            p=0.35,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.25),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.25),
        A.Resize(IMG_SIZE, IMG_SIZE),
        NORMALIZE,
        TO_TENSOR,
    ])


def build_normal_train_transform() -> A.Compose:
    return A.Compose([
        A.OneOf([
            A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            A.Resize(IMG_SIZE, IMG_SIZE),
        ], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.5),
        A.GaussNoise(std_range=(0.04, 0.12), p=0.45),
        A.GaussianBlur(blur_limit=(3, 7), p=0.35),
        A.MotionBlur(blur_limit=7, p=0.25),
        A.MedianBlur(blur_limit=5, p=0.15),
        A.RandomShadow(p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(16, 36),
            hole_width_range=(16, 36),
            fill=0,
            p=0.25,
        ),
        A.Resize(IMG_SIZE, IMG_SIZE),
        NORMALIZE,
        TO_TENSOR,
    ])


def build_eval_transform() -> A.Compose:
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        NORMALIZE,
        TO_TENSOR,
    ])


class ProduceDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int, str]],
        transform_soft: A.Compose | None,
        transform_normal: A.Compose | None,
        eval_transform: A.Compose | None,
        training: bool,
    ):
        self.samples = samples
        self.transform_soft = transform_soft
        self.transform_normal = transform_normal
        self.eval_transform = eval_transform
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target, class_name = self.samples[index]
        image = np.array(Image.open(path).convert("RGB"))
        if self.training:
            if is_soft_class(class_name):
                image = self.transform_soft(image=image)["image"]
            else:
                image = self.transform_normal(image=image)["image"]
        else:
            image = self.eval_transform(image=image)["image"]
        return image, target


def collect_samples(dataset_dir: Path) -> dict[str, list[Path]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = list_images(class_dir)
        if files:
            by_class[class_dir.name] = files
    if not by_class:
        raise RuntimeError(f"No images found under {dataset_dir}")
    return by_class


def stratified_split_samples(
    by_class: dict[str, list[Path]],
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
) -> tuple[list, list, list, list[str], dict[str, int]]:
    classes = sorted(by_class.keys())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_s, val_s, test_s = [], [], []
    rng = random.Random(seed)

    for class_name in classes:
        paths = by_class[class_name][:]
        rng.shuffle(paths)
        n = len(paths)
        train_n, val_n, test_n = split_counts(n, train_r, val_r, test_r)
        train_paths = paths[:train_n]
        val_paths = paths[train_n:train_n + val_n]
        test_paths = paths[train_n + val_n:]
        idx = class_to_idx[class_name]
        for p in train_paths:
            train_s.append((str(p), idx, class_name))
        for p in val_paths:
            val_s.append((str(p), idx, class_name))
        for p in test_paths:
            test_s.append((str(p), idx, class_name))

    return train_s, val_s, test_s, classes, class_to_idx


def write_split_folders(split_root: Path, train_s, val_s, test_s, clear: bool = True) -> None:
    splits = {"train": train_s, "val": val_s, "test": test_s}
    for split_name, samples in splits.items():
        split_dir = split_root / split_name
        if clear and split_dir.is_dir():
            shutil.rmtree(split_dir)
        for path_str, _, class_name in samples:
            src = Path(path_str)
            dst_dir = split_dir / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            if dst.exists():
                dst = dst_dir / f"{src.stem}_dup{src.suffix}"
            shutil.copy2(src, dst)


def build_split_summary(classes, train_s, val_s, test_s) -> pd.DataFrame:
    rows = []
    for c in classes:
        tr = sum(1 for _, _, cn in train_s if cn == c)
        va = sum(1 for _, _, cn in val_s if cn == c)
        te = sum(1 for _, _, cn in test_s if cn == c)
        total = tr + va + te
        rows.append({
            "class": c,
            "train": tr,
            "val": va,
            "test": te,
            "total": total,
            "train_pct": round(tr / total, 4) if total else 0,
            "val_pct": round(va / total, 4) if total else 0,
            "test_pct": round(te / total, 4) if total else 0,
            "augmentation": "soft" if is_soft_class(c) else "normal",
        })
    df = pd.DataFrame(rows)
    total_all = df["total"].sum()
    df = pd.concat([
        df,
        pd.DataFrame([{
            "class": "__TOTAL__",
            "train": df["train"].sum(),
            "val": df["val"].sum(),
            "test": df["test"].sum(),
            "total": total_all,
            "train_pct": round(df["train"].sum() / total_all, 4),
            "val_pct": round(df["val"].sum() / total_all, 4),
            "test_pct": round(df["test"].sum() / total_all, 4),
            "augmentation": "",
        }]),
    ], ignore_index=True)
    return df


def run_epoch(model, loader, criterion, device, optimizer, training, desc):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(1) == labels).sum().item()
        total += batch_size
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total, 1), correct / max(total, 1)


def collect_predictions(model, loader, dataset, device):
    model.eval()
    y_true, y_pred, y_prob, paths = [], [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predicting", leave=False, ncols=100):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            labels_np = labels.numpy()
            batch_start = len(y_true)

            for i in range(len(labels_np)):
                y_true.append(labels_np[i])
                y_pred.append(preds[i])
                y_prob.append(probs[i])
                paths.append(dataset.samples[batch_start + i][0])

    return np.array(y_true), np.array(y_pred), np.array(y_prob), paths


def optimal_thresholds_from_val(y_true, y_prob, classes):
    thresholds = {}
    for i, cname in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        scores = y_prob[:, i]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            thresholds[cname] = 0.5
            continue
        try:
            prec, rec, thr = precision_recall_curve(y_bin, scores)
            if len(thr) == 0:
                thresholds[cname] = 0.5
                continue
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
            best = int(np.argmax(f1))
            thresholds[cname] = float(round(thr[best], 4))
        except Exception:
            thresholds[cname] = 0.5
    return thresholds


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_model_summary(model, path):
    lines = [
        "EfficientNet-B3 produce classifier",
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}",
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        "",
        str(model),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_onnx(model, device, out_path):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def main() -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Produce classification — EfficientNet-B3")
    print("=" * 60)
    print(f"Dataset:  {DATASET_DIR}")
    print(f"Splits:   {SPLIT_ROOT / 'train'}, val, test")
    print(f"Results:  {RESULTS_DIR}")
    print(f"Epochs:   {EPOCHS}  |  Patience: {PATIENCE}")
    print(f"Split:    {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%} (train/val/test)")
    print()

    print("Scanning class folders and building stratified split...")
    by_class = collect_samples(DATASET_DIR)
    train_s, val_s, test_s, classes, class_to_idx = stratified_split_samples(
        by_class, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
    )
    num_classes = len(classes)

    print(f"Classes ({num_classes}): {classes}")
    print(f"Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    if len(train_s) == 0:
        raise RuntimeError("Train split is empty — check DATASET_DIR.")

    split_df = build_split_summary(classes, train_s, val_s, test_s)
    split_df.to_csv(RESULTS_DIR / "train_val_test_split_summary.csv", index=False)

    print("Writing train/val/test folders (originals only)...")
    write_split_folders(SPLIT_ROOT, train_s, val_s, test_s, clear=True)

    soft_tf = build_soft_train_transform()
    normal_tf = build_normal_train_transform()
    eval_tf = build_eval_transform()

    train_ds = ProduceDataset(train_s, soft_tf, normal_tf, eval_tf, training=True)
    val_ds = ProduceDataset(val_s, None, None, eval_tf, training=False)
    test_ds = ProduceDataset(test_s, None, None, eval_tf, training=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    batch_size = choose_batch_size(device)
    num_workers = get_num_workers()
    pin_memory = device.type == "cuda"
    print(f"Batch size: {batch_size}  |  Workers: {num_workers}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    print("\nLoading EfficientNet-B3 (timm, pretrained)...")
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)
    model = model.to(device)
    save_model_summary(model, RESULTS_DIR / "model_summary.txt")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_count = 0
    history_rows = []

    soft_list = sorted({c for c in classes if is_soft_class(c)})
    print("\n" + "=" * 60)
    print("Training (augmentation on-the-fly, train only)")
    print("SOFT aug classes:", soft_list)
    print("NORMAL aug: all others | val/test: no aug")
    print("=" * 60)

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Epochs", ncols=110)

    for epoch in epoch_bar:
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, True,
            f"Epoch {epoch} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, None, False,
            f"Epoch {epoch} val",
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start

        history_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
            "lr": current_lr,
            "seconds": round(elapsed, 2),
        })

        epoch_bar.set_postfix({
            "tr_loss": f"{train_loss:.4f}",
            "va_loss": f"{val_loss:.4f}",
            "tr_acc": f"{train_acc:.4f}",
            "va_acc": f"{val_acc:.4f}",
            "lr": f"{current_lr:.2e}",
        })

        tqdm.write(
            f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {current_lr:.2e} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, RESULTS_DIR / "best_model.pth")
            patience_count = 0
            tqdm.write("  -> best_model.pth saved")
        else:
            patience_count += 1
            tqdm.write(f"  -> patience {patience_count}/{PATIENCE}")

        if patience_count >= PATIENCE:
            tqdm.write("Early stopping.")
            break

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(RESULTS_DIR / "training_history.csv", index=False)
    plot_training_curves(history_df, RESULTS_DIR / "training_curves.png")

    torch.save(model.state_dict(), RESULTS_DIR / "final_model.pth")
    model.load_state_dict(best_weights)

    with open(RESULTS_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    with open(RESULTS_DIR / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print("\nPer-class thresholds from validation set...")
    y_val_true, y_val_pred, y_val_prob, _ = collect_predictions(
        model, val_loader, val_ds, device
    )
    val_acc = accuracy_score(y_val_true, y_val_pred)
    print(f"Validation accuracy (best model): {val_acc:.4f}")

    class_thresholds = optimal_thresholds_from_val(y_val_true, y_val_prob, classes)
    with open(RESULTS_DIR / "class_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(class_thresholds, f, indent=2)

    print("\nTest set evaluation...")
    y_true, y_pred, y_prob, test_paths = collect_predictions(
        model, test_loader, test_ds, device
    )

    test_acc = accuracy_score(y_true, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test precision: {prec_w:.4f}")
    print(f"Test recall:    {rec_w:.4f}")
    print(f"Test F1:        {f1_w:.4f}")

    report_dict = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=classes, zero_division=0
    )
    (RESULTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(RESULTS_DIR / "classification_report.csv")

    per_class_prec, per_class_rec, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    pd.DataFrame({
        "class": classes,
        "precision": per_class_prec,
        "recall": per_class_rec,
        "f1_score": per_class_f1,
        "support": support,
        "threshold": [class_thresholds[c] for c in classes],
        "augmentation": ["soft" if is_soft_class(c) else "normal" for c in classes],
    }).to_csv(RESULTS_DIR / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(RESULTS_DIR / "confusion_matrix.csv")
    plot_confusion_matrix(cm, classes, RESULTS_DIR / "confusion_matrix.png")

    pred_rows = []
    for i, path in enumerate(test_paths):
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        true_name = classes[true_idx]
        pred_name = classes[pred_idx]
        confidence = float(y_prob[i, pred_idx])
        threshold = class_thresholds[pred_name]
        pred_rows.append({
            "image_path": path,
            "true_class": true_name,
            "predicted_class": pred_name,
            "confidence": round(confidence, 6),
            "threshold": threshold,
            "accepted": confidence >= threshold,
            "correct": true_name == pred_name,
        })
    pd.DataFrame(pred_rows).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    print("\nExporting ONNX...")
    export_onnx(model, device, RESULTS_DIR / "efficientnet_b3.onnx")

    (RESULTS_DIR / "requirements.txt").write_text(REQUIREMENTS.strip() + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print("Done. Outputs in:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
