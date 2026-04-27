"""
augment.py  —  Offline Data Augmentation
=========================================
Reads every labeled image from DATASET + MANUAL_LABELS,
applies augmentations, and saves new images + corrected labels
back into the same folders so retrain.py picks them all up.

Augmentations applied:
  1. Rotation        ±15 degrees   (labels transformed correctly)
  2. Brightness      decrease / increase
  3. Contrast        lower / higher
  4. Horizontal flip               (labels mirrored correctly)
  5. Gaussian blur   small / medium
  6. Salt & pepper noise
  7. HSV shift       hue + saturation + value

Weak classes get more copies:
  - Okra, Ginger, Chinese eggplant  → 6 augmented copies per image
  - All other classes                → 3 augmented copies per image

Usage:
  python augment.py
"""

import cv2
import numpy as np
import os
import random

DATASET_DIR = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET - Copy"
LABELS_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MANUAL_LABELS"

WEAK_CLASSES   = {"Okra", "Ginger", "Chinese eggplant"}
NORMAL_COPIES  = 3
WEAK_COPIES    = 6

random.seed(42)
np.random.seed(42)


# ── label coordinate helpers ──────────────────────────────────────────────────

def yolo_to_corners(cx, cy, w, h, W, H):
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return x1, y1, x2, y2


def corners_to_yolo(x1, y1, x2, y2, W, H):
    x1 = max(0.0, min(float(W), x1))
    y1 = max(0.0, min(float(H), y1))
    x2 = max(0.0, min(float(W), x2))
    y2 = max(0.0, min(float(H), y2))
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    if bw < 0.01 or bh < 0.01:   # box too small after transform — discard
        return None
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    return cx, cy, bw, bh


def read_labels(path):
    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append((int(parts[0]),
                                float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4])))
    return labels


def write_labels(path, labels):
    with open(path, "w") as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ── augmentation functions ────────────────────────────────────────────────────

def aug_rotate(img, labels):
    """Rotate image and transform bounding boxes using same matrix."""
    H, W = img.shape[:2]
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    aug = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    new_labels = []
    for cls_id, cx, cy, w, h in labels:
        x1, y1, x2, y2 = yolo_to_corners(cx, cy, w, h, W, H)
        corners = np.array([[x1, y1], [x2, y1],
                             [x2, y2], [x1, y2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        rotated = (M @ np.hstack([corners, ones]).T).T
        result = corners_to_yolo(rotated[:, 0].min(), rotated[:, 1].min(),
                                  rotated[:, 0].max(), rotated[:, 1].max(),
                                  W, H)
        if result:
            new_labels.append((cls_id, *result))
    return aug, new_labels


def aug_brightness(img, labels):
    """Random brightness — darker or brighter."""
    factor = random.choice([
        random.uniform(0.35, 0.65),   # darker
        random.uniform(1.3,  1.7),    # brighter
    ])
    aug = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return aug, labels


def aug_contrast(img, labels):
    """Random contrast stretch/compress."""
    alpha = random.uniform(0.5, 1.9)
    beta  = random.randint(-40, 40)
    aug   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return aug, labels


def aug_flip_h(img, labels):
    """Horizontal mirror — flip cx coordinate."""
    aug = cv2.flip(img, 1)
    new_labels = [(cls_id, 1.0 - cx, cy, w, h)
                  for cls_id, cx, cy, w, h in labels]
    return aug, new_labels


def aug_blur(img, labels):
    """Gaussian blur — simulates out-of-focus camera."""
    k   = random.choice([3, 5, 7])
    aug = cv2.GaussianBlur(img, (k, k), 0)
    return aug, labels


def aug_noise(img, labels):
    """Salt & pepper noise — simulates low-quality camera."""
    noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
    aug   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return aug, labels


def aug_hsv(img, labels):
    """Shift hue, saturation, and brightness in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-12, 12)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.5, 1.5), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.4, 1.6), 0, 255)
    aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return aug, labels


# Augmentation pipelines — each is a list of 1 or 2 functions to chain
PIPELINES = [
    [aug_rotate,     aug_brightness],   # rotated + darker/brighter
    [aug_flip_h,     aug_contrast],     # mirrored + contrast change
    [aug_blur,       aug_hsv],          # blurry + colour shift
    [aug_noise,      aug_brightness],   # noisy + brightness
    [aug_rotate,     aug_flip_h],       # rotated + mirrored
    [aug_hsv,        aug_blur],         # colour shift + blur
    [aug_contrast,   aug_noise],        # contrast + noise  (weak class extra)
]


def apply_pipeline(img, labels, idx):
    """Run pipeline[idx % len(PIPELINES)]."""
    fns = PIPELINES[idx % len(PIPELINES)]
    for fn in fns:
        img, labels = fn(img, list(labels))
        if not labels:
            return None, None
    return img, labels


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    classes = sorted([
        d for d in os.listdir(LABELS_DIR)
        if os.path.isdir(os.path.join(LABELS_DIR, d))
    ])

    print(f"Found {len(classes)} classes.\n")
    grand_total = 0

    for cls in classes:
        lbl_dir  = os.path.join(LABELS_DIR, cls)
        img_dir  = os.path.join(DATASET_DIR, cls)
        n_copies = WEAK_COPIES if cls in WEAK_CLASSES else NORMAL_COPIES

        # Only process ORIGINAL files — skip anything already augmented
        txts = [
            f for f in os.listdir(lbl_dir)
            if f.endswith(".txt") and not f.startswith("aug_")
        ]

        cls_new = 0
        for txt in txts:
            stem       = os.path.splitext(txt)[0]
            label_path = os.path.join(lbl_dir, txt)

            img_path = img_ext = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(img_dir, stem + ext)
                if os.path.exists(candidate):
                    img_path, img_ext = candidate, ext
                    break
            if img_path is None:
                continue

            img    = cv2.imread(img_path)
            labels = read_labels(label_path)
            if img is None or not labels:
                continue

            # skip if all augmented copies already exist
            all_exist = all(
                os.path.exists(os.path.join(img_dir, f"aug_{i}_{stem}{img_ext}"))
                for i in range(n_copies)
            )
            if all_exist:
                continue

            for i in range(n_copies):
                out_name = f"aug_{i}_{stem}"
                if os.path.exists(os.path.join(img_dir, out_name + img_ext)):
                    continue   # this copy already done

                aug_img, aug_labels = apply_pipeline(img.copy(), labels, i)
                if aug_img is None or not aug_labels:
                    continue

                cv2.imwrite(os.path.join(img_dir, out_name + img_ext), aug_img)
                write_labels(os.path.join(lbl_dir, out_name + ".txt"), aug_labels)
                cls_new += 1

        tag = "  << WEAK CLASS" if cls in WEAK_CLASSES else ""
        print(f"  {cls:<30} +{cls_new:>4} images{tag}")
        grand_total += cls_new

    print(f"\n{'-'*50}")
    print(f"Done!  Added {grand_total} augmented images total.")
    print(f"Original labeled images were kept untouched.")
    print(f"\nNext step: upload DATASET + MANUAL_LABELS to Colab")
    print(f"then run:  python retrain.py")


if __name__ == "__main__":
    main()
