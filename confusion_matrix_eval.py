"""
Run a small YOLO validation confusion test.

Use after training, for example in Colab:
    python confusion_matrix_eval.py \
      --model /content/drive/MyDrive/TUNE-DATAPOS/MODEL_CATEGORIES/checkpoints/best_new.pt \
      --data-dir /content/YOLO_CATEGORIES \
      --out-dir /content/confusion_eval

Outputs:
    confusion_matrix.csv
    confusion_report.json
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "banana", "beans", "chilli", "coconut", "dasakai",
    "eggplant", "fruit", "gourd", "ladyfinger", "leafy",
    "onion", "root", "special", "tomato",
]


CATEGORY_LOOKALIKE_MAP = {
    "chilli": ["gourd"],
    "gourd": ["fruit", "chilli"],
    "fruit": ["gourd"],
    "root": ["fruit"],
    "leafy": ["gourd"],
    "eggplant": ["gourd"],
    "beans": ["chilli", "gourd"],
    "onion": ["root"],
}


def yolo_to_xyxy(label, width, height):
    cls_id, cx, cy, bw, bh = label
    x1 = (cx - bw / 2) * width
    y1 = (cy - bh / 2) * height
    x2 = (cx + bw / 2) * width
    y2 = (cy + bh / 2) * height
    return int(cls_id), np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def load_labels(path, width, height):
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append(yolo_to_xyxy(tuple(map(float, parts[:5])), width, height))
    return labels


def evaluate(model_path, data_dir, out_dir, conf, iou):
    from PIL import Image
    from ultralytics import YOLO

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_dir = data_dir / "images" / "val"
    label_dir = data_dir / "labels" / "val"
    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise SystemExit(f"No validation images found in {image_dir}")

    model = YOLO(model_path)
    n = len(CLASS_NAMES)
    missed_idx = n
    extra_idx = n + 1
    matrix = np.zeros((n + 2, n + 2), dtype=int)

    for image_path in image_paths:
        with Image.open(image_path) as im:
            width, height = im.size
        gt = load_labels(label_dir / f"{image_path.stem}.txt", width, height)

        result = model.predict(str(image_path), conf=conf, imgsz=640, verbose=False)[0]
        preds = []
        if len(result.boxes):
            for cls_id, box in zip(result.boxes.cls.cpu().numpy().astype(int), result.boxes.xyxy.cpu().numpy()):
                if 0 <= cls_id < n:
                    preds.append((int(cls_id), box.astype(np.float32)))

        matched_preds = set()
        for true_cls, true_box in gt:
            best_iou = 0.0
            best_pred = None
            for pred_i, (pred_cls, pred_box) in enumerate(preds):
                if pred_i in matched_preds:
                    continue
                score = box_iou(true_box, pred_box)
                if score > best_iou:
                    best_iou = score
                    best_pred = pred_i
            if best_pred is None or best_iou < iou:
                matrix[true_cls, missed_idx] += 1
            else:
                matched_preds.add(best_pred)
                matrix[true_cls, preds[best_pred][0]] += 1

        for pred_i, (pred_cls, _) in enumerate(preds):
            if pred_i not in matched_preds:
                matrix[extra_idx, pred_cls] += 1

    labels = CLASS_NAMES + ["missed", "extra_prediction"]
    csv_path = out_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])

    pairs = []
    for true_i, true_name in enumerate(CLASS_NAMES):
        total = int(matrix[true_i, :].sum())
        correct = int(matrix[true_i, true_i])
        wrong = []
        for pred_i, count in enumerate(matrix[true_i, :n]):
            if pred_i != true_i and count:
                wrong.append({
                    "predicted": CLASS_NAMES[pred_i],
                    "count": int(count),
                    "rate": round(count / total, 4) if total else 0,
                })
        wrong.sort(key=lambda x: x["count"], reverse=True)
        pairs.append({
            "true": true_name,
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0,
            "missed": int(matrix[true_i, missed_idx]),
            "top_confusions": wrong[:5],
            "expected_lookalikes": CATEGORY_LOOKALIKE_MAP.get(true_name, []),
        })

    report = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "images": len(image_paths),
        "conf": conf,
        "iou": iou,
        "classes": CLASS_NAMES,
        "per_class": pairs,
    }
    json_path = out_dir / "confusion_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    print(f"Images: {len(image_paths)}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print("\nTop measured confusions:")
    for item in pairs:
        if not item["top_confusions"]:
            continue
        top = ", ".join(f"{x['predicted']}={x['count']}" for x in item["top_confusions"][:3])
        print(f"  {item['true']:<12} -> {top}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--data-dir", required=True, help="Path to extracted YOLO_CATEGORIES")
    parser.add_argument("--out-dir", default="confusion_eval")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.50)
    args = parser.parse_args()
    evaluate(args.model, args.data_dir, args.out_dir, args.conf, args.iou)


if __name__ == "__main__":
    main()
