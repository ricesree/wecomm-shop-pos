"""
evaluate.py  —  Full model evaluation report
Generates: confusion matrix, per-class bar charts, confidence histogram,
           missed detection chart, and a text summary report.

Usage:
  python evaluate.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from ultralytics import YOLO

MODEL_PATH  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MODEL_MANUAL\yolov8m-prod\weights\best.pt"
TEST_DIR    = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET - Copy"
RESULTS_DIR = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\RESULTS"
CONF        = 0.25   # lower threshold to catch weak detections too
MAX_IMGS    = 20     # images per class to evaluate

FIXED_NAMES = {
    0:  "Banana",                1:  "Banana Flower",
    2:  "Beetroot",              3:  "Bell pepper",
    4:  "Boxed Sweets",          5:  "Cabbage",
    6:  "Cauliflower",           7:  "Chayote",
    8:  "Chinese Green Eggplant",9:  "Chinese eggplant",
    10: "Cilantro",              11: "Coconut",
    12: "Curry leaves",          13: "Dasakai",
    14: "Garlic",                15: "Ginger",
    16: "Guava",                 17: "Home made snacks",
    18: "Indian eggplant",       19: "Karela",
    20: "Lady stickers",         21: "Leaves",
    22: "Lemon",                 23: "Long green beans",
    24: "Mint",                  25: "Muli",
    26: "Mums",                  27: "Okra",
    28: "Pan Leaves",            29: "Papaya",
    30: "Pearl",                 31: "Potato",
    32: "Pumpkin",               33: "Red Onions",
    34: "Roti",                  35: "Snake Guard",
    36: "Squah",                 37: "String beans",
    38: "Sweet Potato",          39: "Thai Chilli",
    40: "Tindora",               41: "Tomato",
    42: "Turai",                 43: "White Onions",
}


def short(name, maxlen=12):
    return name[:maxlen] if len(name) > maxlen else name


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model...")
    model = YOLO(MODEL_PATH)
    model.model.names = FIXED_NAMES

    classes = sorted([
        d for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    ])
    n = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # ── collect predictions ────────────────────────────────────────────────────
    # per class: list of (predicted_name, confidence)
    results_per_class = defaultdict(list)
    conf_matrix = np.zeros((n, n), dtype=int)  # [true][pred]
    missed = {}   # class -> count of images with zero detection

    print(f"\nRunning inference on up to {MAX_IMGS} images per class...\n")
    for cls in classes:
        cls_dir = os.path.join(TEST_DIR, cls)
        imgs = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.startswith("aug_")   # test on originals only
        ][:MAX_IMGS]

        true_idx = cls_to_idx[cls]
        miss_count = 0

        for fname in imgs:
            path = os.path.join(cls_dir, fname)
            res  = model(path, conf=CONF, verbose=False)[0]

            if len(res.boxes) == 0:
                miss_count += 1
                results_per_class[cls].append(("NO DETECTION", 0.0))
                continue

            # pick highest-confidence detection
            best = max(res.boxes, key=lambda b: float(b.conf[0]))
            pred_id   = int(best.cls[0])
            pred_name = FIXED_NAMES.get(pred_id, f"class_{pred_id}")
            pred_conf = float(best.conf[0])

            results_per_class[cls].append((pred_name, pred_conf))

            pred_idx = cls_to_idx.get(pred_name, -1)
            if pred_idx >= 0:
                conf_matrix[true_idx][pred_idx] += 1
            # wrong class or unknown → still counts as miss in matrix
            # (already not in the correct cell)

        missed[cls] = miss_count
        n_tested = len(imgs)
        n_correct = sum(1 for p, _ in results_per_class[cls] if p == cls)
        pct = 100 * n_correct / n_tested if n_tested else 0
        print(f"  {cls:<30}  {n_correct:>2}/{n_tested}  ({pct:.0f}%)")

    # ── per-class accuracy + avg confidence ────────────────────────────────────
    accuracies   = []
    avg_confs    = []
    miss_pcts    = []
    wrong_labels = {}   # cls -> most common wrong prediction

    for cls in classes:
        preds = results_per_class[cls]
        if not preds:
            accuracies.append(0); avg_confs.append(0); miss_pcts.append(0)
            continue
        correct = [p for p, _ in preds if p == cls]
        confs   = [c for p, c in preds if p != "NO DETECTION"]
        accuracies.append(100 * len(correct) / len(preds))
        avg_confs.append(100 * np.mean(confs) if confs else 0)
        miss_pcts.append(100 * missed[cls] / len(preds))

        # most common wrong prediction
        wrong = [p for p, _ in preds if p not in (cls, "NO DETECTION")]
        if wrong:
            from collections import Counter
            wrong_labels[cls] = Counter(wrong).most_common(1)[0][0]

    # ── PLOT 1: Confusion matrix (top 20 most confused classes) ───────────────
    print("\nGenerating plots...")

    # normalize rows
    row_sums = conf_matrix.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = conf_matrix / row_sums

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ticks = range(n)
    labels = [short(c) for c in classes]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, pad=12)
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if v > 0.6 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── PLOT 2: Per-class accuracy bar chart ──────────────────────────────────
    colors = ["#2ecc71" if a >= 80 else "#f39c12" if a >= 50 else "#e74c3c"
              for a in accuracies]
    fig, ax = plt.subplots(figsize=(18, 7))
    bars = ax.bar(range(n), accuracies, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([short(c, 11) for c in classes], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Detection Accuracy", fontsize=13)
    ax.set_ylim(0, 110)
    ax.axhline(80, color="green",  linestyle="--", linewidth=1, alpha=0.6, label="80% target")
    ax.axhline(50, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="50% line")
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{acc:.0f}", ha="center", va="bottom", fontsize=7)
    good  = mpatches.Patch(color="#2ecc71", label=">=80%  Good")
    ok    = mpatches.Patch(color="#f39c12", label="50-79%  OK")
    weak  = mpatches.Patch(color="#e74c3c", label="<50%   Weak")
    ax.legend(handles=[good, ok, weak], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_per_class.png"), dpi=150)
    plt.close()

    # ── PLOT 3: Average confidence bar chart ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(range(n), avg_confs, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([short(c, 11) for c in classes], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Avg Confidence (%)", fontsize=11)
    ax.set_title("Average Detection Confidence Per Class", fontsize=13)
    ax.set_ylim(0, 110)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.6, label="50% conf")
    for i, v in enumerate(avg_confs):
        ax.text(i, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_per_class.png"), dpi=150)
    plt.close()

    # ── PLOT 4: Missed detections ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(range(n), miss_pcts, color="#e74c3c", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([short(c, 11) for c in classes], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Missed (%)", fontsize=11)
    ax.set_title("Missed Detections Per Class (no box drawn at all)", fontsize=13)
    ax.set_ylim(0, 110)
    for i, v in enumerate(miss_pcts):
        if v > 0:
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "missed_detections.png"), dpi=150)
    plt.close()

    # ── PLOT 5: Overall confidence histogram ──────────────────────────────────
    all_confs = [c for cls in classes
                 for p, c in results_per_class[cls]
                 if p != "NO DETECTION"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_confs, bins=20, range=(0, 1), color="#9b59b6",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Confidence Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Detection Confidence Scores", fontsize=13)
    ax.axvline(0.25, color="red",   linestyle="--", label="threshold=0.25")
    ax.axvline(0.50, color="orange",linestyle="--", label="0.50")
    ax.axvline(0.75, color="green", linestyle="--", label="0.75")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confidence_histogram.png"), dpi=150)
    plt.close()

    # ── TEXT REPORT ───────────────────────────────────────────────────────────
    overall_acc = np.mean(accuracies)
    overall_conf = np.mean([c for c in avg_confs if c > 0])

    report_lines = [
        "=" * 72,
        "  VEGETABLE DETECTION MODEL — EVALUATION REPORT",
        f"  Model: {MODEL_PATH}",
        "=" * 72,
        f"\n  Overall accuracy  : {overall_acc:.1f}%",
        f"  Overall avg conf  : {overall_conf:.1f}%",
        f"  Classes evaluated : {n}",
        f"  Images per class  : up to {MAX_IMGS} (originals only)",
        f"  Conf threshold    : {CONF}",
        "",
        f"{'Class':<30} {'Acc':>6} {'AvgConf':>8} {'Missed':>7}  {'Top Wrong Prediction'}",
        "-" * 72,
    ]

    grades = []
    for i, cls in enumerate(classes):
        a  = accuracies[i]
        c  = avg_confs[i]
        m  = miss_pcts[i]
        g  = "EXCELLENT" if a >= 90 else "GOOD" if a >= 75 else "OK" if a >= 50 else "!! WEAK"
        w  = wrong_labels.get(cls, "-")
        report_lines.append(
            f"  {cls:<28} {a:>5.0f}%  {c:>6.0f}%  {m:>5.0f}%   {w:<20}  {g}"
        )
        grades.append(g)

    excellent = grades.count("EXCELLENT")
    good      = grades.count("GOOD")
    ok        = grades.count("OK")
    weak      = grades.count("!! WEAK")

    report_lines += [
        "-" * 72,
        f"\n  EXCELLENT (>=90%) : {excellent} classes",
        f"  GOOD      (75-89%): {good} classes",
        f"  OK        (50-74%): {ok} classes",
        f"  WEAK      (<50%)  : {weak} classes",
        "",
        "  NOTE: This model was trained on epoch-41 checkpoint (training",
        "  was interrupted). Retrain with clean Boxed Sweets data for",
        "  better Boxed Sweets accuracy.",
        "",
        "  Boxed Sweets vs Curry Leaves confusion: see confusion_matrix.png",
        "=" * 72,
    ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(os.path.join(RESULTS_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("  confusion_matrix.png")
    print("  accuracy_per_class.png")
    print("  confidence_per_class.png")
    print("  missed_detections.png")
    print("  confidence_histogram.png")
    print("  report.txt")


if __name__ == "__main__":
    main()
