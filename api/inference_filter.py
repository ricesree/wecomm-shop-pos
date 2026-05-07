"""
Per-class confidence threshold filter for production inference.

Why this reduces false positives:
- Hard classes (ladyfinger, beans) get confused with background at low
  confidence. Raising their threshold filters these uncertain predictions.
- Easy classes (banana, tomato) stay at a lower threshold to maintain
  high recall — they rarely produce false positives anyway.

Usage:
    from inference_filter import load_thresholds, apply_class_thresholds
    CLASS_THRESHOLDS = load_thresholds("/app/thresholds.json", fallback=0.40)
    detections = apply_class_thresholds(raw_detections, CLASS_THRESHOLDS)
"""

import json
import os

_DEFAULT_FALLBACK = 0.40


def load_thresholds(path: str, fallback: float = _DEFAULT_FALLBACK) -> dict:
    """
    Load per-class thresholds from JSON.
    Returns empty dict (triggering fallback in apply_class_thresholds)
    if the file is missing or malformed — never crashes.
    """
    if not os.path.exists(path):
        print(f"[thresholds] {path} not found — using global conf={fallback}")
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[thresholds] Loaded {len(data)} per-class thresholds from {path}")
        return {k: float(v) for k, v in data.items()}
    except Exception as e:
        print(f"[thresholds] Failed to load {path}: {e} — using global conf={fallback}")
        return {}


def apply_class_thresholds(
    detections: list,
    thresholds: dict,
    fallback: float = _DEFAULT_FALLBACK
) -> list:
    """
    Filter detections using per-class thresholds.

    detections : list of dicts, each with keys 'class' (str) and 'confidence' (float)
    thresholds : {class_name: threshold_float} from load_thresholds()
    fallback   : used when a class name is not present in thresholds

    A detection is kept when:
        confidence >= thresholds.get(class_name, fallback)
    """
    return [
        d for d in detections
        if d["confidence"] >= thresholds.get(d["class"], fallback)
    ]


def get_base_conf(thresholds: dict, fallback: float = _DEFAULT_FALLBACK) -> float:
    """
    Returns the minimum threshold across all classes.
    Pass this as conf= to model() so YOLO only suppresses detections
    that no class would ever keep anyway — saves NMS overhead.
    """
    if not thresholds:
        return fallback
    return float(min(thresholds.values()))
