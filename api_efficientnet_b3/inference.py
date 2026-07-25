"""ONNX EfficientNet-B3 inference (no Flask/camera deps — Cloud Run safe)."""

import csv
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_ONNX_PATHS = []
if os.environ.get("ONNX_PATH"):
    CANDIDATE_ONNX_PATHS.append(os.environ.get("ONNX_PATH"))
CANDIDATE_ONNX_PATHS.extend(
    [
        os.path.join(BASE, "efficientnet_b3.onnx"),
        os.path.join(BASE, "..", "results_new", "efficientnet_b3.onnx"),
    ]
)
ONNX_PATH = None
for candidate in CANDIDATE_ONNX_PATHS:
    if candidate and os.path.isfile(candidate):
        ONNX_PATH = candidate
        break
if ONNX_PATH is None:
    ONNX_PATH = CANDIDATE_ONNX_PATHS[0] or os.path.join(BASE, "efficientnet_b3.onnx")

THRESHOLD_PATH = os.environ.get(
    "THRESHOLD_PATH", os.path.join(BASE, "class_thresholds.csv")
)

classes: list[str] = []
thresholds: dict[str, float] = {}
session = None
input_name = None


def _load_thresholds() -> None:
    """Load class names / thresholds (small CSV — safe at import or first use)."""
    global classes, thresholds
    if classes:
        return
    try:
        with open(THRESHOLD_PATH) as f:
            loaded_classes: list[str] = []
            loaded_thresholds: dict[str, float] = {}
            for row in csv.DictReader(f):
                loaded_classes.append(row["class"])
                loaded_thresholds[row["class"]] = float(row["threshold"])
        classes = loaded_classes
        thresholds = loaded_thresholds
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Thresholds file not found at {THRESHOLD_PATH}. "
            "Make sure class_thresholds.csv is copied in the container."
        ) from exc


def _load_model() -> None:
    """Lazy-load ONNX session on first inference (avoids Cloud Run startup timeouts)."""
    global session, input_name
    _load_thresholds()
    if session is not None:
        return

    if not ONNX_PATH or not os.path.isfile(ONNX_PATH):
        raise RuntimeError(
            f"ONNX model not found at {ONNX_PATH}. "
            "Cloud Build must copy efficientnet_b3.onnx into the image."
        )

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            ONNX_PATH,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model from {ONNX_PATH}: {e}") from e


# Thresholds are tiny — load now so /produce-list works without forcing ONNX load.
try:
    _load_thresholds()
except Exception:
    # Allow import when CSV is missing locally; first request will surface the error.
    pass

IMG_SIZE = 300

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _top3_from_probs(probs: np.ndarray) -> list[dict]:
    top3_idx = probs.argsort()[::-1][:3]
    return [
        {
            "label": classes[i],
            "prob": float(probs[i]),
            "threshold": thresholds[classes[i]],
        }
        for i in top3_idx
    ]


def classify(frame_bgr: np.ndarray) -> list[dict]:
    _load_model()
    inp = preprocess(frame_bgr)
    logits = session.run(None, {input_name: inp})[0][0]
    return _top3_from_probs(softmax(logits))


def classify_many(frames_bgr: list[np.ndarray]) -> list[dict]:
    """Run inference on one or more images; average logits when multiple are sent."""
    _load_model()
    if not frames_bgr:
        raise ValueError("At least one image is required")
    if len(frames_bgr) == 1:
        return classify(frames_bgr[0])

    logits_list = []
    for frame_bgr in frames_bgr:
        inp = preprocess(frame_bgr)
        logits_list.append(session.run(None, {input_name: inp})[0][0])
    avg_logits = np.mean(logits_list, axis=0)
    return _top3_from_probs(softmax(avg_logits))
