"""ONNX EfficientNet-B3 inference (no Flask/camera deps — Cloud Run safe)."""

import csv
import os

import cv2
import numpy as np
import onnxruntime as ort

BASE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.environ.get("ONNX_PATH", os.path.join(BASE, "efficientnet_b3.onnx"))
THRESHOLD_PATH = os.environ.get(
    "THRESHOLD_PATH", os.path.join(BASE, "class_thresholds.csv")
)

classes: list[str] = []
thresholds: dict[str, float] = {}
with open(THRESHOLD_PATH) as f:
    for row in csv.DictReader(f):
        classes.append(row["class"])
        thresholds[row["class"]] = float(row["threshold"])

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    ONNX_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
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


def classify(frame_bgr: np.ndarray) -> list[dict]:
    inp = preprocess(frame_bgr)
    logits = session.run(None, {input_name: inp})[0][0]
    probs = softmax(logits)
    top3_idx = probs.argsort()[::-1][:3]
    return [
        {
            "label": classes[i],
            "prob": float(probs[i]),
            "threshold": thresholds[classes[i]],
        }
        for i in top3_idx
    ]
