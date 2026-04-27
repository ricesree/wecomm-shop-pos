"""
Vegetable Detection API
POST /detect           — send image, get detections
POST /feedback         — save correction image + correct label to GCS
GET  /feedback/stats   — see how many corrections saved per class
GET  /health           — health check
"""

import os
import io
import json
import time
import cv2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI(title="Swadesh Food Mart — Vegetable Detection API", version="2.0")
app.mount("/static", StaticFiles(directory="/app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("/app/static/index.html")

MODEL_PATH     = os.environ.get("MODEL_PATH",      "/app/best.pt")
CONF           = float(os.environ.get("CONF_THRESHOLD", "0.30"))
IMGSZ          = int(os.environ.get("IMGSZ",           "320"))
FEEDBACK_BUCKET = os.environ.get("FEEDBACK_BUCKET",   "")

FIXED_NAMES = {
    0:  "Beans Regular",         1:  "Burro Banana",
    2:  "Banana",                3:  "Banana Flower",
    4:  "Beetroot",              5:  "Bell pepper",
    6:  "Boxed Sweets",          7:  "Cabbage",
    8:  "Cauliflower",           9:  "Chayote",
    10: "Chinese Green Eggplant",11: "Chinese eggplant",
    12: "Cilantro",              13: "Coconut",
    14: "Curry leaves",          15: "Dasakai",
    16: "Flat Velor",            17: "Florida Long Chilli",
    18: "Fresh Chikku",          19: "Garlic",
    20: "Ginger",                21: "Guava",
    22: "Home made snacks",      23: "Indian eggplant",
    24: "Karela",                25: "Lady stickers",
    26: "Leaves",                27: "Lemon",
    28: "Long green beans",      29: "Mint",
    30: "Muli",                  31: "Mums",
    32: "Okra",                  33: "Poli",
    34: "Pan Leaves",            35: "Papaya",
    36: "Pearl",                 37: "Potato",
    38: "Pumpkin",               39: "Red Onions",
    40: "Roti",                  41: "Snake Guard",
    42: "Squah",                 43: "String beans",
    44: "Sweet Potato",          45: "Thai Egg Plant",
    46: "Thai Chilli",           47: "Tindora",
    48: "Tomato",                49: "Turai",
    50: "White Onions",
}

print(f"Loading model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
model.model.names = FIXED_NAMES
print("Model ready.")

# GCS client — only loaded if bucket is configured
_gcs_client = None
def get_gcs():
    global _gcs_client
    if _gcs_client is None and FEEDBACK_BUCKET:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "conf_threshold": CONF,
        "feedback_bucket": FEEDBACK_BUCKET or "not configured",
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    t0      = time.time()
    results = model(img, conf=CONF, verbose=False, imgsz=IMGSZ)[0]
    elapsed = round(time.time() - t0, 3)

    detections = []
    for box in results.boxes:
        cls_id     = int(box.cls[0])
        confidence = round(float(box.conf[0]), 4)
        name       = FIXED_NAMES.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0]]
        detections.append({
            "class":      name,
            "confidence": confidence,
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return JSONResponse({
        "detections":   detections,
        "count":        len(detections),
        "inference_ms": int(elapsed * 1000),
        "top":          detections[0]["class"] if detections else None,
    })


@app.post("/feedback")
async def feedback(
    file:            UploadFile = File(...),
    correct_label:   str        = Form(...),
    predicted_label: str        = Form("unknown"),
    feedback_type:   str        = Form("correction"),  # confirmation | correction | new_class
    bbox_x1:         float      = Form(None),
    bbox_y1:         float      = Form(None),
    bbox_x2:         float      = Form(None),
    bbox_y2:         float      = Form(None),
    img_width:       float      = Form(640),
    img_height:      float      = Form(480),
):
    """
    Save training feedback to GCS.
    feedback_type:
      confirmation = model was correct, user confirmed (silent, saves with real bbox)
      correction   = model was wrong, user typed correct label
      new_class    = brand new produce not in the 44 classes
    """
    if not FEEDBACK_BUCKET:
        raise HTTPException(status_code=503, detail="Feedback storage not configured. Set FEEDBACK_BUCKET env var.")

    contents = await file.read()
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    folder_map = {
        "confirmation": "confirmations",
        "correction":   "corrections",
        "new_class":    "new_classes",
    }
    folder   = folder_map.get(feedback_type, "corrections")
    img_path = f"{folder}/{correct_label}/{ts}.jpg"
    lbl_path = f"{folder}/{correct_label}/{ts}.txt"
    meta_path= f"{folder}/{correct_label}/{ts}_info.json"

    try:
        client = get_gcs()
        bucket = client.bucket(FEEDBACK_BUCKET)

        # save image
        bucket.blob(img_path).upload_from_string(contents, content_type="image/jpeg")

        # save YOLO label — real bbox if provided, full-frame placeholder otherwise
        if bbox_x1 is not None and bbox_x2 is not None:
            cx = ((bbox_x1 + bbox_x2) / 2) / img_width
            cy = ((bbox_y1 + bbox_y2) / 2) / img_height
            bw = (bbox_x2 - bbox_x1)       / img_width
            bh = (bbox_y2 - bbox_y1)       / img_height
            cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
            bw, bh = max(0.01, min(1.0, bw)), max(0.01, min(1.0, bh))
            label_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        else:
            label_line = "0 0.500000 0.500000 0.900000 0.900000\n"
        bucket.blob(lbl_path).upload_from_string(label_line.encode(), content_type="text/plain")

        # save metadata
        meta = {
            "timestamp":       ts,
            "correct_label":   correct_label,
            "predicted_label": predicted_label,
            "feedback_type":   feedback_type,
            "has_real_bbox":   bbox_x1 is not None,
        }
        bucket.blob(meta_path).upload_from_string(json.dumps(meta, indent=2), content_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}. Make sure the bucket '{FEEDBACK_BUCKET}' exists and Cloud Run has Storage Object Admin role.")

    return JSONResponse({
        "status":  "saved",
        "path":    img_path,
        "message": f"Saved as '{correct_label}' ({feedback_type}). Will be learned on next retrain.",
    })


@app.get("/feedback/stats")
def feedback_stats():
    """Return count of saved corrections per class."""
    if not FEEDBACK_BUCKET:
        return {"error": "FEEDBACK_BUCKET not configured"}

    client = get_gcs()
    bucket = client.bucket(FEEDBACK_BUCKET)

    corrections  = {}
    new_classes  = {}

    confirmations = {}
    for blob in bucket.list_blobs():
        if not blob.name.endswith(".jpg"):
            continue
        parts = blob.name.split("/")
        if len(parts) < 3:
            continue
        folder, cls = parts[0], parts[1]
        if folder == "confirmations":
            confirmations[cls] = confirmations.get(cls, 0) + 1
        elif folder == "corrections":
            corrections[cls]   = corrections.get(cls, 0) + 1
        elif folder == "new_classes":
            new_classes[cls]   = new_classes.get(cls, 0) + 1

    return {
        "confirmations":         confirmations,
        "corrections":           corrections,
        "new_classes":           new_classes,
        "total_confirmations":   sum(confirmations.values()),
        "total_corrections":     sum(corrections.values()),
        "total_new_class_imgs":  sum(new_classes.values()),
    }
