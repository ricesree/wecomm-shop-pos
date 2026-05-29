"""
Vegetable Detection API — 14-class category model
POST /detect           — send image, get top-4 category detections
POST /feedback         — save correction image + correct label to GCS
GET  /feedback/stats   — see how many corrections saved per class
GET  /health           — health check
"""

import os, io, json, time, cv2, numpy as np
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from inference_filter import load_thresholds, apply_class_thresholds, get_base_conf

app = FastAPI(title="Swadesh Food Mart — POS Detection API", version="3.0")
STATIC_DIR = "/app/static" if os.path.exists("/app/static") else "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root(): return FileResponse(os.path.join(STATIC_DIR, "index.html"))

MODEL_PATH        = os.environ.get(
    "MODEL_PATH",
    "/app/best_new.pt" if os.path.exists("/app/best_new.pt") else "best_new.pt"
)
CONF              = float(os.environ.get("CONF_THRESHOLD", "0.40"))
IMGSZ             = int(os.environ.get("IMGSZ",            "640"))
FEEDBACK_BUCKET   = os.environ.get("FEEDBACK_BUCKET",  "")
THRESHOLDS_PATH   = os.environ.get(
    "THRESHOLDS_PATH",
    "/app/thresholds.json" if os.path.exists("/app/thresholds.json") else "thresholds.json"
)

print(f"Loading model from {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
print(f"Model ready — classes: {list(model.names.values())}")

CLASS_THRESHOLDS = load_thresholds(THRESHOLDS_PATH, fallback=CONF)
BASE_CONF        = get_base_conf(CLASS_THRESHOLDS, fallback=CONF)

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
        "status":          "ok",
        "model":           MODEL_PATH,
        "classes":         list(model.names.values()),
        "conf_threshold":  CONF,
        "feedback_bucket": FEEDBACK_BUCKET or "not configured",
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image.")

    t0      = time.time()
    results = model(img, conf=BASE_CONF, verbose=False, imgsz=IMGSZ)[0]
    elapsed = round(time.time() - t0, 3)

    detections = []
    for box in results.boxes:
        cls_id     = int(box.cls[0])
        confidence = round(float(box.conf[0]), 4)
        name       = model.names[cls_id]
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0]]
        detections.append({
            "class":      name,
            "confidence": confidence,
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

    detections = apply_class_thresholds(detections, CLASS_THRESHOLDS, fallback=CONF)
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
    feedback_type:   str        = Form("correction"),
    bbox_x1:         float      = Form(None),
    bbox_y1:         float      = Form(None),
    bbox_x2:         float      = Form(None),
    bbox_y2:         float      = Form(None),
    img_width:       float      = Form(640),
    img_height:      float      = Form(480),
):
    if not FEEDBACK_BUCKET:
        raise HTTPException(503, "Feedback storage not configured. Set FEEDBACK_BUCKET env var.")

    contents    = await file.read()
    ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    label_clean = correct_label.strip().lower()

    CLASS_TO_ID = {name.lower(): i for i, name in model.names.items()}
    cls_id = CLASS_TO_ID.get(label_clean)
    if cls_id is None:
        raise HTTPException(400, f"Unknown label: {correct_label}")

    if bbox_x1 is None:
        return JSONResponse({"status":"skipped","message":"No bounding box provided — not saved."})

    folder    = {"confirmation":"confirmations","correction":"corrections","new_class":"new_classes"}.get(feedback_type,"corrections")
    img_path  = f"{folder}/{label_clean}/{ts}.jpg"
    lbl_path  = f"{folder}/{label_clean}/{ts}.txt"
    meta_path = f"{folder}/{label_clean}/{ts}_info.json"

    cx = max(0.0, min(1.0, ((bbox_x1+bbox_x2)/2)/img_width))
    cy = max(0.0, min(1.0, ((bbox_y1+bbox_y2)/2)/img_height))
    bw = max(0.01, min(1.0, (bbox_x2-bbox_x1)/img_width))
    bh = max(0.01, min(1.0, (bbox_y2-bbox_y1)/img_height))
    label_line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

    try:
        client = get_gcs()
        bucket = client.bucket(FEEDBACK_BUCKET)
        bucket.blob(img_path).upload_from_string(contents, content_type="image/jpeg")
        bucket.blob(lbl_path).upload_from_string(label_line.encode(), content_type="text/plain")
        meta = {"timestamp":ts,"correct_label":label_clean,"predicted_label":predicted_label,"feedback_type":feedback_type}
        bucket.blob(meta_path).upload_from_string(json.dumps(meta,indent=2), content_type="application/json")
    except Exception as e:
        raise HTTPException(500, f"Storage error: {e}")

    return JSONResponse({"status":"saved","path":img_path,
        "message":f"Saved as '{correct_label}' ({feedback_type}). Will be learned on next retrain."})


@app.get("/feedback/stats")
def feedback_stats():
    if not FEEDBACK_BUCKET:
        return {"error": "FEEDBACK_BUCKET not configured"}
    client = get_gcs()
    bucket = client.bucket(FEEDBACK_BUCKET)
    confirmations = {}; corrections = {}; new_classes = {}
    for blob in bucket.list_blobs():
        if not blob.name.endswith(".jpg"): continue
        parts = blob.name.split("/")
        if len(parts) < 3: continue
        folder, cls = parts[0], parts[1]
        if   folder == "confirmations": confirmations[cls] = confirmations.get(cls,0)+1
        elif folder == "corrections":   corrections[cls]   = corrections.get(cls,0)+1
        elif folder == "new_classes":   new_classes[cls]   = new_classes.get(cls,0)+1
    return {"confirmations":confirmations,"corrections":corrections,"new_classes":new_classes,
            "total_confirmations":sum(confirmations.values()),
            "total_corrections":sum(corrections.values()),
            "total_new_class_imgs":sum(new_classes.values())}
