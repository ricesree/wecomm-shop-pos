"""
EfficientNet-B3 vegetable classifier API + GCS feedback collection.

Local:  python app.py  -> http://localhost:8080
"""

import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gcs_feedback import (
    FEEDBACK_BUCKET,
    PREFIX_CONFIRM,
    PREFIX_CORRECT,
    PREFIX_NEW,
    filter_produce_names,
    list_produce_names,
    normalize_produce_name,
    upload_feedback_image,
)
from inference import classify, classes as MODEL_CLASSES

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="VeggieLens EfficientNet-B3 API",
    description=(
        "32-class produce classifier with camera UI, Swagger docs, and GCS feedback "
        "(confirmations / corrections / new produce)."
    ),
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


class Prediction(BaseModel):
    label: str
    prob: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)


class InferResponse(BaseModel):
    success: bool = True
    inference_time_ms: float
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    status: str
    model: str
    feedback_bucket: str


class ProduceListResponse(BaseModel):
    produce: List[str]


class FeedbackSaveResponse(BaseModel):
    status: str = "saved"
    path: str


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def camera_ui():
    index = STATIC_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index)


@app.get("/health", tags=["System"], summary="Health check", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model="efficientnet_b3.onnx",
        feedback_bucket=FEEDBACK_BUCKET or "not configured",
    )


@app.get(
    "/produce-list",
    tags=["Feedback"],
    summary="List produce names for autocomplete",
    response_model=ProduceListResponse,
)
async def produce_list(q: Optional[str] = Query(None, description="Optional filter prefix")):
    all_names = list_produce_names(MODEL_CLASSES)
    if q:
        return ProduceListResponse(produce=filter_produce_names(all_names, q))
    return ProduceListResponse(produce=all_names)


@app.post(
    "/api/infer",
    tags=["Inference"],
    summary="Classify an image",
    response_model=InferResponse,
    responses={400: {"description": "Invalid or unreadable image file"}},
)
async def infer(
    file: UploadFile = File(
        ...,
        description="Image file (JPEG, PNG, etc.) containing a vegetable or fruit",
    ),
):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    start_time = time.perf_counter()
    results = classify(image)
    inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return InferResponse(
        success=True,
        inference_time_ms=inference_time_ms,
        predictions=results,
    )


async def _read_image(image: UploadFile) -> tuple[bytes, Optional[str], Optional[str]]:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file")
    return contents, image.content_type, image.filename


@app.post(
    "/confirm",
    tags=["Feedback"],
    summary="Confirm prediction — save to confirmations/<label>/",
    response_model=FeedbackSaveResponse,
)
async def confirm_feedback(
    image: UploadFile = File(...),
    label: str = Form(...),
):
    try:
        normalized = normalize_produce_name(label)
        contents, content_type, filename = await _read_image(image)
        path = upload_feedback_image(
            PREFIX_CONFIRM, normalized, contents, content_type, filename
        )
        return FeedbackSaveResponse(path=path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post(
    "/correct",
    tags=["Feedback"],
    summary="Correct prediction — save to corrections/<correct_label>/",
    response_model=FeedbackSaveResponse,
)
async def correct_feedback(
    image: UploadFile = File(...),
    correct_label: str = Form(...),
):
    try:
        normalized = normalize_produce_name(correct_label)
        contents, content_type, filename = await _read_image(image)
        path = upload_feedback_image(
            PREFIX_CORRECT, normalized, contents, content_type, filename
        )
        return FeedbackSaveResponse(path=path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post(
    "/new-produce",
    tags=["Feedback"],
    summary="New produce — save to new/<produce_name>/",
    response_model=FeedbackSaveResponse,
)
async def new_produce_feedback(
    image: UploadFile = File(...),
    produce_name: str = Form(...),
):
    try:
        normalized = normalize_produce_name(produce_name)
        contents, content_type, filename = await _read_image(image)
        path = upload_feedback_image(
            PREFIX_NEW, normalized, contents, content_type, filename
        )
        return FeedbackSaveResponse(path=path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
