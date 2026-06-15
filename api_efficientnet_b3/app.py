"""
EfficientNet-B3 vegetable classifier API for Google Cloud Run.

Local:  python app.py  → http://localhost:8080/docs
"""

import os
import time
from typing import List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from inference import classify

app = FastAPI(
    title="VeggieLens EfficientNet-B3 API",
    description=(
        "Upload a vegetable/fruit image and get top-3 class predictions "
        "with confidence scores and per-class thresholds (EfficientNet-B3, 27 classes)."
    ),
    version="1.0.0",
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


@app.get("/health", tags=["System"], summary="Health check", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model="efficientnet_b3.onnx")


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
