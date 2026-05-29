"""FastAPI endpoints for the small learning detection API."""

from fastapi import FastAPI, File, HTTPException, UploadFile

from model import detector


app = FastAPI(title="Learning YOLO Detection API", version="1.0")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": detector.model_path,
        "confidence_threshold": detector.conf_threshold,
        "image_size": detector.imgsz,
        "classes": detector.classes,
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    contents = await file.read()
    try:
        return detector.detect(contents)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
