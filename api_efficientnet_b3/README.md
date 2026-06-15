# VeggieLens EfficientNet-B3 API

This folder is a **standalone Cloud Run service** for vegetable/fruit image classification.

It is **not** the YOLO live-camera POS API (`api/` on the `main` branch).

## What this service does

- Model: EfficientNet-B3 (ONNX), 27 classes, ~92% accuracy
- Input: single image upload
- Output: top-3 predictions with confidence and per-class threshold
- Swagger UI: `/docs`
- Inference endpoint: `POST /api/infer` (form field: `file`)

## Model files (not in Git)

`.onnx` and `.pt` files are **not stored in GitHub**. Upload the model to GCS once:

```bash
gsutil cp efficientnet_b3.onnx gs://vegdetect-pos-models/models/efficientnet_b3.onnx
```

Cloud Build downloads the model from GCS before building the Docker image.

See `models/README.md` for details.

## Local development

```bash
cd api_efficientnet_b3
pip install -r requirements.txt

# Place efficientnet_b3.onnx in this folder (not committed to Git)
python app.py
```

Open http://localhost:8080/docs

## Deploy to Cloud Run

From the **repository root** (parent of this folder):

```bash
gcloud builds submit --config=api_efficientnet_b3/cloudbuild.yaml .
```

Service name: `vegdetect-efficientnet-b3`  
Region: `us-central1`

## Postman

Import `VeggieLens_API.postman_collection.json`.  
Set variable `baseUrl` to your Cloud Run URL.

## Files in this folder

| File | Purpose |
|------|---------|
| `app.py` | FastAPI application |
| `inference.py` | ONNX inference logic |
| `class_thresholds.csv` | Per-class confidence thresholds |
| `Dockerfile` | Container image |
| `cloudbuild.yaml` | Build and deploy to Cloud Run |
| `requirements.txt` | Python dependencies |
