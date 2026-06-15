# EfficientNet-B3 Branch (`efficientnetb3-model`)

This branch contains **only** the EfficientNet-B3 classification API for Google Cloud Run.

It does **not** include the YOLO POS system, training scripts, or datasets from the `main` branch.

## Branch layout

```
api_efficientnet_b3/     ← Cloud Run service (FastAPI + ONNX)
  app.py
  inference.py
  Dockerfile
  cloudbuild.yaml
  class_thresholds.csv
  VeggieLens_API.postman_collection.json
  models/README.md       ← how to upload .onnx to GCS
```

## Main branch vs this branch

| | `main` branch | `efficientnetb3-model` branch |
|---|---------------|-------------------------------|
| Purpose | Live POS with camera + YOLO detection | Image upload API with EfficientNet-B3 |
| API folder | `api/` | `api_efficientnet_b3/` |
| Model | YOLO `best_new.pt` | EfficientNet-B3 `efficientnet_b3.onnx` |
| Classes | 14 detection categories | 27 classification classes |
| Cloud Run service | `vegdetect-api` | `vegdetect-efficientnet-b3` |

## Model storage (important)

**Do not commit `.onnx` or `.pt` files to Git.**

Upload the ONNX model to GCS:

```bash
gsutil cp efficientnet_b3.onnx gs://vegdetect-pos-models/models/efficientnet_b3.onnx
```

Cloud Build pulls the model from GCS during deployment.

## Deploy

```bash
gcloud builds submit --config=api_efficientnet_b3/cloudbuild.yaml .
```

## API endpoints (after deploy)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/api/infer` | Upload image (`file` field) |

## Postman

Import `api_efficientnet_b3/VeggieLens_API.postman_collection.json` and set `baseUrl` to your Cloud Run URL.

More detail: `api_efficientnet_b3/README.md`
