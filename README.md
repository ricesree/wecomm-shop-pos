# EfficientNet-B3 Branch (`efficientnetb3-model`)

Standalone EfficientNet-B3 produce classifier API for Google Cloud Run (32 classes).

## Layout

```
api_efficientnet_b3/   Cloud Run service (FastAPI + ONNX via GCS)
train.py               Local training pipeline
deploy_cloud_run.ps1   Upload model to GCS + Cloud Build deploy
pull_gcs_feedback.py   Download live feedback from GCS
merge_dataset.py       Merge feedback into dataset_overall/
split_dataset.py       Create train/val/test splits
```

**Do not commit** `.onnx`, `.pth`, datasets, or `results_new/` model binaries.

## Model upload (before deploy)

```powershell
gsutil cp results_new\efficientnet_b3.onnx gs://vegdetect-pos-models/models/efficientnet_b3.onnx
gsutil cp api_efficientnet_b3\class_thresholds.csv gs://vegdetect-pos-models/models/class_thresholds.csv
```

## Deploy

```powershell
gcloud auth login
.\deploy_cloud_run.ps1
```

Service: `vegdetect-api` · Region: `us-central1` · Project: `wezard-similarity-score`

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Camera UI |
| GET | `/docs` | Swagger UI |
| GET | `/health` | Health check |
| POST | `/api/infer` | Classify image (`file` field) |

## Training

```powershell
python train.py
```

Outputs go to `results_new/` (metrics, ONNX, thresholds). Then upload ONNX to GCS and redeploy.
