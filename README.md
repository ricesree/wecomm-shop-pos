# Swadesh Food Mart — AI Vegetable Detection POS

An AI-powered Point of Sale system that uses a live camera to automatically recognize vegetables and generate bills.

## Features
- Real-time vegetable detection using YOLOv8s (14 categories)
- Live camera feed with bounding box overlay
- 2-layer detection: YOLO identifies category → cashier selects specific product
- Manual search and price lookup
- Self-learning feedback loop — staff corrections saved to GCS for retraining
- Hosted on Google Cloud Run (auto-deploy on git push)

## Detection Categories
banana, beans, chilli, coconut, dasakai, eggplant, fruit, gourd, ladyfinger, leafy, onion, root, special, tomato

## Confusion Testing
After training in Colab, run `confusion_matrix_eval.py` with the saved `best_new.pt` and extracted `YOLO_CATEGORIES` folder to generate `confusion_matrix.csv` and `confusion_report.json`. Use that report to tune cross-category lookalikes in the POS UI.

## Tech Stack
- **Model**: YOLOv8s (Ultralytics / PyTorch) — trained on 10,000+ images across 14 classes
- **API**: FastAPI on Google Cloud Run
- **Frontend**: Vanilla JS + Canvas API
- **Feedback Storage**: Google Cloud Storage
- **Training**: Google Colab A100 GPU
- **CI/CD**: Google Cloud Build — auto-deploys on every push to main

## Project Structure
```
api/              — FastAPI app, Dockerfile, Cloud Build config
DATASET/          — Original training images (organized by product)
MANUAL_LABELS/    — YOLO bounding box labels per product
prepare_categories.py  — Builds YOLO_CATEGORIES dataset from raw images
colab_train_categories.py  — Training script (run on Google Colab)
detect_pos.py     — Local POS detection (OpenCV, for offline use)
label.py / label_new.py  — Manual bounding box labeling tools
prices.xlsx       — Product prices
```

## Deployment
Push to `main` → Cloud Build auto-triggers → downloads `best.pt` from GCS → builds Docker image → deploys to Cloud Run.

To update the model: upload new `best.pt` to `gs://vegdetect-feedback-1076778092661/models/best.pt`, then push any change to trigger a redeploy.
