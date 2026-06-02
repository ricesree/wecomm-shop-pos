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
offline/          — local tools, training scripts, and POS demo
api/              — production FastAPI service and real APIs
  learning_api/   — experimental YOLO learning API
tests/            — placeholder for future automated tests
DATASET/          — Original training images (organized by product)
MANUAL_LABELS/    — YOLO bounding box labels per product
```

Important files:
```
offline/detect_pos.py
offline/prepare_categories.py
offline/colab_train_categories.py
offline/label_new.py
offline/build_feedback_dataset.py
offline/merge_new_data.py
offline/confusion_matrix_eval.py
api/app.py
api/learning_api/app.py
```

## How to run locally
- Local POS demo:
```bash
cd offline
python detect_pos.py
```
- Production-like API service:
```bash
cd api
uvicorn app:app --host 0.0.0.0 --port 8080
```
- Experimental learning API:
```bash
cd api/learning_api
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Deployment
Push to `main` → Cloud Build auto-triggers → downloads `best.pt` from GCS → builds Docker image → deploys to Cloud Run.

To update the model: upload new `best.pt` to `gs://vegdetect-feedback-1076778092661/models/best.pt`, then push any change to trigger a redeploy.
