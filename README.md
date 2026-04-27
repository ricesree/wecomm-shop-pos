# Swadesh Food Mart — AI Vegetable Detection POS

An AI-powered Point of Sale system that uses a live camera to automatically recognize vegetables and generate bills.

## Features
- Real-time vegetable detection using YOLOv8-M (51 classes)
- Live camera feed with bounding box overlay
- Auto-add to cart on detection
- Manual search and correction
- Self-learning feedback loop — staff corrections saved to GCS and used for retraining
- Hosted on Google Cloud Run

## Tech Stack
- **Model**: YOLOv8-M (Ultralytics / PyTorch)
- **API**: FastAPI on Google Cloud Run
- **Frontend**: Vanilla JS + Canvas API
- **Feedback Storage**: Google Cloud Storage
- **Training**: Google Colab T4 GPU
