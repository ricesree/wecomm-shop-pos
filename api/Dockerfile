FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (saves ~500MB vs default GPU torch)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install app dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    onnx \
    onnxruntime \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    opencv-python-headless \
    aiofiles \
    google-cloud-storage

# Copy model, app, and web UI
COPY best_14class_img640.onnx /app/best_14class_img640.onnx
COPY app.py              /app/app.py
COPY inference_filter.py /app/inference_filter.py
COPY thresholds.json     /app/thresholds.json
COPY static/             /app/static/

ENV MODEL_PATH=/app/best_14class_img640.onnx
ENV CONF_THRESHOLD=0.25
ENV IMGSZ=640

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
