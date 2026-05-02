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
    onnxruntime \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    opencv-python-headless \
    aiofiles \
    google-cloud-storage

# Copy model, app, and web UI
COPY best.pt        /app/best.pt
COPY app.py         /app/app.py
COPY static/        /app/static/

# Export to ONNX at build time — 2-3x faster CPU inference than PyTorch
RUN python -c "\
from ultralytics import YOLO; \
YOLO('/app/best.pt').export(format='onnx', imgsz=320, simplify=True, opset=12); \
print('ONNX export done')"

ENV MODEL_PATH=/app/best.onnx
ENV CONF_THRESHOLD=0.25
ENV IMGSZ=320

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
