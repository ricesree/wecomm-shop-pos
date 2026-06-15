# EfficientNet-B3 model files — do not commit to Git

Upload models to Google Cloud Storage before deploying:

```bash
gsutil cp efficientnet_b3.onnx gs://vegdetect-pos-models/models/efficientnet_b3.onnx
```

Cloud Build downloads the model from GCS during the Docker build. The `.onnx` file is baked into the image at build time, not stored in this repository.
