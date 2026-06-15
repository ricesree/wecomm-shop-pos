# EfficientNet-B3 model files — do not commit to Git

Upload **both** model files to Google Cloud Storage before deploying:

```bash
gsutil cp efficientnet_b3.onnx gs://vegdetect-pos-models/models/efficientnet_b3.onnx
gsutil cp efficientnet_b3.onnx.data gs://vegdetect-pos-models/models/efficientnet_b3.onnx.data
```

The `.onnx` file and `.onnx.data` file must stay together — ONNX uses the `.data` file for large weight tensors.
