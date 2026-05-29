# Learning API

This folder is a small version of your detection backend. It is only for learning with Swagger or Postman.

It does not use the HTML camera UI.

## Folder Structure

```text
learning_api/
  app.py            FastAPI endpoints only
  model.py          YOLO model loading, image decoding, and inference
  requirements.txt  Python packages needed for this small API
  README.md         How the pieces connect
```

Your trained model is reused from:

```text
../api/best_new.pt
```

## How The Layers Connect

```text
Postman / Swagger
-> POST /detect
-> FastAPI UploadFile
-> app.py reads file bytes
-> model.py receives raw bytes
-> model.py uses NumPy byte array
-> model.py uses OpenCV cv2.imdecode()
-> model.py creates image matrix
-> model.py runs YOLO model loaded from best_new.pt
-> model.py formats result.boxes
-> JSON response
```

## File Responsibilities

```text
app.py
```

Only handles API routes:

```text
GET /health
POST /detect
```

```text
model.py
```

Only handles computer vision logic:

```text
load best_new.pt
decode uploaded image bytes
run YOLO inference
convert boxes to JSON-friendly dictionaries
```

## Endpoints

```text
GET /health
```

Checks that the API is running and the YOLO model loaded.

```text
POST /detect
```

Accepts an image file and returns:

```text
class_id
class
confidence
bbox
inference_ms
```

## Run It

From this folder:

```bash
cd learning_api
uvicorn app:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

## Test In Swagger

1. Open `/docs`
2. Click `GET /health`
3. Click `Try it out`
4. Click `Execute`
5. Then open `POST /detect`
6. Upload a vegetable image
7. Click `Execute`

## Test In Postman

```text
POST http://127.0.0.1:8000/detect
```

Body:

```text
form-data
key: file
type: File
value: choose image
```

## Important Keywords

```text
FastAPI
UploadFile
multipart/form-data
raw bytes
NumPy
OpenCV
cv2.imdecode
YOLO
Ultralytics
best_new.pt
inference
confidence
class_id
bounding box
JSON response
Swagger
Postman
```
