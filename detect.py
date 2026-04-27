"""
STEP 3 — Detect vegetables
Draws bounding boxes + class names on images.

Usage:
  python detect.py                    # run on all dataset images (saves to DETECTIONS/)
  python detect.py camera             # live webcam detection
  python detect.py image path/to.jpg  # detect on one image
"""

import sys
import os
import cv2
from ultralytics import YOLO

MODEL_PATH  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MODEL_MANUAL\yolov8m-prod\weights\best.pt"
TEST_DIR    = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET - Copy"
OUTPUT_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DETECTIONS"
CONF        = 0.30     # confidence threshold (0–1)
SAMPLES     = 5        # images per class when running on full dataset


# Correct class names — fixes the phantom MANUAL_LABELS class shift
# that was caused by an extra folder in MANUAL_LABELS on Colab.
# Classes 0-23 were always correct. Classes 24-43 were shifted by 1.
FIXED_NAMES = {
    0:  "Banana",                1:  "Banana Flower",
    2:  "Beetroot",              3:  "Bell pepper",
    4:  "Boxed Sweets",          5:  "Cabbage",
    6:  "Cauliflower",           7:  "Chayote",
    8:  "Chinese Green Eggplant",9:  "Chinese eggplant",
    10: "Cilantro",              11: "Coconut",
    12: "Curry leaves",          13: "Dasakai",
    14: "Garlic",                15: "Ginger",
    16: "Guava",                 17: "Home made snacks",
    18: "Indian eggplant",       19: "Karela",
    20: "Lady stickers",         21: "Leaves",
    22: "Lemon",                 23: "Long green beans",
    24: "Mint",                  25: "Muli",
    26: "Mums",                  27: "Okra",
    28: "Pan Leaves",            29: "Papaya",
    30: "Pearl",                 31: "Potato",
    32: "Pumpkin",               33: "Red Onions",
    34: "Roti",                  35: "Snake Guard",
    36: "Squah",                 37: "String beans",
    38: "Sweet Potato",          39: "Thai Chilli",
    40: "Tindora",               41: "Tomato",
    42: "Turai",                 43: "White Onions",
    44: "UNUSED",
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: trained model not found at:\n  {MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)
    model = YOLO(MODEL_PATH)
    model.model.names = FIXED_NAMES
    return model


def detect_folder(model):
    """Run on up to SAMPLES images per class and save results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cls in sorted(os.listdir(TEST_DIR)):
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue

        out_cls = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_cls, exist_ok=True)

        imgs = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:SAMPLES]

        for fname in imgs:
            src     = os.path.join(cls_dir, fname)
            results = model(src, conf=CONF, verbose=False)
            annotated = results[0].plot()   # draws boxes + labels
            cv2.imwrite(os.path.join(out_cls, fname), annotated)

        print(f"  {cls}: {len(imgs)} images saved to DETECTIONS/{cls}/")

    print(f"\nAll detection results saved to: {OUTPUT_DIR}")


def detect_image(model, path):
    """Detect on a single image and show it."""
    results   = model(path, conf=CONF)
    annotated = results[0].plot()
    cv2.imshow("Vegetable Detection", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_camera(model):
    """Real-time webcam detection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # Lower capture resolution for faster CPU inference
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    import time
    fps_times = []
    print("Webcam detection running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        # imgsz=320 is 4x faster than 640 on CPU with small accuracy tradeoff
        results   = model(frame, conf=CONF, verbose=False, imgsz=320)
        annotated = results[0].plot()
        elapsed   = time.time() - t0

        fps_times.append(elapsed)
        if len(fps_times) > 10:
            fps_times.pop(0)
        fps = 1.0 / (sum(fps_times) / len(fps_times))

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Vegetable Detection  (press Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    model = load_model()

    mode = sys.argv[1] if len(sys.argv) > 1 else "folder"

    if mode == "camera":
        detect_camera(model)
    elif mode == "image" and len(sys.argv) > 2:
        detect_image(model, sys.argv[2])
    else:
        detect_folder(model)


if __name__ == "__main__":
    main()
