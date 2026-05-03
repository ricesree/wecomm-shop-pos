# =============================================================================
# TRAIN 15-CLASS POS VEGETABLE DETECTOR  —  A100 80GB optimised
# =============================================================================
# Upload YOLO_CATEGORIES.zip to MyDrive/TUNE-DATAPOS/ before running
# =============================================================================

# STEP 1 — Install + Mount
import subprocess, os, shutil, zipfile, yaml
subprocess.run(["pip", "install", "ultralytics", "-q"], check=True)
from google.colab import drive
drive.mount("/content/drive")
print("Setup complete\n")

# STEP 2 — GPU check
import torch
print("GPU:", torch.cuda.get_device_name(0),
      f"  {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB\n")

# STEP 3 — Paths
DRIVE_ROOT = "/content/drive/MyDrive/TUNE-DATAPOS"
DATA_DIR   = "/content/YOLO_CATEGORIES"
MODEL_DIR  = "/content/drive/MyDrive/TUNE-DATAPOS/MODEL_CATEGORIES"
CKPT_DIR   = f"{MODEL_DIR}/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# STEP 4 — Unzip dataset
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
print("Unzipping YOLO_CATEGORIES...")
with zipfile.ZipFile(f"{DRIVE_ROOT}/YOLO_CATEGORIES.zip", "r") as z:
    z.extractall("/content")
print("Done.\n")

# STEP 5 — Fix data.yaml path (Windows path -> Colab path)
with open(f"{DATA_DIR}/data.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["path"]  = DATA_DIR
cfg["train"] = "images/train"
cfg["val"]   = "images/val"
cfg["nc"]    = 15
cfg["names"] = ["banana", "beans", "chilli", "coconut", "dasakai",
                "eggplant", "fruit", "gourd", "ladyfinger", "ladystickers",
                "leafy", "onion", "root", "special", "tomato"]

with open(f"{DATA_DIR}/data.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print(f"Classes ({cfg['nc']}): {cfg['names']}")
for sp in ("train", "val"):
    n = len(os.listdir(f"{DATA_DIR}/images/{sp}"))
    print(f"  {sp}: {n} images")
print()

# STEP 6 — Train from scratch on A100 80GB
from ultralytics import YOLO

model = YOLO("yolov8s.pt")   # small — same speed as old 51-class model, fine for 15 broad categories

results = model.train(
    data    = f"{DATA_DIR}/data.yaml",
    epochs  = 100,
    imgsz   = 640,
    batch   = 128,           # A100 80GB handles this comfortably
    device  = "cuda",
    workers = 16,
    cache   = "ram",         # 167 GB RAM — cache all images in CPU RAM
    amp     = True,

    patience     = 15,
    save_period  = 20,

    cos_lr          = True,
    lr0             = 0.015,
    lrf             = 0.01,
    warmup_epochs   = 3,
    warmup_momentum = 0.8,

    degrees      = 10,
    translate    = 0.1,
    scale        = 0.5,
    shear        = 3.0,
    fliplr       = 0.5,
    flipud       = 0.3,
    hsv_h        = 0.015,
    hsv_s        = 0.7,
    hsv_v        = 0.4,
    mosaic       = 1.0,
    mixup        = 0.2,
    close_mosaic = 10,

    project = MODEL_DIR,
    name    = "yolov8s-15class",
    save    = True,
    plots   = True,
)

# STEP 7 — Save checkpoints to Drive
weights_dir = f"{MODEL_DIR}/yolov8s-15class/weights"
print("\nSaving checkpoints to Drive...")
for ckpt in ["best.pt", "last.pt", "epoch20.pt", "epoch40.pt", "epoch60.pt", "epoch80.pt"]:
    src = f"{weights_dir}/{ckpt}"
    if os.path.exists(src):
        shutil.copy(src, f"{CKPT_DIR}/{ckpt}")
        print(f"  Saved: {ckpt}")

# STEP 8 — Validate
print("\nValidating...")
model   = YOLO(f"{weights_dir}/best.pt")
metrics = model.val(data=f"{DATA_DIR}/data.yaml", imgsz=640, device="cuda", plots=True)

print("\n========== RESULTS ==========")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"\nPer-class mAP50:")
for name, ap in zip(cfg["names"], metrics.box.ap50):
    print(f"  {name:<15} {ap:.4f}")

print(f"\nModel saved at: {weights_dir}/best.pt")
print("Download best.pt -> rename to best_categories.pt -> run detect_pos.py")
