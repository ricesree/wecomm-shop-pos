# =============================================================================
# TRAIN 15-CLASS POS VEGETABLE DETECTOR  —  A100 80GB + DRIVE SAFE
# =============================================================================
# Upload YOLO_CATEGORIES_new.zip to MyDrive/TUNE-DATAPOS/ before running
# =============================================================================

# STEP 1 — Install + Mount
import subprocess, os, shutil, zipfile, yaml, json, csv
import numpy as np
subprocess.run(["pip", "install", "ultralytics", "-q"], check=True)
from google.colab import drive
drive.mount("/content/drive")
print("Setup complete\n")

# STEP 2 — GPU check
import torch
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0),
      f"  {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB\n")

# STEP 3 — Paths (all outputs go directly to Drive)
DRIVE_ROOT  = "/content/drive/MyDrive/TUNE-DATAPOS"
DATA_DIR    = "/content/YOLO_CATEGORIES"
MODEL_DIR   = f"{DRIVE_ROOT}/MODEL_CATEGORIES"
CKPT_DIR    = f"{MODEL_DIR}/checkpoints"
THRESH_DIR  = f"{MODEL_DIR}/thresholds"
os.makedirs(CKPT_DIR,   exist_ok=True)
os.makedirs(THRESH_DIR, exist_ok=True)

CLASS_NAMES = ["banana", "beans", "chilli", "coconut", "dasakai",
               "eggplant", "fruit", "gourd", "ladyfinger", "ladystickers",
               "leafy", "onion", "root", "special", "tomato"]

# STEP 4 — Unzip dataset (Windows-safe)
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
zip_path = f"{DRIVE_ROOT}/YOLO_CATEGORIES_new.zip"
if not os.path.exists(zip_path):
    zip_path = f"{DRIVE_ROOT}/YOLO_CATEGORIES.zip"
print(f"Unzipping {os.path.basename(zip_path)}...")
with zipfile.ZipFile(zip_path, "r") as z:
    for member in z.infolist():
        member.filename = member.filename.replace("\\", "/")
        z.extract(member, "/content")
print("Done.\n")

# STEP 5 — Fix data.yaml
yaml_path = f"{DATA_DIR}/data.yaml"
with open(yaml_path) as f:
    cfg = yaml.safe_load(f)
cfg.update({"path": DATA_DIR, "train": "images/train", "val": "images/val",
            "nc": 15, "names": CLASS_NAMES})
with open(yaml_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print(f"Classes ({cfg['nc']}): {cfg['names']}")
for sp in ("train", "val"):
    print(f"  {sp}: {len(os.listdir(f'{DATA_DIR}/images/{sp}'))} images")
print()

# STEP 6 — Train (saves directly to Drive)
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data    = yaml_path,
    epochs  = 130,
    imgsz   = 640,
    batch   = 256,
    device  = "cuda",
    workers = 16,
    cache   = "ram",
    amp     = True,

    patience     = 20,
    save_period  = 10,

    cos_lr          = True,
    lr0             = 0.01,
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

# STEP 7 — Copy checkpoints to Drive/checkpoints/
weights_dir = str(results.save_dir) + "/weights"
print("\nCopying checkpoints...")
for ckpt in ["best.pt", "last.pt", "epoch30.pt", "epoch50.pt", "epoch70.pt",
             "epoch90.pt", "epoch110.pt", "epoch120.pt", "epoch130.pt"]:
    src = f"{weights_dir}/{ckpt}"
    if os.path.exists(src):
        dest = "best_new.pt" if ckpt == "best.pt" else ckpt
        shutil.copy(src, f"{CKPT_DIR}/{dest}")
        print(f"  Saved: {dest}")

# STEP 8 — Validate
print("\nValidating...")
best_path = f"{weights_dir}/best.pt"
model     = YOLO(best_path)
metrics   = model.val(data=yaml_path, imgsz=640, device="cuda", plots=True)

print("\n========== RESULTS ==========")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print("\nPer-class mAP50:")
for name, ap in zip(CLASS_NAMES, metrics.box.ap50):
    print(f"  {name:<15} {ap:.4f}")

# STEP 9 — Per-class threshold optimization
# Runs inference once on val set at low conf, then sweeps thresholds
# analytically to find the best F1 per class. No retraining needed.
print("\n\nOptimizing per-class confidence thresholds...")

VAL_IMG = f"{DATA_DIR}/images/val"
VAL_LBL = f"{DATA_DIR}/labels/val"
MIN_CONF = 0.05
IOU_THR  = 0.5
STEP     = 0.02

def _iou(b1, b2):
    ix1=max(b1[0],b2[0]); iy1=max(b1[1],b2[1])
    ix2=min(b1[2],b2[2]); iy2=min(b1[3],b2[3])
    inter=max(0.,ix2-ix1)*max(0.,iy2-iy1)
    a=(b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/a if a>1e-6 else 0.

from pathlib import Path
nc         = len(CLASS_NAMES)
all_preds  = [[] for _ in range(nc)]
gt_counts  = [0]*nc

img_files = sorted(Path(VAL_IMG).glob("*.jpg")) + sorted(Path(VAL_IMG).glob("*.png"))
print(f"Running val inference on {len(img_files)} images...")

for img_path in img_files:
    lbl_path = Path(VAL_LBL) / (img_path.stem + ".txt")
    result   = model.predict(str(img_path), conf=MIN_CONF, verbose=False, imgsz=640)[0]
    h, w     = result.orig_shape

    gt = []
    if lbl_path.exists():
        for line in open(lbl_path):
            p = line.strip().split()
            if len(p) < 5: continue
            c = int(p[0]); cx,cy,bw,bh = map(float,p[1:5])
            gt.append([c,(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
            if c < nc: gt_counts[c] += 1

    if not len(result.boxes): continue
    confs   = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    xyxy    = result.boxes.xyxy.cpu().numpy()
    matched = set()

    for idx in np.argsort(-confs):
        cid = int(cls_ids[idx]); cv = float(confs[idx]); pb = xyxy[idx].tolist()
        best_iou, best_gi = 0., -1
        for gi, g in enumerate(gt):
            if g[0]!=cid or gi in matched: continue
            iou = _iou(pb, g[1:])
            if iou > best_iou: best_iou, best_gi = iou, gi
        is_tp = 1 if best_iou >= IOU_THR else 0
        if is_tp: matched.add(best_gi)
        if cid < nc: all_preds[cid].append((cv, is_tp))

sweep   = np.arange(MIN_CONF, 0.96, STEP)
opt     = {}
curves  = {}

for cid, cname in enumerate(CLASS_NAMES):
    preds    = np.array(all_preds[cid]) if all_preds[cid] else np.empty((0,2))
    total_gt = gt_counts[cid]
    if not len(preds) or not total_gt:
        opt[cname] = {"threshold":0.25,"precision":0.,"recall":0.,"f1":0.}
        print(f"  {cname:<15} no data → default 0.25")
        continue
    best_f1, best_t, best_p, best_r = -1., 0.25, 0., 0.
    curve = []
    for t in sweep:
        kept=preds[preds[:,0]>=t]; tp=float(kept[:,1].sum()) if len(kept) else 0.
        fp=float(len(kept)-tp); fn=float(total_gt-tp)
        p=tp/(tp+fp) if (tp+fp)>0 else 0.; r=tp/(tp+fn) if (tp+fn)>0 else 0.
        f=2*p*r/(p+r) if (p+r)>0 else 0.
        curve.append((round(float(t),3),round(p,4),round(r,4),round(f,4)))
        if f > best_f1: best_f1,best_t,best_p,best_r = f,float(t),p,r
    opt[cname]    = {"threshold":round(best_t,2),"precision":round(best_p,4),
                     "recall":round(best_r,4),"f1":round(best_f1,4)}
    curves[cname] = curve
    print(f"  {cname:<15} t={best_t:.2f}  P={best_p:.3f}  R={best_r:.3f}  F1={best_f1:.3f}")

# Save thresholds
simple = {k: v["threshold"] for k,v in opt.items()}
with open(f"{THRESH_DIR}/thresholds.json", "w") as f:
    json.dump(simple, f, indent=2)
with open(f"{THRESH_DIR}/thresholds_detailed.json", "w") as f:
    json.dump(opt, f, indent=2)
with open(f"{THRESH_DIR}/thresholds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class","threshold","precision","recall","f1"])
    for k,v in opt.items():
        w.writerow([k,v["threshold"],v["precision"],v["recall"],v["f1"]])

print(f"\nSaved thresholds → {THRESH_DIR}/")
print("\n========== PER-CLASS THRESHOLDS ==========")
print(json.dumps(simple, indent=2))
print(f"\nbest_new.pt saved at: {CKPT_DIR}/best_new.pt")
print("Upload best_new.pt to GCS bucket manually to deploy.")
