"""
Prepare category-level YOLO dataset from MANUAL_LABELS.
Uses DATASET_FULL (from DATASET.zip) so every folder has its images.
No SAM masking — trains directly on real counter images.

Run:   python prepare_categories.py
Out:   YOLO_CATEGORIES/  -> zip it, upload to Google Drive as YOLO_CATEGORIES.zip
"""

import os, shutil, random, yaml, cv2
import numpy as np

DATASET_DIR = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET_FULL"
LABELS_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MANUAL_LABELS"
OUTPUT_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\YOLO_CATEGORIES"

MIN_TARGET  = 500
TRAIN_SPLIT = 0.85
random.seed(42)

# --- 11 YOLO categories --------------------------------------------------

YOLO_CLASSES = ["banana", "beans", "chilli", "coconut", "dasakai",
                "eggplant", "fruit", "gourd", "ladyfinger", "ladystickers",
                "leafy", "onion", "root", "special", "tomato"]

CLASS_TO_CATEGORY = {
    # BANANA
    "Banana":                 "banana",
    "BURRO BANANA":           "banana",
    "Banana Flower":          "banana",
    # EGGPLANT (dasakai is separate)
    "Indian eggplant":        "eggplant",
    "Chinese eggplant":       "eggplant",
    "Chinese Green Eggplant": "eggplant",
    "THAI EGG PLANT":         "eggplant",
    # DASAKAI - own category
    "Dasakai":                "dasakai",
    # CHILLI
    "FLORIDA  LONG CHILLI":   "chilli",
    "Thai Chilli":            "chilli",
    "Bell pepper":            "chilli",
    # BEANS / PODS (okra and lady stickers are separate)
    "BEANS REGULAR":          "beans",
    "Long green beans":       "beans",
    "String beans":           "beans",
    "FLAT VELOR":             "beans",
    # LADIES FINGER - own category
    "Okra":                   "ladyfinger",
    # LADY STICKERS - own category
    "Lady stickers":          "ladystickers",
    # LEAFY
    "Cabbage":                "leafy",
    "Cauliflower":            "leafy",
    "Mint":                   "leafy",
    "Cilantro":               "leafy",
    "Curry leaves":           "leafy",
    "Leaves":                 "leafy",
    "Pan Leaves":             "leafy",
    # ONION
    "Red Onions":             "onion",
    "White Onions":           "onion",
    # ROOT VEGETABLES
    "Potato":                 "root",
    "Sweet Potato":           "root",
    "Beetroot":               "root",
    "Muli":                   "root",
    "Ginger":                 "root",
    "Garlic":                 "root",
    # GOURD FAMILY (chayote moved to fruit)
    "Pumpkin":                "gourd",
    "Squah":                  "gourd",
    "Snake Guard":            "gourd",
    "Turai":                  "gourd",
    "Karela":                 "gourd",
    "Tindora":                "gourd",
    # FRUIT (chayote included here)
    "Guava":                  "fruit",
    "Papaya":                 "fruit",
    "FRESH CHIKKU":           "fruit",
    "Lemon":                  "fruit",
    "Chayote":                "fruit",
    # COCONUT - own category
    "Coconut":                "coconut",
    # TOMATO
    "Tomato":                 "tomato",
    # STORE ITEMS (non-produce but sold at counter)
    "Boxed Sweets":           "special",
    "Home made snacks":       "special",
    "POLI":                   "special",
    "Roti":                   "special",
    "Mums":                   "special",
    "Pearl":                  "special",
}

CATEGORIES = YOLO_CLASSES
CAT_IDX    = {c: i for i, c in enumerate(CATEGORIES)}


# --- Augmentations -------------------------------------------------------

def clip_bbox(xc, yc, bw, bh):
    x1 = max(0.0, min(1.0, xc - bw / 2))
    y1 = max(0.0, min(1.0, yc - bh / 2))
    x2 = max(0.0, min(1.0, xc + bw / 2))
    y2 = max(0.0, min(1.0, yc + bh / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)

def rotate_bbox(xc, yc, bw, bh, a, iw, ih):
    cx, cy = xc*iw, yc*ih; hw, hh = bw*iw/2, bh*ih/2
    c = np.array([[cx-hw,cy-hh],[cx+hw,cy-hh],[cx+hw,cy+hh],[cx-hw,cy+hh]], dtype=np.float32)
    r = np.deg2rad(a); ca, sa = np.cos(r), np.sin(r); icx, icy = iw/2, ih/2
    rot = np.array([[x*ca-y*sa+icx, x*sa+y*ca+icy] for x, y in c-[icx,icy]])
    x1, y1 = max(0, rot[:,0].min()), max(0, rot[:,1].min())
    x2, y2 = min(iw, rot[:,0].max()), min(ih, rot[:,1].max())
    return (xc,yc,bw,bh) if x2<=x1 or y2<=y1 else ((x1+x2)/2/iw,(y1+y2)/2/ih,(x2-x1)/iw,(y2-y1)/ih)

def tl(lines, fn):
    out = []
    for l in lines:
        p = l.split(); xc,yc,bw,bh = float(p[1]),float(p[2]),float(p[3]),float(p[4])
        xc,yc,bw,bh = fn(xc,yc,bw,bh)
        clipped = clip_bbox(xc, yc, bw, bh)
        if clipped is None:
            continue
        xc,yc,bw,bh = clipped
        out.append(f"{p[0]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return out

def ar(deg):
    def ap(img, lbl):
        h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
        return cv2.warpAffine(img, M, (w,h), borderValue=(255,255,255)), tl(lbl, lambda xc,yc,bw,bh: rotate_bbox(xc,yc,bw,bh,deg,w,h))
    return ap
def afh(img, lbl): return cv2.flip(img, 1), tl(lbl, lambda xc,yc,bw,bh: (1-xc, yc, bw, bh))
def afv(img, lbl): return cv2.flip(img, 0), tl(lbl, lambda xc,yc,bw,bh: (xc, 1-yc, bw, bh))
def afhv(img, lbl): img, lbl = afh(img, lbl); return afv(img, lbl)
def az(f):
    def ap(img, lbl):
        h, w = img.shape[:2]; ch, cw = int(h/f), int(w/f); y0, x0 = (h-ch)//2, (w-cw)//2
        return cv2.resize(img[y0:y0+ch, x0:x0+cw], (w,h), interpolation=cv2.INTER_LINEAR), tl(lbl, lambda xc,yc,bw,bh: (max(0,min(1,(xc-.5)*f+.5)), max(0,min(1,(yc-.5)*f+.5)), min(1,bw*f), min(1,bh*f)))
    return ap
def azo(f):
    def ap(img, lbl):
        h, w = img.shape[:2]; nh, nw = int(h*f), int(w*f); small = cv2.resize(img, (nw, nh))
        canvas = np.ones((h,w,3), dtype=np.uint8)*255; yo, xo = (h-nh)//2, (w-nw)//2
        canvas[yo:yo+nh, xo:xo+nw] = small
        return canvas, tl(lbl, lambda xc,yc,bw,bh: (xc*f+(1-f)/2, yc*f+(1-f)/2, bw*f, bh*f))
    return ap
def abr(f):
    def ap(img, lbl): return np.clip(img.astype(np.float32)*f, 0, 255).astype(np.uint8), lbl[:]
    return ap
def act(f):
    def ap(img, lbl): m = img.mean(); return np.clip((img.astype(np.float32)-m)*f+m, 0, 255).astype(np.uint8), lbl[:]
    return ap
def abl(img, lbl): return cv2.GaussianBlur(img, (5,5), 0), lbl[:]
def ash(img, lbl): return cv2.filter2D(img, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])), lbl[:]
def ch(*fns):
    def ap(img, lbl):
        for fn in fns: img, lbl = fn(img, lbl)
        return img, lbl
    return ap

AUGMENTS = [
    ("rot15",   ar(15)),   ("rot30",   ar(30)),   ("rot45",  ar(45)),
    ("rot_15",  ar(-15)),  ("rot_30",  ar(-30)),
    ("rot90",   ar(90)),   ("rot180",  ar(180)),  ("rot270", ar(270)),
    ("fh",      afh),      ("fv",      afv),      ("fhv",    afhv),
    ("z12",     az(1.2)),  ("z15",     az(1.5)),
    ("z08",     azo(0.8)), ("z07",     azo(0.7)),
    ("br13",    abr(1.3)), ("br07",    abr(0.7)),
    ("ct12",    act(1.2)), ("ct08",    act(0.8)),
    ("blur",    abl),      ("sharp",   ash),
    ("fh_br13", ch(afh, abr(1.3))), ("z12_rot30", ch(az(1.2), ar(30))),
]


# --- Main ----------------------------------------------------------------

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    for sp in ("train", "val"):
        os.makedirs(f"{OUTPUT_DIR}/images/{sp}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{sp}", exist_ok=True)

    cat_pairs = {c: [] for c in CATEGORIES}
    raw_total = skipped = 0

    for cls_name in sorted(os.listdir(LABELS_DIR)):
        cls_lbl_path = os.path.join(LABELS_DIR, cls_name)
        if not os.path.isdir(cls_lbl_path):
            continue
        cat = CLASS_TO_CATEGORY.get(cls_name)
        if cat is None:
            print(f"  SKIP  {cls_name}")
            continue

        img_dir = os.path.join(DATASET_DIR, cls_name)
        if not os.path.isdir(img_dir):
            print(f"  NO IMAGE DIR  {cls_name}")
            continue

        cat_idx = CAT_IDX[cat]
        for txt in sorted(f for f in os.listdir(cls_lbl_path) if f.endswith(".txt")):
            stem = os.path.splitext(txt)[0]
            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                c = os.path.join(img_dir, stem + ext)
                if os.path.exists(c):
                    img_path = c; break
            if img_path is None:
                skipped += 1; continue

            with open(os.path.join(cls_lbl_path, txt)) as f:
                raw_lines = [l.strip() for l in f if l.strip()]
            new_lines = []
            for line in raw_lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                clipped = clip_bbox(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                if clipped is None:
                    continue
                xc, yc, bw, bh = clipped
                new_lines.append(f"{cat_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            if not new_lines:
                skipped += 1
                continue
            cat_pairs[cat].append((img_path, new_lines))
            raw_total += 1

    print(f"\n{'='*55}")
    print(f"Raw labeled images: {raw_total}  (skipped no-image: {skipped})")
    print(f"YOLO Categories ({len(CATEGORIES)}): {CATEGORIES}")
    print(f"{'='*55}")
    for c in CATEGORIES:
        print(f"  {c:<12} {len(cat_pairs[c]):>4} images")

    print(f"\n{'-'*55}")
    print(f"Balancing & augmenting weak categories up to minimum {MIN_TARGET} images...")
    print(f"{'-'*55}")

    grand_total = 0
    for cat in CATEGORIES:
        pairs = cat_pairs[cat]
        n = len(pairs)
        if n == 0:
            print(f"  {cat:<12} NO IMAGES -- skipping"); continue

        augmented = list(pairs); ai = ii = 0
        while len(augmented) < MIN_TARGET:
            ip, sl = pairs[ii % len(pairs)]
            an, af = AUGMENTS[ai % len(AUGMENTS)]
            img = cv2.imread(ip)
            if img is None:
                ii += 1; continue
            try:
                aimg, albl = af(img, sl)
            except Exception:
                ii += 1; continue
            augmented.append((f"__aug_{an}_{ai}", aimg, albl))
            ai += 1
            if ai % len(AUGMENTS) == 0: ii += 1

        random.shuffle(augmented)
        cut = max(1, int(len(augmented) * TRAIN_SPLIT))

        for split, items in {"train": augmented[:cut], "val": augmented[cut:]}.items():
            for item in items:
                if len(item) == 2:
                    ip, lbl = item
                    img = cv2.imread(ip)
                    if img is None: continue
                    fname = f"{cat}_{os.path.splitext(os.path.basename(ip))[0]}"
                else:
                    aid, img, lbl = item
                    fname = f"{cat}{aid}"
                cv2.imwrite(f"{OUTPUT_DIR}/images/{split}/{fname}.jpg", img)
                with open(f"{OUTPUT_DIR}/labels/{split}/{fname}.txt", "w") as f:
                    f.write("\n".join(lbl) + "\n")
                grand_total += 1

        print(f"  {cat:<12} {n:>4} raw  ->  {len(augmented)}  (train {cut} / val {len(augmented)-cut})")

    cfg = {
        "path":  OUTPUT_DIR,
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CATEGORIES),
        "names": CATEGORIES,
    }
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\n{'='*55}")
    print(f"Done -- {grand_total} total images written")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*55}")
    print("\nNext steps:")
    print("  1. Zip YOLO_CATEGORIES folder")
    print("  2. Upload YOLO_CATEGORIES.zip to Google Drive / MyDrive/TUNE-DATAPOS/")
    print("  3. Run colab_train_categories.py in Google Colab")


if __name__ == "__main__":
    main()
