"""
label_feedback.py
-----------------
Labels ONLY the newly downloaded feedback images (files starting with fb_).
Placeholder labels written by download_corrections.py are always overridden.

Controls:
  B     = Rectangle box   (drag)
  P     = Pen / freehand  (hold & draw)
  O     = Sloped/diagonal box  (drag first edge → move for width → click)
  ENTER = save all boxes → next image
  Z     = undo last box
  R     = reset to auto-detected box
  S     = skip (keep placeholder)
  Q     = quit
"""

import cv2
import numpy as np
import os

DATASET_DIR = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\DATASET\DATASET - Copy"
LABELS_DIR  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\MANUAL_LABELS"

COLORS = [
    (0,220,0),(0,180,255),(255,100,0),
    (180,0,255),(0,255,180),(255,220,0),(0,255,255),
]

# ── global state ───────────────────────────────────────────────────────────────
mode      = "box"
holding   = False
cur_sx = cur_sy = cur_ex = cur_ey = -1
pen_pts   = []
boxes     = []

# sloped-box state
obb_phase = 0      # 0=idle  1=dragging edge  2=setting width
obb_p1    = None
obb_p2    = None
obb_mouse = (0, 0)

def obb_corners(p1, p2, mouse):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    length = max(np.hypot(dx, dy), 1)
    ux, uy = dx/length, dy/length
    px, py = -uy, ux
    mx, my = mouse[0]-p1[0], mouse[1]-p1[1]
    w = mx*px + my*py
    c1 = (int(p1[0]),        int(p1[1]))
    c2 = (int(p2[0]),        int(p2[1]))
    c3 = (int(p2[0]+w*px),   int(p2[1]+w*py))
    c4 = (int(p1[0]+w*px),   int(p1[1]+w*py))
    return [c1, c2, c3, c4]

def mouse_cb(event, x, y, flags, param):
    global holding, cur_sx, cur_sy, cur_ex, cur_ey, pen_pts
    global obb_phase, obb_p1, obb_p2, obb_mouse

    obb_mouse = (x, y)

    if mode == "box":
        if event == cv2.EVENT_LBUTTONDOWN:
            holding = True; cur_sx = cur_ex = x; cur_sy = cur_ey = y; pen_pts = []
        elif event == cv2.EVENT_MOUSEMOVE and holding:
            cur_ex, cur_ey = x, y
        elif event == cv2.EVENT_LBUTTONUP and holding:
            holding = False; cur_ex, cur_ey = x, y
            if abs(cur_ex - cur_sx) > 5 and abs(cur_ey - cur_sy) > 5:
                x1, y1 = min(cur_sx, cur_ex), min(cur_sy, cur_ey)
                x2, y2 = max(cur_sx, cur_ex), max(cur_sy, cur_ey)
                boxes.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            cur_sx = cur_sy = cur_ex = cur_ey = -1

    elif mode == "pen":
        if event == cv2.EVENT_LBUTTONDOWN:
            holding = True; pen_pts = [(x,y)]
        elif event == cv2.EVENT_MOUSEMOVE and holding:
            pen_pts.append((x,y))
        elif event == cv2.EVENT_LBUTTONUP and holding:
            holding = False
            if len(pen_pts) > 2:
                xs = [p[0] for p in pen_pts]; ys = [p[1] for p in pen_pts]
                x1,y1 = min(xs),min(ys); x2,y2 = max(xs),max(ys)
                boxes.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            pen_pts = []

    elif mode == "obb":
        if event == cv2.EVENT_LBUTTONDOWN:
            if obb_phase == 0:
                obb_phase = 1; obb_p1 = (x,y); obb_p2 = (x,y)
            elif obb_phase == 2:
                corners = obb_corners(obb_p1, obb_p2, (x,y))
                boxes.append(corners)
                obb_phase = 0; obb_p1 = obb_p2 = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if obb_phase == 1:
                obb_p2 = (x,y)
        elif event == cv2.EVENT_LBUTTONUP and obb_phase == 1:
            if obb_p1 and np.hypot(x-obb_p1[0], y-obb_p1[1]) > 5:
                obb_p2 = (x,y); obb_phase = 2
            else:
                obb_phase = 0

def auto_bbox(img):
    h0, w0 = img.shape[:2]; scale = 0.25
    small = cv2.resize(img, (int(w0*scale), int(h0*scale))); h, w = small.shape[:2]
    mx, my = w//6, h//6; rect = (mx, my, w-2*mx, h-2*my)
    mask = np.zeros((h,w), np.uint8)
    bgd = np.zeros((1,65), np.float64); fgd = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
        k = np.ones((10,10), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            large = [c for c in cnts if cv2.contourArea(c) > w*h*0.015] or \
                    [max(cnts, key=cv2.contourArea)]
            pts = np.concatenate(large); x,y,bw,bh = cv2.boundingRect(pts)
            s = 1/scale; pad = 40
            x  = max(0, int(x*s)-pad);  y  = max(0, int(y*s)-pad)
            bw = min(w0-x, int(bw*s)+2*pad); bh = min(h0-y, int(bh*s)+2*pad)
            return x, y, x+bw, y+bh
    except Exception:
        pass
    px, py = int(w0*0.175), int(h0*0.175)
    return px, py, w0-px, h0-py

def resize_to_fit(img, max_w=1050, max_h=700):
    h, w = img.shape[:2]; sc = min(max_w/w, max_h/h, 1.0)
    if sc < 1.0: img = cv2.resize(img, (int(w*sc), int(h*sc)))
    return img, sc

def build_frame(disp, cls, fname, idx, total):
    global obb_phase, obb_p1, obb_p2, obb_mouse
    vis = disp.copy()

    for i, corners in enumerate(boxes):
        col = COLORS[i % len(COLORS)]
        pts = np.array(corners, np.int32)
        cv2.polylines(vis, [pts], True, col, 2)
        cv2.putText(vis, f"#{i+1}", corners[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    if mode == "box" and holding and cur_sx != -1:
        cv2.rectangle(vis,
                      (min(cur_sx,cur_ex), min(cur_sy,cur_ey)),
                      (max(cur_sx,cur_ex), max(cur_sy,cur_ey)),
                      (180,180,180), 1)
    if mode == "pen" and holding and len(pen_pts) > 1:
        cv2.polylines(vis, [np.array(pen_pts,np.int32)], False, (180,180,180), 1)
    if mode == "obb" and obb_phase >= 1 and obb_p1 and obb_p2:
        if obb_phase == 1:
            cv2.line(vis, obb_p1, obb_p2, (200,200,200), 1)
        elif obb_phase == 2:
            corners = obb_corners(obb_p1, obb_p2, obb_mouse)
            cv2.polylines(vis, [np.array(corners,np.int32)], True, (0,200,255), 2)

    sb_w = 240; dh, dw = vis.shape[:2]
    sb = np.zeros((dh, sb_w, 3), np.uint8)

    def t(s, y, col=(200,200,200), sc=0.47):
        cv2.putText(sb, s, (8,y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)

    t("FEEDBACK LABELER", 20, (100,255,100), 0.52)
    t(f"Class: {cls}", 45, (255,255,255), 0.46)
    t(f"{fname}", 68, (150,150,150), 0.38)
    t(f"Image {idx}/{total}", 90, (150,150,150))
    t(f"Boxes: {len(boxes)}", 112, (255,220,80), 0.52)

    t("── DRAW MODE ────", 138, (70,70,70))
    t("[B] Rectangle",     160, (255,140,0)  if mode=="box" else (55,55,55))
    t("[P] Freehand pen",  181, (0,120,255)  if mode=="pen" else (55,55,55))
    t("[O] Diagonal box",  202, (0,220,220)  if mode=="obb" else (55,55,55))

    t("── ACTIONS ──────", 228, (70,70,70))
    t("ENTER  save & next", 250, (255,255,255))
    t("Z      undo last",   272, (200,200,200))
    t("R      reset auto",  294, (200,200,200))
    t("S      skip image",  316, (200,200,200))
    t("Q      quit",        338, (200,200,200))

    if mode == "obb":
        if obb_phase == 0:   t("Click+drag 1st edge", 375, (0,220,220))
        elif obb_phase == 1: t("Drag to set edge...",  375, (0,220,220))
        elif obb_phase == 2:
            t("Move→ set width,",  375, (0,255,180))
            t("then CLICK",        397, (0,255,180))
    else:
        t("Tip: draw tight", 375, (100,100,200))
        t("box around item", 397, (100,100,200))

    return np.hstack([vis, sb])

def corners_to_yolo(corners, scale, ow, oh):
    xs = [int(c[0]/scale) for c in corners]
    ys = [int(c[1]/scale) for c in corners]
    x1, y1 = max(0,min(xs)), max(0,min(ys))
    x2, y2 = min(ow,max(xs)), min(oh,max(ys))
    xc = (x1+x2)/2/ow; yc = (y1+y2)/2/oh
    bw = (x2-x1)/ow;   bh = (y2-y1)/oh
    return xc, yc, bw, bh

def main():
    global mode, holding, cur_sx, cur_sy, cur_ex, cur_ey, pen_pts, boxes
    global obb_phase, obb_p1, obb_p2, obb_mouse

    # Build class index (same sorted order used during training)
    classes = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))])
    class_index = {cls: i for i, cls in enumerate(classes)}

    def is_placeholder(lbl_path):
        """Return True if the label file is missing or still has the download placeholder."""
        if not os.path.exists(lbl_path):
            return True
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) != 1:
            return False   # multiple boxes = real label
        parts = lines[0].split()
        if len(parts) != 5:
            return False
        try:
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # placeholder is exactly 0.5 0.5 0.9 0.9
            return abs(cx-0.5)<0.001 and abs(cy-0.5)<0.001 and abs(bw-0.9)<0.001 and abs(bh-0.9)<0.001
        except ValueError:
            return False

    def load_label_boxes(lbl_path, scale, ow, oh):
        """Load saved YOLO boxes from a label file and convert to display-space corners."""
        result = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw/2) * ow * scale)
                y1 = int((cy - bh/2) * oh * scale)
                x2 = int((cx + bw/2) * ow * scale)
                y2 = int((cy + bh/2) * oh * scale)
                result.append([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
        return result

    # Collect only fb_ images that still need labeling
    todo = []
    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        lbl_dir = os.path.join(LABELS_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.startswith("fb_") and fname.lower().endswith((".jpg",".jpeg",".png")):
                lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
                if is_placeholder(lbl_path):
                    todo.append((cls, fname))

    if not todo:
        print("No fb_ images found in DATASET_DIR. Nothing to label.")
        return

    print(f"Found {len(todo)} feedback images to label.\n")
    print("Controls: B=box  P=pen  O=diagonal  ENTER=save  Z=undo  R=reset  S=skip  Q=quit\n")

    cv2.namedWindow("Feedback Labeler", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Feedback Labeler", mouse_cb)

    saved = 0; skipped = 0; quit_all = False

    for img_num, (cls, fname) in enumerate(todo, 1):
        if quit_all:
            break

        cls_dir  = os.path.join(DATASET_DIR, cls)
        lbl_dir  = os.path.join(LABELS_DIR, cls)
        os.makedirs(lbl_dir, exist_ok=True)

        img_path = os.path.join(cls_dir, fname)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        class_id = class_index.get(cls, 0)

        orig = cv2.imread(img_path)
        if orig is None:
            print(f"  Cannot read {img_path}, skipping.")
            continue

        disp, scale = resize_to_fit(orig)
        oh, ow = orig.shape[:2]

        ax1, ay1, ax2, ay2 = auto_bbox(orig)
        dax1 = int(ax1*scale); day1 = int(ay1*scale)
        dax2 = int(ax2*scale); day2 = int(ay2*scale)
        auto_corners = [(dax1,day1),(dax2,day1),(dax2,day2),(dax1,day2)]

        mode = "box"; holding = False; pen_pts = []
        # load previously saved boxes, otherwise fall back to auto-detect
        if os.path.exists(lbl_path) and not is_placeholder(lbl_path):
            boxes = load_label_boxes(lbl_path, scale, ow, oh) or [auto_corners[:]]
        else:
            boxes = [auto_corners[:]]
        cur_sx = cur_sy = cur_ex = cur_ey = -1
        obb_phase = 0; obb_p1 = obb_p2 = None

        while True:
            frame = build_frame(disp, cls, fname, img_num, len(todo))
            cv2.imshow("Feedback Labeler", frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):   # ENTER — save
                if boxes:
                    lines = []
                    for corners in boxes:
                        xc, yc, bw, bh = corners_to_yolo(corners, scale, ow, oh)
                        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    with open(lbl_path, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    saved += 1
                    print(f"  [{img_num}/{len(todo)}] Saved {len(boxes)} box(es)  [{cls}] {fname}")
                break

            elif key in (ord('b'), ord('B')):
                mode = "box"; pen_pts = []; holding = False; obb_phase = 0
                cur_sx = cur_sy = cur_ex = cur_ey = -1
            elif key in (ord('p'), ord('P')):
                mode = "pen"; pen_pts = []; holding = False; obb_phase = 0
                cur_sx = cur_sy = cur_ex = cur_ey = -1
            elif key in (ord('o'), ord('O')):
                mode = "obb"; pen_pts = []; holding = False; obb_phase = 0
                cur_sx = cur_sy = cur_ex = cur_ey = -1
            elif key in (ord('z'), ord('Z')):
                if boxes: boxes.pop()
            elif key in (ord('r'), ord('R')):
                mode = "box"; pen_pts = []; holding = False; obb_phase = 0
                boxes = [auto_corners[:]]; cur_sx = cur_sy = cur_ex = cur_ey = -1
            elif key in (ord('s'), ord('S')):
                skipped += 1
                print(f"  [{img_num}/{len(todo)}] Skipped  [{cls}] {fname}")
                break
            elif key in (ord('q'), ord('Q')):
                quit_all = True; break

    cv2.destroyAllWindows()
    print(f"\nDone!  Saved: {saved}  |  Skipped: {skipped}  |  Total: {len(todo)}")
    print(f"Labels written to: {LABELS_DIR}")
    print("\nNEXT STEPS:")
    print("  1. python augment.py          (augment the new fb_ images)")
    print("  2. Zip DATASET + MANUAL_LABELS and upload to Drive")
    print("  3. Run Colab fine-tune from best.pt (30 epochs)")

if __name__ == "__main__":
    main()
