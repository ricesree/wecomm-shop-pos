"""
review_feedback.py
------------------
Shows ALL fb_ images with their saved bounding boxes so you can
verify labels look correct. Relabel any that are wrong.

Controls:
  LEFT / RIGHT arrow  = previous / next image
  B                   = redraw box (rectangle)
  O                   = diagonal box
  P                   = freehand pen
  Z                   = undo last box
  R                   = reset to saved boxes
  ENTER               = save current boxes & next
  S                   = skip (keep as-is) & next
  Q                   = quit
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
obb_phase = 0
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
    return [(int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])),
            (int(p2[0]+w*px),int(p2[1]+w*py)), (int(p1[0]+w*px),int(p1[1]+w*py))]

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
            if abs(cur_ex-cur_sx) > 5 and abs(cur_ey-cur_sy) > 5:
                x1,y1 = min(cur_sx,cur_ex), min(cur_sy,cur_ey)
                x2,y2 = max(cur_sx,cur_ex), max(cur_sy,cur_ey)
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
                xs=[p[0] for p in pen_pts]; ys=[p[1] for p in pen_pts]
                boxes.append([(min(xs),min(ys)),(max(xs),min(ys)),(max(xs),max(ys)),(min(xs),max(ys))])
            pen_pts = []

    elif mode == "obb":
        if event == cv2.EVENT_LBUTTONDOWN:
            if obb_phase == 0:
                obb_phase = 1; obb_p1 = (x,y); obb_p2 = (x,y)
            elif obb_phase == 2:
                boxes.append(obb_corners(obb_p1, obb_p2, (x,y)))
                obb_phase = 0; obb_p1 = obb_p2 = None
        elif event == cv2.EVENT_MOUSEMOVE and obb_phase == 1:
            obb_p2 = (x,y)
        elif event == cv2.EVENT_LBUTTONUP and obb_phase == 1:
            if obb_p1 and np.hypot(x-obb_p1[0], y-obb_p1[1]) > 5:
                obb_p2 = (x,y); obb_phase = 2
            else:
                obb_phase = 0

def resize_to_fit(img, max_w=1050, max_h=700):
    h, w = img.shape[:2]; sc = min(max_w/w, max_h/h, 1.0)
    if sc < 1.0: img = cv2.resize(img, (int(w*sc), int(h*sc)))
    return img, sc

def load_boxes(lbl_path, scale, ow, oh):
    result = []
    if not os.path.exists(lbl_path):
        return result
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

def save_boxes(lbl_path, boxes, class_id, scale, ow, oh):
    lines = []
    for corners in boxes:
        xs = [int(c[0]/scale) for c in corners]
        ys = [int(c[1]/scale) for c in corners]
        x1,y1 = max(0,min(xs)), max(0,min(ys))
        x2,y2 = min(ow,max(xs)), min(oh,max(ys))
        xc = (x1+x2)/2/ow; yc = (y1+y2)/2/oh
        bw = (x2-x1)/ow;   bh = (y2-y1)/oh
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def build_frame(disp, cls, fname, idx, total, is_placeholder):
    global obb_phase, obb_p1, obb_p2, obb_mouse
    vis = disp.copy()

    for i, corners in enumerate(boxes):
        col = COLORS[i % len(COLORS)]
        cv2.polylines(vis, [np.array(corners, np.int32)], True, col, 2)
        cv2.putText(vis, f"#{i+1}", corners[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    if mode == "box" and holding and cur_sx != -1:
        cv2.rectangle(vis, (min(cur_sx,cur_ex),min(cur_sy,cur_ey)),
                           (max(cur_sx,cur_ex),max(cur_sy,cur_ey)), (180,180,180), 1)
    if mode == "pen" and holding and len(pen_pts) > 1:
        cv2.polylines(vis, [np.array(pen_pts,np.int32)], False, (180,180,180), 1)
    if mode == "obb" and obb_phase >= 1 and obb_p1 and obb_p2:
        if obb_phase == 1:
            cv2.line(vis, obb_p1, obb_p2, (200,200,200), 1)
        elif obb_phase == 2:
            cv2.polylines(vis, [np.array(obb_corners(obb_p1,obb_p2,obb_mouse),np.int32)], True, (0,200,255), 2)

    sb_w = 250; dh = vis.shape[0]
    sb = np.zeros((dh, sb_w, 3), np.uint8)

    def t(s, y, col=(200,200,200), sc=0.47):
        cv2.putText(sb, s, (8,y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1)

    status_col = (0,80,255) if is_placeholder else (0,200,80)
    status_txt = "PLACEHOLDER!" if is_placeholder else "LABELED OK"
    t("REVIEW MODE", 22, (100,220,255), 0.54)
    t(status_txt, 46, status_col, 0.50)
    t(f"Class: {cls}", 70, (255,255,255), 0.46)
    t(fname[:28], 92, (150,150,150), 0.38)
    t(f"Image {idx}/{total}", 114, (150,150,150))
    t(f"Boxes: {len(boxes)}", 136, (255,220,80), 0.52)

    t("── DRAW ─────────", 162, (70,70,70))
    t("[B] Rectangle",    184, (255,140,0)  if mode=="box" else (55,55,55))
    t("[P] Freehand",     205, (0,120,255)  if mode=="pen" else (55,55,55))
    t("[O] Diagonal",     226, (0,220,220)  if mode=="obb" else (55,55,55))

    t("── ACTIONS ──────", 252, (70,70,70))
    t("ENTER  save & next", 274, (255,255,255))
    t("← →    prev/next",  296, (200,200,200))
    t("Z      undo last",   318, (200,200,200))
    t("R      reload saved",340, (200,200,200))
    t("S      skip",        362, (200,200,200))
    t("Q      quit",        384, (200,200,200))

    if mode == "obb":
        if obb_phase == 0:   t("Click+drag 1st edge", 415, (0,220,220))
        elif obb_phase == 1: t("Drag to set edge...", 415, (0,220,220))
        elif obb_phase == 2:
            t("Move→set width,", 415, (0,255,180))
            t("then CLICK",      437, (0,255,180))

    return np.hstack([vis, sb])

def main():
    global mode, holding, cur_sx, cur_sy, cur_ex, cur_ey, pen_pts, boxes
    global obb_phase, obb_p1, obb_p2, obb_mouse

    classes = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))])
    class_index = {cls: i for i, cls in enumerate(classes)}

    PLACEHOLDER_BOX = (0.5, 0.5, 0.9, 0.9)

    def is_placeholder_label(lbl_path):
        if not os.path.exists(lbl_path):
            return True
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) != 1:
            return False
        parts = lines[0].split()
        if len(parts) != 5:
            return False
        try:
            cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
            return abs(cx-0.5)<0.001 and abs(cy-0.5)<0.001 and abs(bw-0.9)<0.001 and abs(bh-0.9)<0.001
        except ValueError:
            return False

    # Collect ALL fb_ images
    all_items = []
    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        lbl_dir = os.path.join(LABELS_DIR, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.startswith("fb_") and fname.lower().endswith((".jpg",".jpeg",".png")):
                lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
                placeholder = is_placeholder_label(lbl_path)
                all_items.append((cls, fname, placeholder))

    if not all_items:
        print("No fb_ images found.")
        return

    placeholder_count = sum(1 for _,_,p in all_items if p)
    print(f"Total fb_ images: {len(all_items)}")
    print(f"  Properly labeled: {len(all_items) - placeholder_count}")
    print(f"  Still placeholder: {placeholder_count}")
    print("\nControls: B=box  P=pen  O=diagonal  ←/→=prev/next  ENTER=save  S=skip  Q=quit\n")

    cv2.namedWindow("Review Labeler", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Review Labeler", mouse_cb)

    idx = 0
    resaved = 0
    quit_all = False

    while idx < len(all_items) and not quit_all:
        cls, fname, is_ph = all_items[idx]
        cls_dir  = os.path.join(DATASET_DIR, cls)
        lbl_dir  = os.path.join(LABELS_DIR, cls)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        class_id = class_index.get(cls, 0)
        os.makedirs(lbl_dir, exist_ok=True)

        orig = cv2.imread(os.path.join(cls_dir, fname))
        if orig is None:
            idx += 1; continue
        disp, scale = resize_to_fit(orig)
        oh, ow = orig.shape[:2]

        saved_boxes = load_boxes(lbl_path, scale, ow, oh)

        mode = "box"; holding = False; pen_pts = []
        boxes = saved_boxes if saved_boxes else []
        cur_sx = cur_sy = cur_ex = cur_ey = -1
        obb_phase = 0; obb_p1 = obb_p2 = None

        action = None
        while action is None:
            frame = build_frame(disp, cls, fname, idx+1, len(all_items), is_ph)
            cv2.imshow("Review Labeler", frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):       # ENTER — save
                action = "save"
            elif key == 83 or key == 3 or key == 0xFF & ord('d'):  # right arrow or d
                action = "next"
            elif key == 81 or key == 2 or key == 0xFF & ord('a'):  # left arrow or a
                action = "prev"
            elif key in (ord('s'), ord('S')):
                action = "next"
            elif key in (ord('q'), ord('Q')):
                quit_all = True; action = "quit"
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
                boxes = load_boxes(lbl_path, scale, ow, oh)
                cur_sx = cur_sy = cur_ex = cur_ey = -1

        if action == "save" and boxes:
            save_boxes(lbl_path, boxes, class_id, scale, ow, oh)
            resaved += 1
            print(f"  [{idx+1}/{len(all_items)}] Saved  [{cls}] {fname}")
            idx += 1
        elif action == "prev":
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()
    print(f"\nDone!  Reviewed: {len(all_items)}  |  Resaved: {resaved}")

if __name__ == "__main__":
    main()
