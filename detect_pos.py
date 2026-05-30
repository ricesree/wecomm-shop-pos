"""
POS Vegetable Detection System
-------------------------------
Camera detects vegetable → shows category options with prices → cashier selects → billing.

Controls:
  1-9   = select item by number
  SPACE = confirm selection & add to cart
  Q     = quit

Setup:
  1. Copy best_categories.pt (trained model) to this folder
  2. Edit prices.xlsx to set your real prices
  3. Run:  python detect_pos.py
"""

import cv2, os
import time
import numpy as np
import openpyxl
from ultralytics import YOLO

# Mouse click state shared with main loop
CLICK_INFO = {"cam_w": None, "panel_w": None, "click": None}


def mouse_callback(event, x, y, flags, param):
    # store left-button clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_INFO["click"] = (x, y)

# --- Config ---------------------------------------------------------------

MODEL_PATH   = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\best_categories.pt"
PRICES_PATH  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\prices.xlsx"
CONF_THRESH  = 0.55
IMGSZ        = 640
CAMERA_ID    = 0
STABLE_REQ   = 8
DETECTION_DISPLAY_SECONDS = float(os.environ.get("DETECTION_DISPLAY_SECONDS", "5"))

# --- Product Map: YOLO category → sellable items -------------------------

PRODUCT_MAP = {
    "banana":      ["Banana", "BURRO BANANA", "Banana Flower"],
    "beans":       ["Beans Regular", "Long Green Beans", "String Beans", "FLAT VELOR"],
    "chilli":      ["FLORIDA LONG CHILLI", "Thai Chilli", "Bell Pepper"],
    "coconut":     ["Coconut"],
    "dasakai":     ["Dasakai"],
    "eggplant":    ["Indian Eggplant", "Chinese Eggplant", "Chinese Green Eggplant", "Thai Eggplant", "Graphiti Eggplant"],
    "fruit":       ["Guava", "Papaya", "FRESH CHIKKU", "Lemon", "Chayote"],
    "gourd":       ["Pumpkin", "Snake Gourd", "Ridge Gourd", "Bitter Gourd", "Tindora", "Squash"],
    "ladyfinger":  ["Okra / Ladies Finger"],
    "leafy":       ["Cabbage", "Cauliflower", "Mint", "Cilantro", "Curry Leaves", "Leaves", "Pan Leaves"],
    "onion":       ["Red Onions", "White Onions"],
    "root":        ["Potato", "Sweet Potato", "Beetroot", "Radish", "Ginger", "Garlic", "Edo"],
    "special":     ["Boxed Sweets", "POLI", "Roti", "Mums", "Pearl"],
    "tomato":      ["Tomato"],
}

# --- Load prices from Excel -----------------------------------------------

def load_prices(path):
    prices = {}
    if not os.path.exists(path):
        print(f"WARNING: prices.xlsx not found at {path}")
        return prices
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] and row[1] is not None:
            prices[str(row[0]).strip()] = float(row[1])
    print(f"Loaded {len(prices)} prices from prices.xlsx")
    return prices

# --- Drawing helpers ------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_camera_view(frame, detections):
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cat  = det["category"]
        conf = det["conf"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 60), 3)
        label = f"{cat}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.75, 2)
        cv2.rectangle(vis, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 200, 60), -1)
        cv2.putText(vis, label, (x1 + 4, y1 - 4), FONT, 0.75, (0, 0, 0), 2)
    return vis


def draw_panel(detected_category, options, selected_idx, cart, prices, show_suboptions, running=True):
    W, H = 460, 720
    panel = np.full((H, W, 3), 25, dtype=np.uint8)

    # -- header --
    if detected_category:
        cv2.rectangle(panel, (0, 0), (W, 55), (20, 150, 20), -1)
        cv2.putText(panel, f"  {detected_category.upper()}", (10, 38),
                    FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(panel, "Select item:", (16, 80), FONT, 0.55, (160, 160, 160), 1)

        for i, opt in enumerate(options):
            y   = 108 + i * 52
            sel = (i == selected_idx)
            bg  = (0, 110, 200) if sel else (55, 55, 55)
            cv2.rectangle(panel, (12, y - 26), (W - 12, y + 20), bg, -1)
            cv2.rectangle(panel, (12, y - 26), (W - 12, y + 20), (90, 90, 90), 1)

            price = prices.get(opt)
            price_str = f"  ${price:.2f}/kg" if price is not None else ""
            cv2.putText(panel, f" [{i+1}]  {opt}", (18, y), FONT, 0.58, (255, 255, 255), 2)
            if price_str:
                (tw, _), _ = cv2.getTextSize(f" [{i+1}]  {opt}", FONT, 0.58, 2)
                cv2.putText(panel, price_str, (18 + tw, y), FONT, 0.55, (100, 220, 100), 1)

        if selected_idx is not None and show_suboptions:
            cv2.rectangle(panel, (12, H - 170), (W - 12, H - 138), (10, 170, 70), -1)
            cv2.putText(panel, "  SPACE = Add to cart", (18, H - 148),
                        FONT, 0.58, (255, 255, 255), 1)
            cv2.putText(panel, "  C = Clear detection", (18, H - 128),
                        FONT, 0.5, (200, 200, 200), 1)
        elif selected_idx is not None and not show_suboptions:
            cv2.rectangle(panel, (12, H - 170), (W - 12, H - 138), (10, 170, 70), -1)
            cv2.putText(panel, "  Press item to expand variants", (18, H - 148),
                        FONT, 0.52, (255, 255, 255), 1)
            cv2.putText(panel, "  C = Clear detection", (18, H - 128),
                        FONT, 0.5, (200, 200, 200), 1)
    else:
        if running:
            cv2.putText(panel, "Waiting for", (30, 80),  FONT, 0.9, (120, 120, 120), 2)
            cv2.putText(panel, "detection...", (30, 120), FONT, 0.9, (120, 120, 120), 2)
        else:
            cv2.putText(panel, "DETECTION PAUSED", (30, 80),  FONT, 0.8, (200, 120, 120), 2)
            cv2.putText(panel, "Press S to Start, T to Stop", (30, 120), FONT, 0.6, (180, 180, 180), 1)

    # -- control buttons (Start / Stop / Clear) --
    gap = 8
    btn_w = 110
    btn_h = 36
    y1 = 10
    y2 = y1 + btn_h
    x2_clear = W - 12
    x1_clear = x2_clear - btn_w
    x2_stop = x1_clear - gap
    x1_stop = x2_stop - btn_w
    x2_start = x1_stop - gap
    x1_start = x2_start - btn_w

    # start button
    start_bg = (20, 160, 20) if not running else (80, 200, 80)
    cv2.rectangle(panel, (x1_start, y1), (x2_start, y2), start_bg, -1)
    cv2.putText(panel, " START ", (x1_start + 8, y1 + 24), FONT, 0.6, (255, 255, 255), 2)

    # stop button
    stop_bg = (160, 40, 40) if running else (90, 90, 90)
    cv2.rectangle(panel, (x1_stop, y1), (x2_stop, y2), stop_bg, -1)
    cv2.putText(panel, " STOP ", (x1_stop + 12, y1 + 24), FONT, 0.6, (255, 255, 255), 2)

    # clear button
    clear_bg = (10, 110, 200)
    cv2.rectangle(panel, (x1_clear, y1), (x2_clear, y2), clear_bg, -1)
    cv2.putText(panel, " CLEAR ", (x1_clear + 10, y1 + 24), FONT, 0.6, (255, 255, 255), 2)

    buttons = {
        "start": (x1_start, y1, x2_start, y2),
        "stop": (x1_stop, y1, x2_stop, y2),
        "clear": (x1_clear, y1, x2_clear, y2),
    }

    # -- cart --
    cv2.line(panel, (0, H - 175), (W, H - 175), (60, 60, 60), 1)
    cv2.putText(panel, f"Cart ({len(cart)} items):", (12, H - 158),
                FONT, 0.52, (160, 160, 160), 1)

    cart_y = H - 138
    for item, price in cart[-4:]:
        price_str = f"${price:.2f}/kg" if price is not None else "no price"
        cv2.putText(panel, f"  - {item}  ({price_str})", (12, cart_y),
                    FONT, 0.46, (200, 200, 200), 1)
        cart_y += 22

    # -- grand total --
    total = sum(p for _, p in cart if p is not None)
    cv2.line(panel, (0, H - 34), (W, H - 34), (80, 80, 80), 1)
    cv2.putText(panel, f"  TOTAL: ${total:.2f}", (12, H - 10),
                FONT, 0.75, (0, 220, 120), 2)

    return panel, buttons


# --- Main -----------------------------------------------------------------

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Train on Colab, download best.pt, rename to best_categories.pt, place here.")
        return

    prices = load_prices(PRICES_PATH)

    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("POS System ready.  Press Q to quit.\n")

    detected_category = None
    options           = []
    detail_options    = []
    show_suboptions   = False
    detected_shown_at = None
    selected_idx      = None
    cart              = []   # list of (item_name, price)
    stable_frames     = 0
    last_cat          = None
    running           = False  # detection active when True (press S to start)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        best_cat   = None
        best_conf  = 0.0

        if running:
            results    = model(frame, conf=CONF_THRESH, verbose=False, imgsz=IMGSZ)[0]
            for box in results.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                cat  = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"box": (x1, y1, x2, y2), "category": cat, "conf": conf})
                if conf > best_conf:
                    best_conf = conf; best_cat = cat

            if best_cat == last_cat:
                stable_frames += 1
            else:
                stable_frames = 0
                last_cat      = best_cat
                selected_idx  = None

            if stable_frames >= STABLE_REQ and best_cat is not None:
                if best_cat != detected_category:
                    detected_category = best_cat
                    detail_options = PRODUCT_MAP.get(best_cat, [best_cat.title()])
                    if len(detail_options) > 1:
                        options = [best_cat.title()]
                        show_suboptions = False
                    else:
                        options = detail_options
                        show_suboptions = True
                    selected_idx = 0
                    detected_shown_at = time.time()
                    print(f"\nDetected: {best_cat}  ({best_conf:.2f})")
                    if show_suboptions:
                        for i, o in enumerate(options, 1):
                            p = prices.get(o)
                            print(f"  [{i}] {o}  {'$'+str(p)+'/kg' if p else ''}")
                    else:
                        print(f"  [{1}] {options[0]}")
                        print("  Press the item to expand its variants")

        # Auto-hide detection after configured seconds
        if detected_shown_at is not None:
            try:
                if time.time() - detected_shown_at >= DETECTION_DISPLAY_SECONDS:
                    detected_category = None
                    options = []
                    detail_options = []
                    show_suboptions = False
                    selected_idx = None
                    last_cat = None
                    stable_frames = 0
                    detected_shown_at = None
            except Exception:
                # safe-guard: ignore timing errors
                detected_shown_at = None

        cam_view = draw_camera_view(frame, detections)
        panel, buttons = draw_panel(detected_category, options, selected_idx, cart, prices, show_suboptions, running)

        ph  = panel.shape[0]
        cw  = int(frame.shape[1] * (ph / frame.shape[0]))
        cam = cv2.resize(cam_view, (cw, ph))
        display = np.hstack([cam, panel])

        winname = "POS Vegetable Detection  |  Q=quit"
        cv2.namedWindow(winname)
        cv2.setMouseCallback(winname, mouse_callback)
        cv2.imshow(winname, display)

        # publish current sizes for click coordinate translation
        CLICK_INFO["cam_w"] = cam.shape[1]
        CLICK_INFO["panel_w"] = panel.shape[1]

        key = cv2.waitKey(1) & 0xFF

        # Start detection: clears screen and begins inference
        if key in (ord('s'), ord('S')):
            if not running:
                running = True
                detected_category = None
                options = []
                detail_options = []
                show_suboptions = False
                selected_idx = None
                last_cat = None
                stable_frames = 0
                detected_shown_at = None
                print("Detection STARTED; screen cleared")
            else:
                print("Detection already running")

        # Stop detection: pause inference (display may remain until cleared)
        if key in (ord('t'), ord('T')):
            if running:
                running = False
                print("Detection STOPPED")
            else:
                print("Detection already stopped")

        if detected_category and options:
            for i in range(min(9, len(options))):
                if key == ord(str(i + 1)):
                    if len(detail_options) > 1 and not show_suboptions and options[i].lower() == detected_category:
                        options = detail_options
                        show_suboptions = True
                        selected_idx = 0
                        detected_shown_at = time.time()
                        print(f"  Expanded {detected_category} to show variants")
                    else:
                        selected_idx = i
                        detected_shown_at = time.time()
                        print(f"  Selected: {options[i]}")

        if key == ord(' ') and selected_idx is not None and show_suboptions:
            chosen = options[selected_idx]
            price  = prices.get(chosen)
            cart.append((chosen, price))
            print(f"  CONFIRMED: {chosen}  ${price:.2f}/kg  |  Total: ${sum(p for _,p in cart if p):.2f}")
            detected_category = None
            options           = []
            detail_options    = []
            show_suboptions   = False
            detected_shown_at = None
            selected_idx      = None
            last_cat          = None
            stable_frames     = 0

        # Clear detection immediately (restart detection flow)
        if key in (ord('c'), ord('C')):
            detected_category = None
            options = []
            detail_options = []
            show_suboptions = False
            selected_idx = None
            last_cat = None
            stable_frames = 0
            detected_shown_at = None
            print("Cleared detection; resuming live detection")

        # Handle mouse clicks on the panel (if any)
        click = CLICK_INFO.get("click")
        if click is not None:
            cx, cy = click
            CLICK_INFO["click"] = None
            cam_w = CLICK_INFO.get("cam_w") or 0
            # only handle clicks in the panel area
            if cx >= cam_w:
                px = cx - cam_w
                py = cy
                # check each button rect
                def inside(r, x, y):
                    x1, y1, x2, y2 = r
                    return x >= x1 and x <= x2 and y >= y1 and y <= y2

                if inside(buttons["start"], px, py):
                    if not running:
                        running = True
                        detected_category = None
                        options = []
                        detail_options = []
                        show_suboptions = False
                        selected_idx = None
                        last_cat = None
                        stable_frames = 0
                        detected_shown_at = None
                        print("Detection STARTED via UI; screen cleared")
                    else:
                        print("Detection already running")

                elif inside(buttons["stop"], px, py):
                    if running:
                        running = False
                        print("Detection STOPPED via UI")
                    else:
                        print("Detection already stopped")

                elif inside(buttons["clear"], px, py):
                    detected_category = None
                    options = []
                    detail_options = []
                    show_suboptions = False
                    selected_idx = None
                    last_cat = None
                    stable_frames = 0
                    detected_shown_at = None
                    print("Cleared detection via UI; resuming live detection")

        if key in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*45}")
    print("Session ended.")
    if cart:
        print(f"Items scanned ({len(cart)}):")
        for item, price in cart:
            p = f"${price:.2f}/kg" if price else "no price"
            print(f"  - {item}  ({p})")
        total = sum(p for _, p in cart if p)
        print(f"\nGRAND TOTAL: ${total:.2f}")


if __name__ == "__main__":
    main()
