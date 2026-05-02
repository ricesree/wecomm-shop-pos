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
import numpy as np
import openpyxl
from ultralytics import YOLO

# --- Config ---------------------------------------------------------------

MODEL_PATH   = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\best_categories.pt"
PRICES_PATH  = r"c:\Users\sreet\Desktop\TUNE-DATAPOS\prices.xlsx"
CONF_THRESH  = 0.55
CAMERA_ID    = 0
STABLE_REQ   = 8

# --- Product Map: YOLO category → sellable items -------------------------

PRODUCT_MAP = {
    "banana":      ["Banana", "BURRO BANANA", "Banana Flower"],
    "beans":       ["Beans Regular", "Long Green Beans", "String Beans", "FLAT VELOR"],
    "chilli":      ["FLORIDA LONG CHILLI", "Thai Chilli", "Bell Pepper"],
    "coconut":     ["Coconut"],
    "dasakai":     ["Dasakai"],
    "eggplant":    ["Indian Eggplant", "Chinese Eggplant", "Chinese Green Eggplant", "Thai Eggplant"],
    "fruit":       ["Guava", "Papaya", "FRESH CHIKKU", "Lemon", "Chayote"],
    "gourd":       ["Pumpkin", "Snake Gourd", "Ridge Gourd", "Bitter Gourd", "Tindora", "Squash"],
    "ladyfinger":  ["Okra / Ladies Finger"],
    "ladystickers":["Lady Stickers"],
    "leafy":       ["Cabbage", "Cauliflower", "Mint", "Cilantro", "Curry Leaves", "Leaves", "Pan Leaves"],
    "onion":       ["Red Onions", "White Onions"],
    "root":        ["Potato", "Sweet Potato", "Beetroot", "Radish", "Ginger", "Garlic"],
    "special":     ["Boxed Sweets", "Home made snacks", "POLI", "Roti", "Mums", "Pearl"],
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


def draw_panel(detected_category, options, selected_idx, cart, prices):
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

        if selected_idx is not None:
            cv2.rectangle(panel, (12, H - 170), (W - 12, H - 138), (10, 170, 70), -1)
            cv2.putText(panel, "  SPACE = Add to cart", (18, H - 148),
                        FONT, 0.58, (255, 255, 255), 1)
    else:
        cv2.putText(panel, "Waiting for", (30, 80),  FONT, 0.9, (120, 120, 120), 2)
        cv2.putText(panel, "detection...", (30, 120), FONT, 0.9, (120, 120, 120), 2)

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

    return panel


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
    selected_idx      = None
    cart              = []   # list of (item_name, price)
    stable_frames     = 0
    last_cat          = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, conf=CONF_THRESH, verbose=False)[0]
        detections = []
        best_cat   = None
        best_conf  = 0.0

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
                options = PRODUCT_MAP.get(best_cat, [best_cat.title()])
                selected_idx = 0 if len(options) == 1 else None
                print(f"\nDetected: {best_cat}  ({best_conf:.2f})")
                for i, o in enumerate(options, 1):
                    p = prices.get(o)
                    print(f"  [{i}] {o}  {'$'+str(p)+'/kg' if p else ''}")

        cam_view = draw_camera_view(frame, detections)
        panel    = draw_panel(detected_category, options, selected_idx, cart, prices)

        ph  = panel.shape[0]
        cw  = int(frame.shape[1] * (ph / frame.shape[0]))
        cam = cv2.resize(cam_view, (cw, ph))
        display = np.hstack([cam, panel])

        cv2.imshow("POS Vegetable Detection  |  Q=quit", display)

        key = cv2.waitKey(1) & 0xFF

        if detected_category and options:
            for i in range(min(9, len(options))):
                if key == ord(str(i + 1)):
                    selected_idx = i
                    print(f"  Selected: {options[i]}")

        if key == ord(' ') and selected_idx is not None:
            chosen = options[selected_idx]
            price  = prices.get(chosen)
            cart.append((chosen, price))
            print(f"  CONFIRMED: {chosen}  ${price:.2f}/kg  |  Total: ${sum(p for _,p in cart if p):.2f}")
            detected_category = None
            options           = []
            selected_idx      = None
            last_cat          = None
            stable_frames     = 0

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
