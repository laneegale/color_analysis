import cv2
import numpy as np
import csv
import math
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------- CONFIG ----------
ROI_SIZE = 30

# Default starting positions (user can drag to adjust)
START_X = 330
START_Y = 80
DX = 214
DY = 168

CA_X = 80
CA_Y = 80

ROWS = ["A", "B", "C", "D"]
COLS = ["1", "2", "3", "4", "5", "6"]

# Table layout tuning (reduced gaps + no overlap)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS = 1
LINE_H = 20          # vertical spacing per row
HEADER_GAP = 18      # gap after each header line
TABLE_GAP = 16       # gap between tables

WINDOW_NAME = "RGB + HEX + LAB Sampler"
# ----------------------------

# Image state (loaded via button)
img_bgr = None
img_rgb = None
img_h = 0
img_w = 0

# Sampling squares (centers)
squares = {"CA": [CA_X, CA_Y]}
for r_i, r in enumerate(ROWS):
    for c_i, c in enumerate(COLS):
        squares[r + c] = [START_X + c_i * DX, START_Y + r_i * DY]

selected_label = None

# Data
rows_rgbhex = []
rows_lab = []
rows_de = []

# Reference LAB
ref_lab = {}
ref_path = None

# Current image path (for display)
image_path = None


def clamp_u8(v: int) -> int:
    return max(0, min(255, int(v)))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


# Precise RGB -> LAB conversion (2 decimals), D65/2°
def rgb_to_lab_cie_2dec(rgb_255):
    R, G, B = [float(x) / 255.0 for x in rgb_255]

    def gamma_inv(c):
        return (c / 12.92) if c <= 0.04045 else (((c + 0.055) / 1.055) ** 2.4)

    R = gamma_inv(R)
    G = gamma_inv(G)
    B = gamma_inv(B)

    X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375
    Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
    Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041

    X /= 0.95047
    Y /= 1.0
    Z /= 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fX, fY, fZ = f(X), f(Y), f(Z)

    L = round(116 * fY - 16, 2)
    a = round(500 * (fX - fY), 2)
    b = round(200 * (fY - fZ), 2)
    return L, a, b


# CIEDE2000 (Delta E 2000)
def delta_e_ciede2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0):
    L1, a1, b1 = [float(x) for x in lab1]
    L2, a2, b2 = [float(x) for x in lab2]

    C1 = math.sqrt(a1 * a1 + b1 * b1)
    C2 = math.sqrt(a2 * a2 + b2 * b2)
    C_bar = (C1 + C2) / 2.0

    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - math.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = math.sqrt(a1p * a1p + b1 * b1)
    C2p = math.sqrt(a2p * a2p + b2 * b2)

    def hp(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        ang = math.degrees(math.atan2(b, ap))
        return ang + 360.0 if ang < 0 else ang

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if dh > 180:
            dh -= 360
        elif dh < -180:
            dh += 360
        dhp = dh

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    L_bar_p = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        h_bar_p = h1p + h2p
    else:
        hsum = h1p + h2p
        hdiff = abs(h1p - h2p)
        if hdiff > 180:
            h_bar_p = (hsum + 360.0) / 2.0 if hsum < 360.0 else (hsum - 360.0) / 2.0
        else:
            h_bar_p = hsum / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )

    dtheta = 30.0 * math.exp(-((h_bar_p - 275.0) / 25.0) ** 2)
    Rc = 2.0 * math.sqrt((C_bar_p ** 7) / (C_bar_p ** 7 + 25.0 ** 7))
    Sl = 1.0 + (0.015 * (L_bar_p - 50.0) ** 2) / math.sqrt(20.0 + (L_bar_p - 50.0) ** 2)
    Sc = 1.0 + 0.045 * C_bar_p
    Sh = 1.0 + 0.015 * C_bar_p * T
    Rt = -math.sin(math.radians(2.0 * dtheta)) * Rc

    dE = math.sqrt(
        (dLp / (kL * Sl)) ** 2
        + (dCp / (kC * Sc)) ** 2
        + (dHp / (kH * Sh)) ** 2
        + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh))
    )
    return round(dE, 4)


def ensure_image_loaded():
    return img_bgr is not None and img_rgb is not None and img_h > 0 and img_w > 0


def sample_roi_mean_rgb(cx, cy):
    if not ensure_image_loaded():
        return (0, 0, 0), (0, 0, 0, 0)

    x1 = max(0, cx - ROI_SIZE // 2)
    y1 = max(0, cy - ROI_SIZE // 2)
    x2 = min(img_w, cx + ROI_SIZE // 2)
    y2 = min(img_h, cy + ROI_SIZE // 2)

    roi = img_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return (0, 0, 0), (x1, y1, x2, y2)

    mean_rgb = roi.mean(axis=(0, 1))
    r = clamp_u8(np.ceil(mean_rgb[0]))
    g = clamp_u8(np.ceil(mean_rgb[1]))
    b = clamp_u8(np.ceil(mean_rgb[2]))
    return (r, g, b), (x1, y1, x2, y2)


def compute_tables():
    global rows_rgbhex, rows_lab, rows_de
    rows_rgbhex, rows_lab, rows_de = [], [], []

    if not ensure_image_loaded():
        return

    for label, (cx, cy) in squares.items():
        (r, g, b), _ = sample_roi_mean_rgb(cx, cy)
        hx = rgb_to_hex(r, g, b)
        L, a, b_lab = rgb_to_lab_cie_2dec((r, g, b))

        rows_rgbhex.append((label, hx, r, g, b))
        rows_lab.append((label, L, a, b_lab))

        if label in ref_lab:
            rL, ra, rb = ref_lab[label]
            de = delta_e_ciede2000((L, a, b_lab), (rL, ra, rb))
            rows_de.append((label, rL, ra, rb, de))
        else:
            rows_de.append((label, None, None, None, None))


def average_delta_e():
    vals = [de for (_, _, _, _, de) in rows_de if de is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def choose_ncols(min_col_w: int) -> int:
    return max(1, img_w // min_col_w) if img_w > 0 else 1


def draw_table(canvas_img, title, lines, start_y, min_col_w):
    cv2.putText(canvas_img, title, (10, start_y), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y0 = start_y + HEADER_GAP

    ncols = choose_ncols(min_col_w)
    col_w = max(1, img_w // ncols)
    n = len(lines)
    rows_per_col = (n + ncols - 1) // ncols if n > 0 else 1

    idx = 0
    for col in range(ncols):
        x = 10 + col * col_w
        y = y0
        for _ in range(rows_per_col):
            if idx >= n:
                break
            cv2.putText(canvas_img, lines[idx], (x, y), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA)
            y += LINE_H
            idx += 1

    table_height = HEADER_GAP + rows_per_col * LINE_H
    return start_y + table_height


def draw_placeholder():
    ph = np.zeros((420, 900, 3), dtype=np.uint8)
    cv2.putText(ph, "No image loaded.", (20, 80), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ph, "Use 'Load Image' to select a .tif/.png/.jpg.", (20, 140), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(ph, "Then drag squares to align patches.", (20, 190), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, ph)


def draw_canvas():
    if not ensure_image_loaded():
        draw_placeholder()
        return

    compute_tables()

    t1_lines = [f"{lab}: {hx}  R={r} G={g} B={b}" for (lab, hx, r, g, b) in rows_rgbhex]
    t2_lines = [f"{lab}: L={L} a={a} b={b_}" for (lab, L, a, b_) in rows_lab]
    t3_lines = []
    for (lab, rL, ra, rb, de) in rows_de:
        if de is None:
            t3_lines.append(f"{lab}: no ref")
        else:
            t3_lines.append(f"{lab}: dE00={de:.4f}")

    def table_required_height(num_lines, min_col_w):
        ncols = max(1, img_w // min_col_w)
        rows_per_col = (num_lines + ncols - 1) // ncols if num_lines > 0 else 1
        return HEADER_GAP + rows_per_col * LINE_H

    h1 = table_required_height(len(t1_lines), min_col_w=360)
    h2 = table_required_height(len(t2_lines), min_col_w=320)
    h3 = table_required_height(len(t3_lines), min_col_w=260)

    avg_line_space = LINE_H + 14
    total_tables_h = (12 + h1 + TABLE_GAP + h2 + TABLE_GAP + h3 + avg_line_space + 26)
    canvas_h = img_h + total_tables_h

    canvas_img = np.zeros((canvas_h, img_w, 3), dtype=np.uint8)
    canvas_img[0:img_h, 0:img_w] = img_bgr.copy()

    # Draw squares on image
    for label, (cx, cy) in squares.items():
        _, (x1, y1, x2, y2) = sample_roi_mean_rgb(cx, cy)
        cv2.rectangle(canvas_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas_img, label, (x1, max(0, y1 - 5)), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Tables
    y = img_h + 18

    img_label = f"Image: {image_path}" if image_path else "Image: (loaded)"
    cv2.putText(canvas_img, img_label, (10, y), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    y += 18

    y = draw_table(canvas_img, "Table 1: Hex + RGB (rounded UP)", t1_lines, y, min_col_w=360) + TABLE_GAP
    y = draw_table(canvas_img, "Table 2: LAB (from rounded RGB)", t2_lines, y, min_col_w=320) + TABLE_GAP

    ref_label = f"Reference: {ref_path}" if ref_path else "Reference: (not loaded)"
    y = draw_table(canvas_img, f"Table 3: DeltaE2000 vs Reference | {ref_label}", t3_lines, y, min_col_w=260)

    # Average ΔE line
    avg_de = average_delta_e()
    y += LINE_H + 10
    if avg_de is None:
        avg_text = "Average ΔE2000 (all patches): (no reference data)"
    else:
        avg_text = f"Average ΔE2000 (all patches): {avg_de:.2f}"
    cv2.putText(canvas_img, avg_text, (10, y), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, canvas_img)


def on_mouse(event, x, y, flags, param):
    global selected_label
    if not ensure_image_loaded():
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        for label, (cx, cy) in squares.items():
            if abs(x - cx) <= ROI_SIZE // 2 and abs(y - cy) <= ROI_SIZE // 2:
                selected_label = label
                break
    elif event == cv2.EVENT_LBUTTONUP:
        selected_label = None
    elif event == cv2.EVENT_MOUSEMOVE and selected_label is not None:
        squares[selected_label] = [x, y]
        draw_canvas()


def load_image():
    global img_bgr, img_rgb, img_h, img_w, image_path

    path = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*"),
        ]
    )
    if not path:
        return

    bgr = cv2.imread(path)
    if bgr is None:
        messagebox.showerror("Load Image Failed", "Could not read the selected image.")
        return

    img_bgr = bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img_rgb.shape
    image_path = path

    draw_canvas()


def load_reference_lab():
    global ref_lab, ref_path

    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not path:
        return

    loaded = {}
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("No headers found.")

            headers = {h.replace("\ufeff", "").strip().lower(): h for h in reader.fieldnames}

            if "label" not in headers:
                raise ValueError(f"CSV must contain 'Label'. Found headers: {reader.fieldnames}")

            l_key = headers.get("l")
            a_key = headers.get("a")
            b_key = headers.get("b")
            if not (l_key and a_key and b_key):
                raise ValueError("CSV must contain columns: L, a, b")

            for row in reader:
                label = (row[headers["label"]] or "").strip()
                if not label:
                    continue
                L = float(str(row[l_key]).strip())
                a = float(str(row[a_key]).strip())
                b = float(str(row[b_key]).strip())
                loaded[label] = (round(L, 2), round(a, 2), round(b, 2))

    except Exception as e:
        messagebox.showerror("Reference Load Failed", str(e))
        return

    ref_lab = loaded
    ref_path = path
    draw_canvas()


def save_all_csv():
    if not ensure_image_loaded():
        messagebox.showwarning("Export", "Load an image first.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    compute_tables()
    avg_de = average_delta_e()

    de_map = {label: (rL, ra, rb, de) for (label, rL, ra, rb, de) in rows_de}
    lab_map = {label: (L, a, b) for (label, L, a, b) in rows_lab}

    with open(file_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Label",
            "Hex", "R", "G", "B",
            "L", "a", "b",
            "Ref_L", "Ref_a", "Ref_b",
            "DeltaE2000"
        ])

        for (label, hx, r, g, b) in rows_rgbhex:
            L, a, b_lab = lab_map[label]
            rL, ra, rb, de = de_map.get(label, (None, None, None, None))
            w.writerow([label, hx, r, g, b, L, a, b_lab, rL, ra, rb, de])

        # Average ΔE2000 summary row (exported)
        w.writerow([])
        w.writerow(["AVERAGE_DELTAE2000", f"{avg_de:.2f}" if avg_de is not None else ""])

    print(f"Saved: {file_path}")


# ---------- GUI ----------
root = tk.Tk()
root.title("Sampler Export")
root.geometry("300x190")

btn_img = tk.Button(root, text="Load Image", command=load_image)
btn_img.pack(pady=(16, 6))

btn_ref = tk.Button(root, text="Load Reference LAB CSV", command=load_reference_lab)
btn_ref.pack(pady=6)

btn_save = tk.Button(root, text="Save Hex+RGB+LAB+DE CSV", command=save_all_csv)
btn_save.pack(pady=6)

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

draw_placeholder()
root.mainloop()
cv2.destroyAllWindows()
