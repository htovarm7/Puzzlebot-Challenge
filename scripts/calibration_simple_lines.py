"""
Interactive parameter tuner for the simpler Hough-based line detector.
"""

import cv2 as cv
import numpy as np
import sys
import os


# ─── Defaults ─────────────────────────────────────────────────────────
DEFAULTS = {
    # ROI
    "roi_top": 71,
    "blur": 11,
    "block_size": 61,
    "C": 4,
    "morph": 10,
    "canny_low": 384,
    "canny_high": 385,
    "hough_thr": 85,
    "min_len": 66,
    "max_gap": 170
}

C_OFFSET = 30   # trackbars are unsigned; show C as (slider - C_OFFSET)


# ─── Windows ──────────────────────────────────────────────────────────
WIN_PRE  = "Preprocessing"
WIN_LINE = "Lines"
WIN_DBG  = "Debug"
WIN_BIN  = "Binary (ROI)"
WIN_EDG  = "Edges"


def nothing(_):
    pass


def build_windows():
    cv.namedWindow(WIN_PRE,  cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_LINE, cv.WINDOW_NORMAL)
    cv.resizeWindow(WIN_PRE,  420, 260)
    cv.resizeWindow(WIN_LINE, 420, 220)

    # Preprocessing window
    cv.createTrackbar("ROI top %",   WIN_PRE, DEFAULTS["roi_top"],     99, nothing)
    cv.createTrackbar("blur (odd)",  WIN_PRE, DEFAULTS["blur"],        21, nothing)
    cv.createTrackbar("block (odd)", WIN_PRE, DEFAULTS["block_size"],  61, nothing)
    cv.createTrackbar("C (+offset)", WIN_PRE, DEFAULTS["C"] + C_OFFSET, 60, nothing)
    cv.createTrackbar("morph",       WIN_PRE, DEFAULTS["morph"],       15, nothing)

    # Lines window
    cv.createTrackbar("canny low",   WIN_LINE, DEFAULTS["canny_low"],  500, nothing)
    cv.createTrackbar("canny high",  WIN_LINE, DEFAULTS["canny_high"], 500, nothing)
    cv.createTrackbar("hough thr",   WIN_LINE, DEFAULTS["hough_thr"],  300, nothing)
    cv.createTrackbar("min length",  WIN_LINE, DEFAULTS["min_len"],    300, nothing)
    cv.createTrackbar("max gap",     WIN_LINE, DEFAULTS["max_gap"],    200, nothing)


def reset_windows():
    for win, name, key in [
        (WIN_PRE,  "ROI top %",   "roi_top"),
        (WIN_PRE,  "blur (odd)",  "blur"),
        (WIN_PRE,  "block (odd)", "block_size"),
        (WIN_PRE,  "C (+offset)", None),       # special — apply offset
        (WIN_PRE,  "morph",       "morph"),
        (WIN_LINE, "canny low",   "canny_low"),
        (WIN_LINE, "canny high",  "canny_high"),
        (WIN_LINE, "hough thr",   "hough_thr"),
        (WIN_LINE, "min length",  "min_len"),
        (WIN_LINE, "max gap",     "max_gap"),
    ]:
        if name == "C (+offset)":
            cv.setTrackbarPos(name, win, DEFAULTS["C"] + C_OFFSET)
        else:
            cv.setTrackbarPos(name, win, DEFAULTS[key])


def read_params():
    blur = cv.getTrackbarPos("blur (odd)", WIN_PRE)
    blur = max(1, blur | 1)                      # force odd
    block = cv.getTrackbarPos("block (odd)", WIN_PRE)
    block = max(3, block | 1)                    # force odd, >= 3

    p = {
        "roi_top":    cv.getTrackbarPos("ROI top %", WIN_PRE) / 100.0,
        "blur":       blur,
        "block_size": block,
        "C":          cv.getTrackbarPos("C (+offset)", WIN_PRE) - C_OFFSET,
        "morph":      max(1, cv.getTrackbarPos("morph", WIN_PRE)),
        "canny_low":  cv.getTrackbarPos("canny low",  WIN_LINE),
        "canny_high": cv.getTrackbarPos("canny high", WIN_LINE),
        "hough_thr":  max(1, cv.getTrackbarPos("hough thr",  WIN_LINE)),
        "min_len":    max(1, cv.getTrackbarPos("min length", WIN_LINE)),
        "max_gap":    cv.getTrackbarPos("max gap", WIN_LINE),
    }
    # Sanity: canny low < high
    if p["canny_low"] >= p["canny_high"]:
        p["canny_high"] = p["canny_low"] + 1
    return p


def save_params(p):
    out = "hough_params.txt"
    with open(out, "w") as f:
        for k, v in p.items():
            f.write(f"{k} = {v}\n")
    print(f"[saved] {os.path.abspath(out)}")


# ─── Detection (your original detect_lines, parameterized) ────────────
def detect_lines(frame, p):
    height, width = frame.shape[:2]
    y_top = int(height * p["roi_top"])

    # 1. ROI mask (same shape as original — trapezoid would also work,
    #    but a rectangle matches the slider semantics simply).
    roi_vertices = np.array([[
        (0, height),
        (0, y_top),
        (width, y_top),
        (width, height),
    ]], dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv.fillPoly(mask, roi_vertices, 255)

    # 2. Grayscale + ROI
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_roi = cv.bitwise_and(gray, gray, mask=mask)

    # 3. Blur
    if p["blur"] >= 3:
        blurred = cv.GaussianBlur(gray_roi, (p["blur"], p["blur"]), 0)
    else:
        blurred = gray_roi

    # 4. Adaptive threshold (lines are dark → INV)
    binary = cv.adaptiveThreshold(
        blurred, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        blockSize=p["block_size"],
        C=p["C"],
    )
    # Re-mask: adaptiveThreshold lights up the masked-out region too
    binary = cv.bitwise_and(binary, binary, mask=mask)

    # 5. Morphology
    if p["morph"] >= 2:
        kernel = np.ones((p["morph"], p["morph"]), np.uint8)
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN,  kernel)

    # 6. Canny
    edges = cv.Canny(binary, p["canny_low"], p["canny_high"])

    # 7. Probabilistic Hough
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=p["hough_thr"],
        minLineLength=p["min_len"],
        maxLineGap=p["max_gap"],
    )

    return lines, binary, edges, y_top


def draw_debug(frame, lines, y_top, p):
    debug = frame.copy()
    # ROI line
    cv.line(debug, (0, y_top), (debug.shape[1], y_top), (255, 200, 0), 1)

    n = 0
    if lines is not None:
        n = len(lines)
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            cv.line(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    hud = (f"lines={n}  block={p['block_size']}  C={p['C']}  "
           f"canny={p['canny_low']}/{p['canny_high']}  "
           f"hough thr={p['hough_thr']} len={p['min_len']} gap={p['max_gap']}")
    cv.putText(debug, hud, (10, 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return debug


# ─── Source handling ──────────────────────────────────────────────────
def open_source(arg):
    if arg is None:
        cap = cv.VideoCapture(0)
        return (lambda: cap.read()[1]), False, cap

    if not os.path.exists(arg):
        print(f"error: {arg} not found")
        sys.exit(1)

    ext = os.path.splitext(arg)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        img = cv.imread(arg)
        if img is None:
            print(f"error: cv.imread failed on {arg}")
            sys.exit(1)
        return (lambda: img.copy()), True, None
    else:
        cap = cv.VideoCapture(arg)
        return (lambda: cap.read()[1]), False, cap


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    read_frame, is_image, cap = open_source(arg)

    build_windows()
    cv.namedWindow(WIN_DBG, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_BIN, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_EDG, cv.WINDOW_NORMAL)

    paused = False
    last_frame = None
    print("Keys: [q] quit  [s] save  [r] reset  [space] pause")

    while True:
        if not paused or last_frame is None:
            frame = read_frame()
            if frame is None:
                if is_image:
                    break
                if cap is not None:
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            last_frame = frame

        p = read_params()
        lines, binary, edges, y_top = detect_lines(last_frame.copy(), p)
        debug = draw_debug(last_frame, lines, y_top, p)

        cv.imshow(WIN_DBG, debug)
        cv.imshow(WIN_BIN, binary)
        cv.imshow(WIN_EDG, edges)

        key = cv.waitKey(1 if not is_image else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_params(p)
        elif key == ord('r'):
            reset_windows()
        elif key == ord(' '):
            paused = not paused

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()