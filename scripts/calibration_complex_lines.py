"""Interactive parameter tuner for contour-based line detection."""

import cv2 as cv
import numpy as np
import sys
import os


# Defaults
DEFAULTS = {
    "T_init": 185,
    "T_min": 127,
    "T_max": 222,
    "dark_min_x10": 20,
    "dark_max_x10": 24  ,
    "roi_top_x100": 68,
    "min_area": 3753,
    "blur": 21,
    "morph": 9,
    "turn_angle": 36,
    "shift_max": 130

}


# Tuner UI
WIN_CTRL  = "Controls"
WIN_DEBUG = "Debug"
WIN_BIN   = "Binary (ROI)"


def nothing(_):
    pass


def build_window():
    cv.namedWindow(WIN_CTRL, cv.WINDOW_NORMAL)
    cv.resizeWindow(WIN_CTRL, 460, 520)

    cv.createTrackbar("T init",        WIN_CTRL, DEFAULTS["T_init"],       255, nothing)
    cv.createTrackbar("T min",         WIN_CTRL, DEFAULTS["T_min"],        255, nothing)
    cv.createTrackbar("T max",         WIN_CTRL, DEFAULTS["T_max"],        255, nothing)
    cv.createTrackbar("dark% min x10", WIN_CTRL, DEFAULTS["dark_min_x10"], 500, nothing)
    cv.createTrackbar("dark% max x10", WIN_CTRL, DEFAULTS["dark_max_x10"], 500, nothing)
    cv.createTrackbar("ROI top %",     WIN_CTRL, DEFAULTS["roi_top_x100"],  99, nothing)
    cv.createTrackbar("min area",      WIN_CTRL, DEFAULTS["min_area"],   5000, nothing)
    cv.createTrackbar("blur (odd)",    WIN_CTRL, DEFAULTS["blur"],         21, nothing)
    cv.createTrackbar("morph kernel",  WIN_CTRL, DEFAULTS["morph"],        15, nothing)
    cv.createTrackbar("turn angle",    WIN_CTRL, DEFAULTS["turn_angle"],   90, nothing)
    cv.createTrackbar("shift max px",  WIN_CTRL, DEFAULTS["shift_max"],   200, nothing)


def reset_window():
    for name, key in [
        ("T init", "T_init"), ("T min", "T_min"), ("T max", "T_max"),
        ("dark% min x10", "dark_min_x10"), ("dark% max x10", "dark_max_x10"),
        ("ROI top %", "roi_top_x100"), ("min area", "min_area"),
        ("blur (odd)", "blur"), ("morph kernel", "morph"),
        ("turn angle", "turn_angle"), ("shift max px", "shift_max"),
    ]:
        cv.setTrackbarPos(name, WIN_CTRL, DEFAULTS[key])


def read_params():
    p = {
        "T_init":     cv.getTrackbarPos("T init",       WIN_CTRL),
        "T_min":      cv.getTrackbarPos("T min",        WIN_CTRL),
        "T_max":      cv.getTrackbarPos("T max",        WIN_CTRL),
        "dark_min":   cv.getTrackbarPos("dark% min x10", WIN_CTRL) / 10.0,
        "dark_max":   cv.getTrackbarPos("dark% max x10", WIN_CTRL) / 10.0,
        "roi_top":    cv.getTrackbarPos("ROI top %",    WIN_CTRL) / 100.0,
        "min_area":   cv.getTrackbarPos("min area",     WIN_CTRL),
        "blur":       max(1, cv.getTrackbarPos("blur (odd)", WIN_CTRL) | 1),  # force odd
        "morph":      max(1, cv.getTrackbarPos("morph kernel", WIN_CTRL)),
        "turn_angle": cv.getTrackbarPos("turn angle",   WIN_CTRL),
        "shift_max":  cv.getTrackbarPos("shift max px", WIN_CTRL),
    }
    # Sanity: ensure min < max
    if p["T_min"] >= p["T_max"]:
        p["T_max"] = p["T_min"] + 1
    if p["dark_min"] >= p["dark_max"]:
        p["dark_max"] = p["dark_min"] + 0.1
    return p


def save_params(p):
    out = "params.txt"
    with open(out, "w") as f:
        for k, v in p.items():
            f.write(f"{k} = {v}\n")
    print(f"[saved] {os.path.abspath(out)}")


# Detection
_T_state = DEFAULTS["T_init"]  # threshold persists across frames


def crop_roi(img, roi_top):
    h = img.shape[0]
    y1 = int(h * roi_top)
    return img[y1:, :], y1


def balance_pic(gray, p):
    """Iteratively pick T so dark-pixel % in ROI lands in [dark_min, dark_max]."""
    global _T_state
    T = _T_state
    direction = 0
    for _ in range(10):
        _, binary = cv.threshold(gray, T, 255, cv.THRESH_BINARY_INV)
        crop, _ = crop_roi(binary, p["roi_top"])
        area = crop.shape[0] * crop.shape[1]
        if area == 0:
            return None, T
        perc = 100.0 * cv.countNonZero(crop) / area

        if perc > p["dark_max"]:
            if T <= p["T_min"]:
                _T_state = T
                return crop, T
            if direction == 1:
                _T_state = T
                return crop, T
            T -= 10
            direction = -1
        elif perc < p["dark_min"]:
            if T >= p["T_max"]:
                _T_state = T
                return crop, T
            if direction == -1:
                _T_state = T
                return crop, T
            T += 10
            direction = 1
        else:
            _T_state = T
            return crop, T
    _T_state = T
    return None, T


def detect(frame, p):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if p["blur"] >= 3:
        gray = cv.GaussianBlur(gray, (p["blur"], p["blur"]), 0)

    binary_roi, T_used = balance_pic(gray, p)
    debug = frame.copy()

    # Always draw ROI line
    h = frame.shape[0]
    y_off = int(h * p["roi_top"])
    cv.line(debug, (0, y_off), (frame.shape[1], y_off), (255, 200, 0), 1)

    if binary_roi is None:
        cv.putText(debug, "no balanced threshold", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        blank = np.zeros((100, 300), np.uint8)
        return debug, blank, None, None, T_used

    # Morphological cleanup
    k = p["morph"]
    if k >= 2:
        kernel = np.ones((k, k), np.uint8)
        binary_roi = cv.morphologyEx(binary_roi, cv.MORPH_OPEN, kernel)
        binary_roi = cv.morphologyEx(binary_roi, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(binary_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= p["min_area"]]

    angle = shift = None
    if contours:
        line = max(contours, key=cv.contourArea)
        rect = cv.minAreaRect(line)
        (cx, cy), _, _ = rect
        box = cv.boxPoints(rect)
        box = box[np.argsort(box[:, 1])]
        top_mid    = ((box[0] + box[1]) / 2).astype(int)
        bottom_mid = ((box[2] + box[3]) / 2).astype(int)

        dx = float(bottom_mid[0] - top_mid[0])
        dy = float(bottom_mid[1] - top_mid[1])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle < 0:
            angle += 180

        roi_center_x = binary_roi.shape[1] // 2
        shift = int(cx - roi_center_x)

        # Draw contour, oriented box, centerline on the debug frame
        cv.drawContours(debug, [line + [0, y_off]], -1, (0, 255, 0), 2)
        box_shifted = (box + [0, y_off]).astype(int)
        cv.drawContours(debug, [box_shifted], 0, (255, 0, 255), 1)
        p1 = (top_mid[0],    top_mid[1] + y_off)
        p2 = (bottom_mid[0], bottom_mid[1] + y_off)
        cv.line(debug, p1, p2, (0, 0, 255), 3)

        # Center reference line
        fx = frame.shape[1] // 2
        cv.line(debug, (fx, y_off), (fx, frame.shape[0]), (0, 255, 255), 1)

    # HUD
    if angle is not None:
        cv.putText(debug, f"T={T_used}  angle={angle:5.1f}  shift={shift:+d}",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv.putText(debug, f"T={T_used}  no contour", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    return debug, binary_roi, angle, shift, T_used


# Main
def open_source(arg):
    """Return (read_fn, is_image, cap). read_fn() returns a frame or None."""
    if arg is None:
        cap = cv.VideoCapture(0)
        return (lambda: cap.read()[1]), False, cap

    if not os.path.exists(arg):
        print(f"error: {arg} not found"); sys.exit(1)

    ext = os.path.splitext(arg)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        img = cv.imread(arg)
        if img is None:
            print(f"error: cv.imread failed on {arg}"); sys.exit(1)
        return (lambda: img.copy()), True, None
    else:
        cap = cv.VideoCapture(arg)
        return (lambda: cap.read()[1]), False, cap


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    read_frame, is_image, cap = open_source(arg)

    build_window()
    cv.namedWindow(WIN_DEBUG, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_BIN,   cv.WINDOW_NORMAL)

    paused = False
    last_frame = None

    print("Keys: [q] quit  [s] save params  [r] reset  [space] pause")

    while True:
        if not paused or last_frame is None:
            frame = read_frame()
            if frame is None:
                if is_image:
                    break
                # video ended, loop it
                if cap is not None:
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            last_frame = frame

        p = read_params()
        debug, binary, angle, shift, T_used = detect(last_frame.copy(), p)

        cv.imshow(WIN_DEBUG, debug)
        cv.imshow(WIN_BIN, binary)

        key = cv.waitKey(1 if not is_image else 30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_params(p)
        elif key == ord('r'):
            reset_window()
        elif key == ord(' '):
            paused = not paused

    if cap is not None:
        cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()