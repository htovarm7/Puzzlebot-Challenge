"""
Interactive parameter tuner for contour-based line detection.

Three modes:
  --live              Subscribe to /camera/image_raw (ROS2)         ← preferred
  --image PATH        Static image / video file
  (no flag)           Open default webcam (cv.VideoCapture(0))

Keys:
  q       quit
  s       save params to line_params.yaml (format consumed by line_detector node)
  r       reset trackbars to defaults
  space   pause / unpause (no effect in live mode — frames always flow)
"""
from __future__ import annotations
import cv2 as cv    
import numpy as np
import sys
import os
import argparse
import threading
from pathlib import Path

import yaml


# ─── Defaults ────────────────────────────────────────────────────────
DEFAULTS = {
    "T_init":       185,
    "T_min":        127,
    "T_max":        222,
    "dark_min_x10": 10,   # 1.0 %
    "dark_max_x10": 60,   # 6.0 %
    "roi_top_x100": 60,   # 60 %
    "min_area":     300,
    "blur":         21,
    "morph":        9,
    "turn_angle":   36,
    "shift_max":    130,
}


# ─── Tuner UI ────────────────────────────────────────────────────────
WIN_CTRL  = "Controls"
WIN_DEBUG = "Debug"
WIN_BIN   = "Binary (ROI)"


def nothing(_):
    pass


def _load_saved(yaml_path: Path) -> dict:
    """Load previously saved params from YAML and convert back to trackbar units."""
    init = dict(DEFAULTS)
    if not yaml_path.exists():
        return init
    try:
        with open(yaml_path) as f:
            saved = yaml.safe_load(f) or {}
        if "dark_min" in saved:
            init["dark_min_x10"] = int(round(saved["dark_min"] * 10))
        if "dark_max" in saved:
            init["dark_max_x10"] = int(round(saved["dark_max"] * 10))
        if "roi_top" in saved:
            init["roi_top_x100"] = int(round(saved["roi_top"] * 100))
        for key in ("T_init", "T_min", "T_max", "min_area", "blur", "morph"):
            if key in saved:
                init[key] = int(saved[key])
        print(f"[calibrator] Parámetros cargados desde {yaml_path}")
    except Exception as e:
        print(f"[calibrator] No se pudo cargar YAML: {e}")
    return init


def build_window(yaml_path: Path | None = None):
    init = _load_saved(yaml_path) if yaml_path else dict(DEFAULTS)

    cv.namedWindow(WIN_CTRL, cv.WINDOW_NORMAL)
    cv.resizeWindow(WIN_CTRL, 460, 520)

    cv.createTrackbar("T init",        WIN_CTRL, init["T_init"],       255, nothing)
    cv.createTrackbar("T min",         WIN_CTRL, init["T_min"],        255, nothing)
    cv.createTrackbar("T max",         WIN_CTRL, init["T_max"],        255, nothing)
    cv.createTrackbar("dark% min x10", WIN_CTRL, init["dark_min_x10"], 500, nothing)
    cv.createTrackbar("dark% max x10", WIN_CTRL, init["dark_max_x10"], 500, nothing)
    cv.createTrackbar("ROI top %",     WIN_CTRL, init["roi_top_x100"],  99, nothing)
    cv.createTrackbar("min area",      WIN_CTRL, init["min_area"],    5000, nothing)
    cv.createTrackbar("blur (odd)",    WIN_CTRL, init["blur"],           21, nothing)
    cv.createTrackbar("morph kernel",  WIN_CTRL, init["morph"],          15, nothing)
    cv.createTrackbar("turn angle",    WIN_CTRL, init["turn_angle"],     90, nothing)
    cv.createTrackbar("shift max px",  WIN_CTRL, init["shift_max"],     200, nothing)


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
        "blur":       max(1, cv.getTrackbarPos("blur (odd)", WIN_CTRL) | 1),
        "morph":      max(1, cv.getTrackbarPos("morph kernel", WIN_CTRL)),
        "turn_angle": cv.getTrackbarPos("turn angle",   WIN_CTRL),
        "shift_max":  cv.getTrackbarPos("shift max px", WIN_CTRL),
    }
    if p["T_min"] >= p["T_max"]:
        p["T_max"] = p["T_min"] + 1
    if p["dark_min"] >= p["dark_max"]:
        p["dark_max"] = p["dark_min"] + 0.1
    return p


# Only these keys are consumed by line_detector. turn_angle / shift_max are
# follower-side concepts and live in the follower's config, not here.
_NODE_KEYS = ("T_init", "T_min", "T_max", "dark_min", "dark_max",
              "roi_top", "min_area", "blur", "morph")


def _resolve_default_yaml() -> Path:
    """Mirror the lookup used by hsv_calibrator: pkg/config/line_params.yaml."""
    here     = Path(__file__).resolve().parent           # .../puzzlebot_challenge/
    pkg_root = here.parent                               # src/puzzlebot_challenge/
    return pkg_root / "config" / "line_params.yaml"


def save_params(p: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: p[k] for k in _NODE_KEYS}
    with open(out_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f"[saved] {out_path.resolve()}")


# ─── Detection (parameterized, threshold persists across frames) ─────
_T_state = DEFAULTS["T_init"]


def crop_roi(img, roi_top):
    h = img.shape[0]
    y1 = int(h * roi_top)
    return img[y1:, :], y1


def balance_pic(gray, p):
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
            if T <= p["T_min"] or direction == 1:
                _T_state = T
                return crop, T
            T -= 10
            direction = -1
        elif perc < p["dark_min"]:
            if T >= p["T_max"] or direction == -1:
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

    h = frame.shape[0]
    y_off = int(h * p["roi_top"])
    cv.line(debug, (0, y_off), (frame.shape[1], y_off), (255, 200, 0), 1)

    if binary_roi is None:
        cv.putText(debug, "no balanced threshold", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        blank = np.zeros((100, 300), np.uint8)
        return debug, blank, None, None, T_used

    k = p["morph"]
    if k >= 2:
        kernel = np.ones((k, k), np.uint8)
        binary_roi = cv.morphologyEx(binary_roi, cv.MORPH_OPEN, kernel)
        binary_roi = cv.morphologyEx(binary_roi, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(binary_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= p["min_area"]]

    angle = shift = None
    if contours:
        # Keep top-3 largest, then pick the median by X → always the center line
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:3]

        def _cx(c):
            m = cv.moments(c)
            return int(m["m10"] / m["m00"]) if m["m00"] else 0

        contours.sort(key=_cx)
        line = contours[len(contours) // 2]

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

        cv.drawContours(debug, [line + [0, y_off]], -1, (0, 255, 0), 2)
        box_shifted = (box + [0, y_off]).astype(int)
        cv.drawContours(debug, [box_shifted], 0, (255, 0, 255), 1)
        p1 = (top_mid[0],    top_mid[1] + y_off)
        p2 = (bottom_mid[0], bottom_mid[1] + y_off)
        cv.line(debug, p1, p2, (0, 0, 255), 3)

        fx = frame.shape[1] // 2
        cv.line(debug, (fx, y_off), (fx, frame.shape[0]), (0, 255, 255), 1)

    if angle is not None:
        cv.putText(debug, f"T={T_used}  angle={angle:5.1f}  shift={shift:+d}",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv.putText(debug, f"T={T_used}  no contour", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    return debug, binary_roi, angle, shift, T_used


# ─── Frame source: file / webcam ─────────────────────────────────────
def open_source_file(arg):
    """Static image, video file, or webcam (arg=None)."""
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

    cap = cv.VideoCapture(arg)
    return (lambda: cap.read()[1]), False, cap


# ─── Frame source: live ROS2 topic ───────────────────────────────────
class _LiveFrameBuffer:
    """Same pattern as HsvCalibrator: ROS callback writes, UI reads."""

    def __init__(self):
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()

    def push(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame.copy()

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


def run_live(buf: _LiveFrameBuffer, topic: str, out_path: Path):
    """Bring up a ROS2 node, subscribe to `topic`, push frames into `buf`."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rclpy.init()

    class _CamNode(Node):
        def __init__(self):
            super().__init__("complex_lines_tuner")
            self.bridge = CvBridge()
            self.create_subscription(Image, topic, self._cb, 10)
            self.get_logger().info(f"Subscrito a {topic}")

        def _cb(self, msg: Image):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                buf.push(frame)
            except Exception:
                pass

    node = _CamNode()
    spin_once = lambda: rclpy.spin_once(node, timeout_sec=0.01)
    try:
        _ui_loop(buf, spin_once, out_path, is_image=False, is_live=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ─── Shared UI loop ──────────────────────────────────────────────────
def _ui_loop(source, pump_or_cap, out_path: Path,
             is_image: bool, is_live: bool):
    """
    source:        callable returning a frame (file/webcam mode)
                   OR a _LiveFrameBuffer (live mode)
    pump_or_cap:   cv.VideoCapture for file mode (used for looping video)
                   OR a spin_once callable for live mode
                   OR None for single-image / webcam modes
    """
    cv.namedWindow(WIN_DEBUG, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_BIN,   cv.WINDOW_NORMAL)

    paused = False
    last_frame = None

    print("Keys: [q] quit  [s] save params  [r] reset  [space] pause")
    if is_live:
        print(f"  Esperando frames en topic suscrito...  YAML salida: {out_path}")

    while True:
        # ── obtain a frame ──
        if is_live:
            pump_or_cap()                # rclpy.spin_once
            frame = source.latest()      # may be None until first message
        else:
            if not paused or last_frame is None:
                frame = source()
                if frame is None:
                    if is_image:
                        break
                    if pump_or_cap is not None:   # video — loop it
                        pump_or_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                last_frame = frame
            else:
                frame = last_frame

        # In live mode, the first frame may not have arrived yet.
        if frame is None:
            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            continue

        p = read_params()
        debug, binary, angle, shift, T_used = detect(frame.copy(), p)
        cv.imshow(WIN_DEBUG, debug)
        cv.imshow(WIN_BIN, binary)

        # In live mode keep latency low; in image mode poll more slowly.
        wait_ms = 30 if (is_image and not is_live) else 1
        key = cv.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_params(p, out_path)
        elif key == ord('r'):
            reset_window()
        elif key == ord(' ') and not is_live:
            paused = not paused


# ─── Entry point ─────────────────────────────────────────────────────
def main():
    default_yaml = _resolve_default_yaml()

    ap = argparse.ArgumentParser(description="Tuner de detector de línea (PuzzleBot)")
    ap.add_argument("--live", action="store_true",
                    help="Suscribirse al tópico ROS2 en lugar de abrir archivo/cámara")
    ap.add_argument("--topic", default="/camera/image_raw",
                    help="Tópico ROS2 (solo con --live)")
    ap.add_argument("--image", default=None,
                    help="Ruta a imagen / video. Si se omite (y no hay --live), usa la webcam.")
    ap.add_argument("--out", default=str(default_yaml),
                    help=f"Archivo YAML de salida (default: {default_yaml})")
    args = ap.parse_args()

    out_path = Path(args.out)
    build_window(out_path)  # loads saved YAML values into trackbars if file exists

    if args.live:
        buf = _LiveFrameBuffer()
        try:
            run_live(buf, args.topic, out_path)
        finally:
            cv.destroyAllWindows()
        return

    read_frame, is_image, cap = open_source_file(args.image)
    try:
        _ui_loop(read_frame, cap, out_path, is_image=is_image, is_live=False)
    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
