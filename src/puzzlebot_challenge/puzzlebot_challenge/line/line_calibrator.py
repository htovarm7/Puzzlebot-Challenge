#!/usr/bin/env python3
"""Interactive parameter tuner for the contour-based line detector.

The detection math mirrors line_detector.LineDetection, so what you tune here
is what the running node will do. It saves only the keys the node consumes
into line_params.yaml.

Input modes:
  --live              Subscribe to /camera/image_raw (ROS2)
  --image PATH        Static image or video file
  (no flag)           Open default webcam

Keys:
  q       quit
  s       save params to line_params.yaml
  r       reset trackbars to defaults
  space   pause / unpause (no effect in live mode)

Usage:
  ros2 run puzzlebot_challenge line_calibrator --live
  ros2 run puzzlebot_challenge line_calibrator --image floor.png
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml


# Defaults (match line_detector._DEFAULT_PARAMS)
DEFAULTS = {
    "T_init":        185,
    "T_min":         127,
    "T_max":         222,
    "dark_min_x10":  10,    # 1.0 %
    "dark_max_x10":  60,    # 6.0 %
    "roi_top_x100":  60,    # 60 %
    "min_area":      300,
    "blur":          21,
    "morph":         9,
    "n_track_lines": 3,
    "adaptive":      0,     # 0 = global balanced, 1 = local adaptive
    "adapt_block":   61,    # neighborhood size (odd), larger is smoother
    "adapt_c":       12,    # line must be this many units darker than local mean
    # follower-side display only (not saved to the node YAML)
    "turn_angle":    36,
    "shift_max":     130,
}


# Windows
WIN_CTRL  = "Controls"
WIN_DEBUG = "Debug"
WIN_BIN   = "Binary (ROI)"


def nothing(_):
    pass


# YAML load (back into trackbar units)
def _load_saved(yaml_path: Path) -> dict:
    init = dict(DEFAULTS)
    if not yaml_path or not yaml_path.exists():
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
        for key in ("T_init", "T_min", "T_max", "min_area", "blur", "morph",
                    "n_track_lines", "adaptive", "adapt_block", "adapt_c"):
            if key in saved:
                init[key] = int(saved[key])
        print(f"[calibrator] Parameters loaded from {yaml_path}")
    except Exception as e:
        print(f"[calibrator] Could not load YAML: {e}")
    return init


def build_window(yaml_path: Path | None = None):
    init = _load_saved(yaml_path) if yaml_path else dict(DEFAULTS)

    cv.namedWindow(WIN_CTRL, cv.WINDOW_NORMAL)
    cv.resizeWindow(WIN_CTRL, 470, 640)

    # mode + shared
    cv.createTrackbar("adaptive (0/1)", WIN_CTRL, init["adaptive"],       1, nothing)
    cv.createTrackbar("ROI top %",      WIN_CTRL, init["roi_top_x100"],  99, nothing)
    cv.createTrackbar("blur (odd)",     WIN_CTRL, init["blur"],          31, nothing)
    cv.createTrackbar("morph kernel",   WIN_CTRL, init["morph"],         15, nothing)
    cv.createTrackbar("min area",       WIN_CTRL, init["min_area"],    5000, nothing)
    cv.createTrackbar("n track lines",  WIN_CTRL, init["n_track_lines"],  6, nothing)
    # global-threshold mode
    cv.createTrackbar("T init",         WIN_CTRL, init["T_init"],       255, nothing)
    cv.createTrackbar("T min",          WIN_CTRL, init["T_min"],        255, nothing)
    cv.createTrackbar("T max",          WIN_CTRL, init["T_max"],        255, nothing)
    cv.createTrackbar("dark% min x10",  WIN_CTRL, init["dark_min_x10"], 500, nothing)
    cv.createTrackbar("dark% max x10",  WIN_CTRL, init["dark_max_x10"], 500, nothing)
    # adaptive mode
    cv.createTrackbar("adapt block",    WIN_CTRL, init["adapt_block"],  151, nothing)
    cv.createTrackbar("adapt c",        WIN_CTRL, init["adapt_c"],       40, nothing)
    # follower-side display only
    cv.createTrackbar("turn angle",     WIN_CTRL, init["turn_angle"],    90, nothing)
    cv.createTrackbar("shift max px",   WIN_CTRL, init["shift_max"],    200, nothing)


def reset_window():
    for name, key in [
        ("adaptive (0/1)", "adaptive"), ("ROI top %", "roi_top_x100"),
        ("blur (odd)", "blur"), ("morph kernel", "morph"),
        ("min area", "min_area"), ("n track lines", "n_track_lines"),
        ("T init", "T_init"), ("T min", "T_min"), ("T max", "T_max"),
        ("dark% min x10", "dark_min_x10"), ("dark% max x10", "dark_max_x10"),
        ("adapt block", "adapt_block"), ("adapt c", "adapt_c"),
        ("turn angle", "turn_angle"), ("shift max px", "shift_max"),
    ]:
        cv.setTrackbarPos(name, WIN_CTRL, DEFAULTS[key])


def read_params():
    p = {
        "adaptive":      cv.getTrackbarPos("adaptive (0/1)", WIN_CTRL),
        "roi_top":       cv.getTrackbarPos("ROI top %",      WIN_CTRL) / 100.0,
        "blur":          max(1, cv.getTrackbarPos("blur (odd)",   WIN_CTRL) | 1),
        "morph":         max(1, cv.getTrackbarPos("morph kernel", WIN_CTRL)),
        "min_area":      cv.getTrackbarPos("min area",       WIN_CTRL),
        "n_track_lines": max(1, cv.getTrackbarPos("n track lines", WIN_CTRL)),
        "T_init":        cv.getTrackbarPos("T init",         WIN_CTRL),
        "T_min":         cv.getTrackbarPos("T min",          WIN_CTRL),
        "T_max":         cv.getTrackbarPos("T max",          WIN_CTRL),
        "dark_min":      cv.getTrackbarPos("dark% min x10",  WIN_CTRL) / 10.0,
        "dark_max":      cv.getTrackbarPos("dark% max x10",  WIN_CTRL) / 10.0,
        "adapt_block":   max(3, cv.getTrackbarPos("adapt block", WIN_CTRL) | 1),
        "adapt_c":       cv.getTrackbarPos("adapt c",        WIN_CTRL),
        "turn_angle":    cv.getTrackbarPos("turn angle",     WIN_CTRL),
        "shift_max":     cv.getTrackbarPos("shift max px",   WIN_CTRL),
    }
    if p["T_min"] >= p["T_max"]:
        p["T_max"] = p["T_min"] + 1
    if p["dark_min"] >= p["dark_max"]:
        p["dark_max"] = p["dark_min"] + 0.1
    return p


# Only these keys are consumed by line_detector. turn_angle / shift_max are
# follower-side concepts and live in the follower's config, not here.
_NODE_KEYS = ("T_init", "T_min", "T_max", "dark_min", "dark_max",
              "roi_top", "min_area", "blur", "morph",
              "n_track_lines", "adaptive", "adapt_block", "adapt_c")


def _resolve_default_yaml() -> Path:
    """Path of the installed YAML (same one the launch reads).
    With --symlink-install saving here is picked up without recompiling."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("puzzlebot_challenge")
        return Path(share) / "config" / "line_params.yaml"
    except Exception:
        here = Path(__file__).resolve().parent
        return here.parent / "config" / "line_params.yaml"


def save_params(p: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k in _NODE_KEYS:
        v = p[k]
        # ints stay ints; dark_min/dark_max/roi_top stay floats
        if k in ("dark_min", "dark_max", "roi_top"):
            payload[k] = float(v)
        else:
            payload[k] = int(v)
    with open(out_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f"[saved] {out_path.resolve()}")


# Detection (mirrors line_detector.LineDetection.detect)
# Persistent global threshold state, like the node carries _T_state across frames.
_T_state = DEFAULTS["T_init"]


def _balance(gray, p, y_off):
    """Global balanced threshold, mirror of LineDetection._balance."""
    global _T_state
    T = _T_state
    direction = 0
    for _ in range(10):
        _, binary = cv.threshold(gray, T, 255, cv.THRESH_BINARY_INV)
        crop = binary[y_off:, :]
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


def _adaptive(gray, p, y_off):
    """Local adaptive threshold, mirror of LineDetection._adaptive_roi."""
    block = int(p["adapt_block"]) | 1
    C     = int(p["adapt_c"])
    binary = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, block, C)
    return binary[y_off:, :]


def detect(frame, p):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if p["blur"] >= 3:
        gray = cv.GaussianBlur(gray, (p["blur"], p["blur"]), 0)

    h = frame.shape[0]
    y_off = int(h * p["roi_top"])

    debug = frame.copy()
    cv.line(debug, (0, y_off), (frame.shape[1], y_off), (255, 200, 0), 1)

    adaptive = bool(int(p["adaptive"]))
    if adaptive:
        binary_roi = _adaptive(gray, p, y_off)
        T_used = 0
    else:
        binary_roi, T_used = _balance(gray, p, y_off)
        if binary_roi is None:
            cv.putText(debug, "no balanced threshold", (10, 25),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return debug, np.zeros((100, 300), np.uint8), None, None, T_used

    # Morphology: OPEN then CLOSE, same order as the node
    k = int(p["morph"])
    if k >= 2:
        kernel = np.ones((k, k), np.uint8)
        binary_roi = cv.morphologyEx(binary_roi, cv.MORPH_OPEN,  kernel)
        binary_roi = cv.morphologyEx(binary_roi, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        binary_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv.contourArea(c) >= p["min_area"]]

    angle = shift = None
    if contours:
        # Keep the N largest blobs, then pick the contour whose centroid is
        # closest to the horizontal centre, identical to the node.
        n_lines = int(p["n_track_lines"])
        if len(contours) > n_lines:
            contours = sorted(contours, key=cv.contourArea, reverse=True)[:n_lines]

        center_x = binary_roi.shape[1] // 2

        def _cx(c):
            m = cv.moments(c)
            return (m["m10"] / m["m00"]) if m["m00"] else 0.0

        line = min(contours, key=lambda c: abs(_cx(c) - center_x))

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

        shift = int(cx - center_x)

        cv.drawContours(debug, [line + [0, y_off]], -1, (0, 255, 0), 2)
        box_shifted = (box + [0, y_off]).astype(int)
        cv.drawContours(debug, [box_shifted], 0, (255, 0, 255), 1)
        p1 = (int(top_mid[0]),    int(top_mid[1]) + y_off)
        p2 = (int(bottom_mid[0]), int(bottom_mid[1]) + y_off)
        cv.line(debug, p1, p2, (0, 0, 255), 3)

    fx = frame.shape[1] // 2
    cv.line(debug, (fx, y_off), (fx, frame.shape[0]), (0, 255, 255), 1)

    mode = (f"ADAPT b={int(p['adapt_block'])} c={int(p['adapt_c'])}"
            if adaptive else f"T={T_used}")
    if angle is not None:
        cv.putText(debug, f"{mode}  angle={angle:5.1f}  shift={shift:+d}",
                   (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv.putText(debug, f"{mode}  NO LINE", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    return debug, binary_roi, angle, shift, T_used


# Frame source: file / webcam
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


# Frame source: live ROS2 topic
class _LiveFrameBuffer:
    """ROS callback writes, UI reads."""

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
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rclpy.init()

    class _CamNode(Node):
        def __init__(self):
            super().__init__("line_calibrator")
            self.bridge = CvBridge()
            self.create_subscription(Image, topic, self._cb, 10)
            self.get_logger().info(f"Subscribed to {topic}")

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


# Shared UI loop
def _ui_loop(source, pump_or_cap, out_path: Path, is_image: bool, is_live: bool):
    cv.namedWindow(WIN_DEBUG, cv.WINDOW_NORMAL)
    cv.namedWindow(WIN_BIN,   cv.WINDOW_NORMAL)

    paused = False
    last_frame = None

    print("Keys: [q] quit  [s] save params  [r] reset  [space] pause")
    if is_live:
        print(f"  Waiting for frames...  YAML output: {out_path}")

    while True:
        if is_live:
            pump_or_cap()                 # rclpy.spin_once
            frame = source.latest()       # may be None until first message
        else:
            if not paused or last_frame is None:
                frame = source()
                if frame is None:
                    if is_image:
                        break
                    if pump_or_cap is not None:   # video, loop it
                        pump_or_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                last_frame = frame
            else:
                frame = last_frame

        if frame is None:
            key = cv.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            continue

        p = read_params()
        debug, binary, angle, shift, T_used = detect(frame.copy(), p)
        cv.imshow(WIN_DEBUG, debug)
        cv.imshow(WIN_BIN, binary)

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


# Entry point
def main():
    default_yaml = _resolve_default_yaml()

    ap = argparse.ArgumentParser(description="Line-detector tuner (PuzzleBot)")
    ap.add_argument("--live", action="store_true",
                    help="Subscribe to a ROS2 topic instead of file/webcam")
    ap.add_argument("--topic", default="/camera/image_raw",
                    help="ROS2 topic (with --live)")
    ap.add_argument("--image", default=None,
                    help="Path to image / video. Omitted (and no --live) = webcam.")
    ap.add_argument("--out", default=str(default_yaml),
                    help=f"Output YAML (default: {default_yaml})")
    args = ap.parse_args()

    out_path = Path(args.out)
    build_window(out_path)   # loads saved YAML into trackbars if present

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
