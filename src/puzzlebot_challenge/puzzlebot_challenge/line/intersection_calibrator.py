#!/usr/bin/env python3
"""
intersection_calibrator.py  (v5 — near-band STOP trigger)

Interactive tuner. Reuses the EXACT detection core from
`intersection_detector`, so the tuner matches the runtime node.

Windows
-------
  Source    raw frame with the IPM source quad (tune the 4 floor points)
  Debug     work view (warped) with the shaded near band + dash count
  Binary    the thresholded work image

Tuning order
------------
  1. ipm_enable = 1. Adjust src_top_x/y, src_bot_x/y for a clean top-down.
  2. adapt_block / adapt_c / morph until dashes are crisp in Binary.
  3. seg_min_area / seg_max_area / seg_min_aspect so each dash is one green
     blob and solid lines / noise stay gray.
  4. near_band: height of the red band at the bottom = "at the wheels".
     Make it just thick enough that the crossing row lands inside it right
     as it reaches the robot.
  5. min_dashes: how many dashes in the band trigger STOP (HUD turns red).

Keys : q quit   s save   r reset   space pause (file/webcam only)

Usage
-----
  ros2 run puzzlebot_challenge intersection_calibrator --live
  ros2 run puzzlebot_challenge intersection_calibrator --image floor.png
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

try:
    from puzzlebot_challenge.line.intersection_detector import (
        IntersectionDetection, draw_overlay, draw_src_quad, _DEFAULT_PARAMS,
    )
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from intersection_detector import (  # type: ignore
        IntersectionDetection, draw_overlay, draw_src_quad, _DEFAULT_PARAMS,
    )

WIN_CTRL  = "Intersection — Controls"
WIN_SRC   = "Intersection — Source quad (raw)"
WIN_DEBUG = "Intersection — Debug (work view)"
WIN_BIN   = "Intersection — Binary"


def nothing(_):
    pass


# (label, trackbar_max, scale, param_key)   value = trackbar / scale
_TRACKBARS = [
    ("ipm_enable",         1,     1.0,   "ipm_enable"),
    ("src_top_x %",        49,    100.0, "src_top_x"),
    ("src_top_y %",        99,    100.0, "src_top_y"),
    ("src_bot_x %",        49,    100.0, "src_bot_x"),
    ("src_bot_y %",        99,    100.0, "src_bot_y"),
    ("warp_w",             720,   1.0,   "warp_w"),
    ("warp_h",             720,   1.0,   "warp_h"),
    ("roi_top % (no ipm)", 99,    100.0, "roi_top"),
    ("blur (odd)",         31,    1.0,   "blur"),
    ("adapt_block(odd)",   151,   1.0,   "adapt_block"),
    ("adapt_c",            40,    1.0,   "adapt_c"),
    ("morph",              15,    1.0,   "morph"),
    ("seg_min_area",       4000,  1.0,   "seg_min_area"),
    ("seg_max_area",       30000, 1.0,   "seg_max_area"),
    ("seg_min_asp x10",    200,   10.0,  "seg_min_aspect"),
    ("near_band %",        60,    100.0, "near_band"),
    ("min_dashes",         15,    1.0,   "min_dashes"),
    ("debounce",           10,    1.0,   "debounce"),
]


def _to_trackbar(key, scale, saved):
    val = saved.get(key, _DEFAULT_PARAMS.get(key, 0))
    return int(round(float(val) * scale))


def _load_saved(yaml_path):
    if not yaml_path or not yaml_path.exists():
        return {}
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
        print(f"[calibrator] Loaded {yaml_path}")
        return {k: v for k, v in data.items() if k in _DEFAULT_PARAMS}
    except Exception as e:
        print(f"[calibrator] Could not load YAML: {e}")
        return {}


def build_window(yaml_path=None):
    saved = _load_saved(yaml_path)
    cv.namedWindow(WIN_CTRL, cv.WINDOW_NORMAL)
    cv.resizeWindow(WIN_CTRL, 480, 660)
    for name, vmax, scale, key in _TRACKBARS:
        init = min(_to_trackbar(key, scale, saved), vmax)
        cv.createTrackbar(name, WIN_CTRL, max(0, init), vmax, nothing)


def reset_window():
    for name, vmax, scale, key in _TRACKBARS:
        init = min(int(round(float(_DEFAULT_PARAMS[key]) * scale)), vmax)
        cv.setTrackbarPos(name, WIN_CTRL, max(0, init))


def read_params():
    p = {}
    for name, vmax, scale, key in _TRACKBARS:
        raw = cv.getTrackbarPos(name, WIN_CTRL)
        p[key] = raw / scale if scale != 1.0 else raw
    p["blur"]        = max(1, int(p["blur"]) | 1)
    p["adapt_block"] = max(3, int(p["adapt_block"]) | 1)
    p["morph"]       = max(1, int(p["morph"]))
    p["warp_w"]      = max(60, int(p["warp_w"]))
    p["warp_h"]      = max(60, int(p["warp_h"]))
    if p["src_top_y"] >= p["src_bot_y"]:
        p["src_bot_y"] = min(0.999, p["src_top_y"] + 0.01)
    if p["seg_min_area"] >= p["seg_max_area"]:
        p["seg_max_area"] = p["seg_min_area"] + 1
    p["near_band"] = float(np.clip(p["near_band"], 0.02, 0.99))
    # cast to default types; fall back to defaults if a key is somehow missing
    out = {}
    for k, dv in _DEFAULT_PARAMS.items():
        v = p.get(k, dv)
        out[k] = float(v) if isinstance(dv, float) else int(v)
    return out


def _resolve_default_yaml():
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("puzzlebot_challenge")
        return Path(share) / "config" / "intersection_params.yaml"
    except Exception:
        here = Path(__file__).resolve().parent
        return here.parent / "config" / "intersection_params.yaml"


def save_params(p, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: p[k] for k in _DEFAULT_PARAMS}
    with open(out_path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f"[saved] {out_path.resolve()}")


class _LiveFrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def push(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def latest(self):
        with self._lock:
            return self._frame


def run_live(buf, topic, out_path):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rclpy.init()

    class _CamNode(Node):
        def __init__(self):
            super().__init__("intersection_tuner")
            self.bridge = CvBridge()
            self.create_subscription(Image, topic, self._cb, 10)
            self.get_logger().info(f"Subscribed to {topic}")

        def _cb(self, msg):
            try:
                buf.push(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))
            except Exception:
                pass

    node = _CamNode()
    try:
        _ui_loop(buf, lambda: rclpy.spin_once(node, timeout_sec=0.01),
                 out_path, is_image=False, is_live=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def open_source_file(arg):
    if arg is None:
        cap = cv.VideoCapture(0)
        return (lambda: cap.read()[1]), False, cap
    path = Path(arg)
    if not path.exists():
        print(f"error: {arg} not found")
        raise SystemExit(1)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        img = cv.imread(str(path))
        if img is None:
            print(f"error: imread failed on {arg}")
            raise SystemExit(1)
        return (lambda: img.copy()), True, None
    cap = cv.VideoCapture(str(path))
    return (lambda: cap.read()[1]), False, cap


def _ui_loop(source, pump_or_cap, out_path, is_image, is_live):
    for w in (WIN_SRC, WIN_DEBUG, WIN_BIN):
        cv.namedWindow(w, cv.WINDOW_NORMAL)
    detector = IntersectionDetection()

    paused = False
    last_frame = None
    print("Keys: [q] quit  [s] save  [r] reset  [space] pause")
    if is_live:
        print(f"  Waiting for frames...  YAML output: {out_path}")

    while True:
        if is_live:
            pump_or_cap()
            frame = source.latest()
        else:
            if not paused or last_frame is None:
                frame = source()
                if frame is None:
                    if is_image:
                        break
                    if pump_or_cap is not None:
                        pump_or_cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                last_frame = frame
            else:
                frame = last_frame

        if frame is None:
            if (cv.waitKey(30) & 0xFF) == ord('q'):
                break
            continue

        p = read_params()
        detector.params.update(p)
        r = detector.detect(frame)
        cv.imshow(WIN_SRC, draw_src_quad(frame, p))
        cv.imshow(WIN_DEBUG, draw_overlay(frame, r, p))
        if r["binary"] is not None:
            cv.imshow(WIN_BIN, r["binary"])

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


def main():
    default_yaml = _resolve_default_yaml()
    ap = argparse.ArgumentParser(description="Intersection STOP tuner (PuzzleBot)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--topic", default="/camera/image_raw")
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default=str(default_yaml))
    args = ap.parse_args()

    out_path = Path(args.out)
    build_window(out_path)

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