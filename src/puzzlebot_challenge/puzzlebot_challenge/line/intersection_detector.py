#!/usr/bin/env python3
"""
intersection_detector.py  (v5 — near-band STOP trigger)

Minimal intersection detector: raises a STOP signal when a dashed crossing
line is right at the wheels. No direction logic — after stopping, the sign
reader decides what to do (handled elsewhere).

Pipeline
--------
1. IPM warp        : perspective transform from 4 floor points -> bird's-eye.
                     (Toggle ipm_enable; ROI-crop fallback when off.)
2. Threshold+morph : adaptive threshold + opening/closing.
3. Dash blobs      : contours filtered by area + aspect ratio.
4. NEAR BAND       : count only dashes in the bottom `near_band` fraction of
                     the warp — i.e. on the ground right in front of the robot.
5. STOP            : count >= min_dashes  ->  stop = True.
6. Debounce        : confirm over a couple of frames (in the node).

Because the view is top-down, "at the wheels" is simply "in the bottom strip"
— a fixed real distance, not a perspective guess.

Topics
------
Sub : /camera/image_raw       (sensor_msgs/Image)
Pub : /intersection/stop      (std_msgs/Bool)   <- main trigger
      /intersection/detected  (std_msgs/Bool)   <- same value (compat alias)
      /vision/intersection     (sensor_msgs/Image)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge


_DEFAULT_PARAMS = {
    # ── Inverse Perspective Mapping (bird's-eye) ───────────────────────
    "ipm_enable":     1,
    "src_top_x":      0.30,
    "src_top_y":      0.62,
    "src_bot_x":      0.02,
    "src_bot_y":      0.98,
    "warp_w":         360,
    "warp_h":         480,

    # ── Fallback ROI (used only when ipm_enable = 0) ───────────────────
    "roi_top":        0.45,

    # ── Threshold + morphology ─────────────────────────────────────────
    "blur":           5,
    "adapt_block":    41,
    "adapt_c":        10,
    "morph":          3,

    # ── Dash-shaped blob filter ────────────────────────────────────────
    "seg_min_area":   40.0,
    "seg_max_area":   6000.0,
    "seg_min_aspect": 1.5,

    # ── Near band + trigger ────────────────────────────────────────────
    "near_band":      0.18,   # bottom fraction of the warp = "at the wheels"
    "min_dashes":     3,      # dashes in the band needed to STOP
    "debounce":       2,      # consecutive frames to confirm
}


def _load_params_yaml(path: str) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            raw = yaml.safe_load(f) or {}
        out = dict(_DEFAULT_PARAMS)
        out.update({k: v for k, v in raw.items() if k in _DEFAULT_PARAMS})
        return out
    except Exception:
        return None


def src_quad(frame_shape, params) -> np.ndarray:
    """The 4 source floor points (TL, TR, BR, BL) in raw-image pixels."""
    H, W = frame_shape[:2]
    stx = float(params["src_top_x"]) * W
    sty = float(params["src_top_y"]) * H
    sbx = float(params["src_bot_x"]) * W
    sby = float(params["src_bot_y"]) * H
    return np.float32([[stx, sty], [W - stx, sty],
                       [W - sbx, sby], [sbx, sby]])


def draw_src_quad(frame: np.ndarray, params: dict) -> np.ndarray:
    """For the calibrator: show the IPM source region on the raw frame."""
    vis = frame.copy()
    quad = src_quad(frame.shape, params).astype(int)
    cv2.polylines(vis, [quad], True, (0, 255, 255), 2)
    for (x, y), lbl in zip(quad, ("TL", "TR", "BR", "BL")):
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(vis, lbl, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, "IPM source quad (warp these 4 floor points)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
class IntersectionDetection:

    def __init__(self, params: dict | None = None):
        self.params = dict(_DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    @staticmethod
    def _aspect(box: np.ndarray) -> float:
        e0 = box[1] - box[0]
        e1 = box[2] - box[1]
        n0 = float(np.linalg.norm(e0))
        n1 = float(np.linalg.norm(e1))
        long_, short_ = (n0, n1) if n0 >= n1 else (n1, n0)
        return long_ / max(short_, 1.0)

    def _work_image(self, frame_bgr: np.ndarray):
        """Returns (canvas_bgr, work_gray, oy, Pw, Ph)."""
        p = self.params
        H, W = frame_bgr.shape[:2]
        if int(p["ipm_enable"]):
            Ww = max(60, int(p["warp_w"]))
            Wh = max(60, int(p["warp_h"]))
            src = src_quad(frame_bgr.shape, p)
            dst = np.float32([[0, 0], [Ww, 0], [Ww, Wh], [0, Wh]])
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(frame_bgr, M, (Ww, Wh))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            return warped, gray, 0, Ww, Wh
        roi_y = max(0, min(int(H * float(p["roi_top"])), H - 2))
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return frame_bgr.copy(), gray[roi_y:, :], roi_y, W, (H - roi_y)

    def detect(self, frame_bgr: np.ndarray) -> dict:
        p = self.params
        out: dict = {
            "stop": False, "count": 0,
            "segments": [],
            "canvas": None, "binary": None,
            "oy": 0, "pw": 0, "ph": 0,
            "band_top": 0,                 # work-image y where the near band starts
            "ipm": bool(int(p["ipm_enable"])),
        }
        if frame_bgr is None or frame_bgr.size == 0:
            return out

        canvas, work_gray, oy, Pw, Ph = self._work_image(frame_bgr)
        out["canvas"] = canvas
        out["oy"], out["pw"], out["ph"] = oy, Pw, Ph

        k = int(p["blur"]) | 1
        if k >= 3:
            work_gray = cv2.GaussianBlur(work_gray, (k, k), 0)
        block = max(3, int(p["adapt_block"]) | 1)
        binary = cv2.adaptiveThreshold(
            work_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, float(p["adapt_c"]))
        mk = int(p["morph"])
        if mk >= 2:
            kernel = np.ones((mk, mk), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        out["binary"] = binary

        band = float(np.clip(p["near_band"], 0.02, 0.99))
        band_top = (1.0 - band) * Ph
        out["band_top"] = int(band_top)

        a_min = float(p["seg_min_area"])
        a_max = float(p["seg_max_area"])
        asp   = float(p["seg_min_aspect"])

        count = 0
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            (_cx, cy), _, _ = rect
            is_dash = (a_min <= area <= a_max) and (self._aspect(box) >= asp)
            in_band = cy >= band_top
            if is_dash and in_band:
                count += 1
            out["segments"].append({
                "box": (box + np.array([0, oy])).astype(int),
                "dash": is_dash, "near": in_band,
            })

        out["count"] = count
        out["stop"] = count >= int(p["min_dashes"])
        return out


def draw_overlay(frame: np.ndarray, r: dict, params: dict) -> np.ndarray:
    canvas = r.get("canvas")
    vis = (canvas if canvas is not None else frame).copy()
    oy, Pw, Ph = r["oy"], r["pw"], r["ph"]
    band_top_c = oy + r["band_top"]

    # shade the near band (the "at the wheels" zone)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, band_top_c), (vis.shape[1], oy + Ph),
                  (0, 0, 160), -1)
    vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
    cv2.line(vis, (0, band_top_c), (vis.shape[1], band_top_c), (0, 140, 255), 2)

    # blobs: green = counted dash in band ; yellow = dash above band ; gray = other
    for s in r["segments"]:
        if s["dash"] and s["near"]:
            color = (0, 255, 0)
        elif s["dash"]:
            color = (0, 220, 220)
        else:
            color = (110, 110, 110)
        cv2.drawContours(vis, [s["box"]], 0, color, 2)

    mode = "IPM" if r.get("ipm") else "ROI"
    state = "STOP" if r["stop"] else "clear"
    color = (0, 0, 255) if r["stop"] else (0, 200, 0)
    hud = (f"[{mode}] {state}   near={r['count']}  "
           f"thr={int(params['min_dashes'])}")
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (40, 40, 40), -1)
    cv2.putText(vis, hud, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
class IntersectionDetectorNode(Node):

    def __init__(self):
        super().__init__("intersection_detector")

        self.declare_parameter(
            "image_topic", "/camera/image_raw",
            ParameterDescriptor(description="Input image topic"))
        self.declare_parameter(
            "params_config", "",
            ParameterDescriptor(description="Optional path to intersection_params.yaml"))

        for k, v in _DEFAULT_PARAMS.items():
            self.declare_parameter(
                k, float(v) if isinstance(v, float) else int(v),
                ParameterDescriptor(description=f"Intersection param: {k}"))

        yaml_path = self.get_parameter("params_config").value
        yaml_params = _load_params_yaml(yaml_path)
        if yaml_params:
            self.get_logger().info(f"Params loaded from: {yaml_path}")
            for k, v in yaml_params.items():
                ptype = (rclpy.parameter.Parameter.Type.DOUBLE
                         if isinstance(v, float)
                         else rclpy.parameter.Parameter.Type.INTEGER)
                self.set_parameters(
                    [rclpy.parameter.Parameter(k, ptype, v)])
        else:
            self.get_logger().info("Using default params (no YAML)")

        self.bridge   = CvBridge()
        self.detector = IntersectionDetection(self._snapshot_params())

        self._pending = False
        self._pending_n = 0
        self._confirmed = False

        image_topic = self.get_parameter("image_topic").value
        self.sub_img      = self.create_subscription(
            Image, image_topic, self._on_image, 10)
        self.pub_stop     = self.create_publisher(Bool, "/intersection/stop",     10)
        self.pub_detected = self.create_publisher(Bool, "/intersection/detected", 10)
        self.pub_debug    = self.create_publisher(Image, "/vision/intersection",  10)

        self.get_logger().info(
            f"IntersectionDetectorNode (v5 STOP) ready | topic={image_topic} | "
            f"ipm={'on' if int(self.detector.params['ipm_enable']) else 'off'} | "
            f"near_band={self.detector.params['near_band']:.2f} | "
            f"min_dashes={int(self.detector.params['min_dashes'])}")

    def _snapshot_params(self) -> dict:
        return {k: self.get_parameter(k).value for k in _DEFAULT_PARAMS}

    def _on_image(self, msg: Image):
        self.detector.params.update(self._snapshot_params())
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        r = self.detector.detect(frame)
        raw = bool(r["stop"])

        n_need = int(self.detector.params["debounce"])
        if raw == self._pending:
            self._pending_n += 1
        else:
            self._pending = raw
            self._pending_n = 1
        if self._pending_n >= n_need and self._pending != self._confirmed:
            self._confirmed = self._pending
            self.get_logger().info(
                f"STOP LINE {'DETECTED' if self._confirmed else 'cleared'}")

        msg_out = Bool(); msg_out.data = self._confirmed
        self.pub_stop.publish(msg_out)
        self.pub_detected.publish(msg_out)

        if self.pub_debug.get_subscription_count() > 0 and r["canvas"] is not None:
            vis = draw_overlay(frame, r, self.detector.params)
            out_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            out_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_debug.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = IntersectionDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()