#!/usr/bin/env python3
"""
intersection_detector.py

Detects DASHED / intermittent intersection markings (separate from the solid
line-following module) and reports which arms of the intersection exist
relative to the robot: front, back, left, right.

Pipeline
--------
1. ROI crop (ignore wall / chairs above the floor).
2. Adaptive threshold (dark dashes on light floor) -> binary.
3. Morphology to clean speckle.
4. Contour extraction; keep only "dash-like" blobs (small-to-medium area,
   elongated aspect ratio). This automatically rejects the long SOLID
   following-line (too big) and tiny noise (too small).
5. Each surviving dash votes for an arm based on the image region its
   centroid falls in (left / right column, front / back band) and,
   optionally, its orientation (a side street's dashes look horizontal-ish;
   the straight continuation looks vertical-ish).
6. An arm is reported when its vote count clears a tunable threshold.
7. Temporal debounce stabilises the output before publishing.

Topics
------
Sub : /camera/image_raw       (sensor_msgs/Image)
Pub : /intersection/detected  (std_msgs/Bool)    True if ANY arm present
      /intersection/arms       (std_msgs/String)  e.g. "front,left,right" | "none"
      /vision/intersection     (sensor_msgs/Image) annotated debug frame

NOTE: this is a geometric heuristic baseline meant to be TUNED with
`intersection_calibrator`. "back" detection from a forward-facing camera is
inherently weak (it sits at the very bottom edge of the frame); treat it as
best-effort and lean on front/left/right for navigation decisions.
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
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge


# Fixed publish order so downstream parsing is deterministic.
ARM_ORDER = ("front", "back", "left", "right")

_DEFAULT_PARAMS = {
    # ── Preprocessing ──────────────────────────────────────────────────
    "roi_top":          0.45,   # ignore everything above this fraction of H
    "blur":             5,      # gaussian kernel (odd)
    "adapt_block":      41,     # adaptive threshold neighbourhood (odd)
    "adapt_c":          10,     # dash must be this much darker than local mean
    "morph":            3,      # morphology kernel size

    # ── Dash filtering ─────────────────────────────────────────────────
    "dash_min_area":    60.0,   # reject noise below this
    "dash_max_area":    4000.0, # reject the big SOLID following-line above this
    "dash_min_aspect":  1.8,    # length/width: dashes are elongated
    "dash_max_aspect":  12.0,

    # ── Region geometry (fractions) ────────────────────────────────────
    "left_edge":        0.33,   # cx < left_edge*W              -> LEFT column
    "right_edge":       0.66,   # cx > right_edge*W             -> RIGHT column
    "front_edge":       0.40,   # cy < front_edge*H_roi (far)   -> FRONT band
    "back_edge":        0.80,   # cy > back_edge*H_roi (near)   -> BACK band

    # ── Decision thresholds ────────────────────────────────────────────
    "min_dashes_side":  2,      # dashes needed in L/R column to declare arm
    "min_dashes_front": 2,
    "min_dashes_back":  2,

    # ── Orientation gating (1 = on) ────────────────────────────────────
    "use_orientation":  1,
    "horiz_tol_deg":    35.0,   # within this of horizontal -> "crossing" dash
    "vert_tol_deg":     35.0,   # within this of vertical   -> "straight" dash

    # ── Temporal stabilisation ─────────────────────────────────────────
    "debounce":         3,      # consecutive identical frames to confirm
}


def _load_params_yaml(path: str) -> dict | None:
    """Read a YAML saved by the tuner. None if missing / unreadable."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Pure-CV core (no ROS) — imported by the calibrator so the tuner matches
# the runtime node EXACTLY.
# ─────────────────────────────────────────────────────────────────────────────
class IntersectionDetection:

    def __init__(self, params: dict | None = None):
        self.params = dict(_DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    @staticmethod
    def _orientation(box: np.ndarray) -> tuple[float, float, float]:
        """Return (length, width, angle_deg) of an oriented box.
        angle is version-independent: 0 = horizontal, 90 = vertical, range [0,180)."""
        e0 = box[1] - box[0]
        e1 = box[2] - box[1]
        n0 = float(np.linalg.norm(e0))
        n1 = float(np.linalg.norm(e1))
        if n0 >= n1:
            long_vec, length, width = e0, n0, n1
        else:
            long_vec, length, width = e1, n1, n0
        angle = float(np.degrees(np.arctan2(long_vec[1], long_vec[0]))) % 180.0
        return length, max(width, 1.0), angle

    def detect(self, frame_bgr: np.ndarray) -> dict:
        p = self.params
        out: dict = {
            "detected": False,
            "arms":     {a: False for a in ARM_ORDER},
            "counts":   {a: 0 for a in ARM_ORDER},
            "dashes":   [],          # list of dicts for overlay
            "roi_y":    0,
            "binary":   None,
        }
        if frame_bgr is None or frame_bgr.size == 0:
            return out

        H, W = frame_bgr.shape[:2]
        roi_y = int(H * float(p["roi_top"]))
        roi_y = max(0, min(roi_y, H - 2))
        out["roi_y"] = roi_y

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        k = int(p["blur"]) | 1
        if k >= 3:
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        roi = gray[roi_y:, :]
        Hroi = roi.shape[0]

        block = int(p["adapt_block"]) | 1
        block = max(3, block)
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block, float(p["adapt_c"]))

        mk = int(p["morph"])
        if mk >= 2:
            kernel = np.ones((mk, mk), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        out["binary"] = binary

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        left_x  = float(p["left_edge"])  * W
        right_x = float(p["right_edge"]) * W
        front_y = float(p["front_edge"]) * Hroi
        back_y  = float(p["back_edge"])  * Hroi
        use_ori = bool(int(p["use_orientation"]))
        htol    = float(p["horiz_tol_deg"])
        vtol    = float(p["vert_tol_deg"])

        for c in contours:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            length, width, angle = self._orientation(box)
            aspect = length / width
            (cx, cy_roi), _, _ = rect

            valid = (float(p["dash_min_area"]) <= area <= float(p["dash_max_area"])
                     and float(p["dash_min_aspect"]) <= aspect <= float(p["dash_max_aspect"]))

            # orientation classification (angle measured in ROI/image space)
            d_to_horiz = min(angle, 180.0 - angle)   # 0 if perfectly horizontal
            d_to_vert  = abs(angle - 90.0)            # 0 if perfectly vertical
            is_horiz = d_to_horiz <= htol
            is_vert  = d_to_vert <= vtol

            arm_vote = None
            if valid:
                if cx < left_x:
                    if (not use_ori) or is_horiz:
                        arm_vote = "left"
                elif cx > right_x:
                    if (not use_ori) or is_horiz:
                        arm_vote = "right"
                else:  # centre column
                    if cy_roi < front_y:
                        if (not use_ori) or is_vert:
                            arm_vote = "front"
                    elif cy_roi > back_y:
                        if (not use_ori) or is_vert:
                            arm_vote = "back"

            if arm_vote is not None:
                out["counts"][arm_vote] += 1

            box_global = (box + np.array([0, roi_y])).astype(int)
            out["dashes"].append({
                "box":    box_global,
                "valid":  valid,
                "vote":   arm_vote,
                "angle":  angle,
                "aspect": aspect,
                "area":   area,
            })

        out["arms"]["left"]  = out["counts"]["left"]  >= int(p["min_dashes_side"])
        out["arms"]["right"] = out["counts"]["right"] >= int(p["min_dashes_side"])
        out["arms"]["front"] = out["counts"]["front"] >= int(p["min_dashes_front"])
        out["arms"]["back"]  = out["counts"]["back"]  >= int(p["min_dashes_back"])
        out["detected"] = any(out["arms"].values())
        return out


def arms_to_str(arms: dict) -> str:
    present = [a for a in ARM_ORDER if arms.get(a)]
    return ",".join(present) if present else "none"


def draw_overlay(frame: np.ndarray, r: dict, params: dict) -> np.ndarray:
    """Shared debug renderer used by the node AND the calibrator."""
    vis = frame.copy()
    H, W = vis.shape[:2]
    roi_y = r["roi_y"]
    Hroi = H - roi_y

    # ROI + region boundaries
    cv2.line(vis, (0, roi_y), (W, roi_y), (255, 200, 0), 1)
    lx = int(float(params["left_edge"]) * W)
    rx = int(float(params["right_edge"]) * W)
    fy = roi_y + int(float(params["front_edge"]) * Hroi)
    by = roi_y + int(float(params["back_edge"]) * Hroi)
    cv2.line(vis, (lx, roi_y), (lx, H), (120, 120, 120), 1)
    cv2.line(vis, (rx, roi_y), (rx, H), (120, 120, 120), 1)
    cv2.line(vis, (lx, fy), (rx, fy), (120, 120, 120), 1)
    cv2.line(vis, (lx, by), (rx, by), (120, 120, 120), 1)

    # dashes: green=voted an arm, yellow=valid but unassigned, gray=rejected
    for d in r["dashes"]:
        if d["vote"] is not None:
            color = (0, 255, 0)
        elif d["valid"]:
            color = (0, 220, 220)
        else:
            color = (90, 90, 90)
        cv2.drawContours(vis, [d["box"]], 0, color, 2)

    arms = r["arms"]
    cnt = r["counts"]
    tag = arms_to_str(arms).upper()
    hud = (f"ARMS: {tag}   "
           f"F{cnt['front']} B{cnt['back']} L{cnt['left']} R{cnt['right']}")
    color = (0, 255, 0) if r["detected"] else (0, 165, 255)
    cv2.rectangle(vis, (0, 0), (W, 26), (40, 40, 40), -1)
    cv2.putText(vis, hud, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 node
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

        # debounce state
        self._pending = "none"
        self._pending_n = 0
        self._confirmed = "none"

        image_topic = self.get_parameter("image_topic").value
        self.sub_img       = self.create_subscription(
            Image, image_topic, self._on_image, 10)
        self.pub_detected  = self.create_publisher(Bool,   "/intersection/detected", 10)
        self.pub_arms      = self.create_publisher(String, "/intersection/arms",     10)
        self.pub_debug     = self.create_publisher(Image,  "/vision/intersection",   10)

        self.get_logger().info(
            f"IntersectionDetectorNode ready | topic={image_topic} | "
            f"roi_top={self.detector.params['roi_top']:.2f} | "
            f"orientation={'on' if int(self.detector.params['use_orientation']) else 'off'}")

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
        raw = arms_to_str(r["arms"])

        # temporal debounce
        n_need = int(self.detector.params["debounce"])
        if raw == self._pending:
            self._pending_n += 1
        else:
            self._pending = raw
            self._pending_n = 1
        if self._pending_n >= n_need:
            if self._pending != self._confirmed:
                self._confirmed = self._pending
                self.get_logger().info(f"INTERSECTION: {self._confirmed.upper()}")

        d_msg = Bool();   d_msg.data = (self._confirmed != "none")
        a_msg = String(); a_msg.data = self._confirmed
        self.pub_detected.publish(d_msg)
        self.pub_arms.publish(a_msg)

        if self.pub_debug.get_subscription_count() > 0:
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