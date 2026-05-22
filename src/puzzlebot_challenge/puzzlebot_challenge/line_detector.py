#!/usr/bin/env python3
"""
line_detector.py

Topics
------
Sub : /camera/image_raw   (sensor_msgs/Image)
Pub : /line/shift         (std_msgs/Float32)   
      /line/angle         (std_msgs/Float32)   
      /line/detected      (std_msgs/Bool)
      /vision/line        (sensor_msgs/Image) 
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
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

_DEFAULT_PARAMS = {
    "T_init":      185,
    "T_min":       127,
    "T_max":       222,
    "dark_min":    1.0,
    "dark_max":    6.0,
    "roi_top":     0.60,
    "min_area":    300,
    "blur":        21,
    "morph":       9,
    "n_track_lines": 3,
    # Intersection detection: min fraction of frame width a horizontal contour must span
    "intersection_white_frac": 0.55,
}


def _load_params_yaml(path: str) -> dict | None:
    """Read a YAML file saved by the tuner. Returns None if the file does not exist or fails to load."""
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

class LineDetection:
    """Same mathematics as `complex_lines.detect`, without trackbars."""

    def __init__(self, params: dict | None = None):
        self.params = dict(_DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        # Persistent adaptive threshold between frames.
        self._T_state: int = int(self.params["T_init"])

    def _balance(self, gray: np.ndarray) -> tuple[np.ndarray | None, int, int]:
        p = self.params
        T = self._T_state
        direction = 0
        h = gray.shape[0]
        y_off = int(h * p["roi_top"])

        for _ in range(10):
            _, binary = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY_INV)
            crop = binary[y_off:, :]
            area = crop.shape[0] * crop.shape[1]
            if area == 0:
                return None, T, y_off
            perc = 100.0 * cv2.countNonZero(crop) / area

            if perc > p["dark_max"]:
                if T <= p["T_min"] or direction == 1:
                    self._T_state = T
                    return crop, T, y_off
                T -= 10
                direction = -1
            elif perc < p["dark_min"]:
                if T >= p["T_max"] or direction == -1:
                    self._T_state = T
                    return crop, T, y_off
                T += 10
                direction = 1
            else:
                self._T_state = T
                return crop, T, y_off

        self._T_state = T
        return None, T, y_off

    def detect(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns a dict with:
          detected : bool
          shift    : float  (px from the horizontal center; + = right, - = left)
          angle    : float  (deg, 0..180, 90 = vertical)
          T_used   : int    (threshold finally applied)
          y_off    : int    (top line of the ROI, for overlay)
          contour  : np.ndarray | None  (contour already translated to global coords)
          box      : np.ndarray | None  (oriented box in global coords, 4 corners)
          top_mid, bottom_mid : tuple (x,y) in global coords for overlay
        """
        out: dict = {
            "detected":     False,
            "intersection": False,
            "shift":        0.0,
            "angle":        90.0,
            "T_used":       self._T_state,
            "y_off":        0,
            "contour":      None,
            "box":          None,
            "top_mid":      None,
            "bottom_mid":   None,
        }

        if frame_bgr is None or frame_bgr.size == 0:
            return out

        p = self.params
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if p["blur"] >= 3:
            k = int(p["blur"]) | 1   
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        binary_roi, T_used, y_off = self._balance(gray)
        out["T_used"] = T_used
        out["y_off"]  = y_off
        if binary_roi is None:
            return out

        # Morfología
        mk = int(p["morph"])
        if mk >= 2:
            kernel = np.ones((mk, mk), np.uint8)
            binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN,  kernel)
            binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= p["min_area"]]
        if not contours:
            return out

        # Intersection detection: look for a contour whose bounding box is wide
        # and flat — the crossing/stop line taped perpendicular to travel direction.
        # Criteria: bounding width > 25% of frame AND width > 2× height.
        frame_w = binary_roi.shape[1]
        min_span = float(p.get("intersection_white_frac", 0.55)) * frame_w  # reuse param as span fraction
        out["intersection"] = any(
            bw >= min_span and bw >= 2.0 * bh
            for c in contours
            for (_, _, bw, bh) in [cv2.boundingRect(c)]
        )

        # Keep only the N largest blobs — floor artifacts are always smaller than tape
        n_lines = int(p.get("n_track_lines", 3))
        if len(contours) > n_lines:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n_lines]

        # Sort by horizontal centroid and pick the MEDIAN → always the center line
        def _cx(c):
            m = cv2.moments(c)
            return int(m["m10"] / m["m00"]) if m["m00"] else 0

        contours.sort(key=_cx)
        line = contours[len(contours) // 2]

        rect = cv2.minAreaRect(line)
        (cx, _cy), _, _ = rect
        box = cv2.boxPoints(rect)
        box = box[np.argsort(box[:, 1])]
        top_mid    = ((box[0] + box[1]) / 2).astype(int)
        bottom_mid = ((box[2] + box[3]) / 2).astype(int)

        dx = float(bottom_mid[0] - top_mid[0])
        dy = float(bottom_mid[1] - top_mid[1])
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if angle < 0:
            angle += 180.0

        roi_center_x = binary_roi.shape[1] // 2
        shift = float(cx - roi_center_x)

        contour_global = line + np.array([[0, y_off]])
        box_global     = (box + np.array([0, y_off])).astype(int)
        top_mid_g      = (int(top_mid[0]),    int(top_mid[1]    + y_off))
        bottom_mid_g   = (int(bottom_mid[0]), int(bottom_mid[1] + y_off))

        out.update({
            "detected":   True,
            "shift":      shift,
            "angle":      angle,
            "contour":    contour_global,
            "box":        box_global,
            "top_mid":    top_mid_g,
            "bottom_mid": bottom_mid_g,
        })
        return out


class LineDetectorNode(Node):

    def __init__(self):
        super().__init__("line_detector")

        self.declare_parameter(
            "image_topic", "/camera/image_raw",
            ParameterDescriptor(description="Tópico de imagen de entrada"))
        self.declare_parameter(
            "params_config", "",
            ParameterDescriptor(description="Ruta opcional a line_params.yaml"))

        for k, v in _DEFAULT_PARAMS.items():
            self.declare_parameter(
                k, float(v) if isinstance(v, float) else v,
                ParameterDescriptor(description=f"Param de visión: {k}"))

        yaml_path = self.get_parameter("params_config").value
        yaml_params = _load_params_yaml(yaml_path)
        if yaml_params:
            self.get_logger().info(f"Parámetros cargados desde: {yaml_path}")
            for k, v in yaml_params.items():
                self.set_parameters(
                    [rclpy.parameter.Parameter(k,
                        rclpy.parameter.Parameter.Type.DOUBLE
                        if isinstance(v, float)
                        else rclpy.parameter.Parameter.Type.INTEGER,
                        v)])
        else:
            self.get_logger().info("Usando parámetros por defecto (sin YAML)")

        self.bridge   = CvBridge()
        self.detector = LineDetection(self._snapshot_params())

        image_topic = self.get_parameter("image_topic").value
        self.sub_img           = self.create_subscription(
            Image, image_topic, self._on_image, 10)
        self.pub_shift         = self.create_publisher(Float32, "/line/shift",        10)
        self.pub_angle         = self.create_publisher(Float32, "/line/angle",        10)
        self.pub_detected      = self.create_publisher(Bool,    "/line/detected",     10)
        self.pub_intersection  = self.create_publisher(Bool,    "/line/intersection", 10)
        self.pub_debug         = self.create_publisher(Image,   "/vision/line",       10)

        self.get_logger().info(
            f"LineDetectorNode listo | topic={image_topic} | "
            f"min_area={self.detector.params['min_area']} | "
            f"roi_top={self.detector.params['roi_top']:.2f}"
        )

    def _snapshot_params(self) -> dict:
        """Lee los parámetros ROS y devuelve un dict para LineDetection."""
        snap = {}
        for k in _DEFAULT_PARAMS:
            snap[k] = self.get_parameter(k).value
        return snap

    def _on_image(self, msg: Image):
        self.detector.params.update(self._snapshot_params())

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Fallo conversión de imagen: {e}")
            return

        result = self.detector.detect(frame)

        s_msg = Float32(); s_msg.data = result["shift"]
        a_msg = Float32(); a_msg.data = result["angle"]
        d_msg = Bool();    d_msg.data = result["detected"]
        i_msg = Bool();    i_msg.data = result["intersection"]
        self.pub_shift.publish(s_msg)
        self.pub_angle.publish(a_msg)
        self.pub_detected.publish(d_msg)
        self.pub_intersection.publish(i_msg)

        self._publish_debug(frame, result)

    def _publish_debug(self, frame: np.ndarray, r: dict):
        if self.pub_debug.get_subscription_count() == 0:
            return

        vis = frame.copy()
        y_off = r["y_off"]

        cv2.line(vis, (0, y_off), (vis.shape[1], y_off), (255, 200, 0), 1)
        fx = vis.shape[1] // 2
        cv2.line(vis, (fx, y_off), (fx, vis.shape[0]), (0, 255, 255), 1)

        if r["detected"]:
            cv2.drawContours(vis, [r["contour"]], -1, (0, 255, 0), 2)
            cv2.drawContours(vis, [r["box"]],     0, (255, 0, 255), 1)
            cv2.line(vis, r["top_mid"], r["bottom_mid"], (0, 0, 255), 3)
            intr_tag = "  [INTERSECCION]" if r["intersection"] else ""
            hud = (f"T={r['T_used']}  angle={r['angle']:5.1f}  "
                   f"shift={r['shift']:+.0f}{intr_tag}")
            color = (0, 255, 255) if r["intersection"] else (255, 255, 255)
        else:
            hud = f"T={r['T_used']}  NO LINE"
            color = (0, 165, 255)

        cv2.putText(vis, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LineDetectorNode()
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