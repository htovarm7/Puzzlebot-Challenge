#!/usr/bin/env python3
"""
traffic_controller.py

Detecta el estado del semáforo (red / yellow / green / none) y publica
el resultado en /traffic_light.

La detección combina filtrado HSV + roundness (circularidad de contornos),
por lo que solo el blob más circular de cada color cuenta, ignorando manchas
de color aleatorias en el fondo.

Parámetros ROS2
---------------
  ros2 param set /traffic_light_detector min_area       200
  ros2 param set /traffic_light_detector min_circularity 0.55
  ros2 param set /traffic_light_detector stable_frames  3
  ros2 param set /traffic_light_detector roi_fraction   1.0
  ros2 param set /traffic_light_detector hsv_config     /ruta/traffic_hsv.yaml

Usa `ros2 run puzzlebot_challenge hsv_calibrator` para generar el YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# ── Rangos HSV por defecto (fallback si no hay YAML) ─────────────────────────
_DEFAULT_RANGOS_HSV = {
    "red":    [(np.array([0,   80, 80]), np.array([8,   255, 255])),
               (np.array([172, 80, 80]), np.array([180, 255, 255]))],
    "yellow": [(np.array([18,  80, 80]), np.array([32,  255, 255]))],
    "green":  [(np.array([45,  80, 80]), np.array([85,  255, 255]))],
}


def _load_hsv_yaml(path: str) -> dict | None:
    """
    Carga un archivo traffic_hsv.yaml y lo convierte al formato interno
    {color: [(lo_array, hi_array), ...]}.
    Devuelve None si la ruta está vacía o el archivo no existe.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            raw = yaml.safe_load(f)
        result = {}
        for color, ranges in raw.items():
            pairs = []
            for rk in sorted(ranges.keys()):   # range1 antes que range2
                rv = ranges[rk]
                lo = np.array([rv["h_min"], rv["s_min"], rv["v_min"]])
                hi = np.array([rv["h_max"], rv["s_max"], rv["v_max"]])
                pairs.append((lo, hi))
            result[color] = pairs
        return result
    except Exception:
        return None


# ── Lógica de visión pura (sin ROS) ──────────────────────────────────────────
class TrafficLightDetection:
    """Testeable independientemente de ROS con cualquier imagen BGR."""

    def __init__(self, min_area: int = 200, min_circularity: float = 0.60,
                 roi_fraction: float = 1.0, hsv_ranges: dict | None = None):
        self.min_area        = min_area
        self.min_circularity = min_circularity
        self.roi_fraction    = roi_fraction
        self.hsv_ranges      = hsv_ranges if hsv_ranges is not None else _DEFAULT_RANGOS_HSV

    @staticmethod
    def _best_circle_score(mask: np.ndarray) -> tuple[float, float, tuple]:
        """
        Finds the most circular contour in the mask.
        Returns (circularity, area, center_xy) of the best candidate,
        or (0, 0, None) if none passes the basic size filter.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_circ, best_area, best_center = 0.0, 0.0, None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:               # ignore tiny noise
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = (4.0 * np.pi * area) / (perim ** 2)
            if circ > best_circ:
                best_circ   = circ
                best_area   = area
                M           = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    best_center = (cx, cy)
        return best_circ, best_area, best_center

    def detect_state(self, image_bgr: np.ndarray) -> tuple[str, dict]:
        """
        Returns (state, info) donde:
          state → 'red' | 'yellow' | 'green' | 'none'
          info  → dict with circularity, area, and center per color
        """
        if image_bgr is None or image_bgr.size == 0:
            return "none", {}

        w   = image_bgr.shape[1]
        cut = max(1, int(w * self.roi_fraction))
        roi = image_bgr[:, :cut]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        scores: dict[str, dict] = {}
        for color, rangos in self.hsv_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in rangos:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

            circ, area, center = self._best_circle_score(mask)
            scores[color] = {"circularity": circ, "area": area, "center": center}

        # A color qualifies only if its best blob is large enough AND round enough
        valid = {
            c: s for c, s in scores.items()
            if s["area"] >= self.min_area and s["circularity"] >= self.min_circularity
        }

        if not valid:
            return "none", scores

        # Among valid colors pick the one with the highest circularity
        best = max(valid, key=lambda c: valid[c]["circularity"])
        return best, scores


# ── Nodo ROS2 ─────────────────────────────────────────────────────────────────
class TrafficLightNode(Node):

    def __init__(self, debug: bool = False):
        super().__init__("traffic_light_detector")

        self.declare_parameter(
            "min_area", 200,
            ParameterDescriptor(description="Área mínima (px²) del blob circular"))
        self.declare_parameter(
            "min_circularity", 0.60,
            ParameterDescriptor(description="Circularidad mínima 0-1 (1=círculo perfecto)"))
        self.declare_parameter(
            "stable_frames", 2,
            ParameterDescriptor(description="Frames consecutivos para confirmar un color"))
        self.declare_parameter(
            "none_frames", 8,
            ParameterDescriptor(description="Frames consecutivos de NONE para volver a none"))
        self.declare_parameter(
            "roi_fraction", 1.0,
            ParameterDescriptor(description="Fracción horizontal del ROI desde la izquierda (0-1)"))
        self.declare_parameter(
            "image_topic", "/camera/image_raw",
            ParameterDescriptor(description="Tópico de imagen de entrada"))
        self.declare_parameter(
            "hsv_config", "",
            ParameterDescriptor(description="Ruta al YAML de calibración HSV (traffic_hsv.yaml)"))

        hsv_path   = self.get_parameter("hsv_config").value
        hsv_ranges = _load_hsv_yaml(hsv_path)

        if hsv_ranges:
            self.get_logger().info(f"Rangos HSV cargados desde: {hsv_path}")
        else:
            self.get_logger().info("Usando rangos HSV por defecto (sin YAML)")
            hsv_ranges = None

        self.bridge   = CvBridge()
        self.detector = TrafficLightDetection(
            min_area        = self.get_parameter("min_area").value,
            min_circularity = self.get_parameter("min_circularity").value,
            roi_fraction    = self.get_parameter("roi_fraction").value,
            hsv_ranges      = hsv_ranges,
        )

        image_topic      = self.get_parameter("image_topic").value
        self.sub_img     = self.create_subscription(Image, image_topic, self._on_image, 10)
        self.pub_state   = self.create_publisher(String, "/traffic_light", 10)
        self.pub_debug   = self.create_publisher(Image, "/vision/traffic", 10)

        self._current_state   = "none"
        self._candidate       = "none"
        self._candidate_count = 0

        self.create_timer(0.2, self._republish)

        self.debug = debug
        self.get_logger().info(
            f"TrafficLightNode listo | topic={image_topic} | "
            f"min_area={self.detector.min_area} | "
            f"min_circularity={self.detector.min_circularity} | "
            f"stable_frames={self.get_parameter('stable_frames').value} | "
            f"none_frames={self.get_parameter('none_frames').value}"
        )

    # ── Callback de cámara ────────────────────────────────────────────────────
    def _on_image(self, msg: Image):
        self.detector.min_area        = self.get_parameter("min_area").value
        self.detector.min_circularity = self.get_parameter("min_circularity").value
        self.detector.roi_fraction    = self.get_parameter("roi_fraction").value
        stable_frames                 = self.get_parameter("stable_frames").value
        none_frames                   = self.get_parameter("none_frames").value

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Fallo conversión de imagen: {e}")
            return

        detected, scores = self.detector.detect_state(frame)

        # Log every frame
        s = scores.get(detected, {})
        self.get_logger().info(
            f"[{detected.upper():6s}] circ={s.get('circularity', 0):.2f} "
            f"area={s.get('area', 0):.0f} | confirmed={self._current_state.upper()}"
        )

        # Asymmetric hysteresis:
        #   color → confirmed quickly (stable_frames, default 2)
        #   none  → confirmed slowly  (none_frames,   default 8)
        threshold = none_frames if detected == "none" else stable_frames

        if detected == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = detected
            self._candidate_count = 1

        if self._candidate_count >= threshold and self._candidate != self._current_state:
            self._current_state = self._candidate
            self._publish_now()
            self.get_logger().info(f"*** STATE → {self._current_state.upper()} ***")

        # Debug image → /vision/traffic
        self._publish_debug(frame, detected, scores)

    _OVERLAY_BGR = {
        "red":    (0,   0,   220),
        "yellow": (0,   210, 210),
        "green":  (0,   200, 0),
        "none":   (80,  80,  80),
    }

    def _publish_debug(self, frame: np.ndarray, detected: str, scores: dict):
        if self.pub_debug.get_subscription_count() == 0:
            return

        vis = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Draw HSV mask overlay + circle marker for every color
        for color, rangos in self.detector.hsv_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in rangos:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

            s      = scores.get(color, {})
            center = s.get("center")
            area   = s.get("area", 0)
            circ   = s.get("circularity", 0)
            is_det = (color == detected)

            if is_det and center:
                fill = np.full_like(frame, self._OVERLAY_BGR[color])
                vis[mask > 0] = cv2.addWeighted(frame, 0.3, fill, 0.7, 0)[mask > 0]

            if center and area > 0:
                radius      = int(np.sqrt(area / np.pi))
                border_clr  = self._OVERLAY_BGR[color]
                thickness   = 3 if is_det else 1
                cv2.circle(vis, center, radius, border_clr, thickness)
                cv2.putText(vis, f"{circ:.2f}", (center[0] + radius + 2, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, border_clr, 1)

        banner_color = self._OVERLAY_BGR.get(detected, (80, 80, 80))
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), banner_color, -1)
        cv2.putText(vis, detected.upper(), (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        parts = [
            f"{c[0].upper()}:circ={scores.get(c,{}).get('circularity',0):.2f}"
            for c in ("red", "yellow", "green")
        ]
        cv2.putText(vis, "  ".join(parts), (4, vis.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(msg)

    # ── Publicación ───────────────────────────────────────────────────────────
    def _publish_now(self):
        msg      = String()
        msg.data = self._current_state
        self.pub_state.publish(msg)

    def _republish(self):
        self._publish_now()


def main(args=None):
    debug = "--debug" in sys.argv

    rclpy.init(args=args)
    node = TrafficLightNode(debug=debug)

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
