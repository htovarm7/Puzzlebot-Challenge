#!/usr/bin/env python3
"""
traffic_controller.py

Detecta el estado del semáforo (red / yellow / green / none) y publica
el resultado en /traffic_light.

Parámetros ROS2
---------------
  ros2 param set /traffic_light_detector min_pixels    80
  ros2 param set /traffic_light_detector stable_frames 3
  ros2 param set /traffic_light_detector roi_fraction  1.0
  ros2 param set /traffic_light_detector hsv_config    /ruta/traffic_hsv.yaml

Los rangos HSV se cargan desde el YAML indicado en `hsv_config`. Si el
parámetro está vacío o el archivo no existe, se usan los valores por defecto
hardcodeados. Usa `ros2 run puzzlebot_challenge hsv_calibrator` para generar
o ajustar ese YAML.
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

    def __init__(self, min_pixels: int = 50, roi_fraction: float = 1.0,
                 hsv_ranges: dict | None = None):
        self.min_pixels   = min_pixels
        self.roi_fraction = roi_fraction
        self.hsv_ranges   = hsv_ranges if hsv_ranges is not None else _DEFAULT_RANGOS_HSV

    def detect_state(self, image_bgr: np.ndarray) -> tuple[str, dict]:
        """
        Returns (state, counts) donde state es 'red'|'yellow'|'green'|'none'
        y counts es un dict con el número de píxeles detectados por color.
        """
        if image_bgr is None or image_bgr.size == 0:
            return "none", {c: 0 for c in self.hsv_ranges}

        w   = image_bgr.shape[1]
        cut = max(1, int(w * self.roi_fraction))
        roi = image_bgr[:, :cut]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        counts = {
            color: sum(
                cv2.countNonZero(cv2.inRange(hsv, lo, hi))
                for lo, hi in rangos
            )
            for color, rangos in self.hsv_ranges.items()
        }

        best  = max(counts, key=counts.get)
        state = best if counts[best] >= self.min_pixels else "none"
        return state, counts


# ── Nodo ROS2 ─────────────────────────────────────────────────────────────────
class TrafficLightNode(Node):

    def __init__(self, debug: bool = False):
        super().__init__("traffic_light_detector")

        self.declare_parameter(
            "min_pixels", 50,
            ParameterDescriptor(description="Mínimo de píxeles para validar una detección"))
        self.declare_parameter(
            "stable_frames", 3,
            ParameterDescriptor(description="Frames consecutivos para confirmar un cambio"))
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
            min_pixels   = self.get_parameter("min_pixels").value,
            roi_fraction = self.get_parameter("roi_fraction").value,
            hsv_ranges   = hsv_ranges,
        )

        image_topic      = self.get_parameter("image_topic").value
        self.sub_img     = self.create_subscription(Image, image_topic, self._on_image, 10)
        self.pub_state   = self.create_publisher(String, "/traffic_light", 10)

        self._current_state   = "none"
        self._candidate       = "none"
        self._candidate_count = 0

        self.create_timer(0.2, self._republish)

        self.debug = debug
        self.get_logger().info(
            f"TrafficLightNode listo | topic={image_topic} | "
            f"min_pixels={self.detector.min_pixels} | "
            f"stable_frames={self.get_parameter('stable_frames').value}"
        )

    # ── Callback de cámara ────────────────────────────────────────────────────
    def _on_image(self, msg: Image):
        self.detector.min_pixels   = self.get_parameter("min_pixels").value
        self.detector.roi_fraction = self.get_parameter("roi_fraction").value
        stable_frames              = self.get_parameter("stable_frames").value

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Fallo conversión de imagen: {e}")
            return

        detected, counts = self.detector.detect_state(frame)

        if self.debug:
            self.get_logger().info(
                f"R={counts.get('red',0):5d} "
                f"Y={counts.get('yellow',0):5d} "
                f"G={counts.get('green',0):5d} → {detected}"
            )

        if detected == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = detected
            self._candidate_count = 1

        if self._candidate_count >= stable_frames and self._candidate != self._current_state:
            self.get_logger().info(
                f"Semaforo: {self._current_state} → {self._candidate}"
            )
            self._current_state = self._candidate
            self._publish_now()

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
        node.get_logger().warn("Deteniendo detector.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
