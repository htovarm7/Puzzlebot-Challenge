#!/usr/bin/env python3
"""
traffic_light_circles.py
========================
Detector de semáforo basado en detección de círculos (HoughCircles) +
clasificación HSV dentro del círculo detectado.

Pensado para semáforos que son una pantalla mostrando un círculo de color
(no semáforos físicos). Mucho más robusto que HSV puro porque:
  - Ignora fondos saturados (carros, ropa, paredes) que no son redondos.
  - Solo evalúa regiones que tienen forma circular.
  - Como YOLO pero sin necesidad de modelo, GPU, ni entrenamiento.

Publica en /traffic_light con la misma interfaz que el detector HSV puro.
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


# Rangos HSV — mismos que los otros detectores
RANGOS_HSV = {
    "red":    [(np.array([0,   80, 80]), np.array([8,   255, 255])),
               (np.array([172, 80, 80]), np.array([180, 255, 255]))],
    "yellow": [(np.array([18,  80, 80]), np.array([32,  255, 255]))],
    "green":  [(np.array([45,  80, 80]), np.array([85,  255, 255]))],
}

def classify_color_in_circle(frame_bgr, cx, cy, r):
    """
    Recorta el círculo (con una máscara para no contar pixeles fuera del círculo)
    y devuelve (color_dominante, conteos, fracción_del_círculo_coloreada).
    """
    h, w = frame_bgr.shape[:2]
    # Bounding box del círculo, clipped a la imagen
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    if x2 <= x1 or y2 <= y1:
        return "none", {"red": 0, "yellow": 0, "green": 0}, 0.0

    crop = frame_bgr[y1:y2, x1:x2]

    # Máscara circular dentro del crop (para no contar las esquinas del bbox)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    local_cx = cx - x1
    local_cy = cy - y1
    cv2.circle(mask, (local_cx, local_cy), r, 255, -1)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    counts = {}
    for color, rangos in RANGOS_HSV.items():
        color_mask = sum(cv2.inRange(hsv, lo, hi) for lo, hi in rangos)
        # Intersección con la máscara circular
        final = cv2.bitwise_and(color_mask, color_mask, mask=mask)
        counts[color] = cv2.countNonZero(final)

    circle_area = cv2.countNonZero(mask)
    best = max(counts, key=counts.get)
    fill_ratio = counts[best] / max(1, circle_area)

    return best, counts, fill_ratio


class TrafficLightCirclesNode(Node):

    def __init__(self):
        super().__init__('traffic_light_detector')

        # ── Parámetros ───────────────────────────────────────────────────────
        self.declare_parameter('image_topic',  '/camera/image_raw')
        self.declare_parameter('output_topic', '/traffic_light')
        self.declare_parameter('stable_frames', 3)
        self.declare_parameter('publish_rate',  5.0)
        self.declare_parameter('debug',         False)

        # HoughCircles params
        self.declare_parameter('min_radius',    8)     # px — círculos más chicos se ignoran
        self.declare_parameter('max_radius',    80)    # px — círculos más grandes se ignoran
        self.declare_parameter('min_dist',      30)    # mínima distancia entre centros
        self.declare_parameter('param1',        100.0) # threshold del Canny interno
        self.declare_parameter('param2',        25.0)  # threshold del acumulador (más bajo = más círculos)
        self.declare_parameter('blur_ksize',    7)     # tamaño del kernel del GaussianBlur

        # Aceptación
        self.declare_parameter('min_fill_ratio', 0.20) # % mínimo del círculo que debe estar coloreado

        image_topic  = self.get_parameter('image_topic').value
        output_topic = self.get_parameter('output_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        self.bridge = CvBridge()
        self.sub_img   = self.create_subscription(Image, image_topic, self._on_image, 10)
        self.pub_state = self.create_publisher(String, output_topic, 10)

        # Histéresis
        self._current_state   = "none"
        self._candidate       = "none"
        self._candidate_count = 0

        self.create_timer(1.0 / publish_rate, self._republish)

        self.get_logger().info(
            f"[Circles] listo | in={image_topic} → out={output_topic} | "
            f"radius=[{self.get_parameter('min_radius').value}-"
            f"{self.get_parameter('max_radius').value}px]"
        )

    def _on_image(self, msg: Image):
        # Sync de parámetros (cambio en caliente)
        min_r       = self.get_parameter('min_radius').value
        max_r       = self.get_parameter('max_radius').value
        min_dist    = self.get_parameter('min_dist').value
        param1      = self.get_parameter('param1').value
        param2      = self.get_parameter('param2').value
        blur_ksize  = self.get_parameter('blur_ksize').value
        min_fill    = self.get_parameter('min_fill_ratio').value
        stable_frames = self.get_parameter('stable_frames').value
        debug       = self.get_parameter('debug').value

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Conversión falló: {e}")
            return

        # ── Detección de círculos ────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Kernel impar
        k = max(1, blur_ksize | 1)
        gray = cv2.GaussianBlur(gray, (k, k), 0)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is None:
            if debug:
                self.get_logger().info("  No circles detected")
            self._update("none", stable_frames)
            return

        circles = np.round(circles[0]).astype(int)

        # ── Evalúa cada círculo, escoge el más "lleno" de color válido ──────
        best_state = "none"
        best_score = 0.0    # fill_ratio del mejor candidato

        for (cx, cy, r) in circles:
            color, counts, fill = classify_color_in_circle(frame, cx, cy, r)
            if color == "none" or fill < min_fill:
                if debug:
                    self.get_logger().info(
                        f"  circle ({cx},{cy},r={r}) → {color} fill={fill:.2f} (rejected)"
                    )
                continue

            if debug:
                self.get_logger().info(
                    f"  circle ({cx},{cy},r={r}) → {color} fill={fill:.2f} ✓"
                )

            # Si hay varios círculos válidos (varios candidatos), quédate con el más saturado
            if fill > best_score:
                best_score = fill
                best_state = color

        self._update(best_state, stable_frames)

    def _update(self, detected, stable_frames):
        if detected == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = detected
            self._candidate_count = 1

        if self._candidate_count >= stable_frames and self._candidate != self._current_state:
            self.get_logger().info(f"🚦 {self._current_state} → {self._candidate}")
            self._current_state = self._candidate
            self._publish_now()

    def _publish_now(self):
        msg = String()
        msg.data = self._current_state
        self.pub_state.publish(msg)

    def _republish(self):
        self._publish_now()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightCirclesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()