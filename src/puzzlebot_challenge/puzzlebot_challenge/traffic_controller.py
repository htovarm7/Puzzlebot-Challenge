#!/usr/bin/env python3
"""
traffic_light_detector.py

CLI
---
  python3 traffic_light_detector.py                 # usa /camera/image_raw
  python3 traffic_light_detector.py --debug         # imprime conteos por color

Parámetros ROS2 
----------------------------------------
  ros2 param set /traffic_light_detector min_pixels 80
  ros2 param set /traffic_light_detector stable_frames 3
  ros2 param set /traffic_light_detector roi_fraction 0.5    # mitad izquierda
"""

import sys
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# Rangos HSV
RANGOS_HSV = {
    "red":    [(np.array([0,   80, 80]), np.array([8,   255, 255])),
               (np.array([172, 80, 80]), np.array([180, 255, 255]))],
    "yellow": [(np.array([18,  80, 80]), np.array([32,  255, 255]))],
    "green":  [(np.array([45,  80, 80]), np.array([85,  255, 255]))],
}


# Detector puro sin ROS
class TrafficLightDetection:
    """Lógica de visión pura. No depende de ROS — testeable con cualquier imagen BGR."""

    def __init__(self, min_pixels=50, roi_fraction=0.5):
        self.min_pixels = min_pixels
        self.roi_fraction = roi_fraction   

    def detect_state(self, image_bgr):
        """
        :param image_bgr: imagen BGR de OpenCV
        :return: ('red'|'yellow'|'green'|'none', dict de conteos por color)
        """
        
        if image_bgr is None or image_bgr.size == 0:
            return "none", {"red": 0, "yellow": 0, "green": 0}

        # ROI = mitad izquierda 
        w = image_bgr.shape[1]
        cut = max(1, int(w * self.roi_fraction))
        zona = image_bgr[:, :cut]

        hsv = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)

        pixeles = {
            color: sum(cv2.countNonZero(cv2.inRange(hsv, lo, hi)) for lo, hi in rangos)
            for color, rangos in RANGOS_HSV.items()
        }

        mejor = max(pixeles, key=pixeles.get)
        state = mejor if pixeles[mejor] >= self.min_pixels else "none"
        return state, pixeles

# Nodo ROS2
class TrafficLightNode(Node):

    def __init__(self, debug=False):
        super().__init__('traffic_light_detector')

        # Parámetros 
        self.declare_parameter('min_pixels',    50,
            ParameterDescriptor(description='Mínimo de pixeles para considerar válida una detección'))
        self.declare_parameter('stable_frames', 3,
            ParameterDescriptor(description='Frames consecutivos requeridos para confirmar un cambio'))
        self.declare_parameter('roi_fraction',  0.5,
            ParameterDescriptor(description='Fracción horizontal a analizar desde la izquierda (0.0-1.0)'))
        self.declare_parameter('image_topic',   '/camera/image_raw',
            ParameterDescriptor(description='Topic de imagen de entrada'))

        self.bridge = CvBridge()
        self.detector = TrafficLightDetection(
            min_pixels   = self.get_parameter('min_pixels').value,
            roi_fraction = self.get_parameter('roi_fraction').value,
        )

        image_topic = self.get_parameter('image_topic').value
        self.sub_img = self.create_subscription(Image, image_topic, self._on_image, 10)
        self.pub_state = self.create_publisher(String, '/traffic_light', 10)

        # Filtro de histéresis 
        self._current_state = "none"      
        self._candidate     = "none"      
        self._candidate_count = 0

        # Republica el estado a 5 Hz
        self.create_timer(0.2, self._republish)

        self.debug = debug
        self.get_logger().info(
            f"TrafficLightNode listo | topic={image_topic} | "
            f"min_pixels={self.detector.min_pixels} | "
            f"stable_frames={self.get_parameter('stable_frames').value}"
        )

    # Callback de cámara 
    def _on_image(self, msg: Image):
        self.detector.min_pixels   = self.get_parameter('min_pixels').value
        self.detector.roi_fraction = self.get_parameter('roi_fraction').value
        stable_frames              = self.get_parameter('stable_frames').value

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Fallo conversión de imagen: {e}")
            return

        detected, counts = self.detector.detect_state(frame)

        if self.debug:
            self.get_logger().info(
                f"R={counts['red']:5d} Y={counts['yellow']:5d} G={counts['green']:5d} → {detected}"
            )

        # Histéresis
        if detected == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = detected
            self._candidate_count = 1

        if self._candidate_count >= stable_frames and self._candidate != self._current_state:
            self.get_logger().info(
                f"Estado: {self._current_state} → {self._candidate}"
            )
            self._current_state = self._candidate
            self._publish_now()

    # Publicar estado
    def _publish_now(self):
        msg = String()
        msg.data = self._current_state
        self.pub_state.publish(msg)

    def _republish(self):
        self._publish_now()

def main(args=None):
    debug = '--debug' in sys.argv

    rclpy.init(args=args)
    node = TrafficLightNode(debug=debug)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('Deteniendo detector.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()