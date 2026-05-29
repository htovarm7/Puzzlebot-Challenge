#!/usr/bin/env python3
"""
sign_viewer.py — Muestra en ventana OpenCV el feed anotado de sign_detector_offload.
Suscribe a /vision/signs (sensor_msgs/Image) — ya trae bounding boxes dibujadas.

Uso:
  ros2 run puzzlebot_challenge sign_viewer
"""

import sys
# Use system OpenCV (GTK backend) instead of pip OpenCV (Qt backend)
sys.path.insert(0, '/usr/lib/python3/dist-packages')

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

WINDOW = "Sign Detector — Bounding Boxes"


class SignViewerNode(Node):

    def __init__(self):
        super().__init__("sign_viewer")
        self._bridge = CvBridge()
        self._latest = None

        self.create_subscription(Image, "/vision/signs", self._on_image, 10)
        self.create_timer(0.033, self._show)  # ~30 fps display

        self.get_logger().info("SignViewer listo — suscrito a /vision/signs")

    def _on_image(self, msg: Image):
        try:
            self._latest = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")

    def _show(self):
        if self._latest is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Esperando /vision/signs ...", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        else:
            frame = self._latest
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = SignViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
