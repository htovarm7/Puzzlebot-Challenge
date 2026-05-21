#!/usr/bin/env python3
"""
line_viewer.py
==============
Muestra en una ventana OpenCV la imagen de debug del line_detector.
Requiere display (local o SSH con X11 forwarding: ssh -X).

Suscribe a /vision/line (sensor_msgs/Image).
"""

import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

WINDOW = 'Line Follower — Vision'


class LineViewerNode(Node):

    def __init__(self):
        super().__init__('line_viewer')
        self._bridge = CvBridge()
        self._frame = None
        self._lock = threading.Lock()
        self.create_subscription(Image, '/vision/line', self._on_image, 10)
        self.get_logger().info('LineViewer listo — esperando /vision/line ...')

    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._lock:
                self._frame = frame
        except Exception as e:
            self.get_logger().warn(f'Error convirtiendo imagen: {e}')

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


def main(args=None):
    rclpy.init(args=args)
    node = LineViewerNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 480)

    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'Esperando imagen...', (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow(WINDOW, placeholder)

    try:
        while True:
            frame = node.get_frame()
            if frame is not None:
                # Escala 2x para que sea más visible en la ventana
                display = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(WINDOW, display)

            key = cv2.waitKey(33) & 0xFF  # ~30 Hz
            if key == ord('q') or key == 27:  # Q o ESC para cerrar
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
