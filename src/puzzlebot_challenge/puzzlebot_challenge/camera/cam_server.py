#!/usr/bin/env python3
"""MJPEG server: subscribes to /camera/image_raw and serves it over HTTP.

Useful for viewing the video from a browser on another machine without
installing anything, and for visualizing overlays produced by other nodes
(the source publishes to the topic, the camera is not reopened)."""

import threading
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from flask import Flask, Response


class CamServer(Node):

    def __init__(self):
        super().__init__('cam_server')

        self.declare_parameter('topic', '/camera/image_raw')
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 8080)
        self.declare_parameter('jpeg_quality', 75)

        self.bridge = CvBridge()
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self._quality = int(self.get_parameter('jpeg_quality').value)

        topic = self.get_parameter('topic').value
        self.create_subscription(Image, topic, self._on_image, 10)
        self.get_logger().info(f"Subscribed to {topic}")

        host = self.get_parameter('host').value
        port = int(self.get_parameter('port').value)
        self._app = Flask(__name__)
        self._app.add_url_rule('/', 'index', self._index)
        self._server_thread = threading.Thread(
            target=self._run_flask, args=(host, port), daemon=True
        )
        self._server_thread.start()
        self.get_logger().info(f"MJPEG server at http://{host}:{port}/")

    def _on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            return
        with self._lock:
            self._latest_jpeg = jpg.tobytes()

    def _frames(self):
        while True:
            with self._lock:
                data = self._latest_jpeg
            if data is None:
                # Wait for the first frame without busy-looping the CPU.
                rclpy.spin_once(self, timeout_sec=0.05)
                continue
            yield (b'--f\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

    def _index(self):
        return Response(self._frames(),
                        mimetype='multipart/x-mixed-replace; boundary=f')

    def _run_flask(self, host: str, port: int):
        # threaded=True to serve multiple clients; debug=False avoids the reloader.
        self._app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)


def main(args=None):
    rclpy.init(args=args)
    node = CamServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('Stopping cam_server.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
