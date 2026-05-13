#!/usr/bin/env python3
"""
picam_publisher.py
==================
Publica frames de la PiCam CSI en /camera/image_raw usando GStreamer
con nvarguscamerasrc.

Uso
---
  python3 picam_publisher.py

Verificación
------------
  ros2 topic hz /camera/image_raw
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


OUT_WIDTH  = 320      
OUT_HEIGHT = 240
PUB_FPS    = 30


def gst_pipeline():
    """
    Pipeline con sensor-mode=4 (720p @ 60fps nativo del sensor IMX219).
    nvvidconv escala a 320x240 en hardware.
    """
    return (
        "nvarguscamerasrc sensor-mode=4 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, "
        "format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        f"video/x-raw, width={OUT_WIDTH}, height={OUT_HEIGHT}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1 max-buffers=1"
    )


class PiCamPublisher(Node):

    def __init__(self):
        super().__init__('picam_publisher')

        pipeline = gst_pipeline()
        self.get_logger().info("Abriendo PiCam vía GStreamer (sensor-mode=4)…")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error(
                "No Picam open.\n"
            )
            raise SystemExit(1)

        # Lee un frame de prueba para confirmar
        ok, _ = self.cap.read()
        if not ok:
            self.get_logger().error("Open camera but no frames")
            self.cap.release()
            raise SystemExit(1)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(1.0 / PUB_FPS, self._tick)

        self.get_logger().info(
            f"PiCamPublisher ready: /camera/image_raw "
            f"({OUT_WIDTH}x{OUT_HEIGHT} @ {PUB_FPS} Hz)"
        )

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Empty frame from camera")
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'picam'
        self.pub.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiCamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('Stopping camera.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()