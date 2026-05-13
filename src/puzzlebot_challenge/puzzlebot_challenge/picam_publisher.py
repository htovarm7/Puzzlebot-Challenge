#!/usr/bin/env python3
"""Publica frames de la PiCam CSI (IMX219) en /camera/image_raw vía GStreamer."""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def build_pipeline(sensor_mode: int, src_w: int, src_h: int,
                   src_fps: int, out_w: int, out_h: int) -> str:
    return (
        f"nvarguscamerasrc sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM),width={src_w},height={src_h},"
        f"format=NV12,framerate={src_fps}/1 ! "
        f"nvvidconv ! "
        f"video/x-raw,width={out_w},height={out_h},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 max-buffers=1"
    )


class PiCamPublisher(Node):

    def __init__(self):
        super().__init__('picam_publisher')

        self.declare_parameter('sensor_mode', 3)
        self.declare_parameter('src_width', 1640)
        self.declare_parameter('src_height', 1232)
        self.declare_parameter('src_fps', 30)
        self.declare_parameter('out_width', 320)
        self.declare_parameter('out_height', 240)
        self.declare_parameter('pub_fps', 30.0)
        self.declare_parameter('topic', '/camera/image_raw')
        self.declare_parameter('frame_id', 'picam')

        p = self.get_parameter
        pipeline = build_pipeline(
            p('sensor_mode').value, p('src_width').value, p('src_height').value,
            p('src_fps').value, p('out_width').value, p('out_height').value,
        )

        self.get_logger().info(f"Abriendo PiCam: {pipeline}")
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("No se pudo abrir la PiCam CSI.")
            raise SystemExit(1)

        ok, _ = self.cap.read()
        if not ok:
            self.get_logger().error("La cámara abrió pero no entrega frames.")
            self.cap.release()
            raise SystemExit(1)

        self.frame_id = p('frame_id').value
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, p('topic').value, 10)
        self.timer = self.create_timer(1.0 / float(p('pub_fps').value), self._tick)

        self.get_logger().info(
            f"Publicando en {p('topic').value} "
            f"({p('out_width').value}x{p('out_height').value} @ {p('pub_fps').value} Hz)"
        )

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Frame vacío de la cámara.")
            return
        frame = cv2.flip(frame, -1)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
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
        node.get_logger().warn('Deteniendo cámara.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
