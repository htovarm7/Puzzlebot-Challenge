#!/usr/bin/env python3
"""Publica frames de la PiCam CSI (IMX219) en /camera/image_raw.

Usa GStreamer Python (gi) en lugar de cv2.VideoCapture porque el OpenCV
del Jetson fue compilado sin soporte GStreamer.
"""

import os
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# GStreamer/ARGUS write directly to stderr; ROS2 logger uses stdout — safe to silence
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)

Gst.init(None)


def build_pipeline(sensor_mode: int, src_w: int, src_h: int,
                   src_fps: int, out_w: int, out_h: int) -> str:
    return (
        f"nvarguscamerasrc sensor-mode={sensor_mode} ! "
        f"video/x-raw(memory:NVMM),width={src_w},height={src_h},"
        f"format=NV12,framerate={src_fps}/1 ! "
        f"nvvidconv ! "
        f"video/x-raw,width={out_w},height={out_h},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink name=sink drop=true max-buffers=1 emit-signals=true sync=false"
    )


class PiCamPublisher(Node):

    def __init__(self):
        super().__init__('picam_publisher')

        self.declare_parameter('sensor_mode', 2)
        self.declare_parameter('src_width',   1920)
        self.declare_parameter('src_height',  1080)
        self.declare_parameter('src_fps',     30)
        self.declare_parameter('out_width',   320)
        self.declare_parameter('out_height',  240)
        self.declare_parameter('pub_fps',     30.0)
        self.declare_parameter('topic',       '/camera/image_raw')
        self.declare_parameter('frame_id',    'picam')

        p = self.get_parameter
        self._out_w    = int(p('out_width').value)
        self._out_h    = int(p('out_height').value)
        self._frame_id = p('frame_id').value

        pipeline_str = build_pipeline(
            int(p('sensor_mode').value),
            int(p('src_width').value), int(p('src_height').value),
            int(p('src_fps').value),
            self._out_w, self._out_h,
        )

        self.get_logger().info("Abriendo PiCam (gi/GStreamer)...")

        self._pipeline = Gst.parse_launch(pipeline_str)
        if self._pipeline is None:
            self.get_logger().error("No se pudo parsear el pipeline GStreamer.")
            raise SystemExit(1)

        self._appsink = self._pipeline.get_by_name('sink')
        self._appsink.connect('new-sample', self._on_new_sample)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.get_logger().error("No se pudo iniciar el pipeline GStreamer.")
            raise SystemExit(1)

        self._latest_frame = None
        self._lock         = threading.Lock()

        self.bridge = CvBridge()
        self.pub    = self.create_publisher(Image, p('topic').value, 10)
        self.timer  = self.create_timer(1.0 / float(p('pub_fps').value), self._tick)

        self.get_logger().info(
            f"Publicando en {p('topic').value} "
            f"({self._out_w}x{self._out_h} @ {p('pub_fps').value} Hz)"
        )

    # ── GStreamer callback (hilo de GStreamer) ────────────────────────────
    def _on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.OK

        buf           = sample.get_buffer()
        ok, mapinfo   = buf.map(Gst.MapFlags.READ)
        if ok:
            data  = np.frombuffer(mapinfo.data, dtype=np.uint8)
            frame = data.reshape(self._out_h, self._out_w, 3).copy()
            frame = cv2.flip(frame, -1)
            with self._lock:
                self._latest_frame = frame
        buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    # ── Timer ROS2 (hilo de rclpy) ───────────────────────────────────────
    def _tick(self):
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return
        msg              = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self.pub.publish(msg)

    def destroy_node(self):
        self._pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiCamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
