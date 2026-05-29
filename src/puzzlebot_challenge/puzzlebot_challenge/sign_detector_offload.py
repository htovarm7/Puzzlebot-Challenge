#!/usr/bin/env python3
"""
sign_detector_offload.py — Corre en la LAPTOP y procesa las imágenes de la Jetson.

Jetson  →  /camera/image_raw  →  Laptop (YOLO+HSV)  →  /sign/command  →  Jetson

Tópicos publicados:
  /sign/command   (std_msgs/String)  — stop | go_straight | turn_left | turn_right | workers | none
  /sign/detected  (std_msgs/Bool)    — True si hay señal activa
  /vision/signs   (sensor_msgs/Image) — frame anotado, ver en rqt:
                    ros2 run rqt_image_view rqt_image_view /vision/signs

Ver docs/MULTIPROCESSING.md para setup de red (Tailscale + FastDDS).
"""

import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

# Re-usa toda la lógica de detección de sign_detector.py
from puzzlebot_challenge.sign_detector import (
    _get_model, detect_signs, annotate,
    TemporalSmoother, _DISPLAY,
)

# QoS sensor-like: best effort, keep last 1 — reduce latencia en red
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class SignDetectorOffloadNode(Node):
    """
    Nodo para laptop: recibe imágenes de la Jetson, corre YOLO, publica comandos.
    """

    def __init__(self):
        super().__init__("sign_detector_offload")

        self.declare_parameter("image_topic",    "/camera/image_raw")
        self.declare_parameter("conf_threshold", 0.60)
        self.declare_parameter("model_path",     self._default_model_path())
        self.declare_parameter("imgsz",          320)   # laptop puede con 320+

        image_topic = self.get_parameter("image_topic").value
        self._conf  = float(self.get_parameter("conf_threshold").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        model_path  = self.get_parameter("model_path").value

        self._bridge   = CvBridge()
        self._model    = _get_model(model_path)
        self._smoother = TemporalSmoother()

        self._pending_frame  = None
        self._latest_dets    = []
        self._latest_command = "none"
        self._lock           = threading.Lock()

        # Suscribe con QoS best-effort para tolerar pérdidas en red WiFi
        self.sub_img = self.create_subscription(
            Image, image_topic, self._on_image, _SENSOR_QOS)

        self.pub_command  = self.create_publisher(String, "/sign/command",  10)
        self.pub_detected = self.create_publisher(Bool,   "/sign/detected", 10)
        self.pub_debug    = self.create_publisher(Image,  "/vision/signs",  10)

        self._running    = True
        self._det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._det_thread.start()

        self.get_logger().info(
            f"SignDetectorOffload (laptop) | topic={image_topic} | "
            f"imgsz={self._imgsz} | "
            f"YOLO={'ON' if self._model else 'OFF (fallback only)'}")

    def _default_model_path(self) -> str:
        try:
            share = get_package_share_directory("puzzlebot_challenge")
            # Prefiere .engine si existe (exportado con export_trt.py)
            engine = os.path.join(share, "models", "best.engine")
            if os.path.exists(engine):
                return engine
            return os.path.join(share, "models", "best.pt")
        except Exception:
            # Fallback: busca junto a este archivo
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(here, "..", "utils", "best.pt")

    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        with self._lock:
            self._pending_frame = frame
            dets    = list(self._latest_dets)
            command = self._latest_command

        c_msg = String(); c_msg.data = command
        d_msg = Bool();   d_msg.data = (command != "none")
        self.pub_command.publish(c_msg)
        self.pub_detected.publish(d_msg)
        self._publish_debug(frame, dets, command)

    def _detection_loop(self):
        YOLO_CONF_TRUST = self._conf
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            raw_dets = detect_signs(frame, self._model, imgsz=self._imgsz)

            yolo_direct = [d for d in raw_dets
                           if d[5] >= YOLO_CONF_TRUST
                           and d[0] in {"stop", "workers", "go_straight",
                                        "turn_left", "turn_right"}]
            fallback_raw    = [d for d in raw_dets if d not in yolo_direct]
            fallback_stable = self._smoother.update(fallback_raw)

            yolo_labels = {d[0] for d in yolo_direct}
            final_dets  = yolo_direct + [d for d in fallback_stable
                                         if d[0] not in yolo_labels]

            if final_dets:
                best    = max(final_dets, key=lambda d: d[3] * d[4])
                command = best[0]
                self.get_logger().info(
                    f"DETECTED: {command.upper()} "
                    f"(conf={best[5]:.0%}, {best[3]}x{best[4]}px)")
            else:
                command = "none"

            with self._lock:
                self._latest_dets    = final_dets
                self._latest_command = command

    def _publish_debug(self, frame, dets, command):
        if self.pub_debug.get_subscription_count() == 0:
            return
        vis     = annotate(frame, dets, command)
        out_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(out_msg)

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorOffloadNode()
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
