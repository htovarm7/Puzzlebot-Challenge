#!/usr/bin/env python3
"""
sign_detector_offload.py — Corre en la LAPTOP y procesa las imágenes de la Jetson.

Jetson  →  /camera/image_raw  →  Laptop (YOLO)  →  /sign/command  →  Jetson

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

_YOLO_MODEL  = None
_YOLO_TRIED  = False
_INFER_HALF  = False
_INFER_DEVID = 0


def _resolve_model_path(base_path: str) -> str:
    engine_path = os.path.splitext(base_path)[0] + ".engine"
    if os.path.exists(engine_path):
        return engine_path
    return base_path


def _get_model(model_path: str):
    global _YOLO_MODEL, _YOLO_TRIED, _INFER_HALF, _INFER_DEVID
    if _YOLO_TRIED:
        return _YOLO_MODEL
    _YOLO_TRIED = True

    resolved = _resolve_model_path(model_path)
    if not os.path.exists(resolved):
        print(f"[sign_detector] WARN: model not found at {resolved} — YOLO disabled")
        return None
    try:
        print(f"[sign_detector] Cargando modelo: {resolved}")
        import torch
        from ultralytics import YOLO
        print(f"[sign_detector] torch={torch.__version__}  CUDA={torch.cuda.is_available()}")
        _YOLO_MODEL  = YOLO(resolved)
        using_trt    = resolved.endswith(".engine")

        if torch.cuda.is_available():
            # Let predict(half=True) handle FP16 casting — don't call .model.half() manually
            _INFER_HALF  = not using_trt  # TRT engines embed FP16 natively
            _INFER_DEVID = 0
            print(f"[sign_detector] CUDA available — inference on cuda:{_INFER_DEVID}"
                  f" {'FP16' if _INFER_HALF else 'FP32'}")
        else:
            _INFER_HALF  = False
            _INFER_DEVID = "cpu"
            torch.set_num_threads(os.cpu_count() or 4)
            print(f"[sign_detector] CPU mode — threads={torch.get_num_threads()}")

        print(f"[sign_detector] model loaded: {resolved}"
              f" ({'TensorRT' if using_trt else 'PyTorch'})")
        print(f"                Classes: {list(_YOLO_MODEL.names.values())}")

        _warmup(_YOLO_MODEL, imgsz=192)
    except Exception as e:
        import traceback
        print(f"[sign_detector] ERROR al cargar YOLO: {e}")
        traceback.print_exc()
    return _YOLO_MODEL


def _warmup(model, imgsz: int = 192):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        model.predict(dummy, verbose=False, conf=0.5, imgsz=imgsz,
                      device=_INFER_DEVID, half=_INFER_HALF)
        print("[sign_detector] model warmup done")
    except Exception as e:
        print(f"[sign_detector] warmup skipped: {e}")


def yolo_detect(frame: np.ndarray, model, conf_thr: float = 0.60, imgsz: int = 320) -> list:
    if model is None:
        return []
    results = model.predict(frame, verbose=False, conf=conf_thr, imgsz=imgsz,
                            device=_INFER_DEVID, half=_INFER_HALF)[0]
    dets = []
    for box in results.boxes:
        label = model.names[int(box.cls)].lower().replace("-", "_").replace(" ", "_")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append((label, x1, y1, x2 - x1, y2 - y1, round(float(box.conf), 2)))
    return dets


def annotate(frame: np.ndarray, dets: list, command: str) -> np.ndarray:
    out = frame.copy()
    for label, x, y, w, h, conf in dets:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"{label.upper()} {conf:.0%}", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (50, 50, 50), -1)
    cv2.putText(out, f"CMD: {command.upper()}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class SignDetectorOffloadNode(Node):

    def __init__(self):
        super().__init__("sign_detector_offload")

        self.declare_parameter("image_topic",    "/camera/image_raw")
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("model_path",     self._default_model_path())
        self.declare_parameter("imgsz",          320)
        self.declare_parameter("jetson_api",     "http://100.73.89.116:8081/sign")

        image_topic = self.get_parameter("image_topic").value
        self._conf  = float(self.get_parameter("conf_threshold").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        model_path  = self.get_parameter("model_path").value

        self._bridge      = CvBridge()
        self._model       = _get_model(model_path)
        self._jetson_api  = self.get_parameter("jetson_api").value
        self._last_posted = "none"

        self._pending_frame  = None
        self._latest_dets    = []
        self._latest_command = "none"
        self._lock           = threading.Lock()

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
            f"imgsz={self._imgsz} | conf>={self._conf:.0%} | "
            f"YOLO={'ON' if self._model else 'OFF'}")

    def _default_model_path(self) -> str:
        try:
            share = get_package_share_directory("puzzlebot_challenge")
            engine = os.path.join(share, "models", "best.engine")
            if os.path.exists(engine):
                return engine
            return os.path.join(share, "models", "best.pt")
        except Exception:
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

        if command != self._last_posted:
            self._post_sign(command)

    def _post_sign(self, command: str):
        self._last_posted = command
        if not self._jetson_api:
            return
        try:
            import requests
            requests.post(self._jetson_api,
                          json={"command": command}, timeout=0.2)
        except Exception:
            pass

    def _detection_loop(self):
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            final_dets = yolo_detect(frame, self._model,
                                     conf_thr=self._conf,
                                     imgsz=self._imgsz)

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
