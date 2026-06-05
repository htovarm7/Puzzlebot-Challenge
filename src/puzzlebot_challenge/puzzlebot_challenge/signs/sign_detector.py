#!/usr/bin/env python3
"""
sign_detector.py — Detecta señales de tránsito con YOLO.

Tópicos publicados:
  /sign/command   (std_msgs/String)  — stop | go_straight | turn_left | turn_right | workers | none
  /sign/detected  (std_msgs/Bool)    — True si hay señal activa
  /vision/signs   (sensor_msgs/Image) — frame anotado
"""

import ctypes
import os
import threading
import time

for _lib in (
    '/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0',
    '/usr/lib/aarch64-linux-gnu/libgomp.so.1',
    'libgomp.so.1',
):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

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


def _get_model(model_path: str):
    global _YOLO_MODEL, _YOLO_TRIED, _INFER_HALF, _INFER_DEVID
    if _YOLO_TRIED:
        return _YOLO_MODEL
    _YOLO_TRIED = True

    base = model_path.replace('.pt', '').replace('.onnx', '').replace('.engine', '')
    engine_path = base + '.engine'
    onnx_path   = base + '.onnx'
    if os.path.exists(engine_path):
        load_path = engine_path
    elif os.path.exists(onnx_path):
        load_path = onnx_path
    else:
        print(f"[sign_detector] ERROR: no se encontró .engine ni .onnx en {base} — YOLO disabled", flush=True)
        return None
    try:
        import sys, traceback, torch
        from ultralytics import YOLO
        print(f"[sign_detector] torch={torch.__version__}  CUDA={torch.cuda.is_available()}", flush=True)
        print(f"[sign_detector] loading: {load_path}")
        _YOLO_MODEL = YOLO(load_path)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _INFER_HALF = torch.cuda.is_available()
        print(f"[sign_detector] device={device}  half={_INFER_HALF}")

        print(f"[sign_detector] model loaded: {model_path}")
    except Exception as e:
        import sys, traceback
        print(f"[sign_detector] ERROR al cargar YOLO: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
    return _YOLO_MODEL


def _warmup(model, imgsz: int = 192):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        model.predict(dummy, verbose=False, conf=0.5, imgsz=imgsz)
        print("[sign_detector] model warmup done")
    except Exception as e:
        print(f"[sign_detector] warmup skipped: {e}")

def _contour_arrow_direction(frame, x1, y1, x2, y2):
    """Detecta direccion de flecha por proyeccion de columnas.
    Suma pixeles blancos por columna, suaviza el perfil y busca el pico.
    El pico indica donde esta la base del arrowhead (parte mas ancha).
    Retorna: turn_left | turn_right | go_straight | None
    """
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
    if int(white.sum() // 255) < 300:
        return None
    profile = white.sum(axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (9, 1), 0).flatten()
    peak_x  = int(np.argmax(profile))
    w       = crop.shape[1]
    ratio   = peak_x / w
    if ratio < 0.40:
        return "turn_left"
    if ratio > 0.60:
        return "turn_right"
    return "go_straight"

def yolo_detect(frame: np.ndarray, model, conf_thr: float = 0.60, imgsz: int = 320) -> list:
    if model is None:
        return []
    results = model.predict(frame, verbose=False, conf=conf_thr, imgsz=imgsz,
                            device="cuda:0" if _INFER_HALF else "cpu",
                            half=_INFER_HALF)[0]
    dets = []
    for box in results.boxes:
        label = model.names[int(box.cls)].lower().replace("-", "_").replace(" ", "_")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if label in ("turn_left", "turn_right", "go_straight"):
            contour_dir = _contour_arrow_direction(frame, x1, y1, x2, y2)
            if contour_dir is not None:
                label = contour_dir
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


class SignDetectorNode(Node):

    def __init__(self):
        super().__init__("sign_detector")

        self.declare_parameter("image_topic",    "/camera/image_raw")
        self.declare_parameter("conf_threshold", 0.65)
        self.declare_parameter("model_path",     self._default_model_path())
        self.declare_parameter("imgsz",          320)

        image_topic = self.get_parameter("image_topic").value
        self._conf  = float(self.get_parameter("conf_threshold").value)
        self._imgsz = int(self.get_parameter("imgsz").value)
        model_path  = self.get_parameter("model_path").value

        self._bridge = CvBridge()
        self._model  = None
        self._model_ready = False

        self._latest_dets    = []
        self._latest_command = "none"
        self._last_status_t  = time.monotonic()
        self._frames_in      = 0

        self.sub_img = self.create_subscription(
            Image, image_topic, self._on_image, _SENSOR_QOS)

        self.pub_command  = self.create_publisher(String, "/sign/command",  10)
        self.pub_detected = self.create_publisher(Bool,   "/sign/detected", 10)
        self.pub_debug    = self.create_publisher(Image,  "/vision/signs",  10)

        self.get_logger().info(
            f"SignDetector | topic={image_topic} | "
            f"imgsz={self._imgsz} | conf>={self._conf:.0%} | "
            f"cargando modelo en background...")

        threading.Thread(
            target=self._load_model_bg, args=(model_path,), daemon=True
        ).start()

    def _load_model_bg(self, model_path: str):
        model = _get_model(model_path)
        if model:
            _warmup(model, self._imgsz)
        self._model = model
        self._model_ready = True
        self.get_logger().info(
            f"SignDetector LISTO | YOLO={'ON' if self._model else 'OFF (sin modelo)'}")

    def _default_model_path(self) -> str:
        try:
            share = get_package_share_directory("puzzlebot_challenge")
            return os.path.join(share, "models", "best.pt")
        except Exception:
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(here, "..", "utils", "best.pt")

    def _on_image(self, msg: Image):
        if not self._model_ready:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        self._frames_in += 1

        try:
            final_dets = yolo_detect(frame, self._model,
                                     conf_thr=self._conf,
                                     imgsz=self._imgsz)
        except Exception as e:
            self.get_logger().error(f"[detector] YOLO falló: {e}")
            return

        if final_dets:
            best    = max(final_dets, key=lambda d: d[3] * d[4])
            command = best[0]
            self.get_logger().info(
                f"DETECTED: {command.upper()} "
                f"(conf={best[5]:.0%}, {best[3]}x{best[4]}px)")
        else:
            command = "none"
            now = time.monotonic()
            if now - self._last_status_t >= 5.0:
                self.get_logger().info(f"[detector] nada detectado | frames_in={self._frames_in}")
                self._last_status_t = now

        self._latest_dets    = final_dets
        self._latest_command = command

        c_msg = String(); c_msg.data = command
        d_msg = Bool();   d_msg.data = (command != "none")
        self.pub_command.publish(c_msg)
        self.pub_detected.publish(d_msg)
        self._publish_debug(frame, final_dets, command)

    def _publish_debug(self, frame, dets, command):
        if self.pub_debug.get_subscription_count() == 0:
            return
        vis     = annotate(frame, dets, command)
        out_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(out_msg)

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectorNode()
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
