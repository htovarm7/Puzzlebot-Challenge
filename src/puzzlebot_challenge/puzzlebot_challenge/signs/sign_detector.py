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
        # ultralytics + TRT print to stdout/stderr during YOLO() construction
        _dn = os.open(os.devnull, os.O_WRONLY)
        _o1, _o2 = os.dup(1), os.dup(2)
        os.dup2(_dn, 1); os.dup2(_dn, 2)
        try:
            _YOLO_MODEL = YOLO(load_path, task='detect')
        finally:
            os.dup2(_o1, 1); os.dup2(_o2, 2)
            os.close(_o1); os.close(_o2); os.close(_dn)
        _INFER_HALF = torch.cuda.is_available()
        print(f"[sign_detector] model loaded — CUDA={_INFER_HALF}  path={load_path}", flush=True)
    except Exception as e:
        import sys, traceback
        print(f"[sign_detector] ERROR al cargar YOLO: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
    return _YOLO_MODEL


def _warmup(model, imgsz: int = 320):
    # TRT engine deserialization + cuDNN/cuBLAS init all print to stdout/stderr here
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    _dn = os.open(os.devnull, os.O_WRONLY)
    _o1, _o2 = os.dup(1), os.dup(2)
    os.dup2(_dn, 1); os.dup2(_dn, 2)
    try:
        model.predict(dummy, verbose=False, conf=0.5, imgsz=imgsz,
                      device="cuda:0" if _INFER_HALF else "cpu",
                      half=_INFER_HALF)
    except Exception as e:
        os.dup2(_o1, 1); os.dup2(_o2, 2)
        os.close(_o1); os.close(_o2); os.close(_dn)
        print(f"[sign_detector] warmup skipped: {e}", flush=True)
        return
    os.dup2(_o1, 1); os.dup2(_o2, 2)
    os.close(_o1); os.close(_o2); os.close(_dn)
    print("[sign_detector] warmup done", flush=True)

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

DEBOUNCE_FRAMES = 2   # frames consecutivos para confirmar detección


class SignDetectorNode(Node):

    def __init__(self):
        super().__init__("sign_detector")

        self.declare_parameter("image_topic",    "/camera/image_raw")
        self.declare_parameter("conf_threshold", 0.70)
        self.declare_parameter("model_path",     self._default_model_path())
        self.declare_parameter("imgsz",          320)
        self.declare_parameter("min_det_area",   6000)
        self.declare_parameter("infer_rate_hz",  5.0)

        image_topic         = self.get_parameter("image_topic").value
        self._conf          = float(self.get_parameter("conf_threshold").value)
        self._imgsz         = int(self.get_parameter("imgsz").value)
        self._min_area      = int(self.get_parameter("min_det_area").value)
        self._infer_rate_hz = float(self.get_parameter("infer_rate_hz").value)
        model_path          = self.get_parameter("model_path").value

        self._bridge      = CvBridge()
        self._model       = None
        self._model_ready = False

        # hilo de inferencia
        self._infer_frame  = None
        self._infer_lock   = threading.Lock()
        self._infer_event  = threading.Event()
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True)

        # debounce
        self._pending_cmd    = "none"
        self._pending_count  = 0
        self._confirmed_cmd  = "none"

        self.sub_img = self.create_subscription(
            Image, image_topic, self._on_image, 10)

        self.pub_command  = self.create_publisher(String, "/sign/command",  10)
        self.pub_detected = self.create_publisher(Bool,   "/sign/detected", 10)
        self.pub_debug    = self.create_publisher(Image,  "/vision/signs",  10)

        self.get_logger().info(
            f"SignDetector | topic={image_topic} | imgsz={self._imgsz} | "
            f"conf>={self._conf:.0%} | min_area={self._min_area}px | "
            f"debounce={DEBOUNCE_FRAMES}f | cargando modelo en background...")

        threading.Thread(
            target=self._load_model_bg, args=(model_path,), daemon=True
        ).start()

    def _load_model_bg(self, model_path: str):
        model = _get_model(model_path)
        if model:
            _warmup(model, self._imgsz)
        self._model = model
        self._model_ready = True
        self._infer_thread.start()
        status = "ON" if self._model else "OFF (sin modelo)"
        self.get_logger().info(f"SignDetector LISTO | YOLO={status}")
        self.get_logger().info(
            ">>> Para arrancar el robot: "
            "ros2 topic pub --once /robot/start std_msgs/Empty '{}'")

    def _default_model_path(self) -> str:
        try:
            share = get_package_share_directory("puzzlebot_challenge")
            return os.path.join(share, "models", "best.pt")
        except Exception:
            here = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(here, "..", "utils", "best.pt")

    def _on_image(self, msg: Image):
        """Callback ROS2: solo convierte y pasa frame al hilo de inferencia."""
        if not self._model_ready:
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self._infer_lock:
            self._infer_frame = frame   # siempre el más reciente
        self._infer_event.set()

    def _infer_loop(self):
        """Hilo dedicado: inferencia YOLO sin bloquear el spin de ROS2."""
        try:
            os.nice(10)  # prioridad más baja que line_detector para ceder CPU
        except OSError:
            pass

        min_dt    = 1.0 / max(self._infer_rate_hz, 0.5)
        last_time = 0.0

        # --- DIAGNÓSTICO TEMPORAL: Hz real + detecciones crudas (sin filtrar) ---
        diag_count   = 0
        diag_t0      = time.monotonic()

        while rclpy.ok():
            self._infer_event.wait()
            self._infer_event.clear()

            now = time.monotonic()
            if now - last_time < min_dt:
                continue  # frame demasiado reciente — descartar, esperar el siguiente

            with self._infer_lock:
                frame = self._infer_frame

            if frame is None:
                continue

            last_time = time.monotonic()

            try:
                dets = yolo_detect(frame, self._model,
                                   conf_thr=self._conf,
                                   imgsz=self._imgsz)
            except Exception as e:
                self.get_logger().error(f"YOLO falló: {e}")
                continue

            # --- DIAGNÓSTICO TEMPORAL ---
            diag_count += 1
            diag_elapsed = time.monotonic() - diag_t0
            if diag_elapsed >= 2.0:
                hz = diag_count / diag_elapsed
                if dets:
                    raw_str = ", ".join(
                        f"{lbl} conf={c:.2f} bbox={w}x{h} area={w*h}px"
                        for lbl, _, _, w, h, c in dets)
                else:
                    raw_str = "sin detecciones crudas"
                self.get_logger().info(
                    f"[DIAG] infer_hz={hz:.2f}  frames={diag_count}  {raw_str}")
                diag_count = 0
                diag_t0 = time.monotonic()
            # --- FIN DIAGNÓSTICO ---

            dets = [d for d in dets if d[3] * d[4] >= self._min_area]
            raw_cmd = dets[0][0] if dets else "none"
            # tomar la detección de mayor área si hay varias
            if dets:
                raw_cmd = max(dets, key=lambda d: d[3] * d[4])[0]

            # debounce: confirmar tras DEBOUNCE_FRAMES frames consecutivos
            if raw_cmd == self._pending_cmd:
                self._pending_count += 1
            else:
                self._pending_cmd   = raw_cmd
                self._pending_count = 1

            if self._pending_count >= DEBOUNCE_FRAMES:
                command = self._pending_cmd
            else:
                command = self._confirmed_cmd  # mantener el último confirmado

            if command != self._confirmed_cmd:
                self._confirmed_cmd = command
                best = max(dets, key=lambda d: d[3] * d[4]) if dets else None
                if best:
                    self.get_logger().info(
                        f"DETECTED: {command.upper()} "
                        f"(conf={best[5]:.0%}, area={best[3]*best[4]}px)")
                else:
                    self.get_logger().info("DETECTED: NONE")

            c_msg = String(); c_msg.data = command
            d_msg = Bool();   d_msg.data = (command != "none")
            self.pub_command.publish(c_msg)
            self.pub_detected.publish(d_msg)
            self._publish_debug(frame, dets, command)

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
