#!/usr/bin/env python3
"""
sign_detector.py  –  ROS2 node: traffic-sign detection using YOLOv8 + HSV fallback.

Subscribes : /camera/image_raw        (sensor_msgs/Image)
Publishes  : /sign/command            (std_msgs/String)   stop | go_straight | turn_left | turn_right | workers | none
             /sign/detected           (std_msgs/Bool)
             /vision/signs            (sensor_msgs/Image) annotated debug frame

Detection pipeline (same priority order as original):
  1. YOLOv8  (primary — CNN trained on best.pt / best.engine for TensorRT)
  2. HSV color + polygon shape  (fallback for red signs)
  3. Template matching on blue ROI  (fallback for directional signs)

YOLO detections above the confidence threshold are reported immediately.
Fallback detections pass through TemporalSmoother (majority vote over a
rolling window) to avoid single-frame flickers.

Jetson Nano optimizations:
  - Loads best.engine (TensorRT FP16) if present, else best.pt
  - Inference runs on cuda:0 with half precision
  - Model warmed up at startup to avoid first-frame latency spike
  - Frame decimation: detection thread skips frames when it falls behind
"""

import os
import threading
import time
from collections import deque, Counter

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
_YOLO_MAP = {
    "goleft":     "turn_left",
    "goright":    "turn_right",
    "gostraight": "go_straight",
    "stop":       "stop",
    "workers":    "workers",
}

_DISPLAY = {
    "stop":        "STOP",
    "workers":     "TRABAJADORES",
    "go_straight": "SIGA RECTO",
    "turn_left":   "VUELTA IZQ.",
    "turn_right":  "VUELTA DER.",
    "none":        "Sin señal",
}

_COLORS = {
    "stop":        (0,   0,   220),
    "workers":     (0,   140, 255),
    "go_straight": (30,  200, 30),
    "turn_left":   (200, 180, 0),
    "turn_right":  (200, 180, 0),
}

# ---------------------------------------------------------------------------
# YOLO loader — prefers TensorRT engine (.engine) over weights (.pt)
# Device: cuda:0 (Jetson Nano GPU). Falls back to CPU if CUDA unavailable.
# ---------------------------------------------------------------------------
_YOLO_MODEL  = None
_YOLO_TRIED  = False
_INFER_HALF  = False   # set True once GPU + FP16 confirmed
_INFER_DEVID = 0       # cuda device index


def _resolve_model_path(base_path: str) -> str:
    """Return best.engine if it lives next to best.pt, else return base_path."""
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
        import torch
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(resolved)
        using_trt = resolved.endswith(".engine")

        if torch.cuda.is_available():
            _INFER_HALF  = not using_trt  # TRT engines embed FP16 natively; .pt uses half via predict()
            _INFER_DEVID = 0
            print(f"[sign_detector] CUDA available — inference on cuda:{_INFER_DEVID}"
                  f" {'FP16' if _INFER_HALF else 'FP32'}")
        else:
            _INFER_HALF  = False
            _INFER_DEVID = "cpu"
            # Use all available CPU cores for BLAS/conv ops
            import os as _os
            torch.set_num_threads(_os.cpu_count() or 4)
            print(f"[sign_detector] CPU mode — threads={torch.get_num_threads()}")

        print(f"[sign_detector] model loaded: {resolved}"
              f" ({'TensorRT' if using_trt else 'PyTorch'})")
        print(f"                Classes: {list(_YOLO_MODEL.names.values())}")

        # Warmup: one dummy forward pass to pre-allocate CUDA/TRT buffers
        _warmup(_YOLO_MODEL, imgsz=192)
    except Exception as e:
        print(f"[sign_detector] WARN: could not load YOLO — {e}")
    return _YOLO_MODEL


def _warmup(model, imgsz: int = 192):
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        model.predict(
            dummy, verbose=False, conf=0.5, imgsz=imgsz,
            device=_INFER_DEVID, half=_INFER_HALF,
        )
        print("[sign_detector] model warmup done")
    except Exception as e:
        print(f"[sign_detector] warmup skipped: {e}")


def yolo_detect(frame: np.ndarray, model, conf_thr: float = 0.45, imgsz: int = 192) -> list:
    if model is None:
        return []
    results = model.predict(
        frame, verbose=False, conf=conf_thr, imgsz=imgsz,
        device=_INFER_DEVID, half=_INFER_HALF,
    )[0]
    dets = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls)].lower()
        label = _YOLO_MAP.get(cls_name)
        if label is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        dets.append((label, x1, y1, x2 - x1, y2 - y1, round(conf, 2)))
    return dets


# ---------------------------------------------------------------------------
# Arrow templates for blue-sign template matching
# ---------------------------------------------------------------------------
def _make_arrow(direction: str, size: int = 80) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    hw = max(size // 8, 4)
    hl = size // 3
    sl = size // 3
    if direction == "up":
        shaft = [(cx-hw, cy+sl), (cx+hw, cy+sl), (cx+hw, cy-4), (cx-hw, cy-4)]
        head  = [(cx-hl, cy-4), (cx, cy-sl-hw*2), (cx+hl, cy-4)]
    elif direction == "left":
        shaft = [(cx, cy-hw), (cx+sl, cy-hw), (cx+sl, cy+hw), (cx, cy+hw)]
        head  = [(cx, cy-hl), (cx-sl, cy), (cx, cy+hl)]
    elif direction == "right":
        shaft = [(cx-sl, cy-hw), (cx, cy-hw), (cx, cy+hw), (cx-sl, cy+hw)]
        head  = [(cx, cy-hl), (cx+sl, cy), (cx, cy+hl)]
    elif direction == "right_curve":
        pts = []
        for angle in np.linspace(np.pi*0.6, np.pi*0.05, 20):
            r = size * 0.32
            pts.append((int(cx + r*np.cos(angle)), int(cy - r*np.sin(angle))))
        if len(pts) > 1:
            for i in range(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], 255, hw*2)
        end = pts[-1]
        h = [(end[0]-hl//2, end[1]-hl//2), (end[0]+hl//2, end[1]), (end[0]-hl//2, end[1]+hl//2)]
        cv2.fillPoly(img, [np.array(h, np.int32)], 255)
        return img
    else:
        return img
    cv2.fillPoly(img, [np.array(shaft, np.int32)], 255)
    cv2.fillPoly(img, [np.array(head,  np.int32)], 255)
    return img


TEMPLATES = {
    "go_straight":  _make_arrow("up"),
    "turn_left":    _make_arrow("left"),
    "turn_right":   _make_arrow("right"),
    "turn_right_c": _make_arrow("right_curve"),
}
MATCH_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]

# ---------------------------------------------------------------------------
# HSV ranges
# ---------------------------------------------------------------------------
RED_LO1  = np.array([0,   100, 50],  np.uint8)
RED_HI1  = np.array([12,  255, 255], np.uint8)
RED_LO2  = np.array([155, 100, 50],  np.uint8)
RED_HI2  = np.array([180, 255, 255], np.uint8)
BLUE_LO  = np.array([95,  80,  50],  np.uint8)
BLUE_HI  = np.array([135, 255, 255], np.uint8)
MIN_AREA = 400
MORPH_K  = np.ones((5, 5), np.uint8)


def _red_mask(hsv):
    m = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                       cv2.inRange(hsv, RED_LO2, RED_HI2))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_K)


def _blue_mask(hsv):
    m = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_K)


def _valid_contours(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted([c for c in cnts if cv2.contourArea(c) >= MIN_AREA],
                  key=cv2.contourArea, reverse=True)


def _classify_red_shape(contour):
    peri   = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n = len(approx)
    _, _, w, h = cv2.boundingRect(contour)
    ar = w / (h + 1e-5)
    if n >= 6 and 0.65 < ar < 1.5:
        return "stop"
    if n == 3:
        return "workers"
    if n == 4 and ar > 1.3:
        return "stop"
    return None


def _match_arrow(roi_bw):
    best_label = "unknown"
    best_score = 0.28
    h, w = roi_bw.shape
    for label, tmpl in TEMPLATES.items():
        th, tw = tmpl.shape
        for scale in MATCH_SCALES:
            sw, sh = int(tw * scale), int(th * scale)
            if sw < 10 or sh < 10 or sw > w or sh > h:
                continue
            scaled = cv2.resize(tmpl, (sw, sh))
            res = cv2.matchTemplate(roi_bw, scaled, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if score > best_score:
                best_score = score
                best_label = label
    if best_label == "turn_right_c":
        best_label = "turn_right"
    return best_label, best_score


def detect_signs(frame: np.ndarray, model, imgsz: int = 192) -> list:
    """
    3-stage pipeline identical to original.
    Returns list of (label, x, y, w, h, confidence).
    """
    dets_yolo   = yolo_detect(frame, model, imgsz=imgsz)
    yolo_labels = {d[0] for d in dets_yolo}

    all_labels = {"stop", "workers", "go_straight", "turn_left", "turn_right"}
    if yolo_labels >= all_labels:
        return dets_yolo

    dets_fallback = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_is_close = False
    if "stop" not in yolo_labels and "workers" not in yolo_labels:
        for cnt in _valid_contours(_red_mask(hsv)):
            label = _classify_red_shape(cnt)
            if label is None:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            dets_fallback.append((label, x, y, w, h,
                                  0.82 if label == "stop" else 0.78))
            red_is_close = (w >= 60 and h >= 60)
            break

    directional = {"go_straight", "turn_left", "turn_right"}
    if not (directional & yolo_labels) and not red_is_close:
        for cnt in _valid_contours(_blue_mask(hsv)):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                break
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_bw = cv2.threshold(roi_gray, 140, 255, cv2.THRESH_BINARY)
            label, score = _match_arrow(roi_bw)
            if label != "unknown":
                dets_fallback.append((label, x, y, w, h, round(score, 2)))
            break

    return dets_yolo + dets_fallback


def annotate(frame: np.ndarray, dets: list, command: str) -> np.ndarray:
    out = frame.copy()
    for label, x, y, w, h, conf in dets:
        color = _COLORS.get(label, (255, 255, 255))
        text  = _DISPLAY.get(label, label.upper())
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        cv2.putText(out, f"{text} {conf:.0%}", (x, max(y-6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    bg_color = _COLORS.get(command, (60, 60, 60))
    title    = f"CMD: {_DISPLAY.get(command, command.upper())}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), bg_color, -1)
    cv2.putText(out, title, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Temporal smoother (unchanged from original)
# ---------------------------------------------------------------------------
class TemporalSmoother:
    WINDOW    = 15
    THRESHOLD = 0.45

    def __init__(self):
        self._red_buf  = deque(maxlen=self.WINDOW)
        self._blue_buf = deque(maxlen=self.WINDOW)

    def update(self, raw_dets: list) -> list:
        red_label = blue_label = None
        red_det   = blue_det   = None

        for det in raw_dets:
            label = det[0]
            if label in ("stop", "workers"):
                red_label, red_det = label, det
            else:
                blue_label, blue_det = label, det

        self._red_buf.append(red_label)
        self._blue_buf.append(blue_label)

        stable = []
        for buf, det in ((self._red_buf, red_det), (self._blue_buf, blue_det)):
            counts = Counter(x for x in buf if x is not None)
            if not counts:
                continue
            top_label, top_count = counts.most_common(1)[0]
            if top_count / self.WINDOW >= self.THRESHOLD:
                if det and det[0] == top_label:
                    stable.append(det)
        return stable


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------
class SignDetectorNode(Node):

    def __init__(self):
        super().__init__("sign_detector")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("model_path", self._default_model_path())
        # max YOLO inferences per second; 0 = unlimited (process every frame)
        self.declare_parameter("max_infer_fps", 0.0)  # 0 = unlimited
        self.declare_parameter("imgsz", 192)

        image_topic  = self.get_parameter("image_topic").value
        self._conf        = float(self.get_parameter("conf_threshold").value)
        model_path        = self.get_parameter("model_path").value
        self._imgsz       = int(self.get_parameter("imgsz").value)
        _fps = self.get_parameter("max_infer_fps").value
        self._min_infer_dt = (1.0 / _fps) if _fps > 0 else 0.0

        self._bridge   = CvBridge()
        self._model    = _get_model(model_path)
        self._smoother = TemporalSmoother()

        # Shared state between ROS callback and detection thread
        self._pending_frame  = None
        self._latest_dets    = []
        self._latest_command = "none"
        self._lock           = threading.Lock()
        self._last_infer_t   = 0.0

        self.sub_img      = self.create_subscription(
            Image, image_topic, self._on_image, 10)
        self.pub_command  = self.create_publisher(String, "/sign/command",  10)
        self.pub_detected = self.create_publisher(Bool,   "/sign/detected", 10)
        self.pub_debug    = self.create_publisher(Image,  "/vision/signs",  10)

        # Background thread for YOLO inference (avoids blocking camera callback)
        self._running = True
        self._det_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._det_thread.start()

        self.get_logger().info(
            f"SignDetectorNode ready | topic={image_topic} | "
            f"YOLO={'ON' if self._model else 'OFF (fallback only)'} | "
            f"max_infer_fps={self.get_parameter('max_infer_fps').value}")

    # ------------------------------------------------------------------
    def _default_model_path(self) -> str:
        try:
            share = get_package_share_directory("puzzlebot_challenge")
            return os.path.join(share, "models", "best.pt")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        # Hand off to detection thread (drop frame if thread is still busy)
        with self._lock:
            self._pending_frame = frame
            dets    = list(self._latest_dets)
            command = self._latest_command

        # Publish current results (non-blocking — uses last known state)
        c_msg = String(); c_msg.data = command
        d_msg = Bool();   d_msg.data = (command != "none")
        self.pub_command.publish(c_msg)
        self.pub_detected.publish(d_msg)

        self._publish_debug(frame, dets, command)

    # ------------------------------------------------------------------
    def _detection_loop(self):
        """Runs in background thread — picks up frames and runs full pipeline."""
        YOLO_CONF_TRUST = self._conf
        while self._running:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            # Optional rate cap (max_infer_fps > 0); 0 = unlimited
            if self._min_infer_dt > 0:
                now = time.monotonic()
                wait = self._min_infer_dt - (now - self._last_infer_t)
                if wait > 0:
                    time.sleep(wait)
            self._last_infer_t = time.monotonic()

            raw_dets = detect_signs(frame, self._model, imgsz=self._imgsz)

            # YOLO detections above threshold are trusted immediately
            yolo_direct = [d for d in raw_dets
                           if d[5] >= YOLO_CONF_TRUST
                           and d[0] in {"stop", "workers", "go_straight",
                                        "turn_left", "turn_right"}]
            fallback_raw = [d for d in raw_dets if d not in yolo_direct]
            fallback_stable = self._smoother.update(fallback_raw)

            yolo_labels = {d[0] for d in yolo_direct}
            final_dets  = yolo_direct + [d for d in fallback_stable
                                         if d[0] not in yolo_labels]

            if final_dets:
                # Pick command from detection with largest bounding-box area
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

    # ------------------------------------------------------------------
    def _publish_debug(self, frame, dets, command):
        if self.pub_debug.get_subscription_count() == 0:
            return
        vis     = annotate(frame, dets, command)
        out_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(out_msg)

    # ------------------------------------------------------------------
    def destroy_node(self):
        self._running = False
        super().destroy_node()


# ---------------------------------------------------------------------------
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
