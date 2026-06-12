#!/usr/bin/env python3
"""HSV calibrator for PuzzleBot traffic light.

Adjust HSV ranges with OpenCV trackbars and save to traffic_hsv.yaml.

Modes
-----
Live (default) — subscribes to /camera/image_raw via ROS2:
  ros2 run puzzlebot_challenge hsv_calibrator

Static images — uses docs/ reference images (no ROS needed):
  ros2 run puzzlebot_challenge hsv_calibrator --no-live
  python3 hsv_calibrator.py --no-live

Controls: r/g/y (color)  1/2 (range)  n/p (image, static only)  s (save)  q (quit)
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml

DEFAULT_RANGES: dict = {
    "red": {
        "range1": {"h_min": 0,   "h_max": 10,  "s_min": 80, "s_max": 255, "v_min": 31, "v_max": 255},
        "range2": {"h_min": 172, "h_max": 180,  "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
    },
    "yellow": {
        "range1": {"h_min": 15,  "h_max": 38,  "s_min": 60, "s_max": 255, "v_min": 80, "v_max": 255},
    },
    "green": {
        "range1": {"h_min": 45,  "h_max": 85,  "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
    },
}

COLORS_ORDER  = ["red", "yellow", "green"]
HIGHLIGHT_BGR = {"red": (0, 0, 220), "yellow": (0, 210, 210), "green": (0, 200, 0)}

WIN_CTRL = "HSV Calibrador — Controles"
WIN_ORIG = "Original + Overlay"
WIN_MASK = "Mascara pura"


# ── Core calibrator (mode-agnostic) ──────────────────────────────────────────

class HsvCalibrator:

    def __init__(self, out_path: Path, docs_dir: Path | None = None):
        self.out_path = out_path
        self.ranges   = self._load_or_default(out_path)

        # Static images (optional, used in --no-live mode)
        self.img_list: list[tuple[str, np.ndarray]] = []
        if docs_dir:
            for color in COLORS_ORDER:
                p = docs_dir / f"{color}.png"
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        self.img_list.append((color, img))

        self.img_idx      = 0
        self.active_color = COLORS_ORDER[0]
        self.active_range = "range1"

        # Live frame — written by ROS callback, read by main loop
        self._live_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_ORIG, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CTRL, 500, 260)
        self._rebuild_trackbars()

    # ── Frame from ROS callback ───────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._live_frame = frame.copy()

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_or_default(path: Path) -> dict:
        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                print(f"[INFO] Rangos cargados desde {path}")
                return data
            except Exception as e:
                print(f"[WARN] No se pudo leer {path}: {e} — usando defaults")
        return {
            color: {rk: dict(rv) for rk, rv in ranges.items()}
            for color, ranges in DEFAULT_RANGES.items()
        }

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w") as f:
            yaml.dump(self.ranges, f, default_flow_style=False, sort_keys=False)
        print(f"[OK] Guardado en {self.out_path}")

    # ── Trackbars ─────────────────────────────────────────────────────────────

    def _rebuild_trackbars(self):
        cv2.destroyWindow(WIN_CTRL)
        cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CTRL, 500, 260)

        r = self.ranges[self.active_color][self.active_range]

        def cb(_): pass

        cv2.createTrackbar("H min", WIN_CTRL, r["h_min"], 180, cb)
        cv2.createTrackbar("H max", WIN_CTRL, r["h_max"], 180, cb)
        cv2.createTrackbar("S min", WIN_CTRL, r["s_min"], 255, cb)
        cv2.createTrackbar("S max", WIN_CTRL, r["s_max"], 255, cb)
        cv2.createTrackbar("V min", WIN_CTRL, r["v_min"], 255, cb)
        cv2.createTrackbar("V max", WIN_CTRL, r["v_max"], 255, cb)

        header = np.zeros((50, 500, 3), dtype=np.uint8)
        label  = f"{self.active_color.upper()}  —  {self.active_range}"
        cv2.putText(header, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, HIGHLIGHT_BGR[self.active_color], 2)
        cv2.imshow(WIN_CTRL, header)

    def _sync_trackbars(self):
        r = self.ranges[self.active_color][self.active_range]
        r["h_min"] = cv2.getTrackbarPos("H min", WIN_CTRL)
        r["h_max"] = cv2.getTrackbarPos("H max", WIN_CTRL)
        r["s_min"] = cv2.getTrackbarPos("S min", WIN_CTRL)
        r["s_max"] = cv2.getTrackbarPos("S max", WIN_CTRL)
        r["v_min"] = cv2.getTrackbarPos("V min", WIN_CTRL)
        r["v_max"] = cv2.getTrackbarPos("V max", WIN_CTRL)

    # ── Vision ────────────────────────────────────────────────────────────────

    def _compute_mask(self, frame_bgr: np.ndarray, color: str) -> np.ndarray:
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rv in self.ranges[color].values():
            lo = np.array([rv["h_min"], rv["s_min"], rv["v_min"]])
            hi = np.array([rv["h_max"], rv["s_max"], rv["v_max"]])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask

    def _render(self, frame_bgr: np.ndarray, source_label: str):
        self._sync_trackbars()
        mask = self._compute_mask(frame_bgr, self.active_color)
        n_px = int(np.sum(mask > 0))

        overlay = np.full_like(frame_bgr, HIGHLIGHT_BGR[self.active_color])
        blended = frame_bgr.copy()
        blended[mask > 0] = cv2.addWeighted(frame_bgr, 0.3, overlay, 0.7, 0)[mask > 0]

        has_r2 = "range2" in self.ranges[self.active_color]
        help_r = "1/2:rango  " if has_r2 else ""
        info   = (f"{self.active_color.upper()} | {source_label} | px={n_px} | "
                  f"r/g/y:color  {help_r}n/p:img  s:guardar  q:salir")
        cv2.putText(blended, info, (4, blended.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        for i, (rk, rv) in enumerate(self.ranges[self.active_color].items()):
            txt = (f"{rk}: H[{rv['h_min']}-{rv['h_max']}] "
                   f"S[{rv['s_min']}-{rv['s_max']}] "
                   f"V[{rv['v_min']}-{rv['v_max']}]")
            clr = (0, 255, 255) if rk == self.active_range else (180, 180, 180)
            cv2.putText(blended, txt, (4, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)

        cv2.imshow(WIN_ORIG, blended)
        cv2.imshow(WIN_MASK, mask)

    # ── Key handling ──────────────────────────────────────────────────────────

    def handle_key(self, key: int) -> bool:
        """Returns True if the loop should exit."""
        if key in (ord("q"), 27):
            return True
        elif key == ord("r"):
            self.active_color = "red";    self.active_range = "range1"; self._rebuild_trackbars()
        elif key == ord("g"):
            self.active_color = "green";  self.active_range = "range1"; self._rebuild_trackbars()
        elif key == ord("y"):
            self.active_color = "yellow"; self.active_range = "range1"; self._rebuild_trackbars()
        elif key == ord("1"):
            self.active_range = "range1"; self._rebuild_trackbars()
        elif key == ord("2"):
            if "range2" in self.ranges[self.active_color]:
                self.active_range = "range2"; self._rebuild_trackbars()
        elif key == ord("n") and self.img_list:
            self.img_idx = (self.img_idx + 1) % len(self.img_list)
        elif key == ord("p") and self.img_list:
            self.img_idx = (self.img_idx - 1) % len(self.img_list)
        elif key == ord("s"):
            self.save()
        return False

    # ── Live loop ─────────────────────────────────────────────────────────────

    def run_live(self, spin_once_fn):
        """Main loop for live camera mode. spin_once_fn pumps the ROS executor."""
        print("\n=== Calibrador HSV — MODO LIVE ===")
        print(f"  Topic:  /camera/image_raw")
        print(f"  Salida: {self.out_path}")
        print("  Esperando primer frame...\n")

        while True:
            spin_once_fn()

            with self._frame_lock:
                frame = self._live_frame

            if frame is not None:
                self._render(frame, "LIVE")

            key = cv2.waitKey(1) & 0xFF
            if self.handle_key(key):
                break

        cv2.destroyAllWindows()

    # ── Static loop ───────────────────────────────────────────────────────────

    def run_static(self):
        if not self.img_list:
            print("[ERROR] No hay imágenes en docs/ y el modo live está desactivado.")
            sys.exit(1)

        print("\n=== Calibrador HSV — MODO IMÁGENES ===")
        print(f"  Imágenes: {[n for n, _ in self.img_list]}")
        print(f"  Salida:   {self.out_path}\n")

        while True:
            ref_name, frame = self.img_list[self.img_idx]
            self._render(frame, f"img:{ref_name}")

            key = cv2.waitKey(30) & 0xFF
            if self.handle_key(key):
                break

        cv2.destroyAllWindows()


# ── ROS2 node wrapper ─────────────────────────────────────────────────────────

def _run_with_ros(cal: HsvCalibrator, topic: str):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    rclpy.init()

    class _CamNode(Node):
        def __init__(self):
            super().__init__("hsv_calibrator")
            self.bridge = CvBridge()
            self.create_subscription(Image, topic, self._cb, 10)
            self.get_logger().info(f"Subscrito a {topic}")

        def _cb(self, msg: Image):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cal.push_frame(frame)
            except Exception:
                pass

    node = _CamNode()

    try:
        cal.run_live(lambda: rclpy.spin_once(node, timeout_sec=0.01))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ── Path resolution ───────────────────────────────────────────────────────────

def _resolve_defaults() -> tuple[Path, Path]:
    here     = Path(__file__).resolve().parent
    pkg_root = here.parent
    ws_root  = pkg_root.parent.parent
    return ws_root / "docs", pkg_root / "config" / "traffic_hsv.yaml"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    default_docs, default_out = _resolve_defaults()

    ap = argparse.ArgumentParser(description="Calibrador HSV — semáforo PuzzleBot")
    ap.add_argument("--no-live", action="store_true",
                    help="Usar imágenes estáticas de docs/ en lugar de la cámara")
    ap.add_argument("--topic", default="/camera/image_raw",
                    help="Tópico ROS2 de imagen (default: /camera/image_raw)")
    ap.add_argument("--docs", default=str(default_docs),
                    help=f"Carpeta con imágenes de referencia (default: {default_docs})")
    ap.add_argument("--out", default=str(default_out),
                    help=f"Archivo YAML de salida (default: {default_out})")
    args = ap.parse_args()

    docs_dir = Path(args.docs) if args.no_live else None
    cal      = HsvCalibrator(Path(args.out), docs_dir=docs_dir)

    if args.no_live:
        cal.run_static()
    else:
        _run_with_ros(cal, args.topic)


if __name__ == "__main__":
    main()
