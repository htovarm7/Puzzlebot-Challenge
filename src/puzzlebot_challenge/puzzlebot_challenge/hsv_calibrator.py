#!/usr/bin/env python3
"""HSV calibrator for PuzzleBot traffic light.

Adjust HSV ranges with OpenCV trackbars and save to traffic_hsv.yaml.

Usage: python3 hsv_calibrator.py [--docs DIR] [--out FILE]

Controls: r/g/y (color), 1/2 (range), n/p (image), s (save), q (quit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ── Rangos HSV por defecto ────────────────────────────────────────────────────
# Coinciden con los hardcodeados en traffic_controller.py como punto de partida.
DEFAULT_RANGES: dict = {
    "red": {
        "range1": {"h_min": 0,   "h_max": 8,   "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
        "range2": {"h_min": 172, "h_max": 180,  "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
    },
    "yellow": {
        "range1": {"h_min": 18,  "h_max": 32,   "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
    },
    "green": {
        "range1": {"h_min": 45,  "h_max": 85,   "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
    },
}

COLORS_ORDER = ["red", "yellow", "green"]

# Color de resaltado en BGR para el overlay
HIGHLIGHT_BGR = {
    "red":    (0,   0,   220),
    "yellow": (0,   220, 220),
    "green":  (0,   200, 0),
}

WIN_CTRL = "HSV Calibrador — Controles"
WIN_ORIG = "Original + Overlay"
WIN_MASK = "Mascara pura"


class HsvCalibrator:

    def __init__(self, docs_dir: Path, out_path: Path):
        self.out_path = out_path
        self.ranges   = self._load_or_default(out_path)

        # Cargar imágenes de referencia en orden COLORS_ORDER
        self.img_list: list[tuple[str, np.ndarray]] = []
        for color in COLORS_ORDER:
            p = docs_dir / f"{color}.png"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    self.img_list.append((color, img))

        if not self.img_list:
            print(f"[ERROR] No se encontraron imágenes en {docs_dir}")
            sys.exit(1)

        self.img_idx      = 0
        self.active_color = self.img_list[0][0]
        self.active_range = "range1"

        cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_ORIG, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CTRL, 500, 260)

        self._rebuild_trackbars()

    # ── Persistencia ──────────────────────────────────────────────────────────
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
        # OpenCV no permite eliminar trackbars individualmente; recreamos la ventana.
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

        # Encabezado visual dentro de la ventana de controles
        header = np.zeros((50, 500, 3), dtype=np.uint8)
        label  = f"{self.active_color.upper()}  —  {self.active_range}"
        cv2.putText(header, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, HIGHLIGHT_BGR[self.active_color], 2)
        cv2.imshow(WIN_CTRL, header)

    def _sync_trackbars(self):
        """Lee los trackbars y actualiza self.ranges en memoria."""
        r = self.ranges[self.active_color][self.active_range]
        r["h_min"] = cv2.getTrackbarPos("H min", WIN_CTRL)
        r["h_max"] = cv2.getTrackbarPos("H max", WIN_CTRL)
        r["s_min"] = cv2.getTrackbarPos("S min", WIN_CTRL)
        r["s_max"] = cv2.getTrackbarPos("S max", WIN_CTRL)
        r["v_min"] = cv2.getTrackbarPos("V min", WIN_CTRL)
        r["v_max"] = cv2.getTrackbarPos("V max", WIN_CTRL)

    # ── Detección ─────────────────────────────────────────────────────────────
    def _compute_mask(self, frame_bgr: np.ndarray, color: str) -> np.ndarray:
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rv in self.ranges[color].values():
            lo = np.array([rv["h_min"], rv["s_min"], rv["v_min"]])
            hi = np.array([rv["h_max"], rv["s_max"], rv["v_max"]])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask

    # ── Render ────────────────────────────────────────────────────────────────
    def _render(self, frame_bgr: np.ndarray, ref_name: str):
        self._sync_trackbars()
        mask = self._compute_mask(frame_bgr, self.active_color)
        n_px = int(np.sum(mask > 0))

        # Overlay de color sobre la imagen original
        overlay          = np.full_like(frame_bgr, HIGHLIGHT_BGR[self.active_color])
        blended          = frame_bgr.copy()
        blended[mask > 0] = cv2.addWeighted(
            frame_bgr, 0.3, overlay, 0.7, 0
        )[mask > 0]

        has_r2 = "range2" in self.ranges[self.active_color]
        help_r = "1/2:rango  " if has_r2 else ""
        info   = (f"{self.active_color.upper()} | ref:{ref_name} | "
                  f"px={n_px} | "
                  f"r/g/y:color  {help_r}n/p:img  s:guardar  q:salir")

        h = blended.shape[0]
        cv2.putText(blended, info, (4, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Resumen de rangos activos en la esquina superior
        for i, (rk, rv) in enumerate(self.ranges[self.active_color].items()):
            txt = (f"{rk}: H[{rv['h_min']}-{rv['h_max']}] "
                   f"S[{rv['s_min']}-{rv['s_max']}] "
                   f"V[{rv['v_min']}-{rv['v_max']}]")
            color_txt = (0, 255, 255) if rk == self.active_range else (180, 180, 180)
            cv2.putText(blended, txt, (4, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color_txt, 1)

        cv2.imshow(WIN_ORIG, blended)
        cv2.imshow(WIN_MASK, mask)

    # ── Loop principal ────────────────────────────────────────────────────────
    def run(self):
        print("\n=== Calibrador HSV ===")
        print(f"  Imágenes: {[n for n, _ in self.img_list]}")
        print(f"  Salida:   {self.out_path}")
        print()
        print("  r/g/y   → seleccionar color")
        print("  1/2     → rango activo  (rojo tiene 2 rangos)")
        print("  n/p     → siguiente/anterior imagen")
        print("  s       → guardar YAML")
        print("  q/Esc   → salir\n")

        while True:
            ref_name, frame = self.img_list[self.img_idx]
            self._render(frame, ref_name)

            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                self.active_color = "red"
                self.active_range = "range1"
                self._rebuild_trackbars()
            elif key == ord("g"):
                self.active_color = "green"
                self.active_range = "range1"
                self._rebuild_trackbars()
            elif key == ord("y"):
                self.active_color = "yellow"
                self.active_range = "range1"
                self._rebuild_trackbars()
            elif key == ord("1"):
                self.active_range = "range1"
                self._rebuild_trackbars()
            elif key == ord("2"):
                if "range2" in self.ranges[self.active_color]:
                    self.active_range = "range2"
                    self._rebuild_trackbars()
                else:
                    print(f"[INFO] {self.active_color} solo tiene range1")
            elif key == ord("n"):
                self.img_idx = (self.img_idx + 1) % len(self.img_list)
            elif key == ord("p"):
                self.img_idx = (self.img_idx - 1) % len(self.img_list)
            elif key == ord("s"):
                self.save()

        cv2.destroyAllWindows()


# ── Resolución de rutas por defecto ──────────────────────────────────────────
# Estructura esperada del workspace:
#   <workspace>/
#     docs/                          ← imágenes de referencia
#     src/puzzlebot_challenge/
#       config/traffic_hsv.yaml      ← salida del calibrador
#       puzzlebot_challenge/
#         hsv_calibrator.py          ← este archivo

def _resolve_defaults() -> tuple[Path, Path]:
    here      = Path(__file__).resolve().parent          # .../puzzlebot_challenge/
    pkg_root  = here.parent                              # .../src/puzzlebot_challenge/
    ws_root   = pkg_root.parent.parent                   # workspace root
    docs_dir  = ws_root / "docs"
    out_path  = pkg_root / "config" / "traffic_hsv.yaml"
    return docs_dir, out_path


def main():
    default_docs, default_out = _resolve_defaults()

    ap = argparse.ArgumentParser(description="Calibrador HSV para semáforo PuzzleBot")
    ap.add_argument("--docs", default=str(default_docs),
                    help="Carpeta con red.png, green.png, yellow.png "
                         f"(default: {default_docs})")
    ap.add_argument("--out", default=str(default_out),
                    help="Archivo YAML de salida "
                         f"(default: {default_out})")
    args = ap.parse_args()

    cal = HsvCalibrator(Path(args.docs), Path(args.out))
    cal.run()


if __name__ == "__main__":
    main()
