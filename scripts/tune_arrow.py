#!/usr/bin/env python3
"""
tune_arrow.py — Tunea los parámetros de detección de dirección de flechas.

Uso:
    python3 scripts/tune_arrow.py                        # webcam
    python3 scripts/tune_arrow.py imagen.jpg             # imagen estática
    python3 scripts/tune_arrow.py imagen.jpg --loop      # repite la imagen como video

Controles:
    q / ESC  — salir
    s        — guardar captura en arrow_capture.png
    p        — imprimir parámetros actuales en terminal
"""

import sys
import os
import cv2
import numpy as np

# ── Parámetros iniciales (iguales al sign_detector) ──────────────────────────
DEFAULTS = dict(
    h_min=0,   h_max=180,
    s_min=0,   s_max=70,
    v_min=160, v_max=255,
    left_thr=44,   # cx/w < left_thr/100  → turn_left
    right_thr=56,  # cx/w > right_thr/100 → turn_right
    min_area=100,
)

WIN_CTRL  = "Controles"
WIN_ORIG  = "Original + deteccion"
WIN_MASK  = "Mascara blanca"


def create_trackbars():
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 400, 500)
    for name, val in DEFAULTS.items():
        hi = 180 if name.startswith("h") else 255 if name in ("s_min","s_max","v_min","v_max") else 100 if "thr" in name else 5000
        cv2.createTrackbar(name, WIN_CTRL, val, hi, lambda _: None)


def get_params():
    return {k: cv2.getTrackbarPos(k, WIN_CTRL) for k in DEFAULTS}


def detect_direction(crop, p):
    """Misma lógica que sign_detector._contour_arrow_direction con params tuneables."""
    if crop.size == 0:
        return None, None, None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       (p["h_min"], p["s_min"], p["v_min"]),
                       (p["h_max"], p["s_max"], p["v_max"]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask, None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < p["min_area"]:
        return None, mask, cnt
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, mask, cnt
    cx = M["m10"] / M["m00"]
    w  = crop.shape[1]
    ratio = cx / w
    if ratio < p["left_thr"] / 100:
        direction = "TURN_LEFT"
    elif ratio > p["right_thr"] / 100:
        direction = "TURN_RIGHT"
    else:
        direction = "GO_STRAIGHT"
    return direction, mask, cnt


def draw_overlay(frame, p):
    h, w = frame.shape[:2]
    # Usa el frame completo como crop (para probar con imagen centrada)
    direction, mask, cnt = detect_direction(frame, p)

    vis = frame.copy()

    if cnt is not None:
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
            ratio_pct = int(cx / w * 100)
            cv2.putText(vis, f"cx/w = {ratio_pct}%  [{p['left_thr']}..{p['right_thr']}]",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    color = (0, 200, 0) if direction else (0, 0, 200)
    label = direction or "AMBIGUO"
    cv2.rectangle(vis, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(vis, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    mask_bgr = cv2.cvtColor(mask if mask is not None else np.zeros((h, w), np.uint8),
                            cv2.COLOR_GRAY2BGR)
    return vis, mask_bgr


def main():
    use_image  = len(sys.argv) > 1 and os.path.isfile(sys.argv[1])
    loop_image = "--loop" in sys.argv

    if use_image:
        static = cv2.imread(sys.argv[1])
        if static is None:
            print(f"No se pudo leer: {sys.argv[1]}")
            sys.exit(1)
        cap = None
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se encontró webcam")
            sys.exit(1)
        static = None

    cv2.namedWindow(WIN_ORIG, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_ORIG, 640, 480)
    cv2.resizeWindow(WIN_MASK, 640, 480)
    create_trackbars()

    print("Controles: q/ESC=salir  s=guardar  p=imprimir params")

    while True:
        if static is not None:
            frame = static.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                break

        p = get_params()
        vis, mask_bgr = draw_overlay(frame, p)

        cv2.imshow(WIN_ORIG, vis)
        cv2.imshow(WIN_MASK, mask_bgr)

        key = cv2.waitKey(30 if (cap or loop_image) else 50) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            cv2.imwrite("arrow_capture.png", vis)
            print("Guardado: arrow_capture.png")
        elif key == ord('p'):
            print("\n── Parámetros actuales ──────────────────")
            print(f"  HSV blanco : ({p['h_min']},{p['s_min']},{p['v_min']}) → ({p['h_max']},{p['s_max']},{p['v_max']})")
            print(f"  left_thr   : {p['left_thr']}  (cx/w < {p['left_thr']/100:.2f} → TURN_LEFT)")
            print(f"  right_thr  : {p['right_thr']}  (cx/w > {p['right_thr']/100:.2f} → TURN_RIGHT)")
            print(f"  min_area   : {p['min_area']}")
            print()
            print("  Copia esto en sign_detector.py:")
            print(f"    white_mask = cv2.inRange(hsv, ({p['h_min']}, {p['s_min']}, {p['v_min']}), ({p['h_max']}, {p['s_max']}, {p['v_max']}))")
            print(f"    if ratio < {p['left_thr']/100:.2f}:  return 'turn_left'")
            print(f"    if ratio > {p['right_thr']/100:.2f}:  return 'turn_right'")
            print(f"    return 'go_straight'")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
