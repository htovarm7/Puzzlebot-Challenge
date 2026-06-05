#!/usr/bin/env python3
"""
tune_arrow.py — Tunea detección de dirección de flechas por proyección de columnas.

Uso:
    python3 scripts/tune_arrow.py                   # webcam
    python3 scripts/tune_arrow.py imagen.jpg        # imagen estática

Controles:
    q / ESC  — salir
    s        — guardar captura
    p        — imprimir parámetros listos para copiar en sign_detector.py
"""

import sys
import os
import cv2
import numpy as np

WIN = "tune_arrow  |  q=salir  s=guardar  p=params"

# Nombre legible del slider → (valor default, límite máximo, clave interna)
SLIDERS = [
    ("Blanco  H min (0-180)",   0,   180, "h_min"),
    ("Blanco  H max (0-180)", 180,   180, "h_max"),
    ("Blanco  S min (sat)",     0,   255, "s_min"),
    ("Blanco  S max (sat)",    70,   255, "s_max"),
    ("Blanco  V min (brillo)", 160,  255, "v_min"),
    ("Blanco  V max (brillo)", 255,  255, "v_max"),
    ("Suavizado perfil (impar)",  9,  51, "smooth_k"),
    ("Umbral IZQUIERDA  %",      40, 100, "left_thr"),
    ("Umbral DERECHA    %",      60, 100, "right_thr"),
    ("Min pixeles blancos",     300, 5000, "min_white"),
]


def create_ui():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1000, 760)
    for label, val, lim, _ in SLIDERS:
        cv2.createTrackbar(label, WIN, val, lim, lambda _: None)


def get_params():
    p = {}
    for label, _, _, key in SLIDERS:
        p[key] = cv2.getTrackbarPos(label, WIN)
    p["smooth_k"] = max(1, p["smooth_k"] | 1)
    return p


def process(frame, p):
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv,
                        (p["h_min"], p["s_min"], p["v_min"]),
                        (p["h_max"], p["s_max"], p["v_max"]))
    total = int(white.sum() // 255)

    profile = white.sum(axis=0).astype(np.float32)
    k       = p["smooth_k"]
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), 0).flatten()
    peak_x  = int(np.argmax(profile))
    w       = frame.shape[1]
    ratio   = peak_x / max(w, 1)

    if total < p["min_white"]:
        direction = None
    elif ratio < p["left_thr"] / 100:
        direction = "TURN_LEFT"
    elif ratio > p["right_thr"] / 100:
        direction = "TURN_RIGHT"
    else:
        direction = "GO_STRAIGHT"

    return direction, white, profile, peak_x, ratio


def build_frame(frame, p):
    H, W = frame.shape[:2]
    direction, white, profile, peak_x, ratio = process(frame, p)

    # ── Panel superior izquierda: imagen original anotada ─────────────────
    vis = frame.copy()
    overlay = np.zeros_like(vis)
    overlay[white > 0] = (0, 255, 180)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    left_px  = int(p["left_thr"]  / 100 * W)
    right_px = int(p["right_thr"] / 100 * W)
    cv2.line(vis, (left_px,  0), (left_px,  H), (0, 220, 255), 1)
    cv2.line(vis, (right_px, 0), (right_px, H), (0, 220, 255), 1)
    cv2.line(vis, (peak_x,   0), (peak_x,   H), (0, 0, 255), 2)

    color = {"TURN_LEFT":   (255, 120,   0),
             "TURN_RIGHT":  (  0, 120, 255),
             "GO_STRAIGHT": (  0, 220,   0)}.get(direction, (80, 80, 80))
    label = direction or f"AMBIGUO  (px blancos={int(white.sum()//255)})"
    cv2.rectangle(vis, (0, 0), (W, 38), (20, 20, 20), -1)
    cv2.putText(vis, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(vis, f"pico={int(ratio*100)}%  izq<{p['left_thr']}  der>{p['right_thr']}",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 0), 1)

    # ── Panel superior derecha: máscara blanca ────────────────────────────
    mask_bgr = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, "MASCARA BLANCA (HSV)", (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1)

    # ── Panel inferior: perfil de columnas ────────────────────────────────
    PH = 180
    graph = np.zeros((PH, W, 3), np.uint8)
    left_px2  = int(p["left_thr"]  / 100 * W)
    right_px2 = int(p["right_thr"] / 100 * W)
    graph[:, :left_px2]          = (50, 20, 20)
    graph[:, left_px2:right_px2] = (20, 50, 20)
    graph[:, right_px2:]         = (20, 20, 50)

    max_val = float(profile.max()) if profile.max() > 0 else 1.0
    for i, val in enumerate(profile):
        x0 = int(i * W / len(profile))
        x1 = int((i + 1) * W / len(profile))
        bh = int(val / max_val * (PH - 16))
        cv2.rectangle(graph, (x0, PH - bh), (x1, PH), (180, 180, 50), -1)

    cv2.line(graph, (peak_x,    0), (peak_x,    PH), (0, 0, 255), 2)
    cv2.line(graph, (left_px2,  0), (left_px2,  PH), (0, 220, 255), 1)
    cv2.line(graph, (right_px2, 0), (right_px2, PH), (0, 220, 255), 1)
    cv2.putText(graph, "IZQUIERDA",  (4, PH - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 255), 1)
    cv2.putText(graph, "RECTO",      (left_px2 + 4, PH - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 255, 160), 1)
    cv2.putText(graph, "DERECHA",    (right_px2 + 4, PH - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 255), 1)
    cv2.putText(graph, "PERFIL DE COLUMNAS  (pico rojo = direccion flecha)",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ── Ensamblar ──────────────────────────────────────────────────────────
    sep_v_H = np.full((H,  6, 3), 60, np.uint8)
    sep_v_P = np.full((PH, 6, 3), 60, np.uint8)
    sep_h   = np.full((6, W * 2 + 6, 3), 60, np.uint8)

    top_row = np.hstack([vis, sep_v_H, mask_bgr])
    bot_row = np.hstack([graph, sep_v_P, np.zeros((PH, W, 3), np.uint8)])
    return np.vstack([top_row, sep_h, bot_row])


def main():
    use_image = len(sys.argv) > 1 and os.path.isfile(sys.argv[1])

    if use_image:
        static = cv2.imread(sys.argv[1])
        if static is None:
            print(f"No se pudo leer: {sys.argv[1]}")
            sys.exit(1)
        static = cv2.resize(static, (640, 480))
        cap = None
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se encontró webcam")
            sys.exit(1)
        static = None

    create_ui()
    print("q/ESC=salir  s=guardar  p=imprimir params")

    while True:
        frame = static.copy() if static is not None else None
        if frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (640, 480))

        p        = get_params()
        combined = build_frame(frame, p)
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(30 if cap else 50) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            cv2.imwrite("arrow_capture.png", combined)
            print("Guardado: arrow_capture.png")
        elif key == ord('p'):
            print("\n── Copia esto en sign_detector.py ──────────────────")
            print(f"    white = cv2.inRange(hsv, ({p['h_min']}, {p['s_min']}, {p['v_min']}), ({p['h_max']}, {p['s_max']}, {p['v_max']}))")
            print(f"    if int(white.sum() // 255) < {p['min_white']}: return None")
            print(f"    profile = white.sum(axis=0).astype(np.float32)")
            k = p['smooth_k']
            print(f"    profile = cv2.GaussianBlur(profile.reshape(1,-1), ({k},1), 0).flatten()")
            print(f"    ratio = np.argmax(profile) / len(profile)")
            print(f"    if ratio < {p['left_thr']/100:.2f}:  return 'turn_left'")
            print(f"    if ratio > {p['right_thr']/100:.2f}: return 'turn_right'")
            print(f"    return 'go_straight'")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
