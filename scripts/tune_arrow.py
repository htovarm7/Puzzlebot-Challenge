#!/usr/bin/env python3
"""
tune_arrow.py — Tunea detección de dirección de flechas por proyección de columnas.

La idea: para cada columna x del crop, cuenta cuántos píxeles blancos hay.
Eso da un perfil horizontal. El pico del perfil indica dónde está la parte
más ancha de la flecha (base del arrowhead):
  - Turn-Left  : pico en la mitad izquierda
  - Turn-Right : pico en la mitad derecha
  - Go-Straight: pico centrado, perfil simétrico

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

DEFAULTS = dict(
    h_min=0,   h_max=180,
    s_min=0,   s_max=70,
    v_min=160, v_max=255,
    smooth_k=9,       # kernel gaussiano para suavizar el perfil (impar)
    left_thr=40,      # peak_x/w < left_thr/100  → turn_left
    right_thr=60,     # peak_x/w > right_thr/100 → turn_right
    min_white=300,    # mínimo de píxeles blancos totales para considerar válido
)

WIN_CTRL    = "Controles"
WIN_ORIG    = "Original + deteccion"
WIN_PROFILE = "Perfil de columnas"


def create_trackbars():
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 480)
    limits = dict(h_min=180, h_max=180, s_min=255, s_max=255,
                  v_min=255, v_max=255, smooth_k=51,
                  left_thr=100, right_thr=100, min_white=5000)
    for name, val in DEFAULTS.items():
        cv2.createTrackbar(name, WIN_CTRL, val, limits[name], lambda _: None)


def get_params():
    p = {k: cv2.getTrackbarPos(k, WIN_CTRL) for k in DEFAULTS}
    # smooth_k debe ser impar y >= 1
    p["smooth_k"] = max(1, p["smooth_k"] | 1)
    return p


def white_mask(frame, p):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,
                       (p["h_min"], p["s_min"], p["v_min"]),
                       (p["h_max"], p["s_max"], p["v_max"]))


def column_profile(mask, smooth_k):
    """Suma de píxeles blancos por columna, suavizado con gaussiana."""
    profile = mask.sum(axis=0).astype(np.float32)  # (w,)
    if smooth_k > 1:
        profile = cv2.GaussianBlur(profile.reshape(1, -1),
                                   (smooth_k, 1), 0).flatten()
    return profile


def detect_direction(frame, p):
    mask = white_mask(frame, p)
    total = int(mask.sum() // 255)
    if total < p["min_white"]:
        return None, mask, None, None

    profile = column_profile(mask, p["smooth_k"])
    peak_x  = int(np.argmax(profile))
    w       = frame.shape[1]
    ratio   = peak_x / w

    if ratio < p["left_thr"] / 100:
        direction = "TURN_LEFT"
    elif ratio > p["right_thr"] / 100:
        direction = "TURN_RIGHT"
    else:
        direction = "GO_STRAIGHT"

    return direction, mask, profile, peak_x


def draw_profile(profile, peak_x, w_out, h_out, left_thr, right_thr):
    """Dibuja el perfil de columnas como gráfica de barras."""
    canvas = np.zeros((h_out, w_out, 3), np.uint8)
    if profile is None or len(profile) == 0:
        return canvas

    profile_w = len(profile)
    max_val   = float(profile.max()) if profile.max() > 0 else 1.0
    bar_w     = max(1, w_out // profile_w)

    # Zonas de color de fondo
    left_px  = int(left_thr  / 100 * w_out)
    right_px = int(right_thr / 100 * w_out)
    canvas[:, :left_px]  = (40, 0, 0)    # azul oscuro = zona turn_left
    canvas[:, left_px:right_px] = (0, 40, 0)   # verde oscuro = zona go_straight
    canvas[:, right_px:] = (0, 0, 40)    # rojo oscuro = zona turn_right

    for i, val in enumerate(profile):
        x0 = int(i * w_out / profile_w)
        x1 = x0 + bar_w
        bar_h = int(val / max_val * (h_out - 20))
        color = (200, 200, 50)
        cv2.rectangle(canvas, (x0, h_out - bar_h), (x1, h_out), color, -1)

    # Línea del pico
    peak_draw = int(peak_x * w_out / profile_w)
    cv2.line(canvas, (peak_draw, 0), (peak_draw, h_out), (0, 0, 255), 2)
    cv2.putText(canvas, f"peak={int(peak_x * 100 / profile_w)}%",
                (peak_draw + 4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Líneas de threshold
    cv2.line(canvas, (left_px,  0), (left_px,  h_out), (255, 200, 0), 1)
    cv2.line(canvas, (right_px, 0), (right_px, h_out), (255, 200, 0), 1)

    cv2.putText(canvas, "LEFT", (4, h_out - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
    cv2.putText(canvas, "STRAIGHT", (left_px + 4, h_out - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)
    cv2.putText(canvas, "RIGHT", (right_px + 4, h_out - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
    return canvas


def draw_overlay(frame, p):
    h, w = frame.shape[:2]
    direction, mask, profile, peak_x = detect_direction(frame, p)

    vis = frame.copy()

    # Líneas de threshold verticales sobre la imagen
    left_px  = int(p["left_thr"]  / 100 * w)
    right_px = int(p["right_thr"] / 100 * w)
    cv2.line(vis, (left_px,  0), (left_px,  h), (0, 200, 255), 1)
    cv2.line(vis, (right_px, 0), (right_px, h), (0, 200, 255), 1)

    if peak_x is not None:
        cv2.line(vis, (peak_x, 0), (peak_x, h), (0, 0, 255), 2)
        ratio_pct = int(peak_x * 100 / w)
        cv2.putText(vis, f"peak={ratio_pct}%", (peak_x + 4, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    color = {
        "TURN_LEFT":   (255, 100, 0),
        "TURN_RIGHT":  (0, 100, 255),
        "GO_STRAIGHT": (0, 220, 0),
    }.get(direction, (0, 0, 200))
    label = direction or "AMBIGUO (pocos px blancos)"
    cv2.rectangle(vis, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(vis, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # Overlay de máscara semitransparente
    mask_color = np.zeros_like(frame)
    mask_color[mask > 0] = (0, 255, 200)
    vis = cv2.addWeighted(vis, 0.75, mask_color, 0.25, 0)

    profile_img = draw_profile(profile, peak_x if peak_x is not None else 0,
                                w, 200, p["left_thr"], p["right_thr"])
    return vis, profile_img


def main():
    use_image = len(sys.argv) > 1 and os.path.isfile(sys.argv[1])

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

    cv2.namedWindow(WIN_ORIG,    cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_PROFILE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_ORIG,    640, 480)
    cv2.resizeWindow(WIN_PROFILE, 640, 200)
    create_trackbars()

    print("Controles: q/ESC=salir  s=guardar  p=imprimir params")

    while True:
        frame = static.copy() if static is not None else None
        if frame is None:
            ok, frame = cap.read()
            if not ok:
                break

        p = get_params()
        vis, profile_img = draw_overlay(frame, p)

        cv2.imshow(WIN_ORIG,    vis)
        cv2.imshow(WIN_PROFILE, profile_img)

        key = cv2.waitKey(30 if cap else 50) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            cv2.imwrite("arrow_capture.png", vis)
            print("Guardado: arrow_capture.png")
        elif key == ord('p'):
            print("\n── Parámetros para sign_detector.py ────────────────")
            print(f"  HSV blanco : ({p['h_min']}, {p['s_min']}, {p['v_min']}) "
                  f"→ ({p['h_max']}, {p['s_max']}, {p['v_max']})")
            print(f"  smooth_k   : {p['smooth_k']}")
            print(f"  left_thr   : {p['left_thr']/100:.2f}   (peak < este valor → TURN_LEFT)")
            print(f"  right_thr  : {p['right_thr']/100:.2f}  (peak > este valor → TURN_RIGHT)")
            print(f"  min_white  : {p['min_white']} px")
            print()
            print("  ── Código listo para copiar ──")
            print(f"    white = cv2.inRange(hsv, ({p['h_min']}, {p['s_min']}, {p['v_min']}), ({p['h_max']}, {p['s_max']}, {p['v_max']}))")
            print(f"    profile = white.sum(axis=0).astype('float32')")
            k = p['smooth_k']
            print(f"    profile = cv2.GaussianBlur(profile.reshape(1,-1), ({k},1), 0).flatten()")
            print(f"    peak_ratio = np.argmax(profile) / len(profile)")
            print(f"    if peak_ratio < {p['left_thr']/100:.2f}:   return 'turn_left'")
            print(f"    if peak_ratio > {p['right_thr']/100:.2f}:  return 'turn_right'")
            print(f"    return 'go_straight'")

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
