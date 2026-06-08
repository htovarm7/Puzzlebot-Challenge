#!/usr/bin/env python3
"""
test_model_video.py — Corre el modelo YOLO de señales sobre un video y reporta
el tamaño (w x h, área) de cada bounding box detectado, frame por frame.

Sirve para ver en qué condiciones el modelo deja de detectar (señal lejana,
borrosa, ángulo, iluminación...) y decidir si conviene un fallback de visión
clásica para esos casos.

Uso:
    python3 scripts/test_model_video.py video.mp4
    python3 scripts/test_model_video.py video.mp4 --model /ruta/best.onnx
    python3 scripts/test_model_video.py video.mp4 --conf 0.5 --imgsz 320 --no-show

Controles (con ventana):
    q / ESC  — salir
    espacio  — pausar / reanudar
"""

import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "src", "puzzlebot_challenge",
                             "utils", "best.onnx")


def parse_args():
    p = argparse.ArgumentParser(description="Prueba del modelo YOLO sobre un video")
    p.add_argument("video", help="Ruta al archivo de video")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ruta al modelo (.onnx/.engine/.pt)")
    p.add_argument("--conf",  type=float, default=0.5, help="Umbral de confianza")
    p.add_argument("--imgsz", type=int,   default=320, help="Tamaño de inferencia")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="No mostrar ventana, solo imprimir resultados")
    p.set_defaults(show=True)
    return p.parse_args()


def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"ERROR: modelo no encontrado: {model_path}")
        sys.exit(1)
    from ultralytics import YOLO
    return YOLO(model_path, task="detect")


def detect(model, frame, conf, imgsz):
    results = model.predict(frame, verbose=False, conf=conf, imgsz=imgsz)[0]
    dets = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        dets.append((label, x1, y1, w, h, float(box.conf)))
    return dets


def draw(frame, dets):
    out = frame.copy()
    for label, x, y, w, h, conf in dets:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, f"{label} {conf:.0%} [{w}x{h}={w*h}px]",
                    (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    return out


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video no encontrado: {args.video}")
        sys.exit(1)

    print(f"Cargando modelo: {args.model}")
    model = load_model(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir el video: {args.video}")
        sys.exit(1)

    win = "test_model_video  |  q=salir  espacio=pausa"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            dets = detect(model, frame, args.conf, args.imgsz)
            if dets:
                for label, x, y, w, h, conf in dets:
                    print(f"frame {frame_idx:5d} | {label:14s} "
                          f"conf={conf:.2f}  bbox={w}x{h}  area={w*h}px")
            else:
                print(f"frame {frame_idx:5d} | sin detección")

            if args.show:
                cv2.imshow(win, draw(frame, dets))

        if args.show:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
