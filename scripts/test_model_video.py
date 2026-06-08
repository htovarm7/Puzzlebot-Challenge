#!/usr/bin/env python3
"""
test_model_video.py — Corre el modelo YOLO de señales (best.pt) sobre uno o
varios videos y reporta el tamaño (w x h, área) de cada bounding box
detectado, frame por frame.

Sirve para ver en qué condiciones el modelo deja de detectar (señal lejana,
borrosa, ángulo, iluminación...) y decidir si conviene un fallback de visión
clásica para esos casos.

Uso:
    python3 scripts/test_model_video.py video.mp4
    python3 scripts/test_model_video.py scripts/videos/          # procesa todos los .mp4 de la carpeta
    python3 scripts/test_model_video.py scripts/videos/ --conf 0.5 --imgsz 320 --no-show

Controles (con ventana):
    q / ESC          — salir (pasa al siguiente video / termina si es el último)
    espacio          — pausar / reanudar
    flecha derecha/d — adelantar (salta frames hacia adelante)
    flecha izquierda/a — retroceder (salta frames hacia atrás)
"""

import argparse
import glob
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "src", "puzzlebot_challenge",
                             "utils", "best.pt")
SEEK_FRAMES = 30  # cuántos frames salta cada vez que se adelanta/retrocede

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def parse_args():
    p = argparse.ArgumentParser(description="Prueba del modelo YOLO sobre uno o varios videos")
    p.add_argument("video", help="Ruta a un video o a una carpeta con videos")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ruta al modelo (.pt/.onnx/.engine)")
    p.add_argument("--conf",  type=float, default=0.7, help="Umbral de confianza")
    p.add_argument("--imgsz", type=int,   default=320, help="Tamaño de inferencia")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="No mostrar ventana, solo imprimir resultados")
    p.set_defaults(show=True)
    return p.parse_args()


def collect_videos(path):
    if os.path.isdir(path):
        files = []
        for ext in VIDEO_EXTS:
            files.extend(glob.glob(os.path.join(path, f"*{ext}")))
        return sorted(files)
    return [path]


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


LEFT_KEYS = (ord('a'), 81, 2)    # 'a', flecha izquierda (Linux/Windows)
RIGHT_KEYS = (ord('d'), 83, 3)   # 'd', flecha derecha (Linux/Windows)


def process_video(video_path, model, args, win):
    print(f"\n=== {os.path.basename(video_path)} ===")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir el video: {video_path}")
        return True

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    paused = False
    keep_going = True

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            dets = detect(model, frame, args.conf, args.imgsz)
            if dets:
                for label, x, y, w, h, conf in dets:
                    print(f"frame {frame_idx:5d}/{total_frames} | {label:14s} "
                          f"conf={conf:.2f}  bbox={w}x{h}  area={w*h}px")
            else:
                print(f"frame {frame_idx:5d}/{total_frames} | sin detección")

            if args.show:
                cv2.imshow(win, draw(frame, dets))

        if args.show:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                keep_going = False
                break
            if key == ord(' '):
                paused = not paused
            elif key in RIGHT_KEYS or key in LEFT_KEYS:
                step = SEEK_FRAMES if key in RIGHT_KEYS else -SEEK_FRAMES
                frame_idx = max(0, min(total_frames - 1, frame_idx + step))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    cap.release()
    return keep_going


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: ruta no encontrada: {args.video}")
        sys.exit(1)

    videos = collect_videos(args.video)
    if not videos:
        print(f"ERROR: no se encontraron videos en: {args.video}")
        sys.exit(1)

    print(f"Cargando modelo: {args.model}")
    model = load_model(args.model)

    win = "test_model_video  |  q=siguiente/salir  espacio=pausa  a/d=retroceder/adelantar"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for video_path in videos:
        if not process_video(video_path, model, args, win):
            break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
