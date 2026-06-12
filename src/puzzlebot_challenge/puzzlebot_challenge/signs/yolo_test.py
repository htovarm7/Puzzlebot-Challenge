#!/usr/bin/env python3
"""
yolo_test.py — Test de YOLO optimizado para Jetson Nano (TensorRT + FP16).

Uso:
  python3 yolo_test.py                  # cámara en vivo
  python3 yolo_test.py imagen.jpg       # imagen estática

Para exportar a TensorRT antes de correr (hazlo una vez en la Jetson):
  python3 yolo_test.py --export
"""

import ctypes
for _lib in (
    '/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0',
    '/usr/lib/aarch64-linux-gnu/libgomp.so.1',
):
    try:
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

import sys
import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PT     = '/home/puzzlebot/Puzzlebot-Challenge/install/puzzlebot_challenge/share/puzzlebot_challenge/models/best.pt'
MODEL_ENGINE = MODEL_PT.replace('.pt', '.engine')
CONF         = 0.45
IMGSZ        = 256
DEVICE       = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_HALF     = torch.cuda.is_available()   # FP16 solo en GPU

GSTREAMER = (
    'nvarguscamerasrc sensor-mode=4 ! '
    'video/x-raw(memory:NVMM),width=1280,height=720,format=NV12,framerate=60/1 ! '
    'nvvidconv ! video/x-raw,width=640,height=480,format=BGRx ! '
    'videoconvert ! video/x-raw,format=BGR ! '
    'appsink drop=true max-buffers=1 sync=false'
)


def export_to_trt():
    """Exporta best.pt a TensorRT engine (correr una sola vez en la Jetson)."""
    print(f'Exportando {MODEL_PT} → TensorRT FP16 ...')
    m = YOLO(MODEL_PT)
    m.export(format='engine', half=True, imgsz=IMGSZ, device=0, workspace=2)
    print(f'Engine guardado en: {MODEL_ENGINE}')


def load_model():
    if os.path.exists(MODEL_ENGINE):
        print(f'Cargando TensorRT engine: {MODEL_ENGINE}')
        model = YOLO(MODEL_ENGINE)
    else:
        print(f'Engine no encontrado, cargando .pt: {MODEL_PT}')
        print('  → Corre "python3 yolo_test.py --export" para generar el engine')
        model = YOLO(MODEL_PT)
    return model


def warmup(model):
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(dummy, conf=CONF, imgsz=IMGSZ, device=DEVICE,
                      half=USE_HALF, verbose=False)
    print('Warmup listo.')


# ── Modo export ──────────────────────────────────────────────────────────────
if '--export' in sys.argv:
    export_to_trt()
    sys.exit(0)

# ── Carga de modelo ──────────────────────────────────────────────────────────
print(f'Device: {DEVICE}  |  half={USE_HALF}')
model = load_model()
warmup(model)
try:
    print(f'Clases: {model.names}')
except Exception:
    pass

# ── Modo imagen estática ─────────────────────────────────────────────────────
if len(sys.argv) > 1:
    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f'ERROR: no se pudo leer {sys.argv[1]}')
        sys.exit(1)
    r = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE,
                      half=USE_HALF, verbose=True)[0]
    print('Detecciones:')
    for box in r.boxes:
        name = model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f'  {name:20} {conf:.0%}  bbox=({x1},{y1},{x2},{y2})')
    if not r.boxes:
        print('  (ninguna)')
    sys.exit(0)

# ── Modo cámara en vivo ──────────────────────────────────────────────────────
print('Abriendo cámara...')
cap = cv2.VideoCapture(GSTREAMER, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print('GStreamer falló, intentando /dev/video0...')
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('ERROR: no se pudo abrir la cámara')
    sys.exit(1)

print('Cámara abierta. Detectando (Ctrl+C para salir)...\n')

fps_t  = time.monotonic()
fps_n  = 0
fps    = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('ERROR leyendo frame')
            time.sleep(0.05)
            continue

        t0 = time.monotonic()
        r  = model.predict(frame, conf=CONF, imgsz=IMGSZ, device=DEVICE,
                           half=USE_HALF, verbose=False)[0]

        # FPS rolling (cada 30 frames)
        fps_n += 1
        if fps_n >= 30:
            fps   = fps_n / (time.monotonic() - fps_t)
            fps_t = time.monotonic()
            fps_n = 0

        if r.boxes:
            for box in r.boxes:
                name = model.names[int(box.cls)]
                conf = float(box.conf)
                print(f'DETECTED: {name:20} {conf:.0%}  |  {fps:.1f} FPS', flush=True)
        else:
            print(f'nada detectado  |  {fps:.1f} FPS', flush=True)

except KeyboardInterrupt:
    pass

cap.release()
print('Listo.')
