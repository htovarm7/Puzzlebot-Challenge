#!/usr/bin/env python3
"""
yolo_test.py — Test directo de YOLO con la cámara en el Jetson.

Uso:
  python3 yolo_test.py                  # cámara en vivo
  python3 yolo_test.py imagen.jpg       # imagen estática
"""

import ctypes
ctypes.CDLL('/usr/lib/aarch64-linux-gnu/libgomp.so.1', mode=ctypes.RTLD_GLOBAL)

import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = '/home/puzzlebot/Puzzlebot-Challenge/install/puzzlebot_challenge/share/puzzlebot_challenge/models/best.pt'
CONF       = 0.10
IMGSZ      = 320

GSTREAMER = (
    'nvarguscamerasrc sensor-mode=2 ! '
    'video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1 ! '
    'nvvidconv ! video/x-raw,width=320,height=240,format=BGRx ! '
    'videoconvert ! video/x-raw,format=BGR ! '
    'appsink name=sink drop=true max-buffers=1 emit-signals=true sync=false'
)

print(f'Cargando modelo: {MODEL_PATH}')
model = YOLO(MODEL_PATH)
print(f'Clases: {model.names}')

if len(sys.argv) > 1:
    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f'ERROR: no se pudo leer {sys.argv[1]}')
        sys.exit(1)
    r = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=True)[0]
    print('Detecciones:')
    for box in r.boxes:
        name = model.names[int(box.cls)]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f'  {name:20} {conf:.0%}  bbox=({x1},{y1},{x2},{y2})')
    if not r.boxes:
        print('  (ninguna)')
    sys.exit(0)

print('Abriendo cámara...')
cap = cv2.VideoCapture(GSTREAMER, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print('GStreamer falló, intentando /dev/video0...')
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('ERROR: no se pudo abrir la cámara')
    sys.exit(1)

print('Cámara abierta. Detectando (Ctrl+C para salir)...\n')
frame_n = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print('ERROR leyendo frame')
            time.sleep(0.1)
            continue

        frame_n += 1
        if frame_n % 3 != 0:
            continue

        r = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]

        if r.boxes:
            for box in r.boxes:
                name = model.names[int(box.cls)]
                conf = float(box.conf)
                print(f'DETECTED: {name:20} {conf:.0%}', flush=True)
        else:
            print(f'frame={frame_n}  nada detectado', flush=True)

except KeyboardInterrupt:
    pass

cap.release()
print('Listo.')
