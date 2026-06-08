#!/usr/bin/env python3
"""
test_model_picam.py — Corre el modelo YOLO de señales (.onnx) en vivo sobre
el feed de la PiCam (/camera/image_raw) y reporta por frame la clase
detectada, la confianza y el tamaño del bounding box (w x h, área).

Sirve para validar el modelo exportado a ONNX directamente con la cámara
real, sin pasar por sign_detector.

Uso:
    ros2 run puzzlebot_challenge picam_publisher    # si no está corriendo
    python3 scripts/test_model_picam.py
    python3 scripts/test_model_picam.py --model ruta/a/best.onnx --conf 0.7

Controles (con ventana de preview):
    q / ESC  — salir
"""

import argparse
import os
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "src", "puzzlebot_challenge",
                             "utils", "best.onnx")


def parse_args():
    p = argparse.ArgumentParser(description="Prueba en vivo del modelo ONNX con la PiCam")
    p.add_argument("--topic", default="/camera/image_raw",
                   help="Topic de imagen a usar (default: /camera/image_raw)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ruta al modelo .onnx")
    p.add_argument("--conf",  type=float, default=0.7, help="Umbral de confianza")
    p.add_argument("--imgsz", type=int,   default=320, help="Tamaño de inferencia")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="No mostrar ventana, solo imprimir resultados")
    p.set_defaults(show=True)
    return p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])


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


class PicamFeed(Node):

    def __init__(self, topic):
        super().__init__('test_model_picam')
        self.bridge = CvBridge()
        self.frame  = None
        self.create_subscription(Image, topic, self._on_image, 10)
        self.get_logger().info(f"Esperando frames en {topic}...")

    def _on_image(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


def main():
    rclpy.init()
    args = parse_args()

    print(f"Cargando modelo: {args.model}")
    model = load_model(args.model)

    node = PicamFeed(args.topic)

    win = "test_model_picam  |  q/ESC = salir"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = node.frame
            if frame is None:
                continue
            node.frame = None
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
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
