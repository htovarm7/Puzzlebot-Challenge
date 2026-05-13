#!/usr/bin/env python3
"""
test_detector_images.py
=======================
Prueba la clase TrafficLightDetection con imágenes guardadas en disco.
No requiere ROS, ni cámara, ni nada — solo OpenCV y numpy.

Uso
---
  python3 test_detector_images.py foto_roja.jpg foto_verde.jpg foto_amarilla.jpg
  python3 test_detector_images.py *.jpg
"""

import sys
import cv2

# Importa la clase del detector. Asume que traffic_detector.py
# está en la misma carpeta.
from traffic_detector import TrafficLightDetection


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 test_detector_images.py <imagen1> [imagen2] ...")
        sys.exit(1)

    # Crea el detector con los mismos parámetros default del nodo
    detector = TrafficLightDetection(min_pixels=50, roi_fraction=0.5)

    print(f"\nProbando {len(sys.argv) - 1} imágenes con min_pixels=50, roi=mitad izquierda\n")
    print(f"{'Archivo':<40} {'R':>8} {'Y':>8} {'G':>8}   →  Estado")
    print("-" * 80)

    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            print(f"{path:<40}   ❌ No se pudo leer")
            continue

        state, counts = detector.detect_state(img)

        # Imprime conteos y resultado
        print(f"{path:<40} {counts['red']:>8} {counts['yellow']:>8} {counts['green']:>8}"
              f"   →  {state}")

    print()


if __name__ == '__main__':
    main()