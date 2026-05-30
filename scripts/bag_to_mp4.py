#!/usr/bin/env python3
"""Convierte un rosbag2 con /camera/image_raw a MP4.

Uso:
  python3 bag_to_mp4.py <ruta_al_bag>
  python3 bag_to_mp4.py <ruta_al_bag> -o salida.mp4
  python3 bag_to_mp4.py <ruta_al_bag> -t /camera/image_raw -fps 30
"""

import argparse
import os
import sys
import cv2
import numpy as np

try:
    import rclpy
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    import rosbag2_py
except ImportError:
    sys.exit("Falta rosbag2_py. Asegúrate de tener el entorno ROS2 activado:\n  source /opt/ros/humble/setup.bash")


def bag_to_mp4(bag_path: str, output: str, topic: str, fps: float):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        available = list(topic_types.keys())
        sys.exit(f"Topic '{topic}' no encontrado.\nDisponibles: {available}")

    msg_type = get_message(topic_types[topic])

    filter_ = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(filter_)

    writer = None
    frame_count = 0

    print(f"Leyendo: {bag_path}")
    print(f"Topic:   {topic}")

    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, msg_type)

        # Convierte sensor_msgs/Image a numpy
        h, w = msg.height, msg.width
        enc  = msg.encoding.lower()

        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in ('bgr8', 'rgb8'):
            frame = raw.reshape(h, w, 3)
            if enc == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif enc in ('mono8', '8uc1'):
            frame = cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2BGR)
        elif enc in ('yuv422', 'yuyv'):
            frame = cv2.cvtColor(raw.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        else:
            # Intento genérico
            frame = raw.reshape(h, w, -1)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output, fourcc, fps, (w, h))
            print(f"Resolución: {w}x{h} @ {fps} fps")
            print(f"Salida:  {output}")

        writer.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  {frame_count} frames procesados...", end='\r')

    if writer:
        writer.release()

    if frame_count == 0:
        sys.exit("No se encontraron frames en el bag.")

    print(f"\nListo: {frame_count} frames → {output}")

    # Re-encode con ffmpeg para compatibilidad máxima (H.264)
    tmp = output.replace('.mp4', '_tmp.mp4')
    os.rename(output, tmp)
    ret = os.system(f'ffmpeg -y -i "{tmp}" -vcodec libx264 -crf 23 -preset fast "{output}" -loglevel warning')
    os.remove(tmp)
    if ret == 0:
        size_mb = os.path.getsize(output) / 1e6
        print(f"H.264 re-encoded: {output} ({size_mb:.1f} MB)")
    else:
        os.rename(tmp, output)
        print("ffmpeg no disponible, se dejó como mp4v.")


def main():
    parser = argparse.ArgumentParser(description='Convierte rosbag2 a MP4')
    parser.add_argument('bag', help='Ruta al directorio del bag (ej: ~/rosbags/camera_20250101_120000)')
    parser.add_argument('-o', '--output', default='', help='Archivo de salida (default: mismo nombre que el bag)')
    parser.add_argument('-t', '--topic', default='/camera/image_raw', help='Topic de imagen')
    parser.add_argument('--fps', type=float, default=30.0, help='FPS del video (default: 30)')
    args = parser.parse_args()

    bag_path = os.path.expanduser(args.bag).rstrip('/')
    output   = args.output or bag_path + '.mp4'

    bag_to_mp4(bag_path, output, args.topic, args.fps)


if __name__ == '__main__':
    main()
