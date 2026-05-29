"""
signs_laptop.launch.py
======================
Corre en la LAPTOP. Detecta señales con YOLO (RTX 4060) y envía los
comandos al Jetson vía HTTP POST al sign_api.

  Laptop: sign_detector_offload → YOLO → HTTP POST → Jetson:sign_api → /sign/command

El Jetson publica /camera/image_raw (picam_publisher corre allá).
La laptop se suscribe a ese tópico por DDS y manda el resultado de vuelta.

Uso:
  # Primero encuentra la IP del Jetson:  hostname -I  (en la Jetson)
  ros2 launch puzzlebot_challenge signs_laptop.launch.py jetson_ip:=192.168.1.50

  # Con parámetros extra:
  ros2 launch puzzlebot_challenge signs_laptop.launch.py jetson_ip:=192.168.1.50 conf_threshold:=0.55
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    args = [
        DeclareLaunchArgument(
            'jetson_ip',
            default_value='10.22.171.82',
            description='IP del Jetson donde corre sign_api (puerto 8081)',
        ),
        DeclareLaunchArgument(
            'conf_threshold',
            default_value='0.50',
            description='Umbral de confianza YOLO (0-1)',
        ),
        DeclareLaunchArgument(
            'imgsz',
            default_value='320',
            description='Tamaño de imagen para inferencia',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Tópico de imagen publicado por el Jetson',
        ),
    ]

    sign_detector = Node(
        package='puzzlebot_challenge',
        executable='sign_detector_offload',
        name='sign_detector_offload',
        parameters=[{
            'image_topic':    LaunchConfiguration('image_topic'),
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'imgsz':          LaunchConfiguration('imgsz'),
            'jetson_api': PythonExpression([
                '"http://" + "', LaunchConfiguration('jetson_ip'), '" + ":8081/sign"'
            ]),
        }],
        output='screen',
    )

    return LaunchDescription(args + [sign_detector])
