"""Graba /camera/image_raw (y topics opcionales) en un rosbag.

Uso básico — levanta la cámara y graba:
  ros2 launch puzzlebot_challenge record_camera.launch.py

Sin levantar la cámara (ya está corriendo en otra terminal):
  ros2 launch puzzlebot_challenge record_camera.launch.py with_camera:=false

Grabar también detección de señales:
  ros2 launch puzzlebot_challenge record_camera.launch.py extra_topics:=/sign/command,/vision/signs

Carpeta de destino personalizada:
  ros2 launch puzzlebot_challenge record_camera.launch.py bag_dir:=/home/hector/mis_bags
"""

import os
import datetime
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    bag_dir      = LaunchConfiguration('bag_dir').perform(context)
    extra_topics = LaunchConfiguration('extra_topics').perform(context)
    with_camera  = LaunchConfiguration('with_camera').perform(context)

    # Nombre del bag con timestamp
    stamp    = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_path = os.path.join(os.path.expanduser(bag_dir), f'camera_{stamp}')

    base_topics = ['/camera/image_raw']
    if extra_topics.strip():
        base_topics += [t.strip() for t in extra_topics.split(',') if t.strip()]

    record_cmd = ['ros2', 'bag', 'record', '-o', bag_path] + base_topics

    actions = []

    if with_camera.lower() in ('true', '1', 'yes'):
        pkg_share  = get_package_share_directory('puzzlebot_challenge')
        camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')
        actions.append(Node(
            package='puzzlebot_challenge',
            executable='picam_publisher',
            name='picam_publisher',
            parameters=[camera_cfg],
            output='screen',
        ))

    actions.append(ExecuteProcess(
        cmd=record_cmd,
        output='screen',
    ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_dir',
            default_value='~/rosbags',
            description='Carpeta donde se guarda el bag (se crea si no existe)',
        ),
        DeclareLaunchArgument(
            'with_camera',
            default_value='true',
            description='Levantar picam_publisher (false si ya está corriendo)',
        ),
        DeclareLaunchArgument(
            'extra_topics',
            default_value='',
            description='Topics extra separados por coma, ej: /sign/command,/vision/signs',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
