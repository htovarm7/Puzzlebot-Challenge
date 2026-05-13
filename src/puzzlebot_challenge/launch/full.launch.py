"""Lanza todo el stack: cámara + servidor MJPEG + controlador PID."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_challenge')

    task_arg = DeclareLaunchArgument(
        'task', default_value='SQUARE',
        description='Tarea: SQUARE o WAYPOINTS',
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'camera.launch.py')
        )
    )
    pid = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'pid.launch.py')
        ),
        launch_arguments={'task': LaunchConfiguration('task')}.items(),
    )

    return LaunchDescription([task_arg, camera, pid])
