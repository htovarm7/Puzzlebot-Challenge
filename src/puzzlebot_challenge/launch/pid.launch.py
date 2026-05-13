"""Lanza el controlador PID. La tarea (square/waypoints) se elige con un argumento."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_challenge')
    pid_cfg = os.path.join(pkg_share, 'config', 'pid.yaml')

    task_arg = DeclareLaunchArgument(
        'task', default_value='SQUARE',
        description='Tarea: SQUARE o WAYPOINTS',
    )

    return LaunchDescription([
        task_arg,
        Node(
            package='puzzlebot_challenge',
            executable='pid_controller',
            name='pid_controller',
            parameters=[pid_cfg, {'task': LaunchConfiguration('task')}],
            output='screen',
        ),
    ])
