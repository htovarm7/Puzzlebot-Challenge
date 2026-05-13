"""Lanza el controlador PID. La tarea (square/waypoints) se elige con un argumento."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    task_arg = DeclareLaunchArgument(
        'task', default_value='waypoints',
        description='Tarea: square o waypoints',
    )

    return LaunchDescription([
        task_arg,
        Node(
            package='puzzlebot_challenge',
            executable='pid_controller',
            name='puzzlebot_motion_pd',
            arguments=[LaunchConfiguration('task')],
            output='screen',
        ),
    ])
