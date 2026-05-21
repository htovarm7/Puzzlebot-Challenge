from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='puzzlebot_challenge',
            executable='teleop',
            name='puzzlebot_teleop',
            output='screen',
            emulate_tty=True,
        ),
    ])
