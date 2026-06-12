"""Launch the CSI PiCam publisher and the MJPEG server."""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_challenge')
    camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')

    return LaunchDescription([
        Node(
            package='puzzlebot_challenge',
            executable='picam_publisher',
            name='picam_publisher',
            parameters=[camera_cfg],
            output='screen',
        ),
        Node(
            package='puzzlebot_challenge',
            executable='cam_server',
            name='cam_server',
            parameters=[camera_cfg],
            output='screen',
        ),
    ])
