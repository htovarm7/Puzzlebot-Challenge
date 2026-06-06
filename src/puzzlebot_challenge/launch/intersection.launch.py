"""PiCam publisher + intersection detector launch file."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_challenge')
    camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')
    inter_cfg  = os.path.join(pkg_share, 'config', 'intersection_params.yaml')

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
            executable='intersection_detector',
            name='intersection_detector',
            parameters=[{'params_config': inter_cfg}],
            output='screen',
        ),
    ])