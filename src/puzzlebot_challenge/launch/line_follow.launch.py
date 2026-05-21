"""
line_follow.launch.py
=====================
Launches the full line-following stack:
  1. picam_publisher   – CSI camera driver
  2. line_detector     – vision: publishes /line/shift, /line/angle, /line/detected
  3. line_follower     – PID control: subscribes to /line/* → drives /VelocitySet{L,R}
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_challenge')
    camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')
    line_cfg   = os.path.join(pkg_share, 'config', 'line_params.yaml')

    kp_arg    = DeclareLaunchArgument('kp',          default_value='0.006',  description='PID P gain')
    ki_arg    = DeclareLaunchArgument('ki',          default_value='0.0002', description='PID I gain')
    kd_arg    = DeclareLaunchArgument('kd',          default_value='0.003',  description='PID D gain')
    vbase_arg = DeclareLaunchArgument('v_base',      default_value='0.12',   description='Base forward speed [m/s]')
    vmin_arg  = DeclareLaunchArgument('v_min',       default_value='0.04',   description='Min forward speed [m/s]')

    return LaunchDescription([
        kp_arg, ki_arg, kd_arg, vbase_arg, vmin_arg,

        Node(
            package='puzzlebot_challenge',
            executable='picam_publisher',
            name='picam_publisher',
            parameters=[camera_cfg],
            output='screen',
        ),

        Node(
            package='puzzlebot_challenge',
            executable='line_detector',
            name='line_detector',
            parameters=[{'params_config': line_cfg}],
            output='screen',
        ),

        Node(
            package='puzzlebot_challenge',
            executable='line_follower',
            name='line_follower',
            parameters=[{
                'kp':     LaunchConfiguration('kp'),
                'ki':     LaunchConfiguration('ki'),
                'kd':     LaunchConfiguration('kd'),
                'v_base': LaunchConfiguration('v_base'),
                'v_min':  LaunchConfiguration('v_min'),
            }],
            output='screen',
        ),
    ])
