"""
line_follow.launch.py
=====================
Stack completo de seguimiento de línea:
  1. picam_publisher    – driver cámara CSI
  2. line_detector_v2   – visión v2: publica /line/shift, /line/angle, /line/detected, /vision/line
  3. line_follower      – PID: suscribe /line/* → publica /cmd/VelocitySet{L,R}
  4. motor_watchdog     – seguridad: /cmd/VelocitySet* → /VelocitySet{L,R}, para motores si no llegan comandos
  5. line_viewer        – ventana de debug (requiere display o ssh -X)
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

    kp_arg       = DeclareLaunchArgument('kp',            default_value='1.2',  description='PD P gain (normalised error)')
    kd_arg       = DeclareLaunchArgument('kd',            default_value='0.35', description='PD D gain (normalised error)')
    vbase_arg    = DeclareLaunchArgument('v_base',        default_value='0.12', description='Velocidad base [m/s]')
    vmin_arg     = DeclareLaunchArgument('v_min',         default_value='0.04', description='Velocidad mínima [m/s]')
    ctime_arg    = DeclareLaunchArgument('crossing_time', default_value='3.0',  description='Segundos atravesando intersección recto')
    cooldown_arg = DeclareLaunchArgument('cooldown_time', default_value='3.0',  description='Cooldown entre intersecciones [s]')

    return LaunchDescription([
        kp_arg, kd_arg, vbase_arg, vmin_arg, ctime_arg, cooldown_arg,

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
                'kp':            LaunchConfiguration('kp'),
                'kd':            LaunchConfiguration('kd'),
                'v_base':        LaunchConfiguration('v_base'),
                'v_min':         LaunchConfiguration('v_min'),
                'crossing_time': LaunchConfiguration('crossing_time'),
                'cooldown_time': LaunchConfiguration('cooldown_time'),
            }],
            output='screen',
        ),

        # Watchdog de seguridad: para motores si line_follower muere
        Node(
            package='puzzlebot_challenge',
            executable='motor_watchdog',
            name='motor_watchdog',
            output='screen',
        ),

        # Ventana de debug de visión (requiere DISPLAY — usar ssh -X)
        Node(
            package='puzzlebot_challenge',
            executable='line_viewer',
            name='line_viewer',
            output='screen',
        ),
    ])
