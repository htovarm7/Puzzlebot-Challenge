"""Full challenge launch (Jetson).

Nodes: picam_publisher, line_detector, line_follower, sign_behavior_controller.
sign_detector (YOLO) runs in a separate terminal so inference is not restarted
with this launch; it publishes /sign/command and /traffic_light.

Control priority: traffic_light (red/yellow) > sign behavior > line following.

Usage:
  ros2 launch puzzlebot_challenge final.launch.py
  ros2 launch puzzlebot_challenge final.launch.py v_base:=0.12
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share  = get_package_share_directory('puzzlebot_challenge')
    camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')
    line_cfg   = os.path.join(pkg_share, 'config', 'line_params.yaml')

    args = [
        # Line follower
        DeclareLaunchArgument('kp',             default_value='0.3',  description='P gain'),
        DeclareLaunchArgument('kd',             default_value='0.08', description='D gain'),
        DeclareLaunchArgument('ka',             default_value='0.2',  description='Angle correction weight'),
        DeclareLaunchArgument('v_base',         default_value='0.1',  description='Base speed [m/s]'),
        # Sign behaviors
        DeclareLaunchArgument('give_way_time',  default_value='2.0',  description='give_way stop [s]'),
        DeclareLaunchArgument('stop_hold_time', default_value='1.0',  description='Hold after stop sign disappears [s]'),
        DeclareLaunchArgument('workers_factor', default_value='0.5',  description='Workers speed factor'),
        DeclareLaunchArgument('approach_time',  default_value='0.4',  description='Straight run before turn [s]'),
        DeclareLaunchArgument('turn_time',      default_value='1.8',  description='Turn duration [s]'),
        DeclareLaunchArgument('turn_omega',     default_value='0.7',  description='Turn angular speed [rad/s]'),
        DeclareLaunchArgument('turn_v',         default_value='0.06', description='Turn linear speed [m/s]'),
        DeclareLaunchArgument('straight_time',  default_value='4.0',  description='go_straight override duration [s]'),
        DeclareLaunchArgument('straight_v',     default_value='0.12', description='go_straight speed [m/s]'),
        DeclareLaunchArgument('sign_cooldown',  default_value='1.0',  description='Cooldown between equal signs [s]'),
        DeclareLaunchArgument('wait_for_start', default_value='true', description='Wait for /robot/start before moving'),
    ]

    picam = Node(
        package='puzzlebot_challenge',
        executable='picam_publisher',
        name='picam_publisher',
        parameters=[camera_cfg],
        output='log',
    )

    line_detector = Node(
        package='puzzlebot_challenge',
        executable='line_detector',
        name='line_detector',
        parameters=[{'params_config': line_cfg}],
        output='screen',
    )

    line_follower = Node(
        package='puzzlebot_challenge',
        executable='line_follower',
        name='line_follower',
        parameters=[{
            'kp':     LaunchConfiguration('kp'),
            'kd':     LaunchConfiguration('kd'),
            'ka':     LaunchConfiguration('ka'),
            'v_base': LaunchConfiguration('v_base'),
        }],
        remappings=[
            ('/VelocitySetL', '/line/VelocitySetL'),
            ('/VelocitySetR', '/line/VelocitySetR'),
        ],
        output='screen',
    )

    sign_behavior = Node(
        package='puzzlebot_challenge',
        executable='sign_behavior_controller',
        name='sign_behavior_controller',
        parameters=[{
            'give_way_stop_time': LaunchConfiguration('give_way_time'),
            'stop_hold_time':     LaunchConfiguration('stop_hold_time'),
            'workers_factor':     LaunchConfiguration('workers_factor'),
            'approach_time':      LaunchConfiguration('approach_time'),
            'turn_time':          LaunchConfiguration('turn_time'),
            'turn_omega':         LaunchConfiguration('turn_omega'),
            'turn_v':             LaunchConfiguration('turn_v'),
            'straight_time':      LaunchConfiguration('straight_time'),
            'straight_v':         LaunchConfiguration('straight_v'),
            'sign_cooldown':      LaunchConfiguration('sign_cooldown'),
            'wait_for_start':     LaunchConfiguration('wait_for_start'),
        }],
        output='screen',
    )

    env_actions = [
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
        SetEnvironmentVariable('GST_DEBUG', '0'),
    ]

    return LaunchDescription(env_actions + args + [
        picam,
        line_detector,
        line_follower,
        sign_behavior,
    ])
