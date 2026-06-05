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

    return LaunchDescription([
        DeclareLaunchArgument('kp',     default_value='0.3',  description='P gain'),
        DeclareLaunchArgument('kd',     default_value='0.08', description='D gain'),
        DeclareLaunchArgument('ka',     default_value='0.2',  description='Angle correction weight'),
        DeclareLaunchArgument('v_base', default_value='0.2',  description='Velocidad base [m/s]'),
        DeclareLaunchArgument('v_min',  default_value='0.04', description='Velocidad mínima [m/s]'),

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
                'kd':     LaunchConfiguration('kd'),
                'ka':     LaunchConfiguration('ka'),
                'v_base': LaunchConfiguration('v_base'),
                'v_min':  LaunchConfiguration('v_min'),
            }],
            output='screen',
        ),
    ])
