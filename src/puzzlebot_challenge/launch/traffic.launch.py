"""
Lanza el detector de semáforo con el método elegido.

Métodos:
  hsv      — umbralización HSV pura (rápido, falla con fondos coloreados)
  circles  — HoughCircles + HSV intra-círculo (recomendado para semáforos en pantalla)

Uso
---
  ros2 launch puzzlebot_challenge traffic.launch.py                  # default: circles
  ros2 launch puzzlebot_challenge traffic.launch.py method:=hsv
  ros2 launch puzzlebot_challenge traffic.launch.py method:=circles debug:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_challenge')
    default_config = os.path.join(pkg_share, 'config', 'traffic_hsv.yaml')

    method_arg = DeclareLaunchArgument(
        'method', default_value='circles',
        choices=['hsv', 'circles'],
        description='Detector: hsv | circles'
    )
    config_arg = DeclareLaunchArgument(
        'config', default_value=default_config,
        description='YAML de parámetros'
    )
    debug_arg = DeclareLaunchArgument(
        'debug', default_value='false',
        description='Logs de debug por frame'
    )

    method = LaunchConfiguration('method')
    config = LaunchConfiguration('config')
    debug  = LaunchConfiguration('debug')

    is_hsv     = IfCondition(PythonExpression(["'", method, "' == 'hsv'"]))
    is_circles = IfCondition(PythonExpression(["'", method, "' == 'circles'"]))

    hsv_node = Node(
        package='puzzlebot_challenge', executable='traffic_detector_hsv',
        name='traffic_light_detector', output='screen',
        parameters=[config, {'debug': debug}],
        condition=is_hsv,
    )
    circles_node = Node(
        package='puzzlebot_challenge', executable='traffic_detector_circle',
        name='traffic_light_detector', output='screen',
        parameters=[config, {'debug': debug}],
        condition=is_circles,
    )

    return LaunchDescription([
        method_arg, config_arg, debug_arg,
        LogInfo(msg=['Detector de semáforo método: ', method]),
        hsv_node,
        circles_node,
    ])