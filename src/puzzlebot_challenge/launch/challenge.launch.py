"""
challenge.launch.py
===================
Stack completo para competencia: línea + semáforo + señalamientos.

Nodos (Jetson):
  1. picam_publisher    – driver cámara CSI
  2. line_detector      – /line/shift, /line/angle, /line/detected, /line/intersection
  3. traffic_detector   – /traffic_light  (red | yellow | green | none)
  4. line_follower      – control con prioridades: semáforo → señalamientos → línea
  5. sign_api           – HTTP server para recibir /sign/command desde la laptop
  6. motor_watchdog     – para motores si line_follower muere
  7. line_viewer        – debug visual (requiere DISPLAY / ssh -X)

Nodo (Laptop — correr por separado):
  sign_detector_offload – YOLO en RTX 4060, publica /sign/command vía HTTP al Jetson

Prioridad en control:
  1° /traffic_light  red/yellow → STOP
  2° /sign/command   al detectar intersección → acción (giro, pare, recto)
  3° /line/*         seguimiento de línea PD

Para arrancar en el Jetson:
  ros2 launch puzzlebot_challenge challenge.launch.py

Para arrancar en la laptop (en otra terminal):
  ros2 run puzzlebot_challenge sign_detector_offload
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share  = get_package_share_directory('puzzlebot_challenge')
    camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')
    line_cfg   = os.path.join(pkg_share, 'config', 'line_params.yaml')
    hsv_cfg    = os.path.join(pkg_share, 'config', 'traffic_hsv.yaml')

    # ── Argumentos tuneables ─────────────────────────────────────────────
    kp_arg       = DeclareLaunchArgument('kp',            default_value='1.2',  description='PD P gain')
    kd_arg       = DeclareLaunchArgument('kd',            default_value='0.35', description='PD D gain')
    ka_arg       = DeclareLaunchArgument('ka',            default_value='0.4',  description='Peso corrección ángulo')
    vbase_arg    = DeclareLaunchArgument('v_base',        default_value='0.12', description='Velocidad base [m/s]')
    vmin_arg     = DeclareLaunchArgument('v_min',         default_value='0.04', description='Velocidad mínima [m/s]')
    ctime_arg    = DeclareLaunchArgument('crossing_time', default_value='3.0',  description='Segundos cruzando intersección recto')
    cooldown_arg = DeclareLaunchArgument('cooldown_time', default_value='3.0',  description='Cooldown entre intersecciones [s]')

    return LaunchDescription([
        kp_arg, kd_arg, ka_arg, vbase_arg, vmin_arg, ctime_arg, cooldown_arg,

        # ── Cámara ──────────────────────────────────────────────────────
        Node(
            package='puzzlebot_challenge',
            executable='picam_publisher',
            name='picam_publisher',
            parameters=[camera_cfg],
            output='screen',
        ),

        # ── Visión: línea ────────────────────────────────────────────────
        Node(
            package='puzzlebot_challenge',
            executable='line_detector',
            name='line_detector',
            parameters=[{'params_config': line_cfg}],
            output='screen',
        ),

        # ── Visión: semáforo — PRIORIDAD 1 ───────────────────────────────
        # Publica /traffic_light → red | yellow | green | none
        Node(
            package='puzzlebot_challenge',
            executable='traffic_detector',
            name='traffic_light_detector',
            parameters=[{'hsv_config': hsv_cfg}],
            output='screen',
        ),

        # ── Control: seguidor con prioridades integradas ─────────────────
        # 1° semáforo  2° /sign/command en intersección  3° línea PD
        Node(
            package='puzzlebot_challenge',
            executable='line_follower',
            name='line_follower',
            parameters=[{
                'kp':            LaunchConfiguration('kp'),
                'kd':            LaunchConfiguration('kd'),
                'ka':            LaunchConfiguration('ka'),
                'v_base':        LaunchConfiguration('v_base'),
                'v_min':         LaunchConfiguration('v_min'),
                'crossing_time': LaunchConfiguration('crossing_time'),
                'cooldown_time': LaunchConfiguration('cooldown_time'),
            }],
            output='screen',
        ),

        # ── API HTTP: bridge para /sign/command desde la laptop ──────────
        # La laptop corre sign_detector_offload (YOLO) y hace POST aquí
        # POST http://<JETSON_IP>:8081/sign  {"command": "turn_left"}
        Node(
            package='puzzlebot_challenge',
            executable='sign_api',
            name='sign_api',
            output='screen',
        ),

        # ── Seguridad ────────────────────────────────────────────────────
        Node(
            package='puzzlebot_challenge',
            executable='motor_watchdog',
            name='motor_watchdog',
            output='screen',
        ),

        # ── Debug visual (requiere DISPLAY / ssh -X) ─────────────────────
        Node(
            package='puzzlebot_challenge',
            executable='line_viewer',
            name='line_viewer',
            output='screen',
        ),
    ])
