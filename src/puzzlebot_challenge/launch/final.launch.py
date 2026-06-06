"""
final.launch.py
===============
Launch final para el desafío completo.

Nodos (Jetson):
  1. picam_publisher          – driver cámara CSI
  2. line_detector            – /line/shift, /line/angle, /line/detected, /line/intersection
  3. traffic_detector (HSV)   – /traffic_light  (red | yellow | green | none)
  4. sign_detector (YOLO)     – /sign/command, /sign/detected
  5. line_follower            – control PD de línea
                                 ↳ salida remapeada → /line/VelocitySetL, /line/VelocitySetR
  6. sign_behavior_controller – intercepta velocidades del line_follower
                                 y aplica comportamientos por señal:
                                   give_way       → sigue línea, para 2 s al perder la señal
                                   stop           → para mientras esté visible + hold
                                   workers_ahead  → reduce velocidad a WORKERS_FACTOR
                                   turn_left      → al perder la señal, gira izquierda
                                   turn_right     → al perder la señal, gira derecha
                                   go_straight    → sigue recto al perder la señal
                                 ↳ publica → /VelocitySetL, /VelocitySetR
  7. motor_watchdog           – para motores si no llegan comandos

Prioridad de control:
  1° /traffic_light  red/yellow → stop  (en line_follower)
  2° sign_behavior_controller   → acción por señal
  3° /line/*                    → seguimiento de línea PD

Uso:
  ros2 launch puzzlebot_challenge final.launch.py
  ros2 launch puzzlebot_challenge final.launch.py v_base:=0.12
"""

import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_libgomp() -> str:
    existing = os.environ.get('LD_PRELOAD', '')
    if 'libgomp' in existing:
        return existing
    for search_cmd in (
        'find /home /usr /opt -name "libgomp*.so*" -path "*/torch*" 2>/dev/null | head -1',
        'find /usr/lib -name "libgomp.so.1" 2>/dev/null | head -1',
    ):
        try:
            out = subprocess.check_output(search_cmd, shell=True, text=True, timeout=5).strip()
            if out:
                return out
        except Exception:
            pass
    return ''


def generate_launch_description():
    pkg_share  = get_package_share_directory('puzzlebot_challenge')
    camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')
    line_cfg   = os.path.join(pkg_share, 'config', 'line_params.yaml')
    hsv_cfg    = os.path.join(pkg_share, 'config', 'traffic_hsv.yaml')

    # ── Argumentos ──────────────────────────────────────────────────────────────
    args = [
        # Line follower
        DeclareLaunchArgument('kp',             default_value='0.3',  description='P gain'),
        DeclareLaunchArgument('kd',             default_value='0.08', description='D gain'),
        DeclareLaunchArgument('ka',             default_value='0.2',  description='Peso corrección ángulo'),
        DeclareLaunchArgument('v_base',         default_value='0.2', description='Velocidad base [m/s]'),
        DeclareLaunchArgument('crossing_time',  default_value='3.0',  description='Segundos en intersección recto [s]'),
        DeclareLaunchArgument('cooldown_time',  default_value='3.0',  description='Cooldown entre intersecciones [s]'),
        # Sign behaviors
        DeclareLaunchArgument('give_way_time',  default_value='2.0',  description='Parada give_way [s]'),
        DeclareLaunchArgument('stop_hold_time', default_value='1.0',  description='Hold tras desaparecer stop [s]'),
        DeclareLaunchArgument('workers_factor', default_value='0.5',  description='Factor velocidad workers'),
        DeclareLaunchArgument('approach_time',  default_value='0.4',  description='Tramo recto antes del giro [s]'),
        DeclareLaunchArgument('turn_time',      default_value='1.8',  description='Duración del giro [s]'),
        DeclareLaunchArgument('turn_omega',     default_value='0.7',  description='Velocidad angular giro [rad/s]'),
        DeclareLaunchArgument('turn_v',         default_value='0.06', description='Velocidad lineal durante giro [m/s]'),
        DeclareLaunchArgument('straight_time',  default_value='3.0',  description='Duración go_straight override [s]'),
        DeclareLaunchArgument('straight_v',     default_value='0.12', description='Velocidad go_straight [m/s]'),
        DeclareLaunchArgument('sign_cooldown',  default_value='4.0',  description='Cooldown entre señales iguales [s]'),
        DeclareLaunchArgument('wait_for_start', default_value='true', description='Esperar /robot/start antes de mover'),
        # YOLO
        DeclareLaunchArgument('conf_threshold', default_value='0.60', description='Umbral confianza YOLO (0-1)'),
        DeclareLaunchArgument('min_det_area',   default_value='8000', description='Área mínima bbox para detectar señal [px²]'),
        DeclareLaunchArgument('imgsz',          default_value='320',  description='Tamaño imagen inferencia YOLO'),
    ]

    # ── 1. Cámara CSI ────────────────────────────────────────────────────────────
    picam = Node(
        package='puzzlebot_challenge',
        executable='picam_publisher',
        name='picam_publisher',
        parameters=[camera_cfg],
        output='screen',
    )

    # ── 2. Detector de línea ─────────────────────────────────────────────────────
    line_detector = Node(
        package='puzzlebot_challenge',
        executable='line_detector',
        name='line_detector',
        parameters=[{'params_config': line_cfg}],
        output='screen',
    )

    # ── 3. Detector de semáforo HSV (prioridad máxima) ───────────────────────────
    traffic_detector = Node(
        package='puzzlebot_challenge',
        executable='traffic_detector',
        name='traffic_light_detector',
        parameters=[{'hsv_config': hsv_cfg}],
        output='screen',
    )

    # ── 4. Detector de señales YOLO ──────────────────────────────────────────────
    sign_detector = Node(
        package='puzzlebot_challenge',
        executable='sign_detector',
        name='sign_detector',
        parameters=[{
            'image_topic':    '/camera/image_raw',
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'imgsz':          LaunchConfiguration('imgsz'),
            'min_det_area':   LaunchConfiguration('min_det_area'),
        }],
        output='screen',
    )

    # ── 5. Seguidor de línea (salida remapeada, sign_behavior_controller toma prioridad)
    line_follower = Node(
        package='puzzlebot_challenge',
        executable='line_follower',
        name='line_follower',
        parameters=[{
            'kp':            LaunchConfiguration('kp'),
            'kd':            LaunchConfiguration('kd'),
            'ka':            LaunchConfiguration('ka'),
            'v_base':        LaunchConfiguration('v_base'),
            'crossing_time': LaunchConfiguration('crossing_time'),
            'cooldown_time': LaunchConfiguration('cooldown_time'),
        }],
        remappings=[
            ('/VelocitySetL', '/line/VelocitySetL'),
            ('/VelocitySetR', '/line/VelocitySetR'),
        ],
        output='screen',
    )

    # ── 6. Controlador de comportamiento por señales (salida final) ──────────────
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

    # ── 7. Watchdog de seguridad ─────────────────────────────────────────────────
    motor_watchdog = Node(
        package='puzzlebot_challenge',
        executable='motor_watchdog',
        name='motor_watchdog',
        output='screen',
    )

    # Precarga libgomp para evitar error TLS en Jetson + ROS2
    libgomp = _find_libgomp()
    env_actions = [SetEnvironmentVariable('PYTHONUNBUFFERED', '1')]
    if libgomp:
        env_actions.append(SetEnvironmentVariable('LD_PRELOAD', libgomp))
        env_actions.append(LogInfo(msg=f'[final.launch] LD_PRELOAD={libgomp}'))
    else:
        env_actions.append(
            LogInfo(msg='[final.launch] WARN: libgomp no encontrado — torch puede fallar'))

    return LaunchDescription(env_actions + args + [
        picam,
        line_detector,
        traffic_detector,
        sign_detector,
        line_follower,
        sign_behavior,
        motor_watchdog,
    ])
