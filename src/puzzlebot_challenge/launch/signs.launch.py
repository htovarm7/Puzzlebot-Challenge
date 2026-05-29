"""
signs.launch.py
===============
Stack completo para detección de señales de tránsito con acciones.

Nodos (Jetson — este launch):
  1. picam_publisher          – driver cámara CSI  (desactivar con with_camera:=false)
  2. line_detector            – /line/shift, /line/angle, /line/detected
  3. traffic_detector (HSV)   – /traffic_light  (red | yellow | green | none)
  4. line_follower            – control PD de línea
                                 ↳ salida remapeada → /line/VelocitySetL, /line/VelocitySetR
  5. sign_behavior_controller – intercepta velocidades del line_follower
                                 y aplica comportamientos por señal:
                                   give_way    → para 2 s
                                   stop        → para mientras esté visible + hold
                                   workers     → reduce velocidad a X%
                                   turn_left   → al dejar de ver la señal, gira izquierda
                                   turn_right  → al dejar de ver la señal, gira derecha
                                   go_straight → al dejar de ver la señal, avanza recto
                                 ↳ publica → /VelocitySetL, /VelocitySetR
  6. sign_api                 – HTTP server (puerto 8081) para recibir /sign/command
  7. motor_watchdog           – para motores si no llegan comandos

Laptop (terminal separada):
  ros2 launch puzzlebot_challenge signs_laptop.launch.py jetson_ip:=<IP_JETSON>

Prioridad de control:
  1° /traffic_light  red/yellow → STOP  (en line_follower)
  2° sign_behavior_controller   → acción por señal
  3° /line/*                    → seguimiento de línea PD

Uso:
  ros2 launch puzzlebot_challenge signs.launch.py
  ros2 launch puzzlebot_challenge signs.launch.py v_base:=0.10 debug:=true
"""

import glob
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_libgomp() -> str:
    """
    Devuelve la ruta al libgomp de PyTorch para precargarla con LD_PRELOAD.
    Necesario en Jetson: ROS2 ocupa el TLS block antes de que torch lo pueda usar,
    causando 'cannot allocate memory in static TLS block' al importar torch.
    NO importa torch aquí — el proceso de launch también tendría el mismo conflicto.
    Busca el archivo directamente en el sistema de archivos.
    """
    search_roots = [
        os.path.expanduser('~/.local/lib'),
        '/usr/local/lib',
        '/usr/lib',
        '/opt/conda/lib',
    ]
    for root in search_roots:
        hits = glob.glob(
            os.path.join(root, 'python*', 'site-packages', 'torch', 'torch.libs', 'libgomp*.so*')
        ) + glob.glob(
            os.path.join(root, 'python*', 'dist-packages', 'torch', 'torch.libs', 'libgomp*.so*')
        )
        if hits:
            return hits[0]
    return ''


def generate_launch_description():
    pkg_share  = get_package_share_directory('puzzlebot_challenge')
    camera_cfg = os.path.join(pkg_share, 'config', 'camera.yaml')
    line_cfg   = os.path.join(pkg_share, 'config', 'line_params.yaml')
    hsv_cfg    = os.path.join(pkg_share, 'config', 'traffic_hsv.yaml')

    # ── Argumentos ajustables ────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('kp',               default_value='1.2',  description='PD P gain'),
        DeclareLaunchArgument('kd',               default_value='0.35', description='PD D gain'),
        DeclareLaunchArgument('ka',               default_value='0.4',  description='Peso corrección ángulo'),
        DeclareLaunchArgument('v_base',           default_value='0.12', description='Velocidad base line_follower [m/s]'),
        DeclareLaunchArgument('crossing_time',    default_value='3.0',  description='Segundos cruzando intersección recto'),
        DeclareLaunchArgument('cooldown_time',    default_value='3.0',  description='Cooldown entre intersecciones [s]'),
        DeclareLaunchArgument('give_way_time',    default_value='2.0',  description='Segundos de parada en give_way'),
        DeclareLaunchArgument('stop_hold_time',   default_value='1.0',  description='Espera extra tras desaparecer el stop [s]'),
        DeclareLaunchArgument('workers_factor',   default_value='0.5',  description='Factor de velocidad con señal workers'),
        DeclareLaunchArgument('turn_time',        default_value='1.8',  description='Duración del giro [s]'),
        DeclareLaunchArgument('turn_omega',       default_value='0.7',  description='Velocidad angular del giro [rad/s]'),
        DeclareLaunchArgument('turn_v',           default_value='0.06', description='Velocidad lineal durante el giro [m/s]'),
        DeclareLaunchArgument('straight_time',    default_value='3.0',  description='Duración del override recto [s]'),
        DeclareLaunchArgument('straight_v',       default_value='0.12', description='Velocidad del override recto [m/s]'),
        DeclareLaunchArgument('sign_cooldown',    default_value='4.0',  description='Cooldown entre señales iguales [s]'),
        DeclareLaunchArgument('conf_threshold',   default_value='0.50', description='Umbral de confianza YOLO (0-1)'),
        DeclareLaunchArgument('imgsz',            default_value='320',  description='Tamaño de imagen para inferencia YOLO'),
    ]

    # ── 1. Cámara CSI ────────────────────────────────────────────────────────
    picam = Node(
        package='puzzlebot_challenge',
        executable='picam_publisher',
        name='picam_publisher',
        parameters=[camera_cfg],
        output='screen',
    )

    # ── 2. Detector de línea ─────────────────────────────────────────────────
    line_detector = Node(
        package='puzzlebot_challenge',
        executable='line_detector',
        name='line_detector',
        parameters=[{'params_config': line_cfg}],
        output='screen',
    )

    # ── 3. Detector de semáforo (HSV) — PRIORIDAD 1 ──────────────────────────
    traffic_detector = Node(
        package='puzzlebot_challenge',
        executable='traffic_detector',
        name='traffic_light_detector',
        parameters=[{'hsv_config': hsv_cfg}],
        output='screen',
    )

    # ── 4. Seguidor de línea — salida remapeada a /line/Velocity* ────────────
    #   El sign_behavior_controller lee de /line/VelocitySet{L,R} y publica
    #   la velocidad final en /VelocitySet{L,R}, evitando conflicto de tópicos.
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

    # ── 5. Controlador de comportamiento por señales — PRIORIDAD 2 ────────────
    #   Suscribe: /sign/command, /sign/detected, /line/VelocitySet{L,R}
    #   Publica:  /VelocitySet{L,R}   (salida final a los motores)
    sign_behavior = Node(
        package='puzzlebot_challenge',
        executable='sign_behavior_controller',
        name='sign_behavior_controller',
        parameters=[{
            'give_way_stop_time': LaunchConfiguration('give_way_time'),
            'stop_hold_time':     LaunchConfiguration('stop_hold_time'),
            'workers_factor':     LaunchConfiguration('workers_factor'),
            'turn_time':          LaunchConfiguration('turn_time'),
            'turn_omega':         LaunchConfiguration('turn_omega'),
            'turn_v':             LaunchConfiguration('turn_v'),
            'straight_time':      LaunchConfiguration('straight_time'),
            'straight_v':         LaunchConfiguration('straight_v'),
            'sign_cooldown':      LaunchConfiguration('sign_cooldown'),
        }],
        output='screen',
    )

    # ── 6. Detector de señales YOLO — corre en la Jetson ─────────────────────
    #   Suscribe: /camera/image_raw
    #   Publica:  /sign/command, /sign/detected, /vision/signs (debug)
    sign_detector = Node(
        package='puzzlebot_challenge',
        executable='sign_detector_offload',
        name='sign_detector',
        parameters=[{
            'image_topic':    '/camera/image_raw',
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'imgsz':          LaunchConfiguration('imgsz'),
            'jetson_api':     '',   # vacío = no hace HTTP POST (todo es local)
        }],
        output='screen',
    )

    # ── 7. Watchdog de seguridad ──────────────────────────────────────────────
    motor_watchdog = Node(
        package='puzzlebot_challenge',
        executable='motor_watchdog',
        name='motor_watchdog',
        output='screen',
    )

    # Precarga libgomp de torch para evitar el error de TLS en Jetson+ROS2
    libgomp = _find_libgomp()
    preload = ([SetEnvironmentVariable('LD_PRELOAD', libgomp)] if libgomp else [])

    return LaunchDescription(preload + args + [
        picam,
        line_detector,
        traffic_detector,
        sign_detector,
        line_follower,
        sign_behavior,
        motor_watchdog,
    ])
