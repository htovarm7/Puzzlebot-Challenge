"""All-in-one sign stack (Jetson).

Like final.launch.py but also launches sign_detector (YOLO) and motor_watchdog
in the same launch. The traffic light is detected by the YOLO model itself
(/traffic_light), so no separate HSV detector is needed.

Nodes: picam_publisher, line_detector, sign_detector, line_follower,
sign_behavior_controller, motor_watchdog.

Control priority: traffic_light (red/yellow) > sign behavior > line following.

Usage:
  ros2 launch puzzlebot_challenge signs.launch.py
  ros2 launch puzzlebot_challenge signs.launch.py v_base:=0.10
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_libgomp() -> str:
    """Path to torch's libgomp for LD_PRELOAD.

    Needed on Jetson: ROS2 takes the TLS block before torch can, causing
    'cannot allocate memory in static TLS block' when importing torch.
    """
    import subprocess
    existing = os.environ.get('LD_PRELOAD', '')
    if 'libgomp' in existing:
        return existing

    # Prefer torch's copy (avoid mixing versions), then fall back to the system
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

    args = [
        DeclareLaunchArgument('kp',             default_value='0.3',  description='P gain'),
        DeclareLaunchArgument('kd',             default_value='0.08', description='D gain'),
        DeclareLaunchArgument('ka',             default_value='0.2',  description='Angle correction weight'),
        DeclareLaunchArgument('v_base',         default_value='0.12', description='Base speed [m/s]'),
        DeclareLaunchArgument('give_way_time',  default_value='2.0',  description='give_way stop [s]'),
        DeclareLaunchArgument('stop_hold_time', default_value='1.0',  description='Hold after stop sign disappears [s]'),
        DeclareLaunchArgument('workers_factor', default_value='0.5',  description='Workers speed factor'),
        DeclareLaunchArgument('turn_time',      default_value='1.8',  description='Turn duration [s]'),
        DeclareLaunchArgument('turn_omega',     default_value='0.7',  description='Turn angular speed [rad/s]'),
        DeclareLaunchArgument('turn_v',         default_value='0.06', description='Turn linear speed [m/s]'),
        DeclareLaunchArgument('straight_time',  default_value='3.0',  description='go_straight override duration [s]'),
        DeclareLaunchArgument('straight_v',     default_value='0.12', description='go_straight override speed [m/s]'),
        DeclareLaunchArgument('sign_cooldown',  default_value='4.0',  description='Cooldown between equal signs [s]'),
        DeclareLaunchArgument('conf_threshold', default_value='0.50', description='YOLO confidence threshold (0-1)'),
        DeclareLaunchArgument('imgsz',          default_value='320',  description='YOLO inference image size'),
    ]

    picam = Node(
        package='puzzlebot_challenge',
        executable='picam_publisher',
        name='picam_publisher',
        parameters=[camera_cfg],
        output='screen',
    )

    line_detector = Node(
        package='puzzlebot_challenge',
        executable='line_detector',
        name='line_detector',
        parameters=[{'params_config': line_cfg}],
        output='screen',
    )

    # line_follower output is remapped to /line/Velocity*; sign_behavior_controller
    # reads those and publishes the final /VelocitySet{L,R}.
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
            'turn_time':          LaunchConfiguration('turn_time'),
            'turn_omega':         LaunchConfiguration('turn_omega'),
            'turn_v':             LaunchConfiguration('turn_v'),
            'straight_time':      LaunchConfiguration('straight_time'),
            'straight_v':         LaunchConfiguration('straight_v'),
            'sign_cooldown':      LaunchConfiguration('sign_cooldown'),
        }],
        output='screen',
    )

    sign_detector = Node(
        package='puzzlebot_challenge',
        executable='sign_detector',
        name='sign_detector',
        parameters=[{
            'image_topic':    '/camera/image_raw',
            'conf_threshold': LaunchConfiguration('conf_threshold'),
            'imgsz':          LaunchConfiguration('imgsz'),
        }],
        output='screen',
    )

    motor_watchdog = Node(
        package='puzzlebot_challenge',
        executable='motor_watchdog',
        name='motor_watchdog',
        output='screen',
    )

    # Preload torch's libgomp to avoid the TLS error on Jetson + ROS2
    libgomp = _find_libgomp()
    env_actions = [SetEnvironmentVariable('PYTHONUNBUFFERED', '1')]
    if libgomp:
        env_actions.append(SetEnvironmentVariable('LD_PRELOAD', libgomp))
        env_actions.append(LogInfo(msg=f'[signs.launch] LD_PRELOAD={libgomp}'))
    else:
        env_actions.append(LogInfo(msg='[signs.launch] WARN: libgomp not found — torch may fail under ROS2'))

    return LaunchDescription(env_actions + args + [
        picam,
        line_detector,
        sign_detector,
        line_follower,
        sign_behavior,
        motor_watchdog,
    ])
