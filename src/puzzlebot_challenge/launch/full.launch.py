"""Stack completo: cámara + servidor MJPEG + detector de semáforo + PID.

El detector se incluye aquí directamente (no via traffic.launch.py) para
evitar arrancar picam_publisher dos veces.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_challenge')
    hsv_cfg = os.path.join(pkg_share, 'config', 'traffic_hsv.yaml')

    task_arg = DeclareLaunchArgument(
        'task', default_value='waypoints',
        description='Tarea: square o waypoints',
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'camera.launch.py')
        )
    )
    traffic_detector = Node(
        package='puzzlebot_challenge',
        executable='traffic_detector',
        name='traffic_light_detector',
        parameters=[{'hsv_config': hsv_cfg}],
        output='screen',
    )
    pid = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'pid.launch.py')
        ),
        launch_arguments={'task': LaunchConfiguration('task')}.items(),
    )

    return LaunchDescription([task_arg, camera, traffic_detector, pid])
