from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'puzzlebot_challenge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),  glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('utils/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JLDominguezM',
    maintainer_email='jldm1111@gmail.com',
    description='Nodos ROS2 para el PuzzleBot (cámara CSI, PID, servidor MJPEG).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'picam_publisher  = puzzlebot_challenge.picam_publisher:main',
            'cam_server       = puzzlebot_challenge.cam_server:main',
            'pid_controller   = puzzlebot_challenge.pid_controller:main',
            'pid_tuner        = puzzlebot_challenge.pid_tuner:main',
            'hsv_calibrator   = puzzlebot_challenge.hsv_calibrator:main',
            'traffic_detector = puzzlebot_challenge.traffic_controller_hsv:main',
            'line_detector    = puzzlebot_challenge.line_detector:main',
            'line_calibrator  = puzzlebot_challenge.line_calibrator:main',
            'line_follower    = puzzlebot_challenge.line_follower:main',
            'teleop           = puzzlebot_challenge.teleop:main',
            'motor_watchdog   = puzzlebot_challenge.motor_watchdog:main',
            'line_viewer      = puzzlebot_challenge.line_viewer:main',
            'sign_detector_offload      = puzzlebot_challenge.sign_detector_offload:main',
            'sign_api                   = puzzlebot_challenge.sign_api:main',
            'sign_behavior_controller   = puzzlebot_challenge.sign_behavior_controller:main',
        ],
    },
)
