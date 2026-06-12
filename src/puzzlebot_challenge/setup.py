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
        (os.path.join('share', package_name, 'models'), glob('utils/*.engine') + glob('utils/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JLDominguezM',
    maintainer_email='jldm1111@gmail.com',
    description='ROS2 nodes for the PuzzleBot (CSI camera, line following, sign detection).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # camera
            'picam_publisher  = puzzlebot_challenge.camera.picam_publisher:main',
            'cam_server       = puzzlebot_challenge.camera.cam_server:main',
            # line
            'line_detector    = puzzlebot_challenge.line.line_detector:main',
            'line_calibrator  = puzzlebot_challenge.line.line_calibrator:main',
            'line_follower    = puzzlebot_challenge.line.line_follower:main',
            'line_viewer      = puzzlebot_challenge.line.line_viewer:main',
            # control
            'pid_controller   = puzzlebot_challenge.control.pid_controller:main',
            'pid_tuner        = puzzlebot_challenge.control.pid_tuner:main',
            'motor_watchdog   = puzzlebot_challenge.control.motor_watchdog:main',
            'teleop           = puzzlebot_challenge.control.teleop:main',
            # signs
            'sign_detector              = puzzlebot_challenge.signs.sign_detector:main',
            'sign_behavior_controller   = puzzlebot_challenge.signs.sign_behavior_controller:main',
            'sign_viewer                = puzzlebot_challenge.signs.sign_viewer:main',
        ],
    },
)
