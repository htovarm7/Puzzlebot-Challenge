#!/usr/bin/env python3
"""Record PiCam video by reading /camera/image_raw.

Uses the same resolution sign_detector receives. When finished it asks for a
name and saves the video in the user's home directory.

Usage:
    ros2 run puzzlebot_challenge picam_publisher
    python3 scripts/record_picam_video.py
Controls: q / ESC stops recording and saves.
"""

import argparse
import os
import sys
import tempfile

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def parse_args():
    p = argparse.ArgumentParser(description="Record PiCam video via ROS2")
    p.add_argument("--topic", default="/camera/image_raw",
                   help="Image topic to record (default: /camera/image_raw)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Output file FPS (default: 30.0, same as pub_fps)")
    return p.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])


class PicamRecorder(Node):

    def __init__(self, topic, fps):
        super().__init__('picam_recorder')
        self.bridge   = CvBridge()
        self.writer   = None
        self.tmp_path = None
        self.frame    = None
        self.fps      = fps
        self.create_subscription(Image, topic, self._on_image, 10)
        self.get_logger().info(f"Waiting for frames on {topic}...")

    def _on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.tmp_path = os.path.join(tempfile.gettempdir(), 'picam_recording.mp4')
            self.writer = cv2.VideoWriter(self.tmp_path, fourcc, self.fps, (w, h))
            self.get_logger().info(f"Recording at {w}x{h} @ {self.fps} fps to {self.tmp_path}")
        self.writer.write(frame)
        self.frame = frame


def main():
    rclpy.init()
    args = parse_args()
    node = PicamRecorder(args.topic, args.fps)

    win = "Recording PiCam  |  q/ESC = stop and save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.frame is not None:
                cv2.imshow(win, node.frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if node.writer is not None:
            node.writer.release()

    tmp_path = node.tmp_path
    node.destroy_node()
    rclpy.shutdown()

    if tmp_path is None or not os.path.exists(tmp_path):
        print("No frame was recorded; nothing to save.")
        sys.exit(0)

    name = input("Name to save the video (without extension): ").strip()
    if not name:
        name = "picam_recording"
    if not name.lower().endswith('.mp4'):
        name += '.mp4'

    dest = os.path.expanduser(os.path.join('~', name))
    os.replace(tmp_path, dest)
    print(f"Video saved to: {dest}")


if __name__ == '__main__':
    main()
