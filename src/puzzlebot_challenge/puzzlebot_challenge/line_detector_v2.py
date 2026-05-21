#!/usr/bin/env python3
"""
line_detector_v2.py  –  ROS2 node: dark-tape detection on a light floor.

Subscribes : /camera/image_raw   (sensor_msgs/Image)
Publishes  : /line/shift         (std_msgs/Float32)  px offset from center (+right)
             /line/angle         (std_msgs/Float32)  always 90.0 (unused by follower)
             /line/detected      (std_msgs/Bool)
             /vision/line        (sensor_msgs/Image) annotated debug frame
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge


class CenterLineDetector:
    def __init__(self, cam_w: int = 320, cam_h: int = 240):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.last_center = (cam_w // 2, int(cam_h * 0.85))
        self.frames_since_detect = 999  # large → jump filter loose on startup

        self.roi_top_frac  = 0.60    # start ROI at 60 % of frame height
        self.max_jump_px   = 120     # max allowed jump when actively tracking
        self.min_area      = 100     # minimum contour area in pixels
        self.min_h_box     = 5       # minimum bounding-box height in pixels
        self.otsu_thresh   = 0       # last computed Otsu T (for HUD)
        self.debug_mask    = None
        self.last_valid    = []      # all valid contour centroids from last frame

    def detect(self, image: np.ndarray):
        """
        Returns (cx, cy, detected).
        cx/cy are in full-frame coordinates.

        Strategy: the track has three parallel lines (left-border, center, right-border).
        Sort all valid contours by their X centroid and pick the MEDIAN one — it will
        always be the center line regardless of how many are visible.
        """
        h, w = image.shape[:2]
        y_start = int(h * self.roi_top_frac)

        roi  = image[y_start:h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 2.0)

        # OTSU + BINARY_INV: dark tape → white in mask
        self.otsu_thresh, mask = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Smaller close kernel so adjacent lines are NOT merged into one blob
        k_open  = np.ones((3, 3), np.uint8)
        k_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        self.debug_mask = mask

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            _, _, _, bh = cv2.boundingRect(c)
            if bh < self.min_h_box:
                continue
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"]) + y_start
            valid.append((cx, cy, area))

        if not valid:
            self.frames_since_detect += 1
            return self.last_center[0], self.last_center[1], False

        # Sort all candidates by X position (left → right)
        valid.sort(key=lambda t: t[0])
        self.last_valid = valid  # expose for debug rendering

        if len(valid) >= 3:
            # 3 lines visible: left-border, CENTER, right-border → take median
            mid = len(valid) // 2
            best = (valid[mid][0], valid[mid][1])
        else:
            # Fewer lines visible: pick the one closest to last known center
            best_item = min(valid, key=lambda t:
                (t[0] - self.last_center[0]) ** 2 + (t[1] - self.last_center[1]) ** 2)
            best = (best_item[0], best_item[1])

        # Jump filter: if actively tracking and jump is too large, fall back to
        # the candidate closest to last center (avoids latching onto a border line)
        if self.frames_since_detect < 5:
            dist_jump = ((best[0] - self.last_center[0]) ** 2 +
                         (best[1] - self.last_center[1]) ** 2) ** 0.5
            if dist_jump > self.max_jump_px:
                best_item = min(valid, key=lambda t:
                    (t[0] - self.last_center[0]) ** 2 + (t[1] - self.last_center[1]) ** 2)
                best = (best_item[0], best_item[1])

        self.last_center = best
        self.frames_since_detect = 0
        return best[0], best[1], True


class LineDetectorV2Node(Node):

    def __init__(self):
        super().__init__("line_detector_v2")

        self.declare_parameter("image_topic", "/camera/image_raw")
        image_topic = self.get_parameter("image_topic").value

        self._bridge   = CvBridge()
        self._detector = None  # lazy-init on first frame (learns resolution)

        self.sub_img      = self.create_subscription(
            Image, image_topic, self._on_image, 10)
        self.pub_shift    = self.create_publisher(Float32, "/line/shift",    10)
        self.pub_angle    = self.create_publisher(Float32, "/line/angle",    10)
        self.pub_detected = self.create_publisher(Bool,    "/line/detected", 10)
        self.pub_debug    = self.create_publisher(Image,   "/vision/line",   10)

        self.get_logger().info(
            f"LineDetectorV2Node ready | topic={image_topic}")

    # ------------------------------------------------------------------
    def _on_image(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return

        h, w = frame.shape[:2]
        if self._detector is None:
            self._detector = CenterLineDetector(cam_w=w, cam_h=h)
            self.get_logger().info(f"Detector initialized: {w}x{h}")

        cx, cy, detected = self._detector.detect(frame)
        shift = float(cx - w / 2.0)

        s_msg = Float32(); s_msg.data = shift
        a_msg = Float32(); a_msg.data = 90.0
        d_msg = Bool();    d_msg.data = detected

        self.pub_shift.publish(s_msg)
        self.pub_angle.publish(a_msg)
        self.pub_detected.publish(d_msg)

        self._publish_debug(frame, cx, cy, detected, shift)

    # ------------------------------------------------------------------
    def _publish_debug(self, frame: np.ndarray,
                       cx: int, cy: int,
                       detected: bool, shift: float):
        if self.pub_debug.get_subscription_count() == 0:
            return

        vis = frame.copy()
        h, w = vis.shape[:2]
        y_start = int(h * self._detector.roi_top_frac)
        T = int(self._detector.otsu_thresh)

        # ROI boundary (yellow) and vertical center (cyan)
        cv2.line(vis, (0, y_start), (w, y_start), (0, 200, 255), 1)
        cv2.line(vis, (w // 2, y_start), (w // 2, h), (255, 255, 0), 1)

        if detected:
            # Draw ALL candidate centroids as small gray circles for visibility
            for i, (vx, vy, _) in enumerate(self._detector.last_valid):
                cv2.circle(vis, (vx, vy), 5, (160, 160, 160), -1)
                cv2.putText(vis, str(i), (vx + 6, vy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
            # Selected (center) contour
            cv2.line(vis, (w // 2, cy), (cx, cy), (0, 0, 255), 2)
            cv2.circle(vis, (cx, cy), 8, (0, 255, 0), -1)
            n = len(self._detector.last_valid)
            hud   = f"T={T}  shift={shift:+.0f}  n={n}"
            color = (255, 255, 255)
        else:
            hud   = f"T={T}  no contour"
            color = (0, 165, 255)

        cv2.putText(vis, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_debug.publish(out_msg)


# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = LineDetectorV2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
