#!/usr/bin/env python3
"""Closed-loop PD waypoint follower using wheel-encoder odometry.

Two cascaded PD loops driven by encoder-based odometry: a TURN phase that
rotates in place toward the waypoint, then a STRAIGHT phase that drives to it
while correcting heading. A segment completes when the error stays within
tolerance for SETTLE_TICKS consecutive ticks.

Topics:
  Pub: /VelocitySetL, /VelocitySetR  (Float32, rad/s)
  Sub: /VelocityEncL, /VelocityEncR  (Float32, rad/s measured, sensor-data QoS)

The encoder topics use BEST_EFFORT reliability, so the subscribers use the
sensor-data profile to match.
"""

import sys
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, String


# Robot physical parameters (must match your real PuzzleBot)
WHEEL_RADIUS = 0.05154     # [m]  calibrated via linear test (tape vs odom)
WHEEL_BASE   = 0.19        # [m]  track width (left-right wheel distance)

# Chassis orientation: -1 if the robot's physical front is the controller's
# back. +1 = normal, -1 = flipped. Only forward/back is affected.
FORWARD_SIGN = -1

# PD gains (tune with pid_tuner.py)
KP_DIST = 0.9              # distance loop (linear velocity)
KD_DIST = 0.15
KP_TH   = 2.2             # heading loop (angular velocity)
KD_TH   = 0.25

# Saturation limits
V_MAX     = 0.20           # [m/s]
OMEGA_MAX = 1.2            # [rad/s]
HEADING_GATE = math.radians(20.0)   # if |err_th| > this, slow v down

# Convergence criteria
DIST_TOL      = 0.03               # [m]   stop straight segment within 3 cm
ANG_TOL       = math.radians(2.0)  # [rad] stop turn within 2 deg
SETTLE_TICKS  = 5                  # consecutive in-tolerance ticks needed

CTRL_DT = 0.05             # [s]  20 Hz control loop

# Traffic-light reaction: gain applied to v and omega per state.
# Red is a full stop, yellow slows down, green/none run at full speed.
TRAFFIC_GAIN = {
    "green":  1.0,
    "none":   1.0,
    "yellow": 0.4,
    "red":    0.0,
}

# Task selection + waypoints
TASK = "WAYPOINTS"

WAYPOINTS = [
    (1.00,  0.00),
    (1.00, -0.90),
    (1.80, -0.90),
    (1.80, 0.00),
    (1.00, 0.00)
]

# Square task (for sanity-checking)
SQUARE_SIDE = 0.55


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def unicycle_to_wheels(v: float, omega: float):
    """Convert (v, omega) to (wL, wR) wheel speeds for the PuzzleBot API."""
    v_l = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_r = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_l, v_r


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# FSM states
class State:
    IDLE            = "IDLE"
    TURN_TO_HEADING = "TURN_TO_HEADING"
    DRIVE_TO_POINT  = "DRIVE_TO_POINT"
    GOAL_REACHED    = "GOAL_REACHED"


class PuzzlebotMotionPD(Node):

    def __init__(self):
        super().__init__("puzzlebot_motion_pd")

        self.pub_l = self.create_publisher(Float32, "/VelocitySetL", 10)
        self.pub_r = self.create_publisher(Float32, "/VelocitySetR", 10)
        # Encoders publish BEST_EFFORT, so subscribers must match
        self.create_subscription(
            Float32, "/VelocityEncL", self._enc_l_cb, qos_profile_sensor_data)
        self.create_subscription(
            Float32, "/VelocityEncR", self._enc_r_cb, qos_profile_sensor_data)

        self._traffic_state = "none"
        self.create_subscription(
            String, "/traffic_light", self._on_traffic, 10)

        # Wheel speed measurements (rad/s)
        self.wL = 0.0
        self.wR = 0.0

        # Odometry pose
        self.pose_x  = 0.0
        self.pose_y  = 0.0
        self.pose_th = 0.0

        self.targets = []
        self._build_targets()
        self.tgt_index = 0

        # PD state
        self.prev_err_dist = 0.0
        self.prev_err_th   = 0.0
        self.in_tol_count  = 0

        self.state = State.IDLE
        self._startup_deadline = self.get_clock().now().nanoseconds / 1e9 + 2.0

        self.last_time = self.get_clock().now().nanoseconds / 1e9
        self.timer = self.create_timer(CTRL_DT, self._control_loop)

        self.get_logger().info(
            f"PuzzlebotMotionPD ready.  Task={TASK}  targets={len(self.targets)}  "
            f"Kp_d={KP_DIST} Kd_d={KD_DIST}  Kp_th={KP_TH} Kd_th={KD_TH}"
        )

    def _build_targets(self):
        if TASK == "WAYPOINTS":
            for (x, y) in WAYPOINTS:
                self.targets.append((x, y))
        elif TASK == "SQUARE":
            # 4 corners of a square, returning near origin
            s = SQUARE_SIDE
            self.targets = [(s, 0.0), (s, s), (0.0, s), (0.0, 0.0)]
        else:
            self.get_logger().error(f"Unknown TASK '{TASK}'")

        for i, (x, y) in enumerate(self.targets):
            self.get_logger().info(f"  target[{i+1}] = ({x:+.3f}, {y:+.3f})")

    # Encoder callbacks (measured wheel angular velocities)
    def _enc_l_cb(self, msg: Float32):
        self.wL = float(msg.data)

    def _enc_r_cb(self, msg: Float32):
        self.wR = float(msg.data)

    def _on_traffic(self, msg: String):
        new_state = (msg.data or "none").strip().lower()
        if new_state not in TRAFFIC_GAIN:
            new_state = "none"
        if new_state != self._traffic_state:
            self.get_logger().info(
                f"[traffic] {self._traffic_state.upper()} to {new_state.upper()}"
            )
            self._traffic_state = new_state

    # Odometry update (called each control tick)
    def _update_odom(self, dt: float):
        v_meas = FORWARD_SIGN * WHEEL_RADIUS * (self.wR + self.wL) / 2.0
        w_meas = WHEEL_RADIUS * (self.wR - self.wL) / WHEEL_BASE

        # Midpoint integration (slightly more accurate than Euler)
        th_mid = self.pose_th + 0.5 * w_meas * dt
        self.pose_x  += v_meas * math.cos(th_mid) * dt
        self.pose_y  += v_meas * math.sin(th_mid) * dt
        self.pose_th  = wrap_angle(self.pose_th + w_meas * dt)

    # Motor publish helpers
    def _publish_wheels(self, wL_cmd: float, wR_cmd: float):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wL_cmd), float(wR_cmd)
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)

    def _cmd_unicycle(self, v: float, omega: float):
        v = clamp(v, -V_MAX, V_MAX)
        omega = clamp(omega, -OMEGA_MAX, OMEGA_MAX)
        # Traffic light gating: scale both v and omega so a red light
        # is a real stop and yellow is a smooth slowdown without
        # changing the controller's intent.
        gain = TRAFFIC_GAIN.get(self._traffic_state, 1.0)
        v *= gain
        omega *= gain
        wL, wR = unicycle_to_wheels(FORWARD_SIGN * v, omega)
        self._publish_wheels(wL, wR)

    def _stop(self):
        self._publish_wheels(0.0, 0.0)

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = max(1e-3, now - self.last_time)
        self.last_time = now

        self._update_odom(dt)

        # IDLE: startup pause, then begin first target
        if self.state == State.IDLE:
            self._stop()
            if now >= self._startup_deadline:
                if self.tgt_index < len(self.targets):
                    self._begin_target()
                else:
                    self.state = State.GOAL_REACHED
            return

        # GOAL_REACHED: latch motors off
        if self.state == State.GOAL_REACHED:
            self._stop()
            self.get_logger().info(
                f"GOAL REACHED.  pose=({self.pose_x:+.3f}, {self.pose_y:+.3f}, "
                f"{math.degrees(self.pose_th):+.1f} deg)"
            )
            self.timer.cancel()
            return

        # Current target
        tx, ty = self.targets[self.tgt_index]
        dx = tx - self.pose_x
        dy = ty - self.pose_y
        dist_to_goal = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        err_th = wrap_angle(angle_to_goal - self.pose_th)

        # TURN_TO_HEADING: rotate in place toward waypoint
        if self.state == State.TURN_TO_HEADING:
            d_err_th = (err_th - self.prev_err_th) / dt
            self.prev_err_th = err_th

            omega = KP_TH * err_th + KD_TH * d_err_th
            self._cmd_unicycle(0.0, omega)

            if abs(err_th) < ANG_TOL:
                self.in_tol_count += 1
            else:
                self.in_tol_count = 0

            if self.in_tol_count >= SETTLE_TICKS:
                self._stop()
                self.get_logger().info(
                    f"  [turn done] err={math.degrees(err_th):+.2f} deg, "
                    f"switch to DRIVE"
                )
                self.state = State.DRIVE_TO_POINT
                self.in_tol_count = 0
                self.prev_err_dist = dist_to_goal
                self.prev_err_th = err_th
            return

        # DRIVE_TO_POINT: forward + heading correction
        if self.state == State.DRIVE_TO_POINT:
            # Distance error: signed projection along current heading.
            # This is positive while goal is in front, becomes ~0 at goal,
            # and prevents overshoot from "fighting" past the target.
            err_d = dx * math.cos(self.pose_th) + dy * math.sin(self.pose_th)

            d_err_d  = (err_d  - self.prev_err_dist) / dt
            d_err_th = (err_th - self.prev_err_th)   / dt
            self.prev_err_dist = err_d
            self.prev_err_th   = err_th

            v     = KP_DIST * err_d  + KD_DIST * d_err_d
            omega = KP_TH   * err_th + KD_TH   * d_err_th

            # If heading error is large, attenuate forward speed so the
            # robot doesn't drive in the wrong direction.
            if abs(err_th) > HEADING_GATE:
                v *= max(0.0, 1.0 - abs(err_th) / math.pi)

            # Never reverse during a forward waypoint approach
            v = max(0.0, v)

            self._cmd_unicycle(v, omega)

            # Convergence: close to goal in Euclidean distance
            if dist_to_goal < DIST_TOL:
                self.in_tol_count += 1
            else:
                self.in_tol_count = 0

            if self.in_tol_count >= SETTLE_TICKS:
                self._stop()
                self.get_logger().info(
                    f"  [waypoint {self.tgt_index+1}/{len(self.targets)} reached] "
                    f"pose=({self.pose_x:+.3f}, {self.pose_y:+.3f})  "
                    f"residual={dist_to_goal*100:.1f} cm"
                )
                self.tgt_index += 1
                self.in_tol_count = 0
                if self.tgt_index < len(self.targets):
                    self._begin_target()
                else:
                    self.state = State.GOAL_REACHED
            return

    # Target transition: precompute heading and switch to TURN
    def _begin_target(self):
        tx, ty = self.targets[self.tgt_index]
        dx = tx - self.pose_x
        dy = ty - self.pose_y
        dist = math.hypot(dx, dy)
        target_th = math.atan2(dy, dx)
        err_th = wrap_angle(target_th - self.pose_th)

        self.prev_err_dist = dist
        self.prev_err_th   = err_th
        self.in_tol_count  = 0

        # Skip turn if heading already aligned
        if abs(err_th) < ANG_TOL:
            self.state = State.DRIVE_TO_POINT
            self.get_logger().info(
                f"[target {self.tgt_index+1}/{len(self.targets)}] "
                f"({tx:+.2f},{ty:+.2f})  heading OK, DRIVE directly  "
                f"d={dist:.3f} m"
            )
        else:
            self.state = State.TURN_TO_HEADING
            self.get_logger().info(
                f"[target {self.tgt_index+1}/{len(self.targets)}] "
                f"({tx:+.2f},{ty:+.2f})  TURN {math.degrees(err_th):+.1f} deg "
                f"then DRIVE {dist:.3f} m"
            )


def main(args=None):
    global TASK
    valid = {"square": "SQUARE", "waypoints": "WAYPOINTS"}
    cli_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if cli_args:
        choice = cli_args[0].lower()
        if choice in valid:
            TASK = valid[choice]
        else:
            print(f"[ERROR] Unknown task '{cli_args[0]}', defaulting to {TASK}")
    print(f"[INFO] Task selected: {TASK}\n")

    rclpy.init(args=args)
    node = PuzzlebotMotionPD()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Keyboard interrupt, stopping motors.")
        node._stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()