#!/usr/bin/env python3
"""
puzzlebot_motion.py
===================
ROS2 Humble node for Manchester Robotics PuzzleBot.
Implements an open-loop FSM controller for:
  - Task A: Square path (0.5 m sides, 4 x 90° turns)
  - Task B: Waypoint navigation (user-defined list of (x, y) points)

Topics
------
Publishers : /VelocitySetL  (std_msgs/Float32)
             /VelocitySetR  (std_msgs/Float32)
Subscribers: /VelEncL       (std_msgs/Float32)  — for logging / future use
             /VelEncR       (std_msgs/Float32)

Robustness strategy
-------------------
* Velocity ramp-up: linear speed increases gradually from 0 → v_target
  over RAMP_TIME seconds to reduce wheel slip and inertia overshoot.
* Ramp-down: symmetric deceleration in the last RAMP_TIME seconds of each
  straight segment.
* Turning: constant low angular speed (no ramp) for predictable heading.
* All timing uses ROS clock (monotonic) to be simulation-agnostic.

Author : Armando / MCR2 Mini Challenge 1
"""

import sys
import math
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


# Robot physical parameters  (tune to match your real PuzzleBot)

WHEEL_RADIUS = 0.052  # [m]   radius of each wheel
WHEEL_BASE = 0.19  # [m]   distance between left and right wheels (track)


# Motion parameters  (open-loop, tune empirically)

V_LINEAR = 0.15  # [m/s] desired forward speed during straight segments
OMEGA_TURN = 0.5  # [rad/s] angular speed during turns (in-place)
RAMP_TIME = 0.4  # [s]   duration of velocity ramp at start / end of straight


# Task selector — overridden by CLI argument (see main())

TASK = "SQUARE"  # default; use:  python3 puzzlebot_motion.py square|waypoints


# Task B — Waypoint list (metres, relative to start pose = origin, θ=0)

WAYPOINTS = [
    (0.50, 0.50),  # p1
    (1.00, 0.00),  # p2
    (1.50, 0.50),  # p3
]


# Helper: differential-drive inverse kinematics


def unicycle_to_wheels(v: float, omega: float):
    """
    Convert (v, ω) → (v_L, v_R) in rad/s for the PuzzleBot wheel speed API.
    PuzzleBot's /VelocitySetL and /VelocitySetR accept rad/s.
    """
    v_l = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_r = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_l, v_r


def ramp_factor(elapsed: float, duration: float) -> float:
    """
    Returns a scalar in [0, 1] that ramps up for the first RAMP_TIME seconds
    and ramps down for the last RAMP_TIME seconds of a segment.
    """
    ramp = min(RAMP_TIME, duration / 2.0)  # never exceed half the segment
    if elapsed < ramp:
        return elapsed / ramp
    if elapsed > duration - ramp:
        return (duration - elapsed) / ramp
    return 1.0


# FSM States


class State:
    IDLE = "IDLE"
    MOVING_STRAIGHT = "MOVING_STRAIGHT"
    TURNING = "TURNING"
    GOAL_REACHED = "GOAL_REACHED"


# Main Node


class PuzzlebotMotion(Node):

    def __init__(self):
        super().__init__("puzzlebot_motion")

        self.pub_l = self.create_publisher(Float32, "/VelocitySetL", 10)
        self.pub_r = self.create_publisher(Float32, "/VelocitySetR", 10)

        self.create_subscription(Float32, "/VelEncL", self._enc_l_cb, 10)
        self.create_subscription(Float32, "/VelEncR", self._enc_r_cb, 10)

        self.plan = []  # list of ('straight', dist) or ('turn', angle_rad)
        self._build_plan()

        self.state = State.IDLE
        self.plan_index = 0
        self.seg_start = None  # ROS time at start of current segment
        self.seg_duration = 0.0  # how long this segment should take

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_th = 0.0  # [rad]

        self.timer = self.create_timer(0.05, self._control_loop)
        self.get_logger().info(
            f"PuzzlebotMotion ready.  Task = {TASK}  |  "
            f"{len(self.plan)} motion segments queued.  Waiting 2 s …"
        )

        # Small startup delay so the simulator / hardware is ready
        self._startup_deadline = self.get_clock().now().nanoseconds / 1e9 + 2.0

    # Plan builder

    def _build_plan(self):
        """
        Populate self.plan with a sequence of ('straight', metres) and
        ('turn', radians) primitives.
        """
        if TASK == "SQUARE":
            side = 0.55  # metres
            for _ in range(4):
                self.plan.append(("straight", side))
                self.plan.append(("turn", math.pi / 2.0))  # 90° left turn

            self.get_logger().info(
                f"[SQUARE] Plan built: {len(self.plan)} segments  "
                f"(side={side} m, v={V_LINEAR} m/s, ω={OMEGA_TURN} rad/s)"
            )

        elif TASK == "WAYPOINTS":
            current_th = 0.0  # robot starts facing +X
            current_x, current_y = 0.0, 0.0

            for idx, (wx, wy) in enumerate(WAYPOINTS):
                dx = wx - current_x
                dy = wy - current_y
                dist = math.hypot(dx, dy)
                target_th = math.atan2(dy, dx)

                # Heading error 
                delta_th = target_th - current_th
                delta_th = (delta_th + math.pi) % (2 * math.pi) - math.pi

                if abs(delta_th) > 1e-3:  # skip negligible turns
                    self.plan.append(("turn", delta_th))

                if dist > 1e-3:  # skip zero-length segments
                    self.plan.append(("straight", dist))

                self.get_logger().info(
                    f"  [WP {idx+1}] ({wx:.2f}, {wy:.2f})  "
                    f"Δθ={math.degrees(delta_th):.1f}°  dist={dist:.3f} m"
                )

                current_x, current_y = wx, wy
                current_th = target_th

            # Validate reachability (warn if any segment takes < 0.5 s)
            for kind, value in self.plan:
                if kind == "straight":
                    t = value / V_LINEAR
                elif kind == "turn":
                    t = abs(value) / OMEGA_TURN
                if t < 0.5:
                    self.get_logger().warn(
                        f"  ⚠ Segment ({kind}, {value:.3f}) → t={t:.2f} s  "
                        f"— may be unreliable on real hardware."
                    )
        else:
            self.get_logger().error(
                f"Unknown TASK '{TASK}'. Set TASK='SQUARE' or 'WAYPOINTS'."
            )

    # Encoder callbacks (open-loop: log only)

    def _enc_l_cb(self, msg: Float32):
        pass  # available for future closed-loop upgrade

    def _enc_r_cb(self, msg: Float32):
        pass

    # Motor command helper

    def _publish(self, v_l: float, v_r: float):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(v_l), float(v_r)
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)

    def _stop(self):
        self._publish(0.0, 0.0)

    # FSM — main control loop

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.state == State.IDLE:
            if now >= self._startup_deadline:
                if self.plan_index < len(self.plan):
                    self._start_segment()
                else:
                    self.state = State.GOAL_REACHED
            return

        if self.state == State.GOAL_REACHED:
            self._stop()
            self.get_logger().info(
                f"✅ GOAL REACHED.  "
                f"Estimated pose: x={self.pose_x:.3f} m  y={self.pose_y:.3f} m  "
                f"θ={math.degrees(self.pose_th):.1f}°"
            )
            self.timer.cancel()
            return

        elapsed = now - self.seg_start

        if self.state == State.MOVING_STRAIGHT:
            if elapsed >= self.seg_duration:
                # Segment finished — update estimated pose
                dist = V_LINEAR * self.seg_duration
                self.pose_x += dist * math.cos(self.pose_th)
                self.pose_y += dist * math.sin(self.pose_th)
                self._stop()
                self._advance_plan()
            else:
                # Ramp-shaped velocity profile
                factor = ramp_factor(elapsed, self.seg_duration)
                v = V_LINEAR * factor
                vl, vr = unicycle_to_wheels(v, 0.0)
                self._publish(vl, vr)

        elif self.state == State.TURNING:
            if elapsed >= self.seg_duration:
                # Segment finished — update estimated heading
                angle_rad = self._current_turn_angle
                self.pose_th = (self.pose_th + angle_rad + math.pi) % (
                    2 * math.pi
                ) - math.pi
                self._stop()
                self._advance_plan()
            else:
                # Constant angular velocity (no ramp for turns — more precise)
                omega = np.sign(self._current_turn_angle) * OMEGA_TURN
                vl, vr = unicycle_to_wheels(0.0, omega)
                self._publish(vl, vr)

    # Plan navigation helpers

    def _start_segment(self):
        kind, value = self.plan[self.plan_index]
        self.seg_start = self.get_clock().now().nanoseconds / 1e9

        if kind == "straight":
            self.seg_duration = value / V_LINEAR
            self.state = State.MOVING_STRAIGHT
            self.get_logger().info(
                f"[{self.plan_index+1}/{len(self.plan)}] STRAIGHT  "
                f"{value:.3f} m  →  {self.seg_duration:.2f} s"
            )
        elif kind == "turn":
            self.seg_duration = abs(value) / OMEGA_TURN
            self._current_turn_angle = value
            self.state = State.TURNING
            direction = "LEFT" if value > 0 else "RIGHT"
            self.get_logger().info(
                f"[{self.plan_index+1}/{len(self.plan)}] TURN {direction}  "
                f"{math.degrees(value):.1f}°  →  {self.seg_duration:.2f} s"
            )

    def _advance_plan(self):
        self.plan_index += 1
        if self.plan_index < len(self.plan):
            # Brief pause between segments (helps hardware settle)
            import time

            time.sleep(0.15)
            self._start_segment()
        else:
            self.state = State.GOAL_REACHED


# Entry point


def main(args=None):
    global TASK

    # Usage:
    #   python3 puzzlebot_motion.py square
    #   python3 puzzlebot_motion.py waypoints
    #   ros2 run <pkg> puzzlebot_motion --ros-args ...   (no positional arg → default SQUARE)
    valid = {"square": "SQUARE", "waypoints": "WAYPOINTS"}
    cli_args = [a for a in sys.argv[1:] if not a.startswith("--")]  # skip ROS flags

    if cli_args:
        choice = cli_args[0].lower()
        if choice in valid:
            TASK = valid[choice]
        else:
            print(
                f"[ERROR] Unknown task '{cli_args[0]}'.\n"
                f"  Usage: python3 puzzlebot_motion.py [square|waypoints]\n"
                f"  Falling back to default: {TASK}"
            )
    else:
        print(f"[INFO] No task argument given — using default: {TASK}")

    print(f"[INFO] Task selected: {TASK}\n")

    rclpy.init(args=args)
    node = PuzzlebotMotion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("⚠ Keyboard interrupt — stopping motors.")
        node._stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

