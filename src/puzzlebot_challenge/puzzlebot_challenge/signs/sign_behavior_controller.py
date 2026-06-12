#!/usr/bin/env python3
"""Sign behavior controller.

Intercepts line_follower wheel speeds and applies behaviors based on the
traffic signs detected by YOLO. Turns are timer-triggered.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String, Empty

# Robot geometry
WHEEL_RADIUS = 0.05154
WHEEL_BASE   = 0.19
FORWARD_SIGN = -1   # positive wheel speed = forward

# Default parameters
GIVE_WAY_STOP_TIME = 2.0   # give_way stop duration [s]
STOP_HOLD_TIME     = 1.0   # extra wait after stop sign disappears [s]
WORKERS_FACTOR     = 0.5   # speed factor under workers sign
APPROACH_TIME      = 0.4   # straight run before turn [s]
TURN_TIME          = 4.0   # turn duration [s]
TURN_OMEGA         = 0.7   # turn angular speed [rad/s]
TURN_V             = 0.06  # linear speed during turn [m/s]
STRAIGHT_TIME      = 3.0   # go_straight override duration [s]
STRAIGHT_V         = 0.12  # go_straight override speed [m/s]
SIGN_COOLDOWN      = 4.0   # cooldown before re-triggering the same command [s]
ARM_DELAY          = 2.0   # wait after detection before running the maneuver [s]
CTRL_DT            = 0.05  # control loop period [s]

# State ids
S_IDLE              = "IDLE"
S_PENDING_GIVE_WAY  = "PENDING_GIVE_WAY"
S_GIVE_WAY          = "GIVE_WAY"
S_STOP              = "STOP"
S_STOP_HOLD         = "STOP_HOLD"
S_WORKERS           = "WORKERS"
S_PENDING_LEFT      = "PENDING_LEFT"
S_PENDING_RIGHT     = "PENDING_RIGHT"
S_PENDING_STRAIGHT  = "PENDING_STRAIGHT"
S_APPROACH_LEFT     = "APPROACH_LEFT"
S_APPROACH_RIGHT    = "APPROACH_RIGHT"
S_TURNING_LEFT      = "TURNING_LEFT"
S_TURNING_RIGHT     = "TURNING_RIGHT"
S_GOING_STRAIGHT    = "GOING_STRAIGHT"


def unicycle_to_wheels(v: float, omega: float):
    vl = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    vr = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return vl, vr


class SignBehaviorController(Node):

    def __init__(self):
        super().__init__("sign_behavior_controller")

        self.declare_parameter("give_way_stop_time", GIVE_WAY_STOP_TIME)
        self.declare_parameter("stop_hold_time",     STOP_HOLD_TIME)
        self.declare_parameter("workers_factor",     WORKERS_FACTOR)
        self.declare_parameter("approach_time",      APPROACH_TIME)
        self.declare_parameter("turn_time",          TURN_TIME)
        self.declare_parameter("turn_omega",         TURN_OMEGA)
        self.declare_parameter("turn_v",             TURN_V)
        self.declare_parameter("straight_time",      STRAIGHT_TIME)
        self.declare_parameter("straight_v",         STRAIGHT_V)
        self.declare_parameter("sign_cooldown",      SIGN_COOLDOWN)
        self.declare_parameter("arm_delay",          ARM_DELAY)
        self.declare_parameter("wait_for_start",     True)

        self.create_subscription(String,  "/sign/command",      self._cb_command,       10)
        self.create_subscription(Bool,    "/sign/detected",     self._cb_detected,      10)
        self.create_subscription(String,  "/traffic_light",     self._cb_traffic,       10)
        self.create_subscription(Bool,    "/line/detected",     self._cb_line_detected, 10)
        self.create_subscription(Float32, "/line/VelocitySetL", self._cb_vel_l,         10)
        self.create_subscription(Float32, "/line/VelocitySetR", self._cb_vel_r,         10)
        self.create_subscription(Empty,   "/robot/start",       self._cb_start,         10)

        self._pub_l = self.create_publisher(Float32, "/VelocitySetL", 10)
        self._pub_r = self.create_publisher(Float32, "/VelocitySetR", 10)

        self._state         = S_IDLE
        self._state_start   = self._now()
        self._sign_command  = "none"
        self._sign_detected = False
        self._traffic_state = "none"
        self._prev_detected = False
        self._line_detected = False
        self._line_vel_l    = 0.0
        self._line_vel_r    = 0.0
        self._last_trigger  = {}     # cmd to last trigger timestamp

        _wait = self.get_parameter("wait_for_start").value
        _wait_bool = _wait if isinstance(_wait, bool) else str(_wait).lower() not in ("false", "0", "no")
        self._sign_ready = not _wait_bool

        self.create_timer(CTRL_DT, self._control_loop)
        self.get_logger().info("SignBehaviorController ready")

    # Callbacks

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb_start(self, _msg: Empty):
        if not self._sign_ready:
            self._sign_ready = True
            self.get_logger().info("[SignBehavior] /robot/start received — robot running")

    def _cb_command(self, msg: String):
        self._sign_command = msg.data.lower()

    def _cb_detected(self, msg: Bool):
        self._sign_detected = bool(msg.data)

    def _cb_traffic(self, msg: String):
        self._traffic_state = msg.data.lower()

    def _cb_line_detected(self, msg: Bool):
        self._line_detected = bool(msg.data)

    def _cb_vel_l(self, msg: Float32):
        self._line_vel_l = float(msg.data)

    def _cb_vel_r(self, msg: Float32):
        self._line_vel_r = float(msg.data)

    # Publish helpers

    def _publish(self, wl: float, wr: float):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wl), float(wr)
        self._pub_l.publish(ml)
        self._pub_r.publish(mr)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _passthrough(self):
        """Forward line_follower speeds unchanged."""
        self._publish(self._line_vel_l, self._line_vel_r)

    # State transition

    _STATE_MSG = {
        S_PENDING_GIVE_WAY: "GIVE WAY detected — approaching",
        S_GIVE_WAY:         "GIVE WAY — stopping",
        S_STOP:             "STOP — stopped",
        S_STOP_HOLD:        "STOP — hold",
        S_WORKERS:          "WORKERS — reduced speed",
        S_PENDING_LEFT:     "TURN LEFT detected — waiting arm_delay s before turning",
        S_PENDING_RIGHT:    "TURN RIGHT detected — waiting arm_delay s before turning",
        S_PENDING_STRAIGHT: "GO STRAIGHT detected — waiting arm_delay s before straight",
        S_APPROACH_LEFT:    "TURN LEFT — straight approach",
        S_APPROACH_RIGHT:   "TURN RIGHT — straight approach",
        S_TURNING_LEFT:     "TURN LEFT — turning",
        S_TURNING_RIGHT:    "TURN RIGHT — turning",
        S_GOING_STRAIGHT:   "GO STRAIGHT — driving straight",
        S_IDLE:             "IDLE — following line",
    }

    def _enter(self, state: str, cmd: str = None):
        self._state       = state
        self._state_start = self._now()
        if cmd is not None:
            self._last_trigger[cmd] = self._now()
        msg = self._STATE_MSG.get(state, state)
        self.get_logger().info(f"[SignBehavior] {msg}")

    def _in_cooldown(self, cmd: str) -> bool:
        cooldown = self.get_parameter("sign_cooldown").value
        last = self._last_trigger.get(cmd, -(cooldown * 2))
        return (self._now() - last) < cooldown

    # Main control loop (20 Hz)

    def _control_loop(self):
        if not self._sign_ready:
            self._stop()
            return

        # Top priority: red light stops the robot
        if self._traffic_state == "red":
            self._stop()
            return

        # Yellow light reduces speed
        if self._traffic_state == "yellow":
            wk_fact = self.get_parameter("workers_factor").value
            self._publish(self._line_vel_l * wk_fact,
                          self._line_vel_r * wk_fact)
            return

        elapsed  = self._now() - self._state_start
        cmd      = self._sign_command
        detected = self._sign_detected

        rising  = detected and not self._prev_detected
        falling = not detected and self._prev_detected
        self._prev_detected = detected

        p = self.get_parameter
        gw_time   = p("give_way_stop_time").value
        sh_time   = p("stop_hold_time").value
        wk_fact   = p("workers_factor").value
        app_time  = p("approach_time").value
        t_time    = p("turn_time").value
        t_omg     = p("turn_omega").value
        t_v       = p("turn_v").value
        s_time    = p("straight_time").value
        s_v       = p("straight_v").value
        arm_delay = p("arm_delay").value

        # IDLE: wait for new command
        if self._state == S_IDLE:
            if rising and not self._in_cooldown(cmd):
                if cmd == "give_way":
                    self._enter(S_PENDING_GIVE_WAY, cmd)
                elif cmd == "stop":
                    self._enter(S_STOP, cmd)
                elif cmd == "workers":
                    self._enter(S_WORKERS, cmd)
                elif cmd == "turn_left":
                    self._enter(S_PENDING_LEFT, cmd)
                elif cmd == "turn_right":
                    self._enter(S_PENDING_RIGHT, cmd)
                elif cmd == "go_straight":
                    self._enter(S_PENDING_STRAIGHT, cmd)
            self._passthrough()

        # PENDING_GIVE_WAY: follow line while sign visible
        elif self._state == S_PENDING_GIVE_WAY:
            if falling:
                self._enter(S_GIVE_WAY)
            else:
                self._passthrough()

        # GIVE_WAY: stop 2 s after losing sign, then continue
        elif self._state == S_GIVE_WAY:
            if elapsed < gw_time:
                self._stop()
            else:
                self._enter(S_IDLE)

        # STOP: stop while sign visible
        elif self._state == S_STOP:
            if not detected:
                self._enter(S_STOP_HOLD)
            else:
                self._stop()

        # STOP_HOLD: short extra pause after stop disappears
        elif self._state == S_STOP_HOLD:
            if elapsed < sh_time:
                self._stop()
            else:
                self._enter(S_IDLE)

        # WORKERS: reduce speed while sign visible
        elif self._state == S_WORKERS:
            if detected:
                self._publish(self._line_vel_l * wk_fact,
                              self._line_vel_r * wk_fact)
            else:
                self._enter(S_IDLE)
                self._passthrough()

        # PENDING_*: follow the line until arm_delay, then run the maneuver
        elif self._state in (S_PENDING_LEFT, S_PENDING_RIGHT, S_PENDING_STRAIGHT):
            if elapsed >= arm_delay:
                if self._state == S_PENDING_LEFT:
                    self._enter(S_APPROACH_LEFT)
                elif self._state == S_PENDING_RIGHT:
                    self._enter(S_APPROACH_RIGHT)
                else:
                    self._enter(S_GOING_STRAIGHT)
            elif self._line_detected:
                self._passthrough()
            else:
                # line lost during count: drive straight, do not stop
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, 0.0)
                self._publish(vl, vr)

        # APPROACH_LEFT / APPROACH_RIGHT: straight run before the turn
        elif self._state == S_APPROACH_LEFT:
            if elapsed < app_time:
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, 0.0)
                self._publish(vl, vr)
            else:
                self._enter(S_TURNING_LEFT)

        elif self._state == S_APPROACH_RIGHT:
            if elapsed < app_time:
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, 0.0)
                self._publish(vl, vr)
            else:
                self._enter(S_TURNING_RIGHT)

        # TURNING_LEFT
        elif self._state == S_TURNING_LEFT:
            if elapsed < t_time:
                omega = FORWARD_SIGN * t_omg          # positive omega turns left
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, omega)
                self._publish(vl, vr)
            else:
                self._enter(S_IDLE)

        # TURNING_RIGHT
        elif self._state == S_TURNING_RIGHT:
            if elapsed < t_time:
                omega = -(FORWARD_SIGN * t_omg)       # negative omega turns right
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, omega)
                self._publish(vl, vr)
            else:
                self._enter(S_IDLE)

        # GOING_STRAIGHT: drive straight ignoring line_follower
        elif self._state == S_GOING_STRAIGHT:
            if elapsed < s_time:
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * s_v, 0.0)
                self._publish(vl, vr)
            else:
                self._enter(S_IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = SignBehaviorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()