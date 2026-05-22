#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String

WHEEL_RADIUS = 0.05154
WHEEL_BASE   = 0.19
FORWARD_SIGN = -1

KP = 0.3
KD = 0.08

V_BASE    = 0.6    # m/s cruise speed
V_MIN     = 0.04   # m/s minimum speed
OMEGA_MAX = 2.0    # rad/s saturation

SHIFT_SCALE  = 160.0  # pixels → ±1 normalised error
DEADBAND     = 0.06   # normalised units
LOST_TIMEOUT = 0.5    # s without detection before stopping
CTRL_DT      = 0.05   # 20 Hz control loop
DERIV_ALPHA  = 0.15   # derivative low-pass

CROSSING_TIME  = 3   # s to drive straight through intersection
TURN_TIME      = 1.8   # s to execute a directional turn
TURN_OMEGA     = 0.8   # rad/s for intersection turns
STOP_WAIT      = 3.0   # s to stop at a stop sign before proceeding
APPROACH_TIME  = 0.5   # s of straight driving before executing a turn
COOLDOWN_TIME  = 3.0   # s after crossing before next intersection can trigger


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def unicycle_to_wheels(v, omega):
    vl = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    vr = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return vl, vr


class LineFollowerNode(Node):

    def __init__(self):
        super().__init__("line_follower")

        self.declare_parameter("kp",            KP)
        self.declare_parameter("kd",            KD)
        self.declare_parameter("v_base",        V_BASE)
        self.declare_parameter("v_min",         V_MIN)
        self.declare_parameter("omega_max",     OMEGA_MAX)
        self.declare_parameter("shift_scale",   SHIFT_SCALE)
        self.declare_parameter("deadband",      DEADBAND)
        self.declare_parameter("lost_timeout",  LOST_TIMEOUT)
        self.declare_parameter("deriv_alpha",   DERIV_ALPHA)
        self.declare_parameter("crossing_time", CROSSING_TIME)
        self.declare_parameter("turn_time",     TURN_TIME)
        self.declare_parameter("turn_omega",    TURN_OMEGA)
        self.declare_parameter("stop_wait",     STOP_WAIT)
        self.declare_parameter("cooldown_time", COOLDOWN_TIME)

        self.create_subscription(Float32, "/line/shift",        self._cb_shift,        10)
        self.create_subscription(Bool,    "/line/detected",     self._cb_detected,     10)
        self.create_subscription(Bool,    "/line/intersection", self._cb_intersection, 10)
        self.create_subscription(String,  "/sign/command",      self._cb_sign,         10)

        self.pub_l = self.create_publisher(Float32, "VelocitySetL", 10)
        self.pub_r = self.create_publisher(Float32, "VelocitySetR", 10)

        self._shift           = 0.0
        self._detected        = False
        self._last_seen_t     = self._now()
        self._at_intersection = False
        self._prev_inters     = False   # for rising-edge detection
        self._sign_command    = "none"

        # Intersection state machine
        self._state              = "FOLLOWING"  # "FOLLOWING" | "CROSSING"
        self._crossing_action    = "straight"
        self._crossing_start     = 0.0
        self._last_crossing_end  = -COOLDOWN_TIME  # allow first trigger immediately

        self._prev_err   = 0.0
        self._filtered_d = 0.0
        self._last_t     = self._now()

        self.create_timer(CTRL_DT, self._control_loop)

        self.get_logger().info(
            f"LineFollower ready  Kp={KP}  Kd={KD}  v_base={V_BASE} m/s  "
            f"[intersection support ON]")

    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb_shift(self, msg: Float32):
        self._shift = float(msg.data)

    def _cb_detected(self, msg: Bool):
        self._detected = bool(msg.data)
        if self._detected:
            self._last_seen_t = self._now()

    def _cb_intersection(self, msg: Bool):
        self._at_intersection = bool(msg.data)

    def _cb_sign(self, msg: String):
        self._sign_command = msg.data

    def _publish_wheels(self, wl: float, wr: float):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wl), float(wr)
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)

    def _stop(self):
        self._publish_wheels(0.0, 0.0)

    # ------------------------------------------------------------------
    def _control_loop(self):
        now = self._now()
        dt  = max(1e-3, now - self._last_t)
        self._last_t = now

        v0    = self.get_parameter("v_base").value
        kp    = self.get_parameter("kp").value
        kd    = self.get_parameter("kd").value
        omax  = self.get_parameter("omega_max").value
        sscl  = self.get_parameter("shift_scale").value
        db    = self.get_parameter("deadband").value
        tout  = self.get_parameter("lost_timeout").value
        alpha = self.get_parameter("deriv_alpha").value
        c_time = self.get_parameter("crossing_time").value
        t_time = self.get_parameter("turn_time").value
        t_omg  = self.get_parameter("turn_omega").value
        s_wait = self.get_parameter("stop_wait").value
        cool   = self.get_parameter("cooldown_time").value

        # ── CROSSING STATE ──────────────────────────────────────────────
        if self._state == "CROSSING":
            elapsed = now - self._crossing_start
            action  = self._crossing_action

            if action == "straight":
                if elapsed < c_time:
                    vl, vr = unicycle_to_wheels(FORWARD_SIGN * v0, 0.0)
                    self._publish_wheels(vl, vr)
                else:
                    self._finish_crossing(now)
                return

            elif action == "turn_left":
                # Brief straight approach, then turn
                if elapsed < APPROACH_TIME:
                    vl, vr = unicycle_to_wheels(FORWARD_SIGN * v0, 0.0)
                elif elapsed < APPROACH_TIME + t_time:
                    # Positive omega = left turn (see sign convention in PD controller)
                    omega = FORWARD_SIGN * t_omg
                    vl, vr = unicycle_to_wheels(FORWARD_SIGN * v0 * 0.5, omega)
                else:
                    self._finish_crossing(now)
                    return
                self._publish_wheels(vl, vr)
                return

            elif action == "turn_right":
                if elapsed < APPROACH_TIME:
                    vl, vr = unicycle_to_wheels(FORWARD_SIGN * v0, 0.0)
                elif elapsed < APPROACH_TIME + t_time:
                    omega = -(FORWARD_SIGN * t_omg)
                    vl, vr = unicycle_to_wheels(FORWARD_SIGN * v0 * 0.5, omega)
                else:
                    self._finish_crossing(now)
                    return
                self._publish_wheels(vl, vr)
                return

            elif action == "stop":
                if elapsed < s_wait:
                    self._stop()
                elif elapsed < s_wait + c_time:
                    vl, vr = unicycle_to_wheels(FORWARD_SIGN * v0, 0.0)
                    self._publish_wheels(vl, vr)
                else:
                    self._finish_crossing(now)
                return

        # ── FOLLOWING STATE ─────────────────────────────────────────────

        # Detect intersection rising edge and trigger crossing
        rising_edge = self._at_intersection and not self._prev_inters
        cooldown_ok = (now - self._last_crossing_end) > cool
        self._prev_inters = self._at_intersection

        if rising_edge and cooldown_ok:
            cmd = self._sign_command
            if cmd == "turn_left":
                action = "turn_left"
            elif cmd == "turn_right":
                action = "turn_right"
            elif cmd == "stop":
                action = "stop"
            else:
                # "none", "go_straight", "workers", or anything else → go straight
                action = "straight"

            self._crossing_action = action
            self._crossing_start  = now
            self._state           = "CROSSING"
            self.get_logger().info(
                f"[Intersection] sign='{cmd}'  action={action}")
            # Re-enter so the CROSSING branch runs immediately
            self._prev_err   = 0.0
            self._filtered_d = 0.0
            return

        # Stop if line has been lost for too long
        if not self._detected and (now - self._last_seen_t > tout):
            self._prev_err   = 0.0
            self._filtered_d = 0.0
            self._stop()
            return

        # Hold last steering while crossing a short gap
        if not self._detected:
            return

        # Normalise shift error to [-1, 1]
        err = self._shift / sscl
        if abs(err) < db:
            err = 0.0

        raw_d = (err - self._prev_err) / dt
        self._filtered_d = alpha * raw_d + (1.0 - alpha) * self._filtered_d
        self._prev_err = err

        omega = -(kp * err + kd * self._filtered_d)
        omega = clamp(omega, -omax, omax)

        vl, vr = unicycle_to_wheels(FORWARD_SIGN * v0, FORWARD_SIGN * omega)
        self._publish_wheels(vl, vr)

    def _finish_crossing(self, now: float):
        self._state             = "FOLLOWING"
        self._last_crossing_end = now
        self._prev_inters       = False  # reset so the next real intersection triggers
        self.get_logger().info("[Intersection] Crossing done — resuming line following")


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
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
