#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

WHEEL_RADIUS = 0.05154
WHEEL_BASE   = 0.19
FORWARD_SIGN = -1

# PD gains  (error is normalised to [-1, 1] so gains are resolution-independent)
KP = 1.2    # proportional — how hard to steer NOW
KD = 0.35   # derivative   — how hard to steer based on HOW FAST error is growing

V_BASE    = 0.12   # m/s cruise speed
V_MIN     = 0.04   # m/s minimum speed (never stop mid-line)
OMEGA_MAX = 2.0    # rad/s saturation

SHIFT_SCALE  = 160.0   # pixels that map to ±1 normalised error (half frame width)
LOST_TIMEOUT = 0.5     # seconds without detection before stopping
CTRL_DT      = 0.05    # control loop period (20 Hz)
DERIV_ALPHA  = 0.45    # derivative low-pass: 0=frozen  1=raw  (0.45 = light filter)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def unicycle_to_wheels(v, omega):
    vl = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    vr = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return vl, vr


class LineFollowerNode(Node):

    def __init__(self):
        super().__init__("line_follower")

        self.declare_parameter("kp",           KP)
        self.declare_parameter("kd",           KD)
        self.declare_parameter("v_base",       V_BASE)
        self.declare_parameter("v_min",        V_MIN)
        self.declare_parameter("omega_max",    OMEGA_MAX)
        self.declare_parameter("shift_scale",  SHIFT_SCALE)
        self.declare_parameter("lost_timeout", LOST_TIMEOUT)
        self.declare_parameter("deriv_alpha",  DERIV_ALPHA)

        self.create_subscription(Float32, "/line/shift",    self._cb_shift,    10)
        self.create_subscription(Bool,    "/line/detected", self._cb_detected, 10)

        self.pub_l = self.create_publisher(Float32, "/cmd/VelocitySetL", 10)
        self.pub_r = self.create_publisher(Float32, "/cmd/VelocitySetR", 10)

        self._shift       = 0.0
        self._detected    = False
        self._last_seen_t = self._now()

        self._prev_err   = 0.0   # previous normalised error
        self._filtered_d = 0.0   # low-pass filtered derivative

        self._last_t = self._now()

        self.create_timer(CTRL_DT, self._control_loop)

        self.get_logger().info(
            f"LineFollower (PD) ready  Kp={KP}  Kd={KD}  v_base={V_BASE} m/s")

    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb_shift(self, msg: Float32):
        self._shift = float(msg.data)

    def _cb_detected(self, msg: Bool):
        self._detected = bool(msg.data)
        if self._detected:
            self._last_seen_t = self._now()

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

        kp    = self.get_parameter("kp").value
        kd    = self.get_parameter("kd").value
        v0    = self.get_parameter("v_base").value
        vmin  = self.get_parameter("v_min").value
        omax  = self.get_parameter("omega_max").value
        sscl  = self.get_parameter("shift_scale").value
        tout  = self.get_parameter("lost_timeout").value
        alpha = self.get_parameter("deriv_alpha").value

        # Stop if line has been lost for too long
        if not self._detected and (now - self._last_seen_t > tout):
            self._prev_err   = 0.0
            self._filtered_d = 0.0
            self._stop()
            return

        # Hold last steering while crossing a gap (line temporarily missing)
        if not self._detected:
            return

        # Normalise error to [-1, 1]  (+1 = line is all the way to the right)
        err = self._shift / sscl

        # Derivative: low-pass filtered  (suppresses encoder/detection noise)
        raw_d = (err - self._prev_err) / dt
        self._filtered_d = alpha * raw_d + (1.0 - alpha) * self._filtered_d
        self._prev_err = err

        # PD steering command
        omega = -(kp * err + kd * self._filtered_d)
        omega = clamp(omega, -omax, omax)

        # Adaptive speed: slow down proportionally to |err| AND |d_err|
        # — |err|  slows the robot when it's already off-centre
        # — |d_err| slows it DOWN EARLY when the error is GROWING (anticipates curves)
        curve_load = clamp(abs(err) + 0.4 * abs(self._filtered_d), 0.0, 1.0)
        v = max(vmin, v0 * (1.0 - curve_load))

        vl, vr = unicycle_to_wheels(FORWARD_SIGN * v, omega)
        self._publish_wheels(vl, vr)


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
