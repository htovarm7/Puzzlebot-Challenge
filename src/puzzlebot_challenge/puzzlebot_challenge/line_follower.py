#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

WHEEL_RADIUS = 0.05154
WHEEL_BASE   = 0.19
FORWARD_SIGN = -1

KP = 0.006
KI = 0.0002
KD = 0.003

V_BASE    = 0.12
V_MIN     = 0.04
OMEGA_MAX = 1.8

SHIFT_SCALE  = 160.0
LOST_TIMEOUT = 0.5
CTRL_DT      = 0.05


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def unicycle_to_wheels(v, omega):
    v_l = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_r = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_l, v_r


class LineFollowerNode(Node):

    def __init__(self):
        super().__init__("line_follower")

        self.declare_parameter("kp",           KP)
        self.declare_parameter("ki",           KI)
        self.declare_parameter("kd",           KD)
        self.declare_parameter("v_base",       V_BASE)
        self.declare_parameter("v_min",        V_MIN)
        self.declare_parameter("omega_max",    OMEGA_MAX)
        self.declare_parameter("shift_scale",  SHIFT_SCALE)
        self.declare_parameter("lost_timeout", LOST_TIMEOUT)

        self.create_subscription(Float32, "/line/shift",    self._cb_shift,    10)
        self.create_subscription(Float32, "/line/angle",    self._cb_angle,    10)
        self.create_subscription(Bool,    "/line/detected", self._cb_detected, 10)

        self.pub_l = self.create_publisher(Float32, "/VelocitySetL", 10)
        self.pub_r = self.create_publisher(Float32, "/VelocitySetR", 10)

        self._shift    = 0.0
        self._angle    = 90.0
        self._detected = False
        self._last_seen_t = self.get_clock().now().nanoseconds / 1e9

        self._integral = 0.0
        self._prev_err = 0.0
        self._last_t   = self.get_clock().now().nanoseconds / 1e9

        self.create_timer(CTRL_DT, self._control_loop)

        self.get_logger().info(
            f"LineFollowerNode ready  Kp={KP} Ki={KI} Kd={KD}  v_base={V_BASE} m/s"
        )

    def _cb_shift(self, msg: Float32):
        self._shift = float(msg.data)

    def _cb_angle(self, msg: Float32):
        self._angle = float(msg.data)

    def _cb_detected(self, msg: Bool):
        self._detected = bool(msg.data)
        if self._detected:
            self._last_seen_t = self.get_clock().now().nanoseconds / 1e9

    def _publish_wheels(self, wL: float, wR: float):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wL), float(wR)
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)

    def _stop(self):
        try:
            self._publish_wheels(0.0, 0.0)
        except Exception:
            pass

    def _cmd_unicycle(self, v: float, omega: float):
        v     = clamp(v,     -V_BASE,    V_BASE)
        omega = clamp(omega, -OMEGA_MAX, OMEGA_MAX)
        wL, wR = unicycle_to_wheels(FORWARD_SIGN * v, omega)
        self._publish_wheels(wL, wR)

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt  = max(1e-3, now - self._last_t)
        self._last_t = now

        kp   = self.get_parameter("kp").value
        ki   = self.get_parameter("ki").value
        kd   = self.get_parameter("kd").value
        v0   = self.get_parameter("v_base").value
        vmin = self.get_parameter("v_min").value
        omax = self.get_parameter("omega_max").value
        sscl = self.get_parameter("shift_scale").value
        tout = self.get_parameter("lost_timeout").value

        if not self._detected and (now - self._last_seen_t > tout):
            self._integral = 0.0
            self._prev_err = 0.0
            self._stop()
            return

        if not self._detected:
            return

        err = self._shift
        self._integral = clamp(self._integral + err * dt, -200.0, 200.0)
        derivative     = (err - self._prev_err) / dt
        self._prev_err = err

        omega = -(kp * err + ki * self._integral + kd * derivative)
        omega = clamp(omega, -omax, omax)

        v = v0 * (1.0 - min(1.0, abs(err) / sscl))
        v = max(vmin, v)

        self._cmd_unicycle(v, omega)


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
