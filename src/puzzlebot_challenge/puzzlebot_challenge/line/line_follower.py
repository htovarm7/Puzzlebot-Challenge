#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool

WHEEL_RADIUS = 0.05154
WHEEL_BASE   = 0.19
FORWARD_SIGN = -1

KP = 0.45
KD = 0.14
KA = 0.35

V_BASE    = 0.15    # m/s cruise speed
V_MIN     = 0.05   # m/s velocidad mínima durante corrección
OMEGA_MAX = 1.8    # rad/s — limita cuánto puede corregir de golpe

SHIFT_SCALE  = 160.0
ANGLE_SCALE  = 30.0
DEADBAND     = 0.04
LOST_TIMEOUT = 0.5
CTRL_DT      = 0.05
DERIV_ALPHA  = 0.30


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
        self.declare_parameter("ka",           KA)
        self.declare_parameter("v_base",       V_BASE)
        self.declare_parameter("v_min",        V_MIN)
        self.declare_parameter("omega_max",    OMEGA_MAX)
        self.declare_parameter("shift_scale",  SHIFT_SCALE)
        self.declare_parameter("deadband",     DEADBAND)
        self.declare_parameter("lost_timeout", LOST_TIMEOUT)
        self.declare_parameter("deriv_alpha",  DERIV_ALPHA)

        self.create_subscription(Float32, "/line/shift",    self._cb_shift,    10)
        self.create_subscription(Float32, "/line/angle",    self._cb_angle,    10)
        self.create_subscription(Bool,    "/line/detected", self._cb_detected, 10)

        self.pub_l = self.create_publisher(Float32, "/VelocitySetL", 10)
        self.pub_r = self.create_publisher(Float32, "/VelocitySetR", 10)

        self._shift       = 0.0
        self._angle       = 90.0
        self._detected    = False
        self._last_seen_t = self._now()
        self._prev_err    = 0.0
        self._filtered_d  = 0.0
        self._last_t      = self._now()

        self.create_timer(CTRL_DT, self._control_loop)
        self.get_logger().info(f"LineFollower ready  Kp={KP}  Kd={KD}  v_base={V_BASE} m/s")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb_shift(self, msg: Float32):
        self._shift = float(msg.data)

    def _cb_angle(self, msg: Float32):
        self._angle = float(msg.data)

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

    def _control_loop(self):
        now = self._now()
        dt  = max(1e-3, now - self._last_t)
        self._last_t = now

        v0    = self.get_parameter("v_base").value
        kp    = self.get_parameter("kp").value
        kd    = self.get_parameter("kd").value
        ka    = self.get_parameter("ka").value
        omax  = self.get_parameter("omega_max").value
        sscl  = self.get_parameter("shift_scale").value
        db    = self.get_parameter("deadband").value
        tout  = self.get_parameter("lost_timeout").value
        alpha = self.get_parameter("deriv_alpha").value

        if not self._detected and (now - self._last_seen_t > tout):
            self._prev_err   = 0.0
            self._filtered_d = 0.0
            self._stop()
            return

        if not self._detected:
            return

        shift_err = self._shift / sscl
        if abs(shift_err) < db:
            shift_err = 0.0
        angle_err = (self._angle - 90.0) / ANGLE_SCALE
        err = shift_err + ka * angle_err

        raw_d = (err - self._prev_err) / dt
        self._filtered_d = alpha * raw_d + (1.0 - alpha) * self._filtered_d
        self._prev_err = err

        omega = -(kp * err + kd * self._filtered_d)
        omega = clamp(omega, -omax, omax)

        # Reducir velocidad lineal cuando hay giro pronunciado:
        # a omega=0 → v=v0 ; a omega=±omax → v=v_min
        v_min = self.get_parameter("v_min").value
        speed_factor = 1.0 - clamp(abs(omega) / omax, 0.0, 1.0) * 0.8
        v = max(v_min, v0 * speed_factor)

        vl, vr = unicycle_to_wheels(FORWARD_SIGN * v, FORWARD_SIGN * omega)
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
