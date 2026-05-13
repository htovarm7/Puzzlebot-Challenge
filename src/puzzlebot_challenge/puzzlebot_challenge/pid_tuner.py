#!/usr/bin/env python3
"""
puzzlebot_pid_tuner.py
======================
Nodo de sintonización de PIDs para el PuzzleBot (ROS2 Humble).

Ejecuta UN solo segmento de prueba (recto o giro) en bucle infinito,
publicando métricas en tiempo real para que puedas observar la respuesta
y ajustar las ganancias SIN reiniciar el nodo.

Modos de prueba
---------------
  straight   : avanza TARGET_DIST metros, se detiene, regresa, repite.
  turn       : gira TARGET_ANGLE grados, regresa al origen, repite.

Ajuste en caliente (ROS2 parameters)
-------------------------------------
Mientras el nodo corre, en otra terminal:

  # Ver todos los parámetros actuales
  ros2 param list /pid_tuner

  # Cambiar una ganancia (efecto inmediato en el siguiente ciclo)
  ros2 param set /pid_tuner kp_dist 1.5
  ros2 param set /pid_tuner ki_dist 0.03
  ros2 param set /pid_tuner kd_dist 0.12
  ros2 param set /pid_tuner kp_head 2.8
  ros2 param set /pid_tuner ki_head 0.04
  ros2 param set /pid_tuner kd_head 0.25

  # Cambiar el modo de prueba
  ros2 param set /pid_tuner mode straight   # o: turn

  # Cambiar la distancia / ángulo objetivo
  ros2 param set /pid_tuner target_dist 1.0   # metros
  ros2 param set /pid_tuner target_angle 90.0 # grados

Métricas publicadas (Float32)
------------------------------
  /tuner/error_dist   — error de distancia al objetivo [m]
  /tuner/error_head   — error de heading [rad]
  /tuner/cmd_v        — velocidad lineal comandada [m/s]
  /tuner/cmd_w        — velocidad angular comandada [rad/s]
  /tuner/pose_x       — posición X estimada [m]
  /tuner/pose_y       — posición Y estimada [m]
  /tuner/pose_th      — heading estimado [rad]
  /tuner/travelled    — distancia recorrida en segmento actual [m]

Grafica las métricas en tiempo real con:
  ros2 run rqt_plot rqt_plot /tuner/error_dist /tuner/error_head /tuner/cmd_v

CLI usage
---------
  python3 puzzlebot_pid_tuner.py straight
  python3 puzzlebot_pid_tuner.py turn

Author: Armando / MCR2 Mini Challenge 1 — PID Tuning Tool
"""

import sys
import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange
from std_msgs.msg import Float32


# ─────────────────────────────────────────────────────────────────────────────
# Robot constants
# ─────────────────────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.05   # [m]
WHEEL_BASE   = 0.19   # [m]

V_MAX  = 0.30   # [m/s]
V_MIN  = 0.04   # [m/s]
W_MAX  = 1.20   # [rad/s]
W_MIN  = 0.08   # [rad/s]
I_MAX  = 0.30   # anti-windup clamp

DIST_TOL  = 0.03   # [m]   goal acceptance
ANGLE_TOL = 0.03   # [rad] goal acceptance (~1.7°)

# Pause between repetitions [s]
REPEAT_PAUSE = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def unicycle_to_wheels(v, omega):
    vl = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    vr = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return vl, vr


# ─────────────────────────────────────────────────────────────────────────────
# PID
# ─────────────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral   = 0.0
        self._prev_error = None

    def update_gains(self, kp, ki, kd):
        """Hot-reload gains without resetting integrator."""
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def reset(self):
        self._integral   = 0.0
        self._prev_error = None

    def compute(self, error, dt):
        p = self.kp * error
        self._integral = clamp(self._integral + error * dt, -I_MAX, I_MAX)
        i = self.ki * self._integral
        d = 0.0
        if self._prev_error is not None and dt > 1e-6:
            d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error
        return p + i + d


# ─────────────────────────────────────────────────────────────────────────────
# FSM
# ─────────────────────────────────────────────────────────────────────────────
class Phase:
    WAIT     = "WAIT"       # post-reset pause
    FORWARD  = "FORWARD"    # executing the test segment
    RETURN   = "RETURN"     # returning to origin (opposite segment)


# ─────────────────────────────────────────────────────────────────────────────
# Tuner node
# ─────────────────────────────────────────────────────────────────────────────
class PIDTuner(Node):

    def __init__(self, initial_mode: str):
        super().__init__('pid_tuner')

        # ── ROS2 Parameters (hot-adjustable) ────────────────────────────
        def _fd(desc, lo=0.0, hi=10.0):
            d = ParameterDescriptor(description=desc)
            d.floating_point_range = [FloatingPointRange(from_value=lo, to_value=hi, step=0.0)]
            return d

        self.declare_parameter('kp_dist',      1.20,  _fd('P gain — distance PID'))
        self.declare_parameter('ki_dist',      0.02,  _fd('I gain — distance PID'))
        self.declare_parameter('kd_dist',      0.10,  _fd('D gain — distance PID'))
        self.declare_parameter('kp_head',      2.50,  _fd('P gain — heading PID'))
        self.declare_parameter('ki_head',      0.05,  _fd('I gain — heading PID'))
        self.declare_parameter('kd_head',      0.20,  _fd('D gain — heading PID'))
        self.declare_parameter('target_dist',  1.00,  _fd('Test distance [m]', 0.1, 5.0))
        self.declare_parameter('target_angle', 90.0,  _fd('Test angle [deg]',  10.0, 360.0))
        self.declare_parameter('mode', initial_mode,
                               ParameterDescriptor(description='"straight" or "turn"'))

        # ── Publishers: motor commands ───────────────────────────────────
        self.pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)

        # ── Publishers: telemetry ────────────────────────────────────────
        self._tpub = {
            k: self.create_publisher(Float32, f'/tuner/{k}', 10)
            for k in ['error_dist', 'error_head', 'cmd_v', 'cmd_w',
                      'pose_x', 'pose_y', 'pose_th', 'travelled']
        }

        # ── Subscribers: encoders ────────────────────────────────────────
        self.enc_l = 0.0
        self.enc_r = 0.0
        self.create_subscription(Float32, '/VelEncL', lambda m: setattr(self, 'enc_l', m.data), 10)
        self.create_subscription(Float32, '/VelEncR', lambda m: setattr(self, 'enc_r', m.data), 10)

        # ── PIDs ─────────────────────────────────────────────────────────
        self.pid_dist = PID()
        self.pid_head = PID()
        self._sync_gains()   # load initial values from parameters

        # ── Odometry ─────────────────────────────────────────────────────
        self.x  = 0.0
        self.y  = 0.0
        self.th = 0.0
        self._last_t = None

        # ── Iteration bookkeeping ─────────────────────────────────────────
        self.phase       = Phase.WAIT
        self.phase_start = None
        self.seg_start_x = 0.0
        self.seg_start_y = 0.0
        self.seg_target_th = 0.0   # desired heading for current segment
        self.seg_target_dist = 0.0
        self.iteration   = 0

        # ── Control loop 20 Hz ───────────────────────────────────────────
        self.timer = self.create_timer(0.05, self._loop)
        self.get_logger().info(
            f"PID Tuner ready | mode={initial_mode} | "
            "Adjust gains with:  ros2 param set /pid_tuner kp_dist <value>"
        )

        # Begin first pause
        self._enter_wait()

    # ─────────────────────────────────────────────────────────────────────
    # Gain sync (called each loop tick)
    # ─────────────────────────────────────────────────────────────────────
    def _sync_gains(self):
        self.pid_dist.update_gains(
            self.get_parameter('kp_dist').value,
            self.get_parameter('ki_dist').value,
            self.get_parameter('kd_dist').value,
        )
        self.pid_head.update_gains(
            self.get_parameter('kp_head').value,
            self.get_parameter('ki_head').value,
            self.get_parameter('kd_head').value,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Odometry
    # ─────────────────────────────────────────────────────────────────────
    def _odom(self, now):
        if self._last_t is None:
            self._last_t = now
            return
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0:
            return
        vl = self.enc_l * WHEEL_RADIUS
        vr = self.enc_r * WHEEL_RADIUS
        v     = (vr + vl) / 2.0
        omega = (vr - vl) / WHEEL_BASE
        self.x  += v * math.cos(self.th) * dt
        self.y  += v * math.sin(self.th) * dt
        self.th  = wrap_angle(self.th + omega * dt)

    # ─────────────────────────────────────────────────────────────────────
    # Telemetry publisher
    # ─────────────────────────────────────────────────────────────────────
    def _pub_tel(self, key, val):
        m = Float32()
        m.data = float(val)
        self._tpub[key].publish(m)

    def _publish_motors(self, vl, vr):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(vl), float(vr)
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)

    def _stop(self):
        self._publish_motors(0.0, 0.0)

    # ─────────────────────────────────────────────────────────────────────
    # Phase transitions
    # ─────────────────────────────────────────────────────────────────────
    def _enter_wait(self):
        self._stop()
        self.phase       = Phase.WAIT
        self.phase_start = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().info(f"  [iter {self.iteration}] Pausing {REPEAT_PAUSE} s…")

    def _enter_forward(self):
        mode = self.get_parameter('mode').value
        self._sync_gains()
        self.pid_dist.reset()
        self.pid_head.reset()
        self.phase       = Phase.FORWARD
        self.phase_start = self.get_clock().now().nanoseconds / 1e9
        self.seg_start_x = self.x
        self.seg_start_y = self.y

        if mode == 'straight':
            self.seg_target_dist = self.get_parameter('target_dist').value
            self.seg_target_th   = self.th   # keep current heading
            self.get_logger().info(
                f"  [iter {self.iteration}] FORWARD straight "
                f"{self.seg_target_dist:.2f} m | "
                f"Kp_d={self.pid_dist.kp:.3f} Ki_d={self.pid_dist.ki:.3f} Kd_d={self.pid_dist.kd:.3f}"
            )
        else:   # turn
            angle_deg = self.get_parameter('target_angle').value
            angle_rad = math.radians(angle_deg)
            self.seg_target_th   = wrap_angle(self.th + angle_rad)
            self.seg_target_dist = 0.0
            self.get_logger().info(
                f"  [iter {self.iteration}] FORWARD turn "
                f"{angle_deg:.1f}° | "
                f"Kp_h={self.pid_head.kp:.3f} Ki_h={self.pid_head.ki:.3f} Kd_h={self.pid_head.kd:.3f}"
            )

    def _enter_return(self):
        mode = self.get_parameter('mode').value
        self._sync_gains()
        self.pid_dist.reset()
        self.pid_head.reset()
        self.phase       = Phase.RETURN
        self.phase_start = self.get_clock().now().nanoseconds / 1e9
        self.seg_start_x = self.x
        self.seg_start_y = self.y

        if mode == 'straight':
            self.seg_target_dist = self.get_parameter('target_dist').value
            self.seg_target_th   = wrap_angle(self.th + math.pi)   # reverse direction
            self.get_logger().info(f"  [iter {self.iteration}] RETURN straight")
        else:
            angle_deg = self.get_parameter('target_angle').value
            angle_rad = math.radians(angle_deg)
            self.seg_target_th   = wrap_angle(self.th - angle_rad)  # opposite turn
            self.seg_target_dist = 0.0
            self.get_logger().info(f"  [iter {self.iteration}] RETURN turn -{angle_deg:.1f}°")

    # ─────────────────────────────────────────────────────────────────────
    # Control loop
    # ─────────────────────────────────────────────────────────────────────
    def _loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        self._odom(now)
        self._sync_gains()   # pick up any ros2 param set changes
        dt = 0.05

        mode = self.get_parameter('mode').value

        # ── WAIT ────────────────────────────────────────────────────────
        if self.phase == Phase.WAIT:
            self._stop()
            if now - self.phase_start >= REPEAT_PAUSE:
                self.iteration += 1
                self._enter_forward()
            return

        # ── Shared: compute travelled distance ───────────────────────────
        travelled = math.hypot(self.x - self.seg_start_x,
                               self.y - self.seg_start_y)

        # ── STRAIGHT segments ────────────────────────────────────────────
        if mode == 'straight':
            err_dist = self.seg_target_dist - travelled
            err_head = wrap_angle(self.seg_target_th - self.th)

            # Telemetry
            self._pub_tel('error_dist', err_dist)
            self._pub_tel('error_head', err_head)
            self._pub_tel('travelled',  travelled)
            self._pub_tel('pose_x', self.x)
            self._pub_tel('pose_y', self.y)
            self._pub_tel('pose_th', self.th)

            if err_dist < DIST_TOL:
                self._stop()
                self.get_logger().info(
                    f"    Goal reached | travelled={travelled:.3f} m  "
                    f"err={err_dist*100:.1f} cm  heading_err={math.degrees(err_head):.2f}°"
                )
                if self.phase == Phase.FORWARD:
                    self._enter_return()
                else:
                    self._enter_wait()
                return

            raw_v = self.pid_dist.compute(err_dist, dt)
            if 0 < raw_v < V_MIN:
                raw_v = V_MIN
            v = clamp(raw_v, 0.0, V_MAX)

            raw_w = self.pid_head.compute(err_head, dt)
            omega = clamp(raw_w, -W_MAX, W_MAX)

            self._pub_tel('cmd_v', v)
            self._pub_tel('cmd_w', omega)

            vl, vr = unicycle_to_wheels(v, omega)
            self._publish_motors(vl, vr)

        # ── TURN segments ────────────────────────────────────────────────
        else:
            err_head = wrap_angle(self.seg_target_th - self.th)
            err_dist = 0.0   # not used in turn mode

            self._pub_tel('error_dist', err_dist)
            self._pub_tel('error_head', err_head)
            self._pub_tel('travelled',  abs(err_head))   # repurpose as angle remaining
            self._pub_tel('pose_x', self.x)
            self._pub_tel('pose_y', self.y)
            self._pub_tel('pose_th', self.th)

            if abs(err_head) < ANGLE_TOL:
                self._stop()
                self.get_logger().info(
                    f"    Turn done | err={math.degrees(err_head):.2f}°"
                )
                if self.phase == Phase.FORWARD:
                    self._enter_return()
                else:
                    self._enter_wait()
                return

            raw_w = self.pid_head.compute(err_head, dt)
            if abs(raw_w) < W_MIN:
                raw_w = math.copysign(W_MIN, raw_w)
            omega = clamp(raw_w, -W_MAX, W_MAX)

            self._pub_tel('cmd_v', 0.0)
            self._pub_tel('cmd_w', omega)

            vl, vr = unicycle_to_wheels(0.0, omega)
            self._publish_motors(vl, vr)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    valid = {'straight', 'turn'}
    cli   = [a for a in sys.argv[1:] if not a.startswith('--')]
    mode  = cli[0].lower() if cli and cli[0].lower() in valid else 'straight'

    if not cli or cli[0].lower() not in valid:
        print(f"[INFO] Usage: python3 puzzlebot_pid_tuner.py [straight|turn]")
        print(f"[INFO] Defaulting to mode=straight")

    print(f"[INFO] Tuner mode = {mode}\n")

    rclpy.init(args=args)
    node = PIDTuner(initial_mode=mode)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn('Keyboard interrupt — stopping motors.')
        node._stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
