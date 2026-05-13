#!/usr/bin/env python3
"""Controlador PID en lazo cerrado para PuzzleBot con odometría por encoders.

Sigue una secuencia de 'segmentos' (rectos y giros) construida a partir del
parámetro `task` (SQUARE o WAYPOINTS). Todas las constantes viven en
config/pid.yaml; se ajustan en caliente con `ros2 param set`.
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


IDLE, TURNING, STRAIGHT, DONE = "IDLE", "TURNING", "STRAIGHT", "DONE"


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PID:
    def __init__(self, kp, ki, kd, i_max):
        self.kp, self.ki, self.kd, self.i_max = kp, ki, kd, i_max
        self._i = 0.0
        self._prev = None

    def reset(self):
        self._i = 0.0
        self._prev = None

    def compute(self, err, dt):
        self._i = clamp(self._i + err * dt, -self.i_max, self.i_max)
        d = (err - self._prev) / dt if (self._prev is not None and dt > 1e-6) else 0.0
        self._prev = err
        return self.kp * err + self.ki * self._i + self.kd * d


class PuzzlebotPID(Node):

    def __init__(self):
        super().__init__('pid_controller')

        # Parámetros (todos vienen de config/pid.yaml o de ros2 param set)
        self.declare_parameters(
            namespace='',
            parameters=[
                ('task', 'SQUARE'),
                ('wheel_radius', 0.05),
                ('wheel_base', 0.19),
                ('v_max', 0.20),
                ('v_min', 0.04),
                ('w_max', 0.80),
                ('w_min', 0.08),
                ('kp_dist', 1.20), ('ki_dist', 0.02), ('kd_dist', 0.10),
                ('kp_head', 2.50), ('ki_head', 0.05), ('kd_head', 0.20),
                ('i_max', 0.30),
                ('dist_tol', 0.03),
                ('angle_tol', 0.03),
                ('timeout_factor', 2.5),
                ('control_period', 0.05),
                ('startup_delay', 2.0),
                ('waypoints', [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
                ('square_side', 2.0),
                ('square_turn', math.pi / 2.0),
            ],
        )

        gp = lambda n: self.get_parameter(n).value
        self.task = str(gp('task')).upper()
        self.WHEEL_RADIUS = gp('wheel_radius')
        self.WHEEL_BASE   = gp('wheel_base')
        self.V_MAX = gp('v_max'); self.V_MIN = gp('v_min')
        self.W_MAX = gp('w_max'); self.W_MIN = gp('w_min')
        self.DIST_TOL = gp('dist_tol')
        self.ANGLE_TOL = gp('angle_tol')
        self.TIMEOUT_FACTOR = gp('timeout_factor')

        self.pid_d = PID(gp('kp_dist'), gp('ki_dist'), gp('kd_dist'), gp('i_max'))
        self.pid_h = PID(gp('kp_head'), gp('ki_head'), gp('kd_head'), gp('i_max'))

        self.pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)

        self.enc_l = 0.0
        self.enc_r = 0.0
        self.create_subscription(Float32, '/VelEncL',
                                 lambda m: setattr(self, 'enc_l', m.data), 10)
        self.create_subscription(Float32, '/VelEncR',
                                 lambda m: setattr(self, 'enc_r', m.data), 10)

        self.x = 0.0; self.y = 0.0; self.th = 0.0
        self._last_t = None

        self.plan = []
        self.idx = 0
        self.state = IDLE

        self.seg_start_t = 0.0
        self.seg_timeout = 0.0
        self.seg_target_th = 0.0
        self.seg_target_dist = 0.0
        self.seg_travelled = 0.0

        self._build_plan()

        self._ready_at = self.get_clock().now().nanoseconds / 1e9 + gp('startup_delay')
        self.create_timer(gp('control_period'), self._loop)

        self.get_logger().info(
            f"PuzzlebotPID listo | task={self.task} | {len(self.plan)} segmentos"
        )

    # ── Odometría ────────────────────────────────────────────────────────────
    def _odom(self, now):
        if self._last_t is None:
            self._last_t = now
            return
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0:
            return

        vl = self.enc_l * self.WHEEL_RADIUS
        vr = self.enc_r * self.WHEEL_RADIUS
        v  = (vl + vr) / 2.0
        w  = (vr - vl) / self.WHEEL_BASE

        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.th = wrap(self.th + w * dt)

        self.seg_travelled += abs(v) * dt

    # ── Motores ──────────────────────────────────────────────────────────────
    def _wheels(self, v, w):
        vl = (v - w * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        vr = (v + w * self.WHEEL_BASE / 2.0) / self.WHEEL_RADIUS
        return vl, vr

    def _cmd(self, v, w):
        vl, vr = self._wheels(v, w)
        ml, mr = Float32(), Float32()
        ml.data = float(vl); mr.data = float(vr)
        self.pub_l.publish(ml); self.pub_r.publish(mr)

    def _stop(self):
        self._cmd(0.0, 0.0)

    # ── Plan ─────────────────────────────────────────────────────────────────
    def _build_plan(self):
        if self.task == "SQUARE":
            side = self.get_parameter('square_side').value
            turn = self.get_parameter('square_turn').value
            for _ in range(4):
                self.plan.append(('straight', side))
                self.plan.append(('turn', turn))
            self.get_logger().info(f"[SQUARE] 4 lados x {side} m + 4 giros {math.degrees(turn):.1f}°")

        elif self.task == "WAYPOINTS":
            flat = list(self.get_parameter('waypoints').value)
            wps = list(zip(flat[0::2], flat[1::2]))
            cx, cy, cth = 0.0, 0.0, 0.0
            for i, (wx, wy) in enumerate(wps):
                dx, dy = wx - cx, wy - cy
                dist = math.hypot(dx, dy)
                tgt_th = math.atan2(dy, dx)
                delta_th = wrap(tgt_th - cth)

                if dist < self.DIST_TOL:
                    self.get_logger().warn(
                        f"  WP{i+1} ({wx},{wy}): dist={dist:.3f} m < tol — ignorado"
                    )
                    continue

                if abs(delta_th) > self.ANGLE_TOL:
                    self.plan.append(('turn', delta_th))
                self.plan.append(('straight', dist))

                self.get_logger().info(
                    f"  WP{i+1} ({wx:.2f},{wy:.2f}) | "
                    f"Δθ={math.degrees(delta_th):.1f}° | d={dist:.3f} m"
                )
                cx, cy, cth = wx, wy, tgt_th
        else:
            self.get_logger().error(f"task '{self.task}' no reconocido (use SQUARE o WAYPOINTS).")

    def _start_segment(self):
        kind, value = self.plan[self.idx]
        self.seg_start_t = self.get_clock().now().nanoseconds / 1e9
        self.seg_travelled = 0.0
        self.pid_d.reset(); self.pid_h.reset()

        if kind == 'straight':
            self.seg_target_dist = value
            self.seg_target_th   = self.th
            self.seg_timeout     = (value / self.V_MIN) * self.TIMEOUT_FACTOR
            self.state = STRAIGHT
            self.get_logger().info(
                f"[{self.idx+1}/{len(self.plan)}] STRAIGHT {value:.2f} m | "
                f"θ={math.degrees(self.seg_target_th):.1f}° | timeout={self.seg_timeout:.1f} s"
            )
        elif kind == 'turn':
            self.seg_target_th   = wrap(self.th + value)
            self.seg_target_dist = 0.0
            self.seg_timeout     = (abs(value) / self.W_MIN) * self.TIMEOUT_FACTOR
            self.state = TURNING
            self.get_logger().info(
                f"[{self.idx+1}/{len(self.plan)}] TURN {'L' if value>0 else 'R'} "
                f"{math.degrees(value):.1f}° | θ_target={math.degrees(self.seg_target_th):.1f}° | "
                f"timeout={self.seg_timeout:.1f} s"
            )

    def _advance(self):
        self._stop()
        self.idx += 1
        if self.idx < len(self.plan):
            self._start_segment()
        else:
            self.state = DONE

    # ── Loop de control ──────────────────────────────────────────────────────
    def _loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        self._odom(now)
        dt = self.get_parameter('control_period').value

        if self.state == IDLE:
            if now >= self._ready_at and self.plan:
                self._start_segment()
            return

        if self.state == DONE:
            self._stop()
            self.get_logger().info(
                f"✅ Completado | x={self.x:.3f} m  y={self.y:.3f} m  "
                f"θ={math.degrees(self.th):.1f}°"
            )
            raise SystemExit

        elapsed = now - self.seg_start_t

        if self.state == TURNING:
            err_th = wrap(self.seg_target_th - self.th)
            if abs(err_th) < self.ANGLE_TOL:
                self.get_logger().info(f"  Turn OK | err={math.degrees(err_th):.2f}°")
                self._advance(); return
            if elapsed > self.seg_timeout:
                self.get_logger().warn(f"  ⚠ Turn TIMEOUT ({elapsed:.1f} s) — skip")
                self._advance(); return
            raw_w = self.pid_h.compute(err_th, dt)
            if abs(raw_w) < self.W_MIN:
                raw_w = math.copysign(self.W_MIN, raw_w)
            self._cmd(0.0, clamp(raw_w, -self.W_MAX, self.W_MAX))

        elif self.state == STRAIGHT:
            err_dist = self.seg_target_dist - self.seg_travelled
            if err_dist < self.DIST_TOL:
                self.get_logger().info(f"  Straight OK | viajado={self.seg_travelled:.3f} m")
                self._advance(); return
            if elapsed > self.seg_timeout:
                self.get_logger().warn(
                    f"  ⚠ Straight TIMEOUT ({elapsed:.1f} s) | "
                    f"viajado={self.seg_travelled:.3f}/{self.seg_target_dist:.2f} m — skip"
                )
                self._advance(); return
            raw_v = self.pid_d.compute(err_dist, dt)
            if 0 < raw_v < self.V_MIN:
                raw_v = self.V_MIN
            v = clamp(raw_v, 0.0, self.V_MAX)
            err_th = wrap(self.seg_target_th - self.th)
            w = clamp(self.pid_h.compute(err_th, dt), -self.W_MAX, self.W_MAX)
            self._cmd(v, w)


def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotPID()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().warn('Deteniendo.')
        node._stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
