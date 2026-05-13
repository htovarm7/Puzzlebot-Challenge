#!/usr/bin/env python3
"""
puzzlebot_pid.py  —  FIXED VERSION
====================================
ROS2 Humble — Manchester Robotics PuzzleBot
Controlador PID de lazo cerrado con odometría por encoders.

Bugs corregidos respecto a la versión anterior
-----------------------------------------------
1. _start_segment estaba definida DOS veces en la clase (Python sólo usa la
   última). Ahora existe una sola definición limpia.
2. `travelled` se calculaba con math.hypot + hasattr-fallback-a-0, lo que
   causaba que nunca alcanzara DIST_TOL. Ahora se acumula directamente
   integrando |v| * dt en cada tick — más robusto y no depende de x,y.
3. El watchdog para 2 m con V_MAX/2 daba 40 s, permitiendo que el robot
   avanzara indefinidamente. Ahora usa V_MIN como denominador → watchdog
   más ajustado y realista.

CLI
---
  python3 puzzlebot_pid.py square
  python3 puzzlebot_pid.py waypoints
"""

import sys
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros físicos del robot  ← mide los tuyos
# ─────────────────────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.05    # [m]
WHEEL_BASE   = 0.19    # [m]

# ─────────────────────────────────────────────────────────────────────────────
# Límites de velocidad
# ─────────────────────────────────────────────────────────────────────────────
V_MAX  = 0.20   # [m/s]   — conservador para lazo cerrado
V_MIN  = 0.04   # [m/s]   — vence fricción estática
W_MAX  = 0.80   # [rad/s]
W_MIN  = 0.08   # [rad/s]

# ─────────────────────────────────────────────────────────────────────────────
# Ganancias PID
# ─────────────────────────────────────────────────────────────────────────────
KP_DIST = 1.20;  KI_DIST = 0.02;  KD_DIST = 0.10
KP_HEAD = 2.50;  KI_HEAD = 0.05;  KD_HEAD = 0.20
I_MAX   = 0.30   # anti-windup

# ─────────────────────────────────────────────────────────────────────────────
# Tolerancias de llegada
# ─────────────────────────────────────────────────────────────────────────────
DIST_TOL  = 0.03   # [m]
ANGLE_TOL = 0.03   # [rad]  ≈ 1.7°

# ─────────────────────────────────────────────────────────────────────────────
# Watchdog: tiempo máximo = (dist/V_MIN) * TIMEOUT_FACTOR
# Usar V_MIN como denominador da el peor caso realista (más ajustado que V_MAX/2)
# ─────────────────────────────────────────────────────────────────────────────
TIMEOUT_FACTOR = 2.5

# ─────────────────────────────────────────────────────────────────────────────
# Tarea
# ─────────────────────────────────────────────────────────────────────────────
TASK = "SQUARE"

WAYPOINTS = [
    (1.00, 0.00),
    (1.00, 1.00),
    (0.00, 1.00),
    (0.00, 0.00),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def wheels(v, w):
    vl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    vr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return vl, vr


# ─────────────────────────────────────────────────────────────────────────────
# PID con anti-windup
# ─────────────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev = None

    def reset(self):
        self._i = 0.0
        self._prev = None

    def compute(self, err, dt):
        self._i = clamp(self._i + err * dt, -I_MAX, I_MAX)
        d = (err - self._prev) / dt if (self._prev is not None and dt > 1e-6) else 0.0
        self._prev = err
        return self.kp * err + self.ki * self._i + self.kd * d


# ─────────────────────────────────────────────────────────────────────────────
# Estados FSM
# ─────────────────────────────────────────────────────────────────────────────
IDLE, TURNING, STRAIGHT, DONE = "IDLE", "TURNING", "STRAIGHT", "DONE"


# ─────────────────────────────────────────────────────────────────────────────
# Nodo principal
# ─────────────────────────────────────────────────────────────────────────────
class PuzzlebotPID(Node):

    def __init__(self):
        super().__init__('puzzlebot_pid')

        self.pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)

        self.enc_l = 0.0
        self.enc_r = 0.0
        self.create_subscription(Float32, '/VelEncL', lambda m: setattr(self, 'enc_l', m.data), 10)
        self.create_subscription(Float32, '/VelEncR', lambda m: setattr(self, 'enc_r', m.data), 10)

        # Odometría
        self.x = 0.0;  self.y = 0.0;  self.th = 0.0
        self._last_t = None

        # PIDs
        self.pid_d = PID(KP_DIST, KI_DIST, KD_DIST)
        self.pid_h = PID(KP_HEAD, KI_HEAD, KD_HEAD)

        # Plan y FSM
        self.plan  = []
        self.idx   = 0
        self.state = IDLE

        # Variables de segmento — inicializadas aquí para que nunca sean None
        self.seg_start_t     = 0.0
        self.seg_timeout     = 0.0
        self.seg_target_th   = 0.0
        self.seg_target_dist = 0.0
        self.seg_travelled   = 0.0   # acumulador de distancia del segmento actual

        self._build_plan()
        self._ready_at = self.get_clock().now().nanoseconds / 1e9 + 2.0
        self.create_timer(0.05, self._loop)

        self.get_logger().info(
            f"PuzzlebotPID listo | task={TASK} | {len(self.plan)} segmentos"
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

        vl = self.enc_l * WHEEL_RADIUS
        vr = self.enc_r * WHEEL_RADIUS
        v  = (vl + vr) / 2.0
        w  = (vr - vl) / WHEEL_BASE

        self.x  += v * math.cos(self.th) * dt
        self.y  += v * math.sin(self.th) * dt
        self.th  = wrap(self.th + w * dt)

        # Acumular distancia lineal recorrida en el segmento actual.
        # Se usa |v| para contar también retrocesos por overshoot del PID.
        self.seg_travelled += abs(v) * dt

    # ── Motores ──────────────────────────────────────────────────────────────
    def _cmd(self, v, w):
        vl, vr = wheels(v, w)
        ml, mr = Float32(), Float32()
        ml.data = float(vl);  mr.data = float(vr)
        self.pub_l.publish(ml);  self.pub_r.publish(mr)

    def _stop(self):
        self._cmd(0.0, 0.0)

    # ── Constructor del plan ─────────────────────────────────────────────────
    def _build_plan(self):
        if TASK == "SQUARE":
            for _ in range(4):
                self.plan.append(('straight', 2.0))
                self.plan.append(('turn', math.pi / 2.0))
            self.get_logger().info("[SQUARE] 4 lados x 2 m + 4 giros 90°")

        elif TASK == "WAYPOINTS":
            cx, cy, cth = 0.0, 0.0, 0.0
            for i, (wx, wy) in enumerate(WAYPOINTS):
                dx, dy   = wx - cx, wy - cy
                dist     = math.hypot(dx, dy)
                tgt_th   = math.atan2(dy, dx)
                delta_th = wrap(tgt_th - cth)

                if dist < DIST_TOL:
                    self.get_logger().warn(
                        f"  WP{i+1} ({wx},{wy}): dist={dist:.3f} m < tolerancia — ignorado"
                    )
                    continue

                if abs(delta_th) > ANGLE_TOL:
                    self.plan.append(('turn', delta_th))
                self.plan.append(('straight', dist))

                self.get_logger().info(
                    f"  WP{i+1} ({wx:.2f},{wy:.2f}) | "
                    f"Δθ={math.degrees(delta_th):.1f}° | d={dist:.3f} m"
                )
                cx, cy, cth = wx, wy, tgt_th
        else:
            self.get_logger().error(f"TASK '{TASK}' no reconocido.")

    # ── Inicio de segmento — ÚNICA DEFINICIÓN ────────────────────────────────
    def _start_segment(self):
        kind, value = self.plan[self.idx]

        self.seg_start_t   = self.get_clock().now().nanoseconds / 1e9
        self.seg_travelled = 0.0   # reset del acumulador en cada segmento nuevo
        self.pid_d.reset()
        self.pid_h.reset()

        if kind == 'straight':
            self.seg_target_dist = value
            self.seg_target_th   = self.th                        # conservar heading actual
            self.seg_timeout     = (value / V_MIN) * TIMEOUT_FACTOR
            self.state = STRAIGHT
            self.get_logger().info(
                f"[{self.idx+1}/{len(self.plan)}] STRAIGHT {value:.2f} m | "
                f"θ={math.degrees(self.seg_target_th):.1f}° | timeout={self.seg_timeout:.1f} s"
            )

        elif kind == 'turn':
            self.seg_target_th   = wrap(self.th + value)
            self.seg_target_dist = 0.0
            self.seg_timeout     = (abs(value) / W_MIN) * TIMEOUT_FACTOR
            self.state = TURNING
            self.get_logger().info(
                f"[{self.idx+1}/{len(self.plan)}] TURN {'L' if value>0 else 'R'} "
                f"{math.degrees(value):.1f}° | "
                f"θ_target={math.degrees(self.seg_target_th):.1f}° | timeout={self.seg_timeout:.1f} s"
            )

    # ── Avanzar al siguiente segmento ────────────────────────────────────────
    def _advance(self):
        self._stop()
        self.idx += 1
        if self.idx < len(self.plan):
            self._start_segment()
        else:
            self.state = DONE

    # ── Loop de control 20 Hz ────────────────────────────────────────────────
    def _loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        self._odom(now)

        dt = 0.05

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
            raise SystemExit   # termina el spin limpiamente
            return

        elapsed = now - self.seg_start_t

        # ── TURNING ──────────────────────────────────────────────────────────
        if self.state == TURNING:
            err_th = wrap(self.seg_target_th - self.th)

            if abs(err_th) < ANGLE_TOL:
                self.get_logger().info(f"  Turn OK | err={math.degrees(err_th):.2f}°")
                self._advance()
                return

            if elapsed > self.seg_timeout:
                self.get_logger().warn(f"  ⚠ Turn TIMEOUT ({elapsed:.1f} s) — skip")
                self._advance()
                return

            raw_w = self.pid_h.compute(err_th, dt)
            if abs(raw_w) < W_MIN:
                raw_w = math.copysign(W_MIN, raw_w)
            self._cmd(0.0, clamp(raw_w, -W_MAX, W_MAX))

        # ── STRAIGHT ─────────────────────────────────────────────────────────
        elif self.state == STRAIGHT:
            err_dist = self.seg_target_dist - self.seg_travelled

            if err_dist < DIST_TOL:
                self.get_logger().info(
                    f"  Straight OK | viajado={self.seg_travelled:.3f} m"
                )
                self._advance()
                return

            if elapsed > self.seg_timeout:
                self.get_logger().warn(
                    f"  ⚠ Straight TIMEOUT ({elapsed:.1f} s) | "
                    f"viajado={self.seg_travelled:.3f}/{self.seg_target_dist:.2f} m — skip"
                )
                self._advance()
                return

            raw_v = self.pid_d.compute(err_dist, dt)
            if 0 < raw_v < V_MIN:
                raw_v = V_MIN
            v = clamp(raw_v, 0.0, V_MAX)

            err_th = wrap(self.seg_target_th - self.th)
            w = clamp(self.pid_h.compute(err_th, dt), -W_MAX, W_MAX)

            self._cmd(v, w)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    global TASK
    cli   = [a for a in sys.argv[1:] if not a.startswith('--')]
    valid = {'square': 'SQUARE', 'waypoints': 'WAYPOINTS'}

    if cli and cli[0].lower() in valid:
        TASK = valid[cli[0].lower()]
    else:
        if cli:
            print(f"[ERROR] Tarea '{cli[0]}' no reconocida. Usando: {TASK}")
        print(f"[INFO] Uso: python3 puzzlebot_pid.py [square|waypoints]")

    print(f"[INFO] Task = {TASK}\n")
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

