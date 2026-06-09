#!/usr/bin/env python3
"""
sign_behavior_controller.py
============================
Intercepta las velocidades del line_follower y aplica comportamientos
basados en las señales de tránsito detectadas por YOLO.

** SIN DETECTOR DE INTERSECCIÓN — DISPARO POR TEMPORIZADOR **
El giro ya NO depende de /intersection/stop ni del estado de la línea. Lógica:
  cuando se detecta una señal de giro/recto, el robot sigue la línea con
  normalidad durante arm_delay segundos (por defecto 2 s) y luego ejecuta la
  maniobra hardcodeada (izquierda / derecha / recto), IGNORANDO por completo
  si ve o no la línea. Si la línea se pierde durante la cuenta, avanza recto
  para no detenerse.

Arquitectura de tópicos:
  line_follower  → /line/VelocitySetL, /line/VelocitySetR   (remapeado en launch)
  line_detector  → /line/detected
  sign_detector  → /sign/command, /sign/detected
  este nodo      → /VelocitySetL, /VelocitySetR              (salida final)

Comportamientos:
  give_way    → sigue la línea mientras ve la señal; al perderla, para 2 s y continúa
  stop        → detenerse mientras la señal esté visible + STOP_HOLD_TIME s después
  workers     → reducir velocidad al WORKERS_FACTOR mientras la señal esté visible
  turn_left   → arm_delay s tras detectar la señal, gira a la izquierda
  turn_right  → arm_delay s tras detectar la señal, gira a la derecha
  go_straight → arm_delay s tras detectar la señal, avanza recto
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool, String, Empty

# ── Geometría del robot ───────────────────────────────────────────────────────
WHEEL_RADIUS = 0.05154
WHEEL_BASE   = 0.19
FORWARD_SIGN = -1   # velocidad de rueda positiva = avance

# ── Parámetros por defecto ────────────────────────────────────────────────────
GIVE_WAY_STOP_TIME = 2.0   # s — duración de la parada en give_way
STOP_HOLD_TIME     = 1.0   # s — espera extra tras desaparecer el stop
WORKERS_FACTOR     = 0.5   # factor de velocidad con señal de workers
APPROACH_TIME      = 0.4   # s — avance recto antes del giro
TURN_TIME          = 3.5  # s — duración del giro
TURN_OMEGA         = 0.7   # rad/s — velocidad angular del giro
TURN_V             = 0.06  # m/s — velocidad lineal durante el giro
STRAIGHT_TIME      = 3.0   # s — duración del override recto (go_straight)
STRAIGHT_V         = 0.12  # m/s — velocidad durante el override recto
SIGN_COOLDOWN      = 4.0   # s — cooldown antes de re-disparar el mismo comando
ARM_DELAY          = 2.0   # s — espera tras detectar la señal antes de ejecutar la maniobra
CTRL_DT            = 0.05  # s — ciclo del bucle de control (20 Hz)

# ── Identificadores de estado ─────────────────────────────────────────────────
S_IDLE              = "IDLE"
S_PENDING_GIVE_WAY  = "PENDING_GIVE_WAY"   # sigue línea mientras ve la señal
S_GIVE_WAY          = "GIVE_WAY"           # para 2 s tras perder la señal
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

        # /intersection/stop ya NO se usa — ahora escuchamos /line/detected
        self.create_subscription(String,  "/sign/command",      self._cb_command,       10)
        self.create_subscription(Bool,    "/sign/detected",     self._cb_detected,      10)
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
        self._prev_detected = False
        self._line_detected = False
        self._line_vel_l    = 0.0
        self._line_vel_r    = 0.0
        self._last_trigger  = {}     # cmd → timestamp del último disparo

        _wait = self.get_parameter("wait_for_start").value
        _wait_bool = _wait if isinstance(_wait, bool) else str(_wait).lower() not in ("false", "0", "no")
        self._sign_ready = not _wait_bool

        self.create_timer(CTRL_DT, self._control_loop)
        self.get_logger().info(
            "SignBehaviorController listo — SIN intersección: la maniobra se dispara "
            "arm_delay s después de detectar la señal (ignora el estado de la línea)")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb_start(self, _msg: Empty):
        if not self._sign_ready:
            self._sign_ready = True
            self.get_logger().info("[SignBehavior] /robot/start recibido — robot en marcha")

    def _cb_command(self, msg: String):
        self._sign_command = msg.data.lower()

    def _cb_detected(self, msg: Bool):
        self._sign_detected = bool(msg.data)

    def _cb_line_detected(self, msg: Bool):
        self._line_detected = bool(msg.data)

    def _cb_vel_l(self, msg: Float32):
        self._line_vel_l = float(msg.data)

    def _cb_vel_r(self, msg: Float32):
        self._line_vel_r = float(msg.data)

    # ── Helpers de publicación ────────────────────────────────────────────────

    def _publish(self, wl: float, wr: float):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wl), float(wr)
        self._pub_l.publish(ml)
        self._pub_r.publish(mr)

    def _stop(self):
        self._publish(0.0, 0.0)

    def _passthrough(self):
        """Reenvía las velocidades del line_follower sin modificar."""
        self._publish(self._line_vel_l, self._line_vel_r)

    # ── Transición de estado ──────────────────────────────────────────────────

    _STATE_MSG = {
        S_PENDING_GIVE_WAY: "GIVE WAY detectado — acercándose",
        S_GIVE_WAY:         "GIVE WAY — deteniéndose",
        S_STOP:             "STOP — detenido",
        S_STOP_HOLD:        "STOP — hold",
        S_WORKERS:          "WORKERS — velocidad reducida",
        S_PENDING_LEFT:     "TURN LEFT detectado — esperando arm_delay s antes de girar",
        S_PENDING_RIGHT:    "TURN RIGHT detectado — esperando arm_delay s antes de girar",
        S_PENDING_STRAIGHT: "GO STRAIGHT detectado — esperando arm_delay s antes de recto",
        S_APPROACH_LEFT:    "TURN LEFT — tramo recto previo",
        S_APPROACH_RIGHT:   "TURN RIGHT — tramo recto previo",
        S_TURNING_LEFT:     "TURN LEFT — girando",
        S_TURNING_RIGHT:    "TURN RIGHT — girando",
        S_GOING_STRAIGHT:   "GO STRAIGHT — avanzando recto",
        S_IDLE:             "IDLE — siguiendo línea",
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

    # ── Bucle de control principal (20 Hz) ────────────────────────────────────

    def _control_loop(self):
        if not self._sign_ready:
            self._stop()
            return

        elapsed  = self._now() - self._state_start
        cmd      = self._sign_command
        detected = self._sign_detected

        rising  = detected and not self._prev_detected   # señal aparece
        falling = not detected and self._prev_detected   # señal desaparece
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

        # ── IDLE: espera nuevo comando ─────────────────────────────────────
        if self._state == S_IDLE:
            if rising and not self._in_cooldown(cmd):
                if cmd == "give_way":
                    self._enter(S_PENDING_GIVE_WAY, cmd)   # acercarse primero
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

        # ── PENDING_GIVE_WAY: sigue línea mientras ve la señal ─────────────
        elif self._state == S_PENDING_GIVE_WAY:
            if falling:
                self._enter(S_GIVE_WAY)   # perdió la señal → para 2 s
            else:
                self._passthrough()

        # ── GIVE_WAY: para 2 s tras perder la señal, luego continúa ───────
        elif self._state == S_GIVE_WAY:
            if elapsed < gw_time:
                self._stop()
            else:
                self._enter(S_IDLE)

        # ── STOP: para mientras la señal esté visible ──────────────────────
        elif self._state == S_STOP:
            if not detected:
                self._enter(S_STOP_HOLD)
            else:
                self._stop()

        # ── STOP_HOLD: breve pausa extra tras desaparecer el stop ──────────
        elif self._state == S_STOP_HOLD:
            if elapsed < sh_time:
                self._stop()
            else:
                self._enter(S_IDLE)

        # ── WORKERS: reduce velocidad mientras la señal esté visible ──────
        elif self._state == S_WORKERS:
            if detected:
                self._publish(self._line_vel_l * wk_fact,
                              self._line_vel_r * wk_fact)
            else:
                self._enter(S_IDLE)
                self._passthrough()

        # ── PENDING_*: temporizador. Tras detectar la señal sigue la línea con
        #    normalidad; cuando pasan arm_delay s ejecuta la maniobra, IGNORANDO
        #    si ve o no la línea. Si la línea se pierde durante la cuenta, avanza
        #    recto para no detenerse.
        elif self._state in (S_PENDING_LEFT, S_PENDING_RIGHT, S_PENDING_STRAIGHT):
            if elapsed >= arm_delay:
                if self._state == S_PENDING_LEFT:
                    self._enter(S_APPROACH_LEFT)
                elif self._state == S_PENDING_RIGHT:
                    self._enter(S_APPROACH_RIGHT)
                else:
                    self._enter(S_GOING_STRAIGHT)
            elif self._line_detected:
                self._passthrough()                       # sigue la línea durante la cuenta
            else:
                # línea perdida durante la cuenta: avanza recto, NO te pares
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, 0.0)
                self._publish(vl, vr)

        # ── APPROACH_LEFT / APPROACH_RIGHT: tramo recto previo al giro ─────
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

        # ── TURNING_LEFT ───────────────────────────────────────────────────
        elif self._state == S_TURNING_LEFT:
            if elapsed < t_time:
                omega = FORWARD_SIGN * t_omg          # omega positivo = giro izquierda
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, omega)
                self._publish(vl, vr)
            else:
                self._enter(S_IDLE)

        # ── TURNING_RIGHT ──────────────────────────────────────────────────
        elif self._state == S_TURNING_RIGHT:
            if elapsed < t_time:
                omega = -(FORWARD_SIGN * t_omg)       # omega negativo = giro derecha
                vl, vr = unicycle_to_wheels(FORWARD_SIGN * t_v, omega)
                self._publish(vl, vr)
            else:
                self._enter(S_IDLE)

        # ── GOING_STRAIGHT: avanza recto ignorando el line_follower ────────
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