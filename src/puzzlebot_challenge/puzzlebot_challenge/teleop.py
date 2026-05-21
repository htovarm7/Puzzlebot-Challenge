#!/usr/bin/env python3
"""
teleop.py
=========
ROS2 node for teleoperation of the PuzzleBot via WASD keys in terminal.

Controls
--------
  W : move forward
  S : move backward
  A : turn left  (counter-clockwise)
  D : turn right (clockwise)
  Space / X : emergency stop (zero velocities)
  Q : quit

Topics
------
Publishers : /VelocitySetL  (std_msgs/Float32)  rad/s
             /VelocitySetR  (std_msgs/Float32)  rad/s

The robot uses differential drive:
    wL = (v - omega * B/2) / R
    wR = (v + omega * B/2) / R
where R = wheel radius, B = wheel base.
"""

import sys
import tty
import termios
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# ── Robot physical parameters (must match pid_controller.py) ─────────────────
WHEEL_RADIUS  = 0.05154   # [m]
WHEEL_BASE    = 0.19      # [m]
FORWARD_SIGN  = -1        # -1 because chassis is mounted reversed

# ── Teleop speed settings ────────────────────────────────────────────────────
LINEAR_SPEED  = 0.15      # [m/s]   forward / backward
ANGULAR_SPEED = 0.8       # [rad/s] turning

BANNER = """
╔══════════════════════════════════╗
║   PuzzleBot WASD Teleop          ║
╠══════════════════════════════════╣
║  W  →  Adelante                  ║
║  S  →  Atrás                     ║
║  A  →  Girar izquierda           ║
║  D  →  Girar derecha             ║
║  Space / X  →  Parar             ║
║  Q  →  Salir                     ║
╚══════════════════════════════════╝
"""

KEY_BINDINGS = {
    'w': ( LINEAR_SPEED,  0.0),
    's': (-LINEAR_SPEED,  0.0),
    'a': ( 0.0,           ANGULAR_SPEED),
    'd': ( 0.0,          -ANGULAR_SPEED),
    ' ': ( 0.0,           0.0),
    'x': ( 0.0,           0.0),
}


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def diff_drive(v: float, omega: float):
    """Convert (v, omega) → (wL, wR) in rad/s."""
    wL = FORWARD_SIGN * (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    wR = FORWARD_SIGN * (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return wL, wR


class TeleopNode(Node):
    def __init__(self):
        super().__init__('puzzlebot_teleop')
        self._pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)
        self._pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)
        self.get_logger().info('Teleop node ready.')

    def publish(self, v: float, omega: float):
        wL, wR = diff_drive(v, omega)
        msg_l, msg_r = Float32(), Float32()
        msg_l.data = float(wL)
        msg_r.data = float(wR)
        self._pub_l.publish(msg_l)
        self._pub_r.publish(msg_r)

    def stop(self):
        self.publish(0.0, 0.0)


def main():
    rclpy.init()
    node = TeleopNode()

    # Spin ROS in background so publishers stay alive
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    settings = termios.tcgetattr(sys.stdin)
    print(BANNER)

    try:
        while True:
            key = get_key(settings).lower()

            if key == 'q':
                print('\nSaliendo...')
                break

            if key in KEY_BINDINGS:
                v, omega = KEY_BINDINGS[key]
                node.publish(v, omega)
                action = {
                    'w': 'Adelante',
                    's': 'Atrás',
                    'a': 'Giro izq.',
                    'd': 'Giro der.',
                    ' ': 'STOP',
                    'x': 'STOP',
                }[key]
                print(f'\r[{key.upper()}] {action} — v={v:.2f} m/s  ω={omega:.2f} rad/s    ', end='', flush=True)
            else:
                print(f'\r[?] Tecla desconocida: {repr(key)}    ', end='', flush=True)

    except KeyboardInterrupt:
        print('\nInterrumpido por Ctrl+C')
    finally:
        node.stop()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
