#!/usr/bin/env python3
"""PuzzleBot WASD teleop (pulse mode).

Each key sends a movement pulse for PULSE_DURATION seconds, then stops.
"""

import os
import time
import tty
import termios
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

WHEEL_RADIUS  = 0.05154
WHEEL_BASE    = 0.19
FORWARD_SIGN  = -1

LINEAR_SPEED  = 0.30   # [m/s]
ANGULAR_SPEED = 2.0     # [rad/s]
PULSE_DURATION = 0.25  # [s] duration of each key pulse

BANNER = """
PuzzleBot WASD Teleop
  W          Forward (pulse)
  S          Backward (pulse)
  A          Turn left (pulse)
  D          Turn right (pulse)
  Space / X  Stop
  Q          Quit
"""

KEY_BINDINGS = {
    'w': ( LINEAR_SPEED,  0.0),
    's': (-LINEAR_SPEED,  0.0),
    'a': ( 0.0,           ANGULAR_SPEED),
    'd': ( 0.0,          -ANGULAR_SPEED),
    ' ': ( 0.0,           0.0),
    'x': ( 0.0,           0.0),
}

KEY_LABELS = {
    'w': 'Forward',
    's': 'Backward',
    'a': 'Turn left',
    'd': 'Turn right',
    ' ': 'STOP',
    'x': 'STOP',
}


_tty_fd = None

def _open_tty():
    global _tty_fd
    if _tty_fd is None:
        _tty_fd = open('/dev/tty', 'rb', buffering=0)
    return _tty_fd


def get_key(settings):
    fd = _open_tty().fileno()
    tty.setraw(fd)
    key = os.read(fd, 1).decode('utf-8', errors='replace')
    termios.tcsetattr(fd, termios.TCSADRAIN, settings)
    return key


def diff_drive(v: float, omega: float):
    wL = FORWARD_SIGN * (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    wR = FORWARD_SIGN * (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return wL, wR


class TeleopNode(Node):
    def __init__(self):
        super().__init__('puzzlebot_teleop')
        self._pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)
        self._pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)
        self._stop_timer = None
        self._lock = threading.Lock()

    def publish(self, v: float, omega: float):
        wL, wR = diff_drive(v, omega)
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wL), float(wR)
        self._pub_l.publish(ml)
        self._pub_r.publish(mr)

    def stop(self):
        try:
            self.publish(0.0, 0.0)
        except Exception:
            pass

    def pulse(self, v: float, omega: float):
        """Publish a velocity and schedule an automatic stop after PULSE_DURATION."""
        with self._lock:
            if self._stop_timer is not None:
                self._stop_timer.cancel()
            self.publish(v, omega)
            # Immediate stop needs no timer
            if v == 0.0 and omega == 0.0:
                self._stop_timer = None
                return
            self._stop_timer = threading.Timer(PULSE_DURATION, self._timer_stop)
            self._stop_timer.daemon = True
            self._stop_timer.start()

    def _timer_stop(self):
        with self._lock:
            self._stop_timer = None
        self.stop()


def main():
    rclpy.init()
    node = TeleopNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    tty_fd = _open_tty().fileno()
    settings = termios.tcgetattr(tty_fd)
    print(BANNER)

    try:
        while True:
            key = get_key(settings).lower()

            if key == 'q':
                print('\nQuitting...')
                break

            if key in KEY_BINDINGS:
                v, omega = KEY_BINDINGS[key]
                node.pulse(v, omega)
                print(f'\r[{key.upper()}] {KEY_LABELS[key]:<16}', end='', flush=True)
            else:
                print(f'\r[?] Unrecognized key: {repr(key)}    ', end='', flush=True)

    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, settings)
        with node._lock:
            if node._stop_timer is not None:
                node._stop_timer.cancel()
                node._stop_timer = None
        # Publish stop while spin is still active
        for _ in range(10):
            node.stop()
            time.sleep(0.05)
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
