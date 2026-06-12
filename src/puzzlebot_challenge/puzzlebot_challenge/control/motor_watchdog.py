#!/usr/bin/env python3
"""Safety watchdog for the PuzzleBot motors.

Bridges the control nodes and the motors: if no command arrives within
WATCHDOG_TIMEOUT seconds it publishes zero on both wheels, so the motors stop
even if the control node dies abruptly.
"""

import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

WATCHDOG_TIMEOUT = 0.5   # [s] time without commands before stopping motors
TIMER_HZ         = 20    # watchdog check frequency


class MotorWatchdogNode(Node):

    def __init__(self):
        super().__init__('motor_watchdog')

        self._pub_l = self.create_publisher(Float32, '/VelocitySetL', 10)
        self._pub_r = self.create_publisher(Float32, '/VelocitySetR', 10)

        self.create_subscription(Float32, '/VelocitySetL', self._cb_l, 10)
        self.create_subscription(Float32, '/VelocitySetR', self._cb_r, 10)

        self._last_l = 0.0
        self._last_r = 0.0
        self._last_cmd_time = time.monotonic()
        self._timed_out = False

        self.create_timer(1.0 / TIMER_HZ, self._watchdog_tick)
        self.get_logger().info(
            f'MotorWatchdog ready, timeout={WATCHDOG_TIMEOUT}s')

    def _cb_l(self, msg: Float32):
        self._last_l = msg.data
        self._last_cmd_time = time.monotonic()
        self._timed_out = False

    def _cb_r(self, msg: Float32):
        self._last_r = msg.data
        self._last_cmd_time = time.monotonic()
        self._timed_out = False

    def _watchdog_tick(self):
        elapsed = time.monotonic() - self._last_cmd_time

        if elapsed > WATCHDOG_TIMEOUT:
            if not self._timed_out:
                self.get_logger().warn(
                    f'No commands for {elapsed:.2f}s, stopping motors.')
                self._timed_out = True
            self._publish(0.0, 0.0)
        else:
            self._publish(self._last_l, self._last_r)

    def _publish(self, wl: float, wr: float):
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wl), float(wr)
        self._pub_l.publish(ml)
        self._pub_r.publish(mr)

    def emergency_stop(self):
        try:
            self._publish(0.0, 0.0)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MotorWatchdogNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.emergency_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
