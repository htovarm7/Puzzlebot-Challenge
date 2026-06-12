#!/usr/bin/env python3
"""Step-response PD tuner for the PuzzleBot waypoint controller.

Runs one closed-loop step response (heading or distance) and reports rise
time, overshoot, settling time and steady-state error so you can tune a PD
controller. The loop relies on pose updating from the encoders, so it also
prints odometry diagnostics at startup and traces wL/wR/pose to CSV.

Debugging flags:
  --open-loop   Bypass feedback, send a constant command for a few seconds
                to verify the wheels move and check the encoder sign.
  --flip-left   Negate /VelocityEncL inside the node.
  --flip-right  Negate /VelocityEncR inside the node.
  --swap        Swap left and right encoder readings.

Usage:
  python3 pid_tuner.py heading --open-loop   # check plumbing first
  python3 pid_tuner.py heading 90            # 90 deg step
  python3 pid_tuner.py distance 1.0          # 1.0 m step

Tune heading gains first, then distance.
"""

import sys
import math
import time
import csv
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32


# Robot params (must match pid_controller.py exactly)
WHEEL_RADIUS = 0.048
WHEEL_BASE   = 0.19

# Chassis orientation (see pid_controller.py for details)
FORWARD_SIGN = -1

# Gains under test (edit between runs)
KP_DIST = 0.9
KD_DIST = 0.15
KP_TH   = 2.2
KD_TH   = 0.25

# Saturation
V_MAX        = 0.20
OMEGA_MAX    = 1.2
HEADING_GATE = math.radians(20.0)

# Step / convergence config
CTRL_DT       = 0.05
SETTLE_TOL_TH = math.radians(2.0)
SETTLE_TOL_D  = 0.03
SETTLE_TIME   = 1.0
TIMEOUT       = 15.0
STARTUP_WAIT  = 2.0

# Open-loop test config
OPENLOOP_OMEGA = 0.4    # rad/s for --open-loop heading test
OPENLOOP_V     = 0.10   # m/s   for --open-loop distance test
OPENLOOP_DUR   = 3.0    # seconds


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def unicycle_to_wheels(v, w):
    wl = (v - w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    wr = (v + w * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return wl, wr

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PuzzlebotTuner(Node):

    def __init__(self, mode, step, open_loop, flip_left, flip_right, swap):
        super().__init__("puzzlebot_pid_tuner")
        self.mode = mode
        self.step = step
        self.open_loop = open_loop
        self.flip_left  = flip_left
        self.flip_right = flip_right
        self.swap       = swap

        self.pub_l = self.create_publisher(Float32, "/VelocitySetL", 10)
        self.pub_r = self.create_publisher(Float32, "/VelocitySetR", 10)
        # Encoders publish BEST_EFFORT, so subscribers need the sensor-data QoS
        self.create_subscription(
            Float32, "/VelocityEncL", self._cb_l, qos_profile_sensor_data)
        self.create_subscription(
            Float32, "/VelocityEncR", self._cb_r, qos_profile_sensor_data)

        # Raw encoder values (before sign overrides)
        self._raw_wL = 0.0
        self._raw_wR = 0.0
        self.got_L = False
        self.got_R = False
        self.count_L = 0
        self.count_R = 0

        # Pose
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_th = 0.0

        # Pose at start of test (for runaway detection)
        self.pose_th_at_start = 0.0
        self.pose_xy_at_start = (0.0, 0.0)

        # Targets
        self.target_th = None
        self.target_x  = None
        self.target_y  = None

        self.prev_err    = 0.0
        self.prev_err_th = 0.0

        self.t0 = None
        self._start_real_time = None
        self.last_time = None
        self.last_log  = -1.0
        self.settle_since = None
        self.done = False

        self.peak_err  = 0.0
        self.peak_over = 0.0
        self.t_rise    = None
        self.t_settle  = None

        # CSV
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_openloop" if open_loop else ""
        self.csv_path = f"/tmp/puzzlebot_tuning_{mode}{suffix}_{ts}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["t", "error", "cmd_v", "cmd_w",
                           "wL", "wR", "pose_x", "pose_y", "pose_th_deg"])

        self.timer = self.create_timer(CTRL_DT, self._tick)

        flags = []
        if open_loop:  flags.append("OPEN-LOOP")
        if flip_left:  flags.append("flip-L")
        if flip_right: flags.append("flip-R")
        if swap:       flags.append("swap")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""

        self.get_logger().info(
            f"Tuner mode={mode}  step={step}{flag_str}"
        )
        self.get_logger().info(
            f"  Gains: Kp_d={KP_DIST}  Kd_d={KD_DIST}  "
            f"Kp_th={KP_TH}  Kd_th={KD_TH}"
        )
        self.get_logger().info(
            f"  Waiting {STARTUP_WAIT:.1f}s for encoders to publish..."
        )

    @property
    def wL(self):
        v = self._raw_wR if self.swap else self._raw_wL
        return -v if self.flip_left else v

    @property
    def wR(self):
        v = self._raw_wL if self.swap else self._raw_wR
        return -v if self.flip_right else v

    def _cb_l(self, m):
        self._raw_wL = float(m.data)
        self.got_L = True
        self.count_L += 1

    def _cb_r(self, m):
        self._raw_wR = float(m.data)
        self.got_R = True
        self.count_R += 1

    def _update_odom(self, dt):
        v = FORWARD_SIGN * WHEEL_RADIUS * (self.wR + self.wL) / 2.0
        w = WHEEL_RADIUS * (self.wR - self.wL) / WHEEL_BASE
        th_mid = self.pose_th + 0.5 * w * dt
        self.pose_x += v * math.cos(th_mid) * dt
        self.pose_y += v * math.sin(th_mid) * dt
        self.pose_th = wrap_angle(self.pose_th + w * dt)
        return v, w

    def _cmd(self, v, w):
        v = clamp(v, -V_MAX, V_MAX)
        w = clamp(w, -OMEGA_MAX, OMEGA_MAX)
        wl, wr = unicycle_to_wheels(FORWARD_SIGN * v, w)
        ml, mr = Float32(), Float32()
        ml.data, mr.data = float(wl), float(wr)
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)
        return v, w

    def _stop(self):
        z = Float32(); z.data = 0.0
        self.pub_l.publish(z); self.pub_r.publish(z)

    def _tick(self):
        now = self.get_clock().now().nanoseconds / 1e9

        # one-time init / startup wait
        if self.t0 is None:
            if self._start_real_time is None:
                self._start_real_time = now
                self.last_time = now
                return
            if now - self._start_real_time < STARTUP_WAIT:
                self.last_time = now
                return

            # Encoder diagnostic at end of startup
            self.get_logger().info("--- ENCODER STARTUP DIAGNOSTIC ---")
            self.get_logger().info(
                f"  /VelocityEncL : received {self.count_L} msg  "
                f"(last value = {self._raw_wL:+.4f} rad/s)"
            )
            self.get_logger().info(
                f"  /VelocityEncR : received {self.count_R} msg  "
                f"(last value = {self._raw_wR:+.4f} rad/s)"
            )
            if not self.got_L or not self.got_R:
                self.get_logger().error(
                    "  >>> ONE OR BOTH ENCODER TOPICS ARE NOT PUBLISHING."
                )
                self.get_logger().error(
                    "  >>> Check:  ros2 topic list  |  grep Velocity"
                )
                self.get_logger().error(
                    "  >>> Without encoder data, closed loop CANNOT work."
                )
                if not self.open_loop:
                    self.get_logger().error(
                        "  >>> Aborting. Use --open-loop to test wheels anyway."
                    )
                    self._stop()
                    self.done = True
                    self.timer.cancel()
                    rclpy.shutdown()
                    return
            else:
                self.get_logger().info("  >>> Both encoders OK.")
            self.get_logger().info("----------------------------------")

            # latch target relative to current pose
            self.pose_th_at_start = self.pose_th
            self.pose_xy_at_start = (self.pose_x, self.pose_y)
            if self.mode == "heading":
                self.target_th = wrap_angle(self.pose_th + self.step)
            else:
                self.target_x = self.pose_x + self.step * math.cos(self.pose_th)
                self.target_y = self.pose_y + self.step * math.sin(self.pose_th)
                self.target_th = self.pose_th
            self.t0 = now
            self.last_time = now
            if self.open_loop:
                self.get_logger().info(
                    f"[t=0] OPEN-LOOP test engaged.  "
                    f"Will run {OPENLOOP_DUR:.1f}s then stop."
                )
            else:
                self.get_logger().info(
                    f"[t=0] Closed-loop step engaged.  step={self.step}"
                )
            return

        if self.done:
            self._stop()
            return

        dt = max(1e-3, now - self.last_time)
        self.last_time = now
        t = now - self.t0

        v_meas, w_meas = self._update_odom(dt)

        # OPEN-LOOP MODE: ignore feedback, just send a fixed command
        if self.open_loop:
            if t >= OPENLOOP_DUR:
                self._stop()
                self._open_loop_summary(t)
                return
            if self.mode == "heading":
                v_cmd, w_cmd = 0.0, OPENLOOP_OMEGA
            else:
                v_cmd, w_cmd = OPENLOOP_V, 0.0
            v_real, w_real = self._cmd(v_cmd, w_cmd)

            self.csv.writerow([
                f"{t:.3f}", "0.0", f"{v_real:.4f}", f"{w_real:.4f}",
                f"{self.wL:.4f}", f"{self.wR:.4f}",
                f"{self.pose_x:.4f}", f"{self.pose_y:.4f}",
                f"{math.degrees(self.pose_th):.3f}",
            ])

            if t - self.last_log >= 0.2:
                self.last_log = t
                self.get_logger().info(
                    f"[OL] t={t:4.2f}  cmd v={v_real:+.3f} w={w_real:+.3f}  "
                    f"enc wL={self.wL:+.3f} wR={self.wR:+.3f}  "
                    f"odom v={v_meas:+.3f} w={w_meas:+.3f}  "
                    f"pose=({self.pose_x:+.2f},{self.pose_y:+.2f},"
                    f"{math.degrees(self.pose_th):+6.1f})"
                )
            return

        # CLOSED-LOOP MODE
        if self.mode == "heading":
            err = wrap_angle(self.target_th - self.pose_th)
            d_err = (err - self.prev_err) / dt
            self.prev_err = err
            v_cmd = 0.0
            w_cmd = KP_TH * err + KD_TH * d_err
            settled = abs(err) < SETTLE_TOL_TH
            error_for_log = math.degrees(err)
            error_unit = "deg"
            step_magnitude = math.degrees(self.step)
        else:
            dx = self.target_x - self.pose_x
            dy = self.target_y - self.pose_y
            dist = math.hypot(dx, dy)
            ang_to_goal = math.atan2(dy, dx)
            err_th = wrap_angle(ang_to_goal - self.pose_th)
            err_d = dx * math.cos(self.pose_th) + dy * math.sin(self.pose_th)

            d_err = (err_d - self.prev_err)    / dt
            d_th  = (err_th - self.prev_err_th) / dt
            self.prev_err    = err_d
            self.prev_err_th = err_th

            v_cmd = KP_DIST * err_d + KD_DIST * d_err
            w_cmd = KP_TH   * err_th + KD_TH * d_th
            if abs(err_th) > HEADING_GATE:
                v_cmd *= max(0.0, 1.0 - abs(err_th) / math.pi)
            v_cmd = max(0.0, v_cmd)

            settled = dist < SETTLE_TOL_D
            error_for_log = err_d
            error_unit = "m"
            step_magnitude = self.step

        v_real, w_real = self._cmd(v_cmd, w_cmd)

        self.csv.writerow([
            f"{t:.3f}", f"{error_for_log:.5f}",
            f"{v_real:.4f}", f"{w_real:.4f}",
            f"{self.wL:.4f}", f"{self.wR:.4f}",
            f"{self.pose_x:.4f}", f"{self.pose_y:.4f}",
            f"{math.degrees(self.pose_th):.3f}",
        ])

        # Runaway guard: if we've commanded motion for > 2 s and odometry
        # barely moved, feedback is broken so abort before circling forever.
        if t > 2.0:
            if self.mode == "heading":
                pose_moved = abs(wrap_angle(
                    self.pose_th - self.pose_th_at_start))
                if abs(w_real) > 0.1 and pose_moved < math.radians(5.0):
                    self._abort_runaway(
                        t,
                        f"Commanding w={w_real:+.2f} rad/s for 2 s but "
                        f"pose_th moved only {math.degrees(pose_moved):.2f} deg.",
                    )
                    return
            else:
                dxm = self.pose_x - self.pose_xy_at_start[0]
                dym = self.pose_y - self.pose_xy_at_start[1]
                pose_moved = math.hypot(dxm, dym)
                if abs(v_real) > 0.05 and pose_moved < 0.05:
                    self._abort_runaway(
                        t,
                        f"Commanding v={v_real:+.2f} m/s for 2 s but "
                        f"pose moved only {pose_moved*100:.1f} cm.",
                    )
                    return

        # metrics
        if abs(error_for_log) > self.peak_err:
            self.peak_err = abs(error_for_log)
        if step_magnitude > 0 and error_for_log < -self.peak_over:
            self.peak_over = -error_for_log
        elif step_magnitude < 0 and error_for_log > self.peak_over:
            self.peak_over = error_for_log
        if self.t_rise is None and abs(error_for_log) <= 0.1 * abs(step_magnitude):
            self.t_rise = t

        if settled:
            if self.settle_since is None:
                self.settle_since = t
            elif (t - self.settle_since) >= SETTLE_TIME:
                self.t_settle = self.settle_since
                self._finish(t, error_for_log, error_unit, step_magnitude,
                             reason="SETTLED")
                return
        else:
            self.settle_since = None

        if t > TIMEOUT:
            self._finish(t, error_for_log, error_unit, step_magnitude,
                         reason="TIMEOUT")
            return

        if t - self.last_log >= 0.2:
            self.last_log = t
            self.get_logger().info(
                f"t={t:5.2f}  err={error_for_log:+7.3f}{error_unit}  "
                f"cmd v={v_real:+.3f} w={w_real:+.3f}  "
                f"enc wL={self.wL:+.2f} wR={self.wR:+.2f}  "
                f"pose=({self.pose_x:+.2f},{self.pose_y:+.2f},"
                f"{math.degrees(self.pose_th):+6.1f})"
            )

    def _open_loop_summary(self, t):
        self.done = True
        self.csv_file.flush(); self.csv_file.close()
        moved_th = math.degrees(wrap_angle(
            self.pose_th - self.pose_th_at_start))
        dx = self.pose_x - self.pose_xy_at_start[0]
        dy = self.pose_y - self.pose_xy_at_start[1]
        moved_d = math.hypot(dx, dy)

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"OPEN-LOOP TEST COMPLETE  ({t:.2f} s)")
        if self.mode == "heading":
            expected_th = math.degrees(OPENLOOP_OMEGA * OPENLOOP_DUR)
            self.get_logger().info(
                f"  commanded omega : +{OPENLOOP_OMEGA:.2f} rad/s "
                f"for {OPENLOOP_DUR:.1f}s")
            self.get_logger().info(
                f"  expected pose_th change : ~+{expected_th:.0f} deg (left turn)")
            self.get_logger().info(
                f"  actual pose_th change   :  {moved_th:+.1f} deg")
            self.get_logger().info(
                f"  last enc values         : wL={self.wL:+.3f}  wR={self.wR:+.3f}"
                f"  (after flip/swap flags)")
            self.get_logger().info("")
            self.get_logger().info("Interpretation:")
            self.get_logger().info("  - If robot DID rotate left physically:")
            if abs(moved_th) < 5.0:
                self.get_logger().info(
                    "      .. but pose_th barely changed -> encoder signs wrong.")
                self.get_logger().info(
                    "         Try: --flip-left  OR  --flip-right  OR  --swap")
            elif moved_th < -10.0:
                self.get_logger().info(
                    "      .. but pose_th DECREASED -> sign convention inverted.")
                self.get_logger().info(
                    "         Try: --flip-left AND --flip-right (negate both)")
            else:
                self.get_logger().info(
                    "      .. and pose_th increased correctly. Odometry is GOOD.")
                self.get_logger().info(
                    "         Re-run without --open-loop to start tuning.")
            self.get_logger().info("  - If robot did NOT rotate physically:")
            self.get_logger().info(
                "      .. check wheel power, motor topics, robot/sim state.")
        else:
            expected_d = OPENLOOP_V * OPENLOOP_DUR
            self.get_logger().info(
                f"  commanded v : +{OPENLOOP_V:.2f} m/s for {OPENLOOP_DUR:.1f}s")
            self.get_logger().info(f"  expected travel : ~{expected_d:.2f} m")
            self.get_logger().info(f"  actual travel   :  {moved_d:.3f} m")
            self.get_logger().info(
                f"  last enc        : wL={self.wL:+.3f}  wR={self.wR:+.3f}")
            self.get_logger().info("")
            if moved_d < 0.05:
                self.get_logger().info(
                    "  -> pose did not advance: encoder signs likely wrong.")
                self.get_logger().info(
                    "     If one encoder is negative during forward motion,")
                self.get_logger().info(
                    "     use --flip-left or --flip-right.")
            elif abs(moved_d - expected_d) / expected_d > 0.3:
                self.get_logger().info(
                    "  -> distance off by >30%: check WHEEL_RADIUS calibration.")
            else:
                self.get_logger().info(
                    "  -> odometry tracks well. Re-run without --open-loop.")
        self.get_logger().info(f"  csv trace : {self.csv_path}")
        self.get_logger().info("=" * 60)
        self.timer.cancel()

    def _abort_runaway(self, t, msg):
        self.done = True
        self._stop()
        self.csv_file.flush(); self.csv_file.close()
        self.get_logger().error("=" * 60)
        self.get_logger().error("ABORT: RUNAWAY DETECTED (closed loop not closing)")
        self.get_logger().error(f"  {msg}")
        self.get_logger().error("")
        self.get_logger().error("This almost always means encoder feedback is broken.")
        self.get_logger().error("Diagnose with:")
        self.get_logger().error(
            f"  python3 {sys.argv[0]} {self.mode} --open-loop")
        self.get_logger().error("")
        self.get_logger().error("Check in another terminal:")
        self.get_logger().error("  ros2 topic echo /VelocityEncL --once")
        self.get_logger().error("  ros2 topic echo /VelocityEncR --once")
        self.get_logger().error("  ros2 topic hz   /VelocityEncL")
        self.get_logger().error(f"  csv trace : {self.csv_path}")
        self.get_logger().error("=" * 60)
        self.timer.cancel()

    def _finish(self, t_end, final_err, unit, step_magnitude, reason):
        self.done = True
        self._stop()
        self.csv_file.flush(); self.csv_file.close()
        overshoot_pct = (self.peak_over / abs(step_magnitude) * 100.0
                         if step_magnitude != 0 else 0.0)
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"RUN COMPLETE  reason={reason}")
        self.get_logger().info(f"  mode           : {self.mode}")
        self.get_logger().info(f"  step           : {step_magnitude:+.3f} {unit}")
        self.get_logger().info(
            f"  gains          : Kp_d={KP_DIST} Kd_d={KD_DIST} "
            f"Kp_th={KP_TH} Kd_th={KD_TH}")
        self.get_logger().info(
            f"  rise time (10%): {self.t_rise:.3f} s"
            if self.t_rise else "  rise time      : n/a")
        self.get_logger().info(
            f"  settling time  : {self.t_settle:.3f} s"
            if self.t_settle else "  settling time  : NOT SETTLED")
        self.get_logger().info(
            f"  overshoot      : {self.peak_over:+.4f} {unit}  "
            f"({overshoot_pct:.1f}%)")
        self.get_logger().info(f"  steady-state e : {final_err:+.4f} {unit}")
        self.get_logger().info(f"  total time     : {t_end:.3f} s")
        self.get_logger().info(f"  csv trace      : {self.csv_path}")
        self.get_logger().info("=" * 60)
        self._verdict(overshoot_pct, final_err, unit, step_magnitude)
        self.timer.cancel()

    def _verdict(self, overshoot_pct, ss_err, unit, step_magnitude):
        tips = []
        if self.t_settle is None:
            if abs(ss_err - step_magnitude) < 0.1 * abs(step_magnitude):
                tips.append(
                    "Error stayed near the FULL STEP value -- pose isn't "
                    "updating from encoders. Run with --open-loop and "
                    "fix odometry/signs before tuning."
                )
            else:
                tips.append("Did not settle: lower KP, OR raise KD, OR widen tolerance.")
        if overshoot_pct > 15:
            tips.append("High overshoot: raise KD or lower KP.")
        elif overshoot_pct < 1 and self.t_rise and self.t_rise > 1.0:
            tips.append("Sluggish, no overshoot: raise KP.")
        if abs(ss_err) > (math.degrees(SETTLE_TOL_TH) if unit == "deg"
                          else SETTLE_TOL_D):
            tips.append(
                f"Steady-state error {ss_err:+.3f}{unit}: "
                f"check WHEEL_RADIUS / WHEEL_BASE calibration before adding I."
            )
        if self.t_rise and self.t_rise < 0.15:
            tips.append("Very fast rise -- check for wheel slip on this surface.")
        if not tips:
            tips.append("Looks good. Try the full waypoint run with these gains.")
        self.get_logger().info("Suggested next step:")
        for s in tips:
            self.get_logger().info(f"  - {s}")


def main(args=None):
    raw = sys.argv[1:]
    cli = [a for a in raw if not a.startswith("--ros-args") and a != "--"]

    flags = {a for a in cli if a.startswith("--")}
    positional = [a for a in cli if not a.startswith("--")]

    if not positional or positional[0] not in ("heading", "distance"):
        print("Usage:")
        print("  python3 puzzlebot_pid_tuner.py heading  [deg]  "
              "[--open-loop] [--flip-left] [--flip-right] [--swap]")
        print("  python3 puzzlebot_pid_tuner.py distance [m]    "
              "[--open-loop] [--flip-left] [--flip-right] [--swap]")
        print()
        print("Recommended first run if odometry is unverified:")
        print("  python3 puzzlebot_pid_tuner.py heading --open-loop")
        return

    mode = positional[0]
    if mode == "heading":
        deg = float(positional[1]) if len(positional) > 1 else 90.0
        step = math.radians(deg)
    else:
        step = float(positional[1]) if len(positional) > 1 else 1.0

    rclpy.init(args=args)
    node = PuzzlebotTuner(
        mode=mode,
        step=step,
        open_loop="--open-loop" in flags,
        flip_left="--flip-left" in flags,
        flip_right="--flip-right" in flags,
        swap="--swap" in flags,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted, stopping motors.")
        node._stop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()