# Puzzlebot Challenge

ROS2 (Humble) workspace for Manchester Robotics' **PuzzleBot**, running on a Jetson
Nano with a CSI PiCam (IMX219). It implements an autonomous driving stack that
follows a line, reacts to traffic signs and obeys traffic lights.

Main components:

- **`picam_publisher`** — publishes PiCam frames on `/camera/image_raw`.
- **`cam_server`** — serves the video as MJPEG over HTTP (view it in a browser).
- **`line_detector` / `line_follower`** — detect the line and steer the robot along it.
- **`sign_detector` / `sign_behavior_controller`** — YOLO-based traffic-sign detection and reactive behaviors (stop, turns, workers, go straight).
- **`traffic_controller`** — detects traffic-light color (red / yellow / green).
- **`pid_controller` / `pid_tuner`** — closed-loop PID for `SQUARE` / `WAYPOINTS` trajectories, plus an interactive live-tuning tool.
- **`motor_watchdog`** — safety layer between the controllers and the motors.

---

## Requirements

- **ROS2 Humble** installed (`/opt/ros/humble`).
- Jetson Nano with JetPack ≥ 4.6 (`nvarguscamerasrc` drivers).
- Ethernet cable between the Jetson and your PC, or both on the same network.

### System dependencies

```bash
sudo apt update
sudo apt install -y \
  ros-humble-cv-bridge \
  python3-opencv \
  python3-pip \
  python3-colcon-common-extensions
pip3 install flask
```

`python3-opencv` on the Jetson ships with GStreamer support (required for the CSI PiCam).

### Sign detection (YOLO)

The sign detector uses [Ultralytics](https://github.com/ultralytics/ultralytics):

```bash
pip3 install -r requirements.txt   # ultralytics
```

Model weights live in `src/puzzlebot_challenge/utils/` (`best.pt`, `best.onnx`,
`best.engine`). On the Jetson, the TensorRT `.engine` gives the best inference
speed; see [`utils/export_trt.py`](src/puzzlebot_challenge/utils/export_trt.py) to
regenerate it. Training material is under [`training/`](training/).

---

## Build

```bash
git clone <repo-url> ~/Puzzlebot-Challenge
cd ~/Puzzlebot-Challenge

# (Optional) install dependencies declared in package.xml:
rosdep install --from-paths src -y --ignore-src

# Build the workspace
colcon build --symlink-install

# Load the overlay
source install/setup.bash
```

Add the `source` to your `~/.bashrc` so you don't repeat it in every terminal:

```bash
echo "source ~/Puzzlebot-Challenge/install/setup.bash" >> ~/.bashrc
```

---

## Usage

### Camera + web server (quick visualization)

```bash
ros2 launch puzzlebot_challenge camera.launch.py
```

Open on your PC: <http://JETSON-IP:8080>

### Line following (autonomous)

```bash
ros2 launch puzzlebot_challenge line_follow.launch.py
```

Optional parameters:

```bash
ros2 launch puzzlebot_challenge line_follow.launch.py kp:=0.3 kd:=0.08 v_base:=0.15
```

To run only the vision stack (detection, no movement):

```bash
ros2 launch puzzlebot_challenge line.launch.py
```

### Full stack: line + signs + traffic lights

```bash
ros2 launch puzzlebot_challenge signs.launch.py
```

Useful parameters:

```bash
ros2 launch puzzlebot_challenge signs.launch.py \
  conf_threshold:=0.50 imgsz:=320 \
  v_base:=0.12 turn_time:=1.8 sign_cooldown:=4.0
```

| Parameter        | Default | Meaning                                       |
|------------------|---------|-----------------------------------------------|
| `kp`, `kd`, `ka` | 0.3 / 0.08 / 0.2 | Line-follow P, D and angle-correction gains |
| `v_base`         | 0.12    | Base speed [m/s]                              |
| `give_way_time`  | 2.0     | Stop duration for *give way* [s]              |
| `stop_hold_time` | 1.0     | Hold after a *stop* sign disappears [s]       |
| `workers_factor` | 0.5     | Speed factor in *workers* zones               |
| `turn_time`      | 1.8     | Turn duration [s]                             |
| `turn_omega`     | 0.7     | Turn angular speed [rad/s]                    |
| `straight_time`  | 3.0     | *go straight* override duration [s]           |
| `sign_cooldown`  | 4.0     | Cooldown between identical signs [s]          |
| `conf_threshold` | 0.50    | YOLO confidence threshold (0–1)               |
| `imgsz`          | 320     | YOLO inference image size                     |

The `final.launch.py` variant runs the same behavior using the line-follower
controller without the standalone YOLO detector node.

### Teleop (manual control)

In one terminal, launch the vision stack:

```bash
ros2 launch puzzlebot_challenge line.launch.py
```

In another terminal, launch teleop:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

> **Note:** teleop publishes on `/cmd_vel`. Make sure `motor_watchdog` isn't running
> at the same time, or the commands will conflict.

---

### PID tuning

```bash
ros2 run puzzlebot_challenge pid_tuner straight   # or: turn
```

In another terminal, adjust gains on the fly:

```bash
ros2 param set /pid_tuner kp_dist 1.5
ros2 param set /pid_tuner kd_head 0.25
```

Plot metrics with `rqt_plot`:

```bash
ros2 run rqt_plot rqt_plot /tuner/error_dist /tuner/cmd_v
```

---

## Configuration

Parameters live in `src/puzzlebot_challenge/config/`:

- **`camera.yaml`** — resolution, FPS, topic, MJPEG port.
- **`line_params.yaml`** — line-detection thresholds, ROI, morphology, etc.

Edit the YAML files and relaunch — no `colcon build` needed if you used
`--symlink-install`.

Calibration helpers:

```bash
ros2 run puzzlebot_challenge line_calibrator    # tune line thresholds live
```

---

## Relevant topics

| Topic                 | Type                  | Published by        | Consumed by          |
|-----------------------|-----------------------|---------------------|----------------------|
| `/camera/image_raw`   | `sensor_msgs/Image`   | `picam_publisher`   | `line_detector`, `sign_detector`, `cam_server` |
| `/line/shift`         | `std_msgs/Float32`    | `line_detector`     | `line_follower`      |
| `/line/detected`      | `std_msgs/Bool`       | `line_detector`     | `line_follower`      |
| `/vision/line`        | `sensor_msgs/Image`   | `line_detector`     | `line_viewer`, `cam_server` |
| `/sign/command`       | `std_msgs/String`     | `sign_detector`     | `sign_behavior_controller` |
| `/sign/detected`      | `std_msgs/Bool`       | `sign_detector`     | `sign_behavior_controller` |
| `/vision/signs`       | `sensor_msgs/Image`   | `sign_detector`     | `sign_viewer`        |
| `/traffic_light`      | `std_msgs/String`     | `traffic_controller`| `sign_behavior_controller` |
| `/vision/traffic`     | `sensor_msgs/Image`   | `traffic_controller`| viewer / `cam_server`|
| `/cmd/VelocitySetL/R` | `std_msgs/Float32`    | `line_follower`     | `motor_watchdog`     |
| `/VelocitySetL/R`     | `std_msgs/Float32`    | `motor_watchdog`    | hardware             |
| `/VelEncL`, `/VelEncR`| `std_msgs/Float32`    | hardware            | `pid_controller`     |
| `/tuner/*`            | `std_msgs/Float32`    | `pid_tuner`         | `rqt_plot`           |

`/sign/command` values: `stop`, `go_straight`, `turn_left`, `turn_right`,
`workers`, `none`. `/traffic_light` values: `red`, `yellow`, `green`.

---

## Repository structure

```
Puzzlebot-Challenge/
├── src/
│   └── puzzlebot_challenge/
│       ├── puzzlebot_challenge/   # Python nodes
│       │   ├── camera/            # picam_publisher, cam_server
│       │   ├── line/              # line_detector, line_follower, viewer, calibrator
│       │   ├── signs/             # YOLO sign detector + behavior controller
│       │   ├── traffic/           # traffic-light detection (HSV / circle)
│       │   └── control/           # pid_controller, pid_tuner, motor_watchdog, teleop
│       ├── launch/                # .launch.py files
│       ├── config/                # YAML parameters
│       ├── utils/                 # YOLO model weights + TensorRT export
│       ├── package.xml
│       └── setup.py
├── scripts/                       # non-ROS utilities (quick tests, calibration, recording)
├── training/                      # YOLO sign-detection training (notebook + requirements)
├── docs/                          # extra documentation and reference images
└── README.md
```

More details in [`docs/SETUP.md`](docs/SETUP.md).
