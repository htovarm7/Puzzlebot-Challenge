# Detailed setup — PuzzleBot Jetson

This guide covers what the `README` doesn't detail: connecting to the Jetson,
networking, CSI camera troubleshooting and optional services.

---

## 1. Connecting to the Jetson

### Direct Ethernet (PC ↔ Jetson)

If you connect the Jetson directly to your PC with an Ethernet cable and share the
connection from NetworkManager (Settings → Network → Ethernet → IPv4 →
"Shared to other computers"), the Jetson will get an IP in `10.42.0.0/24`.

To find it from the PC:

```bash
ip neigh show dev <ethernet-interface>
```

Connect over SSH (the PuzzleBot default user is `puzzlebot`):

```bash
ssh puzzlebot@10.42.0.X
```

### Hostname / MagicDNS

- To change the system name: `sudo hostnamectl set-hostname jetson-X` and edit the
  `127.0.1.1` line in `/etc/hosts`.
- If you use Tailscale, edit the name from
  <https://login.tailscale.com/admin/machines> or with
  `sudo tailscale set --hostname=jetson-X`.

---

## 2. CSI camera (IMX219)

### Check that the kernel detects it

```bash
v4l2-ctl --list-devices
# vi-output, imx219 8-0010 (platform:54080000.vi:4):
#     /dev/video0
```

### Quick test without a GUI

```bash
gst-launch-1.0 -e nvarguscamerasrc num-buffers=10 sensor-mode=4 ! \
  "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1" ! \
  nvjpegenc ! multifilesink location=frame_%03d.jpg
```

It should generate 10 JPGs. If it fails with `Failed to create CaptureSession`:

1. Restart the daemon: `sudo systemctl restart nvargus-daemon`.
2. If it persists after a bad kill, **reboot the Jetson**: `sudo reboot`.
3. **Never use `kill -9` on `nvargus-daemon`** — it leaves the sensor in a bad
   state that only recovers with a reboot.

### IMX219 sensor modes

| sensor-mode | Resolution  | FPS  |
|-------------|-------------|------|
| 0           | 3264×2464   | 21   |
| 1           | 3264×1848   | 28   |
| 2           | 1920×1080   | 30   |
| 3           | 1640×1232   | 30   |
| 4           | 1280×720    | 60   |
| 5           | 1280×720    | 120  |

The workspace default is `sensor_mode: 3` (1640×1232 @ 30fps), scaled in hardware
to 320×240 for processing. Change it in
[`config/camera.yaml`](../src/puzzlebot_challenge/config/camera.yaml).

---

## 3. Viewing the video from the PC

Once `ros2 launch puzzlebot_challenge camera.launch.py` is running:

- **Browser**: <http://JETSON-IP:8080>
- **Another node on the PC**: `ros2 topic echo /camera/image_raw` (the Jetson and
  the PC must share the same `ROS_DOMAIN_ID`).
- **RViz**: `rviz2`, add an `Image` display with the `/camera/image_raw` topic.

> The MJPEG server (`cam_server`) streams the `/vision/line` topic by default (the
> annotated line-detection frame). Change the `topic` parameter in
> [`config/camera.yaml`](../src/puzzlebot_challenge/config/camera.yaml) to stream a
> different feed, e.g. `/vision/signs`, `/vision/traffic` or `/camera/image_raw`.

---

## 4. Vision stack feeds

The driving stack publishes several annotated debug images you can inspect with the
MJPEG server, `rqt_image_view` or RViz:

| Topic            | Source              | Shows                                    |
|------------------|---------------------|------------------------------------------|
| `/camera/image_raw` | `picam_publisher` | Raw PiCam frame                          |
| `/vision/line`   | `line_detector`     | Line mask / centroid overlay             |
| `/vision/signs`  | `sign_detector`     | YOLO bounding boxes + labels             |
| `/vision/traffic`| `traffic_controller`| Detected traffic-light color             |

### Sign detection on the Jetson

`sign_detector` loads a YOLO model from `share/puzzlebot_challenge/models/`
(installed from `utils/`). For best performance on the Jetson, use the TensorRT
`best.engine`; regenerate it with
[`utils/export_trt.py`](../src/puzzlebot_challenge/utils/export_trt.py) if the
Ultralytics / TensorRT versions change. If the engine fails to load, the detector
falls back to `best.onnx` / `best.pt`. Tune `conf_threshold` and `imgsz` via the
`signs.launch.py` arguments.

---

## 5. Optional systemd service

To start the camera server automatically when the Jetson boots, create
`/etc/systemd/system/puzzlebot-camera.service`:

```ini
[Unit]
Description=PuzzleBot camera stack
After=network-online.target nvargus-daemon.service

[Service]
Type=simple
User=puzzlebot
WorkingDirectory=/home/puzzlebot/Puzzlebot-Challenge
ExecStart=/bin/bash -c 'source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch puzzlebot_challenge camera.launch.py'
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now puzzlebot-camera
sudo systemctl status puzzlebot-camera
```

---

## 6. Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `Failed to create CaptureSession` | Argus daemon hung | `sudo systemctl restart nvargus-daemon` or reboot |
| `nvbuf_utils: dmabuf_fd -1` | `nveglglessink` over SSH without a display | Use `cam_server` (MJPEG) or launch from the physical monitor |
| `colcon build` can't find `cv_bridge` | Missing package | `sudo apt install ros-humble-cv-bridge` |
| `ImportError: flask` | Missing dependency | `pip3 install flask` |
| `ModuleNotFoundError: ultralytics` | Missing dependency | `pip3 install -r requirements.txt` |
| MJPEG server loads but shows nothing | `picam_publisher` not running or topic mismatch | Check with `ros2 topic hz /camera/image_raw` |
| YOLO is slow / high latency | Running `.pt` instead of TensorRT | Build/use `best.engine`, lower `imgsz`, raise `conf_threshold` |
