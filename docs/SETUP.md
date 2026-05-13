# Setup detallado — PuzzleBot Jetson

Esta guía cubre lo que el `README` no entra al detalle: conexión a la Jetson,
red, troubleshooting de la cámara CSI y servicios opcionales.

---

## 1. Conexión a la Jetson

### Por Ethernet directo (PC ↔ Jetson)

Si conectas la Jetson directo a tu PC con un cable Ethernet y compartes la
conexión desde NetworkManager (Settings → Network → Ethernet → IPv4 →
"Shared to other computers"), la Jetson recibirá una IP en `10.42.0.0/24`.

Para encontrarla desde el PC:

```bash
ip neigh show dev <interfaz-ethernet>
```

Conéctate por SSH (usuario por defecto del Puzzlebot: `puzzlebot`):

```bash
ssh puzzlebot@10.42.0.X
```

### Hostname / MagicDNS

- Para cambiar el nombre del sistema: `sudo hostnamectl set-hostname jetson-X` y
  edita la línea `127.0.1.1` de `/etc/hosts`.
- Si usas Tailscale, edita el nombre desde
  <https://login.tailscale.com/admin/machines> o con
  `sudo tailscale set --hostname=jetson-X`.

---

## 2. Cámara CSI (IMX219)

### Verificar que la detecta el kernel

```bash
v4l2-ctl --list-devices
# vi-output, imx219 8-0010 (platform:54080000.vi:4):
#     /dev/video0
```

### Test rápido sin GUI

```bash
gst-launch-1.0 -e nvarguscamerasrc num-buffers=10 sensor-mode=4 ! \
  "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1" ! \
  nvjpegenc ! multifilesink location=frame_%03d.jpg
```

Debe generar 10 JPGs. Si falla con `Failed to create CaptureSession`:

1. Reinicia el daemon: `sudo systemctl restart nvargus-daemon`.
2. Si persiste tras un mal kill, **reinicia la Jetson**: `sudo reboot`.
3. **Nunca uses `kill -9` con `nvargus-daemon`** — deja el sensor en mal
   estado y solo se recupera con reboot.

### Modos del sensor IMX219

| sensor-mode | Resolución  | FPS  |
|-------------|-------------|------|
| 0           | 3264×2464   | 21   |
| 1           | 3264×1848   | 28   |
| 2           | 1920×1080   | 30   |
| 3           | 1640×1232   | 30   |
| 4           | 1280×720    | 60   |
| 5           | 1280×720    | 120  |

El default del workspace es `sensor_mode: 4` (720p @ 60fps), escalado en
hardware a 320×240 para procesamiento.

---

## 3. Ver el video desde el PC

Una vez corriendo `ros2 launch puzzlebot_challenge camera.launch.py`:

- **Navegador**: <http://IP-DE-LA-JETSON:8080>
- **Otro nodo en el PC**: `ros2 topic echo /camera/image_raw` (necesitas que la
  Jetson y el PC estén en el mismo `ROS_DOMAIN_ID`).
- **RViz**: `rviz2`, agrega un display de tipo `Image` con el tópico
  `/camera/image_raw`.

---

## 4. Servicio systemd opcional

Para que el servidor de cámara arranque solo al encender la Jetson, crea
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

Activa:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now puzzlebot-camera
sudo systemctl status puzzlebot-camera
```

---

## 5. Problemas comunes

| Síntoma | Causa probable | Fix |
|---------|----------------|-----|
| `Failed to create CaptureSession` | Daemon Argus colgado | `sudo systemctl restart nvargus-daemon` o reboot |
| `nvbuf_utils: dmabuf_fd -1` | `nveglglessink` por SSH sin display | Usar `cam_server` (MJPEG) o lanzar desde el monitor físico |
| `colcon build` no encuentra `cv_bridge` | Falta paquete | `sudo apt install ros-humble-cv-bridge` |
| `ImportError: flask` | Falta dependencia | `pip3 install flask` |
| El servidor MJPEG carga pero no muestra | `picam_publisher` no está corriendo o el tópico no coincide | Verifica con `ros2 topic hz /camera/image_raw` |
