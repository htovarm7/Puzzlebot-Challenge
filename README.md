# Puzzlebot Challenge

Workspace ROS2 (Humble) para el **PuzzleBot** de Manchester Robotics con Jetson
Nano + PiCam CSI (IMX219). Contiene:

- **`picam_publisher`** — publica frames de la PiCam en `/camera/image_raw`.
- **`cam_server`** — sirve el video como MJPEG por HTTP (vista en navegador).
- **`pid_controller`** — PID en lazo cerrado para recorrer trayectorias `SQUARE` o `WAYPOINTS`.
- **`pid_tuner`** — herramienta interactiva de sintonización (parámetros en caliente).

---

## Requisitos

- **ROS2 Humble** instalado (`/opt/ros/humble`).
- Jetson Nano con JetPack ≥ 4.6 (drivers `nvarguscamerasrc`).
- Cable Ethernet entre la Jetson y tu PC, o ambas en la misma red.

### Dependencias del sistema

```bash
sudo apt update
sudo apt install -y \
  ros-humble-cv-bridge \
  python3-opencv \
  python3-pip \
  python3-colcon-common-extensions
pip3 install flask
```

`python3-opencv` en Jetson trae soporte GStreamer (necesario para la PiCam CSI).

---

## Build

```bash
git clone <url-del-repo> ~/Puzzlebot-Challenge
cd ~/Puzzlebot-Challenge

# (Opcional) instala dependencias declaradas en package.xml:
rosdep install --from-paths src -y --ignore-src

# Compila el workspace
colcon build --symlink-install

# Carga el overlay
source install/setup.bash
```

Añade el `source` a tu `~/.bashrc` para no repetirlo cada terminal:

```bash
echo "source ~/Puzzlebot-Challenge/install/setup.bash" >> ~/.bashrc
```

---

## Uso

### Cámara + servidor web (visualización rápida)

```bash
ros2 launch puzzlebot_challenge camera.launch.py
```

Abre en tu PC: <http://IP-DE-LA-JETSON:8080>

### Solo controlador PID

```bash
ros2 launch puzzlebot_challenge pid.launch.py task:=SQUARE
# o
ros2 launch puzzlebot_challenge pid.launch.py task:=WAYPOINTS
```

### Todo junto (cámara + servidor + PID)

```bash
ros2 launch puzzlebot_challenge full.launch.py task:=SQUARE
```

### Sintonización de PID

```bash
ros2 run puzzlebot_challenge pid_tuner straight   # o: turn
```

En otra terminal, ajusta ganancias en caliente:

```bash
ros2 param set /pid_tuner kp_dist 1.5
ros2 param set /pid_tuner kd_head 0.25
```

Visualiza métricas con `rqt_plot`:

```bash
ros2 run rqt_plot rqt_plot /tuner/error_dist /tuner/cmd_v
```

---

## Configuración

Los parámetros viven en `src/puzzlebot_challenge/config/`:

- **`camera.yaml`** — resolución, FPS, tópico, puerto del MJPEG.
- **`pid.yaml`** — ganancias, límites de velocidad, tolerancias, waypoints, etc.

Edita los YAML y vuelve a lanzar — no es necesario `colcon build` si usaste
`--symlink-install`.

---

## Tópicos relevantes

| Tópico                | Tipo                  | Quién publica       | Quién consume        |
|-----------------------|-----------------------|---------------------|----------------------|
| `/camera/image_raw`   | `sensor_msgs/Image`   | `picam_publisher`   | `cam_server`, otros  |
| `/VelEncL`, `/VelEncR`| `std_msgs/Float32`    | hardware            | `pid_controller`     |
| `/VelocitySetL/R`     | `std_msgs/Float32`    | `pid_controller`    | hardware             |
| `/tuner/*`            | `std_msgs/Float32`    | `pid_tuner`         | `rqt_plot`           |

---

## Estructura del repo

```
Puzzlebot-Challenge/
├── src/
│   └── puzzlebot_challenge/
│       ├── puzzlebot_challenge/   # código Python (nodos)
│       ├── launch/                # archivos .launch.py
│       ├── config/                # parámetros YAML
│       ├── package.xml
│       └── setup.py
├── scripts/                       # utilidades no-ROS (tests rápidos)
├── docs/                          # documentación extra
└── README.md
```

Más detalles en [`docs/SETUP.md`](docs/SETUP.md).
