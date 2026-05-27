# Detección de Señales de Tráfico

Cubre dos modos de operación: **standalone en Jetson** y **offload a laptop**.

---

## Arquitectura del pipeline

```
Cámara CSI (Jetson)
       │
       ▼
/camera/image_raw
       │
  ┌────┴────────────────────────────┐
  │  Modo A: Jetson solo            │
  │  sign_detector.py               │
  │  YOLO (CPU) + HSV fallback      │
  └────┬────────────────────────────┘
       │          ┌─────────────────────────────────┐
       │          │  Modo B: Offload a laptop        │
       └──────────►  sign_detector_offload.py        │
                  │  YOLO (GPU laptop) + HSV fallback│
                  └────────────┬────────────────────┘
                               │
                               ▼
                    /sign/command   (stop | go_straight |
                    /sign/detected   turn_left | turn_right |
                    /vision/signs    workers | none)
```

---

## Modo A — Jetson standalone (CPU)

### Setup de dependencias (una sola vez)

```bash
# En la Jetson
export LD_PRELOAD=/home/puzzlebot/.local/lib/python3.8/site-packages/torch.libs/libgomp-804f19d4.so.1.0.0
echo 'export LD_PRELOAD=/home/puzzlebot/.local/lib/python3.8/site-packages/torch.libs/libgomp-804f19d4.so.1.0.0' >> ~/.bashrc

pip3 install torch==2.4.1 ultralytics --quiet
```

### Correr el nodo

```bash
source ~/Puzzlebot-Challenge/install/setup.bash
ros2 run puzzlebot_challenge sign_detector
```

### Parámetros ajustables

| Parámetro       | Default | Descripción |
|-----------------|---------|-------------|
| `imgsz`         | `192`   | Tamaño de imagen para YOLO. Menor = más FPS. Probar 160/192/256. |
| `conf_threshold`| `0.45`  | Confianza mínima para YOLO. Bajar si no detecta; subir si hay falsos positivos. |
| `max_infer_fps` | `0`     | Límite de FPS de inferencia. `0` = sin límite. |
| `model_path`    | auto    | Ruta al modelo `.pt` o `.engine`. Auto-detecta `.engine` si existe. |

Ejemplo con parámetros custom:
```bash
ros2 run puzzlebot_challenge sign_detector --ros-args \
  -p imgsz:=160 \
  -p conf_threshold:=0.40
```

### FPS esperados en Jetson Nano (CPU, 4 cores)

| imgsz | FPS aprox. |
|-------|------------|
| 160   | 8–12       |
| 192   | 5–8        |
| 256   | 3–5        |
| 320   | 2–3        |

---

## Modo B — Offload a laptop (recomendado para competencia)

La laptop corre YOLO y publica los resultados. La Jetson solo suscribe.
Ventaja: la laptop tiene CPU/GPU mucho más potente → 30+ FPS de inferencia.

### Requisitos de red

- Laptop y Jetson en la misma red (WiFi local, cable Ethernet, o Tailscale).
- IP de la Jetson en este setup: `100.73.89.116` (Tailscale).

### Setup en ambas máquinas

Agrega esto al `~/.bashrc` de **laptop y Jetson**:

```bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
```

Aplica sin reiniciar:
```bash
source ~/.bashrc
```

### En la Jetson — solo cámara y controladores

```bash
# Terminal 1: cámara
ros2 run puzzlebot_challenge picam_publisher

# Terminal 2: line follower o traffic controller (NO correr sign_detector)
ros2 launch puzzlebot_challenge full.launch.py
```

### En la laptop — YOLO offload

```bash
cd ~/Desktop/Puzzlebot-Challenge
colcon build --packages-select puzzlebot_challenge --symlink-install
source install/setup.bash

ros2 run puzzlebot_challenge sign_detector_offload
```

Para ver el video anotado con las detecciones:
```bash
ros2 run rqt_image_view rqt_image_view /vision/signs
```

### Verificar que la Jetson recibe los comandos

```bash
# En la Jetson
ros2 topic echo /sign/command
```

Debe mostrar `data: stop`, `data: go_straight`, etc. cuando se pone una señal frente a la cámara.

### Parámetros del nodo offload

| Parámetro       | Default | Descripción |
|-----------------|---------|-------------|
| `imgsz`         | `320`   | En laptop se puede usar 320 o más. |
| `conf_threshold`| `0.45`  | Confianza mínima YOLO. |
| `image_topic`   | `/camera/image_raw` | Tópico de imagen de entrada. |

---

## Acelerar con TensorRT (futuro)

Cuando estén disponibles los bindings Python de TensorRT para Python 3.8
en esta imagen, el flujo es:

```bash
# 1. Exportar en laptop (una sola vez)
python3 src/puzzlebot_challenge/utils/export_trt.py

# 2. Copiar el engine a la Jetson
scp src/puzzlebot_challenge/utils/best.engine puzzlebot@100.73.89.116:~/Puzzlebot-Challenge/src/puzzlebot_challenge/utils/

# 3. Rebuildar (el setup.py ya incluye *.engine en data_files)
colcon build --packages-select puzzlebot_challenge --symlink-install

# 4. sign_detector.py auto-carga best.engine si está junto a best.pt
ros2 run puzzlebot_challenge sign_detector
```

FPS esperados con TRT FP16 en Jetson Nano GPU: **15–25 FPS**.

---

## Tópicos publicados

| Tópico           | Tipo                  | Valores |
|------------------|-----------------------|---------|
| `/sign/command`  | `std_msgs/String`     | `stop` \| `go_straight` \| `turn_left` \| `turn_right` \| `workers` \| `none` |
| `/sign/detected` | `std_msgs/Bool`       | `true` si hay señal activa |
| `/vision/signs`  | `sensor_msgs/Image`   | Frame anotado con bounding boxes (solo si hay suscriptores) |

---

## Troubleshooting

| Síntoma | Causa | Fix |
|---------|-------|-----|
| `cannot allocate memory in static TLS block` | libgomp no precargada | `export LD_PRELOAD=.../libgomp-804f19d4.so.1.0.0` (ver Setup) |
| `YOLO=OFF (fallback only)` | torch no instalado o falla import | `pip3 install torch==2.4.1 ultralytics` |
| No llegan mensajes `/sign/command` en Jetson | ROS_DOMAIN_ID distinto | Verificar que ambas máquinas tengan el mismo `ROS_DOMAIN_ID` |
| Latencia alta en modo offload | Imágenes raw por WiFi son pesadas | Usar `image_transport` con compresión JPEG o reducir resolución de cámara |
| `CUDA: False` en Jetson | Incompatibilidad Python 3.8 + CUDA 10.2 | Usar Modo B (offload) o esperar imagen con Python 3.6 |
