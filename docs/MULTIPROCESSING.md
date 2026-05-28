# Procesamiento offload — Laptop + Jetson via Tailscale

Guía para correr inferencia de visión en la laptop mientras el Puzzlebot (Jetson)
corre el stack de control. La laptop suscribe al stream de cámara, corre YOLO y
publica los resultados de vuelta a la Jetson.

```
Jetson  →  /camera/image_raw  →  Laptop (YOLO)  →  /sign/command  →  Jetson
```

---

## Requisitos

- Tailscale instalado y activo en ambas máquinas.
- ROS2 Humble en la laptop con el paquete `puzzlebot_challenge` compilado.
- El paquete `ros-humble-rmw-fastrtps-cpp` instalado (viene por defecto con ROS2).

---

## 1. Obtener tu IP de Tailscale

```bash
tailscale ip -4
```

La IP del Jetson es fija: **`100.73.89.116`**

---

## 2. Crear el archivo de configuración FastDDS

El DDS multicast no funciona sobre Tailscale. Necesitamos configurar unicast
explícito para que los nodos se descubran entre máquinas.

Copia la plantilla del repo y reemplaza `TU_LAPTOP_IP` con tu IP de Tailscale:

```bash
cp ~/Desktop/Puzzlebot-Challenge/config/fastdds_puzzlebot.xml ~/fastdds_puzzlebot.xml
# Edita el archivo y reemplaza TU_LAPTOP_IP con tu IP (ej. 100.90.40.98)
nano ~/fastdds_puzzlebot.xml
```

**El Jetson ya tiene su propio XML configurado** — no necesitas tocarlo.

---

## 3. Agregar variables de entorno al `.bashrc`

Corre esto en **ambas** máquinas:

```bash
cat >> ~/.bashrc << 'EOF'

# Puzzlebot — ROS2 multi-machine (Tailscale)
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_puzzlebot.xml
EOF
source ~/.bashrc
```

---

## 4. Verificar la conexión

Con ambas máquinas configuradas y Tailscale activo:

```bash
# En la laptop — debe mostrar tópicos del robot
ros2 topic list
```

Deberías ver `/VelocityEncL`, `/VelocityEncR`, `/VelocitySetL`, `/VelocitySetR`, etc.

---

## 5. Correr el stack de offload

**Jetson** — arranca cámara y micro_ros_agent:
```bash
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 &
ros2 run puzzlebot_challenge picam_publisher
```

**Laptop** — corre inferencia YOLO:
```bash
ros2 run puzzlebot_challenge sign_detector_offload
```

Para visualizar el video anotado:
```bash
ros2 run rqt_image_view rqt_image_view /vision/signs
```

---

## Troubleshooting

**`/camera/image_raw` no aparece en la laptop**
El terminal donde corre `picam_publisher` no tiene `FASTRTPS_DEFAULT_PROFILES_FILE` seteado.
Solución: `source ~/.bashrc` y reiniciar el publisher.

**`ros2 topic list` solo muestra `/parameter_events` y `/rosout`**
Alguna de las dos máquinas tiene el XML mal configurado o Tailscale no está activo.
Verifica: `ping <IP_TAILSCALE_OTRA_MAQUINA>`.

**Domain ID incorrecto**
El firmware del microcontrolador usa `ROS_DOMAIN_ID=0`. Todo el stack debe usar el mismo dominio.
Verifica con `echo $ROS_DOMAIN_ID` antes de arrancar cualquier nodo.
