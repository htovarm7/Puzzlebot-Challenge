#!/usr/bin/env bash
# Graba /camera/image_raw en un rosbag.
# Uso: ./record_camera.sh [carpeta_destino] [topics_extra...]
#
# Ejemplos:
#   ./record_camera.sh
#   ./record_camera.sh ~/mis_bags
#   ./record_camera.sh ~/mis_bags /sign/command /vision/signs

BAG_DIR="${1:-$HOME/rosbags}"
shift || true  # quita el primer arg si existía; el resto son topics extra

mkdir -p "$BAG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
BAG_PATH="$BAG_DIR/camera_$STAMP"

TOPICS="/camera/image_raw $*"

echo "Grabando en: $BAG_PATH"
echo "Topics: $TOPICS"
echo "Ctrl+C para detener."
echo ""

# shellcheck disable=SC2086
ros2 bag record -o "$BAG_PATH" $TOPICS
