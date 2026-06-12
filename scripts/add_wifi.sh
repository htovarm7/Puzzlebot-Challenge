#!/usr/bin/env bash
# Registra una nueva red WiFi en la Jetson Nano usando nmcli.
# Uso: ./add_wifi.sh [SSID] [CONTRASEÑA]
# Si no se pasan argumentos, los solicita de forma interactiva.

set -e

SSID="${1:-}"
PASS="${2:-}"

if [[ -z "$SSID" ]]; then
    read -rp "SSID (nombre de la red): " SSID
fi

if [[ -z "$SSID" ]]; then
    echo "ERROR: el SSID no puede estar vacío."
    exit 1
fi

read -rsp "Contraseña (dejar vacío para red abierta): " PASS
echo

if [[ -n "$PASS" ]]; then
    sudo nmcli device wifi connect "$SSID" password "$PASS"
else
    sudo nmcli device wifi connect "$SSID"
fi

echo ""
echo "Red '$SSID' registrada y conectada."
echo "Para reconectar automáticamente en el futuro: nmcli connection show '$SSID'"
