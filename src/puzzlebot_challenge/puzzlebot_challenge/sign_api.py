#!/usr/bin/env python3
"""
sign_api.py — Servidor HTTP en el Jetson que recibe comandos de señales
desde la laptop y los publica en /sign/command.

Endpoint:
  POST http://<JETSON_IP>:8081/sign
  Body (JSON): {"command": "turn_left"}
  Comandos válidos: stop | go_straight | turn_left | turn_right | workers | none

Ejemplo desde la laptop:
  curl -X POST http://10.22.171.82:8081/sign \
       -H "Content-Type: application/json" \
       -d '{"command": "turn_left"}'
"""

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from flask import Flask, request, jsonify

VALID_COMMANDS = {"stop", "give_way", "go_straight", "turn_left", "turn_right", "workers", "none"}


class SignApiNode(Node):

    def __init__(self):
        super().__init__('sign_api')
        self.declare_parameter('port', 8081)
        self._pub = self.create_publisher(String, '/sign/command', 10)
        self._last_cmd = 'none'

        port = int(self.get_parameter('port').value)
        app = Flask(__name__)
        app.add_url_rule('/sign', 'sign', self._handle_sign, methods=['POST'])
        app.add_url_rule('/sign/<cmd>', 'sign_get', self._handle_sign_get, methods=['GET'])

        threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=port,
                                   threaded=True, debug=False, use_reloader=False),
            daemon=True
        ).start()
        self.get_logger().info(f'SignAPI listo en http://0.0.0.0:{port}/sign')

    def _publish(self, cmd: str):
        msg = String()
        msg.data = cmd
        self._pub.publish(msg)
        self._last_cmd = cmd
        self.get_logger().info(f'sign/command → {cmd}')

    def _handle_sign(self):
        data = request.get_json(silent=True) or {}
        cmd = str(data.get('command', 'none')).lower()
        if cmd not in VALID_COMMANDS:
            return jsonify({'error': f'Comando inválido: {cmd}'}), 400
        self._publish(cmd)
        return jsonify({'ok': True, 'command': cmd})

    def _handle_sign_get(self, cmd: str):
        cmd = cmd.lower()
        if cmd not in VALID_COMMANDS:
            return jsonify({'error': f'Comando inválido: {cmd}'}), 400
        self._publish(cmd)
        return jsonify({'ok': True, 'command': cmd})


def main(args=None):
    rclpy.init(args=args)
    node = SignApiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
