import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class PuzzlebotSequence(Node):
    def __init__(self):
        super().__init__('puzzlebot_sequence')

        # Publishers para los motores
        self.pub_left = self.create_publisher(Float32, '/VelocitySetL', 10)
        self.pub_right = self.create_publisher(Float32, '/VelocitySetR', 10)

        # Subscribers para los encoders (nombres exactos según el manual MCR2)
        self.sub_left = self.create_subscription(Float32, '/VelEncL', self.enc_l_callback, 10)
        self.sub_right = self.create_subscription(Float32, '/VelEncR', self.enc_r_callback, 10)

        # Bucle de control a 10 Hz (cada 0.1 segundos)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Variables de estado
        self.start_time = self.get_clock().now()
        self.state = 0  # 0: Adelante, 1: Atrás, 2: Izquierda, 3: Derecha, 4: Detenido
        self.speed = 5.0  # Velocidad objetivo (ajusta si es muy rápido/lento)

        self.get_logger().info('Iniciando secuencia: ADELANTE')

    def enc_l_callback(self, msg):
        self.get_logger().info(f'[Encoder L]: {msg.data:.2f}')

    def enc_r_callback(self, msg):
        self.get_logger().info(f'[Encoder R]: {msg.data:.2f}')

    def publish_speeds(self, left, right):
        msg_l = Float32()
        msg_l.data = float(left)
        msg_r = Float32()
        msg_r.data = float(right)
        self.pub_left.publish(msg_l)
        self.pub_right.publish(msg_r)

    def control_loop(self):
        # Calcular el tiempo transcurrido en segundos
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9

        if self.state == 0: # Adelante
            self.publish_speeds(self.speed, self.speed)
            if elapsed > 5.0:
                self.get_logger().info('Cambiando a: ATRÁS')
                self.state = 1
                self.start_time = self.get_clock().now()

        elif self.state == 1: # Atrás
            self.publish_speeds(-self.speed, -self.speed)
            if elapsed > 5.0:
                self.get_logger().info('Cambiando a: GIRO IZQUIERDA')
                self.state = 2
                self.start_time = self.get_clock().now()

        elif self.state == 2: # Izquierda (Rueda derecha avanza, izquierda retrocede)
            self.publish_speeds(-self.speed, self.speed)
            if elapsed > 5.0:
                self.get_logger().info('Cambiando a: GIRO DERECHA')
                self.state = 3
                self.start_time = self.get_clock().now()

        elif self.state == 3: # Derecha (Rueda izquierda avanza, derecha retrocede)
            self.publish_speeds(self.speed, -self.speed)
            if elapsed > 5.0:
                self.get_logger().info('Secuencia completada. DETENIENDO MOTORES.')
                self.state = 4
                self.publish_speeds(0.0, 0.0)
                self.timer.cancel() # Detenemos el ciclo de control

def main(args=None):
    rclpy.init(args=args)
    node = PuzzlebotSequence()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Medida de seguridad: Si presionas Ctrl+C, frena los motores inmediatamente
        node.get_logger().warn('Interrupción manual. Frenando robot...')
        node.publish_speeds(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()