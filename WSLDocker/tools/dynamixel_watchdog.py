#!/usr/bin/env python3
import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Bool


class DynamixelWatchdog(Node):
    def __init__(self, namespace, controller_name, check_count=6, check_interval=1.0):
        super().__init__('dynamixel_watchdog_' + namespace.strip('/').replace('/', '_'))
        self.namespace = namespace.rstrip('/')
        self.controller_name = controller_name
        self.check_count = check_count
        self.check_interval = check_interval

        topic = f"{self.namespace}/joint_states"
        self.get_logger().info(f"Watching joint states on: {topic}")
        self.sub = self.create_subscription(JointState, topic, self.joint_cb, 10)

        self._last_positions = None
        self._stale_counter = 0
        self._healthy = True

        # publish health status
        health_topic = f"{self.namespace}/bus_healthy"
        self.health_pub = self.create_publisher(Bool, health_topic, 1)
        # initial healthy publish
        self.health_pub.publish(Bool(data=True))

        self.srv_client = self.create_client(SwitchController, f"{self.namespace}/controller_manager/switch_controller")
        while not self.srv_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for switch_controller service...')

    def joint_cb(self, msg: JointState):
        positions = list(msg.position)
        if self._last_positions is None:
            self._last_positions = positions
            self._stale_counter = 0
            return

        if len(positions) != len(self._last_positions):
            # different measurement, reset
            self._last_positions = positions
            self._stale_counter = 0
            return

        if all(abs(a - b) < 1e-6 for a, b in zip(positions, self._last_positions)):
            self._stale_counter += 1
            self.get_logger().debug(f'stale_count={self._stale_counter}')
        else:
            self._stale_counter = 0
            self._last_positions = positions
            if not self._healthy:
                self._healthy = True
                try:
                    self.health_pub.publish(Bool(data=True))
                except Exception:
                    pass

        if self._stale_counter >= self.check_count:
            self.get_logger().warn(f'Detected stale joint_states for {self.namespace} ({self._stale_counter} checks). Attempting controller restart')
            self._stale_counter = 0
            # mark unhealthy and attempt restart
            if self._healthy:
                try:
                    self.health_pub.publish(Bool(data=False))
                except Exception:
                    pass
                self._healthy = False
            self._attempt_restart()

    def _attempt_restart(self):
        req = SwitchController.Request()
        req.start_controllers = []
        req.stop_controllers = [self.controller_name]
        req.strictness = 2

        fut = self.srv_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.done() and fut.result() is not None:
            self.get_logger().info(f'Stop request for {self.controller_name} sent')
        else:
            self.get_logger().error('Failed to call stop controller service')
            return

        # wait briefly then start
        time.sleep(1.0)
        req2 = SwitchController.Request()
        req2.start_controllers = [self.controller_name]
        req2.stop_controllers = []
        req2.strictness = 2
        fut2 = self.srv_client.call_async(req2)
        rclpy.spin_until_future_complete(self, fut2, timeout_sec=5.0)
        if fut2.done() and fut2.result() is not None:
            self.get_logger().info(f'Start request for {self.controller_name} sent')
        else:
            self.get_logger().error('Failed to call start controller service')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', '-n', default='/robot2', help='Robot namespace to watch')
    parser.add_argument('--controller', '-c', default='robot2_variable_stiffness', help='Controller to restart')
    parser.add_argument('--checks', type=int, default=6, help='Consecutive unchanged checks before restart')
    parser.add_argument('--interval', type=float, default=1.0, help='Seconds between checks (sub callback rate dependent)')

    args = parser.parse_args()

    rclpy.init()
    node = DynamixelWatchdog(args.namespace, args.controller, check_count=args.checks, check_interval=args.interval)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
