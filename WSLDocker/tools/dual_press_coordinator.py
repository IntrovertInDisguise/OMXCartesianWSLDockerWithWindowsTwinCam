#!/usr/bin/env python3
"""Dual press coordinator

Publishes the same PoseStamped waypoint to two robot controllers simultaneously
after an optional delay. Intended to be launched alongside the dual-hardware
launch to command both robots to press or move at the same time.

Usage:
  python3 dual_press_coordinator.py --namespace1 /robot1 --namespace2 /robot2 --wait 6.5
"""
import argparse
import time
from geometry_msgs.msg import PoseStamped

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers
from std_msgs.msg import Bool


def build_parser():
    p = argparse.ArgumentParser(description='Dual press coordinator')
    p.add_argument('--namespace1', default='/robot1')
    p.add_argument('--namespace2', default='/robot2')
    p.add_argument('--controller1', default='robot1_variable_stiffness')
    p.add_argument('--controller2', default='robot2_variable_stiffness')
    p.add_argument('--wait', type=float, default=5.0,
                   help='Seconds to wait before publishing')
    p.add_argument('--frame', default='absolute', help='PoseStamped header.frame_id')
    p.add_argument('--x', type=float, default=0.20)
    p.add_argument('--y', type=float, default=0.0)
    p.add_argument('--z', type=float, default=0.12)
    p.add_argument('--retry-publishes', type=int, default=5,
                   help='Number of repeated publishes to send')
    p.add_argument('--retry-interval', type=float, default=0.05,
                   help='Seconds between repeated publishes')
    return p


class Coordinator(Node):
    def __init__(self, args):
        super().__init__('dual_press_coordinator')
        t1 = f"{args.namespace1.rstrip('/')}/{args.controller1}/waypoint_command"
        t2 = f"{args.namespace2.rstrip('/')}/{args.controller2}/waypoint_command"
        self.get_logger().info(f'Will publish to: {t1} and {t2} after {args.wait}s')
        self.pub1 = self.create_publisher(PoseStamped, t1, 10)
        self.pub2 = self.create_publisher(PoseStamped, t2, 10)
        # health subscriptions
        h1 = f"{args.namespace1.rstrip('/')}/bus_healthy"
        h2 = f"{args.namespace2.rstrip('/')}/bus_healthy"
        self.health1 = True
        self.health2 = True
        try:
            self.create_subscription(Bool, h1, self._h1_cb, 1)
        except Exception:
            pass
        try:
            self.create_subscription(Bool, h2, self._h2_cb, 1)
        except Exception:
            pass
        self.args = args

    def _h1_cb(self, msg: Bool):
        self.health1 = bool(msg.data)

    def _h2_cb(self, msg: Bool):
        self.health2 = bool(msg.data)

    def wait_for_subscribers(self, timeout=5.0, poll_interval=0.1):
        end = time.time() + float(timeout)
        while time.time() < end:
            c1 = self.pub1.get_subscription_count()
            c2 = self.pub2.get_subscription_count()
            if c1 > 0 and c2 > 0:
                self.get_logger().info(f'both subscribers present (counts={c1},{c2})')
                return True
            self.get_logger().info(f'waiting for subscribers (counts={c1},{c2})')
            time.sleep(poll_interval)
        self.get_logger().warning(f'timeout waiting for subscribers (last counts={c1},{c2})')
        return False

    def wait_for_active_controllers(self, timeout=10.0, poll_interval=0.5):
        """Wait until both variable stiffness controllers are active via controller_manager service."""
        end = time.time() + float(timeout)
        ns1 = f"{self.args.namespace1.rstrip('/')}/controller_manager"
        ns2 = f"{self.args.namespace2.rstrip('/')}/controller_manager"
        client1 = self.create_client(ListControllers, f"{ns1}/list_controllers")
        client2 = self.create_client(ListControllers, f"{ns2}/list_controllers")

        while time.time() < end:
            ready1 = ready2 = False
            if client1.wait_for_service(timeout_sec=0.5):
                req = ListControllers.Request()
                fut = client1.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                if fut.done() and fut.result():
                    for c in fut.result().controller:
                        if c.name == self.args.controller1 and c.state == 'active':
                            ready1 = True
                            break
            if client2.wait_for_service(timeout_sec=0.5):
                req2 = ListControllers.Request()
                fut2 = client2.call_async(req2)
                rclpy.spin_until_future_complete(self, fut2, timeout_sec=1.0)
                if fut2.done() and fut2.result():
                    for c in fut2.result().controller:
                        if c.name == self.args.controller2 and c.state == 'active':
                            ready2 = True
                            break

            self.get_logger().info(f'controller readiness robot1={ready1} robot2={ready2}')
            if ready1 and ready2:
                self.get_logger().info('both controllers active')
                return True
            time.sleep(poll_interval)

        self.get_logger().warning('timeout waiting for active controllers')
        return False

    def publish_once(self):
        p = PoseStamped()
        p.header.frame_id = self.args.frame
        # orientation left as identity (w=1)
        p.pose.orientation.w = 1.0

        # Publish once, updating stamp each time caller requests publish
        p.pose.position.x = float(self.args.x)
        p.pose.position.y = float(self.args.y)
        p.pose.position.z = float(self.args.z)
        p.header.stamp = self.get_clock().now().to_msg()
        self.pub1.publish(p)
        self.pub2.publish(p)
        self.get_logger().info('Published coordinated waypoint to both robots')

    def publish_with_retries(self, repeats=5, interval=0.05):
        for i in range(max(1, int(repeats))):
            p = PoseStamped()
            p.header.frame_id = self.args.frame
            p.pose.position.x = float(self.args.x)
            p.pose.position.y = float(self.args.y)
            p.pose.position.z = float(self.args.z)
            p.pose.orientation.w = 1.0
            p.header.stamp = self.get_clock().now().to_msg()
            if self.health1:
                self.pub1.publish(p)
            else:
                self.get_logger().warning('Skipping publish to robot1: bus unhealthy')
            if self.health2:
                self.pub2.publish(p)
            else:
                self.get_logger().warning('Skipping publish to robot2: bus unhealthy')
            self.get_logger().debug(f'publish attempt {i+1}/{repeats}')
            time.sleep(float(interval))


def main():
    parser = build_parser()
    args = parser.parse_args()

    rclpy.init()
    node = Coordinator(args)
    try:
        # Wait for requested delay to allow controllers to come up
        node.get_logger().info(f'Waiting {args.wait}s before attempting to publish...')
        time.sleep(max(0.0, float(args.wait)))

        # After initial delay, ensure controllers are active and subscribers present
        node.wait_for_active_controllers(timeout=10.0, poll_interval=0.5)
        node.wait_for_subscribers(timeout=5.0, poll_interval=0.1)

        # Publish multiple times to tolerate transient readiness
        node.publish_with_retries(repeats=args.retry_publishes, interval=args.retry_interval)
        # Give a short moment for messages to go out
        time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
