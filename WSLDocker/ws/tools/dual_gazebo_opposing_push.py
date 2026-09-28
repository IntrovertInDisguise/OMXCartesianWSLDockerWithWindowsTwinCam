#!/usr/bin/env python3
"""Dual opposing push scenario for dual_gazebo_variable_stiffness.

1. Spawns a box at center between robot1 and robot2 base links.
2. Waits for both variable stiffness controllers to become active.
3. Sends coordinated waypoint trajectories in straight-line motions for both arms.

Usage:
  source /opt/ros/humble/setup.bash
  source /workspaces/omx_ros2/ws/install/setup.bash
  python3 tools/dual_gazebo_opposing_push.py --duration 30

Notes:
- If you run this inside a script or in ROS2 launch, ensure rclpy is sourced.
- Uses the existing variable stiffness controller waypoint interface.
"""

import argparse
import math
import os
import time

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Bool

SDF_TEMPLATE = """
<sdf version='1.6'>
  <model name='{name}'>
    <static>false</static>
    <link name='link'>
      <pose>0 0 0 0 0 0</pose>
      <collision name='collision'>
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
      </collision>
      <visual name='visual'>
        <geometry>
          <box><size>{sx} {sy} {sz}</size></box>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""


def build_parser():
    p = argparse.ArgumentParser(description='Dual opposing push scenario')
    p.add_argument('--duration', type=float, default=20.0,
                   help='Overall exercise duration in seconds')
    p.add_argument('--box-name', default='opposing_push_box')
    p.add_argument('--box-x', type=float, default=0.16)
    p.add_argument('--box-y', type=float, default=0.0)
    p.add_argument('--box-z', type=float, default=0.08)
    p.add_argument('--box-sx', type=float, default=0.08)
    p.add_argument('--box-sy', type=float, default=0.08)
    p.add_argument('--box-sz', type=float, default=0.12)
    p.add_argument('--robot1-name', default='/robot1')
    p.add_argument('--robot2-name', default='/robot2')
    p.add_argument('--controller1', default='robot1_variable_stiffness')
    p.add_argument('--controller2', default='robot2_variable_stiffness')
    p.add_argument('--wait-before-push', type=float, default=8.0,
                   help='Seconds to wait before first trajectory')
    p.add_argument('--step-rate', type=float, default=1.0,
                   help='Waypoint publish frequency (Hz)')
    p.add_argument('--retracted-distance', type=float, default=0.16,
                   help='Initial retracted Y distance from box centroid')
    p.add_argument('--press-distance', type=float, default=0.05,
                   help='Y distance to press point near box centroid')
    p.add_argument('--press-x-offset', type=float, default=0.01,
                   help='Offset in X in addition to box centroid for press point')
    p.add_argument('--line-steps', type=int, default=10,
                   help='Number of straight-line steps between retracted and press points')
    return p


class DualOpposingPush(Node):
    def __init__(self, args):
        super().__init__('dual_gazebo_opposing_push')

        self.args = args

        self.pub1 = self.create_publisher(
            PoseStamped,
            f"{args.robot1_name.rstrip('/')}/{args.controller1}/waypoint_command",
            10)
        self.pub2 = self.create_publisher(
            PoseStamped,
            f"{args.robot2_name.rstrip('/')}/{args.controller2}/waypoint_command",
            10)

        self.health1 = True
        self.health2 = True
        try:
            self.create_subscription(Bool, f"{args.robot1_name.rstrip('/')}/bus_healthy", self._h1_cb, 1)
        except Exception:
            pass
        try:
            self.create_subscription(Bool, f"{args.robot2_name.rstrip('/')}/bus_healthy", self._h2_cb, 1)
        except Exception:
            pass

    def _h1_cb(self, msg: Bool):
        self.health1 = bool(msg.data)

    def _h2_cb(self, msg: Bool):
        self.health2 = bool(msg.data)

    def wait_for_controllers(self, timeout=60.0):
        ns1 = f"{self.args.robot1_name.rstrip('/')}/controller_manager"
        ns2 = f"{self.args.robot2_name.rstrip('/')}/controller_manager"
        c1 = self.create_client(ListControllers, f"{ns1}/list_controllers")
        c2 = self.create_client(ListControllers, f"{ns2}/list_controllers")

        end = time.time() + timeout
        while time.time() < end:
            r1, r2 = False, False

            if c1.wait_for_service(timeout_sec=0.5):
                req = ListControllers.Request()
                fut = c1.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                if fut.done() and fut.result():
                    r1 = any(c.name == self.args.controller1 and c.state == 'active' for c in fut.result().controller)

            if c2.wait_for_service(timeout_sec=0.5):
                req = ListControllers.Request()
                fut = c2.call_async(req)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=1.0)
                if fut.done() and fut.result():
                    r2 = any(c.name == self.args.controller2 and c.state == 'active' for c in fut.result().controller)

            self.get_logger().info(f'controller readiness robot1={r1} robot2={r2}')
            if r1 and r2:
                return True
            time.sleep(0.5)

        self.get_logger().warning('controllers not both active after timeout (robot1=%s, robot2=%s)', r1, r2)
        # Continue anyway but with warning so we can test and log partial behaviour
        return r1 or r2

    def spawn_box(self):
        client = self.create_client(SpawnEntity, '/spawn_entity')
        if not client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('SpawnEntity service unavailable')
            return False

        sdf = SDF_TEMPLATE.format(
            name=self.args.box_name,
            sx=self.args.box_sx,
            sy=self.args.box_sy,
            sz=self.args.box_sz,
        )
        req = SpawnEntity.Request()
        req.name = self.args.box_name
        req.xml = sdf
        req.initial_pose = Pose()
        req.initial_pose.position.x = self.args.box_x
        req.initial_pose.position.y = self.args.box_y
        req.initial_pose.position.z = self.args.box_z

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.done() and future.result() is not None:
            self.get_logger().info('Spawned box at midpoint')
            return True

        self.get_logger().error('Failed to spawn box')
        return False

    def publish_waypoint(self, x: float, y1: float, y2: float, z: float, frame='absolute'):
        for _ in range(3):
            msg1 = PoseStamped()
            msg1.header.frame_id = frame
            msg1.header.stamp = self.get_clock().now().to_msg()
            msg1.pose.position.x = float(x)
            msg1.pose.position.y = float(y1)
            msg1.pose.position.z = float(z)
            msg1.pose.orientation.w = 1.0

            msg2 = PoseStamped()
            msg2.header.frame_id = frame
            msg2.header.stamp = self.get_clock().now().to_msg()
            msg2.pose.position.x = float(x)
            msg2.pose.position.y = float(y2)
            msg2.pose.position.z = float(z)
            msg2.pose.orientation.w = 1.0

            if self.health1:
                self.pub1.publish(msg1)
            if self.health2:
                self.pub2.publish(msg2)

            self.get_logger().info(f'publish waypoint r1=({x:.3f},{y1:.3f},{z:.3f}) r2=({x:.3f},{y2:.3f},{z:.3f})')
            time.sleep(max(0.01, 1.0 / self.args.step_rate))

    def publish_line(self, start_x, start_y1, start_y2, end_x, end_y1, end_y2, z, steps=10):
        for i in range(steps):
            alpha = i / max(1, steps - 1)
            x = float(start_x + alpha * (end_x - start_x))
            y1 = float(start_y1 + alpha * (end_y1 - start_y1))
            y2 = float(start_y2 + alpha * (end_y2 - start_y2))
            self.publish_waypoint(x, y1, y2, z)

    def run_trajectory(self):
        base_x = self.args.box_x
        z = max(0.01, self.args.box_z + self.args.box_sz / 2.0)

        retracted_y = float(self.args.retracted_distance)
        press_y = float(self.args.press_distance)
        press_x = float(base_x + self.args.press_x_offset)
        retracted_x = float(base_x - 0.06)

        start_t = time.time()
        end_t = start_t + self.args.duration

        self.get_logger().info(f'Waiting {self.args.wait_before_push}s before first move')
        time.sleep(self.args.wait_before_push)

        # Start from retracted positions, outside the box centroid
        self.publish_waypoint(retracted_x, retracted_y, -retracted_y, z)

        # Execute opposing push cycles to the box centroid (with small side offset)
        # Keep x constant for straight-line motion along y axis (as requested).
        retracted_x = press_x

        while time.time() < end_t:
            self.publish_line(
                retracted_x, retracted_y, -retracted_y,
                press_x, press_y, -press_y,
                z, steps=self.args.line_steps,
            )
            self.publish_line(
                press_x, press_y, -press_y,
                retracted_x, retracted_y, -retracted_y,
                z, steps=self.args.line_steps,
            )

        self.get_logger().info('Completed opposing push sequence')


def main():
    args = build_parser().parse_args()

    rclpy.init()
    node = DualOpposingPush(args)

    try:
        if not node.spawn_box():
            raise RuntimeError('box spawn failed')

        if not node.wait_for_controllers(timeout=60.0):
            raise RuntimeError('controllers not ready')

        node.run_trajectory()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
