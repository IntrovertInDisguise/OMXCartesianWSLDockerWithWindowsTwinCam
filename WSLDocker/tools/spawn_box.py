#!/usr/bin/env python3
"""Spawn a simple box model into Gazebo via the SpawnEntity service."""
import argparse
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test_box')
    parser.add_argument('--x', type=float, default=0.2)
    parser.add_argument('--y', type=float, default=0.0)
    parser.add_argument('--z', type=float, default=0.15)
    parser.add_argument('--sx', type=float, default=0.08)
    parser.add_argument('--sy', type=float, default=0.08)
    parser.add_argument('--sz', type=float, default=0.05)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('spawn_box')

    cli = node.create_client(SpawnEntity, '/spawn_entity')
    if not cli.wait_for_service(timeout_sec=5.0):
        node.get_logger().error('SpawnEntity service not available')
        return

    sdf = SDF_TEMPLATE.format(name=args.name, sx=args.sx, sy=args.sy, sz=args.sz)

    req = SpawnEntity.Request()
    req.name = args.name
    req.xml = sdf
    # initial_pose
    from geometry_msgs.msg import Pose
    p = Pose()
    p.position.x = args.x
    p.position.y = args.y
    p.position.z = args.z
    req.initial_pose = p

    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
    if future.done() and future.result() is not None:
        node.get_logger().info(f'Spawned {args.name}')
    else:
        node.get_logger().error('Failed to spawn entity')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
