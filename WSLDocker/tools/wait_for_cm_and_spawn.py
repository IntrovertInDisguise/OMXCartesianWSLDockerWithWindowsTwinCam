#!/usr/bin/env python3
"""
Wait for a controller_manager service for a namespace, then spawn controllers.

Usage: wait_for_cm_and_spawn.py --namespace /robot1 --controller-manager /robot1/controller_manager --controllers joint_state_broadcaster robot1_variable_stiffness
"""
import argparse
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node

from controller_manager_msgs.srv import ListControllers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', required=True)
    parser.add_argument('--controller-manager', required=True)
    parser.add_argument('--controllers', nargs='+', required=True)
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('wait_for_cm_and_spawn')
    svc_name = args.controller_manager + '/list_controllers' if not args.controller_manager.endswith('/list_controllers') else args.controller_manager
    # Ensure svc_name is absolute
    if not svc_name.startswith('/'):
        svc_name = '/' + svc_name

    client = node.create_client(ListControllers, svc_name)
    start_time = time.time()
    node.get_logger().info(f'Waiting for service {svc_name} (timeout={args.timeout}s)')
    while not client.wait_for_service(timeout_sec=1.0):
        if time.time() - start_time > args.timeout:
            node.get_logger().error(f'Timeout waiting for service {svc_name}')
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(2)
        node.get_logger().info(f'service {svc_name} not available yet...')

    # Optionally ensure controller manager responds
    try:
        fut = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    except Exception:
        pass

    node.get_logger().info(f'Service {svc_name} available — spawning controllers: {args.controllers}')
    # Spawn controllers via the controller_manager spawner executable
    for ctrl in args.controllers:
        cmd = ['ros2', 'run', 'controller_manager', 'spawner', ctrl, '--controller-manager', args.controller_manager]
        node.get_logger().info(f'Running: {cmd}')
        try:
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                node.get_logger().warn(f'Spawning {ctrl} exited with {proc.returncode}')
        except FileNotFoundError:
            node.get_logger().error('ros2 CLI not found in PATH; cannot spawn controllers')
            node.destroy_node()
            rclpy.shutdown()
            sys.exit(3)

    node.get_logger().info('Finished spawning controllers')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
