#!/usr/bin/env python3
"""Automated Gazebo + LivePlot + Logger test runner.

This script runs a sequence of Gazebo-based launch scenarios, verifies the
controller manager comes up, controllers are present, log files are written, and
that the launch processes do not crash.

Designed for: dual-robot variable stiffness / gravity compensation tests.

NOTE: This is intended for a desktop environment where Gazebo can run. In CI
environments without X, run with `--no-gui` (it will still launch gzserver).
"""

import argparse
import os
import pathlib
import signal
import subprocess
import sys
import time

import rclpy
from controller_manager_msgs.srv import ListControllers


def run_launch_and_verify(cmd, workspace, timeout=90.0, controllers_expected=None, namespace="/robot1"):
    # Start ros2 launch in a subprocess
    # Run inside a shell so we can source the ROS 2 and workspace setup scripts.
    ros_cmd = "source /opt/ros/humble/setup.bash && source {ws}/install/setup.bash && {cmd}".format(
        ws=workspace,
        cmd=" ".join(cmd),
    )
    proc = subprocess.Popen(['bash', '-lc', ros_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Use rclpy to wait for controller_manager service
    # rclpy must be initialized once per process; keep a single node for the entire script.
    global _rclpy_node
    if '_rclpy_node' not in globals():
        rclpy.init()
        _rclpy_node = rclpy.create_node('test_gazebo_dual')

    svc_name = f"{namespace}/controller_manager/list_controllers"
    client = _rclpy_node.create_client(ListControllers, svc_name)

    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=5)
            raise RuntimeError(f'Launch process exited early with code {proc.returncode}\nstdout:\n{out}\nstderr:\n{err}')
        if client.wait_for_service(timeout_sec=1.0):
            break
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError(f"Timeout waiting for service {svc_name}")

    req = ListControllers.Request()
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(_rclpy_node, fut, timeout_sec=5.0)
    if not fut.done() or fut.result() is None:
        raise RuntimeError('Failed to call list_controllers')

    active = [c.name for c in fut.result().controller if c.state == 'active']

    if controllers_expected:
        missing = [c for c in controllers_expected if c not in active]
        if missing:
            raise RuntimeError(f'Missing active controllers: {missing} (active: {active})')

    return proc

    return proc


def ensure_logs_exist(log_dir, min_files=1, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.isdir(log_dir) and len(list(pathlib.Path(log_dir).glob('**/*'))) >= min_files:
            return
        time.sleep(1.0)
    raise RuntimeError(f'No log files created in {log_dir} after {timeout}s')


def main():
    parser = argparse.ArgumentParser(description='Run Gazebo dual-robot tests with liveplot/logging.')
    parser.add_argument('--no-gui', action='store_true', help='Do not launch gzclient (headless).')
    parser.add_argument('--duration', type=float, default=60.0, help='How long to let each scenario run (seconds).')
    args = parser.parse_args()

    # Prefer the standard workspace location if it exists
    workspace = os.getenv('COLCON_WS')
    if not workspace:
        if os.path.isdir('/workspaces/omx_ros2/ws'):
            workspace = '/workspaces/omx_ros2/ws'
        else:
            workspace = os.getcwd()
    if not os.path.isdir(os.path.join(workspace, 'install')):
        raise RuntimeError('Workspace install directory not found; run colcon build or ensure the correct workspace path.')

    # scenarios to run
    scenarios = [
        {
            'name': 'dual_gravity_comp',
            'launch': ['ros2', 'launch', 'omx_dual_bringup', 'dual_gazebo_gravity_comp.launch.py',
                       'enable_logger:=true', 'enable_live_plot:=true',
                       'gazebo_mode:=dual',
                       f'start_gzclient:={"true" if not args.no_gui else "false"}',
                       'start_rviz:=false',
                       'launch_gazebo:=true'],
            'namespace': '/robot1',
            'controllers': ['joint_state_broadcaster', 'robot1_gravity_comp'],
            'log_dir': os.path.join(workspace, 'logs', 'dual_gravity_comp'),
        },
        {
            'name': 'dual_variable_stiffness',
            'launch': ['ros2', 'launch', 'omx_variable_stiffness_controller', 'dual_gazebo_variable_stiffness.launch.py',
                       'enable_logger:=true', 'enable_live_plot:=true',
                       f'start_gzclient:={"true" if not args.no_gui else "false"}',
                       'launch_gazebo:=true'],
            'namespace': '/robot1',
            'controllers': ['joint_state_broadcaster', 'robot1_variable_stiffness'],
            'log_dir': os.path.join(workspace, 'logs', 'dual_variable_stiffness'),
        },
    ]

    failures = []
    for s in scenarios:
        print(f"\n=== Running scenario: {s['name']} ===")
        proc = None
        try:
            proc = run_launch_and_verify(s['launch'], workspace, timeout=120.0, controllers_expected=s['controllers'], namespace=s['namespace'])
            print(f"Scenario {s['name']} launched successfully, waiting {args.duration}s...")
            time.sleep(args.duration)
            # verify logs
            ensure_logs_exist(s['log_dir'], min_files=2, timeout=30.0)
            print(f"Scenario {s['name']} produced logs in {s['log_dir']}")
        except Exception as e:
            failures.append((s['name'], str(e)))
            print(f"ERROR in scenario {s['name']}: {e}")
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    if failures:
        print('\n=== FAILURES ===')
        for name, msg in failures:
            print(f"{name}: {msg}\n")
        sys.exit(1)

    print('\nAll scenarios completed successfully.')


if __name__ == '__main__':
    try:
        main()
    finally:
        if '_rclpy_node' in globals():
            _rclpy_node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
