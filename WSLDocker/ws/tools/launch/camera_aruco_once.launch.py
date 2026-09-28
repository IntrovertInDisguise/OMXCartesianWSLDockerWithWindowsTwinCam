#!/usr/bin/env python3
"""
Launch a root-namespaced RealSense camera and run camera_aruco.py once.

Examples:
    ros2 launch tools/launch/camera_aruco_once.launch.py
    ros2 launch tools/launch/camera_aruco_once.launch.py depth_camera_serial_no:=<serial>
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    script_path = os.path.join(repo_root, "tools", "camera_aruco.py")

    depth_camera_serial_no = LaunchConfiguration("depth_camera_serial_no")
    capture_timeout_s = LaunchConfiguration("capture_timeout_s")
    startup_delay_s = LaunchConfiguration("startup_delay_s")

    depth_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [
                        FindPackageShare("realsense2_camera"),
                        "launch",
                        "rs_launch.py",
                    ]
                ),
            ]
        ),
        launch_arguments={
            "camera_namespace": "/",
            "align_depth.enable": "true",
            "serial_no": depth_camera_serial_no,
        }.items(),
    )

    capture_once = TimerAction(
        period=startup_delay_s,
        actions=[
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    script_path,
                    "--print-once",
                    "--capture-timeout-s",
                    capture_timeout_s,
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "depth_camera_serial_no",
                default_value="",
                description="Optional RealSense serial number forwarded to rs_launch.py",
            ),
            DeclareLaunchArgument(
                "capture_timeout_s",
                default_value="8.0",
                description="How long camera_aruco.py waits for an RGBD frame bundle",
            ),
            DeclareLaunchArgument(
                "startup_delay_s",
                default_value="4.0",
                description="Delay before the one-shot ArUco capture runs",
            ),
            depth_camera,
            capture_once,
        ]
    )