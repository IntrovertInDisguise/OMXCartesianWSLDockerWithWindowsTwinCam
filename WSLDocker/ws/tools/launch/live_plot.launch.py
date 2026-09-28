#!/usr/bin/env python3
"""
Launch file for the live timeseries plotter.

Runs live_plot_logs.py as a standalone process.  Subscribes directly to
controller topics — no log files are read or written.

Parameters (override with ros2 launch ... key:=value):
    controller  — 'variable_stiffness' or 'gravity_comp'
    namespace   — controller topic namespace for robot 1
    namespace2  — controller topic namespace for robot 2 (enables dual mode)
    window      — rolling time window in seconds (default: 30.0)
    interval    — plot refresh interval in seconds (default: 0.5)

Examples:
    # VS single
    ros2 launch tools/launch/live_plot.launch.py \\
        controller:=variable_stiffness namespace:=/omx/variable_stiffness_controller

    # GC dual
    ros2 launch tools/launch/live_plot.launch.py \\
        controller:=gravity_comp \\
        namespace:=/robot1 namespace2:=/robot2
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    script_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'live_plot_logs.py')
    )

    return LaunchDescription([
        DeclareLaunchArgument('controller',  default_value='variable_stiffness'),
        DeclareLaunchArgument('namespace',   default_value=''),
        DeclareLaunchArgument('namespace2',  default_value=''),
        DeclareLaunchArgument('window',      default_value='30.0'),
        DeclareLaunchArgument('interval',    default_value='0.5'),
        ExecuteProcess(
            cmd=[
                'python3', script_path,
                '--controller', LaunchConfiguration('controller'),
                '--namespace',  LaunchConfiguration('namespace'),
                '--namespace2', LaunchConfiguration('namespace2'),
                '--window',     LaunchConfiguration('window'),
                '--interval',   LaunchConfiguration('interval'),
            ],
            output='screen',
        ),
    ])
