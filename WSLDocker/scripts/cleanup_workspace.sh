#!/bin/bash
set -euo pipefail

# old logs
find /tmp -maxdepth 1 -type f \( -name '*gazebo*' -o -name '*dual_gazebo*' -o -name '*ros2*' \) -mtime +7 -delete
find /workspaces/omx_ros2/logs -type f -mtime +7 -delete 2>/dev/null || true
find /workspaces/omx_ros2/ws/log -type f -mtime +7 -delete 2>/dev/null || true

# build artifacts
rm -rf /workspaces/omx_ros2/ws/build/* 2>/dev/null || true
rm -rf /workspaces/omx_ros2/ws/log/* 2>/dev/null || true
rm -rf /workspaces/omx_ros2/ws/install/* 2>/dev/null || true

# caches
rm -rf ~/.ros/log/* 2>/dev/null || true
rm -rf ~/.gazebo/* 2>/dev/null || true
rm -rf /home/${USER:-root}/.cache/gazebo/* 2>/dev/null || true
find /workspaces/omx_ros2 -type d -name '__pycache__' -exec rm -rf {} +
find /workspaces/omx_ros2 -type f -name '*.pyc' -delete

echo 'cleanup complete'