#!/usr/bin/env bash
# Convenience wrapper to ensure ROS 2 and the workspace overlay are sourced
# before running the `ros2` command. Usage: ./scripts/ros2.sh topic list

set -euo pipefail

# Source system ROS if available
if [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck source=/opt/ros/humble/setup.bash
  source /opt/ros/humble/setup.bash
fi

# Source workspace overlay if available
if [ -f /workspaces/omx_ros2/ws/install/setup.bash ]; then
  # shellcheck source=/workspaces/omx_ros2/ws/install/setup.bash
  source /workspaces/omx_ros2/ws/install/setup.bash
fi

if [ $# -eq 0 ]; then
  echo "Usage: $0 <ros2-args>"
  echo "Example: $0 topic list"
  exit 2
fi

exec ros2 "$@"
