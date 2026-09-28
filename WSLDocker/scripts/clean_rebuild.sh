#!/usr/bin/env bash
set -euo pipefail

echo "Sourcing system ROS2..."
if [ -f "/opt/ros/humble/setup.bash" ]; then
  set +u
  # shellcheck source=/dev/null
  source /opt/ros/humble/setup.bash || true
  set -u
else
  echo "Warning: /opt/ros/humble/setup.bash not found." >&2
fi

if [ -f "ws/install/setup.bash" ]; then
  echo "Sourcing workspace overlay..."
  set +u
  # shellcheck source=/dev/null
  source ws/install/setup.bash || true
  set -u
else
  echo "Workspace overlay not found at ws/install/setup.bash. You may need to run a build first." >&2
fi

echo "Running cleanup script if present..."
if [ -x "scripts/cleanup_workspace.sh" ]; then
  scripts/cleanup_workspace.sh || true
elif [ -f "scripts/cleanup_workspace.sh" ]; then
  bash scripts/cleanup_workspace.sh || true
else
  echo "No cleanup script found at scripts/cleanup_workspace.sh; skipping." 
fi

echo "System memory summary:"
free -h || true

usage(){
  echo "Usage: $0 [--build]"
  echo "  --build    Run 'colcon build --symlink-install' after cleanup"
}

if [ "${1-}" = "--help" ] || [ "${1-}" = "-h" ]; then
  usage
  exit 0
fi

if [ "${1-}" = "--build" ]; then
  echo "Starting colcon build (this may be memory intensive)."
  colcon build --symlink-install
  echo "Sourcing ws/install/setup.bash..."
  # shellcheck source=/dev/null
  if [ -f "ws/install/setup.bash" ]; then
    set +u
    source ws/install/setup.bash || true
    set -u
  fi
  echo "Build finished."
else
  echo "Done. To build, run: ./scripts/clean_rebuild.sh --build"
fi
