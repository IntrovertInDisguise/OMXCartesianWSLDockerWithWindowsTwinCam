#!/usr/bin/env bash
set -euo pipefail

# dual_gazebo_gui.sh
# Start dual-robot Gazebo in GUI mode with robust container-safe behavior.
# - Uses workspace environment setup, no need to source manually.
# - Falls back to Xvfb virtual display when DISPLAY is missing.
# - Retries on early failure up to MAX_RETRIES.

WS="/workspaces/omx_ros2/ws"
WRAPPER="/workspaces/omx_ros2/tools/launch_with_build.sh"
LAUNCH_PKG="omx_variable_stiffness_controller"
LAUNCH_FILE="dual_gazebo_variable_stiffness.launch.py"

MAX_RETRIES=3
RETRY_DELAY=5

# Optional override via env
USE_XVFB=${USE_XVFB:-auto}

if [ ! -f "$WRAPPER" ]; then
  echo "[dual_gazebo_gui.sh] ERROR: wrapper not found: $WRAPPER" >&2
  exit 1
fi

if [ ! -d "$WS" ]; then
  echo "[dual_gazebo_gui.sh] ERROR: workspace not found: $WS" >&2
  exit 1
fi

LAUNCH_ARGS=("$LAUNCH_PKG" "$LAUNCH_FILE" "gui:=true" "enable_live_plot:=false" "enable_controllers:=true")

run_command() {
  local cmd=("$WRAPPER" "--no-build" "--" "${LAUNCH_ARGS[@]}")
  if [ "$USE_XVFB" != "false" ] && [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
    echo "[dual_gazebo_gui.sh] Running under xvfb-run (DISPLAY is empty, GUI will not be host-visible)"
    xvfb-run --auto-servernum --server-args='-screen 0 1280x1024x24 +extension GLX +render -noreset' "${cmd[@]}"
  else
    echo "[dual_gazebo_gui.sh] Running direct ROS2 launch (DISPLAY=${DISPLAY:-unset})"
    "${cmd[@]}"
  fi
}

for attempt in $(seq 1 "$MAX_RETRIES"); do
  echo "[dual_gazebo_gui.sh] Attempt $attempt/$MAX_RETRIES"
  set +e
  run_command
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    echo "[dual_gazebo_gui.sh] Launch exited cleanly (status 0)"
    exit 0
  fi
  echo "[dual_gazebo_gui.sh] Launch failed with status $status"
  if [ "$attempt" -lt "$MAX_RETRIES" ]; then
    echo "[dual_gazebo_gui.sh] Retrying in $RETRY_DELAY seconds..."
    sleep "$RETRY_DELAY"
  else
    echo "[dual_gazebo_gui.sh] Reached max retries ($MAX_RETRIES)."
    exit "$status"
  fi
done
