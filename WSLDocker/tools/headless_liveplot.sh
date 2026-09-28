#!/usr/bin/env bash
# Helper: run dual Gazebo headless and live-plot to PNG screenshots
# Usage: ./tools/headless_liveplot.sh [--screenshot-dir /tmp/live_plot] [--controller variable_stiffness] [--ns1 /robot1/robot1_variable_stiffness] [--ns2 /robot2/robot2_variable_stiffness]

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCREENSHOT_DIR=/tmp/live_plot_screenshots
CONTROLLER=variable_stiffness
NS1=/robot1/robot1_variable_stiffness
NS2=/robot2/robot2_variable_stiffness

while [[ $# -gt 0 ]]; do
  case "$1" in
    --screenshot-dir) SCREENSHOT_DIR="$2"; shift 2;;
    --controller) CONTROLLER="$2"; shift 2;;
    --ns1) NS1="$2"; shift 2;;
    --ns2) NS2="$2"; shift 2;;
    -h|--help) echo "Usage: $0 [--screenshot-dir DIR] [--controller variable_stiffness|gravity_comp] [--ns1 NS] [--ns2 NS]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "Starting headless Gazebo + live-plot (screenshots -> $SCREENSHOT_DIR)"

# Ensure ROS env is sourced (allow setup scripts to reference unset vars)
set +u
source /opt/ros/humble/setup.bash || true
if [ -f /workspaces/omx_ros2/ws/install/setup.bash ]; then
  source /workspaces/omx_ros2/ws/install/setup.bash || true
fi
set -u

# Environment for headless Gazebo
export LIBGL_ALWAYS_SOFTWARE=1
export SDL_AUDIODRIVER=dummy

# Launch Gazebo headless (gui:=false) in background
ros2 launch omx_variable_stiffness_controller dual_gazebo_variable_stiffness.launch.py gui:=false enable_logger:=true enable_live_plot:=false start_rviz:=false &
LAUNCH_PID=$!
echo "Launched Gazebo headless (pid=$LAUNCH_PID). Waiting 8s for bringup..."
sleep 8

# Run live plotter in headless mode saving screenshots
export LIVEPLOT_USE_GUI=0
mkdir -p "$SCREENSHOT_DIR"
python3 "$SCRIPT_DIR/live_plot_logs.py" \
  --controller "$CONTROLLER" \
  --namespace "$NS1" \
  --namespace2 "$NS2" \
  --screenshot-dir "$SCREENSHOT_DIR" \
  --screenshot-rate 2.0 &
PLOT_PID=$!
echo "Live-plot running (pid=$PLOT_PID). Screenshots saving to $SCREENSHOT_DIR"

echo "To stop: kill $PLOT_PID and then kill $LAUNCH_PID (or Ctrl-C this script)."

wait $PLOT_PID || true
echo "Live-plot exited. Cleaning up Gazebo (pid=$LAUNCH_PID)."
kill $LAUNCH_PID 2>/dev/null || true
