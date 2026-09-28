#!/usr/bin/env bash
set -euo pipefail

# dual_gazebo_full_test.sh
# Run dual Gazebo GUI + live plot + box in midpoint + opposing push path.
# Usage:
#   bash tools/dual_gazebo_full_test.sh

# Setup environment
export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-0}
export AMENT_PYTHON_EXECUTABLE=${AMENT_PYTHON_EXECUTABLE:-$(command -v python3)}
set +u
source /opt/ros/humble/setup.bash
source /workspaces/omx_ros2/ws/install/setup.bash
set -u
export LIBGL_ALWAYS_SOFTWARE=1

# Demo parameters
DURATION=${1:-20}
WAIT_BEFORE_PUSH=${2:-12}
LIVE_PLOT_DIR=/tmp/live_plot_screenshots

# Start Gazebo-based dual launch
ros2 launch omx_variable_stiffness_controller dual_gazebo_variable_stiffness.launch.py \
  gui:=true launch_gazebo:=true enable_logger:=true enable_live_plot:=true start_rviz:=false > /tmp/dual_gazebo_full_test_launch.log 2>&1 &
GZPID=$!

# Start live plot tool in foreground mode with screenshots
python3 tools/live_plot_logs.py \
  --controller variable_stiffness \
  --namespace /robot1/robot1_variable_stiffness \
  --namespace2 /robot2/robot2_variable_stiffness \
  --window 60 --interval 0.5 \
  --screenshot-dir $LIVE_PLOT_DIR --screenshot-rate 3 > /tmp/dual_gazebo_full_test_live_plot.log 2>&1 &
PLOT_PID=$!

# Wait for startup and run opposing push
sleep 20
python3 tools/dual_gazebo_opposing_push.py --duration $DURATION --wait-before-push $WAIT_BEFORE_PUSH --line-steps 18 > /tmp/dual_gazebo_full_test_push.log 2>&1 || true

# Cleanup
kill $PLOT_PID 2>/dev/null || true
kill $GZPID 2>/dev/null || true
sleep 2
pkill -f gzserver || true
pkill -f gzclient || true

echo "dual_gazebo_full_test complete"
