#!/usr/bin/env bash
# One-shot manual stable run for dual Open Manipulator X
set -euo pipefail

ROBOT1=${1:-robot1}
ROBOT2=${2:-robot2}
TIMEOUT=${TIMEOUT:-60}
GZ_LOG=${GZ_LOG:-/tmp/gzserver.log}

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "Sourcing ROS environment (if available)..."
if [ -f ws/install/setup.bash ]; then
  # Preferred workspace install
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash || true
  # shellcheck disable=SC1091
  source ws/install/setup.bash || true
elif [ -f install/setup.bash ]; then
  # fallback
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash || true
  # shellcheck disable=SC1091
  source install/setup.bash || true
else
  echo "Warning: could not find workspace install/setup.bash; ensure you source ROS2 and workspace before running." >&2
fi

export LIBGL_ALWAYS_SOFTWARE=1
export SDL_AUDIODRIVER=dummy

echo "Cleaning previous Gazebo processes..."
pkill -f gzserver || true
pkill -f gzclient || true
sleep 0.5

echo "Starting gzserver with minimal ROS plugins (logs: $GZ_LOG)"
nohup gzserver -s libgazebo_ros_init.so -s libgazebo_ros_factory.so -s libgazebo_ros_state.so > "$GZ_LOG" 2>&1 &
GZ_PID=$!
echo "gzserver PID $GZ_PID"

wait_for_topic() {
  TOPIC=$1
  TO=$2
  echo "Waiting for topic/service $TOPIC (timeout ${TO}s)..."
  SECONDS=0
  while [ $SECONDS -lt $TO ]; do
    if ros2 topic list 2>/dev/null | grep -x "$TOPIC" >/dev/null 2>&1; then
      echo "$TOPIC present"
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for $TOPIC" >&2
  return 1
}

# Wait briefly for Gazebo to initialize; wait_and_spawn.sh will perform robust waits too
if ! wait_for_topic /gazebo/model_states "$TIMEOUT"; then
  echo "/gazebo/model_states not visible after $TIMEOUT s; check $GZ_LOG" >&2
  exit 2
fi

# Ensure xacro is available before attempting to render URDFs
if ! command -v xacro >/dev/null 2>&1; then
  echo "xacro not found in PATH. Install ROS2 xacro package or ensure xacro is available." >&2
  exit 3
fi

WAIT_SPAWN="$ROOT_DIR/ws/src/omx_variable_stiffness_controller/scripts/wait_and_spawn.sh"
if [ ! -x "$WAIT_SPAWN" ]; then
  # If script exists but not executable, try to call with bash
  if [ -f "$WAIT_SPAWN" ]; then
    echo "Found wait_and_spawn.sh at $WAIT_SPAWN"
  else
    echo "Could not find wait_and_spawn.sh at $WAIT_SPAWN" >&2
    exit 4
  fi
fi

echo "Spawning $ROBOT1..."
bash "$WAIT_SPAWN" "$ROBOT1" /gazebo/model_states "$TIMEOUT" 0 0 0 0 || {
  echo "Failed to spawn $ROBOT1" >&2
  exit 5
}

echo "Spawning $ROBOT2..."
bash "$WAIT_SPAWN" "$ROBOT2" /gazebo/model_states "$TIMEOUT" 1.0 0 0 0 || {
  echo "Failed to spawn $ROBOT2" >&2
  exit 6
}

echo "Spawning controllers for $ROBOT1 and $ROBOT2"
python3 "$ROOT_DIR/tools/wait_for_cm_and_spawn.py" --namespace /$ROBOT1 --controller-manager /$ROBOT1/controller_manager --controllers joint_state_broadcaster ${ROBOT1}_variable_stiffness || true
python3 "$ROOT_DIR/tools/wait_for_cm_and_spawn.py" --namespace /$ROBOT2 --controller-manager /$ROBOT2/controller_manager --controllers joint_state_broadcaster ${ROBOT2}_variable_stiffness || true

echo "Waiting for joint_states topics..."
if ! wait_for_topic /$ROBOT1/joint_states 30; then
  echo "Timeout waiting for /$ROBOT1/joint_states" >&2
  exit 7
fi
if ! wait_for_topic /$ROBOT2/joint_states 30; then
  echo "Timeout waiting for /$ROBOT2/joint_states" >&2
  exit 8
fi

echo "Running hardware harness (tools/hardware_harness.py)"
python3 "$ROOT_DIR/tools/hardware_harness.py"

echo "Manual stable run completed. If you need to stop gzserver: kill $GZ_PID" 
