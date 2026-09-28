#!/usr/bin/env bash
set -euo pipefail

# dual_gazebo_force_test.sh
# Runs dual Gazebo + GUI + live_plot + logger + ee_force sensor + obstacle + waypoints.
# Writes logs to /tmp and results to /workspaces/omx_ros2/logs/variable_stiffness/dual_gazebo/<timestamp>.

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR="/workspaces/omx_ros2/logs/variable_stiffness/dual_gazebo/$TIMESTAMP"
mkdir -p "$DATA_DIR"

# In this environment, setup scripts may assume AMENT_TRACE_SETUP_FILES.
export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-0}
set +u
source /opt/ros/humble/setup.bash
source /workspaces/omx_ros2/ws/install/setup.bash
set -u
export LIBGL_ALWAYS_SOFTWARE=1

# Cleanup previous
pkill -f 'ros2 launch' || true
pkill -f gzserver || true
pkill -f gzclient || true
sleep 2

LAUNCH_LOG="/tmp/dual_gazebo_$TIMESTAMP.log"
start_gazebo_launch() {
  local gui=$1
  echo "[dual_gazebo_force_test.sh] Starting Gazebo launch (gui=$gui)"
  local cmd=(ros2 launch omx_variable_stiffness_controller dual_gazebo_variable_stiffness.launch.py \
    gui:=$gui enable_logger:=true enable_live_plot:=true start_rviz:=false)

  if [ -z "${DISPLAY:-}" ] && command -v xvfb-run >/dev/null 2>&1; then
    echo "[dual_gazebo_force_test.sh] Using xvfb-run because DISPLAY is unset"
    nohup xvfb-run --auto-servernum --server-args='-screen 0 1280x1024x24 +extension GLX +render -noreset' "${cmd[@]}" > "$LAUNCH_LOG" 2>&1 &
  else
    echo "[dual_gazebo_force_test.sh] Running without xvfb-run (DISPLAY=${DISPLAY:-unset})"
    nohup "${cmd[@]}" > "$LAUNCH_LOG" 2>&1 &
  fi
  GZLAUNCH_PID=$!
  echo $GZLAUNCH_PID > /tmp/dual_gazebo_pid
}

# Start with GUI first, but fallback to headless if gzcient or gzserver dies quickly.
start_gazebo_launch true
sleep 20

# Check whether gzcient/gzserver are alive; if gzclient died, re-run without GUI.
if ! pgrep -f gzserver >/dev/null 2>&1; then
  echo "[dual_gazebo_force_test.sh] WARN: gzserver terminated early, retrying with gui=false"
  pkill -f 'ros2 launch' || true
  pkill -f gzclient || true
  pkill -f gzserver || true
  start_gazebo_launch false
  sleep 20
elif ! pgrep -f gzclient >/dev/null 2>&1; then
  echo "[dual_gazebo_force_test.sh] WARN: gzclient died; switching to gui=false mode"
  pkill -f 'ros2 launch' || true
  pkill -f gzclient || true
  pkill -f gzserver || true
  start_gazebo_launch false
  sleep 20
fi

# Spawn obstacle
python3 /workspaces/omx_ros2/tools/spawn_box.py --name obs_dual --x 0.16 --y 0.08 --z 0.10 --sx 0.12 --sy 0.12 --sz 0.12

# Start ee force sensors
python3 /workspaces/omx_ros2/tools/ee_force_sensor.py -n /robot1 --csv "$DATA_DIR/robot1_ee_force.csv" > "/tmp/robot1_ee_force_$TIMESTAMP.log" 2>&1 &
PID_EE_R1=$!
python3 /workspaces/omx_ros2/tools/ee_force_sensor.py -n /robot2 --csv "$DATA_DIR/robot2_ee_force.csv" > "/tmp/robot2_ee_force_$TIMESTAMP.log" 2>&1 &
PID_EE_R2=$!

sleep 5

# Publish waypoints
python3 - <<'PY'
import time, random, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
rclpy.init(); node=Node('dual_waypoint_runner')
pub1=node.create_publisher(PoseStamped, '/robot1/robot1_variable_stiffness/waypoint_command', 10)
pub2=node.create_publisher(PoseStamped, '/robot2/robot2_variable_stiffness/waypoint_command', 10)
for i in range(12):
    for pub, y0 in [(pub1, 0.08), (pub2, -0.08)]:
        msg = PoseStamped(); msg.header.frame_id='offset'
        msg.pose.position.x = 0.03 + random.uniform(-0.02, 0.04)
        msg.pose.position.y = y0 + random.uniform(-0.02, 0.02)
        msg.pose.position.z = 0.01 + random.uniform(-0.02, 0.02)
        msg.pose.orientation.w = 1.0
        for _ in range(4): pub.publish(msg)
    node.get_logger().info(f'cycle {i+1}')
    time.sleep(1.0)
node.destroy_node(); rclpy.shutdown()
PY

sleep 30

# Collect and print summary status
echo '--- data location:' $DATA_DIR
ls -l "$DATA_DIR"

tail -n 8 "/tmp/dual_gazebo_$TIMESTAMP.log" || true
tail -n 8 "/tmp/robot1_ee_force_$TIMESTAMP.log" || true
tail -n 8 "/tmp/robot2_ee_force_$TIMESTAMP.log" || true

# cleanup
kill $PID_EE_R1 $PID_EE_R2 || true
pkill -f 'ros2 launch' || true
pkill -f gzserver || true
pkill -f gzclient || true

echo 'done (timestamp: ' $TIMESTAMP ')'
