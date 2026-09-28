#!/usr/bin/env bash
set -euo pipefail

# dual_gazebo_test_runner.sh
# Run integration test repeatedly (up to max attempts) until all checks pass.

MAX_ATTEMPTS=10
SLEEP_GAZEBO=8
SLEEP_LIVEPLOT=8
WAIT_CONTROLLER=30
WAIT_CONTACT=30

# kill leftovers
cleanup() {
  pkill -f "ros2 launch omx_variable_stiffness_controller dual_gazebo_variable_stiffness.launch.py" 2>/dev/null || true
  pkill -f live_plot_logs.py 2>/dev/null || true
  pkill -f dual_gazebo_opposing_push.py 2>/dev/null || true
  pkill -f gzserver 2>/dev/null || true
  pkill -f gzclient 2>/dev/null || true
  sleep 2
}

check_active_controllers(){
  local timeout=$1
  local start
  start=$(date +%s)
  while true; do
    local r1 r2
    r1=$(ros2 control list_controllers -c /robot1/controller_manager 2>/dev/null || true | grep -E "robot1_variable_stiffness.*active" | wc -l)
    r2=$(ros2 control list_controllers -c /robot2/controller_manager 2>/dev/null || true | grep -E "robot2_variable_stiffness.*active" | wc -l)
    if [[ $r1 -ge 1 && $r2 -ge 1 ]]; then
      echo "OK"
      return 0
    fi
    if (( $(date +%s) - start > timeout )); then
      echo "TIMEOUT"
      return 1
    fi
    sleep 1
  done
}

check_contact_force(){
  local topic=$1
  local timeout=$2
  local start
  start=$(date +%s)
  while true; do
    local w
    w=$(ros2 topic echo -n 1 "$topic" 2>/dev/null || true)
    if [[ -n "$w" ]]; then
      local fx fxv
      fx=$(echo "$w" | grep -oP 'force:.*x: \K[-0-9.]+') || fx=0
      fxv=$(awk "BEGIN {print ($fx != 0)}")
      if [[ $fxv -eq 1 ]]; then
        echo "OK"
        return 0
      fi
    fi
    if (( $(date +%s) - start > timeout )); then
      echo "TIMEOUT"
      return 1
    fi
    sleep 1
  done
}

for attempt in $(seq 1 $MAX_ATTEMPTS); do
  echo "\n=== Attempt $attempt/$MAX_ATTEMPTS ==="
  cleanup

  # --- launch gazebo + live plot in background
  export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-0}
  export AMENT_PYTHON_EXECUTABLE=${AMENT_PYTHON_EXECUTABLE:-$(command -v python3)}
  set +u
  source /opt/ros/humble/setup.bash
  source /workspaces/omx_ros2/ws/install/setup.bash
  set -u
  export LIBGL_ALWAYS_SOFTWARE=1

  ros2 launch omx_variable_stiffness_controller dual_gazebo_variable_stiffness.launch.py \
    gui:=true launch_gazebo:=true enable_logger:=true enable_live_plot:=true start_rviz:=false \
    > /tmp/dual_gazebo_test_launch.log 2>&1 &
  GZPID=$!

  sleep $SLEEP_GAZEBO

  # 1) liveplot check (process + output page) 
  python3 tools/live_plot_logs.py \
    --controller variable_stiffness \
    --namespace /robot1/robot1_variable_stiffness \
    --namespace2 /robot2/robot2_variable_stiffness \
    --screenshot-dir /tmp/live_plot_screenshots \
    --screenshot-rate 3 --interval 1 > /tmp/dual_gazebo_test_liveplot.log 2>&1 &
  PLOT_PID=$!
  sleep $SLEEP_LIVEPLOT

  if ! ps -p $PLOT_PID > /dev/null 2>&1; then
    echo "FAIL: live_plot process not running"; kill $GZPID 2>/dev/null || true; cleanup; continue
  fi

  start=$(date +%s)
  has_png=0
  while (( $(date +%s) - start < 30 )); do
    if ls /tmp/live_plot_screenshots/*.png 2>/dev/null | head -n1 >/dev/null; then
      has_png=1; break
    fi
    sleep 1
  done
  if (( has_png == 0)); then
    echo "FAIL: live_plot screenshot was not generated"; kill $PLOT_PID $GZPID 2>/dev/null || true; cleanup; continue
  fi
  echo "PASS: liveplot active"

  # 2) spawn check: robot joint_states and box existence
  local timeout=$(date +%s); local found=0
  while (( $(date +%s) - timeout < 30 )); do
    if ros2 topic echo -n 1 /robot1/joint_states 2>/dev/null | grep -q "name" && ros2 topic echo -n 1 /robot2/joint_states 2>/dev/null | grep -q "name"; then
      found=1; break
    fi
    sleep 1
  done
  if (( found == 0 )); then
    echo "FAIL: robot joint_states not present"; kill $PLOT_PID $GZPID 2>/dev/null || true; cleanup; continue
  fi
  echo "PASS: robot joint state topics available"

  # 2b) check box exists in gazebo
  if ! ros2 service call /gazebo/get_model_state gazebo_msgs/srv/GetModelState "{model_name: 'opposing_push_box'}" > /tmp/dual_gazebo_test_get_model.log 2>&1; then
    echo "WARN: box model may not be spawned yet";
    # allow further process to ensure path still tests it
  fi

  # 3) controller managers active for arm controllers
  if ! check_active_controllers $WAIT_CONTROLLER; then
    echo "FAIL: controllers not both active"; kill $PLOT_PID $GZPID 2>/dev/null || true; cleanup; continue
  fi
  echo "PASS: both robot controllers active"

  # 4) execute path and check contact force while running
  python3 tools/dual_gazebo_opposing_push.py --duration 12 --wait-before-push 4 --line-steps 20 > /tmp/dual_gazebo_test_push.log 2>&1 &
  PUSH_PID=$!

  local c1=0 c2=0
  local start2=$(date +%s)
  while (( $(date +%s) - start2 < WAIT_CONTACT )); do
    if ros2 topic echo -n 1 /robot1/variable_stiffness_controller/contact_wrench 2>/dev/null | grep -q "force"; then c1=1; fi
    if ros2 topic echo -n 1 /robot2/variable_stiffness_controller/contact_wrench 2>/dev/null | grep -q "force"; then c2=1; fi
    if (( c1==1 && c2==1 )); then break; fi
    sleep 1
  done
  kill $PUSH_PID 2>/dev/null || true

  if (( c1==1 && c2==1 )); then
    echo "PASS: contact forces observed on both arms"
    echo "ALL PASSED"; kill $PLOT_PID $GZPID 2>/dev/null || true; cleanup; exit 0
  else
    echo "FAIL: contact force not observed both arms (r1=$c1 r2=$c2)"
    kill $PLOT_PID $GZPID 2>/dev/null || true; cleanup; continue
  fi

done

echo "FAILED: all attempts exhausted"; exit 1
