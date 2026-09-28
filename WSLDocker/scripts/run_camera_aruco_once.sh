#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Run the one-shot ArUco world-frame query inside the devcontainer.

This script can either use an already-running camera driver or start one in the
same shell before the one-shot query.

Expected camera topics:
  /camera/color/image_raw
  /camera/color/camera_info
  /camera/aligned_depth_to_color/image_raw
  /camera/aligned_depth_to_color/camera_info

Usage:
  scripts/run_camera_aruco_once.sh [--start-driver] [--serial SERIAL] [--ros-domain-id ID]
                                   [--capture-timeout-s SEC] [--driver-ready-timeout-s SEC]
                                   [--startup-delay-s SEC] [--no-self-check] [-- CAMERA_ARUCO_ARGS...]

Options:
  --start-driver           Start realsense2_camera in the background before capture.
  --serial SERIAL         Optional RealSense serial number used when --start-driver is set.
  --ros-domain-id ID      Override ROS_DOMAIN_ID for this script and any started driver.
  --capture-timeout-s SEC  Timeout for the one-shot RGBD capture. Default: 5.0
  --driver-ready-timeout-s SEC
                          Timeout while waiting for camera topics to publish after --start-driver. Default: 15.0
  --startup-delay-s SEC   Extra delay after driver launch before readiness checks. Default: 2.0
  --no-self-check          Skip the lightweight readiness report before capture.
  -h, --help               Show this help.

Examples:
  scripts/run_camera_aruco_once.sh
  scripts/run_camera_aruco_once.sh --start-driver
  scripts/run_camera_aruco_once.sh --start-driver --serial 234322301234
  scripts/run_camera_aruco_once.sh --capture-timeout-s 8.0
  scripts/run_camera_aruco_once.sh -- --world-marker-id 10 --marker ground_world:10 --marker spring_cap_robot1:14

Environment:
  ROS_SETUP_BASH         Defaults to /opt/ros/humble/setup.bash
  ARUCO_PYTHON_BIN       Defaults to /usr/bin/python3
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
camera_script="$repo_root/tools/camera_aruco.py"
ros_setup_bash="${ROS_SETUP_BASH:-/opt/ros/humble/setup.bash}"
workspace_setup_bash="$repo_root/ws/install/setup.bash"
python_bin="${ARUCO_PYTHON_BIN:-/usr/bin/python3}"
capture_timeout_s="5.0"
driver_ready_timeout_s="15.0"
startup_delay_s="2.0"
run_self_check=1
start_driver=0
serial_no="${DEPTH_CAMERA_SERIAL_NO:-}"
ros_domain_id_override=""
driver_pid=""
camera_args=()

cleanup() {
  if [[ -n "$driver_pid" ]]; then
    if kill -0 "$driver_pid" >/dev/null 2>&1; then
      echo "Stopping background RealSense driver (pid $driver_pid)..."
      kill "$driver_pid" >/dev/null 2>&1 || true
      wait "$driver_pid" >/dev/null 2>&1 || true
    fi
  fi
}

source_setup_file() {
  local setup_file="$1"
  local nounset_enabled=0
  if [[ -o nounset ]]; then
    nounset_enabled=1
    set +u
  fi
  # shellcheck disable=SC1090
  source "$setup_file"
  if [[ "$nounset_enabled" -eq 1 ]]; then
    set -u
  fi
}

wait_for_topic_once() {
  local topic_name="$1"
  local timeout_s="$2"
  if ! timeout "$timeout_s" ros2 topic echo --once "$topic_name" >/dev/null 2>&1; then
    echo "Timed out waiting for topic publication: $topic_name" >&2
    return 1
  fi
}

start_driver_if_requested() {
  local launch_args=()

  [[ "$start_driver" -eq 1 ]] || return 0

  if ! ros2 pkg prefix realsense2_camera >/dev/null 2>&1; then
    echo "ROS package 'realsense2_camera' not found in this environment." >&2
    echo "Auto-start cannot run here. Use scripts/start_realsense_root_ns.sh on the host / outer WSL instead." >&2
    return 1
  fi

  launch_args=(
    "camera_namespace:=/"
    "align_depth.enable:=true"
  )
  if [[ -n "$serial_no" ]]; then
    launch_args+=("serial_no:=${serial_no}")
  fi

  echo "Starting background RealSense driver on /camera/* ..."
  ros2 launch realsense2_camera rs_launch.py "${launch_args[@]}" &
  driver_pid="$!"

  trap cleanup EXIT

  if [[ "$startup_delay_s" != "0" && "$startup_delay_s" != "0.0" ]]; then
    echo "Waiting ${startup_delay_s}s for driver startup..."
    sleep "$startup_delay_s"
  fi

  echo "Waiting for camera topics to publish..."
  wait_for_topic_once "/camera/color/camera_info" "$driver_ready_timeout_s"
  wait_for_topic_once "/camera/aligned_depth_to_color/camera_info" "$driver_ready_timeout_s"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-driver)
      start_driver=1
      shift
      ;;
    --serial)
      [[ $# -ge 2 ]] || { echo "--serial requires a value" >&2; exit 2; }
      serial_no="$2"
      shift 2
      ;;
    --ros-domain-id)
      [[ $# -ge 2 ]] || { echo "--ros-domain-id requires a value" >&2; exit 2; }
      ros_domain_id_override="$2"
      shift 2
      ;;
    --capture-timeout-s)
      [[ $# -ge 2 ]] || { echo "--capture-timeout-s requires a value" >&2; exit 2; }
      capture_timeout_s="$2"
      shift 2
      ;;
    --driver-ready-timeout-s)
      [[ $# -ge 2 ]] || { echo "--driver-ready-timeout-s requires a value" >&2; exit 2; }
      driver_ready_timeout_s="$2"
      shift 2
      ;;
    --startup-delay-s)
      [[ $# -ge 2 ]] || { echo "--startup-delay-s requires a value" >&2; exit 2; }
      startup_delay_s="$2"
      shift 2
      ;;
    --no-self-check)
      run_self_check=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      camera_args+=("$@")
      break
      ;;
    *)
      camera_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$ros_setup_bash" ]]; then
  echo "ROS setup file not found: $ros_setup_bash" >&2
  exit 1
fi

if [[ ! -f "$workspace_setup_bash" ]]; then
  echo "Workspace overlay not found: $workspace_setup_bash" >&2
  echo "Build the workspace first so camera_aruco.py can resolve repo packages cleanly." >&2
  exit 1
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Python interpreter not executable: $python_bin" >&2
  exit 1
fi

if [[ ! -f "$camera_script" ]]; then
  echo "camera_aruco.py not found: $camera_script" >&2
  exit 1
fi

source_setup_file "$ros_setup_bash"
source_setup_file "$workspace_setup_bash"

if [[ -n "$ros_domain_id_override" ]]; then
  export ROS_DOMAIN_ID="$ros_domain_id_override"
fi

start_driver_if_requested

if [[ "$run_self_check" -eq 1 ]]; then
  echo "Running camera_aruco.py self-check..."
  "$python_bin" "$camera_script" --self-check "${camera_args[@]}"
fi

echo "Running one-shot ArUco world-frame query..."
"$python_bin" "$camera_script" \
  --print-once \
  --capture-timeout-s "$capture_timeout_s" \
  "${camera_args[@]}"