#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Start the RealSense ROS 2 driver in the root /camera/* namespace.

Run this outside the devcontainer / outer WSL environment where
the realsense2_camera ROS 2 package is installed.

Usage:
  scripts/start_realsense_root_ns.sh [--serial SERIAL] [--ros-domain-id ID] [-- EXTRA_LAUNCH_ARGS...]

Options:
  --serial SERIAL        Optional RealSense serial number.
  --ros-domain-id ID     Override ROS_DOMAIN_ID for this launch only.
  -h, --help             Show this help.

Examples:
  scripts/start_realsense_root_ns.sh
  scripts/start_realsense_root_ns.sh --serial 234322301234
  scripts/start_realsense_root_ns.sh --ros-domain-id 7 -- pointcloud.enable:=false

Environment:
  ROS_SETUP_BASH         Defaults to /opt/ros/humble/setup.bash
  DEPTH_CAMERA_SERIAL_NO Used when --serial is not supplied
EOF
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

serial_no="${DEPTH_CAMERA_SERIAL_NO:-}"
ros_setup_bash="${ROS_SETUP_BASH:-/opt/ros/humble/setup.bash}"
ros_domain_id_override=""
extra_launch_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_launch_args+=("$@")
      break
      ;;
    *)
      extra_launch_args+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$ros_setup_bash" ]]; then
  echo "ROS setup file not found: $ros_setup_bash" >&2
  exit 1
fi

source_setup_file "$ros_setup_bash"

if [[ -n "$ros_domain_id_override" ]]; then
  export ROS_DOMAIN_ID="$ros_domain_id_override"
fi

if ! ros2 pkg prefix realsense2_camera >/dev/null 2>&1; then
  echo "ROS package 'realsense2_camera' not found." >&2
  echo "Run this script on the host / outer WSL where the RealSense ROS driver is installed." >&2
  exit 1
fi

launch_args=(
  "camera_namespace:=/"
  "align_depth.enable:=true"
)

if [[ -n "$serial_no" ]]; then
  launch_args+=("serial_no:=${serial_no}")
fi

if [[ ${#extra_launch_args[@]} -gt 0 ]]; then
  launch_args+=("${extra_launch_args[@]}")
fi

echo "Launching realsense2_camera on /camera/*"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
if [[ -n "$serial_no" ]]; then
  echo "serial_no=${serial_no}"
fi

exec ros2 launch realsense2_camera rs_launch.py "${launch_args[@]}"