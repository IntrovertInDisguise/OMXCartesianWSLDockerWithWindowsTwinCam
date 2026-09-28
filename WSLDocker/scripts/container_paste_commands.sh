#!/usr/bin/env bash
# Interactive container helper: wait for host recorder READY file and optionally run the harness.
# Usage: copy/paste into the container shell after sourcing the workspace or run directly.

set -euo pipefail

DEFAULT_TRIGGER=/mnt/c/tmp/omx_camera_trigger.txt
DEFAULT_READY=/mnt/c/tmp/realsense_ready.txt
DEFAULT_CALIB_JSON='logs/<calib_dir>/calib_result_*.json'

TRIGGER="$DEFAULT_TRIGGER"
READY="$DEFAULT_READY"
CALIB_JSON="$DEFAULT_CALIB_JSON"
WAIT_READY=false
AUTO_RUN=false
SPECIMEN=""
TIMEOUT=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --trigger PATH       Trigger file inside container (default: $DEFAULT_TRIGGER)
  --ready PATH         Recorder READY file inside container (default: $DEFAULT_READY)
  --wait-ready         Wait for READY file before proceeding
  --auto-run           After READY appears, prompt to auto-run the harness
  --specimen NAME      Pass --spring-specimen NAME to the harness when auto-running
  --calib PATH         Calibration JSON glob (default: $DEFAULT_CALIB_JSON)
  --timeout N          Timeout in seconds when waiting for READY (0 = infinite)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trigger) TRIGGER="$2"; shift 2;;
    --ready) READY="$2"; shift 2;;
    --wait-ready) WAIT_READY=true; shift;;
    --auto-run) AUTO_RUN=true; shift;;
    --specimen) SPECIMEN="$2"; shift 2;;
    --calib) CALIB_JSON="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

echo "Sourcing ROS2 and workspace environment..."
source /opt/ros/humble/setup.bash
source /workspaces/omx_ros2/ws/install/setup.bash

echo
echo "STEP 2 — (Optional) Calibrate spring (if needed)"
echo "  python3 tools/calib_spring.py --spring-specimen <name>"
echo
echo "STEP 3 — Verify ArUco alignment (camera INSIDE container)"
echo "  python3 tools/aruco_alignment_monitor.py"
echo "  # or: ros2 run rqt_image_view rqt_image_view and pick /spring_monitor/aruco_alignment/annotated_image"
echo

HARNESS_CMD=(python3 tools/hardware_harness_contact_gated_with_aruco_v2.py --alignment-policy off --external-camera-trigger-file "$TRIGGER" --calibration-json-path "$CALIB_JSON")
if [[ -n "$SPECIMEN" ]]; then
  HARNESS_CMD+=(--spring-specimen "$SPECIMEN")
fi

echo "STEP 4 — Harness command (camera on HOST; run this after recorder is ready):"
printf '  %s ' "${HARNESS_CMD[@]}"
echo

if [[ "$WAIT_READY" == "true" ]]; then
  echo "Waiting for recorder READY file: $READY"
  elapsed=0
  while [[ ! -f "$READY" ]]; do
    sleep 1
    echo -n "."
    elapsed=$((elapsed+1))
    if [[ $TIMEOUT -gt 0 && $elapsed -ge $TIMEOUT ]]; then
      echo
      echo "Timed out after $TIMEOUT s waiting for $READY" >&2
      exit 2
    fi
  done
  echo
  echo "READY detected: $READY"
  if [[ "$AUTO_RUN" == "true" ]]; then
    read -p "Run harness now? [Y/n] " yn
    yn=${yn:-Y}
    if [[ "$yn" =~ ^[Yy] ]]; then
      echo "Starting harness..."
      "${HARNESS_CMD[@]}"
    else
      echo "Skipping harness run." 
    fi
  else
    echo "Ready. Paste the harness command above to start the run." 
  fi
else
  echo "If you want this script to wait for the host recorder READY file, re-run with --wait-ready [--auto-run]."
fi

echo "Done."
