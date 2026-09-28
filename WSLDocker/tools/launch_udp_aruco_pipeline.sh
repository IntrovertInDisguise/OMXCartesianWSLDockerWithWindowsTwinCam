#!/bin/bash
# launch_udp_aruco_pipeline.sh
# ──────────────────────────────────────────────────────────────────────────────
# Shared helper SOURCED by the single-arm hardware runners
# (run_single_arm_test_hardware.sh / run_single_arm_trial_hardware.sh).
#
# Launches the full IMAGE-FREE UDP ArUco pipeline so the harness gets live
# alignment (and calibration recordings) without any in-container camera frames:
#
#   windows_aruco_udp_sender.py  ──UDP──▶  udp_aruco_ros2_receiver.py
#        (Windows camera)                   ├─▶ /single_arm_aruco/detections_json
#                                           │
#                          udp_aruco_to_calibration.py
#                              ├─▶ /camera_aruco/detections_json  (relay)
#                              └─▶ <output-dir>/detections_*.jsonl (recording)
#
#                          udp_aruco_to_alignment.py
#                              └─▶ /single_arm/aruco_alignment/{valid,
#                                       center_y_error_m, recommended_z_trim_m}
#                                   (the topics the harness subscribes to)
#
# Provides (caller scope):
#   launch_udp_aruco_pipeline [--output-dir DIR] [--camera-info-topic T] [--udp-port N]
#   stop_udp_aruco_pipeline
# Sets caller-scope vars: UDP_ARUCO_PID  CALIB_BRIDGE_PID  ALIGN_BRIDGE_PID
#
# Preconditions: ROS env sourced, cwd at workspace root, Windows sender running.
# ──────────────────────────────────────────────────────────────────────────────

UDP_ARUCO_PID=""
CALIB_BRIDGE_PID=""
ALIGN_BRIDGE_PID=""

launch_udp_aruco_pipeline() {
    local output_dir="/tmp/arx_udp_run"
    local camera_info_topic="/camera/camera/color/camera_info"
    local udp_port=5005

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output-dir)        output_dir="$2"; shift 2 ;;
            --camera-info-topic) camera_info_topic="$2"; shift 2 ;;
            --udp-port)          udp_port="$2"; shift 2 ;;
            *) echo "launch_udp_aruco_pipeline: unknown arg '$1'" >&2; return 1 ;;
        esac
    done

    mkdir -p "$output_dir"
    # The harness subscribes to the alignment topics; enable its consumer side.
    export OMX_HARNESS_SINGLE_ARM_ENABLE_LIVE_ARUCO_ALIGNMENT=1

    echo "Launching UDP ArUco receiver on port ${udp_port}..."
    python3 tools/udp_aruco_ros2_receiver.py --ros-args -p udp_port:="${udp_port}" \
        > /tmp/udp_aruco_receiver.log 2>&1 &
    UDP_ARUCO_PID=$!

    echo "Launching UDP ArUco -> calibration bridge (records to ${output_dir})..."
    python3 tools/udp_aruco_to_calibration.py --output-dir "$output_dir" \
        > /tmp/udp_aruco_calibration.log 2>&1 &
    CALIB_BRIDGE_PID=$!

    echo "Launching UDP ArUco -> alignment bridge (live trim topics)..."
    python3 tools/udp_aruco_to_alignment.py \
        --camera-info-topic "$camera_info_topic" \
        > /tmp/udp_aruco_alignment.log 2>&1 &
    ALIGN_BRIDGE_PID=$!

    echo "UDP ArUco pipeline PIDs: receiver=${UDP_ARUCO_PID} calib=${CALIB_BRIDGE_PID} align=${ALIGN_BRIDGE_PID}"
    echo ""
    echo ">>> Ensure the Windows-side sender is running. Command to use on Windows:"
    bash tools/run_udp_aruco_receiver.sh --print-windows-cmd --udp-port "$udp_port"
    echo ""
}

stop_udp_aruco_pipeline() {
    for _pid in "${UDP_ARUCO_PID}" "${CALIB_BRIDGE_PID}" "${ALIGN_BRIDGE_PID}"; do
        [ -n "$_pid" ] && kill "$_pid" 2>/dev/null || true
    done
    pkill -f udp_aruco_ros2_receiver.py 2>/dev/null || true
    pkill -f udp_aruco_to_calibration.py 2>/dev/null || true
    pkill -f udp_aruco_to_alignment.py 2>/dev/null || true
}
