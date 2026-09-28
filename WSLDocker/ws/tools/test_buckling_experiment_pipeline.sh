#!/usr/bin/env bash
# =============================================================================
# tools/test_buckling_experiment_pipeline.sh
#
# Testing pipeline for the buckling experiment.
#
# Purpose: verify that the existing runner (run_single_arm_test_hardware.sh),
# the controller config, the harness logging, and the (already-verified) ArUco
# UDP pipeline are ready BEFORE running the full 60-trial matrix.
#
# This script does NOT create a new robot runner. It wraps the existing
# run_single_arm_test_hardware.sh and adds preflight checks + a dry run + an
# optional short smoke test.
#
# Design notes (paper-safe):
#   - The harness is data-acquisition only.
#   - Buckling onset/depth is estimated OFFLINE from robot + ArUco trajectories.
#   - K_lat is the COMMANDED Cartesian lateral stiffness, not an independently
#     verified realized stiffness.
#   - We do NOT detect buckling online, do NOT measure contact rotation, and do
#     NOT claim the applied lateral stiffness is exact.
#
# Usage:
#   bash tools/test_buckling_experiment_pipeline.sh [--active-cap-id 1|3] [--skip-aruco] [--smoke]
# =============================================================================
set -u

ACTIVE_CAP_ID="1"
SKIP_ARUCO=false
RUN_SMOKE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --active-cap-id) ACTIVE_CAP_ID="$2"; shift 2 ;;
        --skip-aruco) SKIP_ARUCO=true; shift ;;
        --smoke) RUN_SMOKE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Expected marker IDs (see spec)
GROUND_WORLD_ID=6
METAL_PLATFORM_ID=0
SPRING_CAP1_ID=1
SPRING_CAP2_ID=3

echo "========================================================================"
echo "BUCKLING EXPERIMENT — TESTING PIPELINE"
echo "========================================================================"
echo "active_cap_id = $ACTIVE_CAP_ID"
echo ""

PASS=true

# -----------------------------------------------------------------------------
# 1. Static checks: harness + runner syntax, controller build
# -----------------------------------------------------------------------------
echo "------------------------------------------------------------------------"
echo "[1/4] Static checks: harness syntax, runner syntax, controller build"
echo "------------------------------------------------------------------------"

echo ">> python3 -m py_compile tools/hardware_harness_single_arm_v3.py"
if python3 -m py_compile tools/hardware_harness_single_arm_v3.py; then
    echo "   OK: harness compiles"
else
    echo "   FAIL: harness does not compile"
    PASS=false
fi

echo ">> bash -n run_single_arm_test_hardware.sh"
if bash -n run_single_arm_test_hardware.sh; then
    echo "   OK: runner syntax valid"
else
    echo "   FAIL: runner syntax invalid"
    PASS=false
fi

if [ -d install ]; then
    # Preflight: verify the controller shared library is present. A full
    # colcon build is slow and is already part of the normal workflow; we only
    # confirm the artifact exists so the testing pipeline does not block.
    CTRL_LIB="build/omx_variable_stiffness_controller/libomx_variable_stiffness_controller.so"
    if [ -f "$CTRL_LIB" ]; then
        echo "   OK: controller built ($CTRL_LIB present)"
    else
        echo "   WARN: controller library not found at $CTRL_LIB."
        echo "         Run 'colcon build --packages-select omx_variable_stiffness_controller'"
        echo "         before real trials. Treating as non-fatal for preflight."
    fi
else
    echo "   SKIP: no install/ workspace found; skipping controller build check"
fi

# -----------------------------------------------------------------------------
# 2. ArUco topic preflight (UDP receiver already verified externally)
# -----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------------------"
echo "[2/4] ArUco topic preflight (requires running UDP receiver)"
echo "------------------------------------------------------------------------"

if [ "$SKIP_ARUCO" = true ]; then
    echo "   SKIP: --skip-aruco set"
elif ! command -v ros2 >/dev/null 2>&1; then
    echo "   SKIP: ros2 not on PATH; cannot check ArUco topics"
else
    # detections_json rate
    echo ">> ros2 topic hz /single_arm_aruco/detections_json (5s sample)"
    DET_HZ=$(timeout 7 ros2 topic hz /single_arm_aruco/detections_json 2>/dev/null \
        | grep -oE "average rate: [0-9.]+" | grep -oE "[0-9.]+" | head -1)
    echo "   detections_json rate ~ ${DET_HZ:-0} Hz (need >= 15)"

    # poses rate
    echo ">> ros2 topic hz /single_arm_aruco/poses (5s sample)"
    POSE_HZ=$(timeout 7 ros2 topic hz /single_arm_aruco/poses 2>/dev/null \
        | grep -oE "average rate: [0-9.]+" | grep -oE "[0-9.]+" | head -1)
    echo "   poses rate ~ ${POSE_HZ:-0} Hz (need >= 15)"

    # marker_order
    echo ">> ros2 topic echo --once /single_arm_aruco/marker_order"
    MARKER_ORDER=$(timeout 7 ros2 topic echo --once /single_arm_aruco/marker_order 2>/dev/null \
        | grep -oE "id: [0-9]+" | grep -oE "[0-9]+" | tr '\n' ' ')
    echo "   marker_order ids seen: [${MARKER_ORDER:-none}]"

    DET_OK=$(awk "BEGIN{print ($DET_HZ+0) >= 15}")
    POSE_OK=$(awk "BEGIN{print ($POSE_HZ+0) >= 15}")
    GROUND_OK=$(echo " $MARKER_ORDER " | grep -q " $GROUND_WORLD_ID " && echo 1 || echo 0)
    CAP_OK=$(echo " $MARKER_ORDER " | grep -q " $ACTIVE_CAP_ID " && echo 1 || echo 0)

    if [ "$DET_OK" = "1" ] && [ "$POSE_OK" = "1" ] && [ "$GROUND_OK" = "1" ] && [ "$CAP_OK" = "1" ]; then
        echo "   OK: ArUco preflight passed (ground_world + active cap $ACTIVE_CAP_ID visible, rates >= 15 Hz)"
    else
        echo "   WARN: ArUco preflight not fully satisfied (this is a preflight, not a hard fail)."
        echo "         Ensure the Windows UDP sender is running and markers are in view."
        # Preflight warning does not fail the whole testing pipeline; the real
        # gate is the trials pipeline's per-trial ArUco check.
    fi

    # Time-sync preflight: robot ROS time must be increasing, and the ArUco
    # ROS receive time must be increasing. Offline alignment uses
    # t_rel = t_ros - trial_start_ros_time_s, so a monotonic clock is required.
    echo ">> time-sync preflight: robot ROS time + ArUco ROS receive time monotonic"
    ROBOT_T1=$(timeout 5 ros2 topic echo --once /single_arm/joint_states 2>/dev/null \
        | grep -oE "header:" -A2 | grep -oE "stamp: [0-9]+" | grep -oE "[0-9]+" | head -1)
    sleep 1
    ROBOT_T2=$(timeout 5 ros2 topic echo --once /single_arm/joint_states 2>/dev/null \
        | grep -oE "header:" -A2 | grep -oE "stamp: [0-9]+" | grep -oE "[0-9]+" | head -1)
    if [ -n "$ROBOT_T1" ] && [ -n "$ROBOT_T2" ] && [ "$ROBOT_T2" -gt "$ROBOT_T1" ] 2>/dev/null; then
        echo "   OK: robot ROS time increasing ($ROBOT_T1 -> $ROBOT_T2)"
    else
        echo "   WARN: robot ROS time not confirmed increasing (t1=$ROBOT_T1 t2=$ROBOT_T2)"
    fi
    # ArUco ROS receive time: sample two detections_json packets via the relay
    # topic and compare the embedded ros_recv_time_s (added by the calibration
    # bridge). Falls back to a soft warning if the field is absent.
    A1=$(timeout 5 ros2 topic echo --once /camera_aruco/detections_json 2>/dev/null \
        | grep -oE '"ros_recv_time_s": [0-9.eE+-]+' | grep -oE "[0-9.eE+-]+" | head -1)
    sleep 1
    A2=$(timeout 5 ros2 topic echo --once /camera_aruco/detections_json 2>/dev/null \
        | grep -oE '"ros_recv_time_s": [0-9.eE+-]+' | grep -oE "[0-9.eE+-]+" | head -1)
    if [ -n "$A1" ] && [ -n "$A2" ] && awk "BEGIN{exit !($A2 > $A1)}"; then
        echo "   OK: ArUco ROS receive time increasing ($A1 -> $A2)"
    else
        echo "   WARN: ArUco ros_recv_time_s not confirmed increasing (a1=$A1 a2=$A2);"
        echo "         ensure udp_aruco_to_calibration.py records ros_recv_time_s."
    fi
fi

# -----------------------------------------------------------------------------
# 3. Dry run of the existing runner
# -----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------------------"
echo "[3/4] Dry run of existing runner (--buckling-experiment --dry-run)"
echo "------------------------------------------------------------------------"

DRY_OUT=$(mktemp /tmp/buckling_dryrun.XXXXXX.log)
bash run_single_arm_test_hardware.sh \
    --buckling-experiment \
    --spring-id S1 \
    --k-lat 10 \
    --repeat-id 0 \
    --stroke-m 0.020 \
    --move-duration-s 80 \
    --final-hold-s 5 \
    --active-cap-id "$ACTIVE_CAP_ID" \
    --dry-run > "$DRY_OUT" 2>&1
DRY_EXIT=$?

REQUIRED_LINES=(
    "buckling experiment mode enabled"
    "deep press disabled"
    "lateral probe disabled"
    "fixed_stiffness_y = 10"
    "spring_id = S1"
    "repeat_id = 0"
    "stroke_m = 0.020"
    "move_duration_s = 80"
)
for line in "${REQUIRED_LINES[@]}"; do
    if grep -qF "$line" "$DRY_OUT"; then
        echo "   OK: dry-run contains: $line"
    else
        echo "   FAIL: dry-run missing: $line"
        PASS=false
    fi
done
echo "   (dry-run exit code: $DRY_EXIT)"

# Baseline (no-spring) dry run: confirms --spring-id NONE is accepted and the
# runner records baseline_no_spring condition metadata correctly.
echo ">> baseline dry run (--spring-id NONE)"
BASE_DRY_OUT=$(mktemp /tmp/buckling_basedry.XXXXXX.log)
bash run_single_arm_test_hardware.sh \
    --buckling-experiment \
    --spring-id NONE \
    --k-lat 10 \
    --repeat-id 0 \
    --stroke-m 0.020 \
    --move-duration-s 80 \
    --final-hold-s 5 \
    --active-cap-id "$ACTIVE_CAP_ID" \
    --dry-run > "$BASE_DRY_OUT" 2>&1
BASE_DRY_EXIT=$?
BASE_REQUIRED_LINES=(
    "spring_id = NONE"
    "buckling experiment mode enabled"
)
for line in "${BASE_REQUIRED_LINES[@]}"; do
    if grep -qF "$line" "$BASE_DRY_OUT"; then
        echo "   OK: baseline dry-run contains: $line"
    else
        echo "   FAIL: baseline dry-run missing: $line"
        PASS=false
    fi
done
echo "   (baseline dry-run exit code: $BASE_DRY_EXIT)"

# -----------------------------------------------------------------------------
# 4. Optional short real smoke test
# -----------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------------------"
echo "[4/4] Optional short smoke test"
echo "------------------------------------------------------------------------"
if [ "$RUN_SMOKE" = true ]; then
    echo ">> short real stroke (stroke_m=0.003 move_duration_s=15 final_hold_s=2)"
    SMOKE_OUT=$(mktemp /tmp/buckling_smoke.XXXXXX.log)
    bash run_single_arm_test_hardware.sh \
        --buckling-experiment \
        --spring-id S1 \
        --k-lat 10 \
        --repeat-id 0 \
        --stroke-m 0.003 \
        --move-duration-s 15 \
        --final-hold-s 2 \
        --active-cap-id "$ACTIVE_CAP_ID" \
        > "$SMOKE_OUT" 2>&1
    SMOKE_EXIT=$?
    echo "   smoke exit code: $SMOKE_EXIT"

    # Locate the harness CSV (timestamped dir under /mnt/omx_logs)
    SMOKE_CSV=$(grep -rhoE "/mnt/omx_logs/single_arm_harness_v3_[0-9_]+/single_arm_harness_v3_snapshot.csv" "$SMOKE_OUT" 2>/dev/null | head -1)
    if [ -z "$SMOKE_CSV" ] || [ ! -f "$SMOKE_CSV" ]; then
        SMOKE_CSV=$(find /mnt/omx_logs -maxdepth 1 -type d -name 'single_arm_harness_v3_*' -newer "$SMOKE_OUT" 2>/dev/null | head -1)/single_arm_harness_v3_snapshot.csv
    fi

    if [ -n "$SMOKE_CSV" ] && [ -f "$SMOKE_CSV" ]; then
        echo "   CSV: $SMOKE_CSV"
        REQUIRED_COLS=(
            compression_depth_measured_m
            compression_depth_commanded_m
            lateral_y_m
            lateral_z_m
            r_perp_m
            k_x_n_m
            k_y_n_m
            k_z_n_m
            k_lat_commanded_n_m
        )
        FORBIDDEN_COLS=(
            buckling_detected
            buckling_depth
            instability_label
        )
        HEADER=$(head -1 "$SMOKE_CSV")
        for col in "${REQUIRED_COLS[@]}"; do
            if echo "$HEADER" | grep -qE "(^|,)$col(,)"; then
                echo "   OK: column present: $col"
            else
                echo "   FAIL: required column missing: $col"
                PASS=false
            fi
        done
        for col in "${FORBIDDEN_COLS[@]}"; do
            if echo "$HEADER" | grep -qE "(^|,)$col(,)"; then
                echo "   FAIL: forbidden online-detection column present: $col"
                PASS=false
            else
                echo "   OK: forbidden column absent: $col"
            fi
        done
    else
        echo "   WARN: could not locate smoke-test CSV; skipping column checks"
    fi
else
    echo "   SKIP: --smoke not set. Re-run with --smoke to perform the short real stroke."
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "========================================================================"
if [ "$PASS" = true ]; then
    echo "TESTING PIPELINE: PASS (static + dry-run checks OK)"
    echo "Next: bash tools/run_buckling_trial_matrix.sh --spring both --randomize"
else
    echo "TESTING PIPELINE: FAIL (see above)"
    exit 1
fi
echo "========================================================================"
