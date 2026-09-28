#!/usr/bin/env bash
# =============================================================================
# tools/run_buckling_trial_matrix.sh
#
# Trials pipeline for the buckling experiment.
#
# Repeatedly calls the EXISTING runner (run_single_arm_test_hardware.sh) for the
# full spring x stiffness x repeat matrix. It does NOT implement a robot runner
# of its own — it is a thin orchestration layer around the verified runner.
#
# Design notes (paper-safe):
#   - The harness is data-acquisition only.
#   - Buckling onset/depth is estimated OFFLINE from robot + ArUco trajectories.
#   - K_lat is the COMMANDED Cartesian lateral stiffness, not an independently
#     verified realized stiffness.
#   - We do NOT detect buckling online, do NOT measure contact rotation, and do
#     NOT claim the applied lateral stiffness is exact.
#   - "No buckling by final stroke" is a VALID CENSORED outcome; it never marks
#     a trial failed.
#
# Matrix (default):
#   SPRINGS = (S1 S2)
#   KLATS   = (2.5 5 10 20 40 60)
#   REPEATS = (1 2 3 4 5)
#   => 2 x 6 x 5 = 60 main trials.
#
# Ordering: blocked randomization (default). Each block contains every
# spring x K_lat condition once; within a block the order is randomized. This
# avoids running all low stiffness first or all S1 first.
#
# Usage:
#   bash tools/run_buckling_trial_matrix.sh [options]
# Options:
#   --output-root <path>      root for trial folders + trial_manifest.csv
#   --stroke-m <float>        compression stroke length (default 0.020)
#   --move-duration-s <float> stroke duration (default 80)
#   --final-hold-s <float>    hold after stroke (default 5)
#   --repeats <int>           number of repeats (default 5)
#   --dry-run                 print planned commands, do not run hardware
#   --resume                  skip trials already marked complete
#   --rerun-failed            rerun trials marked failed
#   --overwrite               overwrite existing complete trial folders
#   --randomize               randomize within blocks (same as default)
#   --block-randomize         blocked randomization (default)
#   --spring S1|S2|both       restrict springs (default both)
#   --k-lats "2.5,5,10,20,40,60"   override K_lat set
#   --active-cap-id 1|3       active spring cap marker for ArUco preflight
# =============================================================================
set -u

OUTPUT_ROOT="$(pwd)/buckling_trials"
STROKE_M=0.020
MOVE_DURATION_S=80
FINAL_HOLD_S=5
REPEATS=5
DRY_RUN=false
RESUME=false
RERUN_FAILED=false
OVERWRITE=false
RANDOMIZE_MODE="block"   # block | plain
SPRING_SEL="both"
KLATS_CSV="2.5,5,10,20,40,60"
ACTIVE_CAP_ID="1"
INCLUDE_BASELINE=false
BASELINE_REPEATS=3
BASELINE_KLATS_CSV="2.5,10,40,60"
NO_CAMERA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --stroke-m) STROKE_M="$2"; shift 2 ;;
        --move-duration-s) MOVE_DURATION_S="$2"; shift 2 ;;
        --final-hold-s) FINAL_HOLD_S="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --resume) RESUME=true; shift ;;
        --rerun-failed) RERUN_FAILED=true; shift ;;
        --overwrite) OVERWRITE=true; shift ;;
        --randomize) RANDOMIZE_MODE="plain"; shift ;;
        --block-randomize) RANDOMIZE_MODE="block"; shift ;;
        --spring) SPRING_SEL="$2"; shift 2 ;;
        --k-lats) KLATS_CSV="$2"; shift 2 ;;
        --active-cap-id) ACTIVE_CAP_ID="$2"; shift 2 ;;
        --include-baseline) INCLUDE_BASELINE=true; shift ;;
        --baseline-repeats) BASELINE_REPEATS="$2"; shift 2 ;;
        --baseline-k-lats) BASELINE_KLATS_CSV="$2"; shift 2 ;;
        --no-camera) NO_CAMERA=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Filesystem-safe K_lat label: 2.5 -> 002p5, 5 -> 005, 10 -> 010, ...
klabel() {
    local k="$1"
    # Normalize: strip trailing zeros after decimal, then map.
    case "$k" in
        2.5) echo "002p5" ;;
        5)   echo "005" ;;
        10)  echo "010" ;;
        20)  echo "020" ;;
        40)  echo "040" ;;
        60)  echo "060" ;;
        *)   # generic: replace '.' with 'p', zero-pad integer part to 3 digits
             local intpart frac
             intpart="${k%%.*}"
             frac="${k#*.}"
             [ "$frac" = "$k" ] && frac=""
             while [ ${#intpart} -lt 3 ]; do intpart="0$intpart"; done
             if [ -n "$frac" ]; then echo "${intpart}p${frac}"; else echo "$intpart"; fi
             ;;
    esac
}

# Build spring list
SPRINGS=()
case "$SPRING_SEL" in
    S1) SPRINGS=("S1") ;;
    S2) SPRINGS=("S2") ;;
    both) SPRINGS=("S1" "S2") ;;
    *) echo "Invalid --spring '$SPRING_SEL' (use S1|S2|both)"; exit 1 ;;
esac

# Build K_lat list
IFS=',' read -ra KLATS <<< "$KLATS_CSV"

# Build repeat list
REPEAT_LIST=()
for ((r=1; r<=REPEATS; r++)); do REPEAT_LIST+=("$r"); done

mkdir -p "$OUTPUT_ROOT"
MANIFEST="$OUTPUT_ROOT/trial_manifest.csv"

# Write manifest header if missing. New columns (condition_type .. aruco_status)
# carry time-sync + baseline bookkeeping for offline post-processing.
if [ ! -f "$MANIFEST" ]; then
    echo "trial_id,spring_id,k_lat_commanded_n_m,repeat_id,stroke_m,move_duration_s,final_hold_s,planned_order_index,status,start_time,end_time,output_dir,exit_code,notes,condition_type,spring_present,time_sync_method,trial_start_ros_time_s,trial_end_ros_time_s,aruco_first_frame,aruco_last_frame,aruco_frame_count,aruco_mean_rate_hz,aruco_status" > "$MANIFEST"
fi

# Static per-trial metadata (preserved across manifest updates).
declare -A STATIC_SPRING STATIC_KLAT STATIC_REP STATIC_COND STATIC_SPRESENT

# -----------------------------------------------------------------------------
# Plan all trials (blocked: each repeat = one block of all spring x K_lat)
# -----------------------------------------------------------------------------
PLAN=()   # each entry: "SPRING|KLAT|REPEAT|TRIAL_ID|COND|SPRESENT"
ORDER_IDX=0
for rep in "${REPEAT_LIST[@]}"; do
    # Build the block's condition list
    BLOCK=()
    for sp in "${SPRINGS[@]}"; do
        for k in "${KLATS[@]}"; do
            BLOCK+=("$sp|$k|$rep")
        done
    done
    # Randomize the block
    if [ "$RANDOMIZE_MODE" = "block" ] || [ "$RANDOMIZE_MODE" = "plain" ]; then
        BLOCK=($(printf '%s\n' "${BLOCK[@]}" | shuf))
    fi
    for cond in "${BLOCK[@]}"; do
        IFS='|' read -r sp k rep <<< "$cond"
        kl=$(klabel "$k")
        tid="${sp}_K${kl}_R$(printf '%02d' "$rep")"
        PLAN+=("$sp|$k|$rep|$tid|main_spring|true")
        STATIC_SPRING[$tid]="$sp"; STATIC_KLAT[$tid]="$k"; STATIC_REP[$tid]="$rep"
        STATIC_COND[$tid]="main_spring"; STATIC_SPRESENT[$tid]="true"
        ORDER_IDX=$((ORDER_IDX+1))
    done
done

# Baseline no-spring trials (spring_id=NONE, condition_type=baseline_no_spring).
# Same stroke/move/final as main trials; used offline to set the lateral
# threshold r_perp,thr = max(3mm, mu_base + 5*sigma_base) and to report the
# false-positive rate on baseline (ideally 0).
if [ "$INCLUDE_BASELINE" = true ]; then
    IFS=',' read -ra BKLATS <<< "$BASELINE_KLATS_CSV"
    for ((b=1; b<=BASELINE_REPEATS; b++)); do
        BBLOCK=()
        for k in "${BKLATS[@]}"; do
            BBLOCK+=("NONE|$k|$b")
        done
        if [ "$RANDOMIZE_MODE" = "block" ] || [ "$RANDOMIZE_MODE" = "plain" ]; then
            BBLOCK=($(printf '%s\n' "${BBLOCK[@]}" | shuf))
        fi
        for cond in "${BBLOCK[@]}"; do
            IFS='|' read -r sp k rep <<< "$cond"
            kl=$(klabel "$k")
            tid="BASELINE_K${kl}_R$(printf '%02d' "$rep")"
            PLAN+=("$sp|$k|$rep|$tid|baseline_no_spring|false")
            STATIC_SPRING[$tid]="$sp"; STATIC_KLAT[$tid]="$k"; STATIC_REP[$tid]="$rep"
            STATIC_COND[$tid]="baseline_no_spring"; STATIC_SPRESENT[$tid]="false"
            ORDER_IDX=$((ORDER_IDX+1))
        done
    done
fi

echo "========================================================================"
echo "BUCKLING TRIAL MATRIX"
echo "========================================================================"
echo "output_root   : $OUTPUT_ROOT"
echo "springs       : ${SPRINGS[*]}"
echo "k_lats        : ${KLATS[*]}"
echo "repeats       : $REPEATS"
echo "total planned : ${#PLAN[@]} trials"
echo "randomize     : $RANDOMIZE_MODE"
echo "stroke_m      : $STROKE_M"
echo "move_duration : $MOVE_DURATION_S s"
echo "final_hold    : $FINAL_HOLD_S s"
echo "active_cap_id : $ACTIVE_CAP_ID"
echo "include_base  : $INCLUDE_BASELINE (repeats=$BASELINE_REPEATS, k_lats=${BKLATS[*]:-})"
echo "dry_run       : $DRY_RUN"
echo "========================================================================"

# Helper: update a manifest row's status fields. Static columns (spring_id,
# k_lat, repeat, condition_type, spring_present) are preserved from PLAN.
update_manifest() {
    # args: trial_id status start_time end_time output_dir exit_code notes \
    #       time_sync_method trial_start_ros_time_s trial_end_ros_time_s \
    #       aruco_first_frame aruco_last_frame aruco_frame_count \
    #       aruco_mean_rate_hz aruco_status
    local tid="$1" status="$2" st="$3" et="$4" od="$5" ec="$6" notes="$7"
    local tsm="${8:-ros_receive_time_plus_trial_window}" trs="${9:-}" tes="${10:-}"
    local af="${11:-}" al="${12:-}" ac="${13:-}" ar="${14:-}" ast="${15:-}"
    local sp="${STATIC_SPRING[$tid]:-S1}" k="${STATIC_KLAT[$tid]:-10}" rep="${STATIC_REP[$tid]:-0}"
    local cond="${STATIC_COND[$tid]:-main_spring}" spres="${STATIC_SPRESENT[$tid]:-true}"
    local tmp
    tmp="$(mktemp)"
    grep -v "^${tid}," "$MANIFEST" > "$tmp" 2>/dev/null || true
    echo "${tid},${sp},${k},${rep},${STROKE_M},${MOVE_DURATION_S},${FINAL_HOLD_S},,${status},${st},${et},${od},${ec},${notes},${cond},${spres},${tsm},${trs},${tes},${af},${al},${ac},${ar},${ast}" >> "$tmp"
    mv "$tmp" "$MANIFEST"
}

# Pre-populate manifest with 'planned' rows (idempotent: only if not present)
for entry in "${PLAN[@]}"; do
    IFS='|' read -r sp k rep tid cond spres <<< "$entry"
    if ! grep -q "^${tid}," "$MANIFEST"; then
        echo "${tid},${sp},${k},${rep},${STROKE_M},${MOVE_DURATION_S},${FINAL_HOLD_S},,planned,,,,,,${cond},${spres},ros_receive_time_plus_trial_window,,,,,,,,," >> "$MANIFEST"
    fi
done

# -----------------------------------------------------------------------------
# ArUco preflight (once per run; treat UDP receiver as external dependency)
# -----------------------------------------------------------------------------
aruco_preflight_ok=true
if [ "$DRY_RUN" = false ] && command -v ros2 >/dev/null 2>&1; then
    echo ">> ArUco preflight: /single_arm_aruco/poses rate + marker_order"
    POSE_HZ=$(timeout 7 ros2 topic hz /single_arm_aruco/poses 2>/dev/null \
        | grep -oE "average rate: [0-9.]+" | grep -oE "[0-9.]+" | head -1)
    MARKER_ORDER=$(timeout 7 ros2 topic echo --once /single_arm_aruco/marker_order 2>/dev/null \
        | grep -oE "id: [0-9]+" | grep -oE "[0-9]+" | tr '\n' ' ')
    POSE_OK=$(awk "BEGIN{print ($POSE_HZ+0) >= 15}")
    GROUND_OK=$(echo " $MARKER_ORDER " | grep -q " 6 " && echo 1 || echo 0)
    CAP_OK=$(echo " $MARKER_ORDER " | grep -q " $ACTIVE_CAP_ID " && echo 1 || echo 0)
    if [ "$POSE_OK" = "1" ] && [ "$GROUND_OK" = "1" ] && [ "$CAP_OK" = "1" ]; then
        echo "   OK: ArUco preflight passed"
    else
        echo "   WARN: ArUco preflight not satisfied (poses~${POSE_HZ:-0}Hz, ground=$GROUND_OK, cap$ACTIVE_CAP_ID=$CAP_OK)."
        echo "         Trials will still attempt; per-trial failure rules will catch missing data."
        aruco_preflight_ok=false
    fi
fi

# -----------------------------------------------------------------------------
# Execute
# -----------------------------------------------------------------------------
OVERALL_EXIT=0
IDX=0
TOTAL=${#PLAN[@]}

for entry in "${PLAN[@]}"; do
    IFS='|' read -r sp k rep tid cond spres <<< "$entry"
    IDX=$((IDX+1))
    kl=$(klabel "$k")

    # Resolve output dir. Baseline trials use BASELINE_NO_SPRING as the spring
    # folder so they are clearly separated from main spring trials.
    if [ "$cond" = "baseline_no_spring" ]; then
        TRIAL_DIR="$OUTPUT_ROOT/BASELINE_NO_SPRING/Klat_$kl/repeat_$(printf '%02d' "$rep")"
    else
        TRIAL_DIR="$OUTPUT_ROOT/$sp/Klat_$kl/repeat_$(printf '%02d' "$rep")"
    fi

    # Resume / skip logic
    EXISTING_STATUS=$(grep "^${tid}," "$MANIFEST" | cut -d',' -f9)
    if [ "$RESUME" = true ]; then
        if [ "$EXISTING_STATUS" = "complete" ] && [ "$OVERWRITE" = false ]; then
            echo "[$IDX/$TOTAL] SKIP (complete, resume): $tid"
            continue
        fi
        if [ "$EXISTING_STATUS" = "failed" ] && [ "$RERUN_FAILED" = false ]; then
            echo "[$IDX/$TOTAL] SKIP (failed, no --rerun-failed): $tid"
            continue
        fi
    fi
    if [ -d "$TRIAL_DIR" ] && [ "$OVERWRITE" = false ] && [ "$EXISTING_STATUS" = "complete" ]; then
        echo "[$IDX/$TOTAL] SKIP (folder exists, complete): $tid"
        continue
    fi

    mkdir -p "$TRIAL_DIR"
    update_manifest "$tid" "running" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "" "$TRIAL_DIR" "" "started"

    echo ""
    echo "[$IDX/$TOTAL] TRIAL $tid  (spring=$sp k_lat=$k repeat=$rep cond=$cond)"
    echo "   dir: $TRIAL_DIR"

    if [ "$DRY_RUN" = true ]; then
        echo "   DRY-RUN command:"
        echo "     bash run_single_arm_test_hardware.sh \\"
        echo "       --buckling-experiment --spring-id $sp --k-lat $k \\"
        echo "       --repeat-id $rep --trial-id $tid \\"
        echo "       --stroke-m $STROKE_M --move-duration-s $MOVE_DURATION_S \\"
        echo "       --final-hold-s $FINAL_HOLD_S --active-cap-id $ACTIVE_CAP_ID \\"
        echo "       --output-root $TRIAL_DIR"
        update_manifest "$tid" "complete" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TRIAL_DIR" "0" "dry-run"
        continue
    fi

    # Run the existing runner. The runner writes runner_command.txt + stdout.log
    # + stderr.log into TRIAL_DIR, and the harness writes its timestamped data
    # folder (mapped via trial_manifest.csv notes / output_dir).
    bash run_single_arm_test_hardware.sh \
        --buckling-experiment \
        --spring-id "$sp" \
        --k-lat "$k" \
        --repeat-id "$rep" \
        --trial-id "$tid" \
        --stroke-m "$STROKE_M" \
        --move-duration-s "$MOVE_DURATION_S" \
        --final-hold-s "$FINAL_HOLD_S" \
        --active-cap-id "$ACTIVE_CAP_ID" \
        --output-root "$TRIAL_DIR" \
        ${NO_CAMERA:+--no-camera} \
        > "$TRIAL_DIR/stdout.log" 2> "$TRIAL_DIR/stderr.log"
    RUN_EXIT=$?

    END_T="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Failure rules (do NOT fail because buckling was not detected).
    FAIL_REASON=""
    if [ $RUN_EXIT -ne 0 ]; then
        FAIL_REASON="runner exit nonzero ($RUN_EXIT)"
    elif [ ! -d "$TRIAL_DIR" ]; then
        FAIL_REASON="output folder missing"
    else
        # Locate harness CSV (timestamped dir under /mnt/omx_logs)
        HARNESS_CSV=$(find /mnt/omx_logs -maxdepth 1 -type d -name 'single_arm_harness_v3_*' -newer "$TRIAL_DIR/stdout.log" 2>/dev/null | head -1)/single_arm_harness_v3_snapshot.csv
        if [ -z "$HARNESS_CSV" ] || [ ! -f "$HARNESS_CSV" ]; then
            FAIL_REASON="no robot CSV produced"
        else
            HEADER=$(head -1 "$HARNESS_CSV")
            for col in compression_depth_measured_m lateral_y_m r_perp_m k_y_n_m k_lat_commanded_n_m; do
                echo "$HEADER" | grep -qE "(^|,)$col(,)" || FAIL_REASON="CSV missing column: $col"
            done
            # k_y_n_m must match requested K_lat
            KY=$(python3 - "$HARNESS_CSV" "$k" <<'PY'
import sys,csv
p=sys.argv[1]; klat=float(sys.argv[2])
with open(p) as f:
    r=csv.DictReader(f)
    row=r.__next__(); ky=float(row.get('k_y_n_m', 'nan'))
    print("%.4f" % ky)
PY
            )
            KY_MATCH=$(awk "BEGIN{print (($KY+0) - $k) < 0.5}")
            if [ "$KY_MATCH" != "1" ]; then
                FAIL_REASON="k_y_n_m ($KY) != requested K_lat ($k)"
            fi
            # compression depth must not remain near zero (excluding header)
            MAX_DEPTH=$(python3 - "$HARNESS_CSV" <<'PY'
import sys,csv
p=sys.argv[1]
mx=0.0
with open(p) as f:
    for row in csv.DictReader(f):
        try:
            d=float(row.get('compression_depth_measured_m','nan'))
        except: continue
        if d>mx: mx=d
print("%.6f"%mx)
PY
            )
            DEPTH_OK=$(awk "BEGIN{print ($MAX_DEPTH+0) > 0.0005}")
            if [ "$DEPTH_OK" != "1" ]; then
                FAIL_REASON="compression depth near zero (max=$MAX_DEPTH)"
            fi
        fi
    fi

    # ---- Time-sync / ArUco bookkeeping (offline alignment inputs) ----
    # Read trial start/end ROS time from the harness metadata.yaml written by
    # the runner into TRIAL_DIR. Then count ArUco JSONL frames whose ROS
    # receive time falls inside [trial_start, trial_end].
    T_START=""; T_END=""
    if [ -f "$TRIAL_DIR/metadata.yaml" ]; then
        T_START=$(grep -E "^trial_start_ros_time_s:" "$TRIAL_DIR/metadata.yaml" | grep -oE "[0-9.eE+-]+" | head -1)
        T_END=$(grep -E "^trial_end_ros_time_s:" "$TRIAL_DIR/metadata.yaml" | grep -oE "[0-9.eE+-]+" | head -1)
    fi
    A_FIRST=""; A_LAST=""; A_COUNT="0"; A_RATE=""; A_STATUS="no_aruco"
    if [ -n "$T_START" ] && [ -n "$T_END" ]; then
        # Find the ArUco JSONL produced during this trial (most recent run dir
        # newer than the trial start). Count frames in the trial window.
        ARUCO_RUN=$(ls -dt /tmp/arx_udp_run_* 2>/dev/null | head -1)
        if [ -n "$ARUCO_RUN" ]; then
            A_STATS=$(python3 - "$ARUCO_RUN" "$T_START" "$T_END" <<'PY'
import sys, glob, json, os
run_dir=sys.argv[1]; t0=float(sys.argv[2]); t1=float(sys.argv[3])
frames=[]; first_f=None; last_f=None
for jf in glob.glob(os.path.join(run_dir, "detections_*.jsonl")):
    with open(jf) as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            try: rec=json.loads(line)
            except: continue
            # Prefer ROS receive time (same clock as the robot) for the trial
            # window; fall back to the Windows sender timestamp (ts_ns) if the
            # bridge did not record ros_recv_time_s.
            t_s = rec.get("ros_recv_time_s")
            if t_s is None:
                ts_ns=rec.get("ts_ns")
                if ts_ns is None: continue
                t_s = ts_ns/1e9
            if t0 <= t_s <= t1:
                frames.append(t_s)
                fno=rec.get("frame")
                if first_f is None: first_f=fno
                last_f=fno
            elif t_s > t1:
                break
frames.sort()
n=len(frames)
if n>1:
    rate=(n-1)/(frames[-1]-frames[0])
else:
    rate=0.0
print("%s,%s,%d,%.3f" % (first_f if first_f is not None else "", last_f if last_f is not None else "", n, rate))
PY
            )
            A_FIRST=$(echo "$A_STATS" | cut -d',' -f1)
            A_LAST=$(echo "$A_STATS" | cut -d',' -f2)
            A_COUNT=$(echo "$A_STATS" | cut -d',' -f3)
            A_RATE=$(echo "$A_STATS" | cut -d',' -f4)
        fi
        # Acceptance: >= 0.6 * 25 * trial_duration_s frames expected.
        TRIAL_DUR=$(awk "BEGIN{print ($T_END - $T_START)}")
        MIN_FRAMES=$(awk "BEGIN{printf \"%d\", 0.6*25*$TRIAL_DUR}")
        if [ -n "$A_COUNT" ] && [ "$A_COUNT" -ge "$MIN_FRAMES" ] 2>/dev/null; then
            A_STATUS="ok"
        elif [ "$A_COUNT" = "0" ]; then
            A_STATUS="failed_aruco"
        else
            A_STATUS="partial_aruco"
        fi
    fi

    if [ -n "$FAIL_REASON" ]; then
        echo "   FAIL: $FAIL_REASON"
        update_manifest "$tid" "failed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$END_T" "$TRIAL_DIR" "$RUN_EXIT" "$FAIL_REASON" \
            "ros_receive_time_plus_trial_window" "$T_START" "$T_END" "$A_FIRST" "$A_LAST" "$A_COUNT" "$A_RATE" "$A_STATUS"
        OVERALL_EXIT=1
    else
        echo "   COMPLETE (aruco_status=$A_STATUS, frames=$A_COUNT, rate=${A_RATE}Hz)"
        # Map harness data folder into the trial folder for convenience.
        HARNESS_DIR=$(find /mnt/omx_logs -maxdepth 1 -type d -name 'single_arm_harness_v3_*' -newer "$TRIAL_DIR/stdout.log" 2>/dev/null | head -1)
        NOTES="harness_data=$HARNESS_DIR"
        # If ArUco dropped badly, mark complete_robot_only rather than failed
        # (robot data is valid; only the vision sync is degraded).
        if [ "$A_STATUS" = "failed_aruco" ]; then
            STATUS="complete_robot_only"
        else
            STATUS="complete"
        fi
        update_manifest "$tid" "$STATUS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$END_T" "$TRIAL_DIR" "$RUN_EXIT" "$NOTES" \
            "ros_receive_time_plus_trial_window" "$T_START" "$T_END" "$A_FIRST" "$A_LAST" "$A_COUNT" "$A_RATE" "$A_STATUS"
    fi
done

echo ""
echo "========================================================================"
echo "TRIAL MATRIX COMPLETE"
echo "Manifest: $MANIFEST"
echo "========================================================================"
exit $OVERALL_EXIT
