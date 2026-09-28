#!/usr/bin/env python3
"""
hardware_harness_ablation.py
────────────────────────────
K_lat ablation harness for the dual Open Manipulator-X spring-compression
experiment.  Runs N repetitions per K_lat setpoint (stochastic coverage),
saves per-run CSVs and a consolidated summary compatible with
klat_ablation_results/summary.csv so compare_sim_hw.py can be used directly.

Physical setup
──────────────
    • Two OMX robots facing each other along the x-axis with the same 0.78 m
        base-to-base spacing used on hardware; spring centre ~50 mm above base height.
    Depth camera mounted at approx height 0.61 m and approx horizontal dist 0.46 m from
    spring centre, at angle to have whole setup in the camera frame.
    Spring params need to be calibrated/adjusted in advance (see calib_spring.py) to determine the K_lat threshold values for the ablation cases.
    Check calib_spring.py for details on the calibration procedure and how to interpret the results.
    Log the calibration results (CalibResult) in the same log directory as the ablation runs for traceability and treat each spring as a separate "hardware configuration" in the analysis.
    Log depth camera geometry (spring_length_m, contact_rotation_{L,R}_rad) during the ablation runs to check for consistency with the calibration and to provide context for interpreting the results, along with timestamped frame snapshots in a separate subdirectory if possible.
  • Robot 1 presses in –x direction, robot 2 in +x direction (world frame).
  • Depth camera mounted overhead at an angle / side – can optionally provide lateral
    deflection via topic /spring_monitor/lateral_deflection_m (Float64,
    optional; harness falls back to force-only onset detection if absent).
  • K_lat is set via the /robot*/robot*_variable_stiffness/set_k_lateral
    service or, where the controller exposes it, via a parameter override.
    Adjust SET_KLAT_METHOD below if your controller uses a different API.

Usage
─────
  python3 tools/hardware_harness_ablation.py                     # full sweep, 20 reps
  python3 tools/hardware_harness_ablation.py --reps 5            # quick smoke test
    python3 tools/hardware_harness_ablation.py --cases K_ref K_low
    python3 tools/hardware_harness_ablation.py --k-lat 10 25 50 70 # explicit values <= configured limit
    python3 tools/hardware_harness_ablation.py --gazebo-smoke      # single fixed-stiffness Gazebo shakeout
    python3 tools/hardware_harness_ablation.py --abort-disable-torque \
            --robot1-port /dev/serial/by-id/<ID1> --robot2-port /dev/serial/by-id/<ID2>
  python3 tools/hardware_harness_ablation.py --dry-run           # check liveness only

Output (written to OMX_LOG_DIR or /tmp/variable_stiffness_logs):
  ablation_<timestamp>/
        depth_camera_calibration.json              # fallback/active camera extrinsics metadata
    depth_camera_frames/                       # optional color/depth frame snapshots if camera topics are available
        depth_camera_frames/<run>_raw_stream/     # optional decimated raw color/depth frames captured during active runs
        depth_camera_frames/<run>_camera_aruco/   # optional per-frame camera_aruco summaries + offline dropout/flip exports
    run_<case>_K<klat>_rep<rep>_<ts>.csv     # full 50 Hz log for every run
    run_<case>_K<klat>_rep<rep>_<ts>.json    # per-run scalar summary
    summary.csv                              # consolidated (matches sim summary.csv)
    ablation_log.txt                         # human-readable run log

MANDATORY BRINGUP (in a separate terminal before starting this script):
  source /opt/ros/humble/setup.bash && source /workspaces/omx_ros2/ws/install/setup.bash
  ros2 launch omx_variable_stiffness_controller dual_hardware_variable_stiffness.launch.py \\
    robot1_port:=/dev/serial/by-id/<ID1> robot2_port:=/dev/serial/by-id/<ID2> \\
    enable_logger:=true enable_live_plot:=false start_rviz:=false

Wait until BOTH robots log  state : MOVE_RETURN  before starting this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Pose, PoseStamped, WrenchStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

try:
    from tools.depth_frame_utils import (
        camera_info_to_dict,
        export_saved_sensor_stream_video,
        image_stamp_ns,
        save_sensor_image,
        suggested_image_extension,
    )
except ImportError:
    from depth_frame_utils import (
        camera_info_to_dict,
        export_saved_sensor_stream_video,
        image_stamp_ns,
        save_sensor_image,
        suggested_image_extension,
    )

try:
    from tools.aruco_alignment_utils import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROLL_DEG_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
        evaluate_fallback_alignment,
    )
except ImportError:
    from aruco_alignment_utils import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROLL_DEG_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
        evaluate_fallback_alignment,
    )

try:
    from tools.summarize_camera_aruco_raw_signals import (
        harness_camera_aruco_log_columns,
        harness_camera_aruco_log_row,
        summarize_records,
        write_summary_csv_exports,
    )
except ImportError:
    from summarize_camera_aruco_raw_signals import (
        harness_camera_aruco_log_columns,
        harness_camera_aruco_log_row,
        summarize_records,
        write_summary_csv_exports,
    )

try:
    from tools.ablation_config import (
        CARTESIAN_STIFFNESS_LIMIT_NPM,
        DEFAULT_ALIGNMENT_FALLBACK_PSI_DIFF_TOL_DEG,
        DEFAULT_ALIGNMENT_FALLBACK_W_TOL_M,
        DEFAULT_ALIGNMENT_FALLBACK_YAW_TOL_DEG,
        DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
        DEFAULT_ALIGNMENT_POLICY,
        DEFAULT_ALIGNMENT_SOURCE,
        DEFAULT_DEPTH_CAMERA_HEIGHT_M,
        DEFAULT_DEPTH_CAMERA_HORIZONTAL_OFFSET_M,
        DEFAULT_DEPTH_CAMERA_SOURCE,
        HW_K_LAT_CASES,
        SPRING_CENTER_HEIGHT_M,
        SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD,
        SPRING_NOMINAL_LENGTH_M,
        SPRING_THEORETICAL_BUCKLING_LOAD_N,
        SPRING_THEORETICAL_CRITICAL_K_LAT_NPM,
    )
except ImportError:
    from ablation_config import (
        CARTESIAN_STIFFNESS_LIMIT_NPM,
        DEFAULT_ALIGNMENT_FALLBACK_PSI_DIFF_TOL_DEG,
        DEFAULT_ALIGNMENT_FALLBACK_W_TOL_M,
        DEFAULT_ALIGNMENT_FALLBACK_YAW_TOL_DEG,
        DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
        DEFAULT_ALIGNMENT_POLICY,
        DEFAULT_ALIGNMENT_SOURCE,
        DEFAULT_DEPTH_CAMERA_HEIGHT_M,
        DEFAULT_DEPTH_CAMERA_HORIZONTAL_OFFSET_M,
        DEFAULT_DEPTH_CAMERA_SOURCE,
        HW_K_LAT_CASES,
        SPRING_CENTER_HEIGHT_M,
        SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD,
        SPRING_NOMINAL_LENGTH_M,
        SPRING_THEORETICAL_BUCKLING_LOAD_N,
        SPRING_THEORETICAL_CRITICAL_K_LAT_NPM,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1) EXPERIMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Shared spring and case-model parameters live in ablation_config.py.
# Default K_lat cases are filtered there against the configured controller cap.

# NOTE — very low K_lat detection floor on hardware:
#   P_c ≈ sqrt(K_lat × k_θ), where k_θ comes from the shared spring model.
#   K=1  → P_c ≈ 0.13 N  (below CONTACT_FORCE_ENTER; physical w/yaw onset
#   K=10 → P_c ≈ 0.42 N   unreliable — rely on /spring_monitor margin topics)
#   K=25 → P_c ≈ 0.67 N  (first value where physical onset may be detectable)
#   K=50 → P_c ≈ 0.95 N  (above contact threshold; physical detection expected)
# For K_lat ≤ 25 N/m t_mc_zero is valid; t_phi_4deg / t_w_3mm may stay NaN.

# Default repetitions per K_lat setpoint.
DEFAULT_REPS = 20
DEFAULT_COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_COLOR_CAMERA_INFO_TOPIC = "/camera/color/camera_info"
DEFAULT_DEPTH_IMAGE_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEFAULT_DEPTH_CAMERA_INFO_TOPIC = "/camera/aligned_depth_to_color/camera_info"
DEFAULT_CAMERA_ARUCO_SUMMARY_TOPIC = "/camera_aruco/detections_json"
DEPTH_CAMERA_FRAME_DIRNAME = "depth_camera_frames"
DEFAULT_RAW_CAMERA_STREAM_RATE_HZ = 0.2
DEFAULT_RAW_CAMERA_VIDEO_FPS = 0.0
DEFAULT_CAPTURE_BUFFER_PRE_S = 5.0
DEFAULT_CAPTURE_BUFFER_POST_S = 5.0

# Inter-repetition pause [s] — lets the spring return to rest and joints settle.
REST_BETWEEN_REPS_S = 4.0

# Inter-case pause [s] — extra settling when K_lat changes.
REST_BETWEEN_CASES_S = 8.0

# Physical timing (must match quasi-static ramp; mirror MuJoCo APPROACH/RAMP).
# These bound how long we wait for onset events.
# Sim: DURATION_S=45 s, ramp rate 0.15 N/s, F*=4.5 N → ramp ≈ 30 s + 15 s hold.
FORWARD_PHASE_TIMEOUT_S = 8.0   # max time to observe each controller enter MOVE_FORWARD
PRESS_RAMP_TIMEOUT_S = 40.0  # max ramp duration before declaring no-onset (28 s ramp + margin)
QUASISTATIC_SAMPLE_DELAY = 0.20  # s — shorter updates keep loading phases ramp-like, not step-and-hold

# Spring / geometry constants (shared with calib_spring.py).
SPRING_Z_OFFSET_M  = SPRING_CENTER_HEIGHT_M
P_B_THEORY_N       = SPRING_THEORETICAL_BUCKLING_LOAD_N

# Physical onset thresholds (must match sim).
W_MAX_THRESH_M     = 0.003  # lateral deflection onset: 3 mm
PHI_MAX_THRESH_DEG = 4.0    # yaw onset: 4 degrees

# ── spring calibration ────────────────────────────────────────────────────────
# All quantities are effective experimental values, not material constants.
# Camera gives geometry (L, DeltaL, w_max, psi); manipulators give force (P_s).
CALIB_K_LAT_NPM      = CARTESIAN_STIFFNESS_LIMIT_NPM
CALIB_FORCE_TARGET_N = 2.5     # N    — max force for k_a fit (< P_b ~= 3.49 N)
CALIB_PB_FORCE_MAX_N = 5.5     # N    — compression stop for P_b scan (> 1.5 x P_b)
CALIB_STEP_M         = 0.0001  # m    — incremental compression step during admissible quasi-static ramps
CALIB_ECC_MM         = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]  # eccentric offsets [mm]
CALIB_LOAD_FRAC      = 0.50    # —     fraction of P_b_hat for k_theta eccentric test
CALIB_W_EXCLUDE_M    = 0.002   # m    — exclude w_max > 2 mm from k_a linear fit
CALIB_L0_SAMPLES     = 40      # —     samples for L0 measurement at zero compression

# How K_lat is communicated to the controller.
# Options: "ros2_param"  → rclpy parameter set on the controller node
#          "topic"       → publish Float64 to /robot*/...set_k_lateral
#          "manual"      → script pauses and prompts operator to set manually
SET_KLAT_METHOD = "topic"  # manual remains available as an operator fallback

# Controller node names for ros2_param method (ignored for other methods).
CTRL_NODE_R1 = "/robot1/robot1_variable_stiffness"
CTRL_NODE_R2 = "/robot2/robot2_variable_stiffness"

# K_lat topic names for topic method (ignored for other methods).
KLAT_TOPIC_R1 = "/robot1/robot1_variable_stiffness/set_k_lateral"
KLAT_TOPIC_R2 = "/robot2/robot2_variable_stiffness/set_k_lateral"

# ═══════════════════════════════════════════════════════════════════════════════
# 2) MOTION / FORCE TUNING  (mirror harness_v2 proven values)
# ═══════════════════════════════════════════════════════════════════════════════

IDLE_VEL_LIMIT      = 1.0    # rad/s — joint velocity limit for idle check
MOVE_VEL_LIMIT      = 1.5    # rad/s — abort threshold during motion
HOLD_VEL_LIMIT      = 0.8    # rad/s — abort threshold during hold
IDLE_EE_SPEED_LIMIT_MPS = 0.020  # m/s — EE-speed fallback when joint states are unavailable
MOVE_EE_SPEED_LIMIT_MPS = 0.050  # m/s — EE-speed fallback during motion checks
HOLD_EE_SPEED_LIMIT_MPS = 0.015  # m/s — EE-speed fallback during hold checks
IDLE_WAIT_TIMEOUT_S = 45.0   # s — allow a full asynchronous cycle to return to the start window
IDLE_STABLE_SAMPLE_S = 0.30  # s — desired-pose sample window for idle gating
IDLE_DESIRED_DRIFT_TOL_M = 5e-4  # m — max desired x drift while considered idle
IDLE_START_WINDOW_TRAVEL_M = 0.020  # m — minimum desired-x excursion before accepting a start window
IDLE_START_X_TOL_M = 0.002  # m — how close desired x must be to the observed cycle start extreme
COMMAND_REPEATS     = 2      # publish repeats per waypoint command
COMMAND_DT          = 0.05   # s between publish repeats
COMMAND_SAMPLE_DELAY = 0.45  # s of spin_for after applying a waypoint

# Hardware v3 compression logs showed that 1.5 mm commands at ~0.46 s cadence
# already pushed joint velocity close to MOVE_VEL_LIMIT. Use finer 0.1 mm
# updates at 0.20 s cadence so pre-contact, contact, and calibration loading
# stay near 0.5 mm/s without long settle pauses between steps.
QUASISTATIC_TCP_RATE_MPS = CALIB_STEP_M / QUASISTATIC_SAMPLE_DELAY
CALIB_SAMPLE_DELAY = QUASISTATIC_SAMPLE_DELAY

# Precontact approach (positive press depth follows the controller's forward x direction).
PRECONTACT_START_X  = 0.0    # offset from nominal [m]
PRECONTACT_END_X    = 0.0200 # max forward travel before declaring no-contact
PRESS_STEP          = CALIB_STEP_M # precontact step per iteration [m]
PRESS_END_X         = 0.0240 # total press depth at end of ramp [m]
RAMP_STEP           = CALIB_STEP_M # stage-5 quasi-static compression step [m]
RAMP_SAMPLE_DELAY   = QUASISTATIC_SAMPLE_DELAY
MAX_PRECONTACT_ITER = max(1, int(math.ceil((PRECONTACT_END_X - PRECONTACT_START_X) / PRESS_STEP)))

# Force thresholds (from harness_v2 fast retune, hardware-verified).
CONTACT_FORCE_ENTER  = 0.60  # N — absolute force for contact entry
CONTACT_FORCE_DELTA  = 0.10  # N — delta above idle baseline for contact entry
CONTACT_FORCE_CAP    = 0.60  # N — cap on threshold
FORCE_BALANCE_TOL    = 0.35  # N — asymmetry tolerance during press
MAX_SIDE_EXTRA_PRESS = 0.0060 # m — max extra press to balance
SIDE_PRESS_STEP      = 0.0010 # m — incremental side-press step during precontact/adaptive moves
RAMP_SIDE_PRESS_STEP = RAMP_STEP # smaller side-bias increment during quasi-static compression
FORCE_DIFF_ABORT     = 4.0   # N — force asymmetry hard abort
MAX_FORCE_MAG_ABORT  = 8.0   # N — single-side force hard abort

# Motion safety envelope.
MIN_EE_SEPARATION_M   = 0.020  # m — minimum 3-D EE separation before abort
MIN_EE_SEPARATION_REBASE_MAX_M = 0.003  # m — largest idle-window shortfall allowed for re-baselining
MIN_EE_SEPARATION_REBASE_TOL_M = 5e-4  # m — preserve a small buffer below the accepted idle separation
MAX_EE_X_GAP_MARGIN_M = 0.020  # m — margin above measured home x-gap

# Abort-time torque disable.
DEFAULT_TORQUE_DISABLE_BAUD = 1000000
DISABLE_TORQUE_SCRIPT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "ws",
        "src",
        "omx_dual_bringup",
        "scripts",
        "disable_torque.py",
    )
)


# ═══════════════════════════════════════════════════════════════════════════════
# 3) DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DepthCameraCalibration:
    source: str
    is_approximate: bool
    camera_position_world_m: Tuple[float, float, float]
    target_position_world_m: Tuple[float, float, float]
    camera_right_world: Tuple[float, float, float]
    camera_up_world: Tuple[float, float, float]
    camera_forward_world: Tuple[float, float, float]
    world_to_camera_matrix: List[List[float]]
    camera_to_world_matrix: List[List[float]]
    assumptions: List[str]

@dataclass
class RunResult:
    case:         str
    K_lat_Npm:    float
    rep:          int
    ts:           str
    passed:       bool
    ablation_index: int = 0
    abort_reason: str = ""
    # timing [s]
    t_contact_s:      float = float("nan")  # mutual contact established
    contact_duration_s: float = float("nan")
    t_mb_zero_s:      float = float("nan")  # first sustained m_b < 0 (if margin topic present)
    t_mc_zero_s:      float = float("nan")  # first sustained m_c < 0
    t_w_3mm_s:        float = float("nan")  # lateral deflection > 3 mm
    t_phi_4deg_s:     float = float("nan")  # yaw > 4 deg
    P_s_at_onset_N:  float = float("nan")  # internal spring load at onset
    P_s_max_N:        float = float("nan")  # peak internal spring load
    # margin overlays (continuous values, not threshold-crossing times)
    mb_at_contact_N:  float = float("nan")  # m_b value when bilateral contact first established
    mb_min_N:         float = float("nan")  # minimum m_b seen during the run (most negative = nearest onset)
    mc_at_contact_N:  float = float("nan")  # m_c value at contact establishment
    mc_min_N:         float = float("nan")  # minimum m_c seen during the run
    P_b_th_N:         float = P_B_THEORY_N
    P_c_th_N:         float = float("nan")  # computed from K_lat at run time
    hw_realisable:    bool  = True
    # identified (measured) K_lat
    K_lat_meas_Npm:     float = float("nan")  # Delta_Fx / Delta_xTCP from perturb test
    K_lat_meas_stderr:  float = float("nan")  # std error of linear fit [N/m]
    # file paths
    alignment_source:  str = ""
    alignment_status:  str = ""
    alignment_message: str = ""
    robot1_z_trim_m:   float = 0.0
    robot2_z_trim_m:   float = 0.0
    camera_aruco_records: int = 0
    camera_aruco_valid_records: int = 0
    camera_aruco_all_markers_records: int = 0
    camera_aruco_dropout_run_count: int = 0
    camera_aruco_angle_flip_count: int = 0
    csv_path:  str = ""
    json_path: str = ""
    camera_aruco_raw_jsonl_path: str = ""
    camera_aruco_summary_json_path: str = ""
    camera_aruco_csv_dir_path: str = ""
    external_capture_metadata: Dict[str, Any] = field(default_factory=dict)
    aruco_summary_row: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibResult:
    """Spring parameter calibration results (effective experimental quantities)."""
    ts:                  str   = ""
    L0_m:                float = float("nan")   # unloaded spring length from camera [m]
    L0_std_m:            float = float("nan")   # std across CALIB_L0_SAMPLES samples
    k_a_Npm:             float = float("nan")   # axial stiffness: P_s = k_a*DL + b [N/m]
    k_a_stderr:          float = float("nan")   # OLS standard error [N/m]
    P_b_hat_N:           float = float("nan")   # buckling threshold from change-point [N]
    P_b_stderr_N:        float = float("nan")   # +/- half-step uncertainty [N]
    B_eff_Nm2:           float = float("nan")   # P_b*L0^2/pi^2  (effective bending stiffness)
    k_theta_Nm_per_rad:  float = float("nan")   # contact rotational stiffness [N.m/rad]
    k_theta_stderr:      float = float("nan")   # OLS standard error [N.m/rad]
    K_lat_star_Npm:      float = float("nan")   # P_b^2 / k_theta  [N/m]
    K_lat_meas_Npm:      float = float("nan")   # measured closed-loop K_lat at HW ceiling [N/m]
    K_lat_meas_stderr:   float = float("nan")   # perturbation-test fit SE [N/m]
    json_path:           str   = ""
    external_capture_metadata: Dict[str, Any] = field(default_factory=dict)


class CalibrationSafetyAbort(RuntimeError):
    """Raised when a calibration or perturbation move trips the EE safety envelope."""


# ═══════════════════════════════════════════════════════════════════════════════
# 4) HARNESS NODE
# ═══════════════════════════════════════════════════════════════════════════════

class AblationHarness(Node):
    """
    Extends harness_v2 with:
      • Outer loop over K_lat cases × repetitions.
      • Per-run onset event detection (m_b, m_c margins, w, yaw).
      • Graceful K_lat change between cases (manual / topic / ros2_param).
      • Consolidated summary.csv output compatible with sim summary.csv.
    """

    # ── construction ──────────────────────────────────────────────────────────
    def __init__(
        self,
        log_root: str,
        skip_klat_command: bool = False,
        min_ee_separation: float = MIN_EE_SEPARATION_M,
        enforce_ee_separation: bool = True,
        max_ee_x_gap: Optional[float] = None,
        max_ee_x_gap_margin: float = MAX_EE_X_GAP_MARGIN_M,
        robot1_y_offset: float = 0.0,
        robot2_y_offset: float = 0.0,
        robot1_z_offset: float = 0.0,
        robot2_z_offset: float = 0.0,
        alignment_source: str = DEFAULT_ALIGNMENT_SOURCE,
        alignment_policy: str = DEFAULT_ALIGNMENT_POLICY,
        alignment_max_auto_z_trim: float = DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
        alignment_fallback_w_tol: float = DEFAULT_ALIGNMENT_FALLBACK_W_TOL_M,
        alignment_fallback_yaw_tol_deg: float = DEFAULT_ALIGNMENT_FALLBACK_YAW_TOL_DEG,
        alignment_fallback_psi_diff_tol_deg: float = DEFAULT_ALIGNMENT_FALLBACK_PSI_DIFF_TOL_DEG,
        abort_disable_torque: bool = False,
        robot1_port: Optional[str] = None,
        robot2_port: Optional[str] = None,
        torque_disable_baud: int = DEFAULT_TORQUE_DISABLE_BAUD,
        color_image_topic: str = DEFAULT_COLOR_IMAGE_TOPIC,
        color_camera_info_topic: str = DEFAULT_COLOR_CAMERA_INFO_TOPIC,
        depth_image_topic: str = DEFAULT_DEPTH_IMAGE_TOPIC,
        depth_camera_info_topic: str = DEFAULT_DEPTH_CAMERA_INFO_TOPIC,
        camera_aruco_summary_topic: str = DEFAULT_CAMERA_ARUCO_SUMMARY_TOPIC,
        raw_camera_stream_rate_hz: float = DEFAULT_RAW_CAMERA_STREAM_RATE_HZ,
        raw_camera_video_fps: float = DEFAULT_RAW_CAMERA_VIDEO_FPS,
        capture_cmd_file: Optional[str] = None,
        capture_buffer_pre_s: float = DEFAULT_CAPTURE_BUFFER_PRE_S,
        capture_buffer_post_s: float = DEFAULT_CAPTURE_BUFFER_POST_S,
    ) -> None:
        super().__init__("hw_ablation_harness")
        self.skip_klat_command = bool(skip_klat_command)
        self.min_ee_separation = float(min_ee_separation)
        self.enforce_ee_separation = bool(enforce_ee_separation)
        self.max_ee_x_gap = float(max_ee_x_gap) if max_ee_x_gap is not None else None
        self.max_ee_x_gap_margin = float(max_ee_x_gap_margin)
        self.robot1_y_offset = float(robot1_y_offset)
        self.robot2_y_offset = float(robot2_y_offset)
        self.robot1_z_offset = float(robot1_z_offset)
        self.robot2_z_offset = float(robot2_z_offset)
        self.alignment_source = str(alignment_source)
        self.alignment_policy = str(alignment_policy)
        self.alignment_max_auto_z_trim = float(alignment_max_auto_z_trim)
        self.alignment_fallback_w_tol = float(alignment_fallback_w_tol)
        self.alignment_fallback_yaw_tol_deg = float(alignment_fallback_yaw_tol_deg)
        self.alignment_fallback_psi_diff_tol_deg = float(alignment_fallback_psi_diff_tol_deg)
        self.abort_disable_torque = bool(abort_disable_torque)
        self.robot1_port = robot1_port
        self.robot2_port = robot2_port
        self.torque_disable_baud = int(torque_disable_baud)
        self.color_image_topic = str(color_image_topic)
        self.color_camera_info_topic = str(color_camera_info_topic)
        self.depth_image_topic = str(depth_image_topic)
        self.depth_camera_info_topic = str(depth_camera_info_topic)
        self.camera_aruco_summary_topic = str(camera_aruco_summary_topic)
        self.raw_camera_stream_rate_hz = max(0.0, float(raw_camera_stream_rate_hz))
        self.raw_camera_video_fps = max(0.0, float(raw_camera_video_fps))
        self.capture_cmd_file = (
            str(capture_cmd_file).strip()
            if capture_cmd_file is not None and str(capture_cmd_file).strip()
            else None
        )
        self.capture_buffer_pre_s = max(0.0, float(capture_buffer_pre_s))
        self.capture_buffer_post_s = max(0.0, float(capture_buffer_post_s))
        self._raw_camera_stream_period_s = (
            0.0 if self.raw_camera_stream_rate_hz <= 0.0 else 1.0 / self.raw_camera_stream_rate_hz
        )
        self.disable_torque_script = DISABLE_TORQUE_SCRIPT
        self._torque_disable_requested = False
        self._current_ablation_index = 0

        # ── publishers ────────────────────────────────────────────────────────
        self.pub1 = self.create_publisher(
            PoseStamped, "/robot1/robot1_variable_stiffness/waypoint_command", 10)
        self.pub2 = self.create_publisher(
            PoseStamped, "/robot2/robot2_variable_stiffness/waypoint_command", 10)

        if SET_KLAT_METHOD == "topic":
            self.klat_pub1 = self.create_publisher(Float64, KLAT_TOPIC_R1, 1)
            self.klat_pub2 = self.create_publisher(Float64, KLAT_TOPIC_R2, 1)

        # ── subscriptions ─────────────────────────────────────────────────────
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(JointState, "/robot1/joint_states",      self._cb_js1,   qos)
        self.create_subscription(JointState, "/robot2/joint_states",      self._cb_js2,   qos)
        self.create_subscription(Float64MultiArray, "/robot1/robot1_variable_stiffness/joint_velocities", self._cb_jv1, qos)
        self.create_subscription(Float64MultiArray, "/robot2/robot2_variable_stiffness/joint_velocities", self._cb_jv2, qos)
        self.create_subscription(Point, "/robot1/robot1_variable_stiffness/end_effector_position", self._cb_ee1, qos)
        self.create_subscription(Point, "/robot2/robot2_variable_stiffness/end_effector_position", self._cb_ee2, qos)
        self.create_subscription(Pose,  "/robot1/robot1_variable_stiffness/cartesian_pose_desired", self._cb_des1, qos)
        self.create_subscription(Pose,  "/robot2/robot2_variable_stiffness/cartesian_pose_desired", self._cb_des2, qos)
        self.create_subscription(WrenchStamped, "/robot1/robot1_variable_stiffness/contact_wrench", self._cb_cx1, qos)
        self.create_subscription(WrenchStamped, "/robot2/robot2_variable_stiffness/contact_wrench", self._cb_cx2, qos)
        self.create_subscription(Bool, "/robot1/robot1_variable_stiffness/contact_valid",  self._cb_cv1, qos)
        self.create_subscription(Bool, "/robot2/robot2_variable_stiffness/contact_valid",  self._cb_cv2, qos)
        self.create_subscription(Bool, "/robot1/robot1_variable_stiffness/waypoint_active", self._cb_wp1, qos)
        self.create_subscription(Bool, "/robot2/robot2_variable_stiffness/waypoint_active", self._cb_wp2, qos)
        self.create_subscription(Image, self.color_image_topic, self._cb_color_image, qos)
        self.create_subscription(CameraInfo, self.color_camera_info_topic, self._cb_color_camera_info, qos)
        self.create_subscription(Image, self.depth_image_topic, self._cb_depth_image, qos)
        self.create_subscription(CameraInfo, self.depth_camera_info_topic, self._cb_depth_camera_info, qos)
        self.create_subscription(String, self.camera_aruco_summary_topic, self._cb_camera_aruco_summary, qos)
        self.create_subscription(Bool, ARUCO_ALIGNMENT_VALID_TOPIC, self._cb_aruco_alignment_valid, qos)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC, self._cb_aruco_robot1_z_trim, qos)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC, self._cb_aruco_robot2_z_trim, qos)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_ROLL_DEG_TOPIC, self._cb_aruco_alignment_roll_deg, qos)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC, self._cb_aruco_alignment_center_y_error, qos)

        # Optional instability-margin topics (published by the online margin node).
        # If absent the harness still runs; t_mb_zero / t_mc_zero stay nan.
        self.create_subscription(Float64, "/spring_monitor/buckling_margin",  self._cb_mb,  qos)
        self.create_subscription(Float64, "/spring_monitor/contact_margin",   self._cb_mc,  qos)
        # Optional deflection topics from depth camera processing node.
        self.create_subscription(Float64, "/spring_monitor/lateral_deflection_m", self._cb_w,   qos)
        self.create_subscription(Float64, "/spring_monitor/tcp_yaw_deg",          self._cb_yaw, qos)
        # Optional spring geometry topics (published by camera processing node).
        # spring_length_m            : end-to-end distance between endcap centres
        # contact_rotation_{L,R}_rad : psi = alpha_cap - alpha_s (endcap vs spring axis)
        self.create_subscription(Float64, "/spring_monitor/spring_length_m",        self._cb_sl,    qos)
        self.create_subscription(Float64, "/spring_monitor/contact_rotation_L_rad", self._cb_psi_l, qos)
        self.create_subscription(Float64, "/spring_monitor/contact_rotation_R_rad", self._cb_psi_r, qos)

        # ── state ─────────────────────────────────────────────────────────────
        self.js1: Optional[JointState]  = None
        self.js2: Optional[JointState]  = None
        self.ee1: Optional[Point]       = None
        self.ee2: Optional[Point]       = None
        self.des1: Optional[Pose]       = None
        self.des2: Optional[Pose]       = None
        self.contact_fx1 = float("nan")
        self.contact_fx2 = float("nan")
        self.contact_valid_1 = False
        self.contact_valid_2 = False
        self.wp1_active: Optional[bool] = None
        self.wp2_active: Optional[bool] = None
        self.mb_latest  = float("nan")  # buckling margin (signed)
        self.mc_latest  = float("nan")  # contact-rotation margin (signed)
        self.w_latest   = float("nan")  # lateral deflection [m]
        self.yaw_latest = float("nan")  # TCP yaw [deg]
        self.color_image: Optional[Image] = None
        self.color_camera_info: Optional[CameraInfo] = None
        self.depth_image: Optional[Image] = None
        self.depth_camera_info: Optional[CameraInfo] = None
        # camera geometry (NaN until /spring_monitor/spring_length_m etc. arrive)
        self.spring_length_latest = float("nan")  # spring end-to-end length [m]
        self.psi_L_latest         = float("nan")  # left  contact rotation [rad]
        self.psi_R_latest         = float("nan")  # right contact rotation [rad]
        self.aruco_alignment_valid = False
        self.aruco_robot1_z_trim = float("nan")
        self.aruco_robot2_z_trim = float("nan")
        self.aruco_alignment_roll_deg = float("nan")
        self.aruco_alignment_center_y_error = float("nan")
        self.alignment_robot1_y_trim = 0.0
        self.alignment_robot2_y_trim = 0.0
        self.alignment_robot1_z_trim = 0.0
        self.alignment_robot2_z_trim = 0.0
        self.last_alignment_source = "unchecked"
        self.last_alignment_status = "unchecked"
        self.last_alignment_message = "alignment not checked yet"

        self.current_phase        = "init"
        self.current_offset_x1    = float("nan")
        self.current_offset_x2    = float("nan")
        self.current_press_x1     = 0.0
        self.current_press_x2     = 0.0
        self.press_cmd_sign1: Optional[float] = None
        self.press_cmd_sign2: Optional[float] = None
        self.idle_start_x1: Optional[float] = None
        self.idle_start_x2: Optional[float] = None
        self.current_contact_mode = "none"
        self.baseline_fx1         = float("nan")
        self.baseline_fx2         = float("nan")
        self._last_snap_t: Optional[float] = None
        self._last_ee1_pos: Optional[Tuple[float, float, float]] = None
        self._last_ee2_pos: Optional[Tuple[float, float, float]] = None
        self._last_ee1_vel: Optional[Tuple[float, float, float]] = None
        self._last_ee2_vel: Optional[Tuple[float, float, float]] = None

        # ── per-run onset tracking ─────────────────────────────────────────────
        self._run_start_t         = 0.0
        self._t_contact           = float("nan")
        self._t_mb_zero           = float("nan")
        self._t_mc_zero           = float("nan")
        self._t_w_3mm             = float("nan")
        self._t_phi_4deg          = float("nan")
        self._p_s_onset           = float("nan")
        self._p_s_max              = 0.0
        self._mb_neg_count        = 0   # consecutive steps m_b < 0
        self._mc_neg_count        = 0   # consecutive steps m_c < 0
        _ONSET_HOLD_STEPS         = 3   # steps needed to confirm sustained crossing
        self._ONSET_HOLD_STEPS    = _ONSET_HOLD_STEPS

        # ── logging ───────────────────────────────────────────────────────────
        self.log_root = log_root
        os.makedirs(self.log_root, exist_ok=True)

        # 50 Hz snapshot timer (active during runs, disabled between runs)
        self._log_cols: List[str] = []
        self._run_csv_path        = ""
        self._run_csv_file        = None
        self._run_csv_writer: Optional[csv.DictWriter] = None
        self._logging_active      = False
        self.create_timer(0.02, self._log_tick)  # 50 Hz
        self._active_depth_frame_output_dir: Optional[str] = None
        self._active_depth_frame_label_prefix: Optional[str] = None
        self._active_depth_frame_buildstamp: str = ""
        self._active_raw_camera_stream_dir: Optional[str] = None
        self._raw_camera_stream_last_write_time: Dict[str, float] = {}
        self._raw_camera_stream_saved_counts: Dict[str, int] = {}
        self._raw_camera_stream_records: Dict[str, List[Dict[str, Any]]] = {}
        self._raw_camera_stream_info_paths: Dict[str, str] = {}
        self._raw_camera_stream_error_logged = False
        self.last_camera_aruco_summary: Optional[Dict[str, Any]] = None
        self._active_camera_aruco_capture_dir: Optional[str] = None
        self._camera_aruco_records: List[Dict[str, Any]] = []
        self._camera_aruco_record_sequence = 0
        self._camera_aruco_capture_error_logged = False
        self._last_camera_aruco_capture_artifacts: Dict[str, Any] = {}
        self._depth_frame_unavailable_logged = False
        self._active_external_capture: Optional[Dict[str, Any]] = None
        self._last_external_capture_metadata: Dict[str, Any] = {}
        self._contact_duration_s_total = 0.0
        self._contact_active = False
        self._contact_active_since_s = float("nan")
        self._contact_wall_time_ns: Optional[int] = None
        self._contact_color_image_stamp_ns: Optional[int] = None
        self._contact_depth_image_stamp_ns: Optional[int] = None

        # human-readable log
        self._ablation_log_path = os.path.join(self.log_root, "ablation_log.txt")
        self._ablation_log: List[str] = []

    # ── ROS callbacks ─────────────────────────────────────────────────────────
    def _cb_js1(self, m): self.js1 = m
    def _cb_js2(self, m): self.js2 = m

    def _cb_jv1(self, m: Float64MultiArray):
        if self.js1 is None:
            self.js1 = JointState()
        self.js1.velocity = list(m.data)

    def _cb_jv2(self, m: Float64MultiArray):
        if self.js2 is None:
            self.js2 = JointState()
        self.js2.velocity = list(m.data)

    def _cb_ee1(self, m): self.ee1 = m
    def _cb_ee2(self, m): self.ee2 = m
    def _cb_des1(self, m): self.des1 = m
    def _cb_des2(self, m): self.des2 = m
    def _cb_color_image(self, m: Image):
        self.color_image = m
        self._record_raw_camera_stream_frame("color", self.color_image_topic, m, self.color_camera_info)

    def _cb_color_camera_info(self, m: CameraInfo): self.color_camera_info = m

    def _cb_depth_image(self, m: Image):
        self.depth_image = m
        self._record_raw_camera_stream_frame("aligned_depth", self.depth_image_topic, m, self.depth_camera_info)

    def _cb_depth_camera_info(self, m: CameraInfo): self.depth_camera_info = m

    def _cb_camera_aruco_summary(self, m: String):
        try:
            summary = json.loads(m.data)
        except json.JSONDecodeError as exc:
            if not self._camera_aruco_capture_error_logged:
                self._camera_aruco_capture_error_logged = True
                self._alog(
                    "camera_aruco summary capture skipped invalid JSON from "
                    f"{self.camera_aruco_summary_topic}: {exc}"
                )
            return
        if not isinstance(summary, dict):
            if not self._camera_aruco_capture_error_logged:
                self._camera_aruco_capture_error_logged = True
                self._alog(
                    "camera_aruco summary capture skipped non-dict payload from "
                    f"{self.camera_aruco_summary_topic}"
                )
            return
        self.last_camera_aruco_summary = summary
        if self._active_camera_aruco_capture_dir is None:
            return
        self._camera_aruco_records.append(self._build_camera_aruco_capture_record(summary))
        self._camera_aruco_record_sequence += 1

    def _cb_aruco_alignment_valid(self, m: Bool): self.aruco_alignment_valid = bool(m.data)
    def _cb_aruco_robot1_z_trim(self, m: Float64): self.aruco_robot1_z_trim = m.data
    def _cb_aruco_robot2_z_trim(self, m: Float64): self.aruco_robot2_z_trim = m.data
    def _cb_aruco_alignment_roll_deg(self, m: Float64): self.aruco_alignment_roll_deg = m.data
    def _cb_aruco_alignment_center_y_error(self, m: Float64): self.aruco_alignment_center_y_error = m.data

    def _cb_cx1(self, m: WrenchStamped):
        self.contact_fx1 = abs(m.wrench.force.x)

    def _cb_cx2(self, m: WrenchStamped):
        self.contact_fx2 = abs(m.wrench.force.x)

    def _cb_cv1(self, m: Bool): self.contact_valid_1 = bool(m.data)
    def _cb_cv2(self, m: Bool): self.contact_valid_2 = bool(m.data)
    def _cb_wp1(self, m: Bool): self.wp1_active = bool(m.data)
    def _cb_wp2(self, m: Bool): self.wp2_active = bool(m.data)

    def _cb_mb(self, m: Float64):
        self.mb_latest = m.data
        # track minimum (most negative = nearest instability)
        if math.isnan(self._mb_min) or m.data < self._mb_min:
            self._mb_min = m.data
        # onset detection: sustained negative crossing
        now = self.now_s() - self._run_start_t
        if m.data < 0.0:
            self._mb_neg_count += 1
            if self._mb_neg_count >= self._ONSET_HOLD_STEPS and math.isnan(self._t_mb_zero):
                self._t_mb_zero = now
                ps = self._ps()
                if not math.isnan(ps):
                    self._p_s_onset = ps
        else:
            self._mb_neg_count = 0

    def _cb_mc(self, m: Float64):
        self.mc_latest = m.data
        # track minimum
        if math.isnan(self._mc_min) or m.data < self._mc_min:
            self._mc_min = m.data
        now = self.now_s() - self._run_start_t
        if m.data < 0.0:
            self._mc_neg_count += 1
            if self._mc_neg_count >= self._ONSET_HOLD_STEPS and math.isnan(self._t_mc_zero):
                self._t_mc_zero = now
        else:
            self._mc_neg_count = 0

    def _cb_w(self, m: Float64):
        self.w_latest = m.data
        now = self.now_s() - self._run_start_t
        if m.data > W_MAX_THRESH_M and math.isnan(self._t_w_3mm):
            self._t_w_3mm = now

    def _cb_yaw(self, m: Float64):
        self.yaw_latest = m.data
        now = self.now_s() - self._run_start_t
        if abs(m.data) > PHI_MAX_THRESH_DEG and math.isnan(self._t_phi_4deg):
            self._t_phi_4deg = now

    def _cb_sl(self, m: Float64):    self.spring_length_latest = m.data
    def _cb_psi_l(self, m: Float64): self.psi_L_latest = m.data
    def _cb_psi_r(self, m: Float64): self.psi_R_latest = m.data

    # ── utilities ─────────────────────────────────────────────────────────────
    def _normalize_vec(self, vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
        norm = math.sqrt(sum(component * component for component in vec))
        if norm <= 1e-12:
            raise ValueError("Cannot normalize a near-zero vector")
        return tuple(component / norm for component in vec)

    def _cross_vec(
        self,
        left: Tuple[float, float, float],
        right: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def _dot_vec(
        self,
        left: Tuple[float, float, float],
        right: Tuple[float, float, float],
    ) -> float:
        return sum(a * b for a, b in zip(left, right))

    def default_depth_camera_calibration(self) -> DepthCameraCalibration:
        target = (0.0, 0.0, SPRING_Z_OFFSET_M)
        camera_position = (
            0.0,
            -DEFAULT_DEPTH_CAMERA_HORIZONTAL_OFFSET_M,
            DEFAULT_DEPTH_CAMERA_HEIGHT_M,
        )
        up_guess = (0.0, 0.0, 1.0)

        forward = self._normalize_vec(
            tuple(target[idx] - camera_position[idx] for idx in range(3))
        )
        right = self._normalize_vec(self._cross_vec(forward, up_guess))
        up = self._normalize_vec(self._cross_vec(right, forward))

        rotation_rows = [right, up, forward]
        world_to_camera = [
            [axis[0], axis[1], axis[2], -self._dot_vec(axis, camera_position)]
            for axis in rotation_rows
        ]
        world_to_camera.append([0.0, 0.0, 0.0, 1.0])

        camera_to_world = [
            [right[0], up[0], forward[0], camera_position[0]],
            [right[1], up[1], forward[1], camera_position[1]],
            [right[2], up[2], forward[2], camera_position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

        return DepthCameraCalibration(
            source=DEFAULT_DEPTH_CAMERA_SOURCE,
            is_approximate=True,
            camera_position_world_m=camera_position,
            target_position_world_m=target,
            camera_right_world=right,
            camera_up_world=up,
            camera_forward_world=forward,
            world_to_camera_matrix=world_to_camera,
            camera_to_world_matrix=camera_to_world,
            assumptions=[
                "Spring center is at world (0, 0, SPRING_Z_OFFSET_M)",
                "Camera lies on the negative-Y side of the rig",
                "Camera optical axis points at the spring center",
                "Camera up is aligned with world +Z",
                "Extrinsics are an approximate fallback unless replaced by a separate calibration workflow",
            ],
        )

    def write_depth_camera_calibration(self, output_dir: str) -> str:
        calibration = self.default_depth_camera_calibration()
        calibration_path = os.path.join(output_dir, "depth_camera_calibration.json")
        with open(calibration_path, "w") as handle:
            json.dump(
                {
                    "source": calibration.source,
                    "is_approximate": calibration.is_approximate,
                    "camera_position_world_m": list(calibration.camera_position_world_m),
                    "target_position_world_m": list(calibration.target_position_world_m),
                    "camera_right_world": list(calibration.camera_right_world),
                    "camera_up_world": list(calibration.camera_up_world),
                    "camera_forward_world": list(calibration.camera_forward_world),
                    "world_to_camera_matrix": calibration.world_to_camera_matrix,
                    "camera_to_world_matrix": calibration.camera_to_world_matrix,
                    "assumptions": calibration.assumptions,
                },
                handle,
                indent=2,
            )
        self._alog(
            "Depth-camera extrinsics metadata written to "
            f"{calibration_path} ({calibration.source}, approximate fallback)"
        )
        return calibration_path

    def _message_stamp_ns(self, message: Optional[Any]) -> Optional[int]:
        if message is None:
            return None
        header = getattr(message, "header", None)
        stamp = getattr(header, "stamp", None)
        sec = getattr(stamp, "sec", None)
        nanosec = getattr(stamp, "nanosec", None)
        if sec is None or nanosec is None:
            return None
        try:
            return (int(sec) * 1_000_000_000) + int(nanosec)
        except (TypeError, ValueError):
            return None

    def _format_capture_cmd_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            return f"{value:.6f}"
        text = str(value).strip()
        if not text:
            return None
        return text.replace(" ", "_")

    def _write_capture_cmd(self, action: str, **fields: Any) -> None:
        if not self.capture_cmd_file:
            return
        parts = [action]
        for key, value in fields.items():
            formatted = self._format_capture_cmd_value(value)
            if formatted is None:
                continue
            parts.append(f"{key}={formatted}")
        target_path = os.path.abspath(self.capture_cmd_file)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        temp_path = target_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(" ".join(parts) + "\n")
        os.replace(temp_path, target_path)

    def _start_external_capture_session(
        self,
        *,
        phase: str,
        name: str,
        ablation_index: int = 0,
        case: str = "",
        rep: int = 0,
        k_lat_npm: float = float("nan"),
    ) -> None:
        self._last_external_capture_metadata = {}
        if not self.capture_cmd_file:
            self._active_external_capture = None
            return
        start_wall_time_ns = time.time_ns()
        self._active_external_capture = {
            "phase": phase,
            "name": name,
            "ablation_index": int(ablation_index),
            "case": case,
            "rep": int(rep),
            "k_lat_npm": float(k_lat_npm),
            "capture_start_wall_time_ns": start_wall_time_ns,
            "capture_start_monotonic_s": time.monotonic(),
        }
        self._write_capture_cmd(
            "START",
            name=name,
            phase=phase,
            ablation_index=ablation_index,
            case=case,
            rep=rep,
            k_lat_npm=k_lat_npm,
            capture_start_wall_time_ns=start_wall_time_ns,
            buffer_pre_s=self.capture_buffer_pre_s,
            buffer_post_s=self.capture_buffer_post_s,
        )

    def _mark_contact_sync_event(self) -> None:
        if self._contact_wall_time_ns is not None:
            return
        if self._active_external_capture is None:
            return
        self._contact_wall_time_ns = time.time_ns()
        self._contact_color_image_stamp_ns = self._message_stamp_ns(self.color_image)
        self._contact_depth_image_stamp_ns = self._message_stamp_ns(self.depth_image)

    def _update_contact_duration_state(self) -> None:
        if not math.isfinite(self._run_start_t):
            return
        t_run = self.now_s() - self._run_start_t
        bilateral_contact = self._raw_bilateral_contact_detected()
        if bilateral_contact and not self._contact_active:
            self._contact_active = True
            self._contact_active_since_s = t_run
        elif not bilateral_contact and self._contact_active:
            if self.fin(self._contact_active_since_s):
                self._contact_duration_s_total += max(0.0, t_run - self._contact_active_since_s)
            self._contact_active = False
            self._contact_active_since_s = float("nan")

    def _final_contact_duration_s(self) -> float:
        if not self.fin(self._t_contact):
            return float("nan")
        duration_s = float(self._contact_duration_s_total)
        if self._contact_active and self.fin(self._contact_active_since_s):
            duration_s += max(0.0, (self.now_s() - self._run_start_t) - self._contact_active_since_s)
        return duration_s

    def _stop_external_capture_session(self, *, status: str) -> Dict[str, Any]:
        session = self._active_external_capture
        self._active_external_capture = None
        if session is None:
            self._last_external_capture_metadata = {}
            return {}

        capture_end_wall_time_ns = time.time_ns()
        capture_duration_s = max(0.0, time.monotonic() - session["capture_start_monotonic_s"])
        metadata: Dict[str, Any] = {
            "enabled": True,
            "phase": session["phase"],
            "name": session["name"],
            "ablation_index": session["ablation_index"],
            "case": session["case"] or None,
            "rep": session["rep"] or None,
            "k_lat_npm": session["k_lat_npm"] if self.fin(session["k_lat_npm"]) else None,
            "status": status,
            "buffer_pre_s": self.capture_buffer_pre_s,
            "buffer_post_s": self.capture_buffer_post_s,
            "capture_start_wall_time_ns": session["capture_start_wall_time_ns"],
            "capture_end_wall_time_ns": capture_end_wall_time_ns,
            "capture_duration_s": capture_duration_s,
        }

        if session["phase"] == "ablation_run":
            contact_time_s = self._t_contact if self.fin(self._t_contact) else None
            contact_duration_s = self._final_contact_duration_s()
            contact_window_start_s = None
            contact_window_end_s = None
            contact_window_start_wall_time_ns = None
            contact_window_end_wall_time_ns = None
            if contact_time_s is not None:
                contact_window_start_s = max(0.0, contact_time_s - self.capture_buffer_pre_s)
                contact_window_end_s = contact_time_s + self.capture_buffer_post_s
            if self._contact_wall_time_ns is not None:
                contact_window_start_wall_time_ns = self._contact_wall_time_ns - int(
                    round(self.capture_buffer_pre_s * 1_000_000_000)
                )
                contact_window_end_wall_time_ns = self._contact_wall_time_ns + int(
                    round(self.capture_buffer_post_s * 1_000_000_000)
                )
            metadata.update(
                {
                    "contact_time_s": contact_time_s,
                    "contact_duration_s": contact_duration_s if self.fin(contact_duration_s) else None,
                    "contact_wall_time_ns": self._contact_wall_time_ns,
                    "contact_window_start_s": contact_window_start_s,
                    "contact_window_end_s": contact_window_end_s,
                    "contact_window_start_wall_time_ns": contact_window_start_wall_time_ns,
                    "contact_window_end_wall_time_ns": contact_window_end_wall_time_ns,
                    "contact_color_image_stamp_ns": self._contact_color_image_stamp_ns,
                    "contact_depth_image_stamp_ns": self._contact_depth_image_stamp_ns,
                }
            )

        self._last_external_capture_metadata = metadata
        self._write_capture_cmd("STOP", **metadata)
        return metadata

    def _build_camera_aruco_capture_record(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "record_type": "camera_aruco_harness_capture",
            "source": "hardware_harness_ablated",
            "sequence": self._camera_aruco_record_sequence,
            "wall_time_unix_ns": time.time_ns(),
            "run_time_s": self.now_s() - self._run_start_t if self._run_start_t > 0.0 else None,
            "color_image_stamp_ns": self._message_stamp_ns(self.color_image),
            "color_camera_info_stamp_ns": self._message_stamp_ns(self.color_camera_info),
            "depth_image_stamp_ns": self._message_stamp_ns(self.depth_image),
            "depth_camera_info_stamp_ns": self._message_stamp_ns(self.depth_camera_info),
            "summary": summary,
        }

    def _finalize_camera_aruco_capture(self) -> None:
        self._last_camera_aruco_capture_artifacts = {}
        if self._active_camera_aruco_capture_dir is None or not self._camera_aruco_records:
            return

        try:
            os.makedirs(self._active_camera_aruco_capture_dir, exist_ok=True)
            raw_jsonl_path = os.path.join(self._active_camera_aruco_capture_dir, "raw_signals.jsonl")
            with open(raw_jsonl_path, "w", encoding="utf-8") as handle:
                for record in self._camera_aruco_records:
                    handle.write(json.dumps(record, sort_keys=False))
                    handle.write("\n")

            summary = summarize_records(self._camera_aruco_records)
            summary["input_path"] = os.path.abspath(raw_jsonl_path)
            summary["label_prefix"] = self._active_depth_frame_label_prefix
            summary["capture_topic"] = self.camera_aruco_summary_topic

            summary_json_path = os.path.join(self._active_camera_aruco_capture_dir, "summary.json")
            with open(summary_json_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
                handle.write("\n")

            csv_dir_path = os.path.join(self._active_camera_aruco_capture_dir, "csv")
            csv_paths = write_summary_csv_exports(summary, csv_dir_path)
        except Exception as exc:
            self._alog(f"camera_aruco capture finalization skipped: {exc}")
            return

        self._last_camera_aruco_capture_artifacts = {
            "raw_jsonl_path": raw_jsonl_path,
            "summary_json_path": summary_json_path,
            "csv_dir_path": csv_dir_path,
            **csv_paths,
            "records": int(summary.get("records", 0) or 0),
            "valid_record_count": int(summary.get("valid_record_count", 0) or 0),
            "records_with_all_expected_labels": int(
                summary.get("records_with_all_expected_labels", 0) or 0
            ),
            "dropout_run_count": int(summary.get("dropout_run_count", 0) or 0),
            "angle_flip_candidate_count": int(summary.get("angle_flip_candidate_count", 0) or 0),
        }
        self._alog(
            "camera_aruco capture saved to "
            f"{self._active_camera_aruco_capture_dir} ({len(self._camera_aruco_records)} records)"
        )

    def _camera_aruco_capture_metadata(self) -> Dict[str, Any]:
        artifacts = self._last_camera_aruco_capture_artifacts
        return {
            "camera_aruco_records": int(artifacts.get("records", 0) or 0),
            "camera_aruco_valid_records": int(artifacts.get("valid_record_count", 0) or 0),
            "camera_aruco_all_markers_records": int(
                artifacts.get("records_with_all_expected_labels", 0) or 0
            ),
            "camera_aruco_dropout_run_count": int(artifacts.get("dropout_run_count", 0) or 0),
            "camera_aruco_angle_flip_count": int(
                artifacts.get("angle_flip_candidate_count", 0) or 0
            ),
            "camera_aruco_raw_jsonl_path": str(artifacts.get("raw_jsonl_path", "")),
            "camera_aruco_summary_json_path": str(artifacts.get("summary_json_path", "")),
            "camera_aruco_csv_dir_path": str(artifacts.get("csv_dir_path", "")),
        }

    def _set_active_depth_frame_context(self, output_dir: str, label_prefix: str) -> None:
        self._active_depth_frame_output_dir = output_dir
        self._active_depth_frame_label_prefix = label_prefix
        self._active_depth_frame_buildstamp = os.path.basename(os.path.abspath(output_dir))
        self._active_raw_camera_stream_dir = None
        self._raw_camera_stream_last_write_time = {}
        self._raw_camera_stream_saved_counts = {}
        self._raw_camera_stream_records = {}
        self._raw_camera_stream_info_paths = {}
        self._raw_camera_stream_error_logged = False
        self._active_camera_aruco_capture_dir = os.path.join(
            output_dir,
            DEPTH_CAMERA_FRAME_DIRNAME,
            f"{label_prefix}_camera_aruco",
        )
        self._camera_aruco_records = []
        self._camera_aruco_record_sequence = 0
        self._camera_aruco_capture_error_logged = False
        self._last_camera_aruco_capture_artifacts = {}
        if self._raw_camera_stream_period_s > 0.0:
            self._active_raw_camera_stream_dir = os.path.join(
                output_dir,
                DEPTH_CAMERA_FRAME_DIRNAME,
                f"{label_prefix}_raw_stream",
            )
            os.makedirs(self._active_raw_camera_stream_dir, exist_ok=True)
            self._alog(
                "Depth-camera raw stream capture enabled for "
                f"{label_prefix} at {self.raw_camera_stream_rate_hz:.3f} Hz"
            )
            if self.raw_camera_video_fps > 0.0:
                self._alog(
                    "Depth-camera raw stream video export enabled for "
                    f"{label_prefix} at {self.raw_camera_video_fps:.3f} FPS"
                )
        self._depth_frame_unavailable_logged = False

    def _clear_active_depth_frame_context(self) -> None:
        self._finalize_camera_aruco_capture()
        self._finalize_raw_camera_stream_video_exports()
        self._active_depth_frame_output_dir = None
        self._active_depth_frame_label_prefix = None
        self._active_depth_frame_buildstamp = ""
        self._active_raw_camera_stream_dir = None
        self._active_camera_aruco_capture_dir = None
        self._raw_camera_stream_last_write_time = {}
        self._raw_camera_stream_saved_counts = {}
        self._raw_camera_stream_records = {}
        self._raw_camera_stream_info_paths = {}
        self._raw_camera_stream_error_logged = False
        self._camera_aruco_records = []
        self._camera_aruco_record_sequence = 0
        self._camera_aruco_capture_error_logged = False
        self._depth_frame_unavailable_logged = False

    def _depth_camera_stamp_fields(self, cam_stamp_ns: int, buildstamp: Optional[str] = None) -> Dict[str, Any]:
        wall_time_ns = time.time_ns()
        wall_time_s = wall_time_ns / 1_000_000_000.0
        stamp_buildstamp = buildstamp if buildstamp is not None else self._active_depth_frame_buildstamp
        return {
            "timestamp": wall_time_s,
            "timestamp_ns": wall_time_ns,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(wall_time_s))
            + f".{int((wall_time_s % 1.0) * 1_000_000):06d}Z",
            "buildstamp": stamp_buildstamp,
            "cam_stamp": int(cam_stamp_ns),
            "cam_stamp_ns": int(cam_stamp_ns),
        }

    def _record_raw_camera_stream_frame(
        self,
        stream_name: str,
        topic_name: str,
        image_msg: Image,
        camera_info_msg: Optional[CameraInfo],
    ) -> None:
        if self._active_raw_camera_stream_dir is None or self._raw_camera_stream_period_s <= 0.0:
            return

        now = time.monotonic()
        last_write = self._raw_camera_stream_last_write_time.get(stream_name)
        if last_write is not None and (now - last_write) < self._raw_camera_stream_period_s:
            return

        stamp_ns = image_stamp_ns(image_msg)
        sequence = self._raw_camera_stream_saved_counts.get(stream_name, 0)
        stem = f"{stamp_ns}_{sequence:06d}_{stream_name}"

        try:
            image_path = os.path.join(
                self._active_raw_camera_stream_dir,
                stem + suggested_image_extension(image_msg.encoding),
            )
            image_meta = save_sensor_image(image_msg, image_path)

            meta_payload = {
                "label_prefix": self._active_depth_frame_label_prefix,
                "stream": stream_name,
                "topic": topic_name,
                "capture_mode": "raw_stream",
                **self._depth_camera_stamp_fields(stamp_ns),
                **image_meta,
            }
            if camera_info_msg is not None:
                info_path = self._raw_camera_stream_info_paths.get(stream_name)
                if info_path is None:
                    info_path = os.path.join(
                        self._active_raw_camera_stream_dir,
                        f"{stream_name}_camera_info.json",
                    )
                    with open(info_path, "w") as handle:
                        json.dump(camera_info_to_dict(camera_info_msg), handle, indent=2)
                    self._raw_camera_stream_info_paths[stream_name] = info_path
                meta_payload["camera_info_path"] = info_path

            meta_path = os.path.join(self._active_raw_camera_stream_dir, stem + "_meta.json")
            with open(meta_path, "w") as handle:
                json.dump(meta_payload, handle, indent=2)
        except ValueError as exc:
            if not self._raw_camera_stream_error_logged:
                self._raw_camera_stream_error_logged = True
                self._alog(f"Depth-camera raw stream capture skipped: {exc}")
            return

        self._raw_camera_stream_last_write_time[stream_name] = now
        self._raw_camera_stream_saved_counts[stream_name] = sequence + 1
        self._raw_camera_stream_records.setdefault(stream_name, []).append(
            {
                "stamp_ns": int(stamp_ns),
                "path": image_path,
                "meta_path": meta_path,
            }
        )

    def _finalize_raw_camera_stream_video_exports(self) -> None:
        if self._active_raw_camera_stream_dir is None or self.raw_camera_video_fps <= 0.0:
            return

        for stream_name, records in self._raw_camera_stream_records.items():
            if not records:
                continue
            target_path = os.path.join(
                self._active_raw_camera_stream_dir,
                f"{stream_name}_raw_stream.mp4",
            )
            try:
                export_meta = export_saved_sensor_stream_video(
                    [record["path"] for record in records],
                    target_path,
                    self.raw_camera_video_fps,
                )
            except ValueError as exc:
                self._alog(
                    "Depth-camera raw stream video export skipped for "
                    f"{stream_name}: {exc}"
                )
                continue

            last_stamp_ns = int(records[-1]["stamp_ns"])
            video_meta = {
                "label_prefix": self._active_depth_frame_label_prefix,
                "stream": stream_name,
                "capture_mode": "raw_stream_video",
                "capture_rate_hz": self.raw_camera_stream_rate_hz,
                "cam_stamp_start": int(records[0]["stamp_ns"]),
                "cam_stamp_end": last_stamp_ns,
                "source_frame_count": len(records),
                "source_first_frame_path": records[0]["path"],
                "source_last_frame_path": records[-1]["path"],
                **self._depth_camera_stamp_fields(last_stamp_ns),
                **export_meta,
            }
            meta_path = os.path.join(
                self._active_raw_camera_stream_dir,
                f"{stream_name}_raw_stream_video.json",
            )
            with open(meta_path, "w") as handle:
                json.dump(video_meta, handle, indent=2)
            self._alog(
                "Depth-camera raw stream video written: "
                f"{export_meta['path']} ({export_meta['frame_count']} frames at "
                f"{export_meta['fps']:.3f} FPS)"
            )

    def capture_depth_camera_snapshot(self, output_dir: str, label: str) -> List[str]:
        streams = [
            ("color", self.color_image_topic, self.color_image, self.color_camera_info),
            ("aligned_depth", self.depth_image_topic, self.depth_image, self.depth_camera_info),
        ]
        available_streams = [stream for stream in streams if stream[2] is not None]
        if not available_streams:
            if not self._depth_frame_unavailable_logged:
                self._depth_frame_unavailable_logged = True
                self._alog(
                    "Depth-camera frames unavailable for "
                    f"{label}; no frames received on configured camera topics "
                    f"{self.color_image_topic} or {self.depth_image_topic}"
                )
            return []

        frame_dir = os.path.join(output_dir, DEPTH_CAMERA_FRAME_DIRNAME)
        os.makedirs(frame_dir, exist_ok=True)
        saved_paths: List[str] = []

        for stream_name, topic_name, image_msg, camera_info_msg in available_streams:
            stamp_ns = image_stamp_ns(image_msg)
            stem = f"{stamp_ns}_{label}_{stream_name}"
            try:
                image_path = os.path.join(
                    frame_dir,
                    stem + suggested_image_extension(image_msg.encoding),
                )
                image_meta = save_sensor_image(image_msg, image_path)
            except ValueError as exc:
                self._alog(
                    f"Depth-camera snapshot skipped for {stream_name} ({label}): {exc}"
                )
                continue

            meta_payload = {
                "label": label,
                "stream": stream_name,
                "topic": topic_name,
                **self._depth_camera_stamp_fields(
                    stamp_ns,
                    buildstamp=os.path.basename(os.path.abspath(output_dir)),
                ),
                **image_meta,
            }
            if camera_info_msg is not None:
                info_path = os.path.join(frame_dir, stem + "_camera_info.json")
                with open(info_path, "w") as handle:
                    json.dump(camera_info_to_dict(camera_info_msg), handle, indent=2)
                meta_payload["camera_info_path"] = info_path

            meta_path = os.path.join(frame_dir, stem + "_meta.json")
            with open(meta_path, "w") as handle:
                json.dump(meta_payload, handle, indent=2)

            saved_paths.append(image_path)

        if saved_paths:
            self._alog(
                "Depth-camera snapshot saved for "
                f"{label}: {', '.join(os.path.basename(path) for path in saved_paths)}"
            )

        return saved_paths

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ps(self) -> float:
        """Internal spring load: average of the two inward contact-force magnitudes.

        P_s = (F_L + F_R) / 2  --  this is the quantity used in the scalar margin
        formulas and in all simulation CSVs.  Do NOT sum; summing doubles the axis
        and breaks hw-sim onset comparisons.
        """
        if self.fin(self.contact_fx1) and self.fin(self.contact_fx2):
            return 0.5 * (self.contact_fx1 + self.contact_fx2)
        return float("nan")

    def fin(self, x: float) -> bool:
        return not math.isnan(x) and not math.isinf(x)

    def ee_separation(self) -> float:
        if self.ee1 is None or self.ee2 is None:
            return float("inf")
        return math.sqrt(
            (self.ee1.x - self.ee2.x) ** 2
            + (self.ee1.y - self.ee2.y) ** 2
            + (self.ee1.z - self.ee2.z) ** 2
        )

    def ee_x_gap(self) -> float:
        if self.ee1 is None or self.ee2 is None:
            return 0.0
        return abs(self.ee1.x - self.ee2.x)

    def _ee_separation_check_enabled(self) -> bool:
        return self.enforce_ee_separation and self.min_ee_separation > 0.0

    def _initialize_motion_safety_limits(self) -> None:
        if self.max_ee_x_gap is not None:
            return
        if self.ee1 is not None and self.ee2 is not None:
            self.max_ee_x_gap = self.ee_x_gap() + self.max_ee_x_gap_margin
            return
        if self.des1 is not None and self.des2 is not None:
            self.max_ee_x_gap = (
                abs(self.des1.position.x - self.des2.position.x)
                + self.max_ee_x_gap_margin
            )

    def motion_safety_violation(self) -> Optional[str]:
        self._initialize_motion_safety_limits()

        if self._ee_separation_check_enabled():
            separation = self.ee_separation()
            if separation < self.min_ee_separation:
                message = (
                    f"EE separation {separation:.4f} m below safety limit "
                    f"{self.min_ee_separation:.4f} m"
                )
                self.get_logger().error(message)
                return message

        if self.max_ee_x_gap is not None:
            x_gap = self.ee_x_gap()
            if x_gap > self.max_ee_x_gap:
                message = (
                    f"EE x gap {x_gap:.4f} m above safety limit "
                    f"{self.max_ee_x_gap:.4f} m"
                )
                self.get_logger().error(message)
                return message

        return None

    def _raise_on_motion_safety_violation(self, context: str) -> None:
        violation = self.motion_safety_violation()
        if violation is None:
            return
        message = f"{context}: {violation}"
        self._alog(message)
        self.safe_abort()
        raise CalibrationSafetyAbort(message)

    def spin_for(self, duration: float, step: float = 0.02) -> None:
        end_t = self.now_s() + duration
        while self.now_s() < end_t:
            rclpy.spin_once(self, timeout_sec=step)

    def wait_for_waypoint_idle(self, timeout: float = 3.0) -> bool:
        t0 = self.now_s()
        saw_status = False
        saw_active = False
        while self.now_s() - t0 < timeout:
            statuses = []
            if self.wp1_active is not None:
                statuses.append(bool(self.wp1_active))
            if self.wp2_active is not None:
                statuses.append(bool(self.wp2_active))

            if statuses:
                saw_status = True
                if any(statuses):
                    saw_active = True
                if saw_active and not any(statuses):
                    return True

            rclpy.spin_once(self, timeout_sec=0.05)

        if not saw_status:
            self.spin_for(max(COMMAND_SAMPLE_DELAY, 2.25))
            return True
        return False

    def _make_pose(self, x: float, y: float, z: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "offset"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        return msg

    def publish_offsets(self, x1, y1, z1, x2, y2, z2,
                        repeats: Optional[int] = None,
                        dt:      Optional[float] = None) -> None:
        n  = COMMAND_REPEATS if repeats is None else repeats
        dt = COMMAND_DT      if dt      is None else dt
        self.current_offset_x1 = x1;  self.current_offset_x2 = x2
        self.current_press_x1  = x1;  self.current_press_x2  = x2
        m1 = self._make_pose(
            x1,
            y1 + self.robot1_y_offset + self.alignment_robot1_y_trim,
            z1 + self.robot1_z_offset + self.alignment_robot1_z_trim,
        )
        m2 = self._make_pose(
            x2,
            y2 + self.robot2_y_offset + self.alignment_robot2_y_trim,
            z2 + self.robot2_z_offset + self.alignment_robot2_z_trim,
        )
        for _ in range(n):
            ts = self.get_clock().now().to_msg()
            m1.header.stamp = ts;  m2.header.stamp = ts
            self.pub1.publish(m1); self.pub2.publish(m2)
            rclpy.spin_once(self, timeout_sec=dt)

    @staticmethod
    def _motion_sign(delta_x: float) -> Optional[float]:
        if delta_x > 1e-4:
            return 1.0
        if delta_x < -1e-4:
            return -1.0
        return None

    def _record_forward_motion(self, delta_x1: float, delta_x2: float) -> Tuple[bool, bool]:
        sign1 = self._motion_sign(delta_x1)
        sign2 = self._motion_sign(delta_x2)
        if sign1 is not None:
            self.press_cmd_sign1 = sign1
        if sign2 is not None:
            self.press_cmd_sign2 = sign2
        return sign1 is not None, sign2 is not None

    def _ensure_press_direction(self) -> None:
        if self.press_cmd_sign1 is not None and self.press_cmd_sign2 is not None:
            return

        saw_stable = False
        t0 = self.now_s()
        while self.now_s() - t0 < FORWARD_PHASE_TIMEOUT_S:
            if self.des1 is None or self.des2 is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue

            x0_1 = self.des1.position.x
            x0_2 = self.des2.position.x
            self.spin_for(IDLE_STABLE_SAMPLE_S)
            if self.des1 is None or self.des2 is None:
                continue

            dx1 = self.des1.position.x - x0_1
            dx2 = self.des2.position.x - x0_2
            if abs(dx1) <= IDLE_DESIRED_DRIFT_TOL_M and abs(dx2) <= IDLE_DESIRED_DRIFT_TOL_M:
                saw_stable = True
                continue
            if not saw_stable:
                continue

            moved1, moved2 = self._record_forward_motion(dx1, dx2)
            if moved1 and moved2:
                return

        raise RuntimeError("press direction could not be inferred from controller motion")

    def publish_press_targets(self, press_x1, y1, z1, press_x2, y2, z2,
                              repeats: Optional[int] = None,
                              dt:      Optional[float] = None) -> None:
        # The controller interprets frame_id="offset" as a delta from the
        # current trajectory target. Positive press depths are mapped onto the
        # controller's observed MOVE_FORWARD direction so hardware and Gazebo
        # can use different start/end ordering without a command-line override.
        if (
            (abs(press_x1) > 1e-9 and self.press_cmd_sign1 is None)
            or (abs(press_x2) > 1e-9 and self.press_cmd_sign2 is None)
        ):
            self._ensure_press_direction()
        cmd_x1 = (self.press_cmd_sign1 if self.press_cmd_sign1 is not None else -1.0) * press_x1
        cmd_x2 = (self.press_cmd_sign2 if self.press_cmd_sign2 is not None else -1.0) * press_x2
        kwargs = {}
        if repeats is not None:
            kwargs["repeats"] = repeats
        if dt is not None:
            kwargs["dt"] = dt
        self.publish_offsets(cmd_x1, y1, z1, cmd_x2, y2, z2, **kwargs)
        self.current_press_x1 = press_x1
        self.current_press_x2 = press_x2

    def _set_last_alignment(self, source: str, status: str, message: str) -> None:
        self.last_alignment_source = source
        self.last_alignment_status = status
        self.last_alignment_message = message

    def _prompt_manual_alignment(self, reason: str) -> None:
        self.get_logger().warn(
            f"\n{'='*60}\n"
            "  ACTION REQUIRED: camera alignment check indicates a mismatch.\n"
            f"  {reason}\n"
            "  Realign manually, then press ENTER to re-check.\n"
            f"{'='*60}"
        )
        input("  >> Realign manually, then press ENTER to re-check: ")
        self.spin_for(1.0)

    def _evaluate_alignment(self) -> Tuple[bool, str, str, Optional[Tuple[float, float]]]:
        if self.alignment_source in {"auto", "aruco"} and self.aruco_alignment_valid:
            trim1 = self.aruco_robot1_z_trim
            trim2 = self.aruco_robot2_z_trim
            if self.fin(trim1) and self.fin(trim2):
                max_trim = max(abs(trim1), abs(trim2))
                roll_desc = (
                    f", roll={self.aruco_alignment_roll_deg:+.2f} deg"
                    if self.fin(self.aruco_alignment_roll_deg)
                    else ""
                )
                if self.alignment_policy == "manual":
                    if max_trim <= 5e-4:
                        return True, "aruco", f"ArUco alignment already within tolerance{roll_desc}", None
                    return False, "aruco", (
                        "manual realignment requested; recommended z trims "
                        f"r1={trim1:+.4f} m r2={trim2:+.4f} m{roll_desc}"
                    ), None
                if max_trim <= self.alignment_max_auto_z_trim:
                    return True, "aruco", (
                        f"ArUco alignment trim accepted: r1={trim1:+.4f} m "
                        f"r2={trim2:+.4f} m{roll_desc}"
                    ), (trim1, trim2)
                return False, "aruco", (
                    "ArUco recommends larger z trims than allowed: "
                    f"r1={trim1:+.4f} m r2={trim2:+.4f} m > limit "
                    f"{self.alignment_max_auto_z_trim:.4f} m{roll_desc}"
                ), None

        if self.alignment_source in {"auto", "fallback"}:
            fallback = evaluate_fallback_alignment(
                lateral_deflection_m=self.w_latest,
                yaw_deg=self.yaw_latest,
                psi_left_rad=self.psi_L_latest,
                psi_right_rad=self.psi_R_latest,
                w_threshold_m=self.alignment_fallback_w_tol,
                yaw_threshold_deg=self.alignment_fallback_yaw_tol_deg,
                psi_diff_threshold_deg=self.alignment_fallback_psi_diff_tol_deg,
            )
            if fallback.valid:
                return fallback.alignment_ok, "fallback", fallback.message, None

        return True, "unavailable", "alignment signals unavailable — continuing without camera correction", None

    def ensure_camera_alignment(self, output_dir: str, label: str) -> Tuple[bool, str]:
        if self.alignment_source == "off" or self.alignment_policy == "off":
            self.alignment_robot1_z_trim = 0.0
            self.alignment_robot2_z_trim = 0.0
            self._set_last_alignment("off", "skipped", "camera alignment check disabled")
            return True, "alignment skipped"

        self.spin_for(0.5)
        self.capture_depth_camera_snapshot(output_dir, f"{label}_alignment_precheck")

        attempts = 0
        while True:
            ok, source, message, trims = self._evaluate_alignment()
            if ok:
                status = "ok"
                if trims is not None and self.alignment_policy == "auto_then_manual":
                    self.alignment_robot1_z_trim = trims[0]
                    self.alignment_robot2_z_trim = trims[1]
                    status = "auto_applied"
                elif source != "aruco":
                    self.alignment_robot1_z_trim = 0.0
                    self.alignment_robot2_z_trim = 0.0
                self._set_last_alignment(source, status, message)
                self._alog(f"Alignment [{source}] {message}")
                return True, f"alignment: {message}"

            self._set_last_alignment(source, "needs_manual", message)
            if self.alignment_policy == "warn":
                self._alog(f"Alignment warning [{source}] {message}")
                return True, f"alignment warning: {message}"
            if self.alignment_policy in {"auto_then_manual", "manual"} and attempts < 2:
                self._alog(f"Alignment [{source}] {message}")
                self._prompt_manual_alignment(message)
                attempts += 1
                self.capture_depth_camera_snapshot(output_dir, f"{label}_alignment_retry{attempts}")
                continue
            self._alog(f"Alignment failed [{source}] {message}")
            return False, f"alignment: {message}"

    def safe_abort(self) -> None:
        self.current_phase = "abort"
        if self._active_depth_frame_output_dir and self._active_depth_frame_label_prefix:
            self.capture_depth_camera_snapshot(
                self._active_depth_frame_output_dir,
                f"{self._active_depth_frame_label_prefix}_abort",
            )
        self.get_logger().error("ABORT: zeroing offsets")
        self.publish_press_targets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=3, dt=0.05)
        self._disable_torque_on_abort()
        self.spin_for(0.5)

    def _disable_torque_on_abort(self) -> None:
        if not self.abort_disable_torque or self._torque_disable_requested:
            return

        self._torque_disable_requested = True

        if not os.path.exists(self.disable_torque_script):
            self.get_logger().error(
                f"Abort torque-disable requested but helper script is missing: {self.disable_torque_script}"
            )
            return

        ports = [("robot1", self.robot1_port), ("robot2", self.robot2_port)]
        missing = [name for name, port in ports if not port]
        if missing:
            self.get_logger().error(
                "Abort torque-disable requested but serial ports were not configured for: "
                + ", ".join(missing)
            )
            return

        for name, port in ports:
            cmd = [
                sys.executable,
                self.disable_torque_script,
                "--port",
                str(port),
                "--baud",
                str(self.torque_disable_baud),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
            except Exception as exc:
                self.get_logger().error(
                    f"Abort torque-disable failed for {name} on {port}: {exc}"
                )
                continue

            if result.returncode != 0:
                stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
                self.get_logger().error(
                    f"Abort torque-disable failed for {name} on {port}: {stderr}"
                )
            else:
                self.get_logger().warn(
                    f"Abort torque-disable succeeded for {name} on {port}"
                )

    def max_vel(self, js: Optional[JointState]) -> Optional[float]:
        if js is None or not js.velocity:
            return None
        return max(abs(v) for v in js.velocity)

    def max_ee_speed(self, ee_vel: Optional[Tuple[float, float, float]]) -> Optional[float]:
        if ee_vel is None:
            return None
        return math.sqrt(sum(v * v for v in ee_vel))

    def ee_speed_limit(self, joint_limit: float) -> float:
        if joint_limit <= HOLD_VEL_LIMIT:
            return HOLD_EE_SPEED_LIMIT_MPS
        if joint_limit <= IDLE_VEL_LIMIT:
            return IDLE_EE_SPEED_LIMIT_MPS
        return MOVE_EE_SPEED_LIMIT_MPS

    def arm_vel_ok(
        self,
        js: Optional[JointState],
        ee_vel: Optional[Tuple[float, float, float]],
        limit: float,
    ) -> bool:
        v = self.max_vel(js)
        if v is not None:
            return v < limit

        ee_v = self.max_ee_speed(ee_vel)
        if ee_v is None:
            return False
        return ee_v < self.ee_speed_limit(limit)

    def arm_motion_sample_available(
        self,
        js: Optional[JointState],
        ee_vel: Optional[Tuple[float, float, float]],
    ) -> bool:
        return self.max_vel(js) is not None or self.max_ee_speed(ee_vel) is not None

    def vel_ok(self, limit: float) -> bool:
        return (
            self.arm_vel_ok(self.js1, self._last_ee1_vel, limit)
            and self.arm_vel_ok(self.js2, self._last_ee2_vel, limit)
        )

    def contact_threshold(self, robot_id: int) -> float:
        base = self.baseline_fx1 if robot_id == 1 else self.baseline_fx2
        if not self.fin(base):
            base = 0.0
        return min(CONTACT_FORCE_CAP,
                   max(CONTACT_FORCE_ENTER, base + CONTACT_FORCE_DELTA))

    def contact_detected(self, robot_id: int) -> bool:
        if robot_id == 1:
            return (self.contact_valid_1 and self.fin(self.contact_fx1)
                    and self.contact_fx1 >= self.contact_threshold(1))
        return (self.contact_valid_2 and self.fin(self.contact_fx2)
                and self.contact_fx2 >= self.contact_threshold(2))

    def _raw_bilateral_contact_detected(self) -> bool:
        return (
            self.contact_valid_1
            and self.contact_valid_2
            and self.fin(self.contact_fx1)
            and self.fin(self.contact_fx2)
            and self.contact_fx1 >= self.contact_threshold(1)
            and self.contact_fx2 >= self.contact_threshold(2)
        )

    # ── snapshot (per-row log columns) ────────────────────────────────────────
    LOG_COLS = [
        "t_run_s", "phase", "K_lat_Npm", "case", "rep",
        "offset_x1", "offset_x2", "contact_mode",
        "ee_x_1", "ee_x_2",
        "contact_fx_1", "contact_fx_2", "contact_fx_diff",
        "P_s_est", "P_s_max",
        "baseline_fx1", "baseline_fx2",
        "contact_valid_1", "contact_valid_2",
        "mb", "mc", "w_m", "yaw_deg",
        "spring_length_m", "contact_rotation_l_rad", "contact_rotation_r_rad",
        "t_mb_zero_s", "t_mc_zero_s", "t_w_3mm_s", "t_phi_4deg_s",
        "robot1_max_vel", "robot2_max_vel",
    ] + harness_camera_aruco_log_columns()

    def _snap(self, case: str, K_lat: float, rep: int) -> Dict[str, Any]:
        t_run = self.now_s() - self._run_start_t
        self._update_contact_duration_state()
        ps    = self._ps()
        if self.fin(ps):
            self._p_s_max = max(self._p_s_max, ps)
        row = {
            "t_run_s":        t_run,
            "phase":          self.current_phase,
            "K_lat_Npm":      K_lat,
            "case":           case,
            "rep":            rep,
            "offset_x1":      self.current_offset_x1,
            "offset_x2":      self.current_offset_x2,
            "contact_mode":   self.current_contact_mode,
            "ee_x_1":         self.ee1.x if self.ee1 else float("nan"),
            "ee_x_2":         self.ee2.x if self.ee2 else float("nan"),
            "contact_fx_1":   self.contact_fx1,
            "contact_fx_2":   self.contact_fx2,
            "contact_fx_diff": (self.contact_fx1 - self.contact_fx2
                                if self.fin(self.contact_fx1) and self.fin(self.contact_fx2)
                                else float("nan")),
            "P_s_est":         ps,
            "P_s_max":         self._p_s_max,
            "baseline_fx1":   self.baseline_fx1,
            "baseline_fx2":   self.baseline_fx2,
            "contact_valid_1": float(self.contact_valid_1),
            "contact_valid_2": float(self.contact_valid_2),
            "mb":              self.mb_latest,
            "mc":              self.mc_latest,
            "w_m":             self.w_latest,
            "yaw_deg":         self.yaw_latest,
            "spring_length_m": self.spring_length_latest,
            "contact_rotation_l_rad": self.psi_L_latest,
            "contact_rotation_r_rad": self.psi_R_latest,
            "t_mb_zero_s":    self._t_mb_zero,
            "t_mc_zero_s":    self._t_mc_zero,
            "t_w_3mm_s":      self._t_w_3mm,
            "t_phi_4deg_s":   self._t_phi_4deg,
            "robot1_max_vel":  self.max_vel(self.js1),
            "robot2_max_vel":  self.max_vel(self.js2),
        }
        row.update(harness_camera_aruco_log_row(self.last_camera_aruco_summary))
        return row

    # ── 50 Hz log tick ────────────────────────────────────────────────────────
    def _log_tick(self) -> None:
        if not self._logging_active or self._run_csv_writer is None:
            return
        row = self._snap(self._current_log_case, self._current_log_klat, self._current_log_rep)
        self._run_csv_writer.writerow(row)

    def _start_run_log(self, csv_path: str, case: str, K_lat: float, rep: int) -> None:
        self._current_log_case = case
        self._current_log_klat = K_lat
        self._current_log_rep  = rep
        self._run_csv_file = open(csv_path, "w", newline="")
        self._run_csv_writer = csv.DictWriter(
            self._run_csv_file, fieldnames=self.LOG_COLS, extrasaction="ignore")
        self._run_csv_writer.writeheader()
        self._logging_active = True

    def _stop_run_log(self) -> None:
        self._logging_active = False
        if self._run_csv_file:
            self._run_csv_file.flush()
            self._run_csv_file.close()
            self._run_csv_file = None
            self._run_csv_writer = None

    # ── onset tracking reset ──────────────────────────────────────────────────
    def _reset_onset(self) -> None:
        self._run_start_t  = self.now_s()
        self._t_contact    = float("nan")
        self._contact_duration_s_total = 0.0
        self._contact_active = False
        self._contact_active_since_s = float("nan")
        self._contact_wall_time_ns = None
        self._contact_color_image_stamp_ns = None
        self._contact_depth_image_stamp_ns = None
        self._t_mb_zero    = float("nan")
        self._t_mc_zero    = float("nan")
        self._t_w_3mm      = float("nan")
        self._t_phi_4deg   = float("nan")
        self._p_s_onset    = float("nan")
        self._p_s_max      = 0.0
        self._mb_neg_count = 0
        self._mc_neg_count = 0
        self._mb_min       = float("nan")   # running minimum m_b
        self._mc_min       = float("nan")   # running minimum m_c
        self._mb_at_contact = float("nan")  # m_b snapshot at bilateral contact
        self._mc_at_contact = float("nan")  # m_c snapshot at bilateral contact

    # ── K_lat change ──────────────────────────────────────────────────────────
    def set_k_lateral(self, k_lat: float) -> None:
        if self.skip_klat_command:
            self.get_logger().warn(
                f"Skipping K_lat command for {k_lat:.1f} N/m; assuming Gazebo/controller config is already fixed"
            )
            return

        if k_lat > CARTESIAN_STIFFNESS_LIMIT_NPM:
            raise ValueError(
                f"Requested K_lat={k_lat:.1f} N/m exceeds configured limit "
                f"{CARTESIAN_STIFFNESS_LIMIT_NPM:.1f} N/m"
            )

        if SET_KLAT_METHOD == "manual":
            self.get_logger().warn(
                f"\n{'='*60}\n"
                f"  ACTION REQUIRED: set K_lat = {k_lat:.1f} N/m on BOTH arms.\n"
                f"  Press ENTER when done.\n"
                f"{'='*60}"
            )
            input(f"  >> Set K_lat = {k_lat:.1f} N/m, then press ENTER: ")
            self.spin_for(1.0)

        elif SET_KLAT_METHOD == "topic":
            msg = Float64()
            msg.data = float(k_lat)
            for _ in range(5):
                self.klat_pub1.publish(msg)
                self.klat_pub2.publish(msg)
                self.spin_for(0.1)
            self.get_logger().info(f"K_lat topic set to {k_lat:.1f} N/m")
            self.spin_for(0.5)

        elif SET_KLAT_METHOD == "ros2_param":
            # Requires rclpy to be able to call set_parameters on the
            # controller node.  Works only if harness and controller share
            # the same process or the controller exposes a parameter server.
            import subprocess
            for node in [CTRL_NODE_R1, CTRL_NODE_R2]:
                cmd = [
                    "ros2", "param", "set", node, "k_lateral", str(k_lat)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
                if result.returncode != 0:
                    self.get_logger().warn(
                        f"ros2 param set failed for {node}: {result.stderr.strip()}")
            self.spin_for(0.5)

        else:
            raise ValueError(f"Unknown SET_KLAT_METHOD: {SET_KLAT_METHOD!r}")

    # ── ablation log helper ───────────────────────────────────────────────────
    def _alog(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._ablation_log.append(line)
        self.get_logger().info(msg)

    def _flush_alog(self) -> None:
        with open(self._ablation_log_path, "w") as f:
            f.write("\n".join(self._ablation_log))

    # ══════════════════════════════════════════════════════════════════════════
    # 5) STAGE IMPLEMENTATIONS  (mirrors harness_v2 structure)
    # ══════════════════════════════════════════════════════════════════════════

    def _stage_liveness(self) -> Tuple[bool, str]:
        """Stage 1: verify controller telemetry is flowing."""
        t0 = self.now_s()
        while self.now_s() - t0 < 5.0:
            if self.pub1.get_subscription_count() > 0 and \
               self.pub2.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        while self.now_s() - t0 < 8.0:
            if self.js1 is not None and self.js2 is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        while self.now_s() - t0 < 10.0:
            if (self.ee1 is not None and self.ee2 is not None and
                    self.des1 is not None and self.des2 is not None):
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.pub1.get_subscription_count() == 0 or self.pub2.get_subscription_count() == 0:
            self.get_logger().warn(
                "waypoint_command subscriber discovery is incomplete during liveness; "
                "continuing because controller telemetry is present"
            )
        if self.js1 is None or self.js2 is None:
            self.get_logger().warn(
                "joint_states are unavailable during liveness; using EE-speed fallback for velocity checks"
            )

        ok = (self.ee1 is not None and self.ee2 is not None and
              self.des1 is not None and self.des2 is not None)
        if ok:
            self._initialize_motion_safety_limits()
        return ok, "liveness" + (" ok" if ok else " FAILED")

    def _stage_idle_check(self) -> Tuple[bool, str]:
        """Stage 2: verify both arms are idle (low joint velocity)."""
        self.current_phase = "idle"
        self.idle_start_x1 = None
        self.idle_start_x2 = None
        deadline = self.now_s() + IDLE_WAIT_TIMEOUT_S
        last_violation = None
        arm1_ready = False
        arm2_ready = False
        arm1_ready_x: Optional[float] = None
        arm2_ready_x: Optional[float] = None
        observed_min_x1 = float("inf")
        observed_min_x2 = float("inf")
        observed_max_x1 = float("-inf")
        observed_max_x2 = float("-inf")
        last_motion_sign1: Optional[float] = None
        last_motion_sign2: Optional[float] = None

        while self.now_s() < deadline:
            if self.des1 is None or self.des2 is None:
                self.spin_for(0.1)
                continue

            x0_1 = self.des1.position.x
            x0_2 = self.des2.position.x
            self.spin_for(IDLE_STABLE_SAMPLE_S)

            if self.des1 is None or self.des2 is None:
                continue

            separation = self.ee_separation() if self._ee_separation_check_enabled() else None
            observed_min_x1 = min(observed_min_x1, x0_1, self.des1.position.x)
            observed_min_x2 = min(observed_min_x2, x0_2, self.des2.position.x)
            observed_max_x1 = max(observed_max_x1, x0_1, self.des1.position.x)
            observed_max_x2 = max(observed_max_x2, x0_2, self.des2.position.x)
            dx1 = self.des1.position.x - x0_1
            dx2 = self.des2.position.x - x0_2
            motion_sign1 = self._motion_sign(dx1)
            motion_sign2 = self._motion_sign(dx2)
            if motion_sign1 is not None:
                last_motion_sign1 = motion_sign1
            if motion_sign2 is not None:
                last_motion_sign2 = motion_sign2

            arm1_desired_stable = abs(dx1) <= IDLE_DESIRED_DRIFT_TOL_M
            arm2_desired_stable = abs(dx2) <= IDLE_DESIRED_DRIFT_TOL_M
            if not arm1_desired_stable and not arm2_desired_stable:
                continue

            arm1_near_max = self.des1.position.x >= (observed_max_x1 - IDLE_START_X_TOL_M)
            arm1_near_min = self.des1.position.x <= (observed_min_x1 + IDLE_START_X_TOL_M)
            arm2_near_max = self.des2.position.x >= (observed_max_x2 - IDLE_START_X_TOL_M)
            arm2_near_min = self.des2.position.x <= (observed_min_x2 + IDLE_START_X_TOL_M)

            arm1_start_window = (
                arm1_desired_stable
                and
                (observed_max_x1 - observed_min_x1) >= IDLE_START_WINDOW_TRAVEL_M
                and (
                    (arm1_near_max and last_motion_sign1 == 1.0)
                    or (arm1_near_min and last_motion_sign1 == -1.0)
                )
            )
            arm2_start_window = (
                arm2_desired_stable
                and
                (observed_max_x2 - observed_min_x2) >= IDLE_START_WINDOW_TRAVEL_M
                and (
                    (arm2_near_max and last_motion_sign2 == 1.0)
                    or (arm2_near_min and last_motion_sign2 == -1.0)
                )
            )
            if not arm1_start_window and not arm2_start_window:
                continue

            arm1_quiet = self.arm_vel_ok(self.js1, self._last_ee1_vel, IDLE_VEL_LIMIT) or not self.arm_motion_sample_available(
                self.js1, self._last_ee1_vel
            )
            arm2_quiet = self.arm_vel_ok(self.js2, self._last_ee2_vel, IDLE_VEL_LIMIT) or not self.arm_motion_sample_available(
                self.js2, self._last_ee2_vel
            )
            if arm1_start_window and arm1_quiet:
                if not arm1_ready:
                    arm1_ready_x = self.des1.position.x
                    if last_motion_sign1 is not None:
                        self.press_cmd_sign1 = -last_motion_sign1
                arm1_ready = True
            if arm2_start_window and arm2_quiet:
                if not arm2_ready:
                    arm2_ready_x = self.des2.position.x
                    if last_motion_sign2 is not None:
                        self.press_cmd_sign2 = -last_motion_sign2
                arm2_ready = True
            if not arm1_ready or not arm2_ready:
                continue

            if separation is not None and separation < self.min_ee_separation:
                separation_shortfall = self.min_ee_separation - separation
                if separation_shortfall > MIN_EE_SEPARATION_REBASE_MAX_M:
                    last_violation = (
                        f"EE separation {separation:.4f} m below safety limit "
                        f"{self.min_ee_separation:.4f} m"
                    )
                    self.get_logger().error(last_violation)
                    continue
                self.min_ee_separation = max(
                    0.0, separation - MIN_EE_SEPARATION_REBASE_TOL_M
                )

            current_x_gap_limit = self.ee_x_gap() + self.max_ee_x_gap_margin
            if self.max_ee_x_gap is None or current_x_gap_limit > self.max_ee_x_gap:
                self.max_ee_x_gap = current_x_gap_limit

            last_violation = self.motion_safety_violation()
            if last_violation is not None:
                continue

            self.idle_start_x1 = arm1_ready_x if arm1_ready_x is not None else self.des1.position.x
            self.idle_start_x2 = arm2_ready_x if arm2_ready_x is not None else self.des2.position.x

            return True, "idle check ok"

        if last_violation is not None:
            return False, f"idle: {last_violation}"
        return False, "idle check FAILED — arms never settled into a safe idle window"

    def _stage_approach(self) -> Tuple[bool, str]:
        """Stage 3: advance both TCPs toward spring until bilateral contact."""
        self.current_phase = "approach"

        # Allow a small phase skew between controllers entering MOVE_FORWARD and
        # infer the active press direction from the desired-x motion itself.
        t0 = self.now_s()
        saw_forward_1 = False
        saw_forward_2 = False
        while self.now_s() - t0 < FORWARD_PHASE_TIMEOUT_S:
            if self.des1 is None or self.des2 is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue
            x0_1 = self.des1.position.x
            x0_2 = self.des2.position.x
            self.spin_for(0.30)
            if self.des1 and self.des2:
                moved1, moved2 = self._record_forward_motion(
                    self.des1.position.x - x0_1,
                    self.des2.position.x - x0_2,
                )
                saw_forward_1 = saw_forward_1 or moved1
                saw_forward_2 = saw_forward_2 or moved2
                if saw_forward_1 and saw_forward_2:
                    break
        else:
            return False, "approach: controllers did not enter forward phase"

        self.spin_for(0.10)
        # capture idle baseline for delta-force contact detection
        self.baseline_fx1 = self.contact_fx1 if self.fin(self.contact_fx1) else 0.0
        self.baseline_fx2 = self.contact_fx2 if self.fin(self.contact_fx2) else 0.0

        # record t=0 for onset timing (contact not yet established)
        self._reset_onset()

        off1 = PRECONTACT_START_X
        off2 = PRECONTACT_START_X
        c1 = c2 = False
        for _ in range(MAX_PRECONTACT_ITER):
            if not c1:
                off1 = min(off1 + PRESS_STEP, PRECONTACT_END_X)
            if not c2:
                off2 = min(off2 + PRESS_STEP, PRECONTACT_END_X)

            # Positive press depths are mapped onto each controller's inferred
            # MOVE_FORWARD direction.
            self.current_contact_mode = (
                "both" if (c1 and c2) else
                "r1_only" if c1 else "r2_only" if c2 else "none")
            self.publish_press_targets(off1, 0.0, 0.0,
                                       off2, 0.0, 0.0,
                                       repeats=1)
            self.spin_for(QUASISTATIC_SAMPLE_DELAY)
            self._update_contact_duration_state()

            violation = self.motion_safety_violation()
            if violation is not None:
                return False, f"approach: {violation}"

            # The two controllers are intentionally phase-skewed, so contact can
            # cross threshold on different settled samples. Latch per-arm contact
            # once seen so the second arm can catch up without requiring an
            # artificially simultaneous threshold crossing.
            c1 = c1 or self.contact_detected(1)
            c2 = c2 or self.contact_detected(2)
            if c1 and c2:
                self._t_contact     = self.now_s() - self._run_start_t
                self._mb_at_contact = self.mb_latest
                self._mc_at_contact = self.mc_latest
                self._mark_contact_sync_event()
                return True, "bilateral contact established"

            if not self.vel_ok(MOVE_VEL_LIMIT):
                # The first post-command sample can contain a one-cycle onset
                # transient even when the next settled sample is back under the
                # motion limit, so only abort if the re-sample is also high.
                self.spin_for(COMMAND_DT)
                if not self.vel_ok(MOVE_VEL_LIMIT):
                    return False, "approach: velocity spike"

        return False, "approach: bilateral contact not established within iteration limit"

    def _stage_hold(self) -> Tuple[bool, str]:
        """Stage 4: hold at contact and wait for settlement."""
        self.current_phase = "hold"
        self.publish_press_targets(self.current_press_x1, 0.0, 0.0,
                                   self.current_press_x2, 0.0, 0.0)
        self.spin_for(0.8)
        self._update_contact_duration_state()
        violation = self.motion_safety_violation()
        if violation is not None:
            return False, f"hold: {violation}"
        ok = self.vel_ok(HOLD_VEL_LIMIT)
        return ok, "hold stable" if ok else "hold: still moving after contact"

    def _stage_ramp(self) -> Tuple[bool, str]:
        """Stage 5: slow quasi-static ramp to press_end; detect onsets."""
        self.current_phase = "ramp"
        off1 = self.current_press_x1
        off2 = self.current_press_x2
        t_ramp_start = self.now_s()

        while max(off1, off2) < PRESS_END_X:
            # check ramp timeout
            if self.now_s() - t_ramp_start > PRESS_RAMP_TIMEOUT_S:
                return False, "ramp timeout"

            off1 = min(off1 + RAMP_STEP, PRESS_END_X)
            off2 = min(off2 + RAMP_STEP, PRESS_END_X)

            # force-balance correction (mirrors v2 adaptive press)
            c1 = self.contact_detected(1)
            c2 = self.contact_detected(2)
            if c1 and c2 and self.fin(self.contact_fx1) and self.fin(self.contact_fx2):
                diff = self.contact_fx1 - self.contact_fx2
                if diff < -FORCE_BALANCE_TOL:
                    off1 = min(off1 + RAMP_SIDE_PRESS_STEP, PRESS_END_X + MAX_SIDE_EXTRA_PRESS)
                    self.current_contact_mode = "push_r1_more"
                elif diff > FORCE_BALANCE_TOL:
                    off2 = min(off2 + RAMP_SIDE_PRESS_STEP, PRESS_END_X + MAX_SIDE_EXTRA_PRESS)
                    self.current_contact_mode = "push_r2_more"
                else:
                    self.current_contact_mode = "balanced"
            else:
                self.current_contact_mode = "seeking"

            self.publish_press_targets(off1, 0.0, 0.0,
                                       off2, 0.0, 0.0)
            self.spin_for(RAMP_SAMPLE_DELAY)
            self._update_contact_duration_state()

            self.current_press_x1 = off1
            self.current_press_x2 = off2

            violation = self.motion_safety_violation()
            if violation is not None:
                return False, f"ramp: {violation}"

            # hard abort conditions
            if not self.vel_ok(MOVE_VEL_LIMIT):
                return False, "ramp: velocity spike"
            if (self.fin(self.contact_fx1) and self.fin(self.contact_fx2)):
                if abs(self.contact_fx1 - self.contact_fx2) > FORCE_DIFF_ABORT:
                    return False, "ramp: force asymmetry abort"
                if max(self.contact_fx1, self.contact_fx2) > MAX_FORCE_MAG_ABORT:
                    return False, "ramp: force magnitude abort"

        return True, "ramp complete"

    def _stage_withdraw(self) -> None:
        """Return TCPs to nominal after each run."""
        self.current_phase = "withdraw"
        self.publish_press_targets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=5, dt=0.05)
        self.spin_for(1.5)
        self._update_contact_duration_state()

    # ══════════════════════════════════════════════════════════════════════════
    # 6) K_lat IDENTIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    def identify_klat(self, n_pts: int = 5, amp_m: float = 0.003,
                      settle_s: float = 0.4) -> Tuple[float, float]:
        """Estimate actual closed-loop K_lat via a small lateral perturbation.

        Moves TCP1 (and TCP2 symmetrically) by +/-amp_m in X relative to the
        current idle pose, records Fx contact force at each displaced position,
        and returns the slope Delta_Fx / Delta_xTCP from a linear regression.

        This gives K_lat^meas = Delta_Fx / Delta_xTCP [N/m], the actual
        closed-loop Cartesian lateral stiffness at the compression posture,
        which may differ from the commanded KP_CART due to joint-torque
        saturation, friction, and impedance tuning.

        Returns
        -------
        (slope_N_per_m, stderr_N_per_m)
            slope is K_lat^meas;  stderr is the standard error of the fit.
        NaN, NaN is returned if contact is not valid or motion fails.
        """
        # Operators can spend tens of seconds at the manual K_lat prompt while
        # the background controller keeps cycling. Re-gate on the same safe
        # idle window used by the main run path before perturbing the TCP.
        ok, msg = self._stage_liveness()
        if not ok:
            self.get_logger().warn(f"identify_klat: {msg}")
            return float("nan"), float("nan")

        ok, msg = self._stage_idle_check()
        if not ok:
            self.get_logger().warn(f"identify_klat: {msg}")
            return float("nan"), float("nan")

        if not (self.contact_valid_1 and self.contact_valid_2):
            self.get_logger().warn("identify_klat: no valid contact — skipping.")
            return float("nan"), float("nan")

        import numpy as np
        displacements = np.linspace(-amp_m, amp_m, n_pts)
        xs, fxs = [], []

        # Cache nominal commanded pose for TCP1
        if self.des1 is None:
            self.get_logger().warn("identify_klat: no desired pose for arm1 — skipping.")
            return float("nan"), float("nan")

        nominal_x1 = self.des1.position.x

        for dx in displacements:
            # Keep calibration perturbations on the same offset waypoint path as
            # the rest of the harness so the controller blends around the live
            # trajectory target instead of interpreting them as absolute poses.
            cmd1 = self._make_pose(dx, 0.0, 0.0)
            self.pub1.publish(cmd1)
            self.spin_for(settle_s)   # allow TCP to settle
            self._raise_on_motion_safety_violation("identify_klat")
            if not self.contact_valid_1:
                continue
            xs.append(dx)
            fxs.append(self.contact_fx1)

        # Restore nominal pose
        cmd_restore = self._make_pose(0.0, 0.0, 0.0)
        self.pub1.publish(cmd_restore)
        self.spin_for(settle_s)
        self._raise_on_motion_safety_violation("identify_klat restore")

        if len(xs) < 3:
            self.get_logger().warn("identify_klat: too few valid points.")
            return float("nan"), float("nan")

        xs  = np.array(xs)
        fxs = np.array(fxs)
        # Linear fit: Fx = K_lat * dx  (zero-intercept model)
        slope = float(np.dot(xs, fxs) / np.dot(xs, xs))
        resid = fxs - slope * xs
        stderr = float(np.std(resid) / np.sqrt(np.dot(xs, xs))) if len(xs) > 2 else float("nan")
        self.get_logger().info(
            f"identify_klat: K_lat^meas = {slope:.1f} N/m  stderr = {stderr:.1f} N/m "
            f"(commanded = {self.current_press_x1:.0f} N/m equiv)")
        return slope, stderr

    # ══════════════════════════════════════════════════════════════════════════
    # 7) SINGLE RUN
    # ══════════════════════════════════════════════════════════════════════════

    def _run_one(self, case: str, K_lat: float, rep: int,
                 P_c_th: float, run_dir: str) -> RunResult:
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_k = f"{K_lat:.0f}".replace(".", "p")
        run_label_prefix = f"run_{case}_K{safe_k}_rep{rep:02d}"
        csv_path  = os.path.join(run_dir, f"run_{case}_K{safe_k}_rep{rep:02d}_{ts}.csv")
        json_path = os.path.join(run_dir, f"run_{case}_K{safe_k}_rep{rep:02d}_{ts}.json")

        result = RunResult(case=case, K_lat_Npm=K_lat, rep=rep, ts=ts,
                           passed=False, P_c_th_N=P_c_th,
                           csv_path=csv_path, json_path=json_path)
        result.ablation_index = int(self._current_ablation_index)
        result.alignment_source = self.last_alignment_source
        result.alignment_status = self.last_alignment_status
        result.alignment_message = self.last_alignment_message
        result.robot1_z_trim_m = self.alignment_robot1_z_trim
        result.robot2_z_trim_m = self.alignment_robot2_z_trim

        self._start_run_log(csv_path, case, K_lat, rep)
        self._reset_onset()
        self._set_active_depth_frame_context(run_dir, run_label_prefix)
        self._start_external_capture_session(
            phase="ablation_run",
            name=run_label_prefix,
            ablation_index=result.ablation_index,
            case=case,
            rep=rep,
            k_lat_npm=K_lat,
        )
        self.capture_depth_camera_snapshot(run_dir, f"{run_label_prefix}_start")

        stages = [
            ("liveness",   self._stage_liveness),
            ("idle_check", self._stage_idle_check),
            ("approach",   self._stage_approach),
            ("hold",       self._stage_hold),
            ("ramp",       self._stage_ramp),
        ]

        passed_all = True
        capture_status = "failed"
        capture_meta: Dict[str, Any] = {}
        try:
            for name, fn in stages:
                ok, msg = fn()
                self.capture_depth_camera_snapshot(
                    run_dir,
                    f"{run_label_prefix}_{name}_{'ok' if ok else 'fail'}",
                )
                self._alog(f"    [{case} K={K_lat:.0f} rep{rep}] {name}: {msg}")
                if not ok:
                    self.safe_abort()
                    result.abort_reason = msg
                    passed_all = False
                    break

            if passed_all:
                self._stage_withdraw()
                self.capture_depth_camera_snapshot(
                    run_dir,
                    f"{run_label_prefix}_withdraw_ok",
                )
                result.passed = True
                capture_status = "completed"
        except Exception:
            capture_status = "exception"
            self.capture_depth_camera_snapshot(run_dir, f"{run_label_prefix}_exception")
            raise
        finally:
            self._update_contact_duration_state()
            capture_meta = self._stop_external_capture_session(status=capture_status)
            self._stop_run_log()
            self._clear_active_depth_frame_context()

        # harvest onset times
        result.t_contact_s     = self._t_contact
        result.contact_duration_s = self._final_contact_duration_s()
        result.t_mb_zero_s     = self._t_mb_zero
        result.t_mc_zero_s     = self._t_mc_zero
        result.t_w_3mm_s       = self._t_w_3mm
        result.t_phi_4deg_s    = self._t_phi_4deg
        result.P_s_at_onset_N  = self._p_s_onset
        result.P_s_max_N       = self._p_s_max
        result.mb_at_contact_N = self._mb_at_contact
        result.mb_min_N        = self._mb_min
        result.mc_at_contact_N = self._mc_at_contact
        result.mc_min_N        = self._mc_min
        camera_aruco_meta = self._camera_aruco_capture_metadata()
        result.camera_aruco_records = camera_aruco_meta["camera_aruco_records"]
        result.camera_aruco_valid_records = camera_aruco_meta["camera_aruco_valid_records"]
        result.camera_aruco_all_markers_records = camera_aruco_meta["camera_aruco_all_markers_records"]
        result.camera_aruco_dropout_run_count = camera_aruco_meta["camera_aruco_dropout_run_count"]
        result.camera_aruco_angle_flip_count = camera_aruco_meta["camera_aruco_angle_flip_count"]
        result.camera_aruco_raw_jsonl_path = camera_aruco_meta["camera_aruco_raw_jsonl_path"]
        result.camera_aruco_summary_json_path = camera_aruco_meta["camera_aruco_summary_json_path"]
        result.camera_aruco_csv_dir_path = camera_aruco_meta["camera_aruco_csv_dir_path"]
        result.external_capture_metadata = capture_meta
        result.aruco_summary_row = harness_camera_aruco_log_row(self.last_camera_aruco_summary)

        # write per-run JSON
        with open(json_path, "w") as f:
            payload = {
            "case": case, "K_lat_Npm": K_lat, "rep": rep, "ts": ts,
            "ablation_index": result.ablation_index,
                "passed": result.passed, "abort_reason": result.abort_reason,
                "alignment_source": result.alignment_source,
                "alignment_status": result.alignment_status,
                "alignment_message": result.alignment_message,
                "robot1_z_trim_m": result.robot1_z_trim_m,
                "robot2_z_trim_m": result.robot2_z_trim_m,
                "t_contact_s":     result.t_contact_s,
            "contact_duration_s": result.contact_duration_s,
                "t_mb_zero_s":     result.t_mb_zero_s,
                "t_mc_zero_s":     result.t_mc_zero_s,
                "t_w_3mm_s":       result.t_w_3mm_s,
                "t_phi_4deg_s":    result.t_phi_4deg_s,
                "P_s_at_onset_N":  result.P_s_at_onset_N,
                "P_s_max_N":       result.P_s_max_N,
                "mb_at_contact_N": result.mb_at_contact_N,
                "mb_min_N":        result.mb_min_N,
                "mc_at_contact_N": result.mc_at_contact_N,
                "mc_min_N":        result.mc_min_N,
                "camera_aruco_records": result.camera_aruco_records,
                "camera_aruco_valid_records": result.camera_aruco_valid_records,
                "camera_aruco_all_markers_records": result.camera_aruco_all_markers_records,
                "camera_aruco_dropout_run_count": result.camera_aruco_dropout_run_count,
                "camera_aruco_angle_flip_count": result.camera_aruco_angle_flip_count,
                "camera_aruco_raw_jsonl_path": result.camera_aruco_raw_jsonl_path,
                "camera_aruco_summary_json_path": result.camera_aruco_summary_json_path,
                "camera_aruco_csv_dir_path": result.camera_aruco_csv_dir_path,
                "P_b_th_N":        P_B_THEORY_N,
                "P_c_th_N":        P_c_th,
                "external_capture": result.external_capture_metadata,
            }
            payload.update(result.aruco_summary_row)
            json.dump(payload, f, indent=2)

        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 8) SPRING CALIBRATION
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Estimates effective experimental constants from the rig.  Each method maps
    # to one quantity in the calibration protocol:
    #
    #   _calib_L0()                -> L0 from camera spring_length_m at zero press
    #   _calib_axial_stiffness()   -> k_a from P_s vs DeltaL  (pre-instability)
    #   _calib_buckling_threshold()-> P_b from change-point in w_max(P_s)
    #                                 B_eff = P_b * L0^2 / pi^2
    #   _psi_rad()                 -> contact rotation psi [rad], any source
    #   _calib_k_theta()           -> k_theta from M = P_s*e vs psi  (|psi|<2 deg)
    #   calibrate_spring()         -> full sequence, writes JSON
    #
    # CRITICAL: k_theta is estimated from the INDEPENDENT eccentricity test.
    # Do not use the K_lat ablation sweep to fit k_theta; that would be circular
    # because the ablation sweep validates P_c^2 = K_lat * k_theta.

    def _calib_wait_spring_length(self, n: int = 20, dt: float = 0.05
                                  ) -> "tuple[float, float]":
        """Collect n readings of spring_length_latest; return (mean, std)."""
        import numpy as np
        samples = []
        for _ in range(n):
            self.spin_for(dt)
            if not math.isnan(self.spring_length_latest):
                samples.append(self.spring_length_latest)
        if len(samples) < 3:
            return float("nan"), float("nan")
        return float(np.mean(samples)), float(np.std(samples))

    def _calib_L0(self) -> "tuple[float, float]":
        """Measure unloaded spring length L0 at zero compression (camera topic)."""
        self._alog("CALIB L0: sampling spring length at zero compression")
        mean_L, std_L = self._calib_wait_spring_length(n=CALIB_L0_SAMPLES)
        self._alog(f"  L0 = {mean_L*1000:.2f} +/- {std_L*1000:.2f} mm")
        return mean_L, std_L

    def _calib_axial_stiffness(self, L0: float) -> "tuple[float, float, list]":
        """
        Quasi-static symmetric compression at K_lat = CALIB_K_LAT_NPM.
        Fits P_s = k_a * DeltaL + b over the pre-instability region:
          - stops at P_s > CALIB_FORCE_TARGET_N  (well below P_b)
          - stops at w_max > CALIB_W_EXCLUDE_M   (lateral bending onset)
        Camera spring_length_m is required for DeltaL; if unavailable the fit
        returns NaN but the force data is still logged.

        Returns (k_a [N/m], SE [N/m], data) where data = [(dL_m, P_s_N), ...].
        """
        import numpy as np
        self._alog("CALIB k_a: quasi-static P_s vs DeltaL compression")
        self.set_k_lateral(CALIB_K_LAT_NPM)
        self.spin_for(2.0)
        data = []
        off1 = off2 = 0.0

        while True:
            off1 = min(off1 + CALIB_STEP_M, PRESS_END_X + 0.005)
            off2 = min(off2 + CALIB_STEP_M, PRESS_END_X + 0.005)
            self.publish_press_targets(off1, 0.0, 0.0, off2, 0.0, 0.0)
            self.spin_for(CALIB_SAMPLE_DELAY)
            self._raise_on_motion_safety_violation("CALIB k_a")
            ps = self._ps()
            L  = self.spring_length_latest
            w  = self.w_latest
            if not self.fin(ps):
                continue
            if not self.vel_ok(MOVE_VEL_LIMIT):
                self._alog("CALIB k_a: velocity spike -- stopping ramp")
                break
            if ps > CALIB_FORCE_TARGET_N:
                break
            if (not math.isnan(w)) and w > CALIB_W_EXCLUDE_M:
                self._alog("CALIB k_a: w_max threshold exceeded -- stopping ramp")
                break
            if self.fin(L):
                dL = L0 - L
                if dL > 1e-5:
                    data.append((dL, ps))

        self.publish_press_targets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=5, dt=0.05)
        self.spin_for(2.0)
        self._raise_on_motion_safety_violation("CALIB k_a reset")

        if len(data) < 4:
            self._alog(f"CALIB k_a: only {len(data)} points -- fit unreliable")
            return float("nan"), float("nan"), data

        xs  = np.array([d[0] for d in data])
        ys  = np.array([d[1] for d in data])
        A   = np.column_stack([xs, np.ones_like(xs)])
        res = np.linalg.lstsq(A, ys, rcond=None)
        k_a, _b = res[0]
        resid = ys - A @ res[0]
        n     = len(xs)
        SSx   = float(np.sum((xs - xs.mean()) ** 2))
        MSE   = float(np.sum(resid ** 2) / max(n - 2, 1))
        se    = float(np.sqrt(MSE / SSx)) if SSx > 0 else float("nan")
        self._alog(f"  k_a = {k_a:.1f} N/m  b = {_b:.3f} N  SE = {se:.1f}  n = {n}")
        return float(k_a), se, data

    def _calib_buckling_threshold(self, L0: float) -> "tuple[float, float, float, list]":
        """
        Compress past P_b at K_lat = CALIB_K_LAT_NPM. Record (P_s, w_max) pairs.
        Fit a piecewise-linear change-point model to w_max(P_s) via grid search:

            w_max = a0 + a1*P_s   for P_s < P_b^cp
                  = b0 + b1*P_s   for P_s >= P_b^cp

        The candidate P_b^cp that minimises total RSS is returned.
        Effective bending stiffness: B_eff = P_b * L0^2 / pi^2.
        This avoids separately measuring E, I, and end-restraint factors.

        Returns (P_b_hat [N], half-step uncertainty [N], B_eff [N.m^2], data)
        where data = [(P_s_N, w_max_m), ...].
        w_max topic is required; if absent returns NaN.
        """
        import numpy as np
        self._alog("CALIB P_b: piecewise change-point in w_max(P_s)")
        self.set_k_lateral(CALIB_K_LAT_NPM)
        self.spin_for(2.0)
        data = []
        off1 = off2 = 0.0

        while True:
            off1 = min(off1 + CALIB_STEP_M, PRESS_END_X + 0.010)
            off2 = min(off2 + CALIB_STEP_M, PRESS_END_X + 0.010)
            self.publish_press_targets(off1, 0.0, 0.0, off2, 0.0, 0.0)
            self.spin_for(CALIB_SAMPLE_DELAY)
            self._raise_on_motion_safety_violation("CALIB P_b")
            ps = self._ps()
            w  = self.w_latest
            if not self.fin(ps):
                continue
            if not self.vel_ok(MOVE_VEL_LIMIT):
                self._alog("CALIB P_b: velocity spike -- stopping")
                break
            if ps > CALIB_PB_FORCE_MAX_N:
                break
            if max(self.contact_fx1, self.contact_fx2) > MAX_FORCE_MAG_ABORT:
                break
            if not math.isnan(w):
                data.append((ps, w))

        self.publish_press_targets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=5, dt=0.05)
        self.spin_for(2.0)
        self._raise_on_motion_safety_violation("CALIB P_b reset")

        if len(data) < 8:
            self._alog(f"CALIB P_b: only {len(data)} points -- change-point unreliable")
            return float("nan"), float("nan"), float("nan"), data

        ps_arr = np.array([d[0] for d in data])
        w_arr  = np.array([d[1] for d in data])

        def _seg_rss(x: np.ndarray, y: np.ndarray) -> float:
            if len(x) < 2:
                return 0.0
            A = np.column_stack([x, np.ones_like(x)])
            r = np.linalg.lstsq(A, y, rcond=None)
            return float(np.sum((y - A @ r[0]) ** 2))

        best_rss, best_cp = float("inf"), float("nan")
        for i in range(3, len(ps_arr) - 3):
            cp  = ps_arr[i]
            rss = (_seg_rss(ps_arr[ps_arr <= cp], w_arr[ps_arr <= cp]) +
                   _seg_rss(ps_arr[ps_arr >  cp], w_arr[ps_arr >  cp]))
            if rss < best_rss:
                best_rss, best_cp = rss, cp

        idx  = int(np.searchsorted(ps_arr, best_cp))
        lo   = ps_arr[max(idx - 1, 0)]
        hi   = ps_arr[min(idx + 1, len(ps_arr) - 1)]
        dp   = float((hi - lo) / 2.0)
        B_eff = (float(best_cp) * (L0 ** 2) / (math.pi ** 2)
                 if not math.isnan(best_cp) else float("nan"))
        self._alog(
            f"  P_b^cp = {best_cp:.3f} N  +/-{dp:.3f} N  "
            f"B_eff = {B_eff*1e6:.2f} N.mm^2  n = {len(data)}"
        )
        return float(best_cp), dp, B_eff, data

    def _psi_rad(self) -> float:
        """
        Best-available contact rotation estimate [rad].
        psi = alpha_cap - alpha_s  (endcap orientation minus spring axis angle).
        Prefers left-side camera estimate; falls back to TCP yaw (deg -> rad).
        """
        if not math.isnan(self.psi_L_latest):
            return self.psi_L_latest
        if not math.isnan(self.yaw_latest):
            return math.radians(self.yaw_latest)
        return float("nan")

    def _calib_k_theta(self, L0: float, P_b_hat: float) -> "tuple[float, float, list]":
        """
        Eccentric loading test: apply lateral offset e [mm] at a sub-buckling
        load, measure contact rotation psi, fit M = k_theta * psi + b.
        Applied moment: M = P_s * e  (first-order for small psi).

        Both TCPs shift in +y by e_m, creating eccentricity e relative to the
        spring axis.  Only points with |psi| < 2 deg are used in the fit.

        This test is INDEPENDENT of the main K_lat sweep so that k_theta is not
        circularly identified from the same data used to validate the model.

        Returns (k_theta [N.m/rad], SE [N.m/rad], data)
        where data = [(e_m, M_Nm, psi_rad), ...].
        """
        import numpy as np
        self._alog("CALIB k_theta: eccentric loading M = P_s*e vs psi")
        self.set_k_lateral(CALIB_K_LAT_NPM)
        self.spin_for(2.0)

        P_target = max(0.3, CALIB_LOAD_FRAC * (
            float(P_b_hat) if not math.isnan(P_b_hat) else P_B_THEORY_N))

        # Compress symmetrically to P_target first
        off1 = off2 = 0.0
        for _ in range(80):
            ps = self._ps()
            if self.fin(ps) and ps >= P_target:
                break
            off1 = min(off1 + CALIB_STEP_M, PRESS_END_X + 0.010)
            off2 = min(off2 + CALIB_STEP_M, PRESS_END_X + 0.010)
            self.publish_press_targets(off1, 0.0, 0.0, off2, 0.0, 0.0)
            self.spin_for(CALIB_SAMPLE_DELAY)
            self._raise_on_motion_safety_violation("CALIB k_theta preload")
        self.spin_for(1.0)  # settle at P_target
        self._raise_on_motion_safety_violation("CALIB k_theta preload settle")

        base_x1 = self.current_press_x1
        base_x2 = self.current_press_x2
        data = []

        for e_mm in CALIB_ECC_MM:
            e_m = e_mm * 1e-3
            # Shift both TCPs by e_m in y: creates eccentricity between
            # contact line and spring axis.
            self.publish_press_targets(base_x1, e_m, 0.0,
                                       base_x2, e_m, 0.0)
            self.spin_for(1.0)  # settle
            self._raise_on_motion_safety_violation("CALIB k_theta eccentric shift")

            ps  = self._ps()
            psi = self._psi_rad()
            if self.fin(ps) and self.fin(psi):
                M = ps * e_m
                data.append((e_m, M, psi))
                self._alog(
                    f"  e = {e_mm:+.1f} mm  "
                    f"P_s = {ps:.3f} N  "
                    f"M = {M*1000:.2f} mN.m  "
                    f"psi = {math.degrees(psi):.2f} deg"
                )

            # Restore symmetric compression before next step
            self.publish_press_targets(base_x1, 0.0, 0.0,
                                       base_x2, 0.0, 0.0)
            self.spin_for(0.6)
            self._raise_on_motion_safety_violation("CALIB k_theta restore")

        self.publish_press_targets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=5, dt=0.05)
        self.spin_for(2.0)
        self._raise_on_motion_safety_violation("CALIB k_theta reset")

        if len(data) < 4:
            self._alog(f"CALIB k_theta: only {len(data)} data points -- fit unreliable")
            return float("nan"), float("nan"), data

        psi_arr = np.array([d[2] for d in data])
        M_arr   = np.array([d[1] for d in data])

        # Small-angle filter: |psi| < 2 deg to stay in the linear regime
        mask = np.abs(psi_arr) < math.radians(2.0)
        if mask.sum() < 3:
            self._alog("CALIB k_theta: < 3 small-angle points -- using full dataset")
            mask = np.ones(len(psi_arr), dtype=bool)

        psi_f = psi_arr[mask]
        M_f   = M_arr[mask]
        A     = np.column_stack([psi_f, np.ones_like(psi_f)])
        res   = np.linalg.lstsq(A, M_f, rcond=None)
        k_th, _b = res[0]
        resid = M_f - A @ res[0]
        n     = len(psi_f)
        SSx   = float(np.sum((psi_f - psi_f.mean()) ** 2))
        MSE   = float(np.sum(resid ** 2) / max(n - 2, 1))
        se    = float(np.sqrt(MSE / SSx)) if SSx > 0 else float("nan")
        self._alog(
            f"  k_theta = {k_th:.5f} N.m/rad  SE = {se:.5f}  "
            f"n_fit = {n}/{len(data)}")
        return float(k_th), se, data

    def calibrate_spring(self, run_dir: str) -> CalibResult:
        """
        Full spring calibration sequence.  Run once per session (or after
        spring replacement) to estimate the effective experimental constants
        that cannot be taken from nominal model parameters.

        Quantities estimated and their measurement sources
        --------------------------------------------------
        L0       [m]         Unloaded spring end-to-end length.
                             Source: camera (/spring_monitor/spring_length_m),
                             CALIB_L0_SAMPLES readings at zero compression.

        k_a      [N/m]       Axial spring stiffness.
                             Source: camera (DeltaL) + robot (P_s).
                             Quasi-static compression ramp at K_lat =
                             CALIB_K_LAT_NPM (current controller limit, used to
                             suppress lateral rotation during the axial ramp).
                             OLS fit P_s = k_a*DeltaL + b on points where
                             w_max < CALIB_W_EXCLUDE_M (2 mm) and
                             P_s < CALIB_FORCE_TARGET_N (2.5 N, safely below
                             P_b ~ 3.49 N).

        P_b_hat  [N]         Buckling threshold (compression onset load).
                             Source: camera (w_max) + robot (P_s).
                             Compression ramp to CALIB_PB_FORCE_MAX_N (5.5 N,
                             ~1.6 x P_b).  Grid-search piecewise-linear
                             change-point fit on w_max(P_s) gives a
                             threshold-independent estimate of P_b^cp.
                                                         Theory is computed from the shared spring model in
                                                         ablation_config.py.
                             NOTE: w_max is the in-plane deflection as seen
                             by the top camera.  If the spring buckles
                             out-of-plane, w_max will be biased low and
                             P_b_hat will appear artificially high.  Constrain
                             the spring to buckle in the camera plane.

        B_eff    [N.m^2]     Effective bending stiffness.
                             Derived: B_eff = P_b_hat * L0^2 / pi^2.
                             Collapses E, I, and end-restraint factor K_EC
                             into a single calibrated constant; no need to
                             estimate them separately.
                                                         Nominal EI and end-constraint factor are centralized
                                                         in ablation_config.py.

        k_theta  [N.m/rad]   Contact rotational stiffness.
                             Source: eccentric loading test -- FULLY INDEPENDENT
                             of the K_lat ablation sweep (estimating k_theta
                             from that sweep would be circular, since the
                             prediction K_lat* = P_b^2/k_theta is what the
                             sweep is meant to validate).
                             Protocol: for each lateral offset e in
                             CALIB_ECC_MM ([-3,-2,-1,0,1,2,3] mm), apply
                             sub-buckling load CALIB_LOAD_FRAC*P_b_hat (50%).
                             Bending moment M = P_s * e.  OLS fit M = k_theta*psi
                             on samples with |psi| < 2 deg.
                             psi is read from /spring_monitor/contact_rotation_L_rad
                             (or tcp_yaw_deg as fallback).

        K_lat*   [N/m]       Critical lateral stiffness (theory).
                             Derived: K_lat* = P_b_hat^2 / k_theta.
                                                         Nominal K_lat* is derived from the shared spring
                                                         model in ablation_config.py.
                                                         The default ablation sweep is clipped against the
                                                         configured controller stiffness limit.

        K_lat^m  [N/m]       Measured closed-loop lateral stiffness at the
                             current ceiling setting (CALIB_K_LAT_NPM).
                             Source: TCP perturbation test via identify_klat().
                             Verifies that the impedance controller actually
                             delivers the commanded stiffness at the end-point.

        Data sources summary
        --------------------
          Camera only:   L, DeltaL, w_max, psi  (geometry)
          Robot only:    P_s, P_imb              (force)
          Both needed:   k_a, P_b, k_theta, K_lat^m

        Output
        ------
        Writes calib_result_<ts>.json to run_dir with all scalar estimates,
        their standard errors, and raw data arrays (DeltaL_m, Ps_N, wmax_m
        for axial/buckling stages; e_m, M_Nm, psi_rad for k_theta stage)
        for post-hoc verification and plotting.

        Returns a CalibResult dataclass with all calibrated quantities.
        """
        self._alog("=" * 60)
        self._alog("SPRING CALIBRATION")
        self._alog("=" * 60)
        ts = time.strftime("%Y%m%d_%H%M%S")
        calibration_label = f"calibration_{ts}"
        self._set_active_depth_frame_context(run_dir, calibration_label)
        self._start_external_capture_session(
            phase="calibration",
            name=calibration_label,
            ablation_index=0,
        )
        capture_meta: Dict[str, Any] = {}
        try:
            depth_camera_calibration_path = self.write_depth_camera_calibration(run_dir)
            self.capture_depth_camera_snapshot(run_dir, f"{calibration_label}_start")
            aligned, align_msg = self.ensure_camera_alignment(run_dir, calibration_label)
            if not aligned:
                raise CalibrationSafetyAbort(align_msg)

            # 1. L0
            L0, L0_std = self._calib_L0()
            if math.isnan(L0):
                self._alog(
                    "WARNING: spring_length_m topic not received. "
                    f"Falling back to nominal L0 = {SPRING_NOMINAL_LENGTH_M:.2f} m.")
                L0, L0_std = SPRING_NOMINAL_LENGTH_M, float("nan")
            self.spin_for(REST_BETWEEN_CASES_S)

            # 2. k_a  (camera + robot force)
            k_a, k_a_se, ka_data = self._calib_axial_stiffness(L0)
            self.spin_for(REST_BETWEEN_CASES_S)

            # 3. P_b, B_eff  (camera w_max required)
            P_b_hat, P_b_se, B_eff, pb_data = self._calib_buckling_threshold(L0)
            self.spin_for(REST_BETWEEN_CASES_S)

            # 4. k_theta  (psi topic or yaw fallback; independent eccentric test)
            k_th, k_th_se, kth_data = self._calib_k_theta(L0, P_b_hat)
            self.spin_for(REST_BETWEEN_CASES_S)

            # 5. K_lat* = P_b^2 / k_theta
            K_lat_star = (
                (float(P_b_hat) ** 2) / float(k_th)
                if (self.fin(P_b_hat) and self.fin(k_th) and k_th > 0)
                else float("nan")
            )

            # 6. K_lat^meas at HW ceiling (reuses identify_klat)
            self.set_k_lateral(CALIB_K_LAT_NPM)
            self.spin_for(REST_BETWEEN_CASES_S)
            K_lat_meas, K_lat_se = self.identify_klat()

            self.capture_depth_camera_snapshot(run_dir, f"{calibration_label}_complete")
            capture_meta = self._stop_external_capture_session(status="completed")

            result = CalibResult(
                ts=ts,
                L0_m=L0,               L0_std_m=L0_std,
                k_a_Npm=k_a,           k_a_stderr=k_a_se,
                P_b_hat_N=P_b_hat,     P_b_stderr_N=P_b_se,
                B_eff_Nm2=B_eff,
                k_theta_Nm_per_rad=k_th,  k_theta_stderr=k_th_se,
                K_lat_star_Npm=K_lat_star,
                K_lat_meas_Npm=K_lat_meas, K_lat_meas_stderr=K_lat_se,
                external_capture_metadata=capture_meta,
            )

            # Write JSON (includes raw data arrays for verification / plotting)
            json_path = os.path.join(run_dir, f"calib_result_{ts}.json")
            with open(json_path, "w") as f:
                json.dump({
                    "ts":                  ts,
                    "L0_m":                L0,
                    "L0_std_m":            L0_std,
                    "k_a_Npm":             k_a,
                    "k_a_stderr":          k_a_se,
                    "P_b_hat_N":           P_b_hat,
                    "P_b_stderr_N":        P_b_se,
                    "B_eff_Nm2":           B_eff,
                    "k_theta_Nm_per_rad":  k_th,
                    "k_theta_stderr":      k_th_se,
                    "K_lat_star_Npm":      K_lat_star,
                    "K_lat_meas_Npm":      K_lat_meas,
                    "K_lat_meas_stderr":   K_lat_se,
                    "depth_camera_calibration_path": depth_camera_calibration_path,
                    "depth_camera_calibration_source": DEFAULT_DEPTH_CAMERA_SOURCE,
                    "external_capture":    capture_meta,
                    # raw data for post-hoc plotting / verification
                    "ka_data_delta_L_m":   [d[0] for d in ka_data],
                    "ka_data_P_s_N":       [d[1] for d in ka_data],
                    "pb_data_P_s_N":       [d[0] for d in pb_data],
                    "pb_data_w_max_m":     [d[1] for d in pb_data],
                    "kth_data_e_m":        [d[0] for d in kth_data],
                    "kth_data_M_Nm":       [d[1] for d in kth_data],
                    "kth_data_psi_rad":    [d[2] for d in kth_data],
                }, f, indent=2)
            result.json_path = json_path

            b_str = (f"  B_eff    = {B_eff*1e6:.2f} N.mm^2"
                     if self.fin(B_eff) else "  B_eff    = nan")
            self._alog("\nCALIBRATION SUMMARY")
            self._alog(f"  L0       = {L0*1000:.2f} mm  (+/-{L0_std*1000:.2f} mm)")
            self._alog(f"  k_a      = {k_a:.1f} N/m" +
                       (f"  SE={k_a_se:.1f}" if self.fin(k_a_se) else ""))
            self._alog(f"  P_b^hat  = {P_b_hat:.3f} N  (theory {P_B_THEORY_N:.3f} N)")
            self._alog(b_str)
            self._alog(f"  k_theta  = {k_th:.5f} N.m/rad")
            self._alog(
                f"  K_lat*   = {K_lat_star:.1f} N/m  "
                f"(theory ~{SPRING_THEORETICAL_CRITICAL_K_LAT_NPM:.0f} N/m)"
            )
            self._alog(f"  K_lat^m  = {K_lat_meas:.1f} N/m  (cmd {CALIB_K_LAT_NPM:.0f} N/m)")
            self._alog(f"  JSON     = {json_path}")
            return result
        except Exception:
            if not capture_meta:
                self._stop_external_capture_session(status="aborted")
            raise
        finally:
            self._flush_alog()
            self._clear_active_depth_frame_context()

    # ══════════════════════════════════════════════════════════════════════════
    # 9) OUTER SWEEP LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def run_ablation(self, cases_klat: Dict[str, List[float]],
                     n_reps: int, dry_run: bool = False) -> List[RunResult]:
        """
        Main entry point.  Iterates over every (case, K_lat) pair and runs
        n_reps repetitions, writing results incrementally.
        """
        run_ts  = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.log_root, f"ablation_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        self._ablation_log_path = os.path.join(run_dir, "ablation_log.txt")
        self.write_depth_camera_calibration(run_dir)

        # summary CSV: same columns as sim summary.csv + extra HW columns
        summary_path = os.path.join(run_dir, "summary.csv")
        SUMMARY_COLS = [
            "case", "K_lat_Npm", "K_lat_meas_Npm", "K_lat_meas_stderr",
            "P_b_th_N", "P_c_th_N", "hw_realisable",
            "t_mb_zero_s", "t_mc_zero_s", "t_mb_tau_zero_s", "t_mc_tau_zero_s",
            "t_w_3mm_s", "t_phi_4deg_s", "P_s_max_N",
            # margin overlays (continuous values)
            "mb_at_contact_N", "mb_min_N", "mc_at_contact_N", "mc_min_N",
            # extra HW-only columns
            "camera_aruco_records", "camera_aruco_valid_records",
            "camera_aruco_all_markers_records", "camera_aruco_dropout_run_count",
            "camera_aruco_angle_flip_count",
            "ablation_index", "rep", "passed", "abort_reason",
            "t_contact_s", "contact_duration_s",
            "capture_start_wall_time_ns", "capture_contact_wall_time_ns", "capture_end_wall_time_ns",
            "P_s_at_onset_N",
        ] + harness_camera_aruco_log_columns()
        with open(summary_path, "w", newline="") as sf:
            csv.DictWriter(sf, fieldnames=SUMMARY_COLS,
                           extrasaction="ignore").writeheader()

        all_results: List[RunResult] = []
        n_total = sum(len(v) for v in cases_klat.values()) * n_reps
        n_done  = 0

        if dry_run:
            self._alog("DRY RUN: liveness check only, no motion.")
            ok, msg = self._stage_liveness()
            self._alog(f"Liveness: {msg}")
            self._flush_alog()
            return []

        prev_klat: Optional[float] = None
        for case, klat_list in cases_klat.items():
            for K_lat in klat_list:
                safe_k = f"{K_lat:.0f}".replace(".", "p")

                # ── change K_lat ───────────────────────────────────────────
                klat_meas   = float("nan")
                klat_stderr = float("nan")
                if K_lat != prev_klat:
                    self._alog(f"Setting K_lat = {K_lat:.1f} N/m  [{case}]")
                    self.set_k_lateral(K_lat)
                    # verify idle before starting reps
                    self.spin_for(REST_BETWEEN_CASES_S)
                    aligned, align_msg = self.ensure_camera_alignment(run_dir, f"case_{case}_K{safe_k}")
                    if not aligned:
                        raise CalibrationSafetyAbort(align_msg)
                    # ── identify actual closed-loop K_lat via perturbation ──
                    if self.skip_klat_command:
                        self._alog("  K_lat^meas = skipped for gazebo smoke mode")
                    else:
                        klat_meas, klat_stderr = self.identify_klat()
                        self._alog(
                            f"  K_lat^meas = {klat_meas:.1f} N/m  "
                            f"(commanded {K_lat:.1f}, stderr {klat_stderr:.1f})"
                        )
                    prev_klat = K_lat

                # P_c_th: analytical critical contact force
                # P_c = sqrt(K_lat * k_theta) where k_theta comes from the
                # shared spring model.
                # This is the contact-rotation threshold under dead-centre loading.
                # No division by L: the formula is already in [N], not [N/m].
                K_THETA_NM = SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD
                P_c_th = math.sqrt(max(K_lat, 0.0) * K_THETA_NM)

                self._alog(f"\n{'─'*50}")
                self._alog(f"CASE: {case}  K_lat={K_lat:.1f} N/m  P_c_th={P_c_th:.3f} N  reps={n_reps}")

                for rep in range(1, n_reps + 1):
                    n_done += 1
                    self._current_ablation_index = n_done
                    self._alog(f"  Rep {rep}/{n_reps}  ({n_done}/{n_total} total)")

                    r = self._run_one(case, K_lat, rep, P_c_th, run_dir)
                    r.ablation_index = n_done
                    r.K_lat_meas_Npm    = klat_meas
                    r.K_lat_meas_stderr = klat_stderr
                    all_results.append(r)

                    # append to summary CSV immediately
                    with open(summary_path, "a", newline="") as sf:
                        capture_meta = r.external_capture_metadata or {}
                        summary_row = {
                            "case":           r.case,
                            "K_lat_Npm":      r.K_lat_Npm,
                            "K_lat_meas_Npm": r.K_lat_meas_Npm,
                            "K_lat_meas_stderr": r.K_lat_meas_stderr,
                            "P_b_th_N":       r.P_b_th_N,
                            "P_c_th_N":       r.P_c_th_N,
                            "hw_realisable":  r.hw_realisable,
                            "t_mb_zero_s":    r.t_mb_zero_s,
                            "t_mc_zero_s":    r.t_mc_zero_s,
                            "t_mb_tau_zero_s": float("nan"),   # not available on HW
                            "t_mc_tau_zero_s": float("nan"),
                            "t_w_3mm_s":      r.t_w_3mm_s,
                            "t_phi_4deg_s":   r.t_phi_4deg_s,
                            "P_s_max_N":      r.P_s_max_N,
                            "mb_at_contact_N": r.mb_at_contact_N,
                            "mb_min_N":        r.mb_min_N,
                            "mc_at_contact_N": r.mc_at_contact_N,
                            "mc_min_N":        r.mc_min_N,
                            "camera_aruco_records": r.camera_aruco_records,
                            "camera_aruco_valid_records": r.camera_aruco_valid_records,
                            "camera_aruco_all_markers_records": r.camera_aruco_all_markers_records,
                            "camera_aruco_dropout_run_count": r.camera_aruco_dropout_run_count,
                            "camera_aruco_angle_flip_count": r.camera_aruco_angle_flip_count,
                            "ablation_index": r.ablation_index,
                            "rep":            r.rep,
                            "passed":         r.passed,
                            "abort_reason":   r.abort_reason,
                            "t_contact_s":    r.t_contact_s,
                            "contact_duration_s": r.contact_duration_s,
                            "capture_start_wall_time_ns": capture_meta.get("capture_start_wall_time_ns", ""),
                            "capture_contact_wall_time_ns": capture_meta.get("contact_wall_time_ns", ""),
                            "capture_end_wall_time_ns": capture_meta.get("capture_end_wall_time_ns", ""),
                            "P_s_at_onset_N": r.P_s_at_onset_N,
                        }
                        summary_row.update(r.aruco_summary_row)
                        csv.DictWriter(sf, fieldnames=SUMMARY_COLS,
                                       extrasaction="ignore").writerow(summary_row)

                    # status line
                    status = "PASS" if r.passed else f"FAIL ({r.abort_reason})"
                    self._alog(
                        f"    → {status}  "
                        f"t_w={r.t_w_3mm_s:.2f}s  "
                        f"t_mb={r.t_mb_zero_s:.3f}s  "
                        f"mb_min={r.mb_min_N:.4f}  mc_min={r.mc_min_N:.4f}  "
                        f"P_max={r.P_s_max_N:.2f}N"
                    )

                    self._flush_alog()

                    # rest between reps
                    if rep < n_reps:
                        self.spin_for(REST_BETWEEN_REPS_S)

        self._alog(f"\n{'='*50}")
        n_pass = sum(1 for r in all_results if r.passed)
        self._alog(f"COMPLETE: {n_pass}/{len(all_results)} runs passed.")
        self._alog(f"Summary CSV: {summary_path}")
        self._flush_alog()

        return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# 10) MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K_lat ablation harness")
    p.add_argument("--reps",     type=int, default=DEFAULT_REPS,
                   help="Repetitions per K_lat setpoint (default 20)")
    p.add_argument("--cases",    nargs="*", default=None,
                   help="Subset of case names to run (e.g. baseline contact_dom)")
    p.add_argument("--k-lat",    nargs="*", type=float, default=None,
                   help="Explicit K_lat values to run (overrides --cases)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Liveness check only, no motion")
    p.add_argument("--gazebo-smoke", action="store_true",
                   help="Run a single fixed-stiffness Gazebo smoke test by skipping live K_lat commands")
    p.add_argument("--min-ee-separation", type=float, default=MIN_EE_SEPARATION_M,
                   help="Minimum allowed 3-D EE separation before abort")
    p.add_argument("--max-ee-x-gap", type=float, default=None,
                   help="Maximum allowed absolute EE x-gap before abort (default: derive from home gap + margin)")
    p.add_argument("--max-ee-x-gap-margin", type=float, default=MAX_EE_X_GAP_MARGIN_M,
                   help="Margin added above the measured home EE x-gap when deriving the max gap")
    p.add_argument("--robot1-y-offset", type=float, default=0.0,
                   help="Additional robot1 waypoint y offset in meters")
    p.add_argument("--robot2-y-offset", type=float, default=0.0,
                   help="Additional robot2 waypoint y offset in meters")
    p.add_argument("--robot1-z-offset", type=float, default=0.0,
                   help="Additional robot1 waypoint z offset in meters (positive raises commanded z)")
    p.add_argument("--robot2-z-offset", type=float, default=0.0,
                   help="Additional robot2 waypoint z offset in meters (positive raises commanded z)")
    p.add_argument("--color-image-topic", default=DEFAULT_COLOR_IMAGE_TOPIC,
                   help="Color image topic used for depth-camera snapshots and alignment context")
    p.add_argument("--color-camera-info-topic", default=DEFAULT_COLOR_CAMERA_INFO_TOPIC,
                   help="CameraInfo topic paired with --color-image-topic")
    p.add_argument("--depth-image-topic", default=DEFAULT_DEPTH_IMAGE_TOPIC,
                   help="Aligned depth image topic used for depth-camera snapshots")
    p.add_argument("--depth-camera-info-topic", default=DEFAULT_DEPTH_CAMERA_INFO_TOPIC,
                   help="CameraInfo topic paired with --depth-image-topic")
    p.add_argument("--camera-aruco-summary-topic", default=DEFAULT_CAMERA_ARUCO_SUMMARY_TOPIC,
                   help="camera_aruco JSON summary topic captured per run/calibration for offline paper analysis")
    p.add_argument("--raw-camera-stream-rate-hz", type=float, default=DEFAULT_RAW_CAMERA_STREAM_RATE_HZ,
                   help="Decimated raw color/depth frame capture rate written into the run log tree (0 disables)")
    p.add_argument("--raw-camera-video-fps", type=float, default=DEFAULT_RAW_CAMERA_VIDEO_FPS,
                   help="Optional encoded raw-stream video FPS written per run/calibration (0 disables video export)")
    p.add_argument("--capture-cmd-file", default=None,
                   help="Optional command file path for external buffered capture control")
    p.add_argument("--capture-buffer-pre-s", type=float, default=DEFAULT_CAPTURE_BUFFER_PRE_S,
                   help="Seconds of pre-contact buffered video to request from the external recorder")
    p.add_argument("--capture-buffer-post-s", type=float, default=DEFAULT_CAPTURE_BUFFER_POST_S,
                   help="Seconds of post-contact buffered video to request from the external recorder")
    p.add_argument("--alignment-source", choices=["auto", "aruco", "fallback", "off"],
                   default=DEFAULT_ALIGNMENT_SOURCE,
                   help="Alignment signal preference: ArUco topics first, fallback spring-monitor topics, or off")
    p.add_argument("--alignment-policy", choices=["auto_then_manual", "manual", "warn", "off"],
                   default=DEFAULT_ALIGNMENT_POLICY,
                   help="How to respond when the camera sees a misalignment")
    p.add_argument("--alignment-max-auto-z-trim", type=float, default=DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
                   help="Maximum per-robot z trim applied automatically from the ArUco alignment monitor")
    p.add_argument("--alignment-fallback-w-tol", type=float, default=DEFAULT_ALIGNMENT_FALLBACK_W_TOL_M,
                   help="Fallback acceptable |lateral_deflection_m| threshold before manual re-alignment is requested")
    p.add_argument("--alignment-fallback-yaw-tol-deg", type=float, default=DEFAULT_ALIGNMENT_FALLBACK_YAW_TOL_DEG,
                   help="Fallback acceptable |tcp_yaw_deg| threshold before manual re-alignment is requested")
    p.add_argument("--alignment-fallback-psi-diff-tol-deg", type=float, default=DEFAULT_ALIGNMENT_FALLBACK_PSI_DIFF_TOL_DEG,
                   help="Fallback acceptable |contact_rotation_L - contact_rotation_R| threshold in degrees")
    p.add_argument("--abort-disable-torque", action="store_true",
                   help="Disable Dynamixel torque on both arms whenever the harness aborts")
    p.add_argument("--robot1-port", default=None,
                   help="Serial port for robot1 torque disable on abort")
    p.add_argument("--robot2-port", default=None,
                   help="Serial port for robot2 torque disable on abort")
    p.add_argument("--dxl-baud", type=int, default=DEFAULT_TORQUE_DISABLE_BAUD,
                   help="Baud rate used by the abort-time torque disable helper")
    p.add_argument("--calibrate", action="store_true",
                   help="Run spring calibration before the ablation sweep "
                        "(L0, k_a, P_b, k_theta, K_lat*, K_lat^meas)")
    p.add_argument("--calibrate-only", action="store_true",
                   help="Run spring calibration only; skip the K_lat ablation sweep")
    p.add_argument("--log-dir",  default=os.environ.get("OMX_LOG_DIR",
                                 "/tmp/variable_stiffness_logs"),
                   help="Output directory for run logs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.gazebo_smoke and (args.calibrate or args.calibrate_only):
        raise SystemExit("--gazebo-smoke cannot be combined with calibration runs")

    # Build cases_klat dict
    if args.gazebo_smoke:
        if args.cases:
            raise SystemExit("--gazebo-smoke uses a single fixed stiffness label; do not combine it with --cases")
        if args.k_lat:
            unique_klat = sorted(set(float(value) for value in args.k_lat))
            if len(unique_klat) != 1:
                raise SystemExit("--gazebo-smoke requires exactly one unique --k-lat value")
            cases_klat = {"gazebo_smoke": [unique_klat[0]]}
        else:
            cases_klat = {"gazebo_smoke": [CARTESIAN_STIFFNESS_LIMIT_NPM]}
    elif args.k_lat:
        if any(value > CARTESIAN_STIFFNESS_LIMIT_NPM for value in args.k_lat):
            raise SystemExit(
                f"Custom --k-lat values must be <= {CARTESIAN_STIFFNESS_LIMIT_NPM:.1f} N/m"
            )
        cases_klat = {"custom": args.k_lat}
    else:
        if args.cases:
            cases_klat = {k: v for k, v in HW_K_LAT_CASES.items()
                          if k in args.cases}
            if not cases_klat:
                raise SystemExit(
                    f"No matching cases found. Available: {list(HW_K_LAT_CASES)}")
        else:
            cases_klat = HW_K_LAT_CASES

    rclpy.init()
    node = AblationHarness(
        log_root=args.log_dir,
        skip_klat_command=args.gazebo_smoke,
        min_ee_separation=args.min_ee_separation,
        enforce_ee_separation=not args.gazebo_smoke,
        max_ee_x_gap=args.max_ee_x_gap,
        max_ee_x_gap_margin=args.max_ee_x_gap_margin,
        robot1_y_offset=getattr(args, "robot1_y_offset", 0.0),
        robot2_y_offset=getattr(args, "robot2_y_offset", 0.0),
        robot1_z_offset=getattr(args, "robot1_z_offset", 0.0),
        robot2_z_offset=getattr(args, "robot2_z_offset", 0.0),
        alignment_source=getattr(args, "alignment_source", DEFAULT_ALIGNMENT_SOURCE),
        alignment_policy=getattr(args, "alignment_policy", DEFAULT_ALIGNMENT_POLICY),
        alignment_max_auto_z_trim=getattr(args, "alignment_max_auto_z_trim", DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M),
        alignment_fallback_w_tol=getattr(args, "alignment_fallback_w_tol", DEFAULT_ALIGNMENT_FALLBACK_W_TOL_M),
        alignment_fallback_yaw_tol_deg=getattr(args, "alignment_fallback_yaw_tol_deg", DEFAULT_ALIGNMENT_FALLBACK_YAW_TOL_DEG),
        alignment_fallback_psi_diff_tol_deg=getattr(
            args,
            "alignment_fallback_psi_diff_tol_deg",
            DEFAULT_ALIGNMENT_FALLBACK_PSI_DIFF_TOL_DEG,
        ),
        abort_disable_torque=args.abort_disable_torque,
        robot1_port=args.robot1_port,
        robot2_port=args.robot2_port,
        torque_disable_baud=args.dxl_baud,
        color_image_topic=getattr(args, "color_image_topic", DEFAULT_COLOR_IMAGE_TOPIC),
        color_camera_info_topic=getattr(args, "color_camera_info_topic", DEFAULT_COLOR_CAMERA_INFO_TOPIC),
        depth_image_topic=getattr(args, "depth_image_topic", DEFAULT_DEPTH_IMAGE_TOPIC),
        depth_camera_info_topic=getattr(args, "depth_camera_info_topic", DEFAULT_DEPTH_CAMERA_INFO_TOPIC),
        camera_aruco_summary_topic=getattr(
            args,
            "camera_aruco_summary_topic",
            DEFAULT_CAMERA_ARUCO_SUMMARY_TOPIC,
        ),
        raw_camera_stream_rate_hz=getattr(
            args,
            "raw_camera_stream_rate_hz",
            DEFAULT_RAW_CAMERA_STREAM_RATE_HZ,
        ),
        raw_camera_video_fps=getattr(args, "raw_camera_video_fps", DEFAULT_RAW_CAMERA_VIDEO_FPS),
        capture_cmd_file=getattr(args, "capture_cmd_file", None),
        capture_buffer_pre_s=getattr(args, "capture_buffer_pre_s", DEFAULT_CAPTURE_BUFFER_PRE_S),
        capture_buffer_post_s=getattr(args, "capture_buffer_post_s", DEFAULT_CAPTURE_BUFFER_POST_S),
    )
    try:
        if args.calibrate or args.calibrate_only:
            calib_dir = os.path.join(
                args.log_dir, f"calib_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(calib_dir, exist_ok=True)
            node._ablation_log_path = os.path.join(calib_dir, "calib_log.txt")
            cr = node.calibrate_spring(calib_dir)
            b_str = (f"{cr.B_eff_Nm2*1e6:.2f} N.mm^2"
                     if not math.isnan(cr.B_eff_Nm2) else "nan")
            print(
                f"\nCalibration complete:\n"
                f"  L0      = {cr.L0_m*1000:.2f} mm\n"
                f"  k_a     = {cr.k_a_Npm:.1f} N/m\n"
                f"  P_b     = {cr.P_b_hat_N:.3f} N  (theory {P_B_THEORY_N:.3f} N)\n"
                f"  B_eff   = {b_str}\n"
                f"  k_theta = {cr.k_theta_Nm_per_rad:.5f} N.m/rad\n"
                f"  K_lat*  = {cr.K_lat_star_Npm:.1f} N/m  "
                f"(theory ~{SPRING_THEORETICAL_CRITICAL_K_LAT_NPM:.0f} N/m)\n"
                f"  K_lat^m = {cr.K_lat_meas_Npm:.1f} N/m\n"
                f"  JSON    = {cr.json_path}"
            )
            if args.calibrate_only:
                return

        results = node.run_ablation(cases_klat, n_reps=args.reps,
                                    dry_run=args.dry_run)
        print(f"\nDone. {sum(r.passed for r in results)}/{len(results)} runs passed.")
    except CalibrationSafetyAbort as exc:
        raise SystemExit(str(exc))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
