#!/usr/bin/env python3
"""
Single-Arm Hardware Harness V3
================================

Single-robot version of hardware_harness_v3.py.  Runs the same 5-stage
test sequence (liveness → idle → precontact approach → hold → adaptive
press) using only robot1.  Designed for experiments where one arm
compresses against a fixed spring or rigid wall.

Key simplifications vs. the bilateral v3:
  - Only robot1 topics (joint_states, EE position, desired pose, wrench, etc.)
  - Single-arm contact detection (no bilateral confirmation)
  - Press direction derived from the single-arm force delta (or a fixed axis)
  - No inter-robot safety checks (separation_safe, x_gap_safe removed)
  - No mutual_contact_anchor or directional_press logic
  - Bag recording and publication-quality plots retained

Usage
-----
    python3 tools/hardware_harness_single_arm_v4.py
    python3 tools/hardware_harness_single_arm_v4.py --push-axis-height-z 0.085
    python3 tools/hardware_harness_single_arm_v4.py --use-sim-time

Environment variables:
  OMX_LOG_DIR                   – log root (default /tmp/variable_stiffness_logs)
  OMX_HARNESS_SINGLE_ARM_ROBOT  – robot namespace (default "robot1")
  OMX_HARNESS_SINGLE_ARM_PUSH_AXIS_HEIGHT_Z  – override push-axis Z
    OMX_HARNESS_SINGLE_ARM_PRESS_END_DISTANCE_M – bounded post-contact press depth
    OMX_LIVEPLOT_SINGLE_ARM_K_LAT – K_lat used by compression-time theory overlays

The single-arm shell wrappers expose that depth via either
--press-end-distance-m or --limited-depth-press-m, both of which map to the
same environment variable above. They also accept --k-lat to update the
compression-time K_lat used by the live plot and instability monitor.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point, Pose, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray

try:
    from tools.ablation_config import (
        SPRING_CAP_ROBOT1_CENTER_Y_M,
        SPRING_CAP_ROBOT1_CENTER_Z_M,
        SPRING_CAP_ROBOT1_HALF_SIZE_Y_M,
        SPRING_CAP_ROBOT1_HALF_SIZE_Z_M,
    )
except ImportError:
    from ablation_config import (
        SPRING_CAP_ROBOT1_CENTER_Y_M,
        SPRING_CAP_ROBOT1_CENTER_Z_M,
        SPRING_CAP_ROBOT1_HALF_SIZE_Y_M,
        SPRING_CAP_ROBOT1_HALF_SIZE_Z_M,
    )

try:
    from tools.aruco_alignment_utils_single import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
    )
except ImportError:
    from aruco_alignment_utils_single import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
    )

try:
    from tools.single_arm_trial_analysis import analyze_trial, write_analysis_json
except ImportError:
    from single_arm_trial_analysis import analyze_trial, write_analysis_json

# Publication-quality matplotlib style (same as plot_logs.py)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    matplotlib = None
    plt = None
    np = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

# Publication-quality style parameters
PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "font.size": 22,
    "font.weight": "bold",
    "axes.labelsize": 24,
    "axes.labelweight": "bold",
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.8,
    "xtick.labelsize": 22,
    "xtick.major.width": 1.6,
    "xtick.major.size": 7,
    "xtick.minor.width": 1.0,
    "xtick.minor.size": 4,
    "xtick.minor.visible": True,
    "ytick.labelsize": 22,
    "ytick.major.width": 1.6,
    "ytick.major.size": 7,
    "ytick.minor.width": 1.0,
    "ytick.minor.size": 4,
    "ytick.minor.visible": True,
    "legend.fontsize": 22,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.2",
    "lines.linewidth": 2.8,
    "lines.markersize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (20, 12),
}

# Color palette for single-arm plots
C_ARM = "#00429d"  # deep blue


class SingleArmBagRecorder:
    """Records a subset of topics to a ROS2 bag for the single-arm run."""

    def __init__(
        self,
        run_dir: str,
        record_all_topics: bool = True,
        selected_topics: Optional[Sequence[str]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.record_all_topics = record_all_topics
        self.selected_topics = list(selected_topics or [])
        self.window_index = 0
        self.process: Optional[subprocess.Popen[str]] = None
        self.log_handle = None
        self.active_output_dir: Optional[str] = None
        self.metadata_path = self.run_dir / "contact_windows.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "contact_bags").mkdir(parents=True, exist_ok=True)

    def is_active(self) -> bool:
        return self.process is not None

    def build_command(self, output_dir: str) -> List[str]:
        command = ["ros2", "bag", "record", "-o", output_dir]
        if self.record_all_topics:
            command.append("-a")
        else:
            command.extend(self.selected_topics)
        return command

    def _append_metadata(self, payload: Dict[str, Any]) -> None:
        with open(self.metadata_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def start(self, reason: str) -> Optional[Dict[str, Any]]:
        if self.process is not None:
            return None

        self.window_index += 1
        output_dir = self.run_dir / "contact_bags" / f"window_{self.window_index:03d}"
        log_path = self.run_dir / f"contact_window_{self.window_index:03d}.log"
        command = self.build_command(str(output_dir))

        payload = {
            "event": "start",
            "window_index": self.window_index,
            "reason": reason,
            "timestamp": time.time(),
            "output_dir": str(output_dir),
            "command": command,
        }

        self.log_handle = open(log_path, "w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            payload["event"] = "start_failed"
            payload["error"] = str(error)
            self._append_metadata(payload)
            self.log_handle.close()
            self.log_handle = None
            self.process = None
            return payload

        payload["pid"] = self.process.pid
        self.active_output_dir = str(output_dir)
        self._append_metadata(payload)
        return payload

    def stop(self, reason: str) -> Optional[Dict[str, Any]]:
        if self.process is None:
            return None

        process = self.process
        payload = {
            "event": "stop",
            "reason": reason,
            "timestamp": time.time(),
            "output_dir": self.active_output_dir,
            "pid": process.pid,
        }

        try:
            os.killpg(process.pid, signal.SIGINT)
            return_code = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait(timeout=5.0)
            payload["forced_kill"] = True
        except ProcessLookupError:
            return_code = process.poll()

        payload["return_code"] = return_code
        self._append_metadata(payload)

        if self.log_handle is not None:
            self.log_handle.close()

        self.log_handle = None
        self.process = None
        self.active_output_dir = None
        return payload


class ExternalCameraTrigger:
    """Writes START/STOP markers into a text file watched by a host-side
    high-FPS recorder.

    The recorder polls the file via mtime+size and parses lines like:
        START name=<window_name>
        STOP
        QUIT

    When ``trigger_path`` is None or empty, every method is a no-op.
    """

    def __init__(self, trigger_path: Optional[str], log_path: Optional[str] = None) -> None:
        self.trigger_path = trigger_path or None
        self.log_path = log_path
        self.window_index = 0
        self.active = False

    def is_enabled(self) -> bool:
        return bool(self.trigger_path)

    def _append_log(self, payload: Dict[str, Any]) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError:
            pass

    def _write(self, body: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.trigger_path:
            return None
        try:
            Path(self.trigger_path).write_text(body, encoding="utf-8")
        except OSError as error:
            payload["error"] = str(error)
            self._append_log(payload)
            return payload
        self._append_log(payload)
        return payload

    def start(self, name: str, reason: str) -> Optional[Dict[str, Any]]:
        if not self.trigger_path:
            return None
        self.window_index += 1
        self.active = True
        body = f"START name={name}\n"
        payload = {
            "event": "start",
            "window_index": self.window_index,
            "name": name,
            "reason": reason,
            "timestamp": time.time(),
            "trigger_path": self.trigger_path,
            "body": body.strip(),
        }
        return self._write(body, payload)

    def stop(self, reason: str) -> Optional[Dict[str, Any]]:
        if not self.trigger_path or not self.active:
            return None
        self.active = False
        body = "STOP\n"
        payload = {
            "event": "stop",
            "window_index": self.window_index,
            "reason": reason,
            "timestamp": time.time(),
            "trigger_path": self.trigger_path,
            "body": body.strip(),
        }
        return self._write(body, payload)


class SingleArmHardwareHarnessV4(Node):
    """Single-arm 5-stage hardware harness.

    Stages
    ------
    1. Liveness   – verify publishers, joint states, and pose topics are present
    2. Idle       – wait for low-velocity stabilization; capture baseline forces
    3. Precontact – step the arm toward the contact target until contact is detected
    4. Hold       – sustain contact for a fixed duration
    5. Press      – incrementally compress along the press direction
    """

    DEFAULT_SINGLE_ARM_TOPICS = [
        "/camera/color/image_raw",
        "/camera/color/camera_info",
        "/camera/aligned_depth_to_color/image_raw",
        "/camera/aligned_depth_to_color/camera_info",
    ]

    def __init__(
        self,
        robot_namespace: str = "robot1",
        push_axis_height_z: Optional[float] = None,
        use_sim_time: bool = False,
        plot_save_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            "single_arm_harness_v4",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)
            ],
        )
        self.robot_ns = robot_namespace
        self.use_sim_time = use_sim_time
        self.plot_save_dir = plot_save_dir

        # Publisher: waypoint command to the controller
        self.pub = self.create_publisher(
            PoseStamped,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/waypoint_command",
            10,
        )

        qos_fast = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscriptions: single-arm topics
        self.create_subscription(
            JointState,
            f"/{self.robot_ns}/joint_states",
            self._cb_js,
            qos_fast,
        )
        self.create_subscription(
            Point,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/end_effector_position",
            self._cb_ee,
            qos_fast,
        )
        self.create_subscription(
            Pose,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/cartesian_pose_desired",
            self._cb_des,
            qos_fast,
        )
        self.create_subscription(
            WrenchStamped,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/contact_wrench",
            self._cb_contact,
            qos_fast,
        )
        self.create_subscription(
            Bool,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/contact_valid",
            self._cb_contact_valid,
            qos_fast,
        )
        self.create_subscription(
            Bool,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/waypoint_active",
            self._cb_wp,
            qos_fast,
        )
        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.online_lateral_stiffness_pub = self.create_publisher(
            Float64,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/online_lateral_stiffness_npm",
            qos_latched,
        )
        self.online_lateral_probe_depth_pub = self.create_publisher(
            Float64,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/online_lateral_probe_depth_m",
            qos_latched,
        )
        self.online_lateral_pair_pub = self.create_publisher(
            Float64MultiArray,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/online_lateral_pair",
            qos_latched,
        )
        self.get_logger().info(
            f"[Robot {self.robot_ns}] summary publisher qos reliability={qos_latched.reliability} durability={qos_latched.durability} history={qos_latched.history} depth={qos_latched.depth}"
        )

        # Live ArUco alignment guidance is optional but enabled by default so
        # the run can adjust for small camera-observed drift while it is moving.
        self.enable_live_aruco_alignment = self._env_flag(
            "OMX_HARNESS_SINGLE_ARM_ENABLE_LIVE_ARUCO_ALIGNMENT",
            True,
        )
        self.live_aruco_y_trim_gain = float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_ARUCO_Y_TRIM_GAIN", "1.0")
        )
        self.live_aruco_z_trim_gain = float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_ARUCO_Z_TRIM_GAIN", "1.0")
        )
        self.live_aruco_y_trim_clamp_m = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_ARUCO_Y_TRIM_CLAMP_M", "0.015")
        ))
        self.live_aruco_z_trim_clamp_m = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_ARUCO_Z_TRIM_CLAMP_M", "0.015")
        ))
        self.spring_cap_center_y = SPRING_CAP_ROBOT1_CENTER_Y_M
        self.spring_cap_center_z = SPRING_CAP_ROBOT1_CENTER_Z_M
        self.spring_cap_half_y = SPRING_CAP_ROBOT1_HALF_SIZE_Y_M
        self.spring_cap_half_z = SPRING_CAP_ROBOT1_HALF_SIZE_Z_M
        self.live_aruco_y_trim_clamp_m = min(self.live_aruco_y_trim_clamp_m, self.spring_cap_half_y)
        self.live_aruco_z_trim_clamp_m = min(self.live_aruco_z_trim_clamp_m, self.spring_cap_half_z)
        if self.enable_live_aruco_alignment:
            self.create_subscription(Bool, ARUCO_ALIGNMENT_VALID_TOPIC, self._cb_aruco_alignment_valid, qos_fast)
            self.create_subscription(Float64, ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC, self._cb_aruco_alignment_center_y_error, qos_fast)
            self.create_subscription(Float64, ARUCO_ALIGNMENT_Z_TRIM_TOPIC, self._cb_aruco_z_trim, qos_fast)

        # State
        self.js: Optional[JointState] = None
        self.ee: Optional[Point] = None
        self.des: Optional[Pose] = None
        self.contact_fx_mag = float("nan")
        self.contact_valid = False
        self.wp_active: Optional[bool] = None

        self.contact_force_vec: Optional[Tuple[float, float, float]] = None
        self.contact_torque_vec: Optional[Tuple[float, float, float]] = None
        self.baseline_force_vec: Optional[Tuple[float, float, float]] = None
        self.precontact_baseline_fx = float("nan")

        self.aruco_alignment_valid = False
        self.aruco_alignment_center_y_error = float("nan")
        self.aruco_z_trim = float("nan")
        self.alignment_y_trim = 0.0
        self.alignment_z_trim = 0.0

        # Phase tracking
        self.current_phase = "init"
        self.current_offset_x = float("nan")
        self.current_offset_y = 0.0
        self.current_offset_z = 0.0
        self.current_press_offset_x = 0.0
        self.current_press_offset_y = 0.0
        self.current_press_offset_z = 0.0
        self.current_contact_mode = "none"
        self.current_press_distance = 0.0
        self.contact_active = False
        self.current_press_dir: Optional[Tuple[float, float, float]] = None
        # Idle checks should ignore the gripper because bring-up intentionally
        # closes it before the arm settles at the start pose.
        self.arm_joint_names = ("joint1", "joint2", "joint3", "joint4")

        # Push-axis height override
        self.push_axis_height_z = self._resolve_push_axis_height_z(push_axis_height_z)

        # Tuning parameters (single-arm defaults)
        # Increased idle_vel_limit from 1.0 to 2.0 to accommodate singularity
        # recovery transients (sigma_min oscillation causes 2-3 rad/s spikes)
        self.idle_vel_limit = 2.0
        self.move_vel_limit = 1.5
        self.hold_vel_limit = 0.8
        self.command_repeats = 2
        self.command_dt = 0.05
        self.command_sample_delay = 0.60
        self.waypoint_timeout = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_WAYPOINT_TIMEOUT_S", "10.0")
        ))
        self.align_wrist_with_push_axis = self._env_flag(
            "OMX_HARNESS_SINGLE_ARM_ALIGN_WRIST_WITH_PUSH_AXIS",
            False,
        )
        self.align_wrist_backoff_m = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_ALIGN_WRIST_BACKOFF_M", "0.010")
        ))

        # Precontact approach parameters
        # Robot1 home x is read from controller YAML (start_position[0]).
        # Precontact offsets must be large enough to bridge the gap to the spring.
        self.precontact_start_x_offset = 0.12  # Start 12cm from home (safe approach start)
        self.precontact_end_x_offset = 0.30    # Approach depth (must reach spring assembly)
        self.min_offset_from_base = 0.16       # Minimum distance from robot base (16cm, matches controller MIN_X floor)
        self.press_step = 0.002                # 2mm per step (slightly larger for longer travel)
        self.max_precontact_iterations = 200   # 350mm / 2mm = 175 steps minimum

        # Press parameters
        self.press_end_distance = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_PRESS_END_DISTANCE_M", "0.050")
        ))
        self.max_press_distance = self.precontact_end_x_offset + self.press_end_distance
        self.deep_press_step = 0.0005         # 0.5mm quasistatic post-contact increment
        self.max_side_extra_press = 0.0060
        self.side_press_step = 0.0010

        # Contact detection thresholds
        self.contact_force_enter = 0.60
        self.contact_force_delta_enter = 0.10
        self.contact_force_threshold_cap = 0.60
        self.contact_threshold_override: Optional[float] = None
        self.use_precontact_baseline: bool = True

        # Abort limits
        self.force_diff_abort = 4.0
        self.max_force_mag_abort = 8.0
        self.instability_mode = False
        self.stage5_supported_force_min = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_SUPPORTED_FORCE_MIN_N", "0.20")
        ))
        self.stage5_release_force_max = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_RELEASE_FORCE_MAX_N", "0.12")
        ))
        self.stage5_low_support_limit = int(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_LOW_SUPPORT_LIMIT", "2")
        )

        # Hold / press timing
        self.hold_duration = 6.0
        self.stage5_hold_duration = 6.0
        self.command_refresh_interval = 1.50
        self.enable_deep_press = self._env_flag("OMX_HARNESS_SINGLE_ARM_ENABLE_DEEP_PRESS", False)
        self.stage5_axial_settle_timeout = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_AXIAL_SETTLE_TIMEOUT_S", "15.0")
        ))
        self.stage5_axial_settle_position_tol = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_AXIAL_SETTLE_POSITION_TOL_M", "0.0015")
        ))
        self.stage5_axial_settle_velocity_limit = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_AXIAL_SETTLE_VEL_LIMIT_RAD_S", "0.05")
        ))
        self.stage5_axial_settle_required_samples = max(1, int(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_AXIAL_SETTLE_REQUIRED_SAMPLES", "6")
        ))
        self.stage5_lateral_probe_amplitude_m = abs(float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_LATERAL_PROBE_AMPLITUDE_M", "0.0015")
        ))
        self.stage5_lateral_probe_cycles = max(1.0, float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_LATERAL_PROBE_CYCLES", "1.0")
        ))
        self.stage5_lateral_probe_points = max(4, int(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_LATERAL_PROBE_POINTS", "24")
        ))
        self.stage5_lateral_probe_period_s = max(0.1, float(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_STAGE5_LATERAL_PROBE_PERIOD_S", "1.2")
        ))

        self.beam_boundary_condition = os.environ.get(
            "OMX_HARNESS_SINGLE_ARM_BEAM_BOUNDARY_CONDITION",
            "cantilever",
        )
        self.beam_effective_length_m = float(os.environ.get(
            "OMX_HARNESS_SINGLE_ARM_BEAM_EFFECTIVE_LENGTH_M",
            "0.128",
        ))
        self.aruco_trace_path = os.environ.get("OMX_HARNESS_SINGLE_ARM_ARUCO_TRACE_CSV")
        if not self.aruco_trace_path:
            self.aruco_trace_path = os.environ.get("OMX_HARNESS_SINGLE_ARM_ARUCO_TRACE_JSONL")

        self.online_lateral_stiffness_npm = float("nan")
        self.online_lateral_eigenvalue_npm = float("nan")
        self.online_lateral_probe_depth_m = float("nan")
        self.current_probe_index = -1
        self.current_probe_offset_m = float("nan")
        self.current_probe_force_n = float("nan")
        self.probe_lateral_offsets: List[float] = []
        self.probe_lateral_forces: List[float] = []

        # Logging
        self.log_root = os.environ.get("OMX_LOG_DIR", "/tmp/variable_stiffness_logs")
        self.run_ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_root, f"single_arm_harness_v4_{self.run_ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "single_arm_harness_v4_snapshot.csv")
        self.sync_csv_path = os.path.join(self.run_dir, "single_arm_harness_v4_sync_steps.csv")
        self.results_path = os.path.join(self.run_dir, "single_arm_harness_v4_results.json")
        self.manifest_path = os.path.join(self.run_dir, "single_arm_harness_v4_manifest.json")

        # Log columns
        self.log_columns = [
            "timestamp",
            "phase",
            "offset_x",
            "offset_y",
            "offset_z",
            "press_offset_x",
            "press_offset_y",
            "press_offset_z",
            "press_distance",
            "contact_mode",
            "contact_active",
            "press_dir_x",
            "press_dir_y",
            "press_dir_z",
            "projected_force",
            "probe_lateral_offset_m",
            "probe_lateral_force_n",
            "probe_sample_index",
            "online_lateral_stiffness_npm",
            "online_lateral_eigenvalue_npm",
            "bag_recording_active",
            "arm_max_vel",
            "ee_x",
            "ee_y",
            "ee_z",
            "desired_x",
            "desired_y",
            "desired_z",
            "contact_fx",
            "contact_fy",
            "contact_fz",
            "contact_tx",
            "contact_ty",
            "contact_tz",
            "contact_force_norm",
            "contact_fx_mag",
            "baseline_fx",
            "contact_threshold",
            "contact_valid",
        ]
        self.sync_columns = [
            "timestamp",
            "phase",
            "command_distance",
            "offset_x",
            "offset_y",
            "offset_z",
            "press_offset_x",
            "press_offset_y",
            "press_offset_z",
            "press_distance",
            "contact_mode",
            "contact_active",
            "press_dir_x",
            "press_dir_y",
            "press_dir_z",
            "projected_force",
            "probe_lateral_offset_m",
            "probe_lateral_force_n",
            "probe_sample_index",
            "online_lateral_stiffness_npm",
            "online_lateral_eigenvalue_npm",
            "bag_recording_active",
            "arm_max_vel",
            "ee_x",
            "ee_y",
            "ee_z",
            "desired_x",
            "desired_y",
            "desired_z",
            "contact_fx",
            "contact_fy",
            "contact_fz",
            "contact_tx",
            "contact_ty",
            "contact_tz",
            "contact_force_norm",
            "contact_fx_mag",
            "baseline_fx",
            "contact_threshold",
            "contact_valid",
            "aruco_alignment_valid",
            "aruco_alignment_center_y_error_m",
            "aruco_z_trim_m",
            "alignment_y_trim_m",
            "alignment_z_trim_m",
        ]

        # Bag recorder
        self.record_all_topics = self._env_flag("OMX_HARNESS_SINGLE_ARM_RECORD_ALL_TOPICS", True)
        self.selected_topics = self._default_single_arm_topics()
        self.bag_recorder = SingleArmBagRecorder(
            self.run_dir,
            record_all_topics=self.record_all_topics,
            selected_topics=self.selected_topics,
        )

        # External camera trigger
        self.external_camera_trigger = ExternalCameraTrigger(
            os.environ.get("OMX_HARNESS_SINGLE_ARM_CAMERA_TRIGGER_FILE"),
            os.path.join(self.run_dir, "external_camera_trigger.log"),
        )

        # Write CSV headers and manifest
        self.write_csv_header(self.csv_path, self.log_columns)
        self.write_csv_header(self.sync_csv_path, self.sync_columns)
        self.write_manifest()

        # Timer for periodic logging
        self.create_timer(0.05, self._log_snapshot_row)

    # -----------------------------------------------------------------------
    # Initialization helpers
    # -----------------------------------------------------------------------

    def _resolve_push_axis_height_z(self, push_axis_height_z: Optional[float]) -> Optional[float]:
        if push_axis_height_z is None:
            raw_value = os.environ.get("OMX_HARNESS_SINGLE_ARM_PUSH_AXIS_HEIGHT_Z")
            if raw_value is None or not raw_value.strip():
                return None
            push_axis_height_z = float(raw_value)

        push_axis_height_z = float(push_axis_height_z)
        if not math.isfinite(push_axis_height_z):
            raise ValueError("push_axis_height_z must be finite")
        return push_axis_height_z

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _default_single_arm_topics(self) -> List[str]:
        camera_topics = os.environ.get("OMX_HARNESS_SINGLE_ARM_CAMERA_TOPICS")
        if camera_topics:
            realsense_topics = [topic.strip() for topic in camera_topics.split(",") if topic.strip()]
        else:
            realsense_topics = list(self.DEFAULT_SINGLE_ARM_TOPICS)

        return [
            f"/{self.robot_ns}/joint_states",
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/end_effector_position",
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/cartesian_pose_desired",
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/contact_wrench",
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/contact_valid",
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/waypoint_active",
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/waypoint_command",
            "/tf",
            "/tf_static",
            *realsense_topics,
        ]

    def write_manifest(self) -> None:
        payload = {
            "run_ts": self.run_ts,
            "run_dir": self.run_dir,
            "robot_namespace": self.robot_ns,
            "record_all_topics": self.record_all_topics,
            "selected_topics": self.selected_topics,
            "push_axis_height_z": self.push_axis_height_z,
            "press_end_distance": self.press_end_distance,
            "max_press_distance": self.max_press_distance,
            "beam_boundary_condition": self.beam_boundary_condition,
            "beam_effective_length_m": self.beam_effective_length_m,
            "aruco_trace_path": self.aruco_trace_path,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def _cb_js(self, msg: JointState) -> None:
        self.js = msg

    def _cb_ee(self, msg: Point) -> None:
        self.ee = msg

    def _cb_des(self, msg: Pose) -> None:
        self.des = msg

    def _cb_contact(self, msg: WrenchStamped) -> None:
        self.contact_fx_mag = abs(msg.wrench.force.x)
        self.contact_force_vec = (
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        )
        self.contact_torque_vec = (
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        )

    def _cb_contact_valid(self, msg: Bool) -> None:
        self.contact_valid = bool(msg.data)

    def _cb_wp(self, msg: Bool) -> None:
        self.wp_active = bool(msg.data)

    def _publish_online_lateral_stiffness(self) -> None:
        msg = Float64()
        msg.data = float(self.online_lateral_stiffness_npm)
        self.online_lateral_stiffness_pub.publish(msg)

    def _publish_online_lateral_probe_depth(self) -> None:
        msg = Float64()
        msg.data = float(self.online_lateral_probe_depth_m)
        publisher = getattr(self, "online_lateral_probe_depth_pub", None)
        if publisher is not None:
            publisher.publish(msg)

    def _publish_online_lateral_pair(self) -> None:
        msg = Float64MultiArray()
        msg.data = [float(self.online_lateral_probe_depth_m), float(self.online_lateral_stiffness_npm)]
        publisher = getattr(self, "online_lateral_pair_pub", None)
        if publisher is not None:
            publisher.publish(msg)

    def _cb_aruco_alignment_valid(self, msg: Bool) -> None:
        self.aruco_alignment_valid = bool(msg.data)
        self._refresh_live_aruco_alignment_trim()

    def _cb_aruco_alignment_center_y_error(self, msg: Float64) -> None:
        self.aruco_alignment_center_y_error = float(msg.data)
        self._refresh_live_aruco_alignment_trim()

    def _cb_aruco_z_trim(self, msg: Float64) -> None:
        self.aruco_z_trim = float(msg.data)
        self._refresh_live_aruco_alignment_trim()

    # -----------------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------------

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_for(self, duration: float, step: float = 0.05) -> None:
        end_t = self.now_s() + duration
        while self.now_s() < end_t:
            rclpy.spin_once(self, timeout_sec=step)

    @staticmethod
    def finite(x: float) -> bool:
        return not math.isnan(x) and not math.isinf(x)

    @staticmethod
    def vector_norm(vector: Optional[Tuple[float, float, float]]) -> float:
        if vector is None:
            return float("nan")
        return math.sqrt(sum(component * component for component in vector))

    def normalize_vector(self, vector: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        norm = self.vector_norm(vector)
        if vector is None or not self.finite(norm) or norm < 1e-9:
            return None
        return (vector[0] / norm, vector[1] / norm, vector[2] / norm)

    @staticmethod
    def dot_product(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _normalized_joint_name(name: str) -> str:
        return name.rsplit("/", 1)[-1].rsplit(":", 1)[-1]

    def max_abs_velocity(
        self,
        js: Optional[JointState],
        joint_names: Optional[Sequence[str]] = None,
    ) -> Optional[float]:
        if js is None or js.velocity is None or len(js.velocity) == 0:
            return None
        if joint_names is None or not js.name:
            values = [abs(v) for v in js.velocity if self.finite(v)]
        else:
            joint_name_set = {self._normalized_joint_name(name) for name in joint_names}
            values = [
                abs(velocity)
                for name, velocity in zip(js.name, js.velocity)
                if self._normalized_joint_name(name) in joint_name_set and self.finite(velocity)
            ]
        return max(values) if values else None

    def check_limits(
        self,
        js: Optional[JointState],
        limit: float,
        joint_names: Optional[Sequence[str]] = None,
    ) -> bool:
        vmax = self.max_abs_velocity(js, joint_names)
        return vmax is not None and vmax < limit

    def configured_push_axis_height(self, fallback_z: float) -> float:
        target_z = fallback_z if self.push_axis_height_z is None else self.push_axis_height_z
        return max(
            self.spring_cap_center_z - self.spring_cap_half_z,
            min(self.spring_cap_center_z + self.spring_cap_half_z, target_z),
        )

    def _clamp_to_spring_cap_bounds(self, y: float, z: float) -> Tuple[float, float]:
        clamped_y = max(self.spring_cap_center_y - self.spring_cap_half_y,
                        min(self.spring_cap_center_y + self.spring_cap_half_y, y))
        clamped_z = max(self.spring_cap_center_z - self.spring_cap_half_z,
                        min(self.spring_cap_center_z + self.spring_cap_half_z, z))
        return clamped_y, clamped_z

    def _clamp_push_approach_target(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
    ) -> Tuple[float, float, float]:
        clamped_y, clamped_z = self._clamp_to_spring_cap_bounds(target_y, target_z)
        return target_x, clamped_y, clamped_z

    def _refresh_live_aruco_alignment_trim(self) -> None:
        if not self.enable_live_aruco_alignment or not self.aruco_alignment_valid:
            self.alignment_y_trim = 0.0
            self.alignment_z_trim = 0.0
            return

        y_error = self.aruco_alignment_center_y_error
        if self.finite(y_error):
            y_trim = -self.live_aruco_y_trim_gain * y_error
            y_trim = max(-self.live_aruco_y_trim_clamp_m, min(self.live_aruco_y_trim_clamp_m, y_trim))
            y_trim = max(-self.spring_cap_half_y, min(self.spring_cap_half_y, y_trim))
        else:
            y_trim = 0.0

        z_error = self.aruco_z_trim
        if self.finite(z_error):
            z_trim = self.live_aruco_z_trim_gain * z_error
            z_trim = max(-self.live_aruco_z_trim_clamp_m, min(self.live_aruco_z_trim_clamp_m, z_trim))
            z_trim = max(-self.spring_cap_half_z, min(self.spring_cap_half_z, z_trim))
        else:
            z_trim = 0.0

        self.alignment_y_trim = y_trim
        self.alignment_z_trim = z_trim

    # -----------------------------------------------------------------------
    # Contact detection
    # -----------------------------------------------------------------------

    def capture_precontact_baseline(self) -> None:
        self.precontact_baseline_fx = self.contact_fx_mag if self.finite(self.contact_fx_mag) else 0.0

    def contact_threshold(self) -> float:
        if self.contact_threshold_override is not None:
            return float(self.contact_threshold_override)

        baseline = self.precontact_baseline_fx
        if not self.finite(baseline):
            baseline = 0.0
        if not self.use_precontact_baseline:
            baseline = 0.0

        return min(
            self.contact_force_threshold_cap,
            max(self.contact_force_enter, baseline + self.contact_force_delta_enter),
        )

    def contact_detected(self) -> bool:
        return (
            self.contact_valid
            and self.finite(self.contact_fx_mag)
            and self.contact_fx_mag >= self.contact_threshold()
        )

    def contact_force_delta(self) -> Optional[Tuple[float, float, float]]:
        """Force minus idle baseline; falls back to raw force if baseline not captured."""
        current = self.contact_force_vec
        baseline = self.baseline_force_vec
        if current is None:
            return None
        if baseline is None:
            return current
        return (
            current[0] - baseline[0],
            current[1] - baseline[1],
            current[2] - baseline[2],
        )

    def projected_force(self, direction: Optional[Tuple[float, float, float]]) -> float:
        vector = self.contact_force_vec
        if vector is None or direction is None:
            return float("nan")
        return self.dot_product(vector, direction)

    def compute_press_direction(self) -> Optional[Tuple[float, float, float]]:
        """Compute the press direction from the contact force delta.

        For a single arm pressing against a fixed target, the press direction
        is the direction of the force delta (force - baseline).  If no baseline
        is available, fall back to the raw force direction.
        """
        force_delta = self.contact_force_delta()
        if force_delta is None:
            return None

        # Zero out Z component — press direction should be purely horizontal
        direction = (force_delta[0], force_delta[1], 0.0)
        direction = self.normalize_vector(direction)
        if direction is None:
            # Fallback: use raw force direction
            if self.contact_force_vec is None:
                return None
            direction = (self.contact_force_vec[0], self.contact_force_vec[1], 0.0)
            direction = self.normalize_vector(direction)

        return direction

    # -----------------------------------------------------------------------
    # Command publishing
    # -----------------------------------------------------------------------

    def _push_axis_orientation_quaternion(
        self,
        direction: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """Return the pose quaternion for the current orientation mode.

        Default mode stays horizontal. Optional push-axis alignment yaws the
        wrist so the gripper x-axis follows the commanded horizontal press
        direction.
        """
        if not self.align_wrist_with_push_axis or direction is None:
            return (0.0, 0.0, 0.0, 1.0)

        direction_xy = self.normalize_vector((direction[0], direction[1], 0.0))
        if direction_xy is None:
            return (0.0, 0.0, 0.0, 1.0)

        yaw = math.atan2(direction_xy[1], direction_xy[0])
        half_yaw = yaw * 0.5
        return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))

    def _push_axis_backoff_target(
        self,
        target: Tuple[float, float, float],
        direction: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """Optionally retract slightly along the push axis to preserve tip height.

        The backoff is only used in the optional wrist-aligned mode so the
        gripper can be oriented head-on without consuming the contact-height
        budget at the tip.
        """
        if not self.align_wrist_with_push_axis or direction is None:
            return target

        direction_xy = self.normalize_vector((direction[0], direction[1], 0.0))
        if direction_xy is None:
            return target

        backoff = self.align_wrist_backoff_m
        return (
            target[0] - backoff * direction_xy[0],
            target[1] - backoff * direction_xy[1],
            target[2],
        )

    def make_absolute_pose(
        self,
        x: float,
        y: float,
        z: float,
        direction: Optional[Tuple[float, float, float]] = None,
    ) -> PoseStamped:
        """Create a pose with position (x, y, z) and configurable orientation.

        By default the gripper remains horizontal for stable head-on
        compression. When push-axis alignment is enabled, the yaw follows the
        horizontal press direction.
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "absolute"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        qx, qy, qz, qw = self._push_axis_orientation_quaternion(direction)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def publish_absolute_target(
        self,
        target: Tuple[float, float, float],
        repeats: Optional[int] = None,
        dt: Optional[float] = None,
        direction: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        repeat_count = self.command_repeats if repeats is None else repeats
        step_dt = self.command_dt if dt is None else dt
        self._refresh_live_aruco_alignment_trim()
        adjusted_y = target[1] + self.alignment_y_trim
        adjusted_z = target[2] + self.alignment_z_trim
        adjusted_target = self._clamp_push_approach_target(target[0], adjusted_y, adjusted_z)
        adjusted_y, adjusted_z = adjusted_target[1], adjusted_target[2]
        self.current_offset_y = adjusted_y
        self.current_offset_z = adjusted_z
        self.current_press_offset_y = adjusted_y
        self.current_press_offset_z = adjusted_z

        adjusted_target = self._push_axis_backoff_target(adjusted_target, direction)
        msg = self.make_absolute_pose(*adjusted_target, direction=direction)

        for _ in range(repeat_count):
            stamp = self.get_clock().now().to_msg()
            msg.header.stamp = stamp
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=step_dt)

    def publish_offset(
        self,
        x: float,
        y: float,
        z: float,
        repeats: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> None:
        """Publish an offset-mode waypoint (frame_id='offset')."""
        repeat_count = self.command_repeats if repeats is None else repeats
        step_dt = self.command_dt if dt is None else dt

        self.current_offset_x = x
        self.current_offset_y = y
        self.current_offset_z = z
        self.current_press_offset_x = x
        self.current_press_offset_y = y
        self.current_press_offset_z = z

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "offset"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0

        for _ in range(repeat_count):
            stamp = self.get_clock().now().to_msg()
            msg.header.stamp = stamp
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=step_dt)

    # -----------------------------------------------------------------------
    # Precontact target computation
    # -----------------------------------------------------------------------

    def compute_absolute_precontact_target(
        self,
        home: Tuple[float, float, float],
        x_offset: float,
    ) -> Dict[str, float]:
        """Compute precontact target with z locked to current EE height.

        During precontact approach, both position and orientation are held constant:
        - z is locked to CURRENT EE height (not push_axis_height) to avoid sudden vertical drops
        - gripper stays perfectly horizontal (identity quaternion) for stable alignment

        CRITICAL: Z must NEVER exceed push_axis_height_z during precontact.
        Clamp to min(current EE Z, push_axis_height_z).
        """
        # Clamp Z: never above push_axis_height_z, but never below current EE Z
        pah = self.push_axis_height_z if self.push_axis_height_z is not None else home[2]
        target_x = max(home[0] + x_offset, self.min_offset_from_base)
        target_z = self.configured_push_axis_height(min(home[2], pah))
        target_x, target_y, target_z = self._clamp_push_approach_target(target_x, home[1], target_z)
        return {
            "offset_x": target_x - home[0],
            "offset_y": target_y - home[1],
            "offset_z": target_z - home[2],
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
        }

    def apply_precontact_target(self, target: Dict[str, float], contact_mode: str) -> None:
        orientation_dir: Optional[Tuple[float, float, float]] = None
        if self.align_wrist_with_push_axis:
            orientation_dir = (1.0 if target["offset_x"] >= 0.0 else -1.0, 0.0, 0.0)

        self.current_contact_mode = contact_mode
        self.current_offset_x = target["offset_x"]
        self.current_offset_y = target["offset_y"]
        self.current_offset_z = target["offset_z"]
        self.current_press_offset_x = target["offset_x"]
        self.current_press_offset_y = target["offset_y"]
        self.current_press_offset_z = target["offset_z"]

        adjusted_target = self._push_axis_backoff_target(
            (target["target_x"], target["target_y"], target["target_z"]),
            orientation_dir,
        )
        self.publish_absolute_target(
            adjusted_target,
            repeats=self.command_repeats,
            direction=orientation_dir,
        )

    # -----------------------------------------------------------------------
    # Directional press target
    # -----------------------------------------------------------------------

    def compute_absolute_directional_press_target(
        self,
        distance: float,
        direction: Tuple[float, float, float],
        anchor: Tuple[float, float, float],
    ) -> Dict[str, float]:
        """Compute directional press target with z locked to anchor height.

        During contact and compression phases:
        - z is HARD-LOCKED to anchor height (push_axis_height) to ensure consistent spring contact
        - offset_z is forced to 0.0 to reject spurious Z from force noise
        - gripper stays perfectly horizontal (via make_absolute_pose -> identity quaternion)
        """
        target_x = max(anchor[0] + distance * direction[0], self.min_offset_from_base)
        target_y = anchor[1] + distance * direction[1]
        target_z = self.configured_push_axis_height(anchor[2])  # Lock Z to anchor height within cap bounds
        target_x, target_y, target_z = self._clamp_push_approach_target(target_x, target_y, target_z)

        return {
            "offset_x": distance * direction[0],
            "offset_y": distance * direction[1],
            "offset_z": 0.0,  # Force Z offset to 0
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
        }

    def apply_directional_press_target(
        self,
        distance: float,
        direction: Tuple[float, float, float],
        anchor: Tuple[float, float, float],
        contact_mode: str,
    ) -> Dict[str, float]:
        target = self.compute_absolute_directional_press_target(distance, direction, anchor)
        self.current_contact_mode = contact_mode
        self.current_offset_x = target["offset_x"]
        self.current_offset_y = target["offset_y"]
        self.current_offset_z = target["offset_z"]
        self.current_press_offset_x = target["offset_x"]
        self.current_press_offset_y = target["offset_y"]
        self.current_press_offset_z = target["offset_z"]

        self.publish_absolute_target(
            (target["target_x"], target["target_y"], target["target_z"]),
            repeats=self.command_repeats,
            direction=direction,
        )
        return target

    def _stage5_update_support_state(
        self,
        projected_force: float,
        support_seen: bool,
        low_support_count: int,
    ) -> Tuple[bool, int, bool]:
        """Track whether the spring is still providing meaningful support.

        Returns (support_seen, low_support_count, support_lost).
        """
        if self.finite(projected_force) and abs(projected_force) >= self.stage5_supported_force_min:
            return True, 0, False

        if support_seen and self.finite(projected_force) and abs(projected_force) <= self.stage5_release_force_max:
            low_support_count += 1
            if low_support_count >= self.stage5_low_support_limit:
                return support_seen, low_support_count, True

        return support_seen, low_support_count, False

    @staticmethod
    def _fit_linear_slope(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float]:
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return float("nan"), float("nan")
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(y_values) / len(y_values)
        denom = sum((x - x_mean) ** 2 for x in x_values)
        if denom <= 1e-12:
            return float("nan"), float("nan")
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / denom
        intercept = y_mean - slope * x_mean
        return slope, intercept

    def _update_online_lateral_stiffness(self) -> None:
        slope, _intercept = self._fit_linear_slope(self.probe_lateral_offsets, self.probe_lateral_forces)
        if self.finite(slope):
            self.online_lateral_stiffness_npm = abs(slope)
            self.online_lateral_eigenvalue_npm = self.online_lateral_stiffness_npm
            self._publish_online_lateral_pair()
        else:
            self.online_lateral_stiffness_npm = float("nan")
            self.online_lateral_eigenvalue_npm = float("nan")
        self._publish_online_lateral_stiffness()

    def _wait_for_axial_settle(
        self,
        target_pose: Tuple[float, float, float],
        direction: Optional[Tuple[float, float, float]] = None,
        timeout_s: Optional[float] = None,
    ) -> bool:
        timeout = self.stage5_axial_settle_timeout if timeout_s is None else timeout_s
        end_t = self.now_s() + timeout
        stable_count = 0

        while self.now_s() < end_t:
            self.spin_for(0.05)
            if not self.contact_detected():
                return False

            if self.ee is None:
                continue

            ee_err = math.sqrt(
                (self.ee.x - target_pose[0]) ** 2 +
                (self.ee.y - target_pose[1]) ** 2 +
                (self.ee.z - target_pose[2]) ** 2
            )
            arm_vel = self.max_abs_velocity(self.js, self.arm_joint_names)
            if (
                ee_err <= self.stage5_axial_settle_position_tol
                and arm_vel is not None
                and arm_vel <= self.stage5_axial_settle_velocity_limit
            ):
                stable_count += 1
                if stable_count >= self.stage5_axial_settle_required_samples:
                    if direction is not None:
                        self.get_logger().info(
                            "Stage 5: axial settle complete at depth=%.6f (ee_err=%.4f, vel=%.4f)"
                            % (self.current_press_distance, ee_err, arm_vel)
                        )
                    return True
            else:
                stable_count = 0

        return False

    def _run_lateral_microperturbation(
        self,
        settled_target: Tuple[float, float, float],
        direction: Optional[Tuple[float, float, float]],
        stage_info: List[Dict[str, Any]],
        command_distance: float,
    ) -> bool:
        if not self.contact_detected():
            return False

        points = self.stage5_lateral_probe_points
        step_dt = self.stage5_lateral_probe_period_s / float(points)
        probe_depth_offset = float(getattr(self, "precontact_end_x_offset", 0.0))
        if not hasattr(self, "online_lateral_stiffness_npm"):
            self.online_lateral_stiffness_npm = float("nan")
        if not hasattr(self, "online_lateral_eigenvalue_npm"):
            self.online_lateral_eigenvalue_npm = float("nan")
        self.probe_lateral_offsets = []
        self.probe_lateral_forces = []
        for idx in range(points):
            phase = (2.0 * math.pi * self.stage5_lateral_probe_cycles * idx) / float(points)
            y_offset = self.stage5_lateral_probe_amplitude_m * math.sin(phase)
            probe_target = (
                settled_target[0],
                settled_target[1] + y_offset,
                settled_target[2],
            )
            probe_depth_m = max(0.0, float(command_distance) - probe_depth_offset)
            self.online_lateral_probe_depth_m = probe_depth_m
            self._publish_online_lateral_probe_depth()
            self.current_contact_mode = "lateral_probe"
            self.current_probe_index = idx
            self.current_probe_offset_m = y_offset
            self.publish_absolute_target(probe_target, repeats=1, dt=self.command_dt, direction=None)
            self.spin_for(step_dt)

            if not self.contact_detected():
                return False

            contact_force_vec = getattr(self, "contact_force_vec", None)
            lateral_force = float(contact_force_vec[1]) if contact_force_vec is not None else float("nan")
            self.current_probe_force_n = lateral_force
            if self.finite(y_offset) and self.finite(lateral_force):
                self.probe_lateral_offsets.append(float(y_offset))
                self.probe_lateral_forces.append(float(lateral_force))
                self._update_online_lateral_stiffness()
            snap = self.snapshot()
            snap["command_distance"] = command_distance
            snap["probe_lateral_offset_m"] = y_offset
            snap["probe_lateral_force_n"] = lateral_force
            snap["probe_sample_index"] = idx
            snap["online_lateral_probe_depth_m"] = probe_depth_m
            snap["online_lateral_stiffness_npm"] = self.online_lateral_stiffness_npm
            snap["online_lateral_eigenvalue_npm"] = self.online_lateral_eigenvalue_npm
            stage_info.append(snap)
            self.log_sync_step(command_distance)

        if self.finite(self.online_lateral_stiffness_npm):
            self.get_logger().info(
                "Stage 5 probe summary: depth=%.6f samples=%d K_lat=%.6f"
                % (self.online_lateral_probe_depth_m, len(self.probe_lateral_offsets), self.online_lateral_stiffness_npm)
            )
        else:
            self.get_logger().warn(
                "Stage 5 probe summary: depth=%.6f samples=%d (no finite stiffness estimate)"
                % (self.online_lateral_probe_depth_m, len(self.probe_lateral_offsets))
            )

        self.current_contact_mode = "axial_settled"
        self.publish_absolute_target(settled_target, repeats=self.command_repeats, dt=self.command_dt, direction=direction)
        self.spin_for(self.command_sample_delay)
        return self.contact_detected()

    # -----------------------------------------------------------------------
    # Snapshot / logging
    # -----------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        arm_max_vel = self.max_abs_velocity(self.js, self.arm_joint_names)
        ee_x = self.ee.x if self.ee is not None else float("nan")
        ee_y = self.ee.y if self.ee is not None else float("nan")
        ee_z = self.ee.z if self.ee is not None else float("nan")
        desired_x = self.des.position.x if self.des is not None else float("nan")
        desired_y = self.des.position.y if self.des is not None else float("nan")
        desired_z = self.des.position.z if self.des is not None else float("nan")

        direction = self.current_press_dir
        projected_force = self.projected_force(direction)

        force = self.contact_force_vec or (float("nan"), float("nan"), float("nan"))
        torque = self.contact_torque_vec or (float("nan"), float("nan"), float("nan"))

        return {
            "timestamp": self.now_s(),
            "phase": self.current_phase,
            "offset_x": self.current_offset_x,
            "offset_y": self.current_offset_y,
            "offset_z": self.current_offset_z,
            "press_offset_x": self.current_press_offset_x,
            "press_offset_y": self.current_press_offset_y,
            "press_offset_z": self.current_press_offset_z,
            "press_distance": self.current_press_distance,
            "contact_mode": self.current_contact_mode,
            "contact_active": float(self.contact_active),
            "press_dir_x": direction[0] if direction is not None else float("nan"),
            "press_dir_y": direction[1] if direction is not None else float("nan"),
            "press_dir_z": direction[2] if direction is not None else float("nan"),
            "projected_force": projected_force,
            "probe_lateral_offset_m": self.current_probe_offset_m,
            "probe_lateral_force_n": self.current_probe_force_n,
            "probe_sample_index": self.current_probe_index,
            "online_lateral_stiffness_npm": self.online_lateral_stiffness_npm,
            "online_lateral_eigenvalue_npm": self.online_lateral_eigenvalue_npm,
            "bag_recording_active": float(self.bag_recorder.is_active()),
            "arm_max_vel": arm_max_vel,
            "ee_x": ee_x,
            "ee_y": ee_y,
            "ee_z": ee_z,
            "desired_x": desired_x,
            "desired_y": desired_y,
            "desired_z": desired_z,
            "contact_fx": force[0],
            "contact_fy": force[1],
            "contact_fz": force[2],
            "contact_tx": torque[0],
            "contact_ty": torque[1],
            "contact_tz": torque[2],
            "contact_force_norm": self.vector_norm(self.contact_force_vec),
            "contact_fx_mag": self.contact_fx_mag,
            "baseline_fx": self.precontact_baseline_fx,
            "contact_threshold": self.contact_threshold(),
            "contact_valid": float(self.contact_valid),
            "aruco_alignment_valid": float(self.aruco_alignment_valid),
            "aruco_alignment_center_y_error_m": self.aruco_alignment_center_y_error,
            "aruco_z_trim_m": self.aruco_z_trim,
            "alignment_y_trim_m": self.alignment_y_trim,
            "alignment_z_trim_m": self.alignment_z_trim,
        }

    def write_csv_header(self, path: str, columns: List[str]) -> None:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(columns)

    def append_csv_row(self, path: str, columns: List[str], row: Dict[str, Any]) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writerow(row)

    def _log_snapshot_row(self) -> None:
        self.append_csv_row(self.csv_path, self.log_columns, self.snapshot())

    def log_sync_step(self, command_distance: float) -> None:
        row = self.snapshot()
        row["command_distance"] = command_distance
        self.append_csv_row(self.sync_csv_path, self.sync_columns, row)

    # -----------------------------------------------------------------------
    # Contact state management
    # -----------------------------------------------------------------------

    def set_contact_state(self, active: bool, reason: str) -> None:
        if active == self.contact_active:
            return

        self.contact_active = active
        if active:
            self.bag_recorder.start(reason)
            self.external_camera_trigger.start(self._external_camera_trigger_name(reason), reason)
        else:
            self.current_press_dir = None
            self.bag_recorder.stop(reason)
            self.external_camera_trigger.stop(reason)

    def _external_camera_trigger_name(self, reason: str) -> str:
        safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (reason or "window"))
        window_idx = getattr(self.bag_recorder, "window_index", 0) if self.bag_recorder is not None else 0
        return f"{safe_reason}_window_{window_idx:03d}"

    # -----------------------------------------------------------------------
    # Waypoint completion
    # -----------------------------------------------------------------------

    def wait_for_waypoint_completion(
        self,
        timeout: Optional[float] = None,
        target: Optional[Tuple[float, float, float]] = None,
        position_tol: Optional[float] = None,
    ) -> bool:
        timeout_s = self.waypoint_timeout if timeout is None else timeout
        position_tol_m = 0.002 if position_tol is None else position_tol
        end_t = self.now_s() + timeout_s
        saw_active = bool(self.wp_active)

        while True:
            current_t = self.now_s()
            if current_t >= end_t:
                break

            wp_active = bool(self.wp_active)
            saw_active = saw_active or wp_active

            if target is not None and self.ee is not None:
                ee_target_error = math.sqrt(
                    (self.ee.x - target[0]) ** 2 +
                    (self.ee.y - target[1]) ** 2 +
                    (self.ee.z - target[2]) ** 2
                )
                if ee_target_error <= position_tol_m:
                    return True

            if saw_active and not wp_active:
                return True

            rclpy.spin_once(self, timeout_sec=0.05)

        return False

    # -----------------------------------------------------------------------
    # Stages
    # -----------------------------------------------------------------------

    def stage1_liveness(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "liveness"
        self.get_logger().info("Stage 1: Checking liveness...")

        # Pre-spin: let DDS discovery settle before checking timeouts.
        # With use_sim_time, the node's clock may not be initialized yet,
        # so we use wall-clock time for stage1 timeouts.
        import time as _wall_time
        self.get_logger().info("Stage 1: Waiting for DDS discovery...")
        self.spin_for(2.0)  # Let DDS discover publishers

        # Wait for subscribers
        t0_wall = _wall_time.monotonic()
        while _wall_time.monotonic() - t0_wall < 15.0:
            if self.pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.2)
        else:
            return False, "waypoint subscribers missing", {"pub_subs": self.pub.get_subscription_count()}

        # Wait for joint states
        self.get_logger().info("Stage 1: Waiting for joint_states...")
        t0_wall = _wall_time.monotonic()
        while _wall_time.monotonic() - t0_wall < 15.0:
            if self.js is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.2)
        else:
            return False, "joint_states missing", {"js_seen": self.js is not None}

        # Wait for pose topics
        t0_wall = _wall_time.monotonic()
        while _wall_time.monotonic() - t0_wall < 10.0:
            if self.ee is not None and self.des is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.2)
        else:
            return False, "Cartesian pose topics missing", {
                "ee_seen": self.ee is not None,
                "des_seen": self.des is not None,
            }

        info = {
            "pub_subs": self.pub.get_subscription_count(),
            "js_seen": self.js is not None,
            "ee_seen": self.ee is not None,
            "des_seen": self.des is not None,
            "wp_status_seen": self.wp_active is not None,
            "contact_valid_seen": self.contact_valid,
        }
        return True, "publishers, joint states, and Cartesian pose topics present", info

    def stage2_idle(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "idle"
        self.get_logger().info("Stage 2: Waiting for idle stabilization...")

        # Wait for arms to converge to start_position and correct Z heights
        convergence_timeout = 30.0
        convergence_start = self.now_s()
        converged = False
        z_convergence_tol = 0.010  # 10mm Z tolerance

        # Get target Z from YAML start_position
        target_z = self._get_controller_param_float("start_position.z", default=0.085)

        while self.now_s() - convergence_start < convergence_timeout:
            self.spin_for(0.25)
            info = self.snapshot()

            ee_x = info.get("ee_x", 0.0)
            ee_z = info.get("ee_z", 0.0)

            x_converged = ee_x >= self.min_offset_from_base
            z_converged = abs(ee_z - target_z) < z_convergence_tol

            if x_converged and z_converged:
                converged = True
                self.get_logger().info(
                    f"Stage 2: Arm converged to safe position "
                    f"(X: ee_x={ee_x:.4f} ≥ {self.min_offset_from_base}; "
                    f"Z: ee_z={ee_z:.4f}≈{target_z:.4f})"
                )
                break
            else:
                self.get_logger().info(
                    f"Stage 2: Waiting for convergence "
                    f"(X: ee_x={ee_x:.4f}; Z: ee_z={ee_z:.4f}/{target_z:.4f})"
                )

        if not converged:
            self.get_logger().warn(
                f"Stage 2: Convergence timeout ({convergence_timeout}s) — "
                f"proceeding with caution (ee_x={info.get('ee_x', 0.0):.4f})"
            )

        # Wait for arm velocity to drop below idle limit (quasi-static settle)
        settle_timeout = 15.0
        settle_start = self.now_s()
        velocity_settled = False
        while self.now_s() - settle_start < settle_timeout:
            self.spin_for(0.25)
            arm_vel = self.max_abs_velocity(self.js, self.arm_joint_names)
            if arm_vel is not None and arm_vel < self.idle_vel_limit:
                velocity_settled = True
                self.get_logger().info(
                    f"Stage 2: Arm velocity settled to {arm_vel:.4f} rad/s "
                    f"(limit {self.idle_vel_limit})"
                )
                break
            else:
                self.get_logger().info(
                    f"Stage 2: Waiting for velocity settle "
                    f"(current: {arm_vel if arm_vel is not None else float('nan'):.4f}, "
                    f"limit: {self.idle_vel_limit})"
                )

        if not velocity_settled:
            self.get_logger().warn(
                f"Stage 2: Velocity settle timeout ({settle_timeout}s) — "
                f"proceeding anyway (vel={self.max_abs_velocity(self.js, self.arm_joint_names)})"
            )

        # Capture baseline forces
        self.baseline_force_vec = self.contact_force_vec

        info = self.snapshot()
        ok = self.check_limits(self.js, self.idle_vel_limit, self.arm_joint_names)
        if not ok:
            # Last-chance: relax limit for initial homing transient
            self.get_logger().warn(
                f"Stage 2: Velocity {self.max_abs_velocity(self.js, self.arm_joint_names):.4f} still above "
                f"{self.idle_vel_limit} — relaxing limit to 3.0 for this run"
            )
            ok = self.check_limits(self.js, 3.0, self.arm_joint_names)
        return ok, "low velocity idle", info

    def _get_controller_param_float(self, param_name: str, default: float = 0.0) -> float:
        """Read a parameter from the controller YAML config."""
        try:
            import yaml
            yaml_path = f"/workspaces/omx_ros2/ws/install/omx_variable_stiffness_controller/share/omx_variable_stiffness_controller/config/{self.robot_ns}_variable_stiffness.yaml"
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)

            params = config.get(f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness", {}).get("ros__parameters", {})

            if param_name == "start_position.z":
                start_pos = params.get("start_position", [])
                if len(start_pos) >= 3:
                    return float(start_pos[2])
            elif param_name == "start_position.x":
                start_pos = params.get("start_position", [])
                if len(start_pos) >= 1:
                    return float(start_pos[0])
            elif "." in param_name:
                parts = param_name.split(".")
                value = params
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        return default
                return float(value) if value is not None else default
            else:
                value = params.get(param_name)
                return float(value) if value is not None else default

        except Exception as e:
            self.get_logger().warn(f"Failed to read {self.robot_ns} parameter {param_name}: {e}")
            return default

    def stage3_precontact(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "precontact"
        self.get_logger().info("Stage 3: Approaching contact target...")
        stage_info: List[Dict[str, Any]] = []

        if self.des is None:
            return False, "desired pose unavailable", {"samples": stage_info}

        self.spin_for(0.10)
        self.capture_precontact_baseline()

        if self.ee is not None:
            home = (self.ee.x, self.ee.y, self.ee.z)
        else:
            home = (self.des.position.x, self.des.position.y, self.des.position.z)

        # PHASE 1: Establish z-height at a safe fixed x anchor BEFORE any x-movement
        target_z_offset = self.configured_push_axis_height(home[2]) - home[2]
        current_z_offset = 0.0
        z_step = self.press_step

        self.get_logger().info(
            f"PHASE 1: Establishing z-height first (target_z_offset={target_z_offset:.4f})"
        )

        max_z_iterations = int(abs(target_z_offset) / z_step) + 5
        for z_iter in range(max_z_iterations):
            current_z_offset = self._step_offset_toward(current_z_offset, target_z_offset, z_step)

            target = self.compute_absolute_precontact_target(home, 0.0)
            target["offset_z"] = current_z_offset
            target["target_z"] = home[2] + current_z_offset

            self.apply_precontact_target(target, "none")
            self.spin_for(self.command_sample_delay)

            if abs(current_z_offset - target_z_offset) < 1e-6:
                self.get_logger().info("PHASE 1 complete: z-height established at push_axis")
                break

        # Wait for arm to converge to target Z
        z_settle_timeout = 10.0
        z_settle_start = self.now_s()
        z_settled = False
        z_settle_tol = 0.003

        while self.now_s() - z_settle_start < z_settle_timeout:
            self.spin_for(0.10)
            info = self.snapshot()
            ee_z = info.get("ee_z", 0.0)
            target_z = home[2] + target_z_offset

            if abs(ee_z - target_z) < z_settle_tol:
                z_settled = True
                self.get_logger().info(
                    f"PHASE 1 Z-settle: Arm converged to target Z "
                    f"(ee_z={ee_z:.4f}≈{target_z:.4f})"
                )
                break
            else:
                self.get_logger().info(
                    f"PHASE 1 Z-settle: Waiting for convergence "
                    f"(ee_z={ee_z:.4f}/{target_z:.4f})"
                )

        if not z_settled:
            self.get_logger().warn(
                f"PHASE 1 Z-settle: Timeout ({z_settle_timeout}s) — proceeding with caution"
            )

        # PHASE 2: Move in x to establish contact
        self.get_logger().info("PHASE 2: Moving in x to establish contact")

        offset = self.precontact_start_x_offset
        press_step_x = math.copysign(self.press_step, self.precontact_end_x_offset)

        for _ in range(self.max_precontact_iterations):
            if press_step_x > 0:
                offset = min(offset + press_step_x, self.precontact_end_x_offset)
            else:
                offset = max(offset + press_step_x, self.precontact_end_x_offset)

            offset = max(offset, self.min_offset_from_base)

            target = self.compute_absolute_precontact_target(home, offset)
            target["offset_z"] = current_z_offset
            target["target_z"] = home[2] + current_z_offset

            contact_mode = "contact" if self.contact_detected() else "none"
            self.apply_precontact_target(target, contact_mode)
            self.spin_for(self.command_sample_delay)

            contact = self.contact_detected()
            self.current_press_distance = abs(offset)

            snap = self.snapshot()
            snap["command_distance"] = abs(offset)
            stage_info.append(snap)
            self.log_sync_step(abs(offset))

            if contact:
                at_depth = abs(offset) >= self.precontact_end_x_offset - 1e-9
                if at_depth:
                    self.current_press_distance = self.precontact_end_x_offset
                    self.set_contact_state(True, "contact_established")
                    return True, "contact established", {"samples": stage_info}

        return False, "contact not established", {"samples": stage_info}

    def _step_offset_toward(self, current: float, target: float, step: float) -> float:
        """Step current toward target by at most step."""
        if abs(target - current) <= step:
            return target
        return current + math.copysign(step, target - current)

    def stage4_hold(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "hold"
        self.get_logger().info("Stage 4: Holding contact...")

        if not self.contact_detected():
            self.set_contact_state(False, "hold_requires_contact")
            return False, "contact not maintained for hold", self.snapshot()

        direction = self.compute_press_direction()
        if direction is None:
            self.set_contact_state(False, "hold_direction_unavailable")
            return False, "press direction unavailable", self.snapshot()

        self.current_press_dir = direction
        hold_distance = max(self.current_press_distance, self.press_step)
        self.current_press_distance = hold_distance

        # Capture anchor from current desired pose
        if self.des is None:
            self.set_contact_state(False, "hold_des_unavailable")
            return False, "desired pose unavailable", self.snapshot()

        anchor = (
            self.des.position.x,
            self.des.position.y,
            self.configured_push_axis_height(self.des.position.z),
        )

        target = self.apply_directional_press_target(
            hold_distance,
            direction,
            anchor,
            "directional_hold",
        )

        # Sustain the hold
        end_t = self.now_s() + self.hold_duration
        while self.now_s() < end_t:
            self.publish_absolute_target(
                (target["target_x"], target["target_y"], target["target_z"]),
                repeats=self.command_repeats,
            )
            self.spin_for(self.command_refresh_interval)

            if not self.contact_detected():
                self.set_contact_state(False, "contact_lost_during_hold")
                return False, "contact lost during hold", self.snapshot()

        return True, "directional hold stable", self.snapshot()

    def stage5_adaptive_press(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "compress"
        self.get_logger().info("Stage 5: Adaptive press...")
        stage_info: List[Dict[str, Any]] = []

        if not self.enable_deep_press:
            # Just hold at current depth
            if not self.contact_detected():
                self.set_contact_state(False, "contact_lost_before_stage5_hold")
                return False, "contact lost before sustained directional hold", self.snapshot()

            direction = self.compute_press_direction()
            if direction is None:
                self.set_contact_state(False, "stage5_direction_unavailable")
                return False, "press direction unavailable", self.snapshot()

            self.current_press_dir = direction

            if self.des is None:
                self.set_contact_state(False, "stage5_des_unavailable")
                return False, "desired pose unavailable", self.snapshot()

            anchor = (
                self.des.position.x,
                self.des.position.y,
                self.configured_push_axis_height(self.des.position.z),
            )

            target = self.compute_absolute_directional_press_target(
                self.current_press_distance,
                direction,
                anchor,
            )

            # Sustain the hold
            end_t = self.now_s() + self.stage5_hold_duration
            while self.now_s() < end_t:
                self.publish_absolute_target(
                    (target["target_x"], target["target_y"], target["target_z"]),
                    repeats=self.command_repeats,
                )
                self.spin_for(self.command_refresh_interval)

                if not self.contact_detected():
                    self.set_contact_state(False, "contact_lost_during_stage5_hold")
                    return False, "contact lost during stage 5 hold", self.snapshot()

            self.set_contact_state(False, "directional_hold_complete")
            return True, "sustained directional hold completed", {"samples": [self.snapshot()]}

        # Deep press mode: incrementally compress
        distance = self.current_press_distance

        if self.des is None:
            self.set_contact_state(False, "press_des_unavailable")
            return False, "desired pose unavailable", {"samples": stage_info}

        anchor = (
            self.des.position.x,
            self.des.position.y,
            self.configured_push_axis_height(self.des.position.z),
        )
        support_seen = False
        low_support_count = 0

        while distance < self.max_press_distance - 1e-9:
            if not self.contact_detected():
                self.set_contact_state(False, "contact_lost_during_press")
                return False, "contact lost during compression", {"samples": stage_info}

            direction = self.compute_press_direction()
            if direction is None:
                self.set_contact_state(False, "press_direction_unavailable")
                return False, "press direction unavailable", {"samples": stage_info}

            self.current_press_dir = direction
            target_distance = min(distance + self.deep_press_step, self.max_press_distance)
            self.current_press_distance = target_distance

            projected_force = self.projected_force(direction)
            contact_mode = "compressing" if self.finite(projected_force) else "seeking"

            target = self.apply_directional_press_target(target_distance, direction, anchor, contact_mode)
            target_pose = (target["target_x"], target["target_y"], target["target_z"])
            wait_ok = self.wait_for_waypoint_completion(
                timeout=self.waypoint_timeout,
                target=target_pose,
                position_tol=0.002,
            )

            if not wait_ok:
                snap = self.snapshot()
                snap["command_distance"] = target_distance
                stage_info.append(snap)
                self.log_sync_step(target_distance)
                timeout_expected = (
                    self.contact_detected()
                    and self.finite(snap["projected_force"])
                    and abs(snap["projected_force"]) >= self.stage5_supported_force_min
                )
                timeout_msg = (
                    "Stage 5: waypoint timeout while compressing; target_distance=%.6f current_distance=%.6f"
                    % (target_distance, distance)
                )
                if timeout_expected:
                    self.get_logger().info(timeout_msg + " — continuing quasi-static compression")
                else:
                    self.get_logger().warn(timeout_msg)
                if not self.contact_detected():
                    self.set_contact_state(False, "press_waypoint_timeout")
                    return False, "contact lost during compression", {"samples": stage_info}

            settle_ok = self._wait_for_axial_settle(target_pose, direction=direction)
            if not settle_ok:
                self.get_logger().warn(
                    "Stage 5: axial settle timeout at target_distance=%.6f current_distance=%.6f"
                    % (target_distance, distance)
                )
                if not self.contact_detected():
                    self.set_contact_state(False, "press_axial_settle_timeout")
                    return False, "contact lost during axial settle", {"samples": stage_info}

            probe_ok = self._run_lateral_microperturbation(
                target_pose,
                direction,
                stage_info,
                target_distance,
            )
            if not probe_ok:
                self.set_contact_state(False, "lateral_probe_contact_lost")
                return False, "contact lost during lateral probe", {"samples": stage_info}

            distance = target_distance
            self.current_press_distance = distance

            if not self.contact_detected():
                self.set_contact_state(False, "contact_lost_after_press_step")
                snap = self.snapshot()
                snap["command_distance"] = distance
                stage_info.append(snap)
                self.log_sync_step(distance)
                return False, "contact lost during compression", {"samples": stage_info}

            snap = self.snapshot()
            snap["command_distance"] = distance
            stage_info.append(snap)
            self.log_sync_step(distance)

            projected_force = snap["projected_force"]
            support_seen, low_support_count, support_lost = self._stage5_update_support_state(
                projected_force,
                support_seen,
                low_support_count,
            )
            if support_lost:
                self.instability_mode = True
                self.set_contact_state(False, "spring_lost_support")
                snap["command_distance"] = distance
                stage_info.append(snap)
                self.log_sync_step(distance)
                return False, "spring lost support; graceful release", {"samples": stage_info}

            if self.finite(projected_force) and projected_force > self.max_force_mag_abort:
                self.instability_mode = True
                self.set_contact_state(False, "contact_force_too_high")
                return False, "contact force too high", {"samples": stage_info}

        self.set_contact_state(False, "compression_complete")
        return True, "adaptive compression completed", {"samples": stage_info}

    # -----------------------------------------------------------------------
    # Results and plotting
    # -----------------------------------------------------------------------

    def write_results(self, results: List[Dict[str, Any]]) -> None:
        with open(self.results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        self.generate_publication_plots()
        self.write_trial_analysis()

    def write_trial_analysis(self) -> None:
        if not os.path.exists(self.sync_csv_path):
            return

        analysis_path = os.path.join(self.run_dir, "single_arm_harness_v4_trial_analysis.json")
        try:
            analysis = analyze_trial(
                self.sync_csv_path,
                camera_trace_path=self.aruco_trace_path,
                boundary_condition=self.beam_boundary_condition,
                effective_length_m=self.beam_effective_length_m,
            )
        except Exception as error:
            self.get_logger().warn(f"Trial analysis skipped: {error}")
            return

        write_analysis_json(analysis, analysis_path)
        self.get_logger().info(f"Trial analysis written to {analysis_path}")

    def generate_publication_plots(self) -> None:
        """Generate publication-quality plots from the harness CSV data."""
        if not HAS_MATPLOTLIB or not HAS_PANDAS:
            return

        csv_path = getattr(self, 'csv_path', None)
        if not csv_path or not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: Could not read CSV for plotting: {e}")
            return

        if len(df) < 2:
            return

        # Convert timestamp to relative time
        if 'timestamp' in df.columns:
            t0 = df['timestamp'].iloc[0]
            df['time_s'] = df['timestamp'] - t0
        else:
            df['time_s'] = df.index * 0.02

        # Apply publication style
        matplotlib.rcParams.update(PUB_RC)

        # Define subplot groups
        plot_groups = [
            {
                'title': 'End-Effector Position',
                'columns': [
                    ('ee_x', 'EE X'),
                    ('ee_y', 'EE Y'),
                    ('ee_z', 'EE Z'),
                ],
                'ylabel': 'Position (m)',
            },
            {
                'title': 'Contact Forces',
                'columns': [
                    ('contact_fx', 'Fx'),
                    ('contact_fy', 'Fy'),
                    ('contact_fz', 'Fz'),
                ],
                'ylabel': 'Force (N)',
            },
            {
                'title': 'Pre-Contact Offsets',
                'columns': [
                    ('offset_x', 'Offset X'),
                    ('offset_y', 'Offset Y'),
                    ('offset_z', 'Offset Z'),
                ],
                'ylabel': 'Offset (m)',
            },
        ]

        # Generate one figure per group
        for group in plot_groups:
            valid_cols = []
            for col, label in group['columns']:
                if col in df.columns and df[col].notna().any():
                    valid_cols.append((col, label))

            if not valid_cols:
                continue

            n_sub = len(valid_cols)
            fig, axes = plt.subplots(n_sub, 1, figsize=(20, max(5.5 * n_sub, 8)),
                                     sharex=True, squeeze=False)
            axes = axes.flatten()

            fig.suptitle(
                f"{group['title']}\nSingle-Arm Harness Run {getattr(self, 'run_ts', 'unknown')}",
                fontsize=28, fontweight="bold", y=0.99,
            )

            for idx, (col, label) in enumerate(valid_cols):
                ax = axes[idx]
                ax.plot(
                    df['time_s'].values, df[col].values,
                    color=C_ARM,
                    linewidth=2.8,
                    label=label,
                )
                ax.set_ylabel(label, fontsize=24, fontweight="bold")
                ax.tick_params(axis='both', which='major', labelsize=22)
                ax.tick_params(axis='both', which='minor', labelsize=18)
                ax.minorticks_on()
                ax.grid(True, which='major', alpha=0.45, linewidth=1.0)
                ax.grid(True, which='minor', alpha=0.2, linewidth=0.6, linestyle=':')
                ax.legend(
                    loc='upper left',
                    bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0.0,
                    framealpha=0.95,
                    edgecolor="0.2",
                    fontsize=22,
                )

            axes[-1].set_xlabel("Time (s)", fontsize=24, fontweight="bold")
            plt.tight_layout()

            col_names = '_'.join([c.replace(' ', '_') for c, _ in valid_cols])
            fig_name = f"single_arm_harness_{group['title'].replace(' ', '_')}_{col_names}"
            fig_path_pdf = os.path.join(self.plot_save_dir or self.run_dir, f"{fig_name}.pdf")
            fig_path_png = os.path.join(self.plot_save_dir or self.run_dir, f"{fig_name}.png")

            plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
            plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved publication plot: {fig_path_pdf}")
            print(f"Saved publication plot: {fig_path_png}")

    # -----------------------------------------------------------------------
    # Abort
    # -----------------------------------------------------------------------

    def controlled_withdrawal(
        self,
        steps: int = 80,
        step_size: float = 0.004,
        repeats: Optional[int] = None,
        sample_delay: Optional[float] = None,
        final_pause_s: float = 1.0,
    ) -> None:
        """Withdraw the arm strictly along negative x until contact is lost or home/start pose is reached."""
        self.get_logger().info("Starting controlled withdrawal along -x to home/start pose...")
        repeat_count = self.command_repeats if repeats is None else repeats
        step_delay = self.command_sample_delay if sample_delay is None else sample_delay
        current_offset = self.current_offset_x
        target_offset = 0.0

        if not self.finite(current_offset) or current_offset <= target_offset:
            self.get_logger().info("No withdrawal needed (already at home/start x position)")
            return

        home_x_yaml = self._get_controller_param_float("start_position.x", default=0.18)
        home_y_yaml = self._get_controller_param_float("start_position.y", default=0.0)
        home_z_yaml = self._get_controller_param_float("start_position.z", default=0.085)
        home = (
            home_x_yaml,
            home_y_yaml,
            self.configured_push_axis_height(home_z_yaml),
        )

        if self.des is None:
            self.get_logger().warn("Desired pose unavailable — holding home/start pose")
            self.publish_offset(0.0, 0.0, 0.0, repeats=repeat_count, dt=0.05)
            self.spin_for(final_pause_s)
            return
        retract_dir = (-1.0, 0.0, 0.0)

        for step_idx in range(steps):
            if current_offset <= target_offset + 1e-6:
                self.get_logger().info("Withdrawal reached home/start x position")
                break

            if not self.contact_detected():
                self.get_logger().info("Withdrawal stopped: contact lost during retraction")
                break

            current_offset = max(current_offset - step_size, target_offset)
            target = self.compute_absolute_precontact_target(home, current_offset)
            self.publish_absolute_target(
                (target["target_x"], target["target_y"], target["target_z"]),
                repeats=repeat_count,
                direction=retract_dir,
            )
            self.spin_for(step_delay)

            if step_idx % 5 == 0:
                ee_x = self.ee.x if self.ee is not None else float("nan")
                self.get_logger().info(
                    f"  Withdrawal step {step_idx}: offset_x={current_offset:.4f}, "
                    f"target_x={target['target_x']:.4f}, ee_x={ee_x:.4f}"
                )

        self.get_logger().info("Withdrawal complete — holding home/start pose")
        self.publish_offset(0.0, 0.0, 0.0, repeats=repeat_count, dt=0.05)
        self.spin_for(final_pause_s)

    def safe_abort(self) -> None:
        self.set_contact_state(False, "safe_abort")
        self.current_phase = "abort"
        try:
            self.get_logger().error("ABORT: controlled withdrawal to safe position")
            if self.instability_mode:
                self.get_logger().info("ABORT: instability recovery with gentle release")
                self.spin_for(1.5)
                self.controlled_withdrawal(
                    steps=240,
                    step_size=0.0015,
                    repeats=1,
                    sample_delay=0.5,
                    final_pause_s=3.0,
                )
            else:
                self.controlled_withdrawal()
        except Exception as e:
            try:
                self.get_logger().error(f"ABORT: withdrawal failed ({e}), maintaining safe position")
                # Maintain safe position instead of sending zero offset
                if self.des is not None:
                    home = (self.des.position.x, self.des.position.y, self.des.position.z)
                    safe_offset = self.min_offset_from_base - home[0]
                    if safe_offset > 0:
                        self.publish_offset(safe_offset, 0.0, 0.0, repeats=1, dt=0.05)
                    else:
                        self.publish_offset(0.0, 0.0, 0.0, repeats=1, dt=0.05)
                    self.spin_for(2.0)
            except Exception:
                pass
        finally:
            self.instability_mode = False

    # -----------------------------------------------------------------------
    # Main run loop
    # -----------------------------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        stages = [
            self.stage1_liveness,
            self.stage2_idle,
            self.stage3_precontact,
            self.stage4_hold,
            self.stage5_adaptive_press,
        ]

        results: List[Dict[str, Any]] = []
        try:
            for idx, stage_fn in enumerate(stages, start=1):
                ok, msg, info = stage_fn()
                record = {
                    "stage": idx,
                    "pass": ok,
                    "message": msg,
                    "info": info,
                }
                results.append(record)
                self.get_logger().info(f"Stage {idx}: {msg} -> {ok}")

                if not ok:
                    self.safe_abort()
                    self.write_results(results)
                    return results

            self.current_phase = "done"
            self.set_contact_state(False, "run_complete")
            self.get_logger().info("ALL STAGES PASSED")
            self.controlled_withdrawal()
            self.write_results(results)
            return results
        finally:
            self.set_contact_state(False, "run_finally")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-arm hardware harness v4")
    parser.add_argument(
        "--robot-namespace",
        type=str,
        default=None,
        help="Robot namespace (default: robot1 or OMX_HARNESS_SINGLE_ARM_ROBOT env var)",
    )
    parser.add_argument(
        "--push-axis-height-z",
        type=float,
        default=None,
        help=(
            "Absolute push-axis height in metres. When provided, precontact and press commands "
            "hold the push at this z height instead of the controller trajectory z."
        ),
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        default=False,
        help="Use simulation time (for Gazebo). Required when running against Gazebo simulation.",
    )
    parser.add_argument(
        "--plot-save-dir",
        type=str,
        default=None,
        help="Directory to save publication plots (default: run_dir)",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()

    # Resolve robot namespace
    robot_ns = args.robot_namespace or os.environ.get("OMX_HARNESS_SINGLE_ARM_ROBOT", "robot1")

    # Initialize ROS with or without sim time
    if args.use_sim_time:
        rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    else:
        rclpy.init()

    node = SingleArmHardwareHarnessV4(
        robot_namespace=robot_ns,
        push_axis_height_z=args.push_axis_height_z,
        use_sim_time=args.use_sim_time,
        plot_save_dir=args.plot_save_dir,
    )

    exit_code = 0
    try:
        out = node.run()
        print(json.dumps(out, indent=2))
        # Check if any stage failed
        for stage_result in out:
            if not stage_result.get("pass", True):
                exit_code = 1
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
