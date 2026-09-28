#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped

from hardware_harness_v2 import HardwareHarnessAdaptive

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

# Color palettes
C_R1 = "#00429d"  # robot 1 — deep blue
C_R2 = "#b5002a"  # robot 2 — dark crimson
ROBOT_COLORS = {1: C_R1, 2: C_R2}


class MutualContactBagRecorder:
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
        self.metadata_path = self.run_dir / "mutual_contact_windows.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "mutual_contact_bags").mkdir(parents=True, exist_ok=True)

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
        output_dir = self.run_dir / "mutual_contact_bags" / f"window_{self.window_index:03d}"
        log_path = self.run_dir / f"mutual_contact_window_{self.window_index:03d}.log"
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
    high-FPS recorder (see scripts/Run-CaptureSession.ps1 and
    tools/realsense_text_trigger_capture.py).

    The recorder polls the file via mtime+size and parses lines like:
        START name=<window_name>
        STOP
        QUIT

    When ``trigger_path`` is None or empty, every method is a no-op so the
    harness behaves exactly as before.
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


class HardwareHarnessAdaptiveV3(HardwareHarnessAdaptive):
    DEFAULT_REALSENSE_TOPICS = [
        "/camera/color/image_raw",
        "/camera/color/camera_info",
        "/camera/aligned_depth_to_color/image_raw",
        "/camera/aligned_depth_to_color/camera_info",
    ]

    def __init__(self, push_axis_height_z: Optional[float] = None, use_sim_time: bool = False, plot_save_dir: Optional[str] = None) -> None:
        super().__init__(use_sim_time=use_sim_time)
        self.push_axis_height_z = self.resolve_push_axis_height_z(push_axis_height_z)
        self.plot_save_dir = plot_save_dir  # Override default plot save directory (None = use run_dir)

        old_csv_path = self.csv_path
        old_sync_csv_path = self.sync_csv_path

        self.run_dir = os.path.join(self.log_root, f"hardware_harness_v3_{self.run_ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "hardware_harness_v3_snapshot.csv")
        self.sync_csv_path = os.path.join(self.run_dir, "hardware_harness_v3_sync_steps.csv")
        self.results_path = os.path.join(self.run_dir, "hardware_harness_v3_results.json")
        self.manifest_path = os.path.join(self.run_dir, "hardware_harness_v3_manifest.json")

        self.current_offset_y1 = float("nan")
        self.current_offset_z1 = float("nan")
        self.current_offset_y2 = float("nan")
        self.current_offset_z2 = float("nan")
        self.current_press_offset_y1 = 0.0
        self.current_press_offset_z1 = 0.0
        self.current_press_offset_y2 = 0.0
        self.current_press_offset_z2 = 0.0
        self.current_press_distance_1 = 0.0
        self.current_press_distance_2 = 0.0
        self.directional_press_enabled = False
        self.mutual_contact_active = False
        self.current_direction_alignment = float("nan")
        self.current_mutual_press_dir: Optional[Tuple[float, float, float]] = None
        self.mutual_contact_anchor_1: Optional[Tuple[float, float, float]] = None
        self.mutual_contact_anchor_2: Optional[Tuple[float, float, float]] = None
        self.last_mutual_contact_anchor_1: Optional[Tuple[float, float, float]] = None
        self.last_mutual_contact_anchor_2: Optional[Tuple[float, float, float]] = None
        self.current_directional_target_1: Optional[Tuple[float, float, float]] = None
        self.current_directional_target_2: Optional[Tuple[float, float, float]] = None
        self.last_directional_target_1: Optional[Tuple[float, float, float]] = None
        self.last_directional_target_2: Optional[Tuple[float, float, float]] = None

        self.contact_force_vec_1: Optional[Tuple[float, float, float]] = None
        self.contact_force_vec_2: Optional[Tuple[float, float, float]] = None
        self.contact_torque_vec_1: Optional[Tuple[float, float, float]] = None
        self.contact_torque_vec_2: Optional[Tuple[float, float, float]] = None
        self.baseline_force_vec_1: Optional[Tuple[float, float, float]] = None
        self.baseline_force_vec_2: Optional[Tuple[float, float, float]] = None

        self.precontact_start_x_offset = 0.12  # Start approach from 12cm (120mm) for both robots

        self.precontact_end_x_offset = 0.180  # robot1 moves in +x toward spring (positive = toward spring in local frame)
        self.precontact_end_x_offset_2 = 0.180  # robot2 moves in +x toward spring (positive = toward spring in local frame)
        # Minimum offset from start position to prevent arm folding into the base.
        # Start position is local x=0.11; to ensure push axis originates at least
        # 12cm from base (local x=0), require offset >= 0.12 so EE stays at local x >= 0.12.
        self.min_offset_from_base = 0.12
        # Explicitly set to 50 mm so the directional press after bilateral contact
        # is sufficient to reach 2 N on spring s1 (k=49.4 N/m, needs ~40 mm).
        # This is independent of v2's press_end_x_offset.
        self.press_end_distance = 0.050
        self.precontact_end_distance = abs(self.precontact_end_x_offset)
        self.max_precontact_iterations = 80  # 100mm / 1.5mm step = 67 steps minimum
        # The allowed directional press distance is an absolute distance measured
        # from the home pose: start at the precontact depth then add the
        # additional `press_end_distance` budget plus any side-extra allowance.
        self.max_directional_press_distance = (
            self.precontact_end_distance + self.press_end_distance + self.max_side_extra_press
        )
        # Minimum allowed 3-D Euclidean distance between the two EE positions.
        # If the robots come closer than this the harness aborts to prevent collision.
        # Set via OMX_HARNESS_V3_MIN_EE_SEPARATION (metres); default 0.020 m.
        self.min_ee_separation: float = float(
            os.environ.get("OMX_HARNESS_V3_MIN_EE_SEPARATION", "0.020")
        )
        # Extra safety margin added beyond the full press depth when computing each
        # robot's x floor.  Set via OMX_HARNESS_V3_MIN_X_FLOOR_MARGIN (metres).
        self.min_x_floor_margin: float = float(
            os.environ.get("OMX_HARNESS_V3_MIN_X_FLOOR_MARGIN", "0.020")
        )
        # Maximum allowed absolute x-axis gap between the two grippers. If an
        # explicit absolute limit is not provided, stage 2 derives it from the
        # home-pose x gap plus this margin.
        self.max_ee_x_gap_margin: float = float(
            os.environ.get("OMX_HARNESS_V3_MAX_EE_X_GAP_MARGIN", "0.020")
        )
        max_ee_x_gap_override = os.environ.get("OMX_HARNESS_V3_MAX_EE_X_GAP")
        self.max_ee_x_gap: Optional[float] = (
            float(max_ee_x_gap_override) if max_ee_x_gap_override is not None else None
        )
        # Per-robot x lower bounds, computed at stage 2 from des.position.x.
        # None means not yet computed (stage 2 not yet run).
        self.min_ee_x_1: Optional[float] = None
        self.min_ee_x_2: Optional[float] = None
        self.mutual_direction_alignment_min = 0.0
        self.directional_command_repeats = 1
        self.directional_refresh_interval = 0.75
        self.directional_hold_duration = 4.0
        self.directional_stage5_hold_duration = 4.0
        self.directional_waypoint_timeout = 3.5
        self.enable_directional_deep_press = self._env_flag("OMX_HARNESS_V3_ENABLE_DEEP_PRESS", False)

        self.record_all_topics = self._env_flag("OMX_HARNESS_V3_RECORD_ALL_TOPICS", True)
        self.selected_topics = self.default_mutual_contact_topics()
        self.bag_recorder = MutualContactBagRecorder(
            self.run_dir,
            record_all_topics=self.record_all_topics,
            selected_topics=self.selected_topics,
        )
        self.external_camera_trigger = ExternalCameraTrigger(
            os.environ.get("OMX_HARNESS_V3_CAMERA_TRIGGER_FILE"),
            os.path.join(self.run_dir, "external_camera_trigger.log"),
        )

        self.log_columns = [
            "timestamp",
            "phase",
            "offset_x1",
            "offset_y1",
            "offset_z1",
            "offset_x2",
            "offset_y2",
            "offset_z2",
            "press_offset_x1",
            "press_offset_y1",
            "press_offset_z1",
            "press_offset_x2",
            "press_offset_y2",
            "press_offset_z2",
            "press_distance_1",
            "press_distance_2",
            "contact_mode",
            "mutual_contact_active",
            "directional_press_enabled",
            "mutual_press_dir_x",
            "mutual_press_dir_y",
            "mutual_press_dir_z",
            "direction_alignment",
            "projected_force_1",
            "projected_force_2",
            "bag_recording_active",
            "robot1_max_vel",
            "robot2_max_vel",
            "ee_x_1",
            "ee_y_1",
            "ee_z_1",
            "ee_x_2",
            "ee_y_2",
            "ee_z_2",
            "ee_x_diff",
            "desired_x_1",
            "desired_y_1",
            "desired_z_1",
            "desired_x_2",
            "desired_y_2",
            "desired_z_2",
            "desired_x_diff",
            "contact_fx_1",
            "contact_fy_1",
            "contact_fz_1",
            "contact_tx_1",
            "contact_ty_1",
            "contact_tz_1",
            "contact_fx_2",
            "contact_fy_2",
            "contact_fz_2",
            "contact_tx_2",
            "contact_ty_2",
            "contact_tz_2",
            "contact_force_norm_1",
            "contact_force_norm_2",
            "contact_fx_mag_1",
            "contact_fx_mag_2",
            "contact_fx_diff",
            "baseline_fx_1",
            "baseline_fx_2",
            "contact_threshold_1",
            "contact_threshold_2",
            "contact_valid_1",
            "contact_valid_2",
        ]
        self.sync_columns = [
            "timestamp",
            "phase",
            "command_distance",
            "offset_x1",
            "offset_y1",
            "offset_z1",
            "offset_x2",
            "offset_y2",
            "offset_z2",
            "press_offset_x1",
            "press_offset_y1",
            "press_offset_z1",
            "press_offset_x2",
            "press_offset_y2",
            "press_offset_z2",
            "press_distance_1",
            "press_distance_2",
            "contact_mode",
            "mutual_contact_active",
            "directional_press_enabled",
            "mutual_press_dir_x",
            "mutual_press_dir_y",
            "mutual_press_dir_z",
            "direction_alignment",
            "projected_force_1",
            "projected_force_2",
            "bag_recording_active",
            "robot1_max_vel",
            "robot2_max_vel",
            "ee_x_1",
            "ee_y_1",
            "ee_z_1",
            "ee_x_2",
            "ee_y_2",
            "ee_z_2",
            "ee_x_diff",
            "desired_x_1",
            "desired_y_1",
            "desired_z_1",
            "desired_x_2",
            "desired_y_2",
            "desired_z_2",
            "desired_x_diff",
            "contact_fx_1",
            "contact_fy_1",
            "contact_fz_1",
            "contact_tx_1",
            "contact_ty_1",
            "contact_tz_1",
            "contact_fx_2",
            "contact_fy_2",
            "contact_fz_2",
            "contact_tx_2",
            "contact_ty_2",
            "contact_tz_2",
            "contact_force_norm_1",
            "contact_force_norm_2",
            "contact_fx_mag_1",
            "contact_fx_mag_2",
            "contact_fx_diff",
            "baseline_fx_1",
            "baseline_fx_2",
            "contact_threshold_1",
            "contact_threshold_2",
            "contact_valid_1",
            "contact_valid_2",
        ]

        directional_target_columns = [
            "anchor_x_1",
            "anchor_y_1",
            "anchor_z_1",
            "anchor_x_2",
            "anchor_y_2",
            "anchor_z_2",
            "target_x_1",
            "target_y_1",
            "target_z_1",
            "target_x_2",
            "target_y_2",
            "target_z_2",
        ]
        for column in directional_target_columns:
            if column not in self.log_columns:
                self.log_columns.append(column)
            if column not in self.sync_columns:
                self.sync_columns.append(column)

        for old_path in (old_csv_path, old_sync_csv_path):
            if old_path not in (self.csv_path, self.sync_csv_path) and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        self.write_csv_header(self.csv_path, self.log_columns)
        self.write_csv_header(self.sync_csv_path, self.sync_columns)
        self.write_manifest()

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _get_controller_param_float(self, robot: str, param_name: str, default: float = 0.0) -> float:
        """Read a parameter from the controller YAML config.
        
        Args:
            robot: "robot1" or "robot2"
            param_name: Parameter name like "start_position.z"
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        try:
            import yaml
            yaml_path = f"/workspaces/omx_ros2/ws/install/omx_variable_stiffness_controller/share/omx_variable_stiffness_controller/config/{robot}_variable_stiffness.yaml"
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Navigate to the parameter
            # Format: /robot1/robot1_variable_stiffness: ros__parameters: start_position: [x, y, z]
            params = config.get(f"/{robot}/{robot}_variable_stiffness", {}).get("ros__parameters", {})
            
            if param_name == "start_position.z":
                start_pos = params.get("start_position", [])
                if len(start_pos) >= 3:
                    return float(start_pos[2])
            elif "." in param_name:
                # Handle nested parameters
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
            self.get_logger().warn(f"Failed to read {robot} parameter {param_name}: {e}")
            return default

    def resolve_push_axis_height_z(self, push_axis_height_z: Optional[float]) -> Optional[float]:
        if push_axis_height_z is None:
            raw_value = os.environ.get("OMX_HARNESS_V3_PUSH_AXIS_HEIGHT_Z")
            if raw_value is None or not raw_value.strip():
                return None
            push_axis_height_z = float(raw_value)

        push_axis_height_z = float(push_axis_height_z)
        if not math.isfinite(push_axis_height_z):
            raise ValueError("push_axis_height_z must be finite")
        return push_axis_height_z

    def configured_push_axis_height(self, fallback_z: float) -> float:
        return fallback_z if self.push_axis_height_z is None else self.push_axis_height_z

    def normalize_push_axis_anchor(
        self,
        anchor: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        if self.push_axis_height_z is None:
            return anchor
        return (anchor[0], anchor[1], self.push_axis_height_z)

    def default_mutual_contact_topics(self) -> List[str]:
        camera_topics = os.environ.get("OMX_HARNESS_V3_CAMERA_TOPICS")
        if camera_topics:
            realsense_topics = [topic.strip() for topic in camera_topics.split(",") if topic.strip()]
        else:
            realsense_topics = list(self.DEFAULT_REALSENSE_TOPICS)

        return [
            "/robot1/joint_states",
            "/robot2/joint_states",
            "/robot1/robot1_variable_stiffness/end_effector_position",
            "/robot2/robot2_variable_stiffness/end_effector_position",
            "/robot1/robot1_variable_stiffness/cartesian_pose_desired",
            "/robot2/robot2_variable_stiffness/cartesian_pose_desired",
            "/robot1/robot1_variable_stiffness/contact_wrench",
            "/robot2/robot2_variable_stiffness/contact_wrench",
            "/robot1/robot1_variable_stiffness/contact_valid",
            "/robot2/robot2_variable_stiffness/contact_valid",
            "/robot1/robot1_variable_stiffness/waypoint_active",
            "/robot2/robot2_variable_stiffness/waypoint_active",
            "/robot1/robot1_variable_stiffness/waypoint_command",
            "/robot2/robot2_variable_stiffness/waypoint_command",
            "/tf",
            "/tf_static",
            *realsense_topics,
        ]

    def write_manifest(self) -> None:
        payload = {
            "run_ts": self.run_ts,
            "run_dir": self.run_dir,
            "record_all_topics": self.record_all_topics,
            "selected_topics": self.selected_topics,
            "push_axis_height_z": self.push_axis_height_z,
            "press_end_distance": self.press_end_distance,
            "max_directional_press_distance": self.max_directional_press_distance,
            "max_ee_x_gap": self.max_ee_x_gap,
            "max_ee_x_gap_margin": self.max_ee_x_gap_margin,
            "mutual_direction_alignment_min": self.mutual_direction_alignment_min,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def compute_press_target(self, x_offset1: float, x_offset2: float) -> Optional[Dict[str, float]]:
        if self.des1 is None or self.des2 is None:
            return None

        target_z_1 = self.configured_push_axis_height(self.des1.position.z)
        target_z_2 = self.configured_push_axis_height(self.des2.position.z)
        return {
            "offset_x1": x_offset1,
            "offset_y1": 0.0,
            "offset_z1": target_z_1 - self.des1.position.z,
            "offset_x2": x_offset2,
            "offset_y2": 0.0,
            "offset_z2": target_z_2 - self.des2.position.z,
        }

    # Minimum distance from robot base (12cm) - HARD SAFETY CONSTRAINT
    # Set to 0.12m to avert collisions with metallic platform during withdrawal
    # when z-height may drop before the x-position recovers.
    MIN_DISTANCE_FROM_BASE = 0.12

    def compute_absolute_precontact_target(
        self,
        home1: Tuple[float, float, float],
        home2: Tuple[float, float, float],
        x_offset1: float,
        x_offset2: float,
    ) -> Dict[str, float]:
        """Compute precontact target with z locked to current EE height and horizontal gripper orientation.
        
        During precontact approach, both position AND orientation are held constant:
        - z is locked to CURRENT EE height (not push_axis_height) to avoid sudden vertical drops
        - gripper stays perfectly horizontal (identity quaternion) for stable alignment
        
        CRITICAL: Z must NEVER exceed push_axis_height_z during precontact.
        Clamp to min(current EE Z, push_axis_height_z) — arms descend to contact
        height if they are above it, but never drop below current height.
        """
        # Clamp Z: never above push_axis_height_z, but never below current EE Z
        pah = self.push_axis_height_z if self.push_axis_height_z is not None else min(home1[2], home2[2])
        target_z1 = min(home1[2], pah)
        target_z2 = min(home2[2], pah)
        target_x1 = max(home1[0] + x_offset1, self.MIN_DISTANCE_FROM_BASE)
        target_x2 = max(home2[0] + x_offset2, self.MIN_DISTANCE_FROM_BASE)
        return {
            "offset_x1": target_x1 - home1[0],
            "offset_y1": 0.0,
            "offset_z1": target_z1 - home1[2],
            "offset_x2": target_x2 - home2[0],
            "offset_y2": 0.0,
            "offset_z2": target_z2 - home2[2],
            "target_x1": target_x1,
            "target_y1": home1[1],
            "target_z1": target_z1,
            "target_x2": target_x2,
            "target_y2": home2[1],
            "target_z2": target_z2,
        }

    def apply_press_target(self, target: Dict[str, float], contact_mode: str) -> None:
        if "target_x1" not in target or "target_x2" not in target:
            super().apply_press_target(target, contact_mode)
            return

        self.get_logger().info(
            f"ABSOLUTE MODE: target_x1={target['target_x1']:.4f}, target_x2={target['target_x2']:.4f} "
            f"(offset_x1={target['offset_x1']:.4f}, offset_x2={target['offset_x2']:.4f})"
        )
        self.current_contact_mode = contact_mode
        self.current_offset_x1 = target["offset_x1"]
        self.current_offset_y1 = target["offset_y1"]
        self.current_offset_z1 = target["offset_z1"]
        self.current_offset_x2 = target["offset_x2"]
        self.current_offset_y2 = target["offset_y2"]
        self.current_offset_z2 = target["offset_z2"]
        self.current_press_offset_x1 = target["offset_x1"]
        self.current_press_offset_y1 = target["offset_y1"]
        self.current_press_offset_z1 = target["offset_z1"]
        self.current_press_offset_x2 = target["offset_x2"]
        self.current_press_offset_y2 = target["offset_y2"]
        self.current_press_offset_z2 = target["offset_z2"]
        self.current_directional_target_1 = None
        self.current_directional_target_2 = None

        self.publish_absolute_targets(
            (target["target_x1"], target["target_y1"], target["target_z1"]),
            (target["target_x2"], target["target_y2"], target["target_z2"]),
            repeats=self.command_repeats,
            dt=self.command_dt,
        )

    def cb_contact1(self, msg: WrenchStamped) -> None:
        super().cb_contact1(msg)
        self.contact_force_vec_1 = (
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        )
        self.contact_torque_vec_1 = (
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        )

    def cb_contact2(self, msg: WrenchStamped) -> None:
        super().cb_contact2(msg)
        self.contact_force_vec_2 = (
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        )
        self.contact_torque_vec_2 = (
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        )

    def publish_offsets(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        repeats: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> None:
        self.current_offset_y1 = y1
        self.current_offset_z1 = z1
        self.current_offset_y2 = y2
        self.current_offset_z2 = z2
        self.current_press_offset_y1 = y1
        self.current_press_offset_z1 = z1
        self.current_press_offset_y2 = y2
        self.current_press_offset_z2 = z2
        self.current_directional_target_1 = None
        self.current_directional_target_2 = None
        super().publish_offsets(x1, y1, z1, x2, y2, z2, repeats=repeats, dt=dt)

    def make_absolute_pose(self, x: float, y: float, z: float) -> PoseStamped:
        """Create a pose with position (x, y, z) and perfectly horizontal gripper orientation.
        
        Horizontal orientation (identity quaternion) is critical for:
        - Stable head-on compression against the spring
        - Consistent effective contact point at push_axis_height
        - Preventing tilt/roll that would asymmetrically load the spring
        """
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "absolute"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        # Identity quaternion: (x=0, y=0, z=0, w=1) = horizontal gripper
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def publish_absolute_targets(
        self,
        target1: Tuple[float, float, float],
        target2: Tuple[float, float, float],
        repeats: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> None:
        repeat_count = self.directional_command_repeats if repeats is None else repeats
        step_dt = self.command_dt if dt is None else dt

        msg1 = self.make_absolute_pose(*target1)
        msg2 = self.make_absolute_pose(*target2)

        for _ in range(repeat_count):
            stamp = self.get_clock().now().to_msg()
            msg1.header.stamp = stamp
            msg2.header.stamp = stamp
            self.pub1.publish(msg1)
            self.pub2.publish(msg2)
            rclpy.spin_once(self, timeout_sec=step_dt)

    def ee_separation(self) -> float:
        """3-D Euclidean distance between the two end-effector positions."""
        p1 = self.current_cartesian_position(1)
        p2 = self.current_cartesian_position(2)
        if p1 is None or p2 is None:
            return float("inf")  # unknown → don't trip the safety check
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )

    def ee_x_gap(self) -> float:
        """Absolute x-axis distance between the two end-effector positions."""
        p1 = self.current_cartesian_position(1)
        p2 = self.current_cartesian_position(2)
        if p1 is None or p2 is None:
            return 0.0  # unknown → don't trip the safety check
        return abs(p1[0] - p2[0])

    def separation_safe(self) -> bool:
        """Return False when EEs are closer than min_ee_separation."""
        return self.ee_separation() >= self.min_ee_separation

    def x_gap_safe(self) -> bool:
        """Return False when the grippers' x-axis gap exceeds max_ee_x_gap."""
        if self.max_ee_x_gap is None:
            return True
        return self.ee_x_gap() <= self.max_ee_x_gap

    def pose_safe(self) -> bool:
        """Return False when either EE has moved too far in the -x direction.

        The per-robot x floors are computed once at stage 2 from the desired home
        position minus the full press depth minus a safety margin.  Before stage 2
        the floors are None and the check always passes (returns True).
        """
        p1 = self.current_cartesian_position(1)
        p2 = self.current_cartesian_position(2)
        if self.min_ee_x_1 is not None and p1 is not None and p1[0] < self.min_ee_x_1:
            self.get_logger().error(
                f"Robot1 EE x {p1[0]:.4f} m below floor {self.min_ee_x_1:.4f} m — aborting"
            )
            return False
        if self.min_ee_x_2 is not None and p2 is not None and p2[0] < self.min_ee_x_2:
            self.get_logger().error(
                f"Robot2 EE x {p2[0]:.4f} m below floor {self.min_ee_x_2:.4f} m — aborting"
            )
            return False
        return True

    def current_cartesian_position(self, robot_id: int) -> Optional[Tuple[float, float, float]]:
        point = self.ee1 if robot_id == 1 else self.ee2
        if point is not None:
            return (point.x, point.y, point.z)

        pose = self.des1 if robot_id == 1 else self.des2
        if pose is not None:
            return (pose.position.x, pose.position.y, pose.position.z)

        return None

    def capture_mutual_contact_anchor(self) -> bool:
        anchor1 = self.current_cartesian_position(1)
        anchor2 = self.current_cartesian_position(2)
        if anchor1 is None or anchor2 is None:
            return False

        self.mutual_contact_anchor_1 = self.normalize_push_axis_anchor(anchor1)
        self.mutual_contact_anchor_2 = self.normalize_push_axis_anchor(anchor2)
        self.last_mutual_contact_anchor_1 = self.mutual_contact_anchor_1
        self.last_mutual_contact_anchor_2 = self.mutual_contact_anchor_2
        return True

    def ensure_mutual_contact_anchor(self) -> bool:
        if self.mutual_contact_anchor_1 is not None and self.mutual_contact_anchor_2 is not None:
            return True
        return self.capture_mutual_contact_anchor()

    def directional_waypoint_target_position(self, robot_id: int) -> Optional[Tuple[float, float, float]]:
        target = self.current_directional_target_1 if robot_id == 1 else self.current_directional_target_2
        if target is not None:
            return target

        desired = self.des1 if robot_id == 1 else self.des2
        if desired is None:
            return None

        return (
            desired.position.x,
            desired.position.y,
            desired.position.z,
        )

    def directional_waypoint_pose_error(self, robot_id: int) -> float:
        ee = self.ee1 if robot_id == 1 else self.ee2
        target = self.directional_waypoint_target_position(robot_id)
        if ee is None or target is None:
            return float("nan")
        return self.vector_norm(
            (
                ee.x - target[0],
                ee.y - target[1],
                ee.z - target[2],
            )
        )

    def directional_waypoint_pose_settled(self, position_tolerance_m: float = 0.05) -> bool:
        error_1 = self.directional_waypoint_pose_error(1)
        error_2 = self.directional_waypoint_pose_error(2)
        if not self.finite(error_1) or not self.finite(error_2):
            return False
        if error_1 > position_tolerance_m or error_2 > position_tolerance_m:
            return False
        # Velocity check removed — quasistatic speed enforced by small step size, not monitoring
        return True

    def wait_for_directional_waypoint_completion(self, timeout: Optional[float] = None) -> bool:
        timeout_s = self.directional_waypoint_timeout if timeout is None else timeout
        end_t = self.now_s() + timeout_s
        saw_robot1_active = bool(self.wp1_active)
        saw_robot2_active = bool(self.wp2_active)
        requires_target_tracking = (
            self.current_directional_target_1 is not None
            or self.current_directional_target_2 is not None
        )
        settled_since: Optional[float] = None

        while True:
            current_t = self.now_s()
            if current_t >= end_t:
                break
            robot1_active = bool(self.wp1_active)
            robot2_active = bool(self.wp2_active)
            saw_robot1_active = saw_robot1_active or robot1_active
            saw_robot2_active = saw_robot2_active or robot2_active
            if (
                saw_robot1_active
                and saw_robot2_active
                and not robot1_active
                and not robot2_active
                and not requires_target_tracking
            ):
                return True
            if saw_robot1_active and saw_robot2_active and self.directional_waypoint_pose_settled():
                if settled_since is None:
                    settled_since = current_t
                elif current_t - settled_since >= 0.15:
                    return True
            else:
                settled_since = None
            rclpy.spin_once(self, timeout_sec=0.05)

        return False

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
    def negate_vector(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (-vector[0], -vector[1], -vector[2])

    def bilateral_contact_confirmed(self) -> bool:
        return self.contact_detected(1) and self.contact_detected(2)

    def contact_detected(self, robot_id: int) -> bool:
        contact_valid = self.contact_valid_1 if robot_id == 1 else self.contact_valid_2
        if not contact_valid:
            return False

        axis_load = self.contact_axis_load(robot_id)
        if self.finite(axis_load):
            return axis_load >= self.contact_threshold(robot_id)

        delta = self.contact_force_delta(robot_id)
        delta_norm = self.vector_norm(delta)
        if self.finite(delta_norm):
            return delta_norm >= self.contact_threshold(robot_id)

        return super().contact_detected(robot_id)

    def linear_force(self, robot_id: int) -> Optional[Tuple[float, float, float]]:
        return self.contact_force_vec_1 if robot_id == 1 else self.contact_force_vec_2

    def contact_axis_load(self, robot_id: int) -> float:
        contact_fx_mag = self.contact_fx_mag_1 if robot_id == 1 else self.contact_fx_mag_2
        if self.finite(contact_fx_mag):
            return contact_fx_mag

        vector = self.linear_force(robot_id)
        if vector is None or not self.finite(vector[0]):
            return float("nan")
        return abs(vector[0])

    def contact_force_delta(self, robot_id: int) -> Optional[Tuple[float, float, float]]:
        """Force minus idle baseline; falls back to raw force if baseline not captured."""
        current = self.contact_force_vec_1 if robot_id == 1 else self.contact_force_vec_2
        baseline = self.baseline_force_vec_1 if robot_id == 1 else self.baseline_force_vec_2
        if current is None:
            return None
        if baseline is None:
            return current
        return (
            current[0] - baseline[0],
            current[1] - baseline[1],
            current[2] - baseline[2],
        )

    def projected_force(self, robot_id: int, direction: Optional[Tuple[float, float, float]]) -> float:
        vector = self.linear_force(robot_id)
        if vector is None or direction is None:
            return float("nan")
        if robot_id == 2:
            vector = self.negate_vector(vector)
        return self.dot_product(vector, direction)

    def compute_mutual_press_direction(self) -> Optional[Tuple[float, float, float]]:
        if not self.bilateral_contact_confirmed():
            self.current_direction_alignment = float("nan")
            return None

        direction1 = self.normalize_vector(self.contact_force_delta(1))
        direction2 = self.normalize_vector(self.contact_force_delta(2))
        if direction1 is None or direction2 is None:
            self.current_direction_alignment = float("nan")
            return None

        # Zero out Z component — press direction should be purely horizontal
        # to maintain constant push axis height. Spurious Z forces from noise
        # or gravity compensation errors would otherwise cause arms to drift
        # vertically during the press.
        direction1 = (direction1[0], direction1[1], 0.0)
        direction2 = (direction2[0], direction2[1], 0.0)
        direction1 = self.normalize_vector(direction1)
        direction2 = self.normalize_vector(direction2)
        if direction1 is None or direction2 is None:
            self.current_direction_alignment = float("nan")
            return None

        # When a baseline is captured, check for parallel reaction forces first.
        # Both robots pressing from the SAME SIDE produce force deltas in the SAME
        # direction. The force sensor measures the force ON the spring FROM the arm,
        # so the force delta direction IS the press direction (no negation needed).
        # Example: both arms push spring in +x → force deltas in +x → press +x.
        using_baseline = (
            self.baseline_force_vec_1 is not None
            and self.baseline_force_vec_2 is not None
        )
        if using_baseline:
            same_dir_alignment = self.dot_product(direction1, direction2)
            if same_dir_alignment >= self.mutual_direction_alignment_min:
                self.current_direction_alignment = same_dir_alignment
                combined = (
                    direction1[0] + direction2[0],
                    direction1[1] + direction2[1],
                    0.0,
                )
                common = self.normalize_vector(combined)
                # Return the common direction directly (NOT negated) — this is the
                # direction both arms should move to continue pressing into the spring.
                return common if common is not None else direction1

        # Fallback: check for opposing reaction forces (robots pressing from opposite sides).
        opposing_direction2 = self.negate_vector(direction2)

        alignment = self.dot_product(direction1, opposing_direction2)
        self.current_direction_alignment = alignment
        if alignment < self.mutual_direction_alignment_min:
            return None

        combined = (
            direction1[0] + opposing_direction2[0],
            direction1[1] + opposing_direction2[1],
            0.0,
        )
        direction = self.normalize_vector(combined)
        if direction is not None:
            return self.negate_vector(direction)

        norm1 = self.vector_norm(self.contact_force_vec_1)
        norm2 = self.vector_norm(self.contact_force_vec_2)
        return self.negate_vector(direction1) if norm1 >= norm2 else direction2

    def compute_directional_press_target(
        self,
        distance1: float,
        distance2: float,
        direction: Tuple[float, float, float],
    ) -> Dict[str, float]:
        return {
            "offset_x1": distance1 * direction[0],
            "offset_y1": distance1 * direction[1],
            "offset_z1": distance1 * direction[2],
            "offset_x2": distance2 * direction[0],
            "offset_y2": distance2 * direction[1],
            "offset_z2": distance2 * direction[2],
        }

    def compute_absolute_directional_press_target(
        self,
        distance1: float,
        distance2: float,
        direction: Tuple[float, float, float],
    ) -> Optional[Dict[str, float]]:
        """Compute directional press target with z locked to anchor height and horizontal gripper orientation.
        
        During contact and compression phases:
        - z is HARD-LOCKED to anchor height (push_axis_height) to ensure consistent spring contact
        - offset_z is forced to 0.0 to reject spurious Z from force noise
        - gripper stays perfectly horizontal (via make_absolute_pose -> identity quaternion)
        """
        if not self.ensure_mutual_contact_anchor():
            return None

        anchor1 = self.mutual_contact_anchor_1
        anchor2 = self.mutual_contact_anchor_2
        if anchor1 is None or anchor2 is None:
            return None

        target = self.compute_directional_press_target(distance1, distance2, direction)
        # Enforce minimum distance from base - HARD SAFETY CONSTRAINT
        target_x1 = max(anchor1[0] + target["offset_x1"], self.MIN_DISTANCE_FROM_BASE)
        target_x2 = max(anchor2[0] + target["offset_x2"], self.MIN_DISTANCE_FROM_BASE)
        # Force Z to anchor height — press direction is always horizontal.
        # This prevents spurious Z offsets from force noise or gravity errors
        # from pushing arms above/below the configured push axis height.
        # Horizontal gripper orientation is maintained via make_absolute_pose (identity quaternion).
        target.update(
            {
                "target_x1": target_x1,
                "target_y1": anchor1[1] + target["offset_y1"],
                "target_z1": anchor1[2],
                "target_x2": target_x2,
                "target_y2": anchor2[1] + target["offset_y2"],
                "target_z2": anchor2[2],
                "offset_z1": 0.0,
                "offset_z2": 0.0,
            }
        )
        return target

    def apply_directional_press_target(
        self,
        distance1: float,
        distance2: float,
        direction: Tuple[float, float, float],
        contact_mode: str,
    ) -> Optional[Dict[str, float]]:
        target = self.compute_absolute_directional_press_target(distance1, distance2, direction)
        if target is None:
            return None
        self.publish_directional_press_target(target, contact_mode)
        return target

    def _clamp_offsets_for_safety(self, offset_x1: float, offset_y1: float, offset_z1: float,
                                   offset_x2: float, offset_y2: float, offset_z2: float) -> Tuple[float, float, float, float, float, float]:
        """Clamp offsets so that anchor + offset >= MIN_DISTANCE_FROM_BASE for both arms.
        
        This is a safety clamp to prevent the controller from commanding positions
        below the minimum safe distance from the robot base. The controller applies
        offsets to trajectory targets without clamping, so we must clamp here.
        """
        if self.mutual_contact_anchor_1 is not None and self.mutual_contact_anchor_2 is not None:
            anchor1_x = self.mutual_contact_anchor_1[0]
            anchor2_x = self.mutual_contact_anchor_2[0]
            
            # Compute what the absolute X would be
            abs_x1 = anchor1_x + offset_x1
            abs_x2 = anchor2_x + offset_x2
            
            # Clamp to MIN_DISTANCE_FROM_BASE
            clamped_x1 = max(abs_x1, self.MIN_DISTANCE_FROM_BASE)
            clamped_x2 = max(abs_x2, self.MIN_DISTANCE_FROM_BASE)
            
            # Recompute offsets from clamped absolute positions
            offset_x1 = clamped_x1 - anchor1_x
            offset_x2 = clamped_x2 - anchor2_x
        
        return offset_x1, offset_y1, offset_z1, offset_x2, offset_y2, offset_z2

    def publish_directional_press_target(self, target: Dict[str, float], contact_mode: str) -> None:
        self.current_contact_mode = contact_mode
        self.current_offset_x1 = target["offset_x1"]
        self.current_offset_y1 = target["offset_y1"]
        self.current_offset_z1 = target["offset_z1"]
        self.current_offset_x2 = target["offset_x2"]
        self.current_offset_y2 = target["offset_y2"]
        self.current_offset_z2 = target["offset_z2"]
        self.current_press_offset_x1 = target["offset_x1"]
        self.current_press_offset_y1 = target["offset_y1"]
        self.current_press_offset_z1 = target["offset_z1"]
        self.current_press_offset_x2 = target["offset_x2"]
        self.current_press_offset_y2 = target["offset_y2"]
        self.current_press_offset_z2 = target["offset_z2"]
        self.current_directional_target_1 = (
            target["target_x1"],
            target["target_y1"],
            target["target_z1"],
        )
        self.current_directional_target_2 = (
            target["target_x2"],
            target["target_y2"],
            target["target_z2"],
        )
        if self.mutual_contact_anchor_1 is not None:
            self.last_mutual_contact_anchor_1 = self.mutual_contact_anchor_1
        if self.mutual_contact_anchor_2 is not None:
            self.last_mutual_contact_anchor_2 = self.mutual_contact_anchor_2
        self.last_directional_target_1 = self.current_directional_target_1
        self.last_directional_target_2 = self.current_directional_target_2

        # Publish as absolute targets (frame_id="absolute") to maintain Z-height
        # at push_axis and ensure X >= MIN_DISTANCE_FROM_BASE throughout Stage 4.
        # Using absolute targets prevents drift that occurs with offset mode when
        # the controller's trajectory state diverges from the anchor position.
        self.publish_absolute_targets(
            (float(target["target_x1"]), float(target["target_y1"]), float(target["target_z1"])),
            (float(target["target_x2"]), float(target["target_y2"]), float(target["target_z2"])),
            repeats=self.directional_command_repeats,
        )

    def sustain_directional_press_target(
        self,
        target: Dict[str, float],
        duration: float,
        velocity_limit: float,
        contact_mode: str,
        contact_loss_reason: str,
        contact_loss_message: str,
        velocity_reason: str,
        velocity_message: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        end_t = self.now_s() + duration
        saw_waypoint_activity = False

        while True:
            self.publish_directional_press_target(target, contact_mode)
            cycle_end = min(end_t, self.now_s() + self.directional_refresh_interval)

            while self.now_s() < cycle_end:
                rclpy.spin_once(self, timeout_sec=0.05)
                saw_waypoint_activity = saw_waypoint_activity or bool(self.wp1_active) or bool(self.wp2_active)
                if not self.bilateral_contact_confirmed():
                    self.set_mutual_contact_state(False, contact_loss_reason)
                    return False, contact_loss_message, self.snapshot()
                # NOTE: separation_safe() (3-D Euclidean) is intentionally skipped here.
                # Both robots press co-directionally from the same side with a permanent
                # ~18 mm structural Z-offset, making the 3-D distance permanently below
                # the 20 mm floor.  pose_safe() and x_gap_safe() cover real safety.
                # Velocity check removed — quasistatic speed enforced by small step size.
                if not self.x_gap_safe():
                    self.get_logger().error(
                        f"EE x gap {self.ee_x_gap():.4f} m above safety limit "
                        f"{self.max_ee_x_gap:.4f} m — aborting directional hold"
                    )
                    self.set_mutual_contact_state(False, "x_gap_above_safety_limit")
                    return False, "inter-robot x gap above safety limit", self.snapshot()
                if not self.pose_safe():
                    self.set_mutual_contact_state(False, "pose_below_x_floor_safety_limit")
                    return False, "EE pose below x floor safety limit", self.snapshot()
                # Velocity check removed — quasistatic speed enforced by small step size

            if self.now_s() >= end_t:
                break

        if not saw_waypoint_activity:
            self.set_mutual_contact_state(False, "directional_waypoint_not_observed")
            return False, "directional waypoint activity not observed", self.snapshot()

        return True, "", self.snapshot()

    def set_mutual_contact_state(self, active: bool, reason: str) -> None:
        if active == self.mutual_contact_active:
            return

        self.mutual_contact_active = active
        if active:
            self.capture_mutual_contact_anchor()
            self.bag_recorder.start(reason)
            self.external_camera_trigger.start(self._external_camera_trigger_name(reason), reason)
        else:
            self.directional_press_enabled = False
            self.current_mutual_press_dir = None
            self.current_direction_alignment = float("nan")
            self.mutual_contact_anchor_1 = None
            self.mutual_contact_anchor_2 = None
            self.current_directional_target_1 = None
            self.current_directional_target_2 = None
            self.bag_recorder.stop(reason)
            self.external_camera_trigger.stop(reason)

    def configure_external_camera_trigger(self, trigger_path: Optional[str]) -> None:
        """Point the harness at a host-side text-trigger file.

        Calling this with a non-empty path enables START/STOP writes during
        ``set_mutual_contact_state`` transitions. Calling with ``None`` or
        an empty string disables external-camera triggering. The
        ``mutual_contact_window`` index is preserved across reconfiguration
        so paired START/STOP markers stay numbered consistently.
        """
        previous = self.external_camera_trigger
        log_path = previous.log_path if previous is not None else os.path.join(self.run_dir, "external_camera_trigger.log")
        new_trigger = ExternalCameraTrigger(trigger_path, log_path)
        if previous is not None:
            new_trigger.window_index = previous.window_index
        self.external_camera_trigger = new_trigger

    def _external_camera_trigger_name(self, reason: str) -> str:
        safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (reason or "window"))
        window_idx = getattr(self.bag_recorder, "window_index", 0) if self.bag_recorder is not None else 0
        return f"{safe_reason}_window_{window_idx:03d}"

    def snapshot(self) -> Dict[str, Any]:
        row = super().snapshot()

        ee_y_1 = self.ee1.y if self.ee1 is not None else float("nan")
        ee_z_1 = self.ee1.z if self.ee1 is not None else float("nan")
        ee_y_2 = self.ee2.y if self.ee2 is not None else float("nan")
        ee_z_2 = self.ee2.z if self.ee2 is not None else float("nan")
        desired_y_1 = self.des1.position.y if self.des1 is not None else float("nan")
        desired_z_1 = self.des1.position.z if self.des1 is not None else float("nan")
        desired_y_2 = self.des2.position.y if self.des2 is not None else float("nan")
        desired_z_2 = self.des2.position.z if self.des2 is not None else float("nan")

        direction = self.current_mutual_press_dir
        projected_force_1 = self.projected_force(1, direction)
        projected_force_2 = self.projected_force(2, direction)

        force1 = self.contact_force_vec_1 or (float("nan"), float("nan"), float("nan"))
        force2 = self.contact_force_vec_2 or (float("nan"), float("nan"), float("nan"))
        torque1 = self.contact_torque_vec_1 or (float("nan"), float("nan"), float("nan"))
        torque2 = self.contact_torque_vec_2 or (float("nan"), float("nan"), float("nan"))
        anchor1 = self.mutual_contact_anchor_1 or self.last_mutual_contact_anchor_1 or (float("nan"), float("nan"), float("nan"))
        anchor2 = self.mutual_contact_anchor_2 or self.last_mutual_contact_anchor_2 or (float("nan"), float("nan"), float("nan"))
        target1 = self.current_directional_target_1 or self.last_directional_target_1 or (float("nan"), float("nan"), float("nan"))
        target2 = self.current_directional_target_2 or self.last_directional_target_2 or (float("nan"), float("nan"), float("nan"))

        row.update(
            {
                "offset_y1": self.current_offset_y1,
                "offset_z1": self.current_offset_z1,
                "offset_y2": self.current_offset_y2,
                "offset_z2": self.current_offset_z2,
                "press_offset_y1": self.current_press_offset_y1,
                "press_offset_z1": self.current_press_offset_z1,
                "press_offset_y2": self.current_press_offset_y2,
                "press_offset_z2": self.current_press_offset_z2,
                "press_distance_1": self.current_press_distance_1,
                "press_distance_2": self.current_press_distance_2,
                "mutual_contact_active": float(self.mutual_contact_active),
                "directional_press_enabled": float(self.directional_press_enabled),
                "mutual_press_dir_x": direction[0] if direction is not None else float("nan"),
                "mutual_press_dir_y": direction[1] if direction is not None else float("nan"),
                "mutual_press_dir_z": direction[2] if direction is not None else float("nan"),
                "direction_alignment": self.current_direction_alignment,
                "projected_force_1": projected_force_1,
                "projected_force_2": projected_force_2,
                "bag_recording_active": float(self.bag_recorder.is_active()),
                "ee_y_1": ee_y_1,
                "ee_z_1": ee_z_1,
                "ee_y_2": ee_y_2,
                "ee_z_2": ee_z_2,
                "desired_y_1": desired_y_1,
                "desired_z_1": desired_z_1,
                "desired_y_2": desired_y_2,
                "desired_z_2": desired_z_2,
                "contact_fx_1": force1[0],
                "contact_fy_1": force1[1],
                "contact_fz_1": force1[2],
                "contact_tx_1": torque1[0],
                "contact_ty_1": torque1[1],
                "contact_tz_1": torque1[2],
                "contact_fx_2": force2[0],
                "contact_fy_2": force2[1],
                "contact_fz_2": force2[2],
                "contact_tx_2": torque2[0],
                "contact_ty_2": torque2[1],
                "contact_tz_2": torque2[2],
                "contact_force_norm_1": self.vector_norm(self.contact_force_vec_1),
                "contact_force_norm_2": self.vector_norm(self.contact_force_vec_2),
                "anchor_x_1": anchor1[0],
                "anchor_y_1": anchor1[1],
                "anchor_z_1": anchor1[2],
                "anchor_x_2": anchor2[0],
                "anchor_y_2": anchor2[1],
                "anchor_z_2": anchor2[2],
                "target_x_1": target1[0],
                "target_y_1": target1[1],
                "target_z_1": target1[2],
                "target_x_2": target2[0],
                "target_y_2": target2[1],
                "target_z_2": target2[2],
            }
        )
        return row

    def log_sync_step(self, command_distance: float) -> None:
        row = self.snapshot()
        row["command_distance"] = command_distance
        self.append_csv_row(self.sync_csv_path, self.sync_columns, row)

    def write_results(self, results: List[Dict[str, Any]]) -> None:
        with open(self.results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        # Generate publication-quality plots after writing results
        self.generate_publication_plots()

    def generate_publication_plots(self) -> None:
        """Generate publication-quality plots from the harness CSV data.
        
        Creates figures showing EE positions, contact forces, and offsets
        over time. Saved to the run_dir as PDF and PNG.
        """
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
            df['time_s'] = df.index * 0.02  # Assume 50 Hz default
        
        # Apply publication style
        matplotlib.rcParams.update(PUB_RC)
        
        # Define subplot groups
        plot_groups = [
            {
                'title': 'End-Effector Positions',
                'columns': [
                    ('ee_x_1', 'EE X Robot1'),
                    ('ee_y_1', 'EE Y Robot1'),
                    ('ee_z_1', 'EE Z Robot1'),
                    ('ee_x_2', 'EE X Robot2'),
                    ('ee_y_2', 'EE Y Robot2'),
                    ('ee_z_2', 'EE Z Robot2'),
                ],
                'ylabel': 'Position (m)',
            },
            {
                'title': 'Contact Forces (X-axis)',
                'columns': [
                    ('contact_fx_1', 'Robot1 Fx'),
                    ('contact_fx_2', 'Robot2 Fx'),
                ],
                'ylabel': 'Force (N)',
            },
            {
                'title': 'Contact Forces (Y-axis)',
                'columns': [
                    ('contact_fy_1', 'Robot1 Fy'),
                    ('contact_fy_2', 'Robot2 Fy'),
                ],
                'ylabel': 'Force (N)',
            },
            {
                'title': 'Contact Forces (Z-axis)',
                'columns': [
                    ('contact_fz_1', 'Robot1 Fz'),
                    ('contact_fz_2', 'Robot2 Fz'),
                ],
                'ylabel': 'Force (N)',
            },
            {
                'title': 'Pre-Contact Offsets',
                'columns': [
                    ('offset_x1', 'Robot1 Offset X'),
                    ('offset_y1', 'Robot1 Offset Y'),
                    ('offset_z1', 'Robot1 Offset Z'),
                    ('offset_x2', 'Robot2 Offset X'),
                    ('offset_y2', 'Robot2 Offset Y'),
                    ('offset_z2', 'Robot2 Offset Z'),
                ],
                'ylabel': 'Offset (m)',
            },
        ]
        
        # Check for additional columns from contact-gated load harness
        if 'shared_load_n' in df.columns:
            plot_groups.append({
                'title': 'Shared Load and Load Step',
                'columns': [
                    ('shared_load_n', 'Shared Load (N)'),
                    ('load_step_index', 'Load Step Index'),
                ],
                'ylabel': 'Load (N) / Step',
            })
        
        # Generate one figure per group
        for group in plot_groups:
            # Filter to columns that exist and have data
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
                f"{group['title']}\nHarness Run {getattr(self, 'run_ts', 'unknown')}",
                fontsize=28, fontweight="bold", y=0.99,
            )
            
            for idx, (col, label) in enumerate(valid_cols):
                ax = axes[idx]
                
                # Determine color based on robot number
                if 'robot1' in label.lower() or col.endswith('_1'):
                    color = C_R1
                elif 'robot2' in label.lower() or col.endswith('_2'):
                    color = C_R2
                else:
                    color = "#2d6a2e"
                
                ax.plot(
                    df['time_s'].values, df[col].values,
                    color=color,
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
            
            # Add time label to bottom subplot
            axes[-1].set_xlabel("Time (s)", fontsize=24, fontweight="bold")
            
            plt.tight_layout()
            
            # Save figures
            col_names = '_'.join([c.replace(' ', '_') for c, _ in valid_cols])
            fig_name = f"harness_{group['title'].replace(' ', '_')}_{col_names}"
            fig_path_pdf = os.path.join(self.plot_save_dir or self.run_dir, f"{fig_name}.pdf")
            fig_path_png = os.path.join(self.plot_save_dir or self.run_dir, f"{fig_name}.png")
            
            plt.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
            plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved publication plot: {fig_path_pdf}")
            print(f"Saved publication plot: {fig_path_png}")

    def safe_abort(self) -> None:
        self.set_mutual_contact_state(False, "safe_abort")
        super().safe_abort()

    def stage2_idle(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "idle"
        info = self.snapshot()

        # Wait for arms to converge to start_position (≥ MIN_DISTANCE_FROM_BASE)
        # and correct Z heights before proceeding to Stage 3 push trajectory.
        convergence_timeout = 30.0  # seconds to wait for convergence
        convergence_start = self.now_s()
        converged = False
        z_convergence_tol = 0.010  # 10mm Z tolerance (relaxed for impedance dynamics)
        
        # Get target Z from YAML start_position (not controller's desired_z which changes during homing)
        target_z_1 = self._get_controller_param_float("robot1", "start_position.z", default=0.085)
        target_z_2 = self._get_controller_param_float("robot2", "start_position.z", default=0.080)
        
        while self.now_s() - convergence_start < convergence_timeout:
            self.spin_for(0.25)
            info = self.snapshot()
            
            ee_x_1 = info.get("ee_x_1", 0.0)
            ee_x_2 = info.get("ee_x_2", 0.0)
            ee_z_1 = info.get("ee_z_1", 0.0)
            ee_z_2 = info.get("ee_z_2", 0.0)
            desired_z_1 = info.get("desired_z_1", 0.0)
            desired_z_2 = info.get("desired_z_2", 0.0)
            
            x_converged = (ee_x_1 >= self.MIN_DISTANCE_FROM_BASE and 
                          ee_x_2 >= self.MIN_DISTANCE_FROM_BASE)
            z_converged = (abs(ee_z_1 - target_z_1) < z_convergence_tol and 
                          abs(ee_z_2 - target_z_2) < z_convergence_tol)
            
            if x_converged and z_converged:
                converged = True
                self.get_logger().info(
                    f"Stage 2: Arms converged to safe positions "
                    f"(X: ee_x_1={ee_x_1:.4f}, ee_x_2={ee_x_2:.4f} ≥ {self.MIN_DISTANCE_FROM_BASE}; "
                    f"Z: ee_z_1={ee_z_1:.4f}≈{target_z_1:.4f}, ee_z_2={ee_z_2:.4f}≈{target_z_2:.4f})"
                )
                break
            else:
                self.get_logger().info(
                    f"Stage 2: Waiting for convergence "
                    f"(X: ee_x_1={ee_x_1:.4f}, ee_x_2={ee_x_2:.4f}; "
                    f"Z: ee_z_1={ee_z_1:.4f}/{target_z_1:.4f}, ee_z_2={ee_z_2:.4f}/{target_z_2:.4f})"
                )
        
        if not converged:
            self.get_logger().warn(
                f"Stage 2: Convergence timeout ({convergence_timeout}s) — "
                f"proceeding with caution (ee_x_1={info.get('ee_x_1', 0.0):.4f}, "
                f"ee_x_2={info.get('ee_x_2', 0.0):.4f})"
            )
        
        # Capture baseline forces and compute safety floors
        self.baseline_force_vec_1 = self.contact_force_vec_1
        self.baseline_force_vec_2 = self.contact_force_vec_2
        
        # Compute per-robot x floors from current desired home positions.
        # Floor = home_x - precontact_depth - press_depth - safety_margin.
        # press_end_distance is kept in sync by ContactGatedLoadStepHarness
        # (set equal to max_additional_load_travel_m in its __init__) so the
        # floor is always conservative enough for the actual travel budget.
        # HARD SAFETY: clamp to MIN_DISTANCE_FROM_BASE to prevent base collision.
        if self.des1 is not None:
            computed_floor_1 = (
                self.des1.position.x
                - self.precontact_end_distance
                - self.press_end_distance
                - self.min_x_floor_margin
            )
            self.min_ee_x_1 = max(computed_floor_1, self.MIN_DISTANCE_FROM_BASE)
        if self.des2 is not None:
            computed_floor_2 = (
                self.des2.position.x
                - self.precontact_end_distance
                - self.press_end_distance
                - self.min_x_floor_margin
            )
            self.min_ee_x_2 = max(computed_floor_2, self.MIN_DISTANCE_FROM_BASE)
        if self.max_ee_x_gap is None and self.des1 is not None and self.des2 is not None:
            self.max_ee_x_gap = (
                abs(self.des1.position.x - self.des2.position.x)
                + self.max_ee_x_gap_margin
            )
        return True, "idle stabilization complete", info

    def stage3_sync_move(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "precontact"
        stage_info: List[Dict[str, Any]] = []

        if self.bilateral_contact_confirmed():
            initial_direction = self.compute_mutual_press_direction()
            if initial_direction is not None:
                self.current_press_distance_1 = 0.0
                self.current_press_distance_2 = 0.0
                self.current_mutual_press_dir = initial_direction
                snap = self.snapshot()
                snap["command_distance"] = 0.0
                stage_info.append(snap)
                self.log_sync_step(0.0)
                self.set_mutual_contact_state(True, "bilateral_contact_present_at_stage_start")
                return True, "bilateral contact established", {"forward_phase_observed": False, "started_in_mutual_contact": True, "samples": stage_info}

            self.get_logger().info(
                "Bilateral contact present at stage start but mutual press direction is invalid; continuing precontact search"
            )

        forward_phase_observed = self.wait_for_forward_phase(timeout=10.0)
        if not forward_phase_observed:
            self.get_logger().info(
                "Forward-phase sync not observed within timeout; continuing with cautious precontact approach"
            )

        if self.des1 is None or self.des2 is None:
            return False, "desired poses unavailable", {"forward_phase_observed": forward_phase_observed, "samples": stage_info}

        self.spin_for(0.10)
        self.capture_precontact_baseline()

        precontact_home_1 = (
            self.des1.position.x,
            self.des1.position.y,
            self.des1.position.z,
        )
        precontact_home_2 = (
            self.des2.position.x,
            self.des2.position.y,
            self.des2.position.z,
        )

        offset1 = self.precontact_start_x_offset
        offset2 = self.precontact_start_x_offset
        # PHASE 1: Establish z-height at push_axis BEFORE any x-movement
        # This prevents collision with metallic plate when moving in x at wrong z
        target_z_offset_1 = self.configured_push_axis_height(precontact_home_1[2]) - precontact_home_1[2]
        target_z_offset_2 = self.configured_push_axis_height(precontact_home_2[2]) - precontact_home_2[2]
        current_z_offset_1 = 0.0
        current_z_offset_2 = 0.0
        z_step = self.press_step  # same magnitude as x-step for coordinated approach
        
        self.get_logger().info(
            f"PHASE 1: Establishing z-height first (z_offset_1={target_z_offset_1:.4f}, z_offset_2={target_z_offset_2:.4f})"
        )
        
        # Increment z-offsets only until both reach target z-height
        max_z_iterations = int(max(abs(target_z_offset_1), abs(target_z_offset_2)) / z_step) + 5
        for z_iter in range(max_z_iterations):
            # Step z incrementally toward the push-axis height target
            current_z_offset_1 = self._step_offset_toward(current_z_offset_1, target_z_offset_1, z_step)
            current_z_offset_2 = self._step_offset_toward(current_z_offset_2, target_z_offset_2, z_step)
            
            # Command z-offset only (x-offsets stay at precontact_start_x_offset)
            target = self.compute_absolute_precontact_target(
                precontact_home_1,
                precontact_home_2,
                offset1,  # Keep at start offset, not incrementing yet
                offset2,
            )
            # Override the z-offsets with incrementally stepped values
            target["offset_z1"] = current_z_offset_1
            target["offset_z2"] = current_z_offset_2
            target["target_z1"] = precontact_home_1[2] + current_z_offset_1
            target["target_z2"] = precontact_home_2[2] + current_z_offset_2
            
            self.apply_press_target(target, "none")
            self.spin_for(self.command_sample_delay)
            
            # Check if both z-offsets have reached their targets
            z1_done = abs(current_z_offset_1 - target_z_offset_1) < 1e-6
            z2_done = abs(current_z_offset_2 - target_z_offset_2) < 1e-6
            if z1_done and z2_done:
                self.get_logger().info("PHASE 1 complete: z-height established at push_axis")
                break
        
        # Wait for arms to actually converge to target Z before starting X movement
        z_settle_timeout = 10.0  # seconds to wait for Z convergence
        z_settle_start = self.now_s()
        z_settled = False
        z_settle_tol = 0.003  # 3mm tolerance for Z convergence
        
        while self.now_s() - z_settle_start < z_settle_timeout:
            self.spin_for(0.10)
            info = self.snapshot()
            ee_z_1 = info.get("ee_z_1", 0.0)
            ee_z_2 = info.get("ee_z_2", 0.0)
            target_z_1 = precontact_home_1[2] + target_z_offset_1
            target_z_2 = precontact_home_2[2] + target_z_offset_2
            
            if abs(ee_z_1 - target_z_1) < z_settle_tol and abs(ee_z_2 - target_z_2) < z_settle_tol:
                z_settled = True
                self.get_logger().info(
                    f"PHASE 1 Z-settle: Arms converged to target Z "
                    f"(ee_z_1={ee_z_1:.4f}≈{target_z_1:.4f}, ee_z_2={ee_z_2:.4f}≈{target_z_2:.4f})"
                )
                break
            else:
                self.get_logger().info(
                    f"PHASE 1 Z-settle: Waiting for convergence "
                    f"(ee_z_1={ee_z_1:.4f}/{target_z_1:.4f}, ee_z_2={ee_z_2:.4f}/{target_z_2:.4f})"
                )
        
        if not z_settled:
            self.get_logger().warn(
                f"PHASE 1 Z-settle: Timeout ({z_settle_timeout}s) — proceeding with caution"
            )
        
        # PHASE 2: Now move in x to establish contact (z is already correct)
        self.get_logger().info("PHASE 2: Moving in x to establish bilateral contact")
        
        # The robots face each other in world x, so stage 3 must drive robot1 in -x
        # and robot2 in +x. press_step_2 derives its sign from robot2's configured
        # precontact endpoint so the clamp logic stays correct if that endpoint changes.
        press_step_1 = math.copysign(self.press_step, self.precontact_end_x_offset)
        press_step_2 = math.copysign(self.press_step, self.precontact_end_x_offset_2)
        contact1 = False
        contact2 = False
        for _ in range(self.max_precontact_iterations):
            both = contact1 and contact2
            # When one arm makes contact first, hold it in place until the other also
            # makes contact — then both press together toward full depth.
            # This ensures mutual contact is confirmed before any further pressing.
            if not contact1 or both:
                if press_step_1 > 0:
                    offset1 = min(offset1 + press_step_1, self.precontact_end_x_offset)
                else:
                    offset1 = max(offset1 + press_step_1, self.precontact_end_x_offset)
            if not contact2 or both:
                if press_step_2 > 0:
                    offset2 = min(offset2 + press_step_2, self.precontact_end_x_offset_2)
                else:
                    offset2 = max(offset2 + press_step_2, self.precontact_end_x_offset_2)
            # Enforce minimum distance from base (12 cm): offset >= min_offset_from_base
            offset1 = max(offset1, self.min_offset_from_base)
            offset2 = max(offset2, self.min_offset_from_base)

            # The controller interprets frame_id="offset" as a delta from the
            # current trajectory target. Anchor stage-3 commands once so a held
            # precontact arm does not keep walking away while the other catches up.
            target = self.compute_absolute_precontact_target(
                precontact_home_1,
                precontact_home_2,
                offset1,
                offset2,
            )
            # Maintain the z-offsets established in phase 1
            target["offset_z1"] = current_z_offset_1
            target["offset_z2"] = current_z_offset_2
            target["target_z1"] = precontact_home_1[2] + current_z_offset_1
            target["target_z2"] = precontact_home_2[2] + current_z_offset_2

            contact_mode = (
                "robot1_only"
                if contact1 and not contact2
                else "robot2_only"
                if contact2 and not contact1
                else "bilateral"
                if contact1 and contact2
                else "none"
            )
            self.apply_press_target(target, contact_mode)
            self.spin_for(self.command_sample_delay)

            contact1 = self.contact_detected(1)
            contact2 = self.contact_detected(2)

            self.current_press_distance_1 = abs(offset1)
            self.current_press_distance_2 = abs(offset2)

            snap = self.snapshot()
            snap["command_distance"] = max(abs(offset1), abs(offset2))
            stage_info.append(snap)
            self.log_sync_step(max(abs(offset1), abs(offset2)))

            if contact1 and contact2:
                at_depth = (
                    abs(offset1) >= self.precontact_end_distance - 1e-9
                    and abs(offset2) >= self.precontact_end_distance - 1e-9
                )
                if at_depth:
                    # Keep press distances at the commanded depth so stage4 continues from here
                    # rather than withdrawing to the spring-lagged actual EE position.
                    self.current_press_distance_1 = self.precontact_end_distance
                    self.current_press_distance_2 = self.precontact_end_distance
                    self.set_mutual_contact_state(True, "bilateral_contact_established")
                    # Override the anchor (captured from actual EE by set_mutual_contact_state)
                    # with the commanded home positions.  Stage4 will then compute targets as
                    # anchor + precontact_end_distance * direction = same as the last stage3
                    # command, eliminating the withdrawal caused by spring compliance lag.
                    anchor_z_1 = self.configured_push_axis_height(precontact_home_1[2])
                    anchor_z_2 = self.configured_push_axis_height(precontact_home_2[2])
                    self.mutual_contact_anchor_1 = (
                        precontact_home_1[0],
                        precontact_home_1[1],
                        anchor_z_1,
                    )
                    self.mutual_contact_anchor_2 = (
                        precontact_home_2[0],
                        precontact_home_2[1],
                        anchor_z_2,
                    )
                    return True, "bilateral contact established", {"forward_phase_observed": forward_phase_observed, "started_in_mutual_contact": False, "samples": stage_info}

            # NOTE: separation_safe() (3-D Euclidean) is NOT checked here because both
            # robots press co-directionally from the same side and have a permanent ~18 mm
            # structural Z-offset between their mounts.  That offset makes the 3-D distance
            # permanently below the 20 mm floor regardless of actual contact state.
            # pose_safe() and x_gap_safe() cover all real safety risks.
            # Velocity check removed - low speed is enforced by small step size.
            if not self.x_gap_safe():
                self.get_logger().error(
                    f"EE x gap {self.ee_x_gap():.4f} m above safety limit "
                    f"{self.max_ee_x_gap:.4f} m — aborting"
                )
                return False, "inter-robot x gap above safety limit", {"forward_phase_observed": forward_phase_observed, "started_in_mutual_contact": False, "samples": stage_info}
            if not self.pose_safe():
                return False, "EE pose below x floor safety limit", {"forward_phase_observed": forward_phase_observed, "started_in_mutual_contact": False, "samples": stage_info}

        return False, "bilateral contact not established", {"forward_phase_observed": forward_phase_observed, "started_in_mutual_contact": False, "samples": stage_info}

    def stage4_hold(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "hold"
        if not self.bilateral_contact_confirmed():
            self.set_mutual_contact_state(False, "hold_requires_bilateral_contact")
            return False, "mutual contact not maintained for hold", self.snapshot()

        direction = self.compute_mutual_press_direction()
        if direction is None:
            self.set_mutual_contact_state(False, "hold_direction_unavailable")
            return False, "mutual contact direction unavailable", self.snapshot()

        self.current_mutual_press_dir = direction
        self.directional_press_enabled = True
        hold_distance_1 = max(self.current_press_distance_1, self.press_step)
        hold_distance_2 = max(self.current_press_distance_2, self.press_step)
        self.current_press_distance_1 = hold_distance_1
        self.current_press_distance_2 = hold_distance_2
        target = self.apply_directional_press_target(
            hold_distance_1,
            hold_distance_2,
            direction,
            "directional_hold",
        )
        if target is None:
            self.set_mutual_contact_state(False, "hold_target_unavailable")
            return False, "directional target unavailable", self.snapshot()

        ok, message, info = self.sustain_directional_press_target(
            target,
            duration=self.directional_hold_duration,
            velocity_limit=self.hold_vel_limit,
            contact_mode="directional_hold",
            contact_loss_reason="mutual_contact_lost_during_hold",
            contact_loss_message="mutual contact lost during directional hold",
            velocity_reason="velocity_spike_during_hold",
            velocity_message="velocity spike during directional hold",
        )
        if not ok:
            return False, message, info

        return True, "directional hold stable", info

    def stage5_adaptive_press(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "compress"

        if not self.enable_directional_deep_press:
            if not self.bilateral_contact_confirmed():
                self.set_mutual_contact_state(False, "mutual_contact_lost_before_stage5_hold")
                return False, "mutual contact lost before sustained directional hold", self.snapshot()

            direction = self.compute_mutual_press_direction()
            if direction is None:
                self.set_mutual_contact_state(False, "directional_hold_stage5_direction_unavailable")
                return False, "mutual contact direction unavailable", self.snapshot()

            self.current_mutual_press_dir = direction
            target = self.compute_absolute_directional_press_target(
                self.current_press_distance_1,
                self.current_press_distance_2,
                direction,
            )
            if target is None:
                self.set_mutual_contact_state(False, "directional_hold_stage5_target_unavailable")
                return False, "directional target unavailable", self.snapshot()

            ok, message, info = self.sustain_directional_press_target(
                target,
                duration=self.directional_stage5_hold_duration,
                velocity_limit=self.hold_vel_limit,
                contact_mode="directional_hold_sustain",
                contact_loss_reason="mutual_contact_lost_during_stage5_hold",
                contact_loss_message="mutual contact lost during sustained directional hold",
                velocity_reason="velocity_spike_during_stage5_hold",
                velocity_message="velocity spike during sustained directional hold",
            )
            if not ok:
                return False, message, info

            self.set_mutual_contact_state(False, "directional_hold_complete")
            self.directional_press_enabled = False
            return True, "sustained bilateral directional hold completed", {"samples": [info]}

        stage_info: List[Dict[str, Any]] = []

        distance1 = self.current_press_distance_1
        distance2 = self.current_press_distance_2
        while max(distance1, distance2) < self.max_directional_press_distance - 1e-9:
            if not self.bilateral_contact_confirmed():
                self.set_mutual_contact_state(False, "mutual_contact_lost_before_directional_press")
                return False, "mutual contact lost during directional compression", {"samples": stage_info}

            direction = self.compute_mutual_press_direction()
            if direction is None:
                self.set_mutual_contact_state(False, "directional_press_direction_unavailable")
                return False, "mutual contact direction unavailable", {"samples": stage_info}

            self.current_mutual_press_dir = direction
            self.directional_press_enabled = True

            distance1 = min(distance1 + self.press_step, self.max_directional_press_distance)
            distance2 = min(distance2 + self.press_step, self.max_directional_press_distance)

            projected_force_1 = self.projected_force(1, direction)
            projected_force_2 = self.projected_force(2, direction)
            if self.finite(projected_force_1) and self.finite(projected_force_2):
                if projected_force_1 + self.force_balance_tolerance < projected_force_2:
                    distance1 = min(distance1 + self.side_press_step, self.max_directional_press_distance)
                    contact_mode = "push_robot1_more_along_direction"
                elif projected_force_2 + self.force_balance_tolerance < projected_force_1:
                    distance2 = min(distance2 + self.side_press_step, self.max_directional_press_distance)
                    contact_mode = "push_robot2_more_along_direction"
                else:
                    contact_mode = "directionally_balanced"
            else:
                contact_mode = "directional_seek"

            self.current_press_distance_1 = distance1
            self.current_press_distance_2 = distance2

            target = self.apply_directional_press_target(distance1, distance2, direction, contact_mode)
            if target is None:
                self.set_mutual_contact_state(False, "directional_press_target_unavailable")
                return False, "directional target unavailable", {"samples": stage_info}

            if not self.wait_for_directional_waypoint_completion():
                self.set_mutual_contact_state(False, "directional_press_waypoint_timeout")
                return False, "directional compression command did not settle", {"samples": stage_info}

            if not self.bilateral_contact_confirmed():
                self.set_mutual_contact_state(False, "mutual_contact_lost_after_directional_press_step")
                snap = self.snapshot()
                snap["command_distance"] = max(distance1, distance2)
                stage_info.append(snap)
                self.log_sync_step(max(distance1, distance2))
                return False, "mutual contact lost during directional compression", {"samples": stage_info}

            snap = self.snapshot()
            snap["command_distance"] = max(distance1, distance2)
            stage_info.append(snap)
            self.log_sync_step(max(distance1, distance2))

            # NOTE: separation_safe() (3-D Euclidean) is NOT checked here because the
            # two robot mounts have different Z heights (~18 mm structural offset), making
            # the 3-D distance permanently below the 20 mm floor even before any contact.
            # x_gap_safe() and pose_safe() below provide the relevant safety coverage.
            if not self.x_gap_safe():
                self.get_logger().error(
                    f"EE x gap {self.ee_x_gap():.4f} m above safety limit "
                    f"{self.max_ee_x_gap:.4f} m — aborting directional compression"
                )
                self.set_mutual_contact_state(False, "x_gap_above_safety_limit")
                return False, "inter-robot x gap above safety limit", {"samples": stage_info}
            if not self.pose_safe():
                self.set_mutual_contact_state(False, "pose_below_x_floor_safety_limit")
                return False, "EE pose below x floor safety limit", {"samples": stage_info}
            # Velocity check removed - low speed is enforced by small step size

            projected_force_1 = snap["projected_force_1"]
            projected_force_2 = snap["projected_force_2"]
            force_diff = (
                projected_force_1 - projected_force_2
                if self.finite(projected_force_1) and self.finite(projected_force_2)
                else float("nan")
            )
            if self.finite(force_diff) and abs(force_diff) > self.force_diff_abort:
                self.set_mutual_contact_state(False, "directional_force_asymmetry_too_high")
                return False, "directional force asymmetry too high", {"samples": stage_info}

            if self.finite(projected_force_1) and self.finite(projected_force_2) and max(projected_force_1, projected_force_2) > self.max_force_mag_abort:
                self.set_mutual_contact_state(False, "directional_force_too_high")
                return False, "directional contact force too high", {"samples": stage_info}

        self.set_mutual_contact_state(False, "directional_compression_complete")
        self.directional_press_enabled = False
        return True, "adaptive bilateral directional compression completed", {"samples": stage_info}

    def run(self) -> List[Dict[str, Any]]:
        stages = [
            self.stage1_liveness,
            self.stage2_idle,
            self.stage3_sync_move,
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
            self.directional_press_enabled = False
            self.set_mutual_contact_state(False, "run_complete")
            self.get_logger().info("ALL STAGES PASSED")
            self.write_results(results)
            return results
        finally:
            self.set_mutual_contact_state(False, "run_finally")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bilateral hardware harness v3")
    parser.add_argument(
        "--push-axis-height-z",
        type=float,
        default=None,
        help=(
            "Absolute push-axis height in metres. When provided, precontact and press commands "
            "hold the bilateral push at this z height instead of the controller trajectory z."
        ),
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        default=False,
        help="Use simulation time (for Gazebo). Required when running against Gazebo simulation.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    # Initialize ROS with or without sim time
    if args.use_sim_time:
        rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    else:
        rclpy.init()
    node = HardwareHarnessAdaptiveV3(push_axis_height_z=args.push_axis_height_z, use_sim_time=args.use_sim_time)
    try:
        out = node.run()
        print(json.dumps(out, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()