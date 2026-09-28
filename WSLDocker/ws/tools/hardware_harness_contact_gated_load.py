#!/usr/bin/env python3
"""Contact-gated load step hardware harness for K_lat ablation experiments.
# Gazebo: 5 trials per combo (2×2 = 4 combos, 20 total trials)
python3 tools/contact_gated_ablation_live_gazebo.py \
  --k-lateral-cases-npm 10,49.4 --peak-load-cases-n 0.015,0.025 \
  --repeats-n 5 --gui --display :0 -- --spring-specimen s1 ...

# Hardware: 20 trials per combo
python3 tools/hardware_harness_contact_gated_load.py \
  --ablate --k-lateral-cases-npm 10,49.4 --peak-load-cases-n 0.015,0.025 \
  --repeats-n 20 --spring-specimen s1 ...


"""


from __future__ import annotations

import argparse
import json
import math
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import rclpy
from std_msgs.msg import Float64

from hardware_harness_v3 import HardwareHarnessAdaptiveV3

try:
    from tools.ablation_config import CARTESIAN_STIFFNESS_LIMIT_NPM, SPRING_SPECIMENS
except ImportError:
    from ablation_config import CARTESIAN_STIFFNESS_LIMIT_NPM, SPRING_SPECIMENS


# Load targets should be well above preload contact force (typically 1-2 N after Stage 4)
DEFAULT_LOAD_TARGETS_N = (2.5, 3.0, 3.5, 4.0)
DEFAULT_CASE_LOAD_TARGET_FRACTIONS = (0.50, 0.75, 1.00)
DEFAULT_ABLATION_PEAK_LOAD_FRACTIONS = (0.50, 0.75, 1.00)
DEFAULT_ABLATION_K_LATERAL_FRACTIONS = (0.25, 0.50, 0.75, 1.00, 1.25)


class ContactGatedLoadStepHarness(HardwareHarnessAdaptiveV3):
    """UTM-style bilateral harness with contact-gated stage 3, handoff stage 4, quasi-static load-step stage 5, and optional stage-0 calibration-derived ablation ranges."""

    def __init__(
        self,
        push_axis_height_z: Optional[float] = None,
        load_targets_n: Optional[Sequence[float]] = None,
        hold_duration_s: float = 10.0,
        upper_force_limit_n: Optional[float] = None,
        load_target_tolerance_n: Optional[float] = None,
        load_seek_step_m: float = 0.0005,
        load_balance_step_m: float = 0.00025,
        # Increased from 0.012 m: s1 spring (k=49.4 N/m) needs ~40 mm
        # compression to reach a 2 N load target; 12 mm only produced ~0.6 N.
        # Default raised to 0.330 m (330 mm) so the harness does not
        # restrict load-step travel below ~33 cm in typical hardware runs.
        max_additional_load_travel_m: float = 0.330,
        min_stage_advance_m: float = 0.0,
        stage4_handoff_target_n: Optional[float] = None,
        defer_k_lateral_until_after_precontact: bool = False,
        external_camera_trigger_file: Optional[str] = None,
        ignore_local_frame_ee_safety: Optional[bool] = None,
        precontact_waypoint_timeout_s: float = 8.0,
        precontact_contact_threshold_n: Optional[float] = None,
        precontact_velocity_limit: Optional[float] = None,
        load_step_waypoint_timeout_s: Optional[float] = None,
        calibration_json_path: Optional[str] = None,
        enable_ablation: bool = False,
        n_repeats: int = 1,
        k_lateral_npm: Optional[float] = None,
        k_lateral_case_values_npm: Optional[Sequence[float]] = None,
        peak_load_case_values_n: Optional[Sequence[float]] = None,
        spring_specimen: Optional[str] = None,
        spring_stiffness_npm: Optional[float] = None,
        use_sim_time: bool = False,
        plot_save_dir: Optional[str] = None,
        k_axial_npm: Optional[float] = None,
    ) -> None:
        super().__init__(push_axis_height_z=push_axis_height_z, use_sim_time=use_sim_time, plot_save_dir=plot_save_dir)
        self.configured_load_targets_n = tuple(float(value) for value in (load_targets_n or ()))
        self.load_targets_explicit = bool(self.configured_load_targets_n)
        self.load_targets_n = self.configured_load_targets_n or tuple(DEFAULT_LOAD_TARGETS_N)
        self.load_hold_duration_s = hold_duration_s
        default_upper_force_limit_n = max(self.load_targets_n) if self.load_targets_n else float("nan")
        resolved_upper_force_limit_n = (
            default_upper_force_limit_n if upper_force_limit_n is None else float(upper_force_limit_n)
        )
        if not math.isfinite(resolved_upper_force_limit_n) or resolved_upper_force_limit_n <= 0.0:
            raise ValueError("upper_force_limit_n must be finite and > 0")
        self.upper_force_limit_n = resolved_upper_force_limit_n
        self.upper_force_limit_explicit = upper_force_limit_n is not None
        self.ignore_local_frame_ee_safety = (
            self._env_flag("OMX_CONTACT_GATED_IGNORE_LOCAL_FRAME_EE_SAFETY", False)
            if ignore_local_frame_ee_safety is None
            else bool(ignore_local_frame_ee_safety)
        )
        self.load_target_tolerance_explicit = load_target_tolerance_n is not None
        self.load_target_tolerance_n = max(
            0.0,
            self._default_load_target_tolerance_n()
            if load_target_tolerance_n is None
            else load_target_tolerance_n,
        )
        self.load_seek_step_m = max(1e-5, load_seek_step_m)
        self.load_balance_step_m = max(0.0, load_balance_step_m)
        self.max_additional_load_travel_m = max(0.0, max_additional_load_travel_m)
        # Sync press_end_distance (used by v3 stage2 x-floor formula) so the
        # safety floor accounts for the full load-step travel budget.
        self.press_end_distance = self.max_additional_load_travel_m
        self.max_directional_press_distance = self.press_end_distance + self.max_side_extra_press
        self.min_stage_advance_m = max(0.0, float(min_stage_advance_m))
        self.stage4_handoff_target_n = (
            float(stage4_handoff_target_n)
            if stage4_handoff_target_n is not None and math.isfinite(float(stage4_handoff_target_n)) and float(stage4_handoff_target_n) > 0.0
            else float("nan")
        )
        self.defer_k_lateral_until_after_precontact = bool(defer_k_lateral_until_after_precontact)
        self._pending_case_k_lateral_npm: Optional[float] = None
        self.external_camera_trigger_file = (
            str(external_camera_trigger_file).strip() or None
        ) if external_camera_trigger_file is not None else None
        if self.external_camera_trigger_file:
            # Configure the base-class trigger (so run_dir/log path logic is consistent),
            # then replace it with an atomic-writer wrapper so host-visible
            # trigger writes use an atomic temp-file + rename pattern.
            try:
                self.configure_external_camera_trigger(self.external_camera_trigger_file)
                previous = getattr(self, "external_camera_trigger", None)
            except Exception:
                previous = None

            class AtomicExternalCameraTrigger:
                """Wrapper that provides the same simple interface as the
                harness ExternalCameraTrigger but performs atomic writes
                (temp-file + os.replace) to reduce cross-OS partial-write
                races when the recorder polls the trigger file.

                This wrapper intentionally mirrors the small public API used
                by the harness: `is_enabled()`, `start(name, reason)`,
                `stop(reason)`, and exposes `window_index`, `active`,
                `trigger_path`, and `log_path` attributes for compatibility.
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
                        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                        with open(self.log_path, "a", encoding="utf-8") as handle:
                            handle.write(json.dumps(payload) + "\n")
                    except OSError:
                        pass

                def _atomic_write(self, body: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                    if not self.trigger_path:
                        return None
                    target = Path(self.trigger_path)
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
                        try:
                            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                                handle.write(body)
                                handle.flush()
                                try:
                                    os.fsync(handle.fileno())
                                except Exception:
                                    pass
                        except Exception as err:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                            raise

                        # Attempt to fsync the directory to improve durability where supported.
                        try:
                            dirfd = os.open(str(target.parent), os.O_RDONLY)
                            try:
                                os.fsync(dirfd)
                            finally:
                                os.close(dirfd)
                        except Exception:
                            pass

                        # Atomic replace
                        os.replace(tmp_path, str(target))
                    except OSError as error:
                        payload["error"] = str(error)
                        self._append_log(payload)
                        return payload
                    self._append_log(payload)
                    return payload

                def start(self, name: str, reason: str) -> Optional[Dict[str, Any]]:
                    """Write a START event to the trigger file and log.

                    Increments `window_index`, marks the trigger `active`, and
                    writes a single-line START payload that the host recorder
                    consumes. Returns the payload dict that was appended to the
                    external trigger log (useful for unit tests and debugging),
                    or `None` if the trigger is not configured.
                    """
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
                    return self._atomic_write(body, payload)

                def stop(self, reason: str) -> Optional[Dict[str, Any]]:
                    """Write a STOP event to end the current capture window.

                    If the trigger is not configured or not currently active, this
                    returns `None`. Otherwise it clears `active` and atomically
                    writes the STOP payload; the return value mirrors
                    `start()` and contains the logged payload dict.
                    """
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
                    return self._atomic_write(body, payload)

            # Replace base-class trigger object with atomic wrapper preserving
            # the previously-configured log path and window index where present.
            wrapped = AtomicExternalCameraTrigger(self.external_camera_trigger_file, (previous.log_path if previous is not None else os.path.join(self.run_dir, "external_camera_trigger.log")))
            if previous is not None:
                try:
                    wrapped.window_index = previous.window_index
                    wrapped.active = previous.active
                except Exception:
                    pass
            self.external_camera_trigger = wrapped
        self.force_balance_tolerance = self._default_force_balance_tolerance_n()
        self.precontact_waypoint_timeout_s = max(0.5, precontact_waypoint_timeout_s)
        # Two-phase precontact with velocity control:
        # Phase 1 (fast): ~30mm/s approach until within 2cm of expected contact
        # Phase 2 (slow): 5-10mm/s quasistatic compression for final approach
        # Both robots share a single phase to ensure symmetric, synchronous motion
        # and avoid unilateral tapping.
        # Step sizes tuned for ~1.5Hz harness loop rate:
        #   Fast: 20mm step → ~30mm/s effective velocity
        #   Slow: 5mm step → ~7.5mm/s quasistatic velocity
        self.precontact_FAST_STEP_M = 0.020  # 20mm fast approach (~30mm/s)
        self.precontact_SLOW_STEP_M = 0.005   # 5mm quasistatic approach (~7.5mm/s)
        self.precontact_QUASI_STATIC_THRESHOLD_M = 0.020  # Switch to slow at 20mm from end
        self.precontact_contact_threshold_explicit = precontact_contact_threshold_n is not None
        default_precontact_contact_threshold_n = (
            self._default_local_frame_precontact_threshold_n()
            if self.ignore_local_frame_ee_safety
            else self.contact_force_enter
        )
        self.precontact_contact_threshold_n = max(
            0.0,
            default_precontact_contact_threshold_n
            if precontact_contact_threshold_n is None
            else precontact_contact_threshold_n,
        )
        self.precontact_velocity_limit = max(
            self.move_vel_limit,
            2.0 if precontact_velocity_limit is None else precontact_velocity_limit,
        )
        self.calibration_json_path = calibration_json_path
        self.enable_ablation = bool(enable_ablation)
        self.n_repeats = max(1, int(n_repeats))
        self.k_lateral_npm = float(k_lateral_npm) if k_lateral_npm is not None else float("nan")
        self.k_lateral_explicit = k_lateral_npm is not None
        self.k_axial_npm = float(k_axial_npm) if k_axial_npm is not None else float("nan")
        self.k_axial_explicit = k_axial_npm is not None
        self.k_lateral_case_values_npm = self._unique_sorted_positive(k_lateral_case_values_npm or ())
        self.peak_load_case_values_n = self._unique_sorted_positive(peak_load_case_values_n or ())
        spring_preset = self._spring_specimen_preset(spring_specimen)
        self.spring_specimen = self._normalize_spring_specimen(spring_specimen)
        self.spring_specimen_label = str(spring_preset.get("label", "")) if spring_preset else ""
        self.spring_free_length_m = float(spring_preset.get("free_length_m", float("nan")))
        self.spring_width_m = float(spring_preset.get("width_m", float("nan")))
        self.spring_thickness_m = float(spring_preset.get("thickness_m", float("nan")))
        if spring_stiffness_npm is not None and not self._positive_finite(spring_stiffness_npm):
            raise ValueError("spring_stiffness_npm must be finite and > 0")
        preset_stiffness_npm = float(spring_preset.get("stiffness_npm", float("nan")))
        self.spring_stiffness_npm = (
            float(spring_stiffness_npm)
            if spring_stiffness_npm is not None
            else preset_stiffness_npm
        )
        self.case_load_target_fractions = DEFAULT_CASE_LOAD_TARGET_FRACTIONS
        self.ablation_peak_load_fractions = DEFAULT_ABLATION_PEAK_LOAD_FRACTIONS
        self.ablation_k_lateral_fractions = DEFAULT_ABLATION_K_LATERAL_FRACTIONS

        self.current_load_step_index = 0
        self.current_load_target_n = float("nan")
        self.current_load_step_state = "idle"
        self.current_shared_load_n = float("nan")
        self.current_mean_load_n = float("nan")
        self.current_load_travel_limit_m = float("nan")
        self.current_case_name = "default"
        self.current_case_k_lateral_npm = self.k_lateral_npm
        self.current_case_peak_load_n = self.upper_force_limit_n
        self.current_calibration_source = "none"
        self.current_calibration_json_path = ""
        self.calibration_payload: Dict[str, Any] = {}
        self.calibrated_peak_load_n = float("nan")
        self.calibrated_critical_k_lateral_npm = float("nan")
        self.calibrated_measured_k_lateral_npm = float("nan")
        self.calibrated_push_axis_height_z = float("nan")
        self.push_axis_height_calibration_gain = 0.5
        self.push_axis_height_calibration_deadband_m = 0.003
        self.push_axis_height_calibration_max_adjust_m = 0.020
        self.derived_k_lateral_case_values_npm: Tuple[float, ...] = tuple()
        self.derived_peak_load_case_values_n: Tuple[float, ...] = tuple()

        self.command_repeats = 1
        self.directional_refresh_interval = min(self.directional_refresh_interval, 0.25)
        self.directional_stage5_hold_duration = self.load_hold_duration_s
        if self.ignore_local_frame_ee_safety:
            self.directional_waypoint_timeout = max(
                self.directional_waypoint_timeout,
                self.precontact_waypoint_timeout_s,
            )
        self.load_step_waypoint_timeout = (
            max(self.directional_waypoint_timeout, self.load_hold_duration_s)
            if load_step_waypoint_timeout_s is None
            else max(0.5, float(load_step_waypoint_timeout_s))
        )

        self.klat_pub1 = self.create_publisher(
            Float64, "/robot1/robot1_variable_stiffness/set_k_lateral", 10
        )
        self.klat_pub2 = self.create_publisher(
            Float64, "/robot2/robot2_variable_stiffness/set_k_lateral", 10
        )
        self.kax_pub1 = self.create_publisher(
            Float64, "/robot1/robot1_variable_stiffness/set_k_axial", 10
        )
        self.kax_pub2 = self.create_publisher(
            Float64, "/robot2/robot2_variable_stiffness/set_k_axial", 10
        )

        if self.ignore_local_frame_ee_safety:
            self.get_logger().info(
                "Ignoring local-frame EE separation/x-gap safety checks for this run"
            )

        extra_columns = [
            "load_step_index",
            "load_step_target_n",
            "load_step_state",
            "shared_load_n",
            "mean_load_n",
            "load_travel_limit_m",
            "local_load_source_1",
            "local_load_source_2",
            "force_delta_x_1",
            "force_delta_y_1",
            "force_delta_z_1",
            "force_delta_x_2",
            "force_delta_y_2",
            "force_delta_z_2",
            "baseline_force_x_1",
            "baseline_force_y_1",
            "baseline_force_z_1",
            "baseline_force_x_2",
            "baseline_force_y_2",
            "baseline_force_z_2",
            "case_name",
            "case_k_lateral_npm",
            "case_peak_load_n",
            "calibration_source",
            "calibrated_peak_load_n",
            "calibrated_critical_k_lateral_npm",
            "calibrated_measured_k_lateral_npm",
        ]
        for column in extra_columns:
            if column not in self.log_columns:
                self.log_columns.append(column)
            if column not in self.sync_columns:
                self.sync_columns.append(column)

        self.write_csv_header(self.csv_path, self.log_columns)
        self.write_csv_header(self.sync_csv_path, self.sync_columns)

    def compute_press_target(self, x_offset1: float, x_offset2: float) -> Optional[Dict[str, float]]:
        """Override V3 to use absolute targets based on actual end-effector positions.

        The controller interprets frame_id='offset' as a delta from a fixed
        reference (the controller's initial desired pose), not from the current
        position. This causes both robots to converge to the same position when
        given the same offset magnitude. By using frame_id='absolute' with
        targets computed from actual end-effector positions, each robot moves
        from its own real position by the specified offset.
        """
        if self.ee1 is None or self.ee2 is None:
            return None
        
        # Compute absolute targets from actual end-effector positions + offsets
        # Clamp Z: never above push_axis_height_z, but never below current EE Z
        pah = self.push_axis_height_z if self.push_axis_height_z is not None else min(self.ee1.z, self.ee2.z)
        target_z1 = min(self.ee1.z, pah)
        target_z2 = min(self.ee2.z, pah)

        raw_target_x1 = self.ee1.x + x_offset1
        target_x1 = max(raw_target_x1, self.MIN_DISTANCE_FROM_BASE)
        target_y1 = self.ee1.y
        
        raw_target_x2 = self.ee2.x + x_offset2
        target_x2 = max(raw_target_x2, self.MIN_DISTANCE_FROM_BASE)
        target_y2 = self.ee2.y

        # Preserve absolute-command semantics while guaranteeing the hard x-floor.
        # Offsets must reflect the clamped target so telemetry and later logic stay consistent.
        clamped_offset_x1 = target_x1 - self.ee1.x
        clamped_offset_x2 = target_x2 - self.ee2.x
        offset_z1 = target_z1 - self.ee1.z
        offset_z2 = target_z2 - self.ee2.z
        
        return {
            "offset_x1": clamped_offset_x1,
            "offset_y1": 0.0,
            "offset_z1": offset_z1,
            "offset_x2": clamped_offset_x2,
            "offset_y2": 0.0,
            "offset_z2": offset_z2,
            "target_x1": target_x1,
            "target_y1": target_y1,
            "target_z1": target_z1,
            "target_x2": target_x2,
            "target_y2": target_y2,
            "target_z2": target_z2,
        }

    def _local_axis_load_source(self, robot_id: int) -> str:
        force = self.linear_force(robot_id)
        if force is not None and self.fin(force[0]):
            return "raw_force_x"

        contact_fx_mag = self.contact_fx_mag_1 if robot_id == 1 else self.contact_fx_mag_2
        if self.fin(contact_fx_mag):
            return "contact_fx_mag"
        return "unavailable"

    def snapshot(self) -> Dict[str, Any]:
        row = super().snapshot()
        delta_1 = self.contact_force_delta(1) or (float("nan"), float("nan"), float("nan"))
        delta_2 = self.contact_force_delta(2) or (float("nan"), float("nan"), float("nan"))
        baseline_1 = self.baseline_force_vec_1 or (float("nan"), float("nan"), float("nan"))
        baseline_2 = self.baseline_force_vec_2 or (float("nan"), float("nan"), float("nan"))
        row.update(
            {
                "load_step_index": self.current_load_step_index,
                "load_step_target_n": self.current_load_target_n,
                "load_step_state": self.current_load_step_state,
                "shared_load_n": self.current_shared_load_n,
                "mean_load_n": self.current_mean_load_n,
                "load_travel_limit_m": self.current_load_travel_limit_m,
                "local_load_source_1": self._local_axis_load_source(1),
                "local_load_source_2": self._local_axis_load_source(2),
                "force_delta_x_1": delta_1[0],
                "force_delta_y_1": delta_1[1],
                "force_delta_z_1": delta_1[2],
                "force_delta_x_2": delta_2[0],
                "force_delta_y_2": delta_2[1],
                "force_delta_z_2": delta_2[2],
                "baseline_force_x_1": baseline_1[0],
                "baseline_force_y_1": baseline_1[1],
                "baseline_force_z_1": baseline_1[2],
                "baseline_force_x_2": baseline_2[0],
                "baseline_force_y_2": baseline_2[1],
                "baseline_force_z_2": baseline_2[2],
                "case_name": self.current_case_name,
                "case_k_lateral_npm": self.current_case_k_lateral_npm,
                "case_peak_load_n": self.current_case_peak_load_n,
                "calibration_source": self.current_calibration_source,
                "calibrated_peak_load_n": self.calibrated_peak_load_n,
                "calibrated_critical_k_lateral_npm": self.calibrated_critical_k_lateral_npm,
                "calibrated_measured_k_lateral_npm": self.calibrated_measured_k_lateral_npm,
            }
        )
        return row

    def fin(self, value: float) -> bool:
        return self.finite(value)

    @staticmethod
    def _positive_finite(value: Any) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0

    @classmethod
    def _unique_sorted_positive(cls, values: Sequence[float]) -> Tuple[float, ...]:
        ordered: List[float] = []
        for value in sorted(float(item) for item in values if cls._positive_finite(item)):
            rounded = round(value, 6)
            if not ordered or abs(rounded - ordered[-1]) > 1e-9:
                ordered.append(rounded)
        return tuple(ordered)

    @staticmethod
    def _resolve_calibration_json_path(path: str) -> str:
        if os.path.isfile(path):
            return path
        if not os.path.isdir(path):
            raise FileNotFoundError(path)

        candidates: List[str] = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                if name.startswith("calib_result_") and name.endswith(".json"):
                    candidates.append(os.path.join(root, name))
        if not candidates:
            raise FileNotFoundError(f"No calib_result_*.json found under {path}")
        candidates.sort()
        return candidates[-1]

    @staticmethod
    def _normalize_spring_specimen(value: Optional[str]) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().lower()

    @classmethod
    def _spring_specimen_preset(cls, specimen: Optional[str]) -> Dict[str, Any]:
        key = cls._normalize_spring_specimen(specimen)
        if not key:
            return {}
        preset = SPRING_SPECIMENS.get(key)
        if preset is None:
            allowed = ", ".join(sorted(SPRING_SPECIMENS))
            raise ValueError(f"unknown spring_specimen {specimen!r}; expected one of: {allowed}")
        return dict(preset)

    def _explicit_spring_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.spring_specimen:
            payload["spring_specimen"] = self.spring_specimen
        if self.spring_specimen_label:
            payload["spring_specimen_label"] = self.spring_specimen_label
        if self._positive_finite(self.spring_free_length_m):
            payload["spring_free_length_m"] = self.spring_free_length_m
        if self._positive_finite(self.spring_width_m):
            payload["spring_width_m"] = self.spring_width_m
        if self._positive_finite(self.spring_thickness_m):
            payload["spring_thickness_m"] = self.spring_thickness_m
        if self._positive_finite(self.spring_stiffness_npm):
            payload["spring_stiffness_npm"] = self.spring_stiffness_npm
            payload["K_lat_meas_Npm"] = self.spring_stiffness_npm
        return payload

    def _has_ablation_k_source(self) -> bool:
        return bool(
            self.calibration_json_path
            or self.k_lateral_case_values_npm
            or self.k_lateral_explicit
            or self._positive_finite(self.spring_stiffness_npm)
        )

    def _has_ablation_peak_source(self) -> bool:
        return bool(
            self.calibration_json_path
            or self.peak_load_case_values_n
            or self.upper_force_limit_explicit
            or self.load_targets_explicit
        )

    def _load_calibration_payload(self) -> Tuple[Dict[str, Any], str]:
        if not self.calibration_json_path:
            return {}, ""
        resolved_path = self._resolve_calibration_json_path(self.calibration_json_path)
        with open(resolved_path, "r", encoding="utf-8") as handle:
            return json.load(handle), resolved_path

    def _derive_peak_load_cases(self, peak_reference_n: float) -> Tuple[float, ...]:
        if self.peak_load_case_values_n:
            return self.peak_load_case_values_n
        if not self._positive_finite(peak_reference_n):
            return tuple()
        if not self.enable_ablation:
            return (round(float(peak_reference_n), 6),)
        return self._unique_sorted_positive(
            peak_reference_n * fraction for fraction in self.ablation_peak_load_fractions
        )

    def _derive_case_load_targets(self, peak_load_n: float) -> Tuple[float, ...]:
        if not self.enable_ablation and self.load_targets_explicit:
            return self.load_targets_n
        return self._unique_sorted_positive(
            peak_load_n * fraction for fraction in self.case_load_target_fractions
        )

    def _derive_k_lateral_cases(self, calibration_payload: Dict[str, Any]) -> Tuple[float, ...]:
        if self.k_lateral_case_values_npm:
            return self._unique_sorted_positive(
                min(value, CARTESIAN_STIFFNESS_LIMIT_NPM) for value in self.k_lateral_case_values_npm
            )

        measured = float(calibration_payload.get("K_lat_meas_Npm", float("nan")))
        critical = float(calibration_payload.get("K_lat_star_Npm", float("nan")))
        ceiling = CARTESIAN_STIFFNESS_LIMIT_NPM
        if self._positive_finite(measured):
            ceiling = min(ceiling, measured)

        if self._positive_finite(critical):
            case_values = [critical * fraction for fraction in self.ablation_k_lateral_fractions]
            if self._positive_finite(measured):
                case_values.append(measured)
            return self._unique_sorted_positive(min(value, ceiling) for value in case_values)

        if self.k_lateral_explicit and self._positive_finite(self.k_lateral_npm):
            return (round(min(self.k_lateral_npm, ceiling), 6),)

        if self.enable_ablation and self._positive_finite(measured) and self._positive_finite(self.spring_stiffness_npm):
            return self._unique_sorted_positive(
                min(measured * fraction, ceiling) for fraction in self.ablation_k_lateral_fractions
            )

        if not self.enable_ablation:
            return tuple()
        return (round(ceiling, 6),)

    def _reference_load_n(self) -> float:
        reference_load_n = float("nan")
        finite_targets = [float(value) for value in self.load_targets_n if self._positive_finite(value)]
        if finite_targets:
            reference_load_n = min(finite_targets)
        if self._positive_finite(self.upper_force_limit_n):
            if self._positive_finite(reference_load_n):
                reference_load_n = min(reference_load_n, self.upper_force_limit_n)
            else:
                reference_load_n = self.upper_force_limit_n
        return reference_load_n

    def _default_load_target_tolerance_n(self) -> float:
        reference_load_n = self._reference_load_n()
        if not self._positive_finite(reference_load_n):
            return 0.02
        tolerance_floor_n = 0.001 if self._using_local_frame_load_logic() else 0.0005
        return max(tolerance_floor_n, min(0.02, 0.05 * reference_load_n))

    def _default_force_balance_tolerance_n(self) -> float:
        reference_load_n = self._reference_load_n()
        if not self._positive_finite(reference_load_n):
            return 0.05
        return max(0.0025, min(0.02, 0.10 * reference_load_n))

    def _load_targets_ready(self) -> bool:
        if not self.load_targets_n:
            return False
        return all(self._positive_finite(value) for value in self.load_targets_n)

    def _default_local_frame_precontact_threshold_n(self) -> float:
        reference_load_n = self._reference_load_n()
        if not self._positive_finite(reference_load_n):
            return 0.02
        return max(0.005, min(0.02, 0.25 * reference_load_n))

    def _stage3_local_contact_threshold_n(self) -> float:
        reference_load_n = self._reference_load_n()
        if not self._positive_finite(reference_load_n):
            return self.precontact_contact_threshold_n
        return max(0.002, min(self.precontact_contact_threshold_n, 0.25 * reference_load_n))

    def _stage3_contact_detected(self, robot_id: int) -> bool:
        if not self._using_local_frame_load_logic():
            return self.contact_detected(robot_id)
        contact_valid = self.contact_valid_1 if robot_id == 1 else self.contact_valid_2
        if not contact_valid:
            return False
        local_load = self._local_raw_axis_load(robot_id)
        if not (self.fin(local_load) and local_load >= self._stage3_local_contact_threshold_n()):
            return False
        # Collision filter: reject contact if EE is over the platform
        # BUT during Stage 3, the robots ARE supposed to make contact at the platform,
        # so only apply this check if we haven't started the precontact approach yet
        # (i.e., if we're still in early stages like homing)
        # For now, disable this check during Stage 3 to allow legitimate contact
        # if self._ee_over_platform(robot_id):
        #     self.get_logger().warn(
        #         f"Robot{robot_id} force detected but EE is over platform (collision). Rejecting as false contact."
        #     )
        #     return False
        return True

    def _ee_over_platform(self, robot_id: int) -> bool:
        """Check if end-effector is over the metallic platform (likely a collision)."""
        pos = self.current_cartesian_position(robot_id)
        if pos is None:
            return False
        ee_world_x = self._local_to_world_x(pos[0], robot_id)
        # Platform spans from -0.15 to +0.15 (300mm centered at world origin)
        platform_left_x = -0.15
        platform_right_x = 0.15
        # Add small margin for IK tolerance
        margin = 0.01
        return (platform_left_x - margin) <= ee_world_x <= (platform_right_x + margin)

    def _local_to_world_x(self, local_x: float, robot_id: int) -> float:
        """Convert local robot x to world x coordinate."""
        # Robot1 base at -0.39, Robot2 base at +0.39
        if robot_id == 1:
            return -0.39 + local_x
        else:  # robot_id == 2
            return 0.39 - local_x

    def _stage3_bilateral_contact_confirmed(self) -> bool:
        return self._stage3_contact_detected(1) and self._stage3_contact_detected(2)

    def _stage3_handoff_direction(self) -> Optional[Tuple[float, float, float]]:
        direction = self.compute_mutual_press_direction()
        if direction is not None:
            return direction
        if not self._using_local_frame_load_logic():
            return None
        if self.current_mutual_press_dir is not None:
            return self.current_mutual_press_dir
        return (1.0, 0.0, 0.0)

    def _load_step_contact_threshold_n(self, target_load_n: float) -> float:
        if not self._using_local_frame_load_logic():
            return self.precontact_contact_threshold_n
        if not self._positive_finite(target_load_n):
            return self.precontact_contact_threshold_n
        threshold_scale = 0.25
        if self.current_contact_mode.startswith("load_step_") and (
            "_seek:" in self.current_contact_mode
            or "_unload:" in self.current_contact_mode
            or "_hold_recover:" in self.current_contact_mode
        ):
            threshold_scale = 0.15
        return max(0.001, min(self.precontact_contact_threshold_n, threshold_scale * target_load_n))

    def _load_step_contact_confirmed(self, target_load_n: float) -> bool:
        if self.bilateral_contact_confirmed():
            return True
        if not self._using_local_frame_load_logic():
            return False

        direction = self.current_mutual_press_dir or (1.0, 0.0, 0.0)
        load_1, load_2, shared_load, _mean_load = self._update_load_metrics(direction)
        threshold_n = self._load_step_contact_threshold_n(target_load_n)
        return (
            self.fin(load_1)
            and self.fin(load_2)
            and self.fin(shared_load)
            and load_1 >= threshold_n
            and load_2 >= threshold_n
            and shared_load >= threshold_n
        )

    def _resolve_load_step_direction(self, target_load_n: float) -> Optional[Tuple[float, float, float]]:
        direction = self.current_mutual_press_dir
        if direction is not None:
            return direction

        direction = self.compute_mutual_press_direction()
        if direction is not None:
            return direction

        if self._using_local_frame_load_logic() and self._load_step_contact_confirmed(target_load_n):
            return (1.0, 0.0, 0.0)
        return None

    def _append_stage_sample(
        self,
        stage_info: List[Dict[str, Any]],
        *,
        command_distance: float,
    ) -> Dict[str, Any]:
        snap = self.snapshot()
        snap["command_distance"] = command_distance
        stage_info.append(snap)
        self.log_sync_step(command_distance)
        return snap

    def configured_push_axis_height(self, fallback_z: float) -> float:
        if self._positive_finite(self.calibrated_push_axis_height_z):
            return float(self.calibrated_push_axis_height_z)
        return super().configured_push_axis_height(fallback_z)

    def _update_push_axis_height_calibration(self) -> None:
        samples: List[float] = []
        if self.ee1 is not None and self.fin(self.ee1.z):
            samples.append(float(self.ee1.z))
        if self.ee2 is not None and self.fin(self.ee2.z):
            samples.append(float(self.ee2.z))
        if len(samples) < 2:
            return

        observed_z = 0.5 * (samples[0] + samples[1])
        base_z = super().configured_push_axis_height(observed_z)
        if not self.fin(base_z):
            return

        if self.push_axis_height_z is None:
            calibrated = observed_z
        else:
            delta = observed_z - base_z
            if abs(delta) <= self.push_axis_height_calibration_deadband_m:
                calibrated = base_z
            else:
                bounded_delta = max(
                    -self.push_axis_height_calibration_max_adjust_m,
                    min(self.push_axis_height_calibration_max_adjust_m, delta),
                )
                calibrated = base_z + self.push_axis_height_calibration_gain * bounded_delta

        if self.fin(calibrated):
            self.calibrated_push_axis_height_z = float(calibrated)

    def _smooth_withdraw_to_zero(self, reason: str) -> None:
        direction = self.current_mutual_press_dir
        retreat_step_m = max(0.0005, self.load_seek_step_m)

        if direction is not None and (
            self.current_press_distance_1 > 1e-9 or self.current_press_distance_2 > 1e-9
        ):
            while self.current_press_distance_1 > 1e-9 or self.current_press_distance_2 > 1e-9:
                next_distance_1 = max(0.0, self.current_press_distance_1 - retreat_step_m)
                next_distance_2 = max(0.0, self.current_press_distance_2 - retreat_step_m)
                if (
                    next_distance_1 >= self.current_press_distance_1 - 1e-12
                    and next_distance_2 >= self.current_press_distance_2 - 1e-12
                ):
                    break

                self.current_press_distance_1 = next_distance_1
                self.current_press_distance_2 = next_distance_2
                target = self.apply_directional_press_target(
                    next_distance_1,
                    next_distance_2,
                    direction,
                    f"withdraw:{reason}",
                )
                if target is None:
                    break
                if not self._wait_for_directional_target_settle(min(1.0, self.load_step_waypoint_timeout)):
                    break

        x1 = self.current_offset_x1 if self.fin(self.current_offset_x1) else 0.0
        y1 = self.current_offset_y1 if self.fin(self.current_offset_y1) else 0.0
        z1 = self.current_offset_z1 if self.fin(self.current_offset_z1) else 0.0
        x2 = self.current_offset_x2 if self.fin(self.current_offset_x2) else 0.0
        y2 = self.current_offset_y2 if self.fin(self.current_offset_y2) else 0.0
        z2 = self.current_offset_z2 if self.fin(self.current_offset_z2) else 0.0
        max_mag = max(abs(x1), abs(y1), abs(z1), abs(x2), abs(y2), abs(z2))
        if max_mag > 1e-9:
            # Use absolute targets for withdrawal to maintain Z-height and ensure X >= MIN_DISTANCE_FROM_BASE
            anchor1 = self.mutual_contact_anchor_1 if self.mutual_contact_anchor_1 is not None else (0.0, 0.0, 0.0)
            anchor2 = self.mutual_contact_anchor_2 if self.mutual_contact_anchor_2 is not None else (0.0, 0.0, 0.0)
            # Use the configured push axis height (safe, consistent Z for both arms)
            observed_z = 0.5 * (anchor1[2] + anchor2[2]) if self.fin(anchor1[2]) and self.fin(anchor2[2]) else 0.0
            safe_z = self.configured_push_axis_height(observed_z)
            
            ramp_steps = max(5, min(25, int(math.ceil(max_mag / 0.002))))
            for idx in range(1, ramp_steps + 1):
                scale = max(0.0, 1.0 - (idx / float(ramp_steps)))
                # Compute absolute targets from scaled offsets, clamped to safety limits
                abs_x1 = max(anchor1[0] + x1 * scale, self.MIN_DISTANCE_FROM_BASE)
                abs_y1 = anchor1[1] + y1 * scale
                abs_z1 = safe_z
                abs_x2 = max(anchor2[0] + x2 * scale, self.MIN_DISTANCE_FROM_BASE)
                abs_y2 = anchor2[1] + y2 * scale
                abs_z2 = safe_z
                
                self.publish_absolute_targets(
                    (abs_x1, abs_y1, abs_z1),
                    (abs_x2, abs_y2, abs_z2),
                    repeats=1,
                    dt=0.02,
                )
                self.spin_for(0.02)

    def _publish_zero_offsets(self) -> None:
        self._smooth_withdraw_to_zero("case_reset")
        self.publish_offsets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=3, dt=0.05)
        self.spin_for(0.5)

    def _reset_case_state(self) -> None:
        self.current_phase = "init"
        self.current_contact_mode = "none"
        self.current_load_step_index = 0
        self.current_load_target_n = float("nan")
        self.current_load_step_state = "idle"
        self.current_shared_load_n = float("nan")
        self.current_mean_load_n = float("nan")
        self.current_load_travel_limit_m = float("nan")
        self.current_offset_x1 = float("nan")
        self.current_offset_y1 = float("nan")
        self.current_offset_z1 = float("nan")
        self.current_offset_x2 = float("nan")
        self.current_offset_y2 = float("nan")
        self.current_offset_z2 = float("nan")
        self.current_press_offset_x1 = 0.0
        self.current_press_offset_y1 = 0.0
        self.current_press_offset_z1 = 0.0
        self.current_press_offset_x2 = 0.0
        self.current_press_offset_y2 = 0.0
        self.current_press_offset_z2 = 0.0
        self.current_press_distance_1 = 0.0
        self.current_press_distance_2 = 0.0
        self.directional_press_enabled = False
        self.current_mutual_press_dir = None
        self.current_direction_alignment = float("nan")
        self.mutual_contact_anchor_1 = None
        self.mutual_contact_anchor_2 = None

    def set_k_lateral(self, k_lateral_npm: float) -> None:
        if not self._positive_finite(k_lateral_npm):
            raise ValueError("k_lateral_npm must be finite and > 0")
        if k_lateral_npm > CARTESIAN_STIFFNESS_LIMIT_NPM:
            raise ValueError(
                f"k_lateral_npm {k_lateral_npm:.2f} exceeds controller limit "
                f"{CARTESIAN_STIFFNESS_LIMIT_NPM:.2f}"
            )

        msg = Float64()
        msg.data = float(k_lateral_npm)
        self.current_case_k_lateral_npm = float(k_lateral_npm)

        deadline = self.now_s() + 3.0
        while self.now_s() < deadline:
            if (
                self.klat_pub1.get_subscription_count() > 0
                and self.klat_pub2.get_subscription_count() > 0
            ):
                break
            rclpy.spin_once(self, timeout_sec=0.05)

        for _ in range(5):
            self.klat_pub1.publish(msg)
            self.klat_pub2.publish(msg)
            self.spin_for(0.10)

    def set_k_axial(self, k_axial_npm: float) -> None:
        if not self._positive_finite(k_axial_npm):
            raise ValueError("k_axial_npm must be finite and > 0")
        if k_axial_npm > CARTESIAN_STIFFNESS_LIMIT_NPM:
            raise ValueError(
                f"k_axial_npm {k_axial_npm:.2f} exceeds controller limit "
                f"{CARTESIAN_STIFFNESS_LIMIT_NPM:.2f}"
            )

        msg = Float64()
        msg.data = float(k_axial_npm)

        deadline = self.now_s() + 3.0
        while self.now_s() < deadline:
            if (
                self.kax_pub1.get_subscription_count() > 0
                and self.kax_pub2.get_subscription_count() > 0
            ):
                break
            rclpy.spin_once(self, timeout_sec=0.05)

        for _ in range(5):
            self.kax_pub1.publish(msg)
            self.kax_pub2.publish(msg)
            self.spin_for(0.10)

    def stage0_calibration(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "calibration"

        if self.enable_ablation and (not self._has_ablation_k_source() or not self._has_ablation_peak_source()):
            info = {
                "calibration_source": "missing",
                "enable_ablation": self.enable_ablation,
                "has_k_source": self._has_ablation_k_source(),
                "has_peak_source": self._has_ablation_peak_source(),
            }
            return False, "ablation requires calibration JSON, measured spring stiffness, or explicit peak/K_lat cases", info

        try:
            raw_calibration_payload, calibration_json_path = self._load_calibration_payload()
        except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
            return False, f"calibration load failed: {error}", {"calibration_source": "load_error"}

        spring_payload = self._explicit_spring_payload()
        calibration_payload = dict(raw_calibration_payload)
        calibration_payload.update(spring_payload)
        self.calibration_payload = calibration_payload
        has_json_payload = bool(raw_calibration_payload)
        has_spring_payload = bool(spring_payload)
        if has_json_payload and has_spring_payload:
            self.current_calibration_source = "json+spring_specimen"
        elif has_json_payload:
            self.current_calibration_source = "json"
        elif has_spring_payload:
            self.current_calibration_source = "spring_specimen"
        else:
            self.current_calibration_source = "heuristic"
        self.current_calibration_json_path = calibration_json_path

        peak_reference_n = float(calibration_payload.get("P_b_hat_N", float("nan")))
        if not self._positive_finite(peak_reference_n):
            peak_reference_n = self.upper_force_limit_n
        if not self._positive_finite(peak_reference_n) and self.load_targets_n:
            peak_reference_n = max(self.load_targets_n)

        if not self._positive_finite(peak_reference_n):
            info = {
                "calibration_source": self.current_calibration_source,
                "calibration_json_path": self.current_calibration_json_path,
            }
            return False, "unable to derive calibrated peak load", info

        self.calibrated_peak_load_n = peak_reference_n
        self.calibrated_critical_k_lateral_npm = float(
            calibration_payload.get("K_lat_star_Npm", float("nan"))
        )
        self.calibrated_measured_k_lateral_npm = float(
            calibration_payload.get("K_lat_meas_Npm", float("nan"))
        )
        self.derived_peak_load_case_values_n = self._derive_peak_load_cases(peak_reference_n)
        self.derived_k_lateral_case_values_npm = self._derive_k_lateral_cases(calibration_payload)

        if not self.load_targets_explicit:
            derived_targets = self._derive_case_load_targets(
                self.derived_peak_load_case_values_n[0]
                if self.derived_peak_load_case_values_n
                else peak_reference_n
            )
            if derived_targets:
                self.load_targets_n = derived_targets
        if not self.upper_force_limit_explicit:
            self.upper_force_limit_n = peak_reference_n
        if not self.load_target_tolerance_explicit:
            self.load_target_tolerance_n = self._default_load_target_tolerance_n()
        self.force_balance_tolerance = self._default_force_balance_tolerance_n()
        if self.ignore_local_frame_ee_safety and not self.precontact_contact_threshold_explicit:
            self.precontact_contact_threshold_n = self._default_local_frame_precontact_threshold_n()

        self.current_case_peak_load_n = self.upper_force_limit_n
        if len(self.derived_k_lateral_case_values_npm) == 1 and not self.k_lateral_explicit:
            self.k_lateral_npm = self.derived_k_lateral_case_values_npm[0]
            self.current_case_k_lateral_npm = self.k_lateral_npm

        info = {
            "calibration_source": self.current_calibration_source,
            "calibration_json_path": self.current_calibration_json_path,
            "calibrated_peak_load_n": self.calibrated_peak_load_n,
            "calibrated_critical_k_lateral_npm": self.calibrated_critical_k_lateral_npm,
            "calibrated_measured_k_lateral_npm": self.calibrated_measured_k_lateral_npm,
            "derived_peak_load_case_values_n": list(self.derived_peak_load_case_values_n),
            "derived_k_lateral_case_values_npm": list(self.derived_k_lateral_case_values_npm),
            "load_targets_n": list(self.load_targets_n),
            "upper_force_limit_n": self.upper_force_limit_n,
            "k_lateral_npm": self.k_lateral_npm,
            "enable_ablation": self.enable_ablation,
        }
        for key in ("P_b_hat_N", "K_lat_star_Npm", "K_lat_meas_Npm", "k_a_Npm"):
            if key in calibration_payload:
                info[key] = calibration_payload[key]
        for key in (
            "spring_specimen",
            "spring_specimen_label",
            "spring_free_length_m",
            "spring_width_m",
            "spring_thickness_m",
            "spring_stiffness_npm",
        ):
            if key in calibration_payload:
                info[key] = calibration_payload[key]

        if has_json_payload and has_spring_payload:
            message = "calibration grid derived from calibration JSON and measured spring specimen"
        elif has_json_payload:
            message = "calibration grid derived from calibration JSON"
        elif has_spring_payload:
            message = "calibration grid prepared from measured spring specimen"
        else:
            message = "heuristic calibration grid prepared"
        return True, message, info

    def separation_safe(self) -> bool:
        if self.ignore_local_frame_ee_safety:
            return True
        return super().separation_safe()

    def x_gap_safe(self) -> bool:
        if self.ignore_local_frame_ee_safety:
            return True
        return super().x_gap_safe()

    def _using_local_frame_load_logic(self) -> bool:
        return self.ignore_local_frame_ee_safety

    def _local_press_sign(self, robot_id: int) -> float:
        offset = self.current_press_offset_x1 if robot_id == 1 else self.current_press_offset_x2
        if self.fin(offset) and abs(offset) > 1e-9:
            return math.copysign(1.0, offset)

        fallback = self.precontact_end_x_offset if robot_id == 1 else self.precontact_end_x_offset_2
        if self.fin(fallback) and abs(fallback) > 1e-9:
            return math.copysign(1.0, fallback)
        return -1.0

    def _local_raw_axis_load(self, robot_id: int) -> float:
        # Use baseline-subtracted force to avoid false contact detection
        # from persistent gravity/compensation offsets in Gazebo.
        delta = self.contact_force_delta(robot_id)
        if delta is not None and self.fin(delta[0]):
            return abs(delta[0])
        force = self.linear_force(robot_id)
        if force is not None and self.fin(force[0]):
            return abs(force[0])

        contact_fx_mag = self.contact_fx_mag_1 if robot_id == 1 else self.contact_fx_mag_2
        if self.fin(contact_fx_mag):
            return abs(contact_fx_mag)
        return float("nan")

    def _local_axis_load(self, robot_id: int) -> float:
        return self._local_raw_axis_load(robot_id)

    def _apply_precontact_target(self, target: Dict[str, float], contact_mode: str) -> None:
        # Route through V3's apply_press_target which detects target_x1/target_x2
        # and uses frame_id="absolute" to send actual world coordinates.
        # This ensures each robot moves to its own correct position rather than
        # both converging to a shared offset reference.
        self.apply_press_target(target, contact_mode)

    def _step_offset_toward(
        self,
        current_offset: float,
        target_offset: float,
        step_magnitude: float,
    ) -> float:
        if step_magnitude <= 0.0 or abs(current_offset - target_offset) <= 1e-12:
            return current_offset
        if current_offset < target_offset:
            return min(current_offset + step_magnitude, target_offset)
        return max(current_offset - step_magnitude, target_offset)

    def contact_detected(self, robot_id: int) -> bool:
        contact_valid = self.contact_valid_1 if robot_id == 1 else self.contact_valid_2
        if not contact_valid:
            return False

        if self._using_local_frame_load_logic():
            local_load = self._local_raw_axis_load(robot_id)
            return self.fin(local_load) and local_load >= self.precontact_contact_threshold_n

        delta_norm = self.vector_norm(self.contact_force_delta(robot_id))
        if self.fin(delta_norm):
            return delta_norm >= self.precontact_contact_threshold_n

        contact_fx_mag = self.contact_fx_mag_1 if robot_id == 1 else self.contact_fx_mag_2
        return self.fin(contact_fx_mag) and contact_fx_mag >= self.precontact_contact_threshold_n

    def projected_force(self, robot_id: int, direction: Optional[Tuple[float, float, float]]) -> float:
        if self._using_local_frame_load_logic():
            return self._local_axis_load(robot_id)
        return super().projected_force(robot_id, direction)

    def compute_mutual_press_direction(self) -> Optional[Tuple[float, float, float]]:
        if self._using_local_frame_load_logic():
            if not self.bilateral_contact_confirmed():
                self.current_direction_alignment = float("nan")
                return None
            self.current_direction_alignment = 1.0
            return (1.0, 0.0, 0.0)
        return super().compute_mutual_press_direction()

    def compute_directional_press_target(
        self,
        distance1: float,
        distance2: float,
        direction: Tuple[float, float, float],
    ) -> Dict[str, float]:
        if self._using_local_frame_load_logic():
            return {
                "offset_x1": self._local_press_sign(1) * abs(distance1),
                "offset_y1": 0.0,
                "offset_z1": 0.0,
                "offset_x2": self._local_press_sign(2) * abs(distance2),
                "offset_y2": 0.0,
                "offset_z2": 0.0,
            }
        return super().compute_directional_press_target(distance1, distance2, direction)

    def _wait_for_precontact_waypoint_completion(
        self,
        *,
        expect_robot1_motion: bool,
        expect_robot2_motion: bool,
        start_desired_x_1: Optional[float] = None,
        start_desired_x_2: Optional[float] = None,
        expected_delta_x1: float = 0.0,
        expected_delta_x2: float = 0.0,
        republish: Optional[Callable[[], None]] = None,
    ) -> bool:
        end_t = self.now_s() + self.precontact_waypoint_timeout_s
        saw_robot1_active = not expect_robot1_motion
        saw_robot2_active = not expect_robot2_motion
        response_1_observed = (
            not expect_robot1_motion
            or start_desired_x_1 is None
            or not self.fin(start_desired_x_1)
            or abs(expected_delta_x1) <= 1e-12
        )
        response_2_observed = (
            not expect_robot2_motion
            or start_desired_x_2 is None
            or not self.fin(start_desired_x_2)
            or abs(expected_delta_x2) <= 1e-12
        )
        last_republish_t = self.now_s()
        last_log_t = self.now_s()
        settled_since: Optional[float] = None

        while True:
            current_t = self.now_s()
            if current_t >= end_t:
                break
            robot1_active = bool(self.wp1_active)
            robot2_active = bool(self.wp2_active)
            saw_robot1_active = saw_robot1_active or robot1_active
            saw_robot2_active = saw_robot2_active or robot2_active
            desired_x_1 = self.des1.position.x if self.des1 is not None else float("nan")
            desired_x_2 = self.des2.position.x if self.des2 is not None else float("nan")
            if not response_1_observed and self.fin(desired_x_1):
                response_1_observed = abs(desired_x_1 - start_desired_x_1) >= 1e-4
            if not response_2_observed and self.fin(desired_x_2):
                response_2_observed = abs(desired_x_2 - start_desired_x_2) >= 1e-4
            if (
                republish is not None
                and ((expect_robot1_motion and not response_1_observed) or (expect_robot2_motion and not response_2_observed))
                and current_t - last_republish_t >= 0.25
            ):
                republish()
                last_republish_t = current_t
                # Log a compact reprobe status when we republish the waypoint
                try:
                    ee1 = self.ee1
                    ee2 = self.ee2
                    des1_pos = self.des1.position if self.des1 is not None else None
                    des2_pos = self.des2.position if self.des2 is not None else None
                    err1 = float('nan')
                    err2 = float('nan')
                    if ee1 is not None and des1_pos is not None:
                        err1 = self.vector_norm((ee1.x - des1_pos.x, ee1.y - des1_pos.y, ee1.z - des1_pos.z))
                    if ee2 is not None and des2_pos is not None:
                        err2 = self.vector_norm((ee2.x - des2_pos.x, ee2.y - des2_pos.y, ee2.z - des2_pos.z))
                    self.get_logger().debug(
                        f"precontact: republish t={current_t:.2f} saw1={saw_robot1_active} saw2={saw_robot2_active} "
                        f"wp1={robot1_active} wp2={robot2_active} resp1={response_1_observed} resp2={response_2_observed} "
                        f"des1_x={desired_x_1:.4f} des2_x={desired_x_2:.4f} err1={err1:.4f} err2={err2:.4f}"
                    )
                except Exception:
                    pass
            if (
                saw_robot1_active
                and saw_robot2_active
                and response_1_observed
                and response_2_observed
                and not robot1_active
                and not robot2_active
            ):
                return True
            if (
                saw_robot1_active
                and saw_robot2_active
                and response_1_observed
                and response_2_observed
                and self.directional_waypoint_pose_settled()
            ):
                if settled_since is None:
                    settled_since = current_t
                elif current_t - settled_since >= 0.15:
                    return True
            elif response_1_observed and response_2_observed and self.directional_waypoint_pose_settled():
                if settled_since is None:
                    settled_since = current_t
                elif current_t - settled_since >= 0.15:
                    return True
            else:
                settled_since = None
            # Periodic lightweight diagnostic logging to help CI debugging when
            # precontact waypoints do not appear to settle. Logged at DEBUG level
            # to avoid spamming normal runs.
            try:
                if current_t - last_log_t >= 0.5:
                    ee1 = self.ee1
                    ee2 = self.ee2
                    des1_pos = self.des1.position if self.des1 is not None else None
                    des2_pos = self.des2.position if self.des2 is not None else None
                    err1 = float('nan')
                    err2 = float('nan')
                    if ee1 is not None and des1_pos is not None:
                        err1 = self.vector_norm((ee1.x - des1_pos.x, ee1.y - des1_pos.y, ee1.z - des1_pos.z))
                    if ee2 is not None and des2_pos is not None:
                        err2 = self.vector_norm((ee2.x - des2_pos.x, ee2.y - des2_pos.y, ee2.z - des2_pos.z))

                    self.get_logger().debug(
                        f"precontact: t={current_t:.2f} saw1={saw_robot1_active} saw2={saw_robot2_active} "
                        f"wp1={robot1_active} wp2={robot2_active} resp1={response_1_observed} resp2={response_2_observed} "
                        f"des1_x={desired_x_1:.4f} des2_x={desired_x_2:.4f} err1={err1:.4f} err2={err2:.4f} settled={self.directional_waypoint_pose_settled()}"
                    )
                    last_log_t = current_t
            except Exception:
                pass

            rclpy.spin_once(self, timeout_sec=0.05)

        return False

    def _wait_for_arms_idle(self, timeout_s: float = 12.0) -> bool:
        """Idle wait removed — quasistatic speed enforced by small step size, not velocity monitoring."""
        self.spin_for(0.25)
        return True

    def _directional_target_settle_position_tolerance_m(self) -> float:
        if self._using_local_frame_load_logic() and (
            (
                self.current_contact_mode.startswith("load_step_")
                and (
                    "_seek:" in self.current_contact_mode
                    or "_unload:" in self.current_contact_mode
                    or "_hold_recover:" in self.current_contact_mode
                )
            )
            or self.current_contact_mode.startswith("directional_hold_")
        ):
            return max(0.0015, 6.0 * self.load_seek_step_m)
        return 0.0015

    def _directional_target_pose_settled(self, position_tolerance_m: float) -> bool:
        if self._using_local_frame_load_logic() and (
            self.current_contact_mode.startswith("load_step_")
            or self.current_contact_mode.startswith("directional_hold_")
        ):
            ee_1 = self.ee1
            ee_2 = self.ee2
            desired_1 = self.des1
            desired_2 = self.des2
            if ee_1 is not None and ee_2 is not None and desired_1 is not None and desired_2 is not None:
                error_1 = self.vector_norm(
                    (
                        ee_1.x - desired_1.position.x,
                        ee_1.y - desired_1.position.y,
                        ee_1.z - desired_1.position.z,
                    )
                )
                error_2 = self.vector_norm(
                    (
                        ee_2.x - desired_2.position.x,
                        ee_2.y - desired_2.position.y,
                        ee_2.z - desired_2.position.z,
                    )
                )
                if self.finite(error_1) and self.finite(error_2):
                    if error_1 > position_tolerance_m or error_2 > position_tolerance_m:
                        return False
                    # Velocity check removed — quasistatic speed enforced by small step size
                    return True

        return self.directional_waypoint_pose_settled(position_tolerance_m=position_tolerance_m)

    def _wait_for_directional_target_settle(self, max_wait_s: float) -> bool:
        max_wait_s = max(0.0, max_wait_s)
        end_t = self.now_s() + max_wait_s
        settle_duration_s = min(0.15, max_wait_s)
        settled_since: Optional[float] = None
        position_tolerance_m = self._directional_target_settle_position_tolerance_m()

        while True:
            current_t = self.now_s()
            if self._directional_target_pose_settled(position_tolerance_m=position_tolerance_m):
                if settled_since is None:
                    settled_since = current_t
                elif current_t - settled_since >= settle_duration_s:
                    return True
            else:
                settled_since = None

            if current_t >= end_t:
                break
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.0, end_t - current_t)))

        return bool(self._directional_target_pose_settled(position_tolerance_m=position_tolerance_m))

    def _projected_loads(self, direction: Optional[Tuple[float, float, float]]) -> Tuple[float, float]:
        if direction is None:
            return float("nan"), float("nan")
        load_1 = self.projected_force(1, direction)
        load_2 = self.projected_force(2, direction)
        return load_1, load_2

    def _update_load_metrics(self, direction: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
        load_1, load_2 = self._projected_loads(direction)
        if self.fin(load_1) and self.fin(load_2):
            # Use absolute values to ensure shared_load is positive for compression
            self.current_shared_load_n = min(abs(load_1), abs(load_2))
            self.current_mean_load_n = 0.5 * (abs(load_1) + abs(load_2))
        else:
            self.current_shared_load_n = float("nan")
            self.current_mean_load_n = float("nan")
        return load_1, load_2, self.current_shared_load_n, self.current_mean_load_n

    def _directional_target_matches_current(self, target: Dict[str, float], contact_mode: Optional[str] = None) -> bool:
        target_1 = (
            target["target_x1"],
            target["target_y1"],
            target["target_z1"],
        )
        target_2 = (
            target["target_x2"],
            target["target_y2"],
            target["target_z2"],
        )
        if contact_mode is not None and self.current_contact_mode != contact_mode:
            return False
        return self.current_directional_target_1 == target_1 and self.current_directional_target_2 == target_2

    def _upper_force_limit_reached(self, load_1: float, load_2: float) -> bool:
        finite_loads = [load for load in (load_1, load_2) if self.fin(load)]
        return bool(finite_loads) and max(finite_loads) >= self.upper_force_limit_n

    def _next_unload_distances(
        self,
        direction: Tuple[float, float, float],
    ) -> Tuple[float, float, str]:
        next_distance_1 = max(0.0, self.current_press_distance_1 - self.load_seek_step_m)
        next_distance_2 = max(0.0, self.current_press_distance_2 - self.load_seek_step_m)
        load_1, load_2, _shared, _mean = self._update_load_metrics(direction)
        if self.fin(load_1) and self.fin(load_2) and self.load_balance_step_m > 0.0:
            if load_1 > load_2 + self.force_balance_tolerance:
                next_distance_2 = max(0.0, next_distance_2 - self.load_balance_step_m)
                return next_distance_1, next_distance_2, "unload_robot2_more"
            if load_2 > load_1 + self.force_balance_tolerance:
                next_distance_1 = max(0.0, next_distance_1 - self.load_balance_step_m)
                return next_distance_1, next_distance_2, "unload_robot1_more"
        return next_distance_1, next_distance_2, "unload_balanced"

    def _append_load_sample(
        self,
        stage_info: List[Dict[str, Any]],
        *,
        command_distance: float,
        step_index: int,
        target_load_n: float,
        state: str,
    ) -> Dict[str, Any]:
        self.current_load_step_index = step_index
        self.current_load_target_n = target_load_n
        self.current_load_step_state = state
        snap = self.snapshot()
        snap["command_distance"] = command_distance
        stage_info.append(snap)
        self.log_sync_step(command_distance)
        return snap

    def _stage4_handoff_target_load_n(self) -> float:
        if not self._using_local_frame_load_logic():
            return float("nan")
        # Allow callers to decouple the stage-4 handoff load from the load
        # schedule, so handoff completes as soon as bilateral contact is
        # established without requiring stage 4 to drive against the spring.
        if self._positive_finite(self.stage4_handoff_target_n):
            return float(self.stage4_handoff_target_n)
        for target_load_n in self.load_targets_n:
            if self._positive_finite(target_load_n):
                return float(target_load_n)
        if self._positive_finite(self.upper_force_limit_n):
            return float(self.upper_force_limit_n)
        return float("nan")

    def _stage4_handoff_load_ready(self, shared_load_n: float) -> bool:
        target_load_n = self._stage4_handoff_target_load_n()
        if not self._positive_finite(target_load_n):
            return True
        tolerance_n = self.load_target_tolerance_n if self._positive_finite(self.load_target_tolerance_n) else 0.0
        return self.fin(shared_load_n) and shared_load_n + tolerance_n >= target_load_n

    def _stage4_handoff_direction(self, entry_contact_ready: bool) -> Optional[Tuple[float, float, float]]:
        if entry_contact_ready:
            direction = self.compute_mutual_press_direction()
            if direction is not None:
                return direction
        if self._using_local_frame_load_logic() and self.current_mutual_press_dir is not None:
            return self.current_mutual_press_dir
        return None

    def _reset_local_frame_load_step_reference(self) -> bool:
        if not self._using_local_frame_load_logic():
            return False
        # Always anchor from ACTUAL end-effector positions (not commanded targets).
        # On hardware, the arms can't reach commanded targets due to gravity/compliance,
        # creating a persistent gap. Anchoring from commanded targets makes the preload-relief
        # target unreachable, causing timeout. Use actual EE X/Y but keep safe Z.
        if not self.capture_mutual_contact_anchor():
            self.get_logger().error("_reset_local_frame_load_step_reference: capture_mutual_contact_anchor failed")
            return False
        # Override Z with safe push-axis height to keep both arms at consistent height
        anchor_z_1 = self.configured_push_axis_height(self.mutual_contact_anchor_1[2])
        anchor_z_2 = self.configured_push_axis_height(self.mutual_contact_anchor_2[2])
        self.mutual_contact_anchor_1 = (
            self.mutual_contact_anchor_1[0],
            self.mutual_contact_anchor_1[1],
            anchor_z_1,
        )
        self.mutual_contact_anchor_2 = (
            self.mutual_contact_anchor_2[0],
            self.mutual_contact_anchor_2[1],
            anchor_z_2,
        )
        self.last_mutual_contact_anchor_1 = self.mutual_contact_anchor_1
        self.last_mutual_contact_anchor_2 = self.mutual_contact_anchor_2
        self.current_press_distance_1 = 0.0
        self.current_press_distance_2 = 0.0
        return True

    def stage4_hold(self) -> Tuple[bool, str, Dict[str, Any]]:
        if not self._using_local_frame_load_logic():
            return super().stage4_hold()

        self.current_phase = "hold"
        stage_info: List[Dict[str, Any]] = []

        # Allow the robot to settle at the stage 3 anchor position before
        # checking contact. This prevents false negatives from transient forces.
        if not self._wait_for_arms_idle(timeout_s=5.0):
            self.get_logger().warn("Arms did not settle before stage 4 pressing; proceeding with caution")

        entry_contact_ready = self.bilateral_contact_confirmed()
        direction = self._stage4_handoff_direction(entry_contact_ready)
        if direction is None:
            if not entry_contact_ready:
                self.set_mutual_contact_state(False, "hold_requires_bilateral_contact")
                return False, "mutual contact not maintained for hold", {"samples": stage_info}
            self.set_mutual_contact_state(False, "hold_direction_unavailable")
            return False, "mutual contact direction unavailable", {"samples": stage_info}

        self.current_mutual_press_dir = direction
        self.directional_press_enabled = True
        self.current_press_distance_1 = max(self.current_press_distance_1, self.press_step)
        self.current_press_distance_2 = max(self.current_press_distance_2, self.press_step)

        handoff_window_s = max(
            self.directional_refresh_interval,
            min(self.directional_waypoint_timeout, max(self.directional_hold_duration, 1.0)),
        )
        handoff_deadline = self.now_s() + handoff_window_s
        _handoff_step_budget_s = max(
            self.directional_hold_duration * 2.0,
            self.directional_waypoint_timeout * 4.0,
        )
        handoff_target_load_n = self._stage4_handoff_target_load_n()
        recovery_limit_m = max(self.current_press_distance_1, self.current_press_distance_2) + max(
            self.max_additional_load_travel_m if self._positive_finite(handoff_target_load_n) else 0.0,
            self.load_seek_step_m,
            self.load_balance_step_m,
            self.press_step,
        )
        # Velocity limits removed — quasistatic speed enforced by small step size

        while True:
            target = self.compute_absolute_directional_press_target(
                self.current_press_distance_1,
                self.current_press_distance_2,
                direction,
            )
            if target is None:
                self.set_mutual_contact_state(False, "hold_target_unavailable")
                return False, "directional target unavailable", {"samples": stage_info}

            self.publish_directional_press_target(target, "directional_hold_handoff")
            handoff_settled = self._wait_for_directional_target_settle(
                min(self.directional_waypoint_timeout, max(0.0, handoff_deadline - self.now_s()))
            )
            violation = self._directional_abort_message()
            self._update_load_metrics(direction)
            self._append_stage_sample(
                stage_info,
                command_distance=max(self.current_press_distance_1, self.current_press_distance_2),
            )
            handoff_contact_ready = self.bilateral_contact_confirmed()
            handoff_load_ready = self._stage4_handoff_load_ready(self.current_shared_load_n)
            if violation is not None:
                return False, violation, {"samples": stage_info, "local_frame_handoff": True}
            if handoff_contact_ready and handoff_load_ready:
                return True, "directional hold handoff stable", {
                    "samples": stage_info,
                    "local_frame_handoff": True,
                }
            if self.now_s() >= handoff_deadline:
                # If contact is good and we still have travel room and step budget,
                # extend the deadline for one more recovery step rather than failing.
                _travel_remaining = recovery_limit_m - max(
                    self.current_press_distance_1, self.current_press_distance_2
                )
                if (
                    handoff_contact_ready
                    and not handoff_load_ready
                    and _travel_remaining > self.load_seek_step_m
                    and _handoff_step_budget_s > 0.0
                ):
                    _extension = min(self.directional_waypoint_timeout, _handoff_step_budget_s)
                    handoff_deadline = self.now_s() + _extension
                    _handoff_step_budget_s -= _extension
                elif handoff_contact_ready:
                    self.set_mutual_contact_state(False, "handoff_preload_below_first_load_target")
                    return False, "directional hold preload below first load target", {
                        "samples": stage_info,
                        "local_frame_handoff": True,
                        "required_preload_target_n": handoff_target_load_n,
                    }
                else:
                    break
            if not handoff_settled:
                continue

            next_distance_1 = min(self.current_press_distance_1 + self.load_seek_step_m, recovery_limit_m)
            next_distance_2 = min(self.current_press_distance_2 + self.load_seek_step_m, recovery_limit_m)
            next_distance_1, next_distance_2, contact_mode = self._balanced_distances(
                next_distance_1,
                next_distance_2,
                direction,
                recovery_limit_m,
            )
            if (
                next_distance_1 <= self.current_press_distance_1 + 1e-12
                and next_distance_2 <= self.current_press_distance_2 + 1e-12
            ):
                break

            self.current_press_distance_1 = next_distance_1
            self.current_press_distance_2 = next_distance_2
            target = self.apply_directional_press_target(
                next_distance_1,
                next_distance_2,
                direction,
                f"directional_hold_recover:{contact_mode}",
            )
            if target is None:
                self.set_mutual_contact_state(False, "hold_recovery_target_unavailable")
                return False, "directional target unavailable during hold recovery", {
                    "samples": stage_info,
                    "local_frame_handoff": True,
                }
            recovery_settled = self._wait_for_directional_target_settle(
                min(self.directional_waypoint_timeout, max(0.0, handoff_deadline - self.now_s()))
            )
            if not recovery_settled:
                self.set_mutual_contact_state(False, "hold_recovery_waypoint_timeout")
                return False, "directional hold recovery command did not settle", {
                    "samples": stage_info,
                    "local_frame_handoff": True,
                }

            # Velocity limits removed — quasistatic speed enforced by small step size
            violation = self._directional_abort_message()
            self._update_load_metrics(direction)
            self._append_stage_sample(
                stage_info,
                command_distance=max(self.current_press_distance_1, self.current_press_distance_2),
            )
            handoff_contact_ready = self.bilateral_contact_confirmed()
            handoff_load_ready = self._stage4_handoff_load_ready(self.current_shared_load_n)
            if violation is not None:
                return False, violation, {"samples": stage_info, "local_frame_handoff": True}
            if handoff_contact_ready and handoff_load_ready:
                return True, "directional hold handoff recovered preload", {
                    "samples": stage_info,
                    "local_frame_handoff": True,
                }

        self.set_mutual_contact_state(False, "mutual_contact_lost_during_stage4_handoff")
        return False, "mutual contact lost during directional hold", {
            "samples": stage_info,
            "local_frame_handoff": True,
        }

    def _quasistatic_unload(
        self,
        step_index: int,
        target_load_n: float,
        stage_info: List[Dict[str, Any]],
        reason: str,
    ) -> Tuple[bool, str]:
        self.current_phase = "unload"
        self.current_load_step_index = step_index
        self.current_load_target_n = target_load_n
        self._append_load_sample(
            stage_info,
            command_distance=max(self.current_press_distance_1, self.current_press_distance_2),
            step_index=step_index,
            target_load_n=target_load_n,
            state="upper_force_limit_reached",
        )

        while self.current_press_distance_1 > 1e-12 or self.current_press_distance_2 > 1e-12:
            direction = self.current_mutual_press_dir
            if direction is None:
                direction = self.compute_mutual_press_direction()
                if direction is None:
                    return False, "mutual contact direction unavailable during quasi-static unload"
            self.current_mutual_press_dir = direction
            self.directional_press_enabled = True

            next_distance_1, next_distance_2, contact_mode = self._next_unload_distances(direction)
            if (
                next_distance_1 >= self.current_press_distance_1 - 1e-12
                and next_distance_2 >= self.current_press_distance_2 - 1e-12
            ):
                break

            self.current_press_distance_1 = next_distance_1
            self.current_press_distance_2 = next_distance_2
            target = self.apply_directional_press_target(
                next_distance_1,
                next_distance_2,
                direction,
                f"load_step_{step_index}_unload:{contact_mode}:{reason}",
            )
            if target is None:
                return False, "directional target unavailable during quasi-static unload"
            if not self._wait_for_directional_target_settle(self.load_step_waypoint_timeout):
                return False, "quasi-static unload command did not settle"

            violation = self._directional_abort_message(self.move_vel_limit)
            self._append_load_sample(
                stage_info,
                command_distance=max(next_distance_1, next_distance_2),
                step_index=step_index,
                target_load_n=target_load_n,
                state="unloading",
            )
            if violation is not None:
                return False, violation

        self.set_mutual_contact_state(False, "upper_force_limit_quasistatic_unload_complete")
        self.directional_press_enabled = False
        self._append_load_sample(
            stage_info,
            command_distance=max(self.current_press_distance_1, self.current_press_distance_2),
            step_index=step_index,
            target_load_n=target_load_n,
            state="unloaded_after_force_limit",
        )
        return (
            True,
            f"upper force limit {self.upper_force_limit_n:.2f} N reached; quasi-static unload complete",
        )

    def _relieve_preload_before_load_steps(
        self,
        step_index: int,
        target_load_n: float,
        stage_info: List[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        preload_shared_ceiling_n = min(target_load_n, self.upper_force_limit_n)
        # If the target load is already at or below the current shared load,
        # backing off would only destroy contact (Gazebo bilateral residual
        # often exceeds tiny early-stage targets). Skip preload relief in that
        # case and let the seek loop handle convergence directly.
        if (
            self.fin(self.current_shared_load_n)
            and self.current_shared_load_n + self.load_target_tolerance_n
            >= preload_shared_ceiling_n
        ):
            return True, ""

        while True:
            if not self._load_step_contact_confirmed(target_load_n):
                self.set_mutual_contact_state(False, "mutual_contact_lost_during_preload_relief")
                return False, "mutual contact lost while relieving preload before load steps"

            direction = self._resolve_load_step_direction(target_load_n)
            if direction is None:
                self.set_mutual_contact_state(False, "preload_relief_direction_unavailable")
                return False, "mutual contact direction unavailable during preload relief"
            self.current_mutual_press_dir = direction
            self.directional_press_enabled = True

            load_1, load_2, shared_load, _mean_load = self._update_load_metrics(direction)
            finite_loads = [load for load in (load_1, load_2) if self.fin(load)]
            if not finite_loads:
                return True, ""

            max_load = max(finite_loads)
            shared_preload_ok = not self.fin(shared_load) or shared_load <= preload_shared_ceiling_n
            self.get_logger().info(f"preload_relief iteration: press_distance_1={self.current_press_distance_1:.4f}, press_distance_2={self.current_press_distance_2:.4f}, max_load={max_load:.3f}, shared_load={shared_load:.3f}, shared_preload_ok={shared_preload_ok}")
            if max_load <= self.upper_force_limit_n and shared_preload_ok:
                return True, ""

            next_distance_1, next_distance_2, contact_mode = self._next_unload_distances(direction)
            self.get_logger().info(f"preload_relief next distances: next_distance_1={next_distance_1:.4f}, next_distance_2={next_distance_2:.4f}, contact_mode={contact_mode}")
            stopping_near_target_floor = (
                next_distance_1 <= 1e-12
                and next_distance_2 <= 1e-12
                and self.fin(shared_load)
                and shared_load <= preload_shared_ceiling_n + self.force_balance_tolerance
                and max_load <= self.upper_force_limit_n
            )
            if stopping_near_target_floor:
                return True, ""
            if (
                next_distance_1 >= self.current_press_distance_1 - 1e-12
                and next_distance_2 >= self.current_press_distance_2 - 1e-12
            ):
                at_preload_floor = (
                    self.current_press_distance_1 <= 1e-12
                    and self.current_press_distance_2 <= 1e-12
                )
                if at_preload_floor and max_load <= self.upper_force_limit_n:
                    return True, ""
                return False, "unable to relieve preload before load steps"

            self.current_press_distance_1 = next_distance_1
            self.current_press_distance_2 = next_distance_2
            target = self.apply_directional_press_target(
                next_distance_1,
                next_distance_2,
                direction,
                f"load_step_{step_index}_preload_relief:{contact_mode}",
            )
            if target is None:
                return False, "directional target unavailable during preload relief"
            # Use target-settle check instead of waypoint-completion: preload relief
            # may command tiny motions where wp_active flags never trigger
            if not self._wait_for_directional_target_settle(self.load_step_waypoint_timeout):
                return False, "preload-relief command did not settle"

            violation = self._directional_abort_message(self.move_vel_limit)
            self._append_load_sample(
                stage_info,
                command_distance=max(next_distance_1, next_distance_2),
                step_index=step_index,
                target_load_n=target_load_n,
                state="preload_relief",
            )
            if violation is not None:
                return False, violation

    def _set_contact_anchor_from_desired(self) -> None:
        if self.des1 is None or self.des2 is None:
            return
        self._update_push_axis_height_calibration()
        anchor_1 = (
            self.des1.position.x,
            self.des1.position.y,
            self.des1.position.z,
        )
        anchor_2 = (
            self.des2.position.x,
            self.des2.position.y,
            self.des2.position.z,
        )
        anchor_z_1 = self.configured_push_axis_height(anchor_1[2])
        anchor_z_2 = self.configured_push_axis_height(anchor_2[2])
        self.mutual_contact_anchor_1 = (
            anchor_1[0],
            anchor_1[1],
            anchor_z_1,
        )
        self.mutual_contact_anchor_2 = (
            anchor_2[0],
            anchor_2[1],
            anchor_z_2,
        )
        self.last_mutual_contact_anchor_1 = self.mutual_contact_anchor_1
        self.last_mutual_contact_anchor_2 = self.mutual_contact_anchor_2

    def _directional_abort_message(self, velocity_limit: float = None) -> Optional[str]:
        """Check for safety violations during directional pressing.
        
        Note: velocity_limit parameter is deprecated - low velocity is enforced
        by command step size, not by monitoring joint velocities.
        """
        if not self.separation_safe():
            self.get_logger().error(
                f"EE separation {self.ee_separation():.4f} m below safety limit "
                f"{self.min_ee_separation:.4f} m"
            )
            self.set_mutual_contact_state(False, "separation_below_safety_limit")
            return "inter-robot separation below safety limit"
        if not self.x_gap_safe():
            self.get_logger().error(
                f"EE x gap {self.ee_x_gap():.4f} m above safety limit "
                f"{self.max_ee_x_gap:.4f} m"
            )
            self.set_mutual_contact_state(False, "x_gap_above_safety_limit")
            return "inter-robot x gap above safety limit"
        if not self.pose_safe():
            self.set_mutual_contact_state(False, "pose_below_x_floor_safety_limit")
            return "EE pose below x floor safety limit"
        # Velocity check removed - low speed is enforced by small step size
        # and low move_vel_limit in YAML, not by monitoring joint velocities

        direction = self.current_mutual_press_dir
        load_1, load_2, _shared, _mean = self._update_load_metrics(direction)
        if self.fin(load_1) and self.fin(load_2):
            if abs(load_1 - load_2) > self.force_diff_abort:
                self.set_mutual_contact_state(False, "directional_force_asymmetry_too_high")
                return "directional force asymmetry too high"
            if max(load_1, load_2) > self.max_force_mag_abort:
                self.set_mutual_contact_state(False, "directional_force_too_high")
                return "directional contact force too high"
        return None

    def _balanced_distances(
        self,
        distance1: float,
        distance2: float,
        direction: Tuple[float, float, float],
        travel_limit_m: float,
    ) -> Tuple[float, float, str]:
        load_1, load_2, _shared, _mean = self._update_load_metrics(direction)
        if self.fin(load_1) and self.fin(load_2):
            if load_1 + self.force_balance_tolerance < load_2:
                distance1 = min(distance1 + self.load_balance_step_m, travel_limit_m)
                return distance1, distance2, "push_robot1_more_along_direction"
            if load_2 + self.force_balance_tolerance < load_1:
                distance2 = min(distance2 + self.load_balance_step_m, travel_limit_m)
                return distance1, distance2, "push_robot2_more_along_direction"
            return distance1, distance2, "directionally_balanced"
        return distance1, distance2, "directional_seek"

    def _seek_load_target(
        self,
        step_index: int,
        target_load_n: float,
        stage_info: List[Dict[str, Any]],
        travel_limit_m: float,
    ) -> Tuple[bool, str, bool]:
        self.current_load_step_index = step_index
        self.current_load_target_n = target_load_n
        self.current_load_travel_limit_m = travel_limit_m
        stage_entry_press_distance = max(self.current_press_distance_1, self.current_press_distance_2)

        while True:
            if not self._load_step_contact_confirmed(target_load_n):
                self.set_mutual_contact_state(False, "mutual_contact_lost_before_target_load")
                return False, "mutual contact lost before reaching target load", False

            direction = self._resolve_load_step_direction(target_load_n)
            if direction is None:
                self.set_mutual_contact_state(False, "target_load_direction_unavailable")
                return False, "mutual contact direction unavailable", False
            self.current_mutual_press_dir = direction
            self.directional_press_enabled = True

            load_1, load_2, shared_load, _mean_load = self._update_load_metrics(direction)
            force_target_satisfied = (
                self.fin(shared_load) and shared_load + self.load_target_tolerance_n >= target_load_n
            )
            current_max_press = max(self.current_press_distance_1, self.current_press_distance_2)
            stage_advance_m = current_max_press - stage_entry_press_distance
            stage_advance_satisfied = stage_advance_m + 1e-12 >= self.min_stage_advance_m
            if force_target_satisfied and stage_advance_satisfied:
                self._append_load_sample(
                    stage_info,
                    command_distance=current_max_press,
                    step_index=step_index,
                    target_load_n=target_load_n,
                    state="target_reached",
                )
                return True, "", False
            if self._upper_force_limit_reached(load_1, load_2):
                ok, message = self._quasistatic_unload(
                    step_index,
                    target_load_n,
                    stage_info,
                    "seek",
                )
                return ok, message, ok

            next_distance_1 = min(self.current_press_distance_1 + self.load_seek_step_m, travel_limit_m)
            next_distance_2 = min(self.current_press_distance_2 + self.load_seek_step_m, travel_limit_m)
            if (
                next_distance_1 <= self.current_press_distance_1 + 1e-12
                and next_distance_2 <= self.current_press_distance_2 + 1e-12
            ):
                self._append_load_sample(
                    stage_info,
                    command_distance=max(self.current_press_distance_1, self.current_press_distance_2),
                    step_index=step_index,
                    target_load_n=target_load_n,
                    state="travel_limit_reached",
                )
                self.set_mutual_contact_state(False, "target_load_not_reached_before_travel_limit")
                return False, f"target load {target_load_n:.2f} N not reached before travel limit", False

            next_distance_1, next_distance_2, contact_mode = self._balanced_distances(
                next_distance_1,
                next_distance_2,
                direction,
                travel_limit_m,
            )
            self.current_press_distance_1 = next_distance_1
            self.current_press_distance_2 = next_distance_2

            target = self.apply_directional_press_target(
                next_distance_1,
                next_distance_2,
                direction,
                f"load_step_{step_index}_seek:{contact_mode}",
            )
            if target is None:
                self.set_mutual_contact_state(False, "target_load_target_unavailable")
                return False, "directional target unavailable", False
            seek_settled = self._wait_for_directional_target_settle(self.load_step_waypoint_timeout)
            if not seek_settled:
                self.set_mutual_contact_state(False, "target_load_waypoint_timeout")
                return False, "contact-gated load command did not settle", False

            violation = self._directional_abort_message(self.move_vel_limit)
            self._append_load_sample(
                stage_info,
                command_distance=max(next_distance_1, next_distance_2),
                step_index=step_index,
                target_load_n=target_load_n,
                state="seeking",
            )
            if violation is not None:
                return False, violation, False

    def _hold_load_target(
        self,
        step_index: int,
        target_load_n: float,
        stage_info: List[Dict[str, Any]],
        travel_limit_m: float,
    ) -> Tuple[bool, str, bool]:
        end_t = self.now_s() + self.load_hold_duration_s
        self.current_load_step_index = step_index
        self.current_load_target_n = target_load_n
        self.current_load_travel_limit_m = travel_limit_m
        hold_reacquire_used = False
        allow_seek_to_hold_ringdown = self.current_contact_mode.startswith(f"load_step_{step_index}_seek") or (
            step_index == 1 and self.current_contact_mode.startswith("directional_hold_")
        )
        hold_contact_grace_samples_remaining = 2 if allow_seek_to_hold_ringdown else 0
        hold_velocity_grace_samples_remaining = 2 if allow_seek_to_hold_ringdown else 0

        while self.now_s() < end_t:
            if not self._load_step_contact_confirmed(target_load_n):
                if hold_contact_grace_samples_remaining > 0:
                    hold_contact_grace_samples_remaining -= 1
                    continue
                self.set_mutual_contact_state(False, "mutual_contact_lost_during_load_hold")
                return False, "mutual contact lost during load hold", False

            direction = self._resolve_load_step_direction(target_load_n)
            if direction is None:
                self.set_mutual_contact_state(False, "load_hold_direction_unavailable")
                return False, "mutual contact direction unavailable", False
            self.current_mutual_press_dir = direction
            self.directional_press_enabled = True

            load_1, load_2, shared_load, _mean_load = self._update_load_metrics(direction)
            if not self.fin(shared_load):
                self.set_mutual_contact_state(False, "load_hold_projection_unavailable")
                return False, "projected load unavailable during hold", False

            if shared_load + self.load_target_tolerance_n < target_load_n:
                if self._upper_force_limit_reached(load_1, load_2):
                    ok, message = self._quasistatic_unload(
                        step_index,
                        target_load_n,
                        stage_info,
                        "hold",
                    )
                    return ok, message, ok
                hold_target = self.compute_absolute_directional_press_target(
                    self.current_press_distance_1,
                    self.current_press_distance_2,
                    direction,
                )
                if (
                    not hold_reacquire_used
                    and hold_target is not None
                    and self._directional_target_matches_current(hold_target, f"load_step_{step_index}_hold")
                ):
                    self.publish_directional_press_target(
                        hold_target,
                        f"load_step_{step_index}_hold_recover:reacquire",
                    )
                    hold_reacquire_used = True
                else:
                    next_distance_1 = min(self.current_press_distance_1 + self.load_seek_step_m, travel_limit_m)
                    next_distance_2 = min(self.current_press_distance_2 + self.load_seek_step_m, travel_limit_m)
                    if (
                        next_distance_1 <= self.current_press_distance_1 + 1e-12
                        and next_distance_2 <= self.current_press_distance_2 + 1e-12
                    ):
                        self._append_load_sample(
                            stage_info,
                            command_distance=max(self.current_press_distance_1, self.current_press_distance_2),
                            step_index=step_index,
                            target_load_n=target_load_n,
                            state="hold_travel_limit_reached",
                        )
                        self.set_mutual_contact_state(False, "load_hold_not_maintainable_before_travel_limit")
                        return False, f"target load {target_load_n:.2f} N not maintainable before travel limit", False

                    next_distance_1, next_distance_2, contact_mode = self._balanced_distances(
                        next_distance_1,
                        next_distance_2,
                        direction,
                        travel_limit_m,
                    )
                    self.current_press_distance_1 = next_distance_1
                    self.current_press_distance_2 = next_distance_2
                    target = self.apply_directional_press_target(
                        next_distance_1,
                        next_distance_2,
                        direction,
                        f"load_step_{step_index}_hold_recover:{contact_mode}",
                    )
                    if target is None:
                        self.set_mutual_contact_state(False, "load_hold_target_unavailable")
                        return False, "directional target unavailable during hold", False
                    hold_reacquire_used = False
                hold_recover_settled = self._wait_for_directional_target_settle(
                    min(self.load_step_waypoint_timeout, max(0.0, end_t - self.now_s()))
                )
                if not hold_recover_settled:
                    self.set_mutual_contact_state(False, "load_hold_recovery_waypoint_timeout")
                    return False, "load-hold recovery command did not settle", False
                sample_state = "hold_recover"
            else:
                hold_reacquire_used = False
                target = self.compute_absolute_directional_press_target(
                    self.current_press_distance_1,
                    self.current_press_distance_2,
                    direction,
                )
                if target is None:
                    self.set_mutual_contact_state(False, "load_hold_target_unavailable")
                    return False, "directional target unavailable during hold", False
                hold_mode = f"load_step_{step_index}_hold"
                if not self._directional_target_matches_current(target, hold_mode):
                    target_positions_match = self._directional_target_matches_current(target)
                    if target_positions_match:
                        self.current_contact_mode = hold_mode
                    else:
                        self.publish_directional_press_target(target, hold_mode)
                        hold_settled = self._wait_for_directional_target_settle(
                            min(self.load_step_waypoint_timeout, max(0.0, end_t - self.now_s()))
                        )
                        if not hold_settled:
                            self.set_mutual_contact_state(False, "load_hold_waypoint_timeout")
                            return False, "load-hold command did not settle", False
                sample_state = "holding"

            # Velocity limits removed — quasistatic speed enforced by small step size
            violation = self._directional_abort_message()
            self._append_load_sample(
                stage_info,
                command_distance=max(self.current_press_distance_1, self.current_press_distance_2),
                step_index=step_index,
                target_load_n=target_load_n,
                state=sample_state,
            )
            if violation is not None:
                return False, violation, False
            if hold_contact_grace_samples_remaining > 0:
                hold_contact_grace_samples_remaining -= 1
            if hold_velocity_grace_samples_remaining > 0:
                hold_velocity_grace_samples_remaining -= 1

        return True, "", False

    def stage3_sync_move(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "precontact"
        stage_info: List[Dict[str, Any]] = []

        if self._stage3_bilateral_contact_confirmed():
            initial_direction = self._stage3_handoff_direction()
            if initial_direction is not None:
                if self._using_local_frame_load_logic():
                    self._set_contact_anchor_from_desired()
                    handoff_distance = max(
                        abs(self.press_step),
                        self.current_press_distance_1,
                        self.current_press_distance_2,
                    )
                    self.current_press_distance_1 = handoff_distance
                    self.current_press_distance_2 = handoff_distance
                else:
                    self.current_press_distance_1 = 0.0
                    self.current_press_distance_2 = 0.0
                self.current_mutual_press_dir = initial_direction
                self._update_load_metrics(initial_direction)
                snap = self.snapshot()
                snap["command_distance"] = 0.0
                stage_info.append(snap)
                self.log_sync_step(0.0)
                self.set_mutual_contact_state(True, "bilateral_contact_present_at_stage_start")
                return True, "bilateral contact established", {
                    "forward_phase_observed": False,
                    "started_in_mutual_contact": True,
                    "samples": stage_info,
                }

            self.get_logger().info(
                "Bilateral contact present at stage start but mutual press direction is invalid; continuing precontact search"
            )

        forward_phase_observed = self.wait_for_forward_phase(timeout=10.0)
        if not forward_phase_observed:
            self.get_logger().info(
                "Forward-phase sync not observed within timeout; continuing with cautious precontact approach"
            )

        self.spin_for(0.10)
        self.capture_precontact_baseline()

        # Use offset-based commands throughout to maintain consistent frame_id mode.
        # The controller must not see a switch between "absolute" and "offset" frame_id
        # mid-approach, as it can reset the trajectory reference and cause large jumps.
        offset1 = self.precontact_start_x_offset
        offset2 = self.precontact_start_x_offset

        # Publish an initial zero-offset so the controller is anchored in "offset" mode
        # before any axial motion begins.
        init_target = self.compute_press_target(offset1, offset2)
        if init_target is not None:
            self._apply_precontact_target(init_target, "none")
            self._wait_for_precontact_waypoint_completion(
                expect_robot1_motion=False,
                expect_robot2_motion=False,
                republish=lambda: self._apply_precontact_target(init_target, "none"),
            )
        
        self.get_logger().info(
            f"Precontact two-phase approach: FAST={self.precontact_FAST_STEP_M*1000:.1f}mm, "
            f"SLOW={self.precontact_SLOW_STEP_M*1000:.1f}mm, "
            f"threshold={self.precontact_QUASI_STATIC_THRESHOLD_M*1000:.0f}mm"
        )
        press_step_1_base = abs(self.precontact_SLOW_STEP_M)
        press_step_2_base = abs(math.copysign(self.precontact_SLOW_STEP_M, self.precontact_end_x_offset_2))
        fast_step_1 = abs(self.precontact_FAST_STEP_M)
        fast_step_2 = abs(math.copysign(self.precontact_FAST_STEP_M, self.precontact_end_x_offset_2))
        contact1 = False
        contact2 = False
        phase = "fast"  # Shared phase: both robots switch together for symmetric approach
        for _ in range(self.max_precontact_iterations):
            prev_offset1 = offset1
            prev_offset2 = offset2
            
            # Two-phase approach: fast until within threshold, then slow quasi-static
            # Both robots share the same phase to ensure symmetric, synchronous motion
            remaining1 = abs(self.precontact_end_x_offset - offset1)
            remaining2 = abs(self.precontact_end_x_offset_2 - offset2)
            
            # Switch to slow phase when BOTH robots are within threshold (symmetric transition)
            max_remaining = max(remaining1, remaining2)
            if max_remaining <= self.precontact_QUASI_STATIC_THRESHOLD_M and phase == "fast":
                self.get_logger().info(
                    f"Both robots switching to SLOW phase: "
                    f"robot1 offset={offset1:.4f} (remaining={remaining1:.4f}m), "
                    f"robot2 offset={offset2:.4f} (remaining={remaining2:.4f}m)"
                )
                phase = "slow"
            
            # Select step size based on shared phase
            press_step_1 = fast_step_1 if phase == "fast" else press_step_1_base
            press_step_2 = fast_step_2 if phase == "fast" else press_step_2_base
            
            if not contact1:
                offset1 = self._step_offset_toward(
                    offset1,
                    self.precontact_end_x_offset,
                    press_step_1,
                )
            elif contact1 and not contact2:
                offset1 = self._step_offset_toward(
                    offset1,
                    self.precontact_start_x_offset,
                    press_step_1,
                )
            if not contact2:
                offset2 = self._step_offset_toward(
                    offset2,
                    self.precontact_end_x_offset_2,
                    press_step_2,
                )
            elif contact2 and not contact1:
                offset2 = self._step_offset_toward(
                    offset2,
                    self.precontact_start_x_offset,
                    press_step_2,
                )

            # Compute the press target with the current offsets.
            # Both robots approach each other: robot1 in -X, robot2 in +X.
            target = self.compute_press_target(offset1, offset2)
            if target is None:
                return False, "desired poses unavailable", {
                    "forward_phase_observed": forward_phase_observed,
                    "started_in_mutual_contact": False,
                    "samples": stage_info,
                }

            contact_mode = (
                "robot1_only"
                if contact1 and not contact2
                else "robot2_only"
                if contact2 and not contact1
                else "none"
            )
            self._apply_precontact_target(target, contact_mode)
            start_desired_x_1 = self.des1.position.x if self.des1 is not None else None
            start_desired_x_2 = self.des2.position.x if self.des2 is not None else None
            if not self._wait_for_precontact_waypoint_completion(
                expect_robot1_motion=abs(offset1 - prev_offset1) > 1e-12,
                expect_robot2_motion=abs(offset2 - prev_offset2) > 1e-12,
                start_desired_x_1=start_desired_x_1,
                start_desired_x_2=start_desired_x_2,
                expected_delta_x1=offset1 - prev_offset1,
                expected_delta_x2=offset2 - prev_offset2,
                republish=lambda target=target, contact_mode=contact_mode: self._apply_precontact_target(target, contact_mode),
            ):
                return False, "precontact command did not settle", {
                    "forward_phase_observed": forward_phase_observed,
                    "started_in_mutual_contact": False,
                    "samples": stage_info,
                }

            self.spin_for(0.05)

            contact1 = self._stage3_contact_detected(1)
            contact2 = self._stage3_contact_detected(2)

            self.current_press_distance_1 = abs(offset1)
            self.current_press_distance_2 = abs(offset2)

            snap = self.snapshot()
            snap["command_distance"] = max(abs(offset1), abs(offset2))
            stage_info.append(snap)
            self.log_sync_step(max(abs(offset1), abs(offset2)))

            if contact1 and contact2:
                self.set_mutual_contact_state(True, "bilateral_contact_established")
                self._set_contact_anchor_from_desired()
                if self._using_local_frame_load_logic():
                    handoff_distance = max(
                        self.press_step,
                        min(self.current_press_distance_1, self.current_press_distance_2),
                    )
                    self.current_press_distance_1 = handoff_distance
                    self.current_press_distance_2 = handoff_distance
                direction = self._stage3_handoff_direction()
                if direction is not None:
                    self.current_mutual_press_dir = direction
                    self._update_load_metrics(direction)
                return True, "bilateral contact established", {
                    "forward_phase_observed": forward_phase_observed,
                    "started_in_mutual_contact": False,
                    "samples": stage_info,
                }

            if not self.separation_safe():
                self.get_logger().error(
                    f"EE separation {self.ee_separation():.4f} m below safety limit "
                    f"{self.min_ee_separation:.4f} m — aborting"
                )
                return False, "inter-robot separation below safety limit", {
                    "forward_phase_observed": forward_phase_observed,
                    "started_in_mutual_contact": False,
                    "samples": stage_info,
                }
            if not self.x_gap_safe():
                self.get_logger().error(
                    f"EE x gap {self.ee_x_gap():.4f} m above safety limit "
                    f"{self.max_ee_x_gap:.4f} m — aborting"
                )
                return False, "inter-robot x gap above safety limit", {
                    "forward_phase_observed": forward_phase_observed,
                    "started_in_mutual_contact": False,
                    "samples": stage_info,
                }
            if not self.pose_safe():
                return False, "EE pose below x floor safety limit", {
                    "forward_phase_observed": forward_phase_observed,
                    "started_in_mutual_contact": False,
                    "samples": stage_info,
                }
            # Velocity check removed for precontact — quasistatic speed is enforced
            # by small step size (3mm fast, 1mm slow), not by monitoring joint velocities.
            # This is especially important in Gazebo where controller oscillations
            # can trigger false velocity spikes.
            # if not self.check_limits(self.js1, self.precontact_velocity_limit) or not self.check_limits(self.js2, self.precontact_velocity_limit):
            #     return False, "velocity spike during precontact approach", {
            #         "forward_phase_observed": forward_phase_observed,
            #         "started_in_mutual_contact": False,
            #         "samples": stage_info,
            #     }

        return False, "bilateral contact not established", {
            "forward_phase_observed": forward_phase_observed,
            "started_in_mutual_contact": False,
            "samples": stage_info,
        }

    def stage5_adaptive_press(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Stage 5: Execute quasi-static load-step compression and hold."""
        self.current_phase = "adaptive_load_step"
        stage_info: List[Dict[str, Any]] = []

        if not self._load_targets_ready():
            self.set_mutual_contact_state(False, "load_targets_not_configured")
            return False, "load targets not configured", {"samples": stage_info}

        direction = self._resolve_load_step_direction(self.load_targets_n[0])
        if direction is None:
            self.set_mutual_contact_state(False, "load_step_direction_unavailable")
            return False, "mutual contact direction unavailable", {"samples": stage_info}

        self.current_mutual_press_dir = direction
        self.directional_press_enabled = True
        self._update_load_metrics(direction)

        if self._using_local_frame_load_logic():
            if not self._reset_local_frame_load_step_reference():
                self.set_mutual_contact_state(False, "load_step_anchor_unavailable")
                self.directional_press_enabled = False
                return False, "unable to establish local-frame load-step anchor", {"samples": stage_info}
        elif not self.ensure_mutual_contact_anchor():
            self.set_mutual_contact_state(False, "load_step_anchor_unavailable")
            self.directional_press_enabled = False
            return False, "unable to capture mutual-contact anchor", {"samples": stage_info}

        for step_index, target_load_n in enumerate(self.load_targets_n, start=1):
            stage_entry_distance = max(self.current_press_distance_1, self.current_press_distance_2)
            travel_limit_m = stage_entry_distance + self.max_additional_load_travel_m

            if step_index == 1:
                ok, message = self._relieve_preload_before_load_steps(step_index, target_load_n, stage_info)
                if not ok:
                    self.directional_press_enabled = False
                    return False, message, {
                        "samples": stage_info,
                        "failed_step_index": step_index,
                        "failed_target_load_n": target_load_n,
                    }

            seek_ok, seek_message, seek_unloaded = self._seek_load_target(
                step_index,
                target_load_n,
                stage_info,
                travel_limit_m,
            )
            if not seek_ok:
                self.directional_press_enabled = False
                return False, seek_message, {
                    "samples": stage_info,
                    "failed_step_index": step_index,
                    "failed_target_load_n": target_load_n,
                }
            if seek_unloaded:
                self.directional_press_enabled = False
                return True, seek_message, {
                    "samples": stage_info,
                    "completed_until_step_index": step_index,
                    "upper_force_limit_n": self.upper_force_limit_n,
                }

            hold_ok, hold_message, hold_unloaded = self._hold_load_target(
                step_index,
                target_load_n,
                stage_info,
                travel_limit_m,
            )
            if not hold_ok:
                self.directional_press_enabled = False
                return False, hold_message, {
                    "samples": stage_info,
                    "failed_step_index": step_index,
                    "failed_target_load_n": target_load_n,
                }
            if hold_unloaded:
                self.directional_press_enabled = False
                return True, hold_message, {
                    "samples": stage_info,
                    "completed_until_step_index": step_index,
                    "upper_force_limit_n": self.upper_force_limit_n,
                }

        self.current_load_step_state = "complete"
        self.set_mutual_contact_state(False, "load_schedule_complete")
        self.directional_press_enabled = False
        return True, "all load steps completed", {
            "samples": stage_info,
            "load_targets_n": list(self.load_targets_n),
            "upper_force_limit_n": self.upper_force_limit_n,
        }

    def _prepare_case(
        self,
        *,
        case_name: str,
        peak_load_n: float,
        load_targets_n: Sequence[float],
        k_lateral_npm: Optional[float],
    ) -> None:
        self.set_mutual_contact_state(False, f"{case_name}_prepare")
        self._publish_zero_offsets()
        self._wait_for_arms_idle(timeout_s=12.0)
        self._reset_case_state()
        self.current_case_name = case_name
        self.current_case_peak_load_n = peak_load_n
        self.current_case_k_lateral_npm = float("nan") if k_lateral_npm is None else float(k_lateral_npm)
        self.load_targets_n = tuple(float(value) for value in load_targets_n)
        self.upper_force_limit_n = float(peak_load_n)
        if not self.load_target_tolerance_explicit:
            self.load_target_tolerance_n = self._default_load_target_tolerance_n()
        self.force_balance_tolerance = self._default_force_balance_tolerance_n()
        if self.ignore_local_frame_ee_safety and not self.precontact_contact_threshold_explicit:
            self.precontact_contact_threshold_n = self._default_local_frame_precontact_threshold_n()
        if k_lateral_npm is not None and self._positive_finite(k_lateral_npm):
            if self.defer_k_lateral_until_after_precontact:
                # Defer the K_lat command until after stage 3 so high lateral
                # stiffness does not amplify pose noise during the precontact
                # approach.
                self._pending_case_k_lateral_npm = float(k_lateral_npm)
            else:
                self.set_k_lateral(float(k_lateral_npm))
                self._pending_case_k_lateral_npm = None
        else:
            self._pending_case_k_lateral_npm = None

        # Apply axial stiffness if explicitly provided
        if self.k_axial_explicit and self._positive_finite(self.k_axial_npm):
            self.set_k_axial(float(self.k_axial_npm))

    def _run_case(
        self,
        *,
        case_name: str,
        peak_load_n: float,
        load_targets_n: Sequence[float],
        k_lateral_npm: Optional[float],
    ) -> Dict[str, Any]:
        try:
            self._prepare_case(
                case_name=case_name,
                peak_load_n=peak_load_n,
                load_targets_n=load_targets_n,
                k_lateral_npm=k_lateral_npm,
            )
        except ValueError as error:
            return {
                "case": case_name,
                "pass": False,
                "message": str(error),
                "k_lateral_npm": k_lateral_npm,
                "upper_force_limit_n": peak_load_n,
                "load_targets_n": list(load_targets_n),
                "stages": [],
            }

        stages = [
            self.stage1_liveness,
            self.stage2_idle,
            self.stage3_sync_move,
            self.stage4_hold,
            self.stage5_adaptive_press,
        ]

        stage_results: List[Dict[str, Any]] = []
        for idx, stage_fn in enumerate(stages, start=1):
            ok, msg, info = stage_fn()
            record = {
                "stage": idx,
                "pass": ok,
                "message": msg,
                "info": info,
            }
            stage_results.append(record)
            self.get_logger().info(f"Case {case_name} stage {idx}: {msg} -> {ok}")
            if not ok:
                self.safe_abort()
                return {
                    "case": case_name,
                    "pass": False,
                    "message": msg,
                    "k_lateral_npm": k_lateral_npm,
                    "upper_force_limit_n": peak_load_n,
                    "load_targets_n": list(load_targets_n),
                    "stages": stage_results,
                }
            if idx == 3 and self._pending_case_k_lateral_npm is not None:
                # Apply deferred K_lat now that bilateral contact is established
                # and before stage 4 directional handoff starts.
                try:
                    self.set_k_lateral(float(self._pending_case_k_lateral_npm))
                except ValueError as err:
                    self.get_logger().warn(
                        f"Case {case_name} deferred K_lat apply failed: {err}"
                    )
                finally:
                    self._pending_case_k_lateral_npm = None

        self.current_phase = "done"
        self.directional_press_enabled = False
        self.set_mutual_contact_state(False, f"{case_name}_complete")
        self._publish_zero_offsets()
        return {
            "case": case_name,
            "pass": True,
            "message": "all stages passed",
            "k_lateral_npm": k_lateral_npm,
            "upper_force_limit_n": peak_load_n,
            "load_targets_n": list(load_targets_n),
            "stages": stage_results,
        }

    def run(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        try:
            ok, msg, info = self.stage0_calibration()
            stage0_record = {
                "stage": 0,
                "pass": ok,
                "message": msg,
                "info": info,
            }
            results.append(stage0_record)
            self.get_logger().info(f"Stage 0: {msg} -> {ok}")
            if not ok:
                self.write_results(results)
                return results

            if not self.enable_ablation:
                case_record = self._run_case(
                    case_name=self.current_case_name,
                    peak_load_n=self.upper_force_limit_n,
                    load_targets_n=self.load_targets_n,
                    k_lateral_npm=self.k_lateral_npm if self._positive_finite(self.k_lateral_npm) else None,
                )
                results.extend(case_record["stages"])
                self.write_results(results)
                return results

            k_lateral_cases = self.derived_k_lateral_case_values_npm or self.k_lateral_case_values_npm
            peak_load_cases = self.derived_peak_load_case_values_n or self.peak_load_case_values_n
            case_index = 0
            for k_lateral_npm in k_lateral_cases:
                for peak_load_n in peak_load_cases:
                    case_index += 1
                    combo_tag = (
                        f"case_{case_index:02d}_k{str(round(k_lateral_npm, 3)).replace('.', 'p')}_"
                        f"p{str(round(peak_load_n, 3)).replace('.', 'p')}"
                    )
                    case_load_targets_n = self._derive_case_load_targets(peak_load_n)
                    trial_records: List[Dict[str, Any]] = []
                    for trial_idx in range(1, self.n_repeats + 1):
                        if self.n_repeats > 1:
                            case_name = f"{combo_tag}_trial_{trial_idx:02d}"
                        else:
                            case_name = combo_tag
                        trial_record = self._run_case(
                            case_name=case_name,
                            peak_load_n=peak_load_n,
                            load_targets_n=case_load_targets_n,
                            k_lateral_npm=k_lateral_npm,
                        )
                        trial_record["trial_index"] = trial_idx
                        trial_record["combo"] = combo_tag
                        trial_records.append(trial_record)
                    combo_pass_count = sum(1 for t in trial_records if t.get("pass"))
                    combo_record: Dict[str, Any] = {
                        "combo": combo_tag,
                        "case": combo_tag,
                        "k_lateral_npm": k_lateral_npm,
                        "peak_load_n": peak_load_n,
                        "pass": combo_pass_count == self.n_repeats,
                        "pass_count": combo_pass_count,
                        "trial_count": self.n_repeats,
                        "trials": trial_records,
                        "message": (
                            f"{combo_pass_count}/{self.n_repeats} trials passed"
                        ),
                    }
                    results.append(combo_record)

            self.write_results(results)
            return results
        finally:
            self.set_mutual_contact_state(False, "run_finally")


def _parse_load_targets(value: str) -> List[float]:
    raw_tokens = [token.strip() for token in value.replace(";", ",").split(",")]
    targets = [float(token) for token in raw_tokens if token]
    if not targets:
        raise argparse.ArgumentTypeError("at least one load target is required")
    previous = 0.0
    for target in targets:
        if target <= 0.0:
            raise argparse.ArgumentTypeError("load targets must be > 0")
        if target <= previous:
            raise argparse.ArgumentTypeError("load targets must be strictly increasing")
        previous = target
    return targets


def _parse_positive_values(value: str) -> List[float]:
    raw_tokens = [token.strip() for token in value.replace(";", ",").split(",")]
    values = [float(token) for token in raw_tokens if token]
    if not values:
        raise argparse.ArgumentTypeError("at least one positive value is required")
    for item in values:
        if item <= 0.0:
            raise argparse.ArgumentTypeError("all values must be > 0")
    return values


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the contact-gated bilateral load-step harness",
        add_help=add_help,
    )
    parser.add_argument(
        "--push-axis-height-z",
        type=float,
        default=None,
        help=(
            "Absolute push-axis height in metres. When provided, precontact and "
            "load-step commands hold the bilateral push at this z height."
        ),
    )
    parser.add_argument(
        "--load-targets-n",
        type=_parse_load_targets,
        default=None,
        help=(
            "Comma-separated strictly increasing bilateral load targets in newtons, "
            f"for example: {','.join(str(x) for x in DEFAULT_LOAD_TARGETS_N)}. "
            "When omitted, stage 0 may derive the range from calibration."
        ),
    )
    parser.add_argument(
        "--hold-duration-s",
        type=float,
        default=10.0,
        help="Seconds to maintain each achieved load target before advancing.",
    )
    parser.add_argument(
        "--upper-force-limit-n",
        type=float,
        default=None,
        help=(
            "Upper per-arm projected load limit in newtons. When reached during stage 5, "
            "the harness stops increasing compression and starts quasi-static unloading. "
            "Defaults to the largest configured load target."
        ),
    )
    parser.add_argument(
        "--load-target-tolerance-n",
        type=float,
        default=None,
        help=(
            "Allowable load undershoot when deciding a target has been reached/maintained. "
            "When omitted, the harness scales the tolerance from the active load schedule."
        ),
    )
    parser.add_argument(
        "--load-seek-step-m",
        type=float,
        default=0.0005,
        help="Quasi-static forward distance increment used while seeking the next load target.",
    )
    parser.add_argument(
        "--load-balance-step-m",
        type=float,
        default=0.00025,
        help="Extra per-arm distance bias used to rebalance asymmetric bilateral load.",
    )
    parser.add_argument(
        "--max-additional-load-travel-m",
        type=float,
        default=0.330,
        help="Maximum additional travel allowed after first bilateral contact while seeking later load steps.",
    )
    parser.add_argument(
        "--min-stage-advance-m",
        type=float,
        default=0.0,
        help=(
            "Minimum forward press distance each load step must advance from its entry pose before "
            "the force criterion alone may declare target_reached. Use a non-zero value (e.g. 0.001) "
            "to force visible quasi-static compression at distinct (pd1, shared_load) plateaus across "
            "a K_lat ablation, even when the post-handoff residual force already meets the early targets."
        ),
    )
    parser.add_argument(
        "--stage4-handoff-target-n",
        type=float,
        default=None,
        help=(
            "Optional fixed shared-load target (Newtons) for the stage 4 directional-hold handoff. "
            "When set, decouples handoff from the load schedule so stage 4 completes as soon as "
            "bilateral contact is established (e.g. 0.005), avoiding lateral instability at high K_lat."
        ),
    )
    parser.add_argument(
        "--defer-k-lateral-until-after-precontact",
        action="store_true",
        help=(
            "Defer applying the per-case K_lat command until after stage 3 (bilateral contact "
            "established). Use for high K_lat values where the lateral stiffness term amplifies "
            "pose noise during the precontact approach and causes stage 3 velocity spikes."
        ),
    )
    parser.add_argument(
        "--external-camera-trigger-file",
        type=str,
        default=None,
        help=(
            "Path to a host-side text-trigger file watched by "
            "tools/realsense_text_trigger_capture.py. When set, the harness writes "
            "'START name=<reason>_window_<idx>' on each mutual-contact-window open and "
            "'STOP' on each close, so a high-FPS host-side recorder can capture per-window "
            "video synchronized with the harness state. Leave unset to disable."
        ),
    )
    parser.add_argument(
        "--ignore-local-frame-ee-safety",
        action="store_true",
        help=(
            "Ignore EE separation/x-gap safety checks that compare per-robot local end-effector frames. "
            "Use this for Gazebo validation only when those local frames do not share a common world origin."
        ),
    )
    parser.add_argument(
        "--precontact-waypoint-timeout-s",
        type=float,
        default=5.0,
        help="Seconds to wait for each quasi-static precontact waypoint to settle before sending the next one.",
    )
    parser.add_argument(
        "--precontact-contact-threshold-n",
        type=float,
        default=None,
        help=(
            "Bilateral force-delta threshold used only to declare first contact in stage 3. "
            "When omitted, Gazebo runs using --ignore-local-frame-ee-safety default to 0.05 N."
        ),
    )
    parser.add_argument(
        "--precontact-velocity-limit",
        type=float,
        default=None,
        help="Joint-velocity limit used only during the quasi-static stage-3 approach.",
    )
    parser.add_argument(
        "--load-step-waypoint-timeout-s",
        type=float,
        default=None,
        help=(
            "Seconds to wait for each stage-5 directional load-step command to settle. "
            "When omitted, the harness uses the larger of the stage-5 hold duration and the directional waypoint timeout."
        ),
    )
    parser.add_argument(
        "--calibration-json",
        default=None,
        help=(
            "Calibration JSON file or directory containing calib_result_*.json. "
            "When provided, stage 0 derives the peak-load and K_lat ablation ranges from it."
        ),
    )
    parser.add_argument(
        "--spring-specimen",
        choices=sorted(SPRING_SPECIMENS.keys()),
        default=None,
        help=(
            "Measured spring specimen preset used to seed stage-0 K_lat derivation. "
            "s1 = 49.4 N/m at 128 mm free length; s2 = 87.1 N/m at 170 mm free length."
        ),
    )
    parser.add_argument(
        "--spring-stiffness-npm",
        type=float,
        default=None,
        help=(
            "Measured spring stiffness in N/m from UTM data. "
            "When supplied with --spring-specimen, this value overrides the preset stiffness."
        ),
    )
    parser.add_argument(
        "--k-lateral-npm",
        type=float,
        default=None,
        help="Optional per-arm lateral stiffness command for a single run.",
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="Run the contact-gated harness over a stage-0-derived K_lat x peak-load ablation grid.",
    )
    parser.add_argument(
        "--k-lateral-cases-npm",
        type=_parse_positive_values,
        default=None,
        help="Optional comma-separated K_lat case overrides for ablation mode.",
    )
    parser.add_argument(
        "--peak-load-cases-n",
        type=_parse_positive_values,
        default=None,
        help="Optional comma-separated peak-load case overrides for ablation mode.",
    )
    parser.add_argument(
        "--repeats-n",
        type=int,
        default=1,
        help=(
            "Number of times each (K_lat, peak-load) combo is repeated. "
            "Trials are labelled _trial_01, _trial_02, … when repeats > 1. "
            "A failed trial does not abort the run; the combo pass/fail is "
            "reported as pass_count/trial_count."
        ),
    )
    parser.add_argument(
        "--use-sim-time",
        action="store_true",
        default=False,
        help="Use simulation time (for Gazebo). Required when running against Gazebo simulation.",
    )
    parser.add_argument(
        "--plot-save",
        type=str,
        default=None,
        help=(
            "Directory to save publication-quality plots. "
            "When omitted, plots are saved in the run log directory alongside the CSV/JSON results."
        ),
    )
    parser.add_argument(
        "--klat",
        type=float,
        default=None,
        dest="k_lateral_npm",
        help="Alias for --k-lateral-npm. Runtime lateral stiffness (N/m) for both robots.",
    )
    parser.add_argument(
        "--kax",
        type=float,
        default=None,
        dest="k_axial_npm",
        help="Runtime axial stiffness (N/m) for both robots. Overrides the controller YAML stiffness_profile_x.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser(add_help=True)
    args, _unknown = parser.parse_known_args(argv)
    return args


def main() -> None:
    args = parse_args()
    # Initialize ROS with or without sim time
    if args.use_sim_time:
        rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=true'])
    else:
        rclpy.init()
    node = ContactGatedLoadStepHarness(
        push_axis_height_z=args.push_axis_height_z,
        load_targets_n=args.load_targets_n,
        hold_duration_s=args.hold_duration_s,
        upper_force_limit_n=args.upper_force_limit_n,
        load_target_tolerance_n=args.load_target_tolerance_n,
        load_seek_step_m=args.load_seek_step_m,
        load_balance_step_m=args.load_balance_step_m,
        max_additional_load_travel_m=args.max_additional_load_travel_m,
        min_stage_advance_m=args.min_stage_advance_m,
        stage4_handoff_target_n=args.stage4_handoff_target_n,
        defer_k_lateral_until_after_precontact=args.defer_k_lateral_until_after_precontact,
        external_camera_trigger_file=args.external_camera_trigger_file,
        ignore_local_frame_ee_safety=args.ignore_local_frame_ee_safety,
        precontact_waypoint_timeout_s=args.precontact_waypoint_timeout_s,
        precontact_contact_threshold_n=args.precontact_contact_threshold_n,
        precontact_velocity_limit=args.precontact_velocity_limit,
        load_step_waypoint_timeout_s=args.load_step_waypoint_timeout_s,
        calibration_json_path=args.calibration_json,
        enable_ablation=args.ablate,
        n_repeats=args.repeats_n,
        k_lateral_npm=args.k_lateral_npm,
        k_lateral_case_values_npm=args.k_lateral_cases_npm,
        peak_load_case_values_n=args.peak_load_cases_n,
        spring_specimen=args.spring_specimen,
        spring_stiffness_npm=args.spring_stiffness_npm,
        use_sim_time=args.use_sim_time,
        plot_save_dir=args.plot_save,
        k_axial_npm=args.k_axial_npm,
    )
    try:
        out = node.run()
        print(json.dumps(out, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
