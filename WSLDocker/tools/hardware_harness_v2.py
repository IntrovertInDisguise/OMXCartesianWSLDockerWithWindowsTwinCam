#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Pose, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


class HardwareHarnessAdaptive(Node):
    def __init__(self, use_sim_time: bool = False) -> None:
        super().__init__("hw_harness_adaptive", parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time)
        ])

        self.pub1 = self.create_publisher(
            PoseStamped, "/robot1/robot1_variable_stiffness/waypoint_command", 10
        )
        self.pub2 = self.create_publisher(
            PoseStamped, "/robot2/robot2_variable_stiffness/waypoint_command", 10
        )

        qos_fast = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(JointState, "/robot1/joint_states", self.cb_js1, qos_fast)
        self.create_subscription(JointState, "/robot2/joint_states", self.cb_js2, qos_fast)
        self.create_subscription(
            Point,
            "/robot1/robot1_variable_stiffness/end_effector_position",
            self.cb_ee1,
            qos_fast,
        )
        self.create_subscription(
            Point,
            "/robot2/robot2_variable_stiffness/end_effector_position",
            self.cb_ee2,
            qos_fast,
        )
        self.create_subscription(
            Pose,
            "/robot1/robot1_variable_stiffness/cartesian_pose_desired",
            self.cb_des1,
            qos_fast,
        )
        self.create_subscription(
            Pose,
            "/robot2/robot2_variable_stiffness/cartesian_pose_desired",
            self.cb_des2,
            qos_fast,
        )
        self.create_subscription(
            WrenchStamped,
            "/robot1/robot1_variable_stiffness/contact_wrench",
            self.cb_contact1,
            qos_fast,
        )
        self.create_subscription(
            WrenchStamped,
            "/robot2/robot2_variable_stiffness/contact_wrench",
            self.cb_contact2,
            qos_fast,
        )
        self.create_subscription(
            Bool,
            "/robot1/robot1_variable_stiffness/contact_valid",
            self.cb_contact_valid1,
            qos_fast,
        )
        self.create_subscription(
            Bool,
            "/robot2/robot2_variable_stiffness/contact_valid",
            self.cb_contact_valid2,
            qos_fast,
        )
        self.create_subscription(
            Bool,
            "/robot1/robot1_variable_stiffness/waypoint_active",
            self.cb_wp1,
            qos_fast,
        )
        self.create_subscription(
            Bool,
            "/robot2/robot2_variable_stiffness/waypoint_active",
            self.cb_wp2,
            qos_fast,
        )

        self.js1: Optional[JointState] = None
        self.js2: Optional[JointState] = None
        self.ee1: Optional[Point] = None
        self.ee2: Optional[Point] = None
        self.des1: Optional[Pose] = None
        self.des2: Optional[Pose] = None
        self.contact_fx_mag_1 = float("nan")
        self.contact_fx_mag_2 = float("nan")
        self.contact_valid_1 = False
        self.contact_valid_2 = False
        self.wp1_active: Optional[bool] = None
        self.wp2_active: Optional[bool] = None

        self.current_phase = "init"
        self.current_offset_x1 = float("nan")
        self.current_offset_x2 = float("nan")
        self.current_press_offset_x1 = 0.0
        self.current_press_offset_x2 = 0.0
        self.current_contact_mode = "none"
        self.precontact_baseline_fx_1 = float("nan")
        self.precontact_baseline_fx_2 = float("nan")

        # Per-robot alignment trims (meters). These are intended to be
        # populated by upstream alignment helpers (e.g. ArUco-based trim
        # publishers). Default to zero so harnesses without camera-based
        # alignment behave the same.
        self.alignment_robot1_z_trim = 0.0
        self.alignment_robot2_z_trim = 0.0
        self.alignment_robot1_y_trim = 0.0
        self.alignment_robot2_y_trim = 0.0

        self.idle_vel_limit = 1.0
        self.move_vel_limit = 1.5
        self.hold_vel_limit = 0.8
        self.command_repeats = 2
        self.command_dt = 0.05
        self.command_sample_delay = 0.45

        # Load optional harness overrides from YAML (see variable_stiffness_controller.yaml)
        try:
            self._load_harness_config()
        except Exception:
            # Don't fail harness startup for missing/invalid YAML
            pass

        self.precontact_start_x_offset = 0.05
        self.precontact_end_x_offset = -0.0100
        self.hold_x_offset = -0.0100
        self.press_end_x_offset = -0.0600
        self.press_step = 0.0015

        self.contact_force_enter = 0.60
        self.contact_force_delta_enter = 0.10
        self.contact_force_threshold_cap = 0.60
        # Optional fixed contact threshold override (None = disabled)
        self.contact_threshold_override: Optional[float] = None
        # Allow disabling use of precontact baseline when computing thresholds
        self.use_precontact_baseline: bool = True
        self.force_balance_tolerance = 0.35
        self.max_side_extra_press = 0.0060
        self.side_press_step = 0.0010
        self.force_diff_abort = 4.0
        self.max_force_mag_abort = 8.0
        self.max_precontact_iterations = 12

        self.log_root = os.environ.get("OMX_LOG_DIR", "/tmp/variable_stiffness_logs")
        self.run_ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.log_root, exist_ok=True)

        self.csv_path = os.path.join(self.log_root, f"hardware_harness_adaptive_{self.run_ts}.csv")
        self.sync_csv_path = os.path.join(self.log_root, f"hardware_harness_sync_steps_{self.run_ts}.csv")

        self.log_columns = [
            "timestamp",
            "phase",
            "offset_x1",
            "offset_x2",
            "press_offset_x1",
            "press_offset_x2",
            "contact_mode",
            "robot1_max_vel",
            "robot2_max_vel",
            "ee_x_1",
            "ee_x_2",
            "ee_x_diff",
            "desired_x_1",
            "desired_x_2",
            "desired_x_diff",
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
            "command_x_offset",
            "offset_x1",
            "offset_x2",
            "press_offset_x1",
            "press_offset_x2",
            "contact_mode",
            "robot1_max_vel",
            "robot2_max_vel",
            "ee_x_1",
            "ee_x_2",
            "ee_x_diff",
            "desired_x_1",
            "desired_x_2",
            "desired_x_diff",
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

        self.write_csv_header(self.csv_path, self.log_columns)
        self.write_csv_header(self.sync_csv_path, self.sync_columns)
        self.create_timer(0.05, self.log_snapshot_row)

    def _load_harness_config(self) -> None:
        """Load harness-related overrides from a YAML file.

        The YAML path may be set via `OMX_HARNESS_CONFIG` env var. By default
        this reads the controller YAML so operators can keep a single source
        of truth (`ws/src/omx_variable_stiffness_controller/config/variable_stiffness_controller.yaml`).
        """
        yaml_path = os.environ.get(
            "OMX_HARNESS_CONFIG",
            "/workspaces/omx_ros2/ws/src/omx_variable_stiffness_controller/config/variable_stiffness_controller.yaml",
        )
        if not os.path.exists(yaml_path):
            return
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().warning(f"Failed to load harness YAML {yaml_path}: {e}")
            return

        params: Dict[str, Any] = {}
        # Look for controller node parameters first
        node_key = "/omx/variable_stiffness_controller"
        if isinstance(data, dict):
            if node_key in data and isinstance(data[node_key], dict) and "ros__parameters" in data[node_key]:
                params = data[node_key]["ros__parameters"]
            else:
                # Merge any ros__parameters blocks found anywhere in the file
                for v in data.values():
                    if isinstance(v, dict) and "ros__parameters" in v and isinstance(v["ros__parameters"], dict):
                        params.update(v["ros__parameters"])
                # Also allow top-level harness: { ... } map
                if "harness" in data and isinstance(data["harness"], dict):
                    params.update(data["harness"])

        # Map recognized harness keys to internal attributes
        if "harness_idle_vel_limit" in params:
            self.idle_vel_limit = float(params["harness_idle_vel_limit"])
        if "harness_move_vel_limit" in params:
            self.move_vel_limit = float(params["harness_move_vel_limit"])
        if "harness_hold_vel_limit" in params:
            self.hold_vel_limit = float(params["harness_hold_vel_limit"])
        if "harness_command_repeats" in params:
            self.command_repeats = int(params["harness_command_repeats"])
        if "harness_command_dt" in params:
            self.command_dt = float(params["harness_command_dt"])
        if "harness_press_step" in params:
            self.press_step = float(params["harness_press_step"])
        # Contact detection tuning
        if "harness_contact_force_enter" in params:
            self.contact_force_enter = float(params["harness_contact_force_enter"])
        if "harness_contact_force_delta_enter" in params:
            self.contact_force_delta_enter = float(params["harness_contact_force_delta_enter"])
        if "harness_contact_force_threshold_cap" in params:
            self.contact_force_threshold_cap = float(params["harness_contact_force_threshold_cap"])
        if "harness_contact_threshold" in params:
            # Fixed absolute threshold (overrides baseline-based computation)
            try:
                self.contact_threshold_override = float(params["harness_contact_threshold"])
            except Exception:
                pass
        if "harness_use_precontact_baseline" in params:
            try:
                self.use_precontact_baseline = bool(params["harness_use_precontact_baseline"])
            except Exception:
                pass

        self.get_logger().info(
            f"Loaded harness YAML {yaml_path}: repeats={self.command_repeats}, dt={self.command_dt}, move_vel_limit={self.move_vel_limit}, contact_enter={self.contact_force_enter}, contact_delta={self.contact_force_delta_enter}, contact_threshold_override={self.contact_threshold_override}, use_precontact_baseline={self.use_precontact_baseline}"
        )

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def cb_js1(self, msg: JointState) -> None:
        self.js1 = msg

    def cb_js2(self, msg: JointState) -> None:
        self.js2 = msg

    def cb_ee1(self, msg: Point) -> None:
        self.ee1 = msg

    def cb_ee2(self, msg: Point) -> None:
        self.ee2 = msg

    def cb_des1(self, msg: Pose) -> None:
        self.des1 = msg

    def cb_des2(self, msg: Pose) -> None:
        self.des2 = msg

    def cb_contact1(self, msg: WrenchStamped) -> None:
        self.contact_fx_mag_1 = abs(msg.wrench.force.x)

    def cb_contact2(self, msg: WrenchStamped) -> None:
        self.contact_fx_mag_2 = abs(msg.wrench.force.x)

    def cb_contact_valid1(self, msg: Bool) -> None:
        self.contact_valid_1 = bool(msg.data)

    def cb_contact_valid2(self, msg: Bool) -> None:
        self.contact_valid_2 = bool(msg.data)

    def cb_wp1(self, msg: Bool) -> None:
        self.wp1_active = bool(msg.data)

    def cb_wp2(self, msg: Bool) -> None:
        self.wp2_active = bool(msg.data)

    def spin_for(self, duration: float, step: float = 0.05) -> None:
        end_t = self.now_s() + duration
        while self.now_s() < end_t:
            rclpy.spin_once(self, timeout_sec=step)

    def finite(self, x: float) -> bool:
        return not math.isnan(x) and not math.isinf(x)

    def clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def make_offset_pose(self, x: float, y: float, z: float) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "offset"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        return msg

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
        repeat_count = self.command_repeats if repeats is None else repeats
        step_dt = self.command_dt if dt is None else dt

        self.current_offset_x1 = x1
        self.current_offset_x2 = x2
        self.current_press_offset_x1 = x1
        self.current_press_offset_x2 = x2

        # Apply any configured per-robot alignment trims to the y/z
        # components so downstream harnesses and controllers see a
        # consistent push-axis height and lateral alignment.
        tz1 = z1 + getattr(self, "alignment_robot1_z_trim", 0.0)
        tz2 = z2 + getattr(self, "alignment_robot2_z_trim", 0.0)
        ty1 = y1 + getattr(self, "alignment_robot1_y_trim", 0.0)
        ty2 = y2 + getattr(self, "alignment_robot2_y_trim", 0.0)

        msg1 = self.make_offset_pose(x1, ty1, tz1)
        msg2 = self.make_offset_pose(x2, ty2, tz2)

        for _ in range(repeat_count):
            stamp = self.get_clock().now().to_msg()
            msg1.header.stamp = stamp
            msg2.header.stamp = stamp
            self.pub1.publish(msg1)
            self.pub2.publish(msg2)
            rclpy.spin_once(self, timeout_sec=step_dt)

    def wait_for_subscribers(self, timeout: float = 3.0) -> bool:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            if self.pub1.get_subscription_count() > 0 and self.pub2.get_subscription_count() > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def wait_joint_states(self, timeout: float = 5.0) -> bool:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            if self.js1 is not None and self.js2 is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def wait_for_poses(self, timeout: float = 3.0) -> bool:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            if self.ee1 is not None and self.ee2 is not None and self.des1 is not None and self.des2 is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def max_abs_velocity(self, js: Optional[JointState]) -> Optional[float]:
        if js is None or js.velocity is None or len(js.velocity) == 0:
            return None
        return max(abs(v) for v in js.velocity)

    def check_limits(self, js: Optional[JointState], limit: float) -> bool:
        vmax = self.max_abs_velocity(js)
        return vmax is not None and vmax < limit

    def capture_precontact_baseline(self) -> None:
        self.precontact_baseline_fx_1 = self.contact_fx_mag_1 if self.finite(self.contact_fx_mag_1) else 0.0
        self.precontact_baseline_fx_2 = self.contact_fx_mag_2 if self.finite(self.contact_fx_mag_2) else 0.0

    def contact_threshold(self, robot_id: int) -> float:
        # If a fixed override is configured, use it directly.
        if self.contact_threshold_override is not None:
            return float(self.contact_threshold_override)

        baseline = self.precontact_baseline_fx_1 if robot_id == 1 else self.precontact_baseline_fx_2
        if not self.finite(baseline):
            baseline = 0.0
        if not self.use_precontact_baseline:
            baseline = 0.0

        return min(
            self.contact_force_threshold_cap,
            max(self.contact_force_enter, baseline + self.contact_force_delta_enter),
        )

    def contact_detected(self, robot_id: int) -> bool:
        if robot_id == 1:
            return (
                self.contact_valid_1
                and self.finite(self.contact_fx_mag_1)
                and self.contact_fx_mag_1 >= self.contact_threshold(1)
            )
        return (
            self.contact_valid_2
            and self.finite(self.contact_fx_mag_2)
            and self.contact_fx_mag_2 >= self.contact_threshold(2)
        )

    def wait_for_forward_phase(self, timeout: float = 10.0) -> bool:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            if self.des1 is None or self.des2 is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                continue
            start1 = self.des1.position.x
            start2 = self.des2.position.x
            self.spin_for(0.30)
            if self.des1 is None or self.des2 is None:
                continue
            delta1 = self.des1.position.x - start1
            delta2 = self.des2.position.x - start2
            if delta1 < -1e-4 and delta2 < -1e-4:
                return True
        return False

    def compute_press_target(self, x_offset1: float, x_offset2: float) -> Optional[Dict[str, float]]:
        if self.des1 is None or self.des2 is None:
            return None

        return {
            "offset_x1": x_offset1,
            "offset_y1": 0.0,
            "offset_z1": 0.0,
            "offset_x2": x_offset2,
            "offset_y2": 0.0,
            "offset_z2": 0.0,
        }

    def snapshot(self) -> Dict[str, Any]:
        robot1_max_vel = self.max_abs_velocity(self.js1)
        robot2_max_vel = self.max_abs_velocity(self.js2)
        ee_x_1 = self.ee1.x if self.ee1 is not None else float("nan")
        ee_x_2 = self.ee2.x if self.ee2 is not None else float("nan")
        desired_x_1 = self.des1.position.x if self.des1 is not None else float("nan")
        desired_x_2 = self.des2.position.x if self.des2 is not None else float("nan")

        ee_x_diff = ee_x_1 - ee_x_2 if self.finite(ee_x_1) and self.finite(ee_x_2) else float("nan")
        desired_x_diff = desired_x_1 - desired_x_2 if self.finite(desired_x_1) and self.finite(desired_x_2) else float("nan")
        contact_fx_diff = (
            self.contact_fx_mag_1 - self.contact_fx_mag_2
            if self.finite(self.contact_fx_mag_1) and self.finite(self.contact_fx_mag_2)
            else float("nan")
        )

        return {
            "timestamp": self.now_s(),
            "phase": self.current_phase,
            "offset_x1": self.current_offset_x1,
            "offset_x2": self.current_offset_x2,
            "press_offset_x1": self.current_press_offset_x1,
            "press_offset_x2": self.current_press_offset_x2,
            "contact_mode": self.current_contact_mode,
            "robot1_max_vel": robot1_max_vel,
            "robot2_max_vel": robot2_max_vel,
            "ee_x_1": ee_x_1,
            "ee_x_2": ee_x_2,
            "ee_x_diff": ee_x_diff,
            "desired_x_1": desired_x_1,
            "desired_x_2": desired_x_2,
            "desired_x_diff": desired_x_diff,
            "contact_fx_mag_1": self.contact_fx_mag_1,
            "contact_fx_mag_2": self.contact_fx_mag_2,
            "contact_fx_diff": contact_fx_diff,
            "baseline_fx_1": self.precontact_baseline_fx_1,
            "baseline_fx_2": self.precontact_baseline_fx_2,
            "contact_threshold_1": self.contact_threshold(1),
            "contact_threshold_2": self.contact_threshold(2),
            "contact_valid_1": float(self.contact_valid_1),
            "contact_valid_2": float(self.contact_valid_2),
        }

    def write_csv_header(self, path: str, columns: List[str]) -> None:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(columns)

    def append_csv_row(self, path: str, columns: List[str], row: Dict[str, Any]) -> None:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writerow(row)

    def log_snapshot_row(self) -> None:
        self.append_csv_row(self.csv_path, self.log_columns, self.snapshot())

    def log_sync_step(
        self,
        command_x_offset: float,
    ) -> None:
        row = self.snapshot()
        row["command_x_offset"] = command_x_offset
        self.append_csv_row(self.sync_csv_path, self.sync_columns, row)

    def apply_press_target(self, target: Dict[str, float], contact_mode: str) -> None:
        self.current_contact_mode = contact_mode
        self.publish_offsets(
            x1=target["offset_x1"],
            y1=target["offset_y1"],
            z1=target["offset_z1"],
            x2=target["offset_x2"],
            y2=target["offset_y2"],
            z2=target["offset_z2"],
        )

    def _current_offset_vector(self) -> Tuple[float, float, float, float, float, float]:
        """Return the latest commanded offsets, falling back to zero if unknown."""
        x1 = self.current_offset_x1 if self.finite(self.current_offset_x1) else 0.0
        y1 = self.current_offset_y1 if self.finite(self.current_offset_y1) else 0.0
        z1 = self.current_offset_z1 if self.finite(self.current_offset_z1) else 0.0
        x2 = self.current_offset_x2 if self.finite(self.current_offset_x2) else 0.0
        y2 = self.current_offset_y2 if self.finite(self.current_offset_y2) else 0.0
        z2 = self.current_offset_z2 if self.finite(self.current_offset_z2) else 0.0
        return (x1, y1, z1, x2, y2, z2)

    def _gentle_return_to_nominal(self, reason: str, settle_pause_s: float = 0.5) -> None:
        """Ease both arms back to the nominal zero-offset trajectory."""
        offsets = self._current_offset_vector()
        max_mag = max(abs(value) for value in offsets)

        if max_mag <= 1e-9:
            self.publish_offsets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=1, dt=0.05)
            self.spin_for(settle_pause_s)
            return

        # Smaller steps and a short settle pause keep the recovery from feeling like a snap.
        ramp_steps = max(12, min(60, int(math.ceil(max_mag / 0.0015))))
        self.get_logger().info(
            f"ABORT: gentle return to nominal trajectory ({reason}, {ramp_steps} steps)"
        )

        for idx in range(1, ramp_steps + 1):
            scale = max(0.0, 1.0 - (idx / float(ramp_steps)))
            self.publish_offsets(
                offsets[0] * scale,
                offsets[1] * scale,
                offsets[2] * scale,
                offsets[3] * scale,
                offsets[4] * scale,
                offsets[5] * scale,
                repeats=1,
                dt=0.04,
            )
            self.spin_for(0.04)

        self.publish_offsets(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, repeats=2, dt=0.05)
        self.spin_for(settle_pause_s)

    def safe_abort(self) -> None:
        self.current_phase = "abort"
        try:
            self.get_logger().error("ABORT: returning gently to nominal trajectory")
            self._gentle_return_to_nominal("safe_abort")
        except Exception as e:
            # During interrupt/shutdown, ROS2 context may be invalid
            # This is expected and safe to ignore
            try:
                self.get_logger().debug(f"safe_abort() encountered expected error during shutdown: {e}")
            except:
                pass  # Even logging might fail

    def stage1_liveness(self) -> Tuple[bool, str, Dict[str, Any]]:
        pubs_ok = self.wait_for_subscribers(timeout=10.0)
        js_ok = self.wait_joint_states(timeout=5.0)
        pose_ok = self.wait_for_poses(timeout=3.0)
        ok = pubs_ok and js_ok and pose_ok
        info = {
            "pub1_subs": self.pub1.get_subscription_count(),
            "pub2_subs": self.pub2.get_subscription_count(),
            "js1_seen": self.js1 is not None,
            "js2_seen": self.js2 is not None,
            "ee1_seen": self.ee1 is not None,
            "ee2_seen": self.ee2 is not None,
            "des1_seen": self.des1 is not None,
            "des2_seen": self.des2 is not None,
            "wp1_status_seen": self.wp1_active is not None,
            "wp2_status_seen": self.wp2_active is not None,
            "contact_valid_topics_seen": self.contact_valid_1 or self.contact_valid_2,
        }
        if ok:
            message = "publishers, joint states, and Cartesian pose topics present"
        elif not pubs_ok:
            message = "waypoint subscribers missing"
        elif not js_ok:
            message = "joint_states missing"
        else:
            message = "Cartesian pose topics missing"
        return ok, message, info

    def stage2_idle(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "idle"
        self.spin_for(0.5)
        info = self.snapshot()
        ok = self.check_limits(self.js1, self.idle_vel_limit) and self.check_limits(self.js2, self.idle_vel_limit)
        return ok, "low velocity idle", info

    def stage3_sync_move(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "precontact"
        stage_info: List[Dict[str, Any]] = []

        if not self.wait_for_forward_phase(timeout=10.0):
            return False, "controllers did not enter forward phase", {"samples": stage_info}

        self.spin_for(0.10)
        self.capture_precontact_baseline()

        offset1 = self.precontact_start_x_offset
        offset2 = self.precontact_start_x_offset
        contact1 = False
        contact2 = False
        for step_index in range(self.max_precontact_iterations):
            if not contact1:
                offset1 = max(offset1 - self.press_step, self.precontact_end_x_offset)
            if not contact2:
                offset2 = max(offset2 - self.press_step, self.precontact_end_x_offset)

            target = self.compute_press_target(offset1, offset2)
            if target is None:
                return False, "desired poses unavailable", {"samples": stage_info}
            contact_mode = "robot1_only" if contact1 and not contact2 else "robot2_only" if contact2 and not contact1 else "none"
            self.apply_press_target(target, contact_mode)
            self.spin_for(self.command_sample_delay)

            contact1 = self.contact_detected(1)
            contact2 = self.contact_detected(2)

            snap = self.snapshot()
            snap["command_x_offset"] = min(offset1, offset2)
            stage_info.append(snap)
            self.log_sync_step(min(offset1, offset2))

            if contact1 and contact2:
                return True, "bilateral contact established", {"samples": stage_info}

            if not self.check_limits(self.js1, self.move_vel_limit) or not self.check_limits(self.js2, self.move_vel_limit):
                return False, "velocity spike during precontact approach", {"samples": stage_info}

        return False, "bilateral contact not established", {"samples": stage_info}

    def stage4_hold(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "hold"
        target = self.compute_press_target(self.current_press_offset_x1, self.current_press_offset_x2)
        if target is None:
            return False, "desired poses unavailable", self.snapshot()

        self.apply_press_target(target, "both")
        self.spin_for(0.8)
        info = self.snapshot()
        ok = self.check_limits(self.js1, self.hold_vel_limit) and self.check_limits(self.js2, self.hold_vel_limit)
        return ok, "hold stable", info

    def stage5_adaptive_press(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.current_phase = "compress"
        stage_info: List[Dict[str, Any]] = []

        offset1 = self.current_press_offset_x1
        offset2 = self.current_press_offset_x2
        while min(offset1, offset2) >= self.press_end_x_offset - 1e-9:
            offset1 -= self.press_step
            offset2 -= self.press_step

            contact1 = self.contact_detected(1)
            contact2 = self.contact_detected(2)
            if contact1 and contact2 and self.finite(self.contact_fx_mag_1) and self.finite(self.contact_fx_mag_2):
                if self.contact_fx_mag_1 + self.force_balance_tolerance < self.contact_fx_mag_2:
                    offset1 = max(offset1 - self.side_press_step, self.press_end_x_offset - self.max_side_extra_press)
                    contact_mode = "push_robot1_more"
                elif self.contact_fx_mag_2 + self.force_balance_tolerance < self.contact_fx_mag_1:
                    offset2 = max(offset2 - self.side_press_step, self.press_end_x_offset - self.max_side_extra_press)
                    contact_mode = "push_robot2_more"
                else:
                    contact_mode = "balanced"
            elif contact1 and not contact2:
                offset1 = self.current_press_offset_x1
                offset2 = max(offset2 - self.side_press_step, self.press_end_x_offset - self.max_side_extra_press)
                contact_mode = "push_robot2_to_contact"
            elif contact2 and not contact1:
                offset2 = self.current_press_offset_x2
                offset1 = max(offset1 - self.side_press_step, self.press_end_x_offset - self.max_side_extra_press)
                contact_mode = "push_robot1_to_contact"
            else:
                contact_mode = "seeking_contact"

            target = self.compute_press_target(offset1, offset2)
            if target is None:
                return False, "desired poses unavailable", {"samples": stage_info}

            self.apply_press_target(target, contact_mode)
            self.spin_for(self.command_sample_delay)

            snap = self.snapshot()
            snap["command_x_offset"] = min(offset1, offset2)
            stage_info.append(snap)
            self.log_sync_step(min(offset1, offset2))

            self.current_press_offset_x1 = offset1
            self.current_press_offset_x2 = offset2

            if not self.check_limits(self.js1, self.move_vel_limit) or not self.check_limits(self.js2, self.move_vel_limit):
                return False, "velocity spike during compression", {"samples": stage_info}

            f1 = snap["contact_fx_mag_1"]
            f2 = snap["contact_fx_mag_2"]
            fd = snap["contact_fx_diff"]
            both_contact = bool(snap["contact_valid_1"]) and bool(snap["contact_valid_2"])
            if both_contact and self.finite(fd) and abs(fd) > self.force_diff_abort:
                return False, "force asymmetry too high", {"samples": stage_info}
            if both_contact and self.finite(f1) and self.finite(f2) and max(f1, f2) > self.max_force_mag_abort:
                return False, "contact force too high", {"samples": stage_info}

        return True, "adaptive bilateral compression completed", {"samples": stage_info}

    def run(self) -> List[Dict[str, Any]]:
        stages = [
            self.stage1_liveness,
            self.stage2_idle,
            self.stage3_sync_move,
            self.stage4_hold,
            self.stage5_adaptive_press,
        ]

        results: List[Dict[str, Any]] = []

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
                return results

        self.current_phase = "done"
        self.get_logger().info("ALL STAGES PASSED")
        return results


def main() -> None:
    rclpy.init()
    node = HardwareHarnessAdaptive()
    try:
        out = node.run()
        print(json.dumps(out, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
