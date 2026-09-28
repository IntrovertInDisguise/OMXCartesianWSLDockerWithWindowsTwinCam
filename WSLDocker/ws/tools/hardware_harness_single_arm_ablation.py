#!/usr/bin/env python3
"""
hardware_harness_single_arm_ablation.py
────────────────────────────────────────
K_lat ablation harness for a single Open Manipulator-X arm pressing against
a fixed spring or rigid wall.  Sweeps lateral stiffness setpoints, runs N
repetitions per setpoint, and logs per-run CSVs + JSONs with a consolidated
summary.csv for downstream analysis.

Physical setup
──────────────
  • One OMX robot (robot1 by default) pressing in –x against a fixed spring
    or wall.  Spring parameters should be calibrated in advance.
  • K_lat is set via the /robot1/robot1_variable_stiffness/set_k_lateral topic
    (or the configured SET_KLAT_METHOD).
  • Optional depth camera for lateral deflection monitoring.

Usage
─────
  python3 tools/hardware_harness_single_arm_ablation.py                     # full sweep, 20 reps
  python3 tools/hardware_harness_single_arm_ablation.py --reps 5            # quick smoke test
  python3 tools/hardware_harness_single_arm_ablation.py --k-lat 10 25 50 70 # explicit K_lat values
  python3 tools/hardware_harness_single_arm_ablation.py --spring s1 s2      # per-spring sweep
  python3 tools/hardware_harness_single_arm_ablation.py --gazebo-smoke      # single fixed-stiffness Gazebo test
  python3 tools/hardware_harness_single_arm_ablation.py --dry-run           # liveness check only

Output (written to OMX_LOG_DIR or /tmp/variable_stiffness_logs):
  single_arm_ablation_<timestamp>/
      run_<spring>_K<klat>_rep<rep>_<ts>.csv   # full 50 Hz log per run
      run_<spring>_K<klat>_rep<rep>_<ts>.json  # per-run scalar summary
      summary.csv                              # consolidated results
      ablation_log.txt                         # human-readable log

MANDATORY BRINGUP (in a separate terminal before starting this script):
  source /opt/ros/humble/setup.bash && source /workspaces/omx_ros2/ws/install/setup.bash
  ros2 launch omx_variable_stiffness_controller single_arm_hardware.launch.py \\
    robot1_port:=/dev/serial/by-id/<ID> enable_logger:=true enable_live_plot:=false start_rviz:=false

Wait until the robot logs  state : MOVE_RETURN  before starting this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point, Pose, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64

# ── Shared ablation config ────────────────────────────────────────────────────
try:
    from tools.ablation_config import (
        CARTESIAN_STIFFNESS_LIMIT_NPM,
        HW_K_LAT_CASES,
        SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD,
        SPRING_SPECIMENS,
    )
except ImportError:
    from ablation_config import (
        CARTESIAN_STIFFNESS_LIMIT_NPM,
        HW_K_LAT_CASES,
        SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD,
        SPRING_SPECIMENS,
    )

# ── Single-arm ablation defaults ─────────────────────────────────────────────
DEFAULT_REPS = 20
DEFAULT_K_LAT_VALUES = [10.0, 25.0, 40.0, 55.0, CARTESIAN_STIFFNESS_LIMIT_NPM]
REST_BETWEEN_REPS_S = 4.0
REST_BETWEEN_CASES_S = 6.0

# K_lat topic for single arm
KLAT_TOPIC_R1 = "/robot1/robot1_variable_stiffness/set_k_lateral"
SET_KLAT_METHOD = "topic"

# Motion / force tuning (single-arm defaults from hardware_harness_single_arm_v3)
IDLE_VEL_LIMIT = 1.0
MOVE_VEL_LIMIT = 1.5
HOLD_VEL_LIMIT = 0.8
COMMAND_REPEATS = 2
COMMAND_DT = 0.05
COMMAND_SAMPLE_DELAY = 0.45

# Precontact approach
PRECONTACT_START_X = 0.12   # offset from home [m]
PRECONTACT_END_X = 0.30     # max forward travel
PRESS_END_X = 0.050         # total press depth
PRESS_STEP = 0.0005         # 0.5 mm quasi-static step
RAMP_SAMPLE_DELAY = 0.20

# Force thresholds
CONTACT_FORCE_ENTER = 0.60  # N
CONTACT_FORCE_DELTA = 0.10  # N
FORCE_DIFF_ABORT = 4.0      # N

# Timing
FORWARD_PHASE_TIMEOUT_S = 8.0
PRESS_RAMP_TIMEOUT_S = 40.0
IDLE_WAIT_TIMEOUT_S = 45.0
HOLD_DURATION_S = 6.0

# Torque disable on abort
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
DEFAULT_TORQUE_DISABLE_BAUD = 1000000


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    """Per-run result for a single-arm K_lat ablation run."""
    spring:       str = ""
    case:         str = ""
    K_lat_Npm:    float = 0.0
    rep:          int = 0
    ts:           str = ""
    passed:       bool = False
    ablation_index: int = 0
    abort_reason: str = ""
    # timing [s]
    t_contact_s:      float = float("nan")
    contact_duration_s: float = float("nan")
    # force / displacement
    P_s_max_N:        float = float("nan")
    P_s_at_contact_N: float = float("nan")
    ee_x_at_contact:  float = float("nan")
    ee_x_at_max:      float = float("nan")
    max_press_depth:  float = float("nan")
    # derived
    P_c_th_N:         float = float("nan")
    K_lat_meas_Npm:   float = float("nan")
    # files
    csv_path:  str = ""
    json_path: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# HARNESS NODE
# ═══════════════════════════════════════════════════════════════════════════════

class SingleArmAblationHarness(Node):
    """Single-arm K_lat ablation harness.

    Iterates over spring/K_lat setpoints, runs N reps per setpoint,
    and logs per-run CSVs + JSONs with a consolidated summary.csv.
    """

    def __init__(
        self,
        robot_namespace: str = "robot1",
        log_root: str = "/tmp/variable_stiffness_logs",
        skip_klat_command: bool = False,
        abort_disable_torque: bool = False,
        robot1_port: Optional[str] = None,
        torque_disable_baud: int = DEFAULT_TORQUE_DISABLE_BAUD,
        use_sim_time: bool = False,
    ) -> None:
        super().__init__(
            "single_arm_ablation_harness",
            parameter_overrides=[
                Parameter("use_sim_time", Parameter.Type.BOOL, use_sim_time),
            ],
        )
        self.robot_ns = robot_namespace
        self.skip_klat_command = skip_klat_command
        self.abort_disable_torque = abort_disable_torque
        self.robot1_port = robot1_port
        self.torque_disable_baud = torque_disable_baud
        self.log_root = log_root

        # ── publishers ────────────────────────────────────────────────────────
        self.pub_wp = self.create_publisher(
            PoseStamped,
            f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/waypoint_command",
            10,
        )
        if SET_KLAT_METHOD == "topic":
            self.klat_pub = self.create_publisher(Float64, KLAT_TOPIC_R1, 1)

        # ── subscriptions ─────────────────────────────────────────────────────
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, f"/{self.robot_ns}/joint_states", self._cb_js, qos)
        self.create_subscription(Point, f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/end_effector_position", self._cb_ee, qos)
        self.create_subscription(Pose, f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/cartesian_pose_desired", self._cb_des, qos)
        self.create_subscription(WrenchStamped, f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/contact_wrench", self._cb_wrench, qos)
        self.create_subscription(Bool, f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/contact_valid", self._cb_contact_valid, qos)
        self.create_subscription(Bool, f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness/waypoint_active", self._cb_wp, qos)

        # ── state ─────────────────────────────────────────────────────────────
        self.js: Optional[JointState] = None
        self.ee: Optional[Point] = None
        self.des: Optional[Pose] = None
        self.contact_fx = float("nan")
        self.contact_fy = float("nan")
        self.contact_fz = float("nan")
        self.contact_tx = float("nan")
        self.contact_ty = float("nan")
        self.contact_tz = float("nan")
        self.contact_force_norm = float("nan")
        self.contact_valid = False
        self.wp_active: Optional[bool] = None
        self.baseline_fx = float("nan")

        # ── per-run logging ───────────────────────────────────────────────────
        self._logging_active = False
        self._run_csv_path = ""
        self._run_csv_file = None
        self._run_csv_writer: Optional[csv.DictWriter] = None
        self._log_cols: List[str] = []

        # ── per-run onset tracking ────────────────────────────────────────────
        self._run_start_t = 0.0
        self._t_contact = float("nan")
        self._p_s_max = 0.0
        self._p_s_at_contact = float("nan")
        self._ee_x_at_contact = float("nan")
        self._ee_x_at_max = float("nan")
        self._max_press_depth = 0.0

        # ── ablation log ──────────────────────────────────────────────────────
        self._ablation_log: List[str] = []
        self._ablation_log_path = ""

        self.arm_joint_names = ("joint1", "joint2", "joint3", "joint4")

    # ── ROS callbacks ─────────────────────────────────────────────────────────
    def _cb_js(self, m: JointState) -> None:
        self.js = m

    def _cb_ee(self, m: Point) -> None:
        self.ee = m

    def _cb_des(self, m: Pose) -> None:
        self.des = m

    def _cb_wrench(self, m: WrenchStamped) -> None:
        self.contact_fx = m.wrench.force.x
        self.contact_fy = m.wrench.force.y
        self.contact_fz = m.wrench.force.z
        self.contact_tx = m.wrench.torque.x
        self.contact_ty = m.wrench.torque.y
        self.contact_tz = m.wrench.torque.z
        self.contact_force_norm = math.sqrt(
            self.contact_fx**2 + self.contact_fy**2 + self.contact_fz**2
        )

    def _cb_contact_valid(self, m: Bool) -> None:
        self.contact_valid = m.data

    def _cb_wp(self, m: Bool) -> None:
        self.wp_active = m.data

    # ── helpers ───────────────────────────────────────────────────────────────
    def spin_for(self, dt: float) -> None:
        end = time.time() + dt
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _max_joint_vel(self) -> float:
        if self.js is None or not self.js.velocity:
            return 0.0
        return max(abs(v) for v in self.js.velocity)

    def _ee_pos(self) -> Tuple[float, float, float]:
        if self.ee is None:
            return (float("nan"), float("nan"), float("nan"))
        return (self.ee.x, self.ee.y, self.ee.z)

    def _des_pos(self) -> Tuple[float, float, float]:
        if self.des is None:
            return (float("nan"), float("nan"), float("nan"))
        return (self.des.position.x, self.des.position.y, self.des.position.z)

    def _alog(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._ablation_log.append(line)
        self.get_logger().info(msg)

    def _flush_alog(self) -> None:
        if self._ablation_log_path:
            with open(self._ablation_log_path, "w") as f:
                f.write("\n".join(self._ablation_log))

    # ── K_lat control ────────────────────────────────────────────────────────
    def set_k_lateral(self, k_lat: float) -> None:
        """Set the lateral stiffness for the single arm."""
        if self.skip_klat_command:
            self.get_logger().warn(
                f"Skipping K_lat command for {k_lat:.1f} N/m (gazebo-smoke mode)"
            )
            return

        if k_lat > CARTESIAN_STIFFNESS_LIMIT_NPM:
            raise ValueError(
                f"Requested K_lat={k_lat:.1f} N/m exceeds limit "
                f"{CARTESIAN_STIFFNESS_LIMIT_NPM:.1f} N/m"
            )

        if SET_KLAT_METHOD == "topic":
            msg = Float64()
            msg.data = float(k_lat)
            for _ in range(5):
                self.klat_pub.publish(msg)
                self.spin_for(0.1)
            self.get_logger().info(f"K_lat set to {k_lat:.1f} N/m via topic")
            self.spin_for(0.5)

        elif SET_KLAT_METHOD == "ros2_param":
            ctrl_node = f"/{self.robot_ns}/{self.robot_ns}_variable_stiffness"
            cmd = ["ros2", "param", "set", ctrl_node, "k_lateral", str(k_lat)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
            if result.returncode != 0:
                self.get_logger().warn(f"ros2 param set failed: {result.stderr.strip()}")
            self.spin_for(0.5)

        elif SET_KLAT_METHOD == "manual":
            self.get_logger().warn(
                f"\n{'='*60}\n"
                f"  ACTION REQUIRED: set K_lat = {k_lat:.1f} N/m.\n"
                f"  Press ENTER when done.\n"
                f"{'='*60}"
            )
            input(f"  >> Set K_lat = {k_lat:.1f} N/m, then press ENTER: ")
            self.spin_for(1.0)

    # ── waypoint publishing ──────────────────────────────────────────────────
    def publish_waypoint(self, x: float, y: float, z: float) -> None:
        """Publish a waypoint command to the controller."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        for _ in range(COMMAND_REPEATS):
            self.pub_wp.publish(msg)
            self.spin_for(COMMAND_DT)

    # ── per-run CSV logging ──────────────────────────────────────────────────
    def _start_run_log(self, csv_path: str, spring: str, k_lat: float, rep: int) -> None:
        self._log_cols = [
            "timestamp", "spring", "K_lat_Npm", "rep",
            "ee_x", "ee_y", "ee_z",
            "des_x", "des_y", "des_z",
            "contact_fx", "contact_fy", "contact_fz",
            "contact_tx", "contact_ty", "contact_tz",
            "contact_force_norm", "contact_valid",
            "arm_max_vel", "wp_active",
        ]
        self._run_csv_path = csv_path
        self._run_csv_file = open(csv_path, "w", newline="")
        self._run_csv_writer = csv.DictWriter(self._run_csv_file, fieldnames=self._log_cols)
        self._run_csv_writer.writeheader()
        self._logging_active = True

    def _log_tick(self) -> None:
        if not self._logging_active or self._run_csv_writer is None:
            return
        ee = self._ee_pos()
        des = self._des_pos()
        row = {
            "timestamp": time.time(),
            "spring": "",
            "K_lat_Npm": "",
            "rep": "",
            "ee_x": ee[0], "ee_y": ee[1], "ee_z": ee[2],
            "des_x": des[0], "des_y": des[1], "des_z": des[2],
            "contact_fx": self.contact_fx,
            "contact_fy": self.contact_fy,
            "contact_fz": self.contact_fz,
            "contact_tx": self.contact_tx,
            "contact_ty": self.contact_ty,
            "contact_tz": self.contact_tz,
            "contact_force_norm": self.contact_force_norm,
            "contact_valid": self.contact_valid,
            "arm_max_vel": self._max_joint_vel(),
            "wp_active": self.wp_active,
        }
        self._run_csv_writer.writerow(row)

    def _stop_run_log(self) -> None:
        self._logging_active = False
        if self._run_csv_file is not None:
            self._run_csv_file.close()
            self._run_csv_file = None
            self._run_csv_writer = None

    # ── onset tracking ───────────────────────────────────────────────────────
    def _reset_onset(self) -> None:
        self._run_start_t = time.time()
        self._t_contact = float("nan")
        self._p_s_max = 0.0
        self._p_s_at_contact = float("nan")
        self._ee_x_at_contact = float("nan")
        self._ee_x_at_max = float("nan")
        self._max_press_depth = 0.0

    # ── stage: liveness ──────────────────────────────────────────────────────
    def _stage_liveness(self) -> Tuple[bool, str]:
        """Verify publishers and subscriptions are alive."""
        timeout = 10.0
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.js is not None and self.ee is not None:
                return True, "liveness OK"
            self.spin_for(0.1)
        missing = []
        if self.js is None:
            missing.append("joint_states")
        if self.ee is None:
            missing.append("end_effector_position")
        return False, f"missing topics: {missing}"

    # ── stage: idle ──────────────────────────────────────────────────────────
    def _stage_idle(self) -> Tuple[bool, str]:
        """Wait for low-velocity stabilization and capture baseline forces."""
        t0 = time.time()
        stable_count = 0
        baseline_samples: List[float] = []
        while time.time() - t0 < IDLE_WAIT_TIMEOUT_S:
            vel = self._max_joint_vel()
            if vel < IDLE_VEL_LIMIT:
                stable_count += 1
                if not math.isnan(self.contact_fx):
                    baseline_samples.append(abs(self.contact_fx))
            else:
                stable_count = 0
                baseline_samples.clear()
            if stable_count >= 10 and len(baseline_samples) >= 5:
                self.baseline_fx = sum(baseline_samples) / len(baseline_samples)
                return True, f"idle OK (baseline_fx={self.baseline_fx:.3f} N)"
            self.spin_for(0.1)
        return False, f"idle timeout after {IDLE_WAIT_TIMEOUT_S:.0f}s"

    # ── stage: precontact approach ───────────────────────────────────────────
    def _stage_precontact(self) -> Tuple[bool, str]:
        """Step the arm toward the contact target until contact is detected."""
        n_steps = int((PRECONTACT_END_X - PRECONTACT_START_X) / 0.002)
        for i in range(n_steps):
            x_offset = PRECONTACT_START_X + i * 0.002
            self.publish_waypoint(x_offset, 0.0, 0.1)
            self.spin_for(COMMAND_SAMPLE_DELAY)

            # Check for contact
            if not math.isnan(self.contact_fx):
                fx_abs = abs(self.contact_fx)
                delta = fx_abs - (self.baseline_fx if not math.isnan(self.baseline_fx) else 0.0)
                if fx_abs > CONTACT_FORCE_ENTER and delta > CONTACT_FORCE_DELTA:
                    self._t_contact = time.time() - self._run_start_t
                    self._p_s_at_contact = fx_abs
                    self._ee_x_at_contact = self._ee_pos()[0]
                    self._alog(f"      Contact detected at t={self._t_contact:.2f}s, F={fx_abs:.3f}N")
                    return True, f"contact at offset={x_offset:.4f}m, F={fx_abs:.3f}N"

        return False, "no contact detected during precontact approach"

    # ── stage: hold ──────────────────────────────────────────────────────────
    def _stage_hold(self) -> Tuple[bool, str]:
        """Sustain contact for a fixed duration."""
        t0 = time.time()
        while time.time() - t0 < HOLD_DURATION_S:
            if not math.isnan(self.contact_force_norm) and self.contact_force_norm > FORCE_DIFF_ABORT:
                return False, f"force too high during hold: {self.contact_force_norm:.2f}N"
            self.spin_for(0.1)
        return True, f"hold OK ({HOLD_DURATION_S:.0f}s)"

    # ── stage: quasi-static ramp ─────────────────────────────────────────────
    def _stage_ramp(self, spring: str, k_lat: float, rep: int) -> Tuple[bool, str]:
        """Incrementally compress along the press direction, logging 50Hz data."""
        # Start 50Hz logging
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_k = f"{k_lat:.0f}".replace(".", "p")
        safe_spring = spring.replace(".", "p").replace(" ", "_")
        run_label = f"run_{safe_spring}_K{safe_k}_rep{rep:02d}"
        csv_path = os.path.join(self._current_run_dir, f"{run_label}_{ts}.csv")
        json_path = os.path.join(self._current_run_dir, f"{run_label}_{ts}.json")

        self._current_csv_path = csv_path
        self._current_json_path = json_path
        self._start_run_log(csv_path, spring, k_lat, rep)

        t0 = time.time()
        press_depth = 0.0
        max_force = 0.0
        ee_x_at_max = float("nan")

        while press_depth < PRESS_END_X and (time.time() - t0) < PRESS_RAMP_TIMEOUT_S:
            press_depth += PRESS_STEP
            x_offset = PRECONTACT_START_X + press_depth
            self.publish_waypoint(x_offset, 0.0, 0.1)
            self._log_tick()
            self.spin_for(RAMP_SAMPLE_DELAY)

            # Track max force
            if not math.isnan(self.contact_force_norm):
                if self.contact_force_norm > max_force:
                    max_force = self.contact_force_norm
                    ee_x_at_max = self._ee_pos()[0]

            # Abort on excessive force
            if max_force > FORCE_DIFF_ABORT:
                self._alog(f"      Force abort: {max_force:.2f}N > {FORCE_DIFF_ABORT}N")
                self._stop_run_log()
                return False, f"force abort at depth={press_depth:.4f}m, F={max_force:.2f}N"

        self._p_s_max = max_force
        self._ee_x_at_max = ee_x_at_max
        self._max_press_depth = press_depth
        self._stop_run_log()
        return True, f"ramp OK (max_F={max_force:.3f}N, depth={press_depth:.4f}m)"

    # ── stage: withdraw ──────────────────────────────────────────────────────
    def _stage_withdraw(self) -> Tuple[bool, str]:
        """Retract to home position."""
        self.publish_waypoint(0.22, 0.0, 0.1)
        self.spin_for(3.0)
        return True, "withdraw OK"

    # ── safe abort ───────────────────────────────────────────────────────────
    def safe_abort(self) -> None:
        """Retract and optionally disable torque."""
        self.get_logger().warn("SAFE ABORT — retracting")
        self.publish_waypoint(0.22, 0.0, 0.1)
        self.spin_for(2.0)
        if self.abort_disable_torque and self.robot1_port:
            self._disable_torque()

    def _disable_torque(self) -> None:
        if not os.path.isfile(self.robot1_port or ""):
            self.get_logger().warn(f"Torque disable: port {self.robot1_port} not found")
            return
        if not os.path.isfile(DISABLE_TORQUE_SCRIPT):
            self.get_logger().warn(f"Torque disable script not found: {DISABLE_TORQUE_SCRIPT}")
            return
        cmd = [
            sys.executable, DISABLE_TORQUE_SCRIPT,
            "--port", self.robot1_port,
            "--baud", str(self.torque_disable_baud),
        ]
        try:
            subprocess.run(cmd, timeout=5.0, capture_output=True, text=True)
        except Exception as e:
            self.get_logger().warn(f"Torque disable failed: {e}")

    # ── single run ───────────────────────────────────────────────────────────
    def _run_one(
        self,
        spring: str,
        case: str,
        K_lat: float,
        rep: int,
        P_c_th: float,
        run_dir: str,
    ) -> RunResult:
        """Execute one ablation run (liveness → idle → precontact → hold → ramp → withdraw)."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._current_run_dir = run_dir
        result = RunResult(
            spring=spring,
            case=case,
            K_lat_Npm=K_lat,
            rep=rep,
            ts=ts,
            passed=False,
            P_c_th_N=P_c_th,
        )

        self._reset_onset()

        stages = [
            ("liveness",   self._stage_liveness),
            ("idle",       self._stage_idle),
            ("precontact", self._stage_precontact),
            ("hold",       self._stage_hold),
        ]

        passed_all = True
        try:
            for name, fn in stages:
                ok, msg = fn()
                self._alog(f"    [{spring} K={K_lat:.0f} rep{rep}] {name}: {msg}")
                if not ok:
                    self.safe_abort()
                    result.abort_reason = msg
                    passed_all = False
                    break

            if passed_all:
                ok, msg = self._stage_ramp(spring, K_lat, rep)
                self._alog(f"    [{spring} K={K_lat:.0f} rep{rep}] ramp: {msg}")
                if not ok:
                    self.safe_abort()
                    result.abort_reason = msg
                    passed_all = False

            if passed_all:
                self._stage_withdraw()
                result.passed = True
        except Exception as e:
            self._alog(f"    [{spring} K={K_lat:.0f} rep{rep}] EXCEPTION: {e}")
            result.abort_reason = str(e)
            self.safe_abort()
        finally:
            self._stop_run_log()

        # Populate result from onset tracking
        result.t_contact_s = self._t_contact
        result.P_s_max_N = self._p_s_max
        result.P_s_at_contact_N = self._p_s_at_contact
        result.ee_x_at_contact = self._ee_x_at_contact
        result.ee_x_at_max = self._ee_x_at_max
        result.max_press_depth = self._max_press_depth
        result.csv_path = getattr(self, "_current_csv_path", "")
        result.json_path = getattr(self, "_current_json_path", "")

        # Write per-run JSON
        if result.json_path:
            with open(result.json_path, "w") as f:
                json.dump({
                    "spring": spring,
                    "case": case,
                    "K_lat_Npm": K_lat,
                    "rep": rep,
                    "ts": ts,
                    "ablation_index": result.ablation_index,
                    "passed": result.passed,
                    "abort_reason": result.abort_reason,
                    "t_contact_s": result.t_contact_s,
                    "P_s_max_N": result.P_s_max_N,
                    "P_s_at_contact_N": result.P_s_at_contact,
                    "ee_x_at_contact": result.ee_x_at_contact,
                    "ee_x_at_max": result.ee_x_at_max,
                    "max_press_depth": result.max_press_depth,
                    "P_c_th_N": P_c_th,
                    "K_lat_meas_Npm": result.K_lat_meas_Npm,
                }, f, indent=2)

        return result

    # ── main ablation loop ───────────────────────────────────────────────────
    def run_ablation(
        self,
        springs_klat: Dict[str, List[Tuple[str, float]]],
        n_reps: int,
        dry_run: bool = False,
    ) -> List[RunResult]:
        """
        Main entry point.  Iterates over every (spring, K_lat) pair and runs
        n_reps repetitions, writing results incrementally.

        Parameters
        ----------
        springs_klat : dict
            Mapping of spring label → list of (case_label, K_lat_value) tuples.
            Example: {"s1": [("K_low", 10.0), ("K_ref", 65.0)], "s2": [...]}
        n_reps : int
            Number of repetitions per (spring, K_lat) setpoint.
        dry_run : bool
            If True, run liveness check only without motion.
        """
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.log_root, f"single_arm_ablation_{run_ts}")
        os.makedirs(run_dir, exist_ok=True)
        self._ablation_log_path = os.path.join(run_dir, "ablation_log.txt")

        # Summary CSV
        summary_path = os.path.join(run_dir, "summary.csv")
        SUMMARY_COLS = [
            "spring", "case", "K_lat_Npm", "K_lat_meas_Npm",
            "P_c_th_N",
            "t_contact_s", "P_s_at_contact_N",
            "P_s_max_N", "max_press_depth",
            "ee_x_at_contact", "ee_x_at_max",
            "ablation_index", "rep", "passed", "abort_reason",
            "csv_path", "json_path",
        ]
        with open(summary_path, "w", newline="") as sf:
            csv.DictWriter(sf, fieldnames=SUMMARY_COLS, extrasaction="ignore").writeheader()

        all_results: List[RunResult] = []
        n_total = sum(len(v) for v in springs_klat.values()) * n_reps
        n_done = 0

        if dry_run:
            self._alog("DRY RUN: liveness check only, no motion.")
            ok, msg = self._stage_liveness()
            self._alog(f"Liveness: {msg}")
            self._flush_alog()
            return []

        prev_klat: Optional[float] = None
        for spring, klat_list in springs_klat.items():
            for case_label, K_lat in klat_list:
                # Compute P_c_th
                K_THETA_NM = SPRING_CONTACT_ROTATIONAL_STIFFNESS_NM_PER_RAD
                P_c_th = math.sqrt(max(K_lat, 0.0) * K_THETA_NM)

                # Change K_lat if needed
                if K_lat != prev_klat:
                    self._alog(f"Setting K_lat = {K_lat:.1f} N/m  [{spring}/{case_label}]")
                    self.set_k_lateral(K_lat)
                    self.spin_for(REST_BETWEEN_CASES_S)
                    prev_klat = K_lat

                self._alog(f"\n{'─'*50}")
                self._alog(f"SPRING: {spring}  CASE: {case_label}  K_lat={K_lat:.1f} N/m  P_c_th={P_c_th:.3f} N  reps={n_reps}")

                for rep in range(1, n_reps + 1):
                    n_done += 1
                    self._alog(f"  Rep {rep}/{n_reps}  ({n_done}/{n_total} total)")

                    r = self._run_one(spring, case_label, K_lat, rep, P_c_th, run_dir)
                    r.ablation_index = n_done
                    all_results.append(r)

                    # Append to summary CSV
                    with open(summary_path, "a", newline="") as sf:
                        csv.DictWriter(sf, fieldnames=SUMMARY_COLS, extrasaction="ignore").writerow({
                            "spring": r.spring,
                            "case": r.case,
                            "K_lat_Npm": r.K_lat_Npm,
                            "K_lat_meas_Npm": r.K_lat_meas_Npm,
                            "P_c_th_N": r.P_c_th_N,
                            "t_contact_s": r.t_contact_s,
                            "P_s_at_contact_N": r.P_s_at_contact,
                            "P_s_max_N": r.P_s_max_N,
                            "max_press_depth": r.max_press_depth,
                            "ee_x_at_contact": r.ee_x_at_contact,
                            "ee_x_at_max": r.ee_x_at_max,
                            "ablation_index": r.ablation_index,
                            "rep": r.rep,
                            "passed": r.passed,
                            "abort_reason": r.abort_reason,
                            "csv_path": r.csv_path,
                            "json_path": r.json_path,
                        })

                    # Status line
                    status = "PASS" if r.passed else f"FAIL ({r.abort_reason})"
                    self._alog(
                        f"    → {status}  "
                        f"t_contact={r.t_contact_s:.2f}s  "
                        f"P_max={r.P_s_max_N:.2f}N  "
                        f"depth={r.max_press_depth:.4f}m"
                    )
                    self._flush_alog()

                    # Rest between reps
                    if rep < n_reps:
                        self.spin_for(REST_BETWEEN_REPS_S)

        self._alog(f"\n{'='*50}")
        n_pass = sum(1 for r in all_results if r.passed)
        self._alog(f"COMPLETE: {n_pass}/{len(all_results)} runs passed.")
        self._alog(f"Summary CSV: {summary_path}")
        self._flush_alog()

        return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-arm K_lat ablation harness")
    p.add_argument("--reps", type=int, default=DEFAULT_REPS,
                   help="Repetitions per K_lat setpoint (default 20)")
    p.add_argument("--spring", nargs="*", default=None,
                   help="Spring specimens to sweep (e.g. s1 s2). Uses SPRING_SPECIMENS from ablation_config.")
    p.add_argument("--k-lat", nargs="*", type=float, default=None,
                   help="Explicit K_lat values to run (overrides --spring)")
    p.add_argument("--cases", nargs="*", default=None,
                   help="Subset of case names to run (e.g. K_low K_mid K_ref)")
    p.add_argument("--dry-run", action="store_true",
                   help="Liveness check only, no motion")
    p.add_argument("--gazebo-smoke", action="store_true",
                   help="Single fixed-stiffness Gazebo smoke test")
    p.add_argument("--robot-namespace", type=str, default="robot1",
                   help="Robot namespace (default: robot1)")
    p.add_argument("--abort-disable-torque", action="store_true",
                   help="Disable Dynamixel torque on abort")
    p.add_argument("--robot1-port", default=None,
                   help="Serial port for torque disable on abort")
    p.add_argument("--dxl-baud", type=int, default=DEFAULT_TORQUE_DISABLE_BAUD,
                   help="Baud rate for torque disable")
    p.add_argument("--use-sim-time", action="store_true", default=False,
                   help="Use simulation time (for Gazebo)")
    p.add_argument("--log-dir", default=os.environ.get("OMX_LOG_DIR", "/tmp/variable_stiffness_logs"),
                   help="Output directory for run logs")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def build_springs_klat(args: argparse.Namespace) -> Dict[str, List[Tuple[str, float]]]:
    """Build the springs_klat mapping from CLI args.

    Returns
    -------
    dict
        {spring_label: [(case_label, K_lat_value), ...]}
    """
    if args.gazebo_smoke:
        if args.k_lat:
            unique = sorted(set(float(v) for v in args.k_lat))
            if len(unique) != 1:
                raise SystemExit("--gazebo-smoke requires exactly one unique --k-lat value")
            k_val = unique[0]
        else:
            k_val = CARTESIAN_STIFFNESS_LIMIT_NPM
        return {"gazebo_smoke": [("gazebo_smoke", k_val)]}

    if args.k_lat:
        if any(v > CARTESIAN_STIFFNESS_LIMIT_NPM for v in args.k_lat):
            raise SystemExit(
                f"--k-lat values must be <= {CARTESIAN_STIFFNESS_LIMIT_NPM:.1f} N/m"
            )
        # Single-arm: all K_lat values under one "single_arm" spring label
        # unless --spring is specified
        springs = args.spring or ["single_arm"]
        result: Dict[str, List[Tuple[str, float]]] = {}
        for sp in springs:
            result[sp] = [("custom", v) for v in args.k_lat]
        return result

    # Use SPRING_SPECIMENS if available and --spring is given
    if args.spring and SPRING_SPECIMENS:
        result = {}
        for sp_name in args.spring:
            if sp_name not in SPRING_SPECIMENS:
                raise SystemExit(
                    f"Unknown spring '{sp_name}'. Available: {list(SPRING_SPECIMENS.keys())}"
                )
            specimen = SPRING_SPECIMENS[sp_name]
            # Build K_lat sweep based on the specimen's stiffness
            # Use fractions of the specimen stiffness
            k_nom = specimen["stiffness_npm"]
            k_values = sorted(set(
                v for v in [
                    k_nom * 0.15,
                    k_nom * 0.30,
                    k_nom * 0.50,
                    k_nom * 0.75,
                    k_nom,
                    CARTESIAN_STIFFNESS_LIMIT_NPM,
                ]
                if v <= CARTESIAN_STIFFNESS_LIMIT_NPM and v > 0
            ))
            # Build case labels
            cases = []
            case_names = args.cases if args.cases else None
            if case_names:
                # Filter to requested cases
                all_cases = HW_K_LAT_CASES
                for cn in case_names:
                    if cn in all_cases:
                        for v in all_cases[cn]:
                            if v <= CARTESIAN_STIFFNESS_LIMIT_NPM:
                                cases.append((cn, v))
            else:
                for k in k_values:
                    case_label = "K_low" if k <= CARTESIAN_STIFFNESS_LIMIT_NPM * 0.4 else \
                                 "K_mid" if k < CARTESIAN_STIFFNESS_LIMIT_NPM else "K_ref"
                    cases.append((case_label, k))
            if not cases:
                cases = [("K_ref", CARTESIAN_STIFFNESS_LIMIT_NPM)]
            result[sp_name] = cases
        return result

    # Default: use HW_K_LAT_CASES with a single "single_arm" label
    if args.cases:
        cases_klat = {k: v for k, v in HW_K_LAT_CASES.items() if k in args.cases}
        if not cases_klat:
            raise SystemExit(f"No matching cases. Available: {list(HW_K_LAT_CASES.keys())}")
    else:
        cases_klat = HW_K_LAT_CASES

    result = {"single_arm": []}
    for case_name, k_values in cases_klat.items():
        for k in k_values:
            result["single_arm"].append((case_name, k))
    return result


def main() -> int:
    args = parse_args()

    if args.use_sim_time:
        rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    else:
        rclpy.init()

    node = SingleArmAblationHarness(
        robot_namespace=args.robot_namespace,
        log_root=args.log_dir,
        skip_klat_command=args.gazebo_smoke,
        abort_disable_torque=args.abort_disable_torque,
        robot1_port=args.robot1_port,
        torque_disable_baud=args.dxl_baud,
        use_sim_time=args.use_sim_time,
    )

    exit_code = 0
    try:
        springs_klat = build_springs_klat(args)
        node._alog(f"Spring/K_lat plan:")
        for sp, klat_list in springs_klat.items():
            for case_label, k_val in klat_list:
                node._alog(f"  {sp} / {case_label}: K_lat={k_val:.1f} N/m")
        node._flush_alog()

        results = node.run_ablation(
            springs_klat=springs_klat,
            n_reps=args.reps,
            dry_run=args.dry_run,
        )

        if results:
            n_pass = sum(1 for r in results if r.passed)
            print(f"\nAblation complete: {n_pass}/{len(results)} passed.")
            if n_pass < len(results):
                exit_code = 1
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
        node.safe_abort()
        exit_code = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
