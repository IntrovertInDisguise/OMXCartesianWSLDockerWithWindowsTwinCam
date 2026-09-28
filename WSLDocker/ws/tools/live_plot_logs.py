#!/usr/bin/env python3
"""
Live timeseries plotter — works for both controller variants.

Subscribes directly to ROS2 topics — no log files are ever read or written.
Uses the same publication-quality matplotlib style as plot_logs.py.

Two controller modes:

  variable_stiffness  Subscribe to all detailed controller state topics
                      (cartesian pose, EE velocity, stiffness/damping, etc.)
                      Published by OmxVariableStiffnessController.

  gravity_comp        Subscribe to /joint_states (from joint_state_broadcaster)
                      and plot joint positions, velocities and compensation
                      efforts.  Published by joint_state_broadcaster alongside
                      OmxGravityCompController.

Usage:
    # Variable-stiffness single robot
    python3 live_plot_logs.py --controller variable_stiffness \\
        --namespace /omx/variable_stiffness_controller

    # Variable-stiffness dual robot
    python3 live_plot_logs.py --controller variable_stiffness \\
        --namespace /robot1/robot1_variable_stiffness \\
        --namespace2 /robot2/robot2_variable_stiffness

    # Gravity-comp single robot
    python3 live_plot_logs.py --controller gravity_comp --namespace /omx

    # Gravity-comp dual robot
    python3 live_plot_logs.py --controller gravity_comp \\
        --namespace /robot1 --namespace2 /robot2

    # Tune rolling window and refresh rate
    python3 live_plot_logs.py --window 60 --interval 0.5

    # Via ROS2 launch file (args forwarded automatically)
    ros2 launch tools/launch/live_plot.launch.py
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

# ---------------------------------------------------------------------------
# matplotlib — must be configured before any other import uses it.
# ---------------------------------------------------------------------------
import matplotlib

def _choose_matplotlib_backend():
    # Use GUI backend when requested via env var, otherwise fall back to headless.
    use_gui = os.environ.get('LIVEPLOT_USE_GUI', '0') in ('1', 'true', 'True')
    display = os.environ.get('DISPLAY', '')

    if use_gui and display:
        for backend in ['TkAgg', 'Qt5Agg', 'QtAgg', 'GTK3Agg']:
            try:
                matplotlib.use(backend)
                print(f'[live_plot_logs] Using matplotlib GUI backend: {backend}')
                return
            except Exception as exc:
                print(f'[live_plot_logs] Failed to set backend {backend}: {exc}')
        print('[live_plot_logs] No GUI backend available; falling back to Agg.')
    else:
        if use_gui and not display:
            print('[live_plot_logs] LIVEPLOT_USE_GUI set but DISPLAY not set; falling back to Agg.')
        else:
            print('[live_plot_logs] using Agg backend (headless).')

    matplotlib.use('Agg')

_choose_matplotlib_backend()
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# ROS2 — required.
# ---------------------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Pose, Point, Vector3, WrenchStamped
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, Float64MultiArray, Bool
except ImportError as exc:
    print(f"Error: rclpy not available — is the ROS2 workspace sourced?\n  {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Import publication-quality helpers directly from plot_logs.py so that the
# style (rcParams, colours, grids, legends) is 100% identical.
# ---------------------------------------------------------------------------
_tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _tools_dir)
try:
    import plot_logs as pl
    from plot_logs import (
        apply_pub_style,
        _enable_minor_grid,
        _style_legend,
        _add_time_xlabel,
        _HC_PALETTE,
        ROBOT_COLORS,
        TIMESERIES_GROUPS,
        ALL_LABELS,
    )
except ImportError as exc:
    print(f"Error: could not import plot_logs.py: {exc}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Controller mode constants
# ---------------------------------------------------------------------------
MODE_VS = "variable_stiffness"
MODE_GC = "gravity_comp"

# ---------------------------------------------------------------------------
# Gravity-comp column groups (joint_state_broadcaster topics)
# ---------------------------------------------------------------------------
GC_POS_COLS = {
    "jpos1": "Joint 1 Position (rad)",
    "jpos2": "Joint 2 Position (rad)",
    "jpos3": "Joint 3 Position (rad)",
    "jpos4": "Joint 4 Position (rad)",
}
GC_VEL_COLS = {
    "jvel1": "Joint 1 Velocity (rad/s)",
    "jvel2": "Joint 2 Velocity (rad/s)",
    "jvel3": "Joint 3 Velocity (rad/s)",
    "jvel4": "Joint 4 Velocity (rad/s)",
}
GC_EFF_COLS = {
    "jeff1": "Joint 1 Grav-Comp Effort (Nm)",
    "jeff2": "Joint 2 Grav-Comp Effort (Nm)",
    "jeff3": "Joint 3 Grav-Comp Effort (Nm)",
    "jeff4": "Joint 4 Grav-Comp Effort (Nm)",
}
GC_TIMESERIES_GROUPS = [
    ("Joint Positions", GC_POS_COLS),
    ("Joint Velocities", GC_VEL_COLS),
    ("Gravity Compensation Efforts", GC_EFF_COLS),
]
GC_ALL_LABELS: Dict[str, str] = {}
for _d in [GC_POS_COLS, GC_VEL_COLS, GC_EFF_COLS]:
    GC_ALL_LABELS.update(_d)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_NS_VS  = "variable_stiffness_controller"
DEFAULT_NS_GC  = "omx"
DEFAULT_WINDOW_S   = 30.0
DEFAULT_INTERVAL_S = 0.5
MAX_SAMPLES        = 20_000


@dataclass(frozen=True)
class LiteReferenceLine:
    label: str
    value: float
    color: str
    linestyle: str = ":"


@dataclass(frozen=True)
class SpringInstabilityParams:
    l_0: float
    k_a: float
    b_eff: float
    k_theta_0: float
    chi: float
    k_lat: float


@dataclass(frozen=True)
class QoSSummary:
    reliability: ReliabilityPolicy
    durability: DurabilityPolicy
    history: HistoryPolicy
    depth: int


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def load_single_arm_spring_params_from_env() -> Optional[SpringInstabilityParams]:
    if os.environ.get("OMX_LIVEPLOT_SINGLE_ARM_SHOW_EXPECTED_DEPTHS", "1") not in ("1", "true", "True"):
        return None
    return SpringInstabilityParams(
        l_0=_env_float("OMX_LIVEPLOT_SINGLE_ARM_L_0", 0.128),
        k_a=_env_float("OMX_LIVEPLOT_SINGLE_ARM_K_A", 49.4),
        b_eff=_env_float("OMX_LIVEPLOT_SINGLE_ARM_B_EFF", 1.74e-3),
        k_theta_0=_env_float("OMX_LIVEPLOT_SINGLE_ARM_K_THETA_0", 0.00558),
        chi=_env_float("OMX_LIVEPLOT_SINGLE_ARM_CHI", 0.019),
        k_lat=_env_float("OMX_LIVEPLOT_SINGLE_ARM_K_LAT", 25.3),
    )


def _solve_monotonic_root(func, lo: float, hi: float, steps: int = 80) -> Optional[float]:
    if not (hi > lo):
        return None
    f_lo = func(lo)
    f_hi = func(hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return None
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None
    left, right = lo, hi
    for _ in range(steps):
        mid = 0.5 * (left + right)
        f_mid = func(mid)
        if not math.isfinite(f_mid):
            return None
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid <= 0:
            right = mid
            f_hi = f_mid
        else:
            left = mid
            f_lo = f_mid
    return 0.5 * (left + right)


def _expected_buckling_depth_m(params: SpringInstabilityParams) -> Optional[float]:
    max_depth = max(0.0, params.l_0 * 0.999)

    def margin_at_depth(depth: float) -> float:
        p = params.k_a * depth
        l = params.l_0 - depth
        if l <= 0:
            return -1e6
        return (math.pi * math.pi * params.b_eff) / (l * l) - p

    return _solve_monotonic_root(margin_at_depth, 0.0, max_depth)


def _expected_contact_rotation_depth_m(params: SpringInstabilityParams) -> Optional[float]:
    max_depth = max(0.0, params.l_0 * 0.999)

    def margin_at_depth(depth: float) -> float:
        p = params.k_a * depth
        l = params.l_0 - depth
        if l <= 0 or params.k_lat <= 0:
            return -1e6
        return params.k_theta_0 / l - p * params.chi * l - (p * p) / params.k_lat

    return _solve_monotonic_root(margin_at_depth, 0.0, max_depth)


def _expected_first_instability_depth_m(params: SpringInstabilityParams) -> tuple[Optional[float], Optional[str]]:
    buckle_depth = _expected_buckling_depth_m(params)
    rotation_depth = _expected_contact_rotation_depth_m(params)

    candidates: List[tuple[float, str]] = []
    if buckle_depth is not None:
        candidates.append((buckle_depth, "buckling"))
    if rotation_depth is not None:
        candidates.append((rotation_depth, "contact rotation"))

    if not candidates:
        return None, None

    depth, mode = min(candidates, key=lambda item: item[0])
    return depth, mode


def build_single_arm_lite_reference_lines() -> List[LiteReferenceLine]:
    """Return the default reference lines for the single-arm Gazebo layout."""
    return [
        LiteReferenceLine("robot base x", -0.39, "#7f7f7f"),
        LiteReferenceLine("platform edge x", -0.15, "#bcbd22"),
        LiteReferenceLine("platform top z", 0.05, "#9467bd"),
        LiteReferenceLine("wall face x", 0.131, "#d62728"),
        LiteReferenceLine("spring cap center x", 0.0865, "#2ca02c"),
    ]


# ---------------------------------------------------------------------------
# Thread-safe rolling buffer
# ---------------------------------------------------------------------------
class LiveBuffer:
    """Accumulates ROS2 callbacks into per-column deques (thread-safe)."""

    VS_COLUMNS: List[str] = [
        "time_s",
        "actual_x", "actual_y", "actual_z",
        "desired_x", "desired_y", "desired_z",
        "ee_x", "ee_y", "ee_z",
        "ee_roll", "ee_pitch", "ee_yaw",
        "ee_vx", "ee_vy", "ee_vz", "ee_wx", "ee_wy", "ee_wz",
        "jv1", "jv2", "jv3", "jv4",
        "tau1", "tau2", "tau3", "tau4",
        "Ktx", "Kty", "Ktz", "Krx", "Kry", "Krz",
        "Dtx", "Dty", "Dtz", "Drx", "Dry", "Drz",
        "manip_yoshikawa", "manip_sigma_min", "manip_sigma_max",
        "manip_condition_number",
        "contact_fx", "contact_fy", "contact_fz",
        "contact_tx", "contact_ty", "contact_tz",
        "contact_valid", "waypoint_active",
        "online_lateral_stiffness_npm",
        "online_lateral_probe_depth_m",
        "online_lateral_pair_depth_m",
        "online_lateral_pair_stiffness_npm",
    ]

    GC_COLUMNS: List[str] = [
        "time_s",
        "jpos1", "jpos2", "jpos3", "jpos4",
        "jvel1", "jvel2", "jvel3", "jvel4",
        "jeff1", "jeff2", "jeff3", "jeff4",
    ]

    def __init__(self, controller_mode: str, maxlen: int = MAX_SAMPLES) -> None:
        self._columns = (
            self.VS_COLUMNS if controller_mode == MODE_VS else self.GC_COLUMNS
        )
        self._lock: threading.Lock = threading.Lock()
        self._deques: Dict[str, Deque[float]] = {
            c: collections.deque(maxlen=maxlen) for c in self._columns
        }
        self._t0: Optional[float] = None

    def push(self, row: dict) -> None:
        t_now = time.monotonic()
        with self._lock:
            if self._t0 is None:
                self._t0 = t_now
            self._deques["time_s"].append(t_now - self._t0)
            for col in self._columns:
                if col == "time_s":
                    continue
                self._deques[col].append(row.get(col, float("nan")))

    def snapshot(self) -> Dict[str, list]:
        with self._lock:
            return {c: list(q) for c, q in self._deques.items()}

    def __len__(self) -> int:
        with self._lock:
            return len(self._deques["time_s"])


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------
class LivePlotNode(Node):
    """
    Subscribes to the appropriate topics for the selected controller mode.
    Snapshots at 50 Hz via timer into LiveBuffers.
    No files are read or written.
    """

    def __init__(
        self,
        buffers: Dict[int, LiveBuffer],
        namespaces: Dict[int, str],
        controller_mode: str,
        screenshot_dir: Optional[str] = None,
    ) -> None:
        super().__init__("live_plot_logs")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._buffers = buffers
        self._state: Dict[int, dict] = {rn: {} for rn in namespaces}
        self._controller_mode = controller_mode
        qos_custom = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._custom_summary_qos = QoSSummary(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._subscription_topics: List[str] = []
        self._subscriptions: List[object] = []
        self._marker_path = None
        self._debug_path = None
        self._stiffness_topic_debug_logged = False
        self._probe_depth_topic_debug_logged = False
        if screenshot_dir:
            try:
                os.makedirs(screenshot_dir, exist_ok=True)
                self._marker_path = os.path.join(screenshot_dir, '.live_plot_active')
                self._debug_path = os.path.join(screenshot_dir, '.live_plot_debug.txt')
                with open(self._marker_path, 'a') as handle:
                    handle.write(f'node:{os.getpid()}\n')
            except Exception:
                self._marker_path = None
                self._debug_path = None

        def _subscribe(*args, **kwargs):
            subscription = self.create_subscription(*args, **kwargs)
            self._subscriptions.append(subscription)
            return subscription

        for rn, ns in namespaces.items():
            base = ns.rstrip("/")
            r = rn  # capture loop variable

            if controller_mode == MODE_GC:
                # gravity_comp: subscribe to joint_states from joint_state_broadcaster
                _subscribe(
                    JointState, f"{base}/joint_states",
                    lambda m, _r=r: self._joint_states_cb(m, _r), qos)
                self.get_logger().info(
                    f"[Robot {rn}] subscribed to {base}/joint_states (gravity_comp)")
            else:
                # variable_stiffness: subscribe to all detailed controller topics
                _subscribe(
                    Pose, f"{base}/cartesian_pose_actual",
                    lambda m, _r=r: self._actual_pose_cb(m, _r), qos)
                _subscribe(
                    Pose, f"{base}/cartesian_pose_desired",
                    lambda m, _r=r: self._desired_pose_cb(m, _r), qos)
                _subscribe(
                    Point, f"{base}/end_effector_position",
                    lambda m, _r=r: self._ee_pos_cb(m, _r), qos)
                _subscribe(
                    Vector3, f"{base}/end_effector_orientation",
                    lambda m, _r=r: self._ee_orient_cb(m, _r), qos)
                _subscribe(
                    Float64MultiArray, f"{base}/end_effector_velocities",
                    lambda m, _r=r: self._ee_vel_cb(m, _r), qos)
                _subscribe(
                    Float64MultiArray, f"{base}/joint_velocities",
                    lambda m, _r=r: self._joint_vel_cb(m, _r), qos)
                _subscribe(
                    Float64MultiArray, f"{base}/torque_values",
                    lambda m, _r=r: self._torque_cb(m, _r), qos)
                _subscribe(
                    Float64MultiArray, f"{base}/stiffness_state",
                    lambda m, _r=r: self._stiffness_cb(m, _r), qos)
                _subscribe(
                    Float64MultiArray, f"{base}/manipulability_metrics",
                    lambda m, _r=r: self._manip_cb(m, _r), qos)
                _subscribe(
                    WrenchStamped, f"{base}/contact_wrench",
                    lambda m, _r=r: self._contact_wrench_cb(m, _r), qos)
                _subscribe(
                    Bool, f"{base}/contact_valid",
                    lambda m, _r=r: self._contact_valid_cb(m, _r), qos)
                _subscribe(
                    Bool, f"{base}/waypoint_active",
                    lambda m, _r=r: self._waypoint_active_cb(m, _r), qos)
                stiffness_topic = f"{base}/online_lateral_stiffness_npm"
                depth_topic = f"{base}/online_lateral_probe_depth_m"
                pair_topic = f"{base}/online_lateral_pair"
                self._subscription_topics.extend([stiffness_topic, depth_topic, pair_topic])
                self.get_logger().info(f"[Robot {rn}] subscribing to {stiffness_topic}")
                _subscribe(
                    Float64, stiffness_topic,
                    lambda m, _r=r: self._online_lateral_stiffness_cb(m, _r), qos_custom)
                self.get_logger().info(f"[Robot {rn}] subscribing to {depth_topic}")
                _subscribe(
                    Float64, depth_topic,
                    lambda m, _r=r: self._online_lateral_probe_depth_cb(m, _r), qos_custom)
                self.get_logger().info(f"[Robot {rn}] subscribing to {pair_topic}")
                _subscribe(
                    Float64MultiArray, pair_topic,
                    lambda m, _r=r: self._online_lateral_pair_cb(m, _r), qos_custom)
                self.get_logger().info(
                    f"[Robot {rn}] custom summary qos reliability={self._custom_summary_qos.reliability} durability={self._custom_summary_qos.durability} history={self._custom_summary_qos.history} depth={self._custom_summary_qos.depth}"
                )
                self.get_logger().info(
                    f"[Robot {rn}] subscribed to {base}/* (variable_stiffness)")

        self.create_timer(0.02, self._snapshot_timer)

    # --- gravity comp callback ---

    def _joint_states_cb(self, msg: JointState, r: int) -> None:
        n = min(4, len(msg.position))
        for i in range(n):
            self._state[r][f"jpos{i+1}"] = msg.position[i] if i < len(msg.position) else float("nan")
            self._state[r][f"jvel{i+1}"] = msg.velocity[i] if i < len(msg.velocity) else float("nan")
            self._state[r][f"jeff{i+1}"] = msg.effort[i]   if i < len(msg.effort)   else float("nan")

    # --- variable stiffness callbacks ---

    def _actual_pose_cb(self, msg: Pose, r: int) -> None:
        p = msg.position
        self._state[r].update(actual_x=p.x, actual_y=p.y, actual_z=p.z)

    def _desired_pose_cb(self, msg: Pose, r: int) -> None:
        p = msg.position
        self._state[r].update(desired_x=p.x, desired_y=p.y, desired_z=p.z)

    def _ee_pos_cb(self, msg: Point, r: int) -> None:
        self._state[r].update(ee_x=msg.x, ee_y=msg.y, ee_z=msg.z)

    def _ee_orient_cb(self, msg: Vector3, r: int) -> None:
        self._state[r].update(ee_roll=msg.x, ee_pitch=msg.y, ee_yaw=msg.z)

    def _ee_vel_cb(self, msg: Float64MultiArray, r: int) -> None:
        d = msg.data
        keys = ["ee_vx", "ee_vy", "ee_vz", "ee_wx", "ee_wy", "ee_wz"]
        self._state[r].update({k: d[i] for i, k in enumerate(keys) if i < len(d)})

    def _joint_vel_cb(self, msg: Float64MultiArray, r: int) -> None:
        d = msg.data
        self._state[r].update({f"jv{i+1}": d[i] for i in range(min(4, len(d)))})

    def _torque_cb(self, msg: Float64MultiArray, r: int) -> None:
        d = msg.data
        self._state[r].update({f"tau{i+1}": d[i] for i in range(min(4, len(d)))})

    def _stiffness_cb(self, msg: Float64MultiArray, r: int) -> None:
        d = msg.data
        keys = ["Ktx", "Kty", "Ktz", "Krx", "Kry", "Krz",
                "Dtx", "Dty", "Dtz", "Drx", "Dry", "Drz"]
        self._state[r].update({k: d[i] for i, k in enumerate(keys) if i < len(d)})

    def _manip_cb(self, msg: Float64MultiArray, r: int) -> None:
        d = msg.data
        keys = ["manip_condition_number", "manip_det", "manip_yoshikawa",
                "manip_sigma_min", "manip_sigma_max"]
        self._state[r].update({k: d[i] for i, k in enumerate(keys) if i < len(d)})

    def _contact_wrench_cb(self, msg: WrenchStamped, r: int) -> None:
        f, t = msg.wrench.force, msg.wrench.torque
        self._state[r].update(
            contact_fx=f.x, contact_fy=f.y, contact_fz=f.z,
            contact_tx=t.x, contact_ty=t.y, contact_tz=t.z,
        )

    def _contact_valid_cb(self, msg: Bool, r: int) -> None:
        self._state[r]["contact_valid"] = float(msg.data)

    def _waypoint_active_cb(self, msg: Bool, r: int) -> None:
        self._state[r]["waypoint_active"] = float(msg.data)

    def _online_lateral_stiffness_cb(self, msg: Float64, r: int) -> None:
        self._state[r]["online_lateral_stiffness_npm"] = float(msg.data)
        if not self._stiffness_topic_debug_logged:
            self._stiffness_topic_debug_logged = True
            self.get_logger().info(
                f"[Robot {r}] received online_lateral_stiffness_npm={float(msg.data):.6f}"
            )
            if self._debug_path:
                try:
                    with open(self._debug_path, 'a') as handle:
                        handle.write(f'callback stiffness robot={r} value={float(msg.data):.6f}\n')
                except Exception:
                    pass
        elif self._debug_path:
            try:
                with open(self._debug_path, 'a') as handle:
                    handle.write(f'callback stiffness robot={r} value={float(msg.data):.6f}\n')
            except Exception:
                pass

    def _online_lateral_probe_depth_cb(self, msg: Float64, r: int) -> None:
        self._state[r]["online_lateral_probe_depth_m"] = float(msg.data)
        if not self._probe_depth_topic_debug_logged:
            self._probe_depth_topic_debug_logged = True
            self.get_logger().info(
                f"[Robot {r}] received online_lateral_probe_depth_m={float(msg.data):.6f}"
            )
            if self._debug_path:
                try:
                    with open(self._debug_path, 'a') as handle:
                        handle.write(f'callback depth robot={r} value={float(msg.data):.6f}\n')
                except Exception:
                    pass
        elif self._debug_path:
            try:
                with open(self._debug_path, 'a') as handle:
                    handle.write(f'callback depth robot={r} value={float(msg.data):.6f}\n')
            except Exception:
                pass

    def _online_lateral_pair_cb(self, msg: Float64MultiArray, r: int) -> None:
        data = list(msg.data)
        depth_val = float(data[0]) if len(data) >= 1 else float("nan")
        stiffness_val = float(data[1]) if len(data) >= 2 else float("nan")
        self.get_logger().info(
            f"PAIR CALLBACK robot={r} data={data} len={len(data)} depth={depth_val:.6f} stiffness={stiffness_val:.6f} finite={math.isfinite(depth_val)} {math.isfinite(stiffness_val)}"
        )
        if len(data) < 2:
            return

        self._state[r]["online_lateral_pair_depth_m"] = depth_val
        self._state[r]["online_lateral_pair_stiffness_npm"] = stiffness_val
        self._state[r]["online_lateral_probe_depth_m"] = depth_val
        self._state[r]["online_lateral_stiffness_npm"] = stiffness_val
        self.get_logger().info(f"[Robot {r}] pair state={self._state[r]}")

        if self._debug_path:
            try:
                with open(self._debug_path, 'a') as handle:
                    handle.write(
                        f'callback pair robot={r} depth={depth_val:.6f} stiffness={stiffness_val:.6f}\n'
                    )
            except Exception:
                pass

    def _snapshot_timer(self) -> None:
        for rn, state in self._state.items():
            if state:
                if self._debug_path:
                    try:
                        with open(self._debug_path, 'a') as handle:
                            handle.write(
                                f"snapshot robot={rn} depth={float(state.get('online_lateral_pair_depth_m', float('nan'))):.6f} stiffness={float(state.get('online_lateral_pair_stiffness_npm', float('nan'))):.6f}\n"
                            )
                    except Exception:
                        pass
                self._buffers[rn].push(dict(state))


def _maximize_figure(fig) -> None:
    try:
        if plt.get_backend().lower().endswith("agg"):
            return
        manager = getattr(fig.canvas, "manager", None)
        if manager is None:
            return
        window = getattr(manager, "window", None)
        if window is None:
            return
        try:
            window.state("zoomed")
        except Exception:
            pass
        try:
            window.attributes("-fullscreen", True)
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Live figure manager
# ---------------------------------------------------------------------------
class LiveFigureManager:
    """
    Single figure with one subplot per timeseries group.  All columns in a
    group share one axes (overlaid lines, coloured per-robot × per-column).
    Fits on screen regardless of mode.
    """

    def __init__(
        self,
        robot_nums: List[int],
        window_s: float,
        robot_count: int,
        controller_mode: str,
        screenshot_dir: Optional[str] = None,
        screenshot_rate: float = 2.0,
    ) -> None:
        self._robot_nums    = robot_nums
        self._window_s      = window_s
        self._robot_count   = robot_count
        self._controller_mode = controller_mode
        self._screenshot_dir = screenshot_dir
        self._screenshot_rate = float(screenshot_rate)
        self._last_screenshot = 0.0

        # If screenshots are requested, ensure the directory exists and create
        # a small marker file so external monitors can detect the live-plot
        # process even if PNG creation is delayed or failing.
        self._marker_path = None
        self._debug_path = None
        if self._screenshot_dir:
            try:
                os.makedirs(self._screenshot_dir, exist_ok=True)
                self._marker_path = os.path.join(self._screenshot_dir, '.live_plot_active')
                self._debug_path = os.path.join(self._screenshot_dir, '.live_plot_debug.txt')
                with open(self._marker_path, 'w') as mf:
                    mf.write(f'pid:{os.getpid()}\nstart:{int(time.time())}\n')
            except Exception as exc:
                print(f'[live_plot_logs] Failed to create screenshot dir/marker: {exc}')

        # Select the right group/label dictionaries
        if controller_mode == MODE_GC:
            self._timeseries_groups = GC_TIMESERIES_GROUPS
            self._all_labels        = GC_ALL_LABELS
        else:
            self._timeseries_groups = TIMESERIES_GROUPS
            self._all_labels        = ALL_LABELS

        apply_pub_style()  # identical rcParams to plot_logs.py

        self._figs: List = []
        self._axes_flat: List = []
        # lines[group_idx][(rn, col)] = Line2D
        self._lines: Dict[int, Dict[tuple, object]] = {}
        self._built = False

    def _maximize_figures(self) -> None:
        for fig in self._figs:
            _maximize_figure(fig)

    # ---- one-time construction ------------------------------------------
    def _build(self) -> None:
        n_groups = len(self._timeseries_groups)
        mode_label = (
            "Gravity Compensation" if self._controller_mode == MODE_GC
            else "Variable Stiffness"
        )
        robot_label = "Dual-Robot" if self._robot_count > 1 else "Single-Robot"

        # Split the groups across up to 3 separate figures for readability.
        num_plots = min(3, n_groups) if n_groups > 0 else 1
        groups_per_plot = (n_groups + num_plots - 1) // num_plots

        # Compact font sizes for the combined view
        tick_size = max(8, 14 - n_groups // 4)

        self._figs = []
        self._axes_flat = []

        for p in range(num_plots):
            start = p * groups_per_plot
            end = min(start + groups_per_plot, n_groups)
            n_sub = max(1, end - start)

            fig, axes = plt.subplots(
                n_sub, 1,
                figsize=(18, max(2.4 * n_sub, 4)),
                sharex=True, squeeze=False,
            )
            axes_list = axes.flatten()
            self._figs.append(fig)

            # Title only on the first figure
            if p == 0:
                fig.suptitle(
                    f"{robot_label} {mode_label} — Live",
                    fontsize=18, fontweight="bold", y=0.99,
                )

            for local_idx, g_idx in enumerate(range(start, end)):
                group_title, col_map = self._timeseries_groups[g_idx]
                ax = axes_list[local_idx]
                ax.set_ylabel(group_title, fontsize=max(8, 13 - n_groups // 5),
                              fontweight="bold")
                ax.tick_params(labelsize=tick_size)
                _enable_minor_grid(ax)
                self._lines[g_idx] = {}

                col_names = list(col_map.keys())
                col_labels = list(col_map.values())

                for rn in self._robot_nums:
                    robot_prefix = "" if self._robot_count == 1 else f"R{rn} "
                    for c_idx, col in enumerate(col_names):
                        color = _HC_PALETTE[c_idx % len(_HC_PALETTE)]
                        ls = "-" if rn == 1 else "--"
                        lw = 1.8 if self._robot_count == 1 else 1.4
                        short_label = col_labels[c_idx].split("(")[0].strip()
                        lbl = f"{robot_prefix}{short_label}"
                        (line,) = ax.plot([], [], color=color, linewidth=lw,
                                         linestyle=ls, label=lbl)
                        self._lines[g_idx][(rn, col)] = line

                ax.legend(fontsize=max(6, 9 - n_groups // 5), loc="upper left",
                          ncol=max(1, len(col_map)),
                          framealpha=0.7, borderpad=0.3, handlelength=1.5)

                self._axes_flat.append(ax)

            # Layout per figure
            fig.tight_layout(rect=[0, 0.01, 1.0, 0.96])
            fig.subplots_adjust(hspace=0.35)

        # Add time xlabel to the last axis of the last figure
        if self._axes_flat:
            _add_time_xlabel(self._axes_flat[-1])
            self._axes_flat[-1].tick_params(labelsize=tick_size)

        self._maximize_figures()
        self._built = True

    # ---- per-cycle refresh ----------------------------------------------
    def update(self, snapshots: Dict[int, dict]) -> None:
        if not self._built:
            if all(len(s.get("time_s", [])) > 1 for s in snapshots.values()):
                self._build()
            else:
                return

        for g_idx, (_group_title, col_map) in enumerate(self._timeseries_groups):
            ax = self._axes_flat[g_idx]
            for col in col_map:
                for rn in self._robot_nums:
                    snap   = snapshots.get(rn, {})
                    ts_all = snap.get("time_s", [])
                    ys_all = snap.get(col, [])
                    if not ts_all or not ys_all:
                        continue
                    n = min(len(ts_all), len(ys_all))
                    ts_all = ts_all[:n]
                    ys_all = ys_all[:n]
                    t_end   = ts_all[-1]
                    t_start = t_end - self._window_s
                    offset = 0
                    for offset in range(n):
                        if ts_all[offset] >= t_start:
                            break
                    ts = ts_all[offset:]
                    ys = ys_all[offset:]
                    line = self._lines[g_idx].get((rn, col))
                    if line is not None:
                        line.set_data(ts, ys)
            ax.relim()
            ax.autoscale_view()

        # Draw each figure
        for fig in self._figs:
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

        # In headless mode or when user requested it, save snapshot images
        if self._screenshot_dir:
            now = time.time()
            if now - self._last_screenshot >= self._screenshot_rate:
                if not os.path.isdir(self._screenshot_dir):
                    os.makedirs(self._screenshot_dir, exist_ok=True)
                for i, fig in enumerate(self._figs, start=1):
                    fname = os.path.join(
                        self._screenshot_dir,
                        f"live_plot_{i}_{int(now)}.png"
                    )
                    try:
                        fig.savefig(fname, dpi=120)
                    except Exception as exc:
                        print(f'[live_plot_logs] Failed to save screenshot {fname}: {exc}')
                # update marker file timestamp so external monitors see activity
                if getattr(self, '_marker_path', None):
                    try:
                        with open(self._marker_path, 'a') as mf:
                            mf.write(f'ts:{int(now)}\n')
                    except Exception:
                        pass
                self._last_screenshot = now
        plt.pause(0.001)


# ---------------------------------------------------------------------------
# Lite live figure manager
# ---------------------------------------------------------------------------
class LiteLiveFigureManager:
    """Single-figure live plot with force, position, velocity, and stiffness subplots."""

    def __init__(
        self,
        robot_nums: List[int],
        window_s: float,
        robot_count: int,
        controller_mode: str,
        screenshot_dir: Optional[str] = None,
        screenshot_rate: float = 2.0,
        reference_lines: Optional[List[LiteReferenceLine]] = None,
    ) -> None:
        self._robot_nums = robot_nums
        self._window_s = window_s
        self._robot_count = robot_count
        self._controller_mode = controller_mode
        self._screenshot_dir = screenshot_dir
        self._screenshot_rate = float(screenshot_rate)
        self._last_screenshot = 0.0
        self._reference_lines = reference_lines or build_single_arm_lite_reference_lines()
        self._spring_params = load_single_arm_spring_params_from_env()
        self._contact_anchor_x: Optional[float] = None
        self._expected_depth_line = None
        self._expected_depth_label: Optional[str] = None
        self._stiffness_latest_text = None
        self._stiffness_debug_logged = False
        self._stiffness_topic_debug_logged = False
        self._probe_depth_topic_debug_logged = False

        self._marker_path = None
        self._debug_path = None
        if self._screenshot_dir:
            try:
                os.makedirs(self._screenshot_dir, exist_ok=True)
                self._marker_path = os.path.join(self._screenshot_dir, '.live_plot_active')
                self._debug_path = os.path.join(self._screenshot_dir, '.live_plot_debug.txt')
                with open(self._marker_path, 'w') as mf:
                    mf.write(f'pid:{os.getpid()}\nstart:{int(time.time())}\n')
            except Exception as exc:
                print(f'[live_plot_logs] Failed to create screenshot dir/marker: {exc}')

        apply_pub_style()

        self._fig = None
        self._axes = []
        self._lines: Dict[str, Dict[tuple, object]] = {"force": {}, "position": {}, "velocity": {}, "stiffness": {}}
        self._built = False

    def _build(self) -> None:
        mode_label = (
            "Gravity Compensation" if self._controller_mode == MODE_GC
            else "Variable Stiffness"
        )
        robot_label = "Dual-Robot" if self._robot_count > 1 else "Single-Robot"

        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=False, squeeze=False)
        self._fig = fig
        self._axes = list(axes.flatten())

        # The first three panels are timeseries plots and should share the
        # common time axis. The stiffness-vs-depth panel uses a different x
        # semantics, so it must remain independent.
        self._axes[1].sharex(self._axes[0])
        self._axes[2].sharex(self._axes[0])

        fig.suptitle(
            f"{robot_label} {mode_label} — Lite",
            fontsize=18,
            fontweight="bold",
            y=0.98,
        )

        force_ax, pos_ax, vel_ax, stiff_ax = self._axes
        label_size = 11
        force_ax.set_ylabel("Force (N)", fontweight="bold", fontsize=label_size, labelpad=4)
        pos_ax.set_ylabel("Position (m)", fontweight="bold", fontsize=label_size, labelpad=4)
        vel_ax.set_ylabel("Velocity (m/s)", fontweight="bold", fontsize=label_size, labelpad=4)
        stiff_ax.set_ylabel("K_lat (N/m)", fontweight="bold", fontsize=label_size, labelpad=4)
        vel_ax.set_xlabel("Time (s)", fontsize=8, labelpad=2)
        stiff_ax.set_xlabel("Compression depth (m)", fontsize=8, labelpad=2)

        force_ax.tick_params(labelbottom=False)
        pos_ax.tick_params(labelbottom=False)

        for ax in self._axes:
            _enable_minor_grid(ax)
            ax.tick_params(labelsize=9)

        force_cols = [("contact_fx", "Fx"), ("contact_fy", "Fy"), ("contact_fz", "Fz")]
        pos_cols = [
            ("actual_x", "actual x"), ("actual_y", "actual y"), ("actual_z", "actual z"),
            ("desired_x", "desired x"), ("desired_y", "desired y"), ("desired_z", "desired z"),
        ]
        vel_cols = [("ee_vx", "vx"), ("ee_vy", "vy"), ("ee_vz", "vz")]

        component_colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}

        for col, short_label in force_cols:
            color = component_colors[col[-1]]
            (line,) = force_ax.plot([], [], color=color, linewidth=1.8, label=f"EE {short_label}")
            self._lines["force"][(1, col)] = line

        for col, short_label in pos_cols:
            color = component_colors[col[-1]]
            linestyle = "-" if col.startswith("actual_") else "--"
            linewidth = 1.8 if col.startswith("actual_") else 1.4
            (line,) = pos_ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=short_label)
            self._lines["position"][(1, col)] = line

        for ref in self._reference_lines:
            pos_ax.axhline(
                ref.value,
                color=ref.color,
                linestyle=ref.linestyle,
                linewidth=1.0,
                alpha=0.85,
                label=ref.label,
            )

        if self._spring_params is not None:
            (instability_line,) = pos_ax.plot(
                [], [],
                color="#111111",
                linewidth=2.2,
                linestyle=(0, (5, 2)),
                label="expected instability depth",
            )
            self._expected_depth_line = instability_line

        for col, short_label in vel_cols:
            color = component_colors[col[-1]]
            (line,) = vel_ax.plot([], [], color=color, linewidth=1.8, label=f"EE {short_label}")
            self._lines["velocity"][(1, col)] = line

        for rn in self._robot_nums:
            robot_prefix = "" if self._robot_count == 1 else f"R{rn} "
            (line,) = stiff_ax.plot(
                [], [],
                color="#9467bd",
                linewidth=2.6,
                linestyle="-",
                marker="o",
                markersize=7.0,
                markerfacecolor="#9467bd",
                markeredgecolor="#5b3f7f",
                markeredgewidth=0.8,
                zorder=5,
                label=f"{robot_prefix}online K_lat vs depth",
            )
            self._lines["stiffness"][(rn, "online_lateral_stiffness_npm")] = line

        self._stiffness_latest_text = stiff_ax.text(
            0.98,
            0.92,
            "waiting for stiffness samples",
            transform=stiff_ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#5b3f7f",
            bbox={"facecolor": "#f3edf9", "edgecolor": "#9467bd", "boxstyle": "round,pad=0.25", "alpha": 0.95},
        )

        force_ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.7, borderpad=0.3, handlelength=1.5)
        pos_ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.7, borderpad=0.3, handlelength=1.5)
        vel_ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.7, borderpad=0.3, handlelength=1.5)
        stiff_ax.legend(fontsize=8, loc="upper left", ncol=1, framealpha=0.7, borderpad=0.3, handlelength=1.5)

        fig.tight_layout(rect=[0.04, 0.01, 1.0, 0.96])
        fig.subplots_adjust(left=0.11, right=0.995, bottom=0.07, hspace=0.45)
        _maximize_figure(fig)
        self._built = True

    def _update_axis(self, axis, line_map: Dict[tuple, object], snapshots: Dict[int, dict], series: List[str]) -> None:
        for col in series:
            for rn in self._robot_nums:
                snap = snapshots.get(rn, {})
                ts_all = snap.get("time_s", [])
                ys_all = snap.get(col, [])
                if not ts_all or not ys_all:
                    continue
                n = min(len(ts_all), len(ys_all))
                ts_all = ts_all[:n]
                ys_all = ys_all[:n]
                t_end = ts_all[-1]
                t_start = t_end - self._window_s
                offset = 0
                for offset in range(n):
                    if ts_all[offset] >= t_start:
                        break
                ts = ts_all[offset:]
                ys = ys_all[offset:]
                line = line_map.get((rn, col))
                if line is not None:
                    line.set_data(ts, ys)
        axis.relim()
        axis.autoscale_view()

    def _update_stiffness_depth_axis(self, snapshots: Dict[int, dict]) -> None:
        axis = self._axes[3]
        for rn in self._robot_nums:
            snap = snapshots.get(rn, {})
            pair_depth_all = snap.get("online_lateral_pair_depth_m", [])
            pair_stiffness_all = snap.get("online_lateral_pair_stiffness_npm", [])
            stiffness_all = snap.get("online_lateral_stiffness_npm", [])
            depth_all = snap.get("online_lateral_probe_depth_m", [])
            depth_vals: List[float] = []
            stiffness_vals: List[float] = []
            used_pair_stream = False

            if pair_depth_all and pair_stiffness_all:
                for depth_val, stiffness_val in zip(pair_depth_all, pair_stiffness_all):
                    depth_float = float(depth_val)
                    stiffness_float = float(stiffness_val)
                    if math.isfinite(depth_float) and depth_float >= 0.0 and math.isfinite(stiffness_float):
                        depth_vals.append(depth_float)
                        stiffness_vals.append(stiffness_float)
                used_pair_stream = bool(depth_vals)

            if not depth_vals and stiffness_all and depth_all:
                n = min(len(depth_all), len(stiffness_all))
                latest_depth: Optional[float] = None
                latest_stiffness: Optional[float] = None
                for idx in range(n):
                    depth_val = depth_all[idx]
                    stiffness_val = stiffness_all[idx]
                    if math.isfinite(float(depth_val)) and float(depth_val) >= 0.0:
                        latest_depth = float(depth_val)
                    if math.isfinite(float(stiffness_val)):
                        latest_stiffness = float(stiffness_val)
                    if latest_depth is None or latest_stiffness is None:
                        continue
                    depth_vals.append(latest_depth)
                    stiffness_vals.append(latest_stiffness)

            if depth_vals and stiffness_vals:
                compressed_depth_vals: List[float] = []
                compressed_stiffness_vals: List[float] = []
                for depth_val, stiffness_val in zip(depth_vals, stiffness_vals):
                    if compressed_depth_vals and abs(depth_val - compressed_depth_vals[-1]) <= 1e-9:
                        compressed_stiffness_vals[-1] = stiffness_val
                        continue
                    compressed_depth_vals.append(depth_val)
                    compressed_stiffness_vals.append(stiffness_val)
                ordered_pairs = sorted(zip(compressed_depth_vals, compressed_stiffness_vals), key=lambda pair: pair[0])
                if len(ordered_pairs) == 1:
                    depth_val, stiffness_val = ordered_pairs[0]
                    depth_step = max(depth_val * 0.05, 0.0005)
                    ordered_pairs = [
                        (depth_val, stiffness_val),
                        (depth_val + depth_step, stiffness_val),
                    ]
                if ordered_pairs:
                    compressed_depth_vals = [pair[0] for pair in ordered_pairs]
                    compressed_stiffness_vals = [pair[1] for pair in ordered_pairs]
                depth_vals = compressed_depth_vals
                stiffness_vals = compressed_stiffness_vals

            line = self._lines["stiffness"].get((rn, "online_lateral_stiffness_npm"))
            if line is not None:
                line.set_data(depth_vals, stiffness_vals)
            if depth_vals and not self._stiffness_debug_logged:
                self._stiffness_debug_logged = True
                print(
                    f"[live_plot_logs] [Robot {rn}] stiffness plot samples={len(depth_vals)} depth0={depth_vals[0]:.6f} K0={stiffness_vals[0]:.6f} source={'pair' if used_pair_stream else 'carry-forward'}"
                )
                if self._debug_path:
                    try:
                        with open(self._debug_path, 'a') as handle:
                            handle.write(
                                f"robot={rn} samples={len(depth_vals)} depth0={depth_vals[0]:.6f} K0={stiffness_vals[0]:.6f} source={'pair' if used_pair_stream else 'carry-forward'}\n"
                            )
                    except Exception:
                        pass

            if self._stiffness_latest_text is not None:
                if depth_vals and stiffness_vals:
                    latest_text = f"latest: depth={depth_vals[-1]:.4f} m\nK_lat={stiffness_vals[-1]:.4f} N/m"
                else:
                    latest_text = (
                        f"raw samples: depth={len(depth_all)} K_lat={len(stiffness_all)} pair={len(pair_depth_all)}\n"
                        "awaiting finite pair"
                    )
                self._stiffness_latest_text.set_text(latest_text)
                self._stiffness_latest_text.set_visible(True)

        if depth_vals and stiffness_vals:
            depth_min = min(depth_vals)
            depth_max = max(depth_vals)
            stiffness_min = min(stiffness_vals)
            stiffness_max = max(stiffness_vals)
            depth_span = max(depth_max - depth_min, 1e-4)
            stiffness_span = max(stiffness_max - stiffness_min, 1e-4)
            depth_pad = max(depth_span * 0.15, 0.0005)
            stiffness_pad = max(stiffness_span * 0.20, 0.0005)
            left = max(0.0, depth_min - depth_pad)
            right = max(left + 0.001, depth_max + depth_pad)
            bottom = max(0.0, stiffness_min - stiffness_pad)
            top = max(bottom + 0.001, stiffness_max + stiffness_pad)
            axis.set_xlim(left=left, right=right)
            axis.set_ylim(bottom=bottom, top=top)
        else:
            axis.relim()
            axis.autoscale_view()
            left, right = axis.get_xlim()
            bottom, top = axis.get_ylim()
            if not math.isfinite(left) or left < 0.0:
                left = 0.0
            if not math.isfinite(bottom) or bottom < 0.0:
                bottom = 0.0
            if not math.isfinite(right) or right <= left:
                right = left + 0.001
            if not math.isfinite(top) or top <= bottom:
                top = bottom + 0.001
            axis.set_xlim(left=left, right=right)
            axis.set_ylim(bottom=bottom, top=top)

    def update(self, snapshots: Dict[int, dict]) -> None:
        if not self._built:
            if all(len(s.get("time_s", [])) > 1 for s in snapshots.values()):
                self._build()
            else:
                return

        self._update_axis(self._axes[0], self._lines["force"], snapshots, ["contact_fx", "contact_fy", "contact_fz"])
        self._update_axis(
            self._axes[1],
            self._lines["position"],
            snapshots,
            ["actual_x", "actual_y", "actual_z", "desired_x", "desired_y", "desired_z"],
        )
        self._update_axis(self._axes[2], self._lines["velocity"], snapshots, ["ee_vx", "ee_vy", "ee_vz"])
        self._update_stiffness_depth_axis(snapshots)

        if self._spring_params is not None and self._axes:
            self._update_expected_depth_overlays(snapshots)

        if self._fig is not None:
            try:
                self._fig.canvas.draw_idle()
            except Exception:
                pass

        if self._screenshot_dir:
            now = time.time()
            if now - self._last_screenshot >= self._screenshot_rate:
                if not os.path.isdir(self._screenshot_dir):
                    os.makedirs(self._screenshot_dir, exist_ok=True)
                fname = os.path.join(self._screenshot_dir, f"live_plot_lite_{int(now)}.png")
                try:
                    self._fig.savefig(fname, dpi=120)
                except Exception as exc:
                    print(f'[live_plot_logs] Failed to save screenshot {fname}: {exc}')
                if getattr(self, '_marker_path', None):
                    try:
                        with open(self._marker_path, 'a') as mf:
                            mf.write(f'ts:{int(now)}\n')
                    except Exception:
                        pass
                self._last_screenshot = now
        plt.pause(0.001)

    def _update_expected_depth_overlays(self, snapshots: Dict[int, dict]) -> None:
        if self._contact_anchor_x is None:
            for rn in self._robot_nums:
                snap = snapshots.get(rn, {})
                if not snap:
                    continue
                contact_valid = snap.get("contact_valid", [])
                if contact_valid and float(contact_valid[-1]) >= 1.0:
                    actual_x = snap.get("actual_x", [])
                    desired_x = snap.get("desired_x", [])
                    if actual_x:
                        self._contact_anchor_x = float(actual_x[-1])
                    elif desired_x:
                        self._contact_anchor_x = float(desired_x[-1])
                    if self._contact_anchor_x is not None:
                        break

        if self._contact_anchor_x is None:
            return

        depth_m, mode = _expected_first_instability_depth_m(self._spring_params)
        if depth_m is None or self._expected_depth_line is None:
            return

        expected_x = self._contact_anchor_x + depth_m
        self._expected_depth_line.set_data([0.0, self._window_s], [expected_x, expected_x])

        if mode != self._expected_depth_label:
            if mode == "buckling":
                color = "#ff7f0e"
                label = "expected instability depth (buckling)"
            else:
                color = "#1f77b4"
                label = "expected instability depth (contact rotation)"
            self._expected_depth_line.set_color(color)
            self._expected_depth_line.set_label(label)
            self._expected_depth_label = mode

            legend = self._axes[1].legend(
                fontsize=8,
                loc="upper left",
                ncol=2,
                framealpha=0.8,
                borderpad=0.35,
                handlelength=1.6,
            )
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color("#111111")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live timeseries plotter — subscribes to ROS2 topics, "
                    "no log files read or written.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--controller", "-c",
        choices=[MODE_VS, MODE_GC],
        default=MODE_VS,
        help=f"Controller mode: '{MODE_VS}' or '{MODE_GC}' (default: {MODE_VS})",
    )
    p.add_argument(
        "--namespace", "-n",
        default=None,
        help=(
            f"Topic namespace for robot 1. "
            f"VS default: {DEFAULT_NS_VS}  |  GC default: {DEFAULT_NS_GC}"
        ),
    )
    p.add_argument(
        "--namespace2", default=None,
        help="Topic namespace for robot 2 (enables dual-robot mode)",
    )
    p.add_argument(
        "--window", type=float, default=DEFAULT_WINDOW_S,
        help=f"Rolling time window in seconds (default: {DEFAULT_WINDOW_S})",
    )
    p.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL_S,
        help=f"Plot refresh interval in seconds (default: {DEFAULT_INTERVAL_S})",
    )
    p.add_argument(
        "--screenshot-dir", default=None,
        help="Directory to save periodic plot screenshots (enables non-interactive mode)",
    )
    p.add_argument(
        "--screenshot-rate", type=float, default=2.0,
        help="Seconds between screenshots when screenshot-dir is set",
    )
    p.add_argument(
        "--layout",
        choices=["full", "lite"],
        default="full",
        help="Plot layout: full multi-figure view or lite single-window view",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_cli().parse_args()

    print(
        f"[live_plot_logs] script={os.path.abspath(__file__)} pid={os.getpid()} host={os.uname().nodename} start={int(time.time())}"
    )

    controller_mode = args.controller

    # Apply mode-aware namespace defaults
    if args.namespace is None:
        args.namespace = DEFAULT_NS_VS if controller_mode == MODE_VS else DEFAULT_NS_GC

    namespaces: Dict[int, str] = {1: args.namespace}
    if args.namespace2 and args.namespace2.strip():
        namespaces[2] = args.namespace2

    robot_nums  = list(namespaces.keys())
    robot_count = len(robot_nums)

    buffers: Dict[int, LiveBuffer] = {
        rn: LiveBuffer(controller_mode) for rn in robot_nums
    }

    rclpy.init()
    node = LivePlotNode(buffers, namespaces, controller_mode, screenshot_dir=args.screenshot_dir)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig_mgr_cls = LiteLiveFigureManager if args.layout == "lite" else LiveFigureManager
    fig_mgr = fig_mgr_cls(
        robot_nums,
        args.window,
        robot_count,
        controller_mode,
        screenshot_dir=args.screenshot_dir,
        screenshot_rate=args.screenshot_rate,
    )
    plt.ion()

    ns_str = ", ".join(f"robot{rn}={ns}" for rn, ns in namespaces.items())
    print(f"Live plotter started — controller={controller_mode} — layout={args.layout} — {ns_str}")
    print(f"Rolling window: {args.window}s  |  Refresh: {args.interval}s")
    print("Press Ctrl-C to exit.\n")

    try:
        while True:
            snapshots = {rn: buf.snapshot() for rn, buf in buffers.items()}
            fig_mgr.update(snapshots)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nLive plotter shutting down.")
        plt.close("all")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
