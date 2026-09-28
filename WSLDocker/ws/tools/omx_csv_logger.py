#!/usr/bin/env python3
"""
omx_csv_logger.py — standalone, crash-safe CSV logger for the single-arm
buckling experiment.

Design goals (why this exists instead of the harness snapshot CSV):
  * Runs as its OWN process from bringup to shutdown, so logging does not
    depend on the harness spinning, on stages, or on any test logic.
  * Fixed-rate sampling (default 50 Hz) with rclpy spinning in the main
    thread only — no spin starvation.
  * Every row is flushed to disk immediately; the file is valid CSV at every
    instant, so Ctrl+C, SIGKILL, arm/bus disconnect, or power loss loses at
    most the row being written.
  * Units are in every column header.
  * Records per-topic data age so post-processing can see exactly when the
    bus/controller died (columns keep logging with growing age instead of
    silently stopping).

Usage (normally launched by the runner script):
  python3 omx_csv_logger.py --ns robot1 --rate 50 \
      --out-dir /mnt/omx_logs/csv_logs \
      --yaml install/.../robot1_variable_stiffness.yaml \
      --spring-id S1 --k-lat 25.3 --note "trial 3"

Stop with SIGINT/SIGTERM (runner cleanup does this); the file is closed
cleanly, but is complete even if the process is killed hard.
"""

import argparse
import csv
import datetime
import json
import math
import os
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Pose, WrenchStamped
from std_msgs.msg import Bool, Float64, String

NAN = float("nan")


def load_yaml_params(yaml_path, ns):
    """Best-effort read of controller params for metadata + constant columns."""
    out = {}
    try:
        import yaml  # PyYAML
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        params = data.get(f"/{ns}/{ns}_variable_stiffness", {}).get("ros__parameters", {})
        for k in (
            "start_position", "end_position", "move_duration", "homing_duration",
            "fixed_stiffness_x", "fixed_stiffness_y", "fixed_stiffness_z",
            "fixed_damping_x", "fixed_damping_y", "fixed_damping_z",
            "stiffness_homing", "hold_final_compression", "torque_scale",
        ):
            if k in params:
                out[k] = params[k]
        cm = data.get(f"/{ns}/controller_manager", {}).get("ros__parameters", {})
        if "update_rate" in cm:
            out["update_rate"] = cm["update_rate"]
    except Exception as e:  # noqa: BLE001 — metadata is best-effort
        out["yaml_read_error"] = str(e)
    return out


class OmxCsvLogger(Node):
    def __init__(self, args):
        super().__init__("omx_csv_logger")
        self.args = args
        ns = args.ns
        ctrl = f"/{ns}/{ns}_variable_stiffness"

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ── Latest-value store + arrival stamps ─────────────────────────
        now = time.time()
        self.t0 = now
        self.js = None;        self.t_js = None
        self.ee = None;        self.t_ee = None
        self.des = None;       self.t_des = None
        self.wrench = None;    self.t_wrench = None
        self.contact_valid = None
        self.waypoint_active = None
        self.buck_margin = NAN
        self.rot_margin = NAN
        self.instab_channel = ""
        self.rows_written = 0

        # ── Subscriptions (all best-effort; NaN until first message) ────
        self.create_subscription(JointState, f"/{ns}/joint_states", self._cb_js, qos)
        self.create_subscription(Point, f"{ctrl}/end_effector_position", self._cb_ee, qos)
        self.create_subscription(Pose, f"{ctrl}/cartesian_pose_desired", self._cb_des, qos)
        self.create_subscription(WrenchStamped, f"{ctrl}/contact_wrench", self._cb_wrench, qos)
        self.create_subscription(Bool, f"{ctrl}/contact_valid", self._cb_cvalid, qos)
        self.create_subscription(Bool, f"{ctrl}/waypoint_active", self._cb_wp, qos)
        # Fast instability monitor (optional — stays NaN if not running)
        mon = args.monitor_node.rstrip("/")
        self.create_subscription(Float64, f"{mon}/buckling_margin", self._cb_bm, qos)
        self.create_subscription(Float64, f"{mon}/contact_rotation_margin", self._cb_rm, qos)
        self.create_subscription(String, f"{mon}/instability_channel", self._cb_ch, qos)

        # ── Metadata + file setup ────────────────────────────────────────
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        tag = f"{args.spring_id}_klat{args.k_lat}" if args.spring_id else "run"
        os.makedirs(args.out_dir, exist_ok=True)
        base = os.path.join(args.out_dir, f"omx_log_{stamp}_{tag}")
        self.csv_path = base + ".csv"
        self.meta_path = base + "_meta.json"

        self.yaml_params = load_yaml_params(args.yaml, ns) if args.yaml else {}
        meta = {
            "created_utc": stamp,
            "spring_id": args.spring_id,
            "k_lat_commanded_N_per_m": args.k_lat,
            "note": args.note,
            "sample_rate_hz": args.rate,
            "namespace": ns,
            "controller_yaml": os.path.abspath(args.yaml) if args.yaml else None,
            "controller_params": self.yaml_params,
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Column definitions: (name_with_units, getter)
        self.joint_names = ["joint1", "joint2", "joint3", "joint4"]
        self.columns = self._build_columns()

        # newline='' per csv docs; buffering=1 => line buffered
        self._fh = open(self.csv_path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._fh)
        self._writer.writerow([c[0] for c in self.columns])
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._last_fsync = time.time()

        self.create_timer(1.0 / args.rate, self._sample)
        self.get_logger().info(
            f"CSV logger: {self.csv_path} @ {args.rate} Hz "
            f"(gains from YAML: Kx={self.yaml_params.get('fixed_stiffness_x')}, "
            f"Ky={self.yaml_params.get('fixed_stiffness_y')}, "
            f"Kz={self.yaml_params.get('fixed_stiffness_z')})"
        )

    # ── Callbacks ────────────────────────────────────────────────────────
    def _cb_js(self, m):     self.js = m;     self.t_js = time.time()
    def _cb_ee(self, m):     self.ee = m;     self.t_ee = time.time()
    def _cb_des(self, m):    self.des = m;    self.t_des = time.time()
    def _cb_wrench(self, m): self.wrench = m; self.t_wrench = time.time()
    def _cb_cvalid(self, m): self.contact_valid = bool(m.data)
    def _cb_wp(self, m):     self.waypoint_active = bool(m.data)
    def _cb_bm(self, m):     self.buck_margin = float(m.data)
    def _cb_rm(self, m):     self.rot_margin = float(m.data)
    def _cb_ch(self, m):     self.instab_channel = str(m.data)

    # ── Column table ─────────────────────────────────────────────────────
    def _build_columns(self):
        cols = [
            ("time_s [s since logger start]", lambda: time.time() - self.t0),
            ("unix_time [s epoch]",           lambda: time.time()),
        ]

        # EE actual (m)
        for ax in "xyz":
            cols.append((f"ee_{ax} [m]",
                         lambda a=ax: getattr(self.ee, a) if self.ee else NAN))
        # EE desired (m)
        for ax in "xyz":
            cols.append((f"ee_desired_{ax} [m]",
                         lambda a=ax: getattr(self.des.position, a) if self.des else NAN))
        # Tracking error (m)
        for ax in "xyz":
            def err(a=ax):
                if self.ee and self.des:
                    return getattr(self.des.position, a) - getattr(self.ee, a)
                return NAN
            cols.append((f"pos_error_{ax} [m]", err))

        # Contact force (N) and torque (Nm)
        for ax in "xyz":
            cols.append((f"contact_f{ax} [N]",
                         lambda a=ax: getattr(self.wrench.wrench.force, a) if self.wrench else NAN))
        for ax in "xyz":
            cols.append((f"contact_t{ax} [Nm]",
                         lambda a=ax: getattr(self.wrench.wrench.torque, a) if self.wrench else NAN))
        cols.append(("contact_force_norm [N]", self._force_norm))
        cols.append(("contact_valid [bool 0/1]",
                     lambda: "" if self.contact_valid is None else int(self.contact_valid)))

        # Joints: position (rad), velocity (rad/s), effort (raw Dynamixel units)
        for jn in self.joint_names:
            cols.append((f"{jn}_pos [rad]",   lambda j=jn: self._joint(j, "position")))
        for jn in self.joint_names:
            cols.append((f"{jn}_vel [rad/s]", lambda j=jn: self._joint(j, "velocity")))
        for jn in self.joint_names:
            cols.append((f"{jn}_eff [raw Dynamixel units]",
                         lambda j=jn: self._joint(j, "effort")))

        # Instability monitor (dimensionless margins)
        cols.append(("buckling_margin [-]",         lambda: self.buck_margin))
        cols.append(("contact_rotation_margin [-]", lambda: self.rot_margin))
        cols.append(("instability_channel [text]",  lambda: self.instab_channel))
        cols.append(("waypoint_active [bool 0/1]",
                     lambda: "" if self.waypoint_active is None else int(self.waypoint_active)))

        # Data ages — post-processing sees bus dropouts explicitly
        cols.append(("joint_states_age [s]", lambda: self._age(self.t_js)))
        cols.append(("ee_age [s]",           lambda: self._age(self.t_ee)))
        cols.append(("wrench_age [s]",       lambda: self._age(self.t_wrench)))
        cols.append(("bus_alive [bool 0/1: joint_states_age<0.5s]",
                     lambda: int(self._age(self.t_js) < 0.5) if self.t_js else 0))

        # Constant per-run gain columns (from YAML) — units N/m and N·s/m
        yp = self.yaml_params
        cols.append(("k_x [N/m]", lambda: yp.get("fixed_stiffness_x", NAN)))
        cols.append(("k_lat_y [N/m]", lambda: yp.get("fixed_stiffness_y", NAN)))
        cols.append(("k_z [N/m]", lambda: yp.get("fixed_stiffness_z", NAN)))
        cols.append(("d_x [N.s/m]", lambda: yp.get("fixed_damping_x", NAN)))
        cols.append(("d_y [N.s/m]", lambda: yp.get("fixed_damping_y", NAN)))
        cols.append(("d_z [N.s/m]", lambda: yp.get("fixed_damping_z", NAN)))
        return cols

    def _joint(self, name, field):
        if not self.js:
            return NAN
        try:
            i = list(self.js.name).index(name)   # map BY NAME (order is scrambled!)
            arr = getattr(self.js, field)
            return arr[i] if i < len(arr) else NAN
        except ValueError:
            return NAN

    def _force_norm(self):
        if not self.wrench:
            return NAN
        f = self.wrench.wrench.force
        return math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)

    def _age(self, t):
        return (time.time() - t) if t else NAN

    # ── Sampling ─────────────────────────────────────────────────────────
    def _sample(self):
        row = []
        for _, getter in self.columns:
            try:
                v = getter()
            except Exception:  # noqa: BLE001 — a bad field must not kill logging
                v = NAN
            row.append(f"{v:.6f}" if isinstance(v, float) else v)
        self._writer.writerow(row)
        self.rows_written += 1
        # line-buffered already; fsync periodically so even SIGKILL loses <2 s
        if time.time() - self._last_fsync > 2.0:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._last_fsync = time.time()

    def close(self):
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
        self.get_logger().info(
            f"CSV logger closed: {self.rows_written} rows -> {self.csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="robot1")
    ap.add_argument("--rate", type=float, default=50.0, help="sample rate [Hz]")
    ap.add_argument("--out-dir", default="/mnt/omx_logs/csv_logs")
    ap.add_argument("--yaml", default="", help="controller YAML for gain metadata")
    ap.add_argument("--spring-id", default="")
    ap.add_argument("--k-lat", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--monitor-node", default="/fast_instability_monitor",
                    help="node namespace of the instability monitor")
    args = ap.parse_args()

    rclpy.init()
    node = OmxCsvLogger(args)

    stop = {"flag": False}

    def handler(signum, frame):  # noqa: ARG001
        stop["flag"] = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    try:
        while rclpy.ok() and not stop["flag"]:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
