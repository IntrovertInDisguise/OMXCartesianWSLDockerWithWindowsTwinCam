#!/usr/bin/env python3
"""
CartesianImpedance.py

Cartesian impedance control for OpenMANIPULATOR-X.
ROS 2 Humble. Single file. No new package.

    tau = J^T [ Kc (x_d - x) - Dc xdot ] + gravity_feedforward

The script talks to the Dynamixels directly with dynamixel_sdk. It does NOT
use ros2_control.

Why not ros2_control:
  1. A plain Python script cannot claim a ros2_control command interface.
     Controllers there are C++ plugins. A new one would be a new package.
  2. The dynamixel_hardware_interface sync read on this arm fails every few
     seconds (-3001 / -3002), losing 20 to 55 ms each time. Position control
     tolerates that. Impedance control does not: the damping term goes out of
     phase and injects energy instead of removing it. The arm oscillates.
     The plain SDK on the same cable ran 1000 pings with zero failures.

The script still publishes /joint_states, so RViz and tf keep working.

--------------------------------------------------------------------
IMPORTANT: ros2_control must NOT be running. It holds /dev/ttyUSB0.
    Stop the bringup launch first.
--------------------------------------------------------------------

Two phases.

  PHASE 1   python3 CartesianImpedance.py --calibrate

    Runs the trajectory in position control (Operating Mode 3), which is
    what already works on this arm. At each waypoint it holds still and
    records Present Current. That current is the current needed to hold
    the arm against gravity at that pose. Saved to a JSON file.

    This measures gravity on your hardware. It does not compute it from
    link masses and a torque constant, so no unverified constant enters.

  PHASE 2   python3 CartesianImpedance.py --run

    Switches to current control (Operating Mode 0) and follows the same
    waypoints with Cartesian impedance. Gravity feedforward comes from
    the recorded table. The impedance term is added on top.

--------------------------------------------------------------------
UNITS WARNING

Stiffness KC and damping DC below are in Dynamixel current units per
metre, NOT newtons per metre. Converting to N/m needs the motor torque
constant, which this script does not know and does not guess. See the
note next to KC.
--------------------------------------------------------------------
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from typing import List, Optional, Tuple

from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncRead,
    GroupSyncWrite,
    COMM_SUCCESS,
)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# ============================================================
# USER SETTINGS
# ============================================================

PORT_NAME = "/dev/ttyUSB0"
BAUD_RATE = 1000000
DXL_IDS = [11, 12, 13, 14]              # joint1..joint4. Gripper (15) untouched.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]

CALIB_FILE = os.path.expanduser("~/.omx_gravity_calib.json")

# Tip pitch, radians. 0.0 = pointing front. -pi/2 = pointing down.
PITCH_RAD = 0.0

# Absolute Cartesian XYZ waypoints in metres, in the link1 frame.
CARTESIAN_WAYPOINTS_M: List[List[float]] = [
    [0.220, 0.000, 0.11],
    [0.230, 0.000, 0.11],
    [0.240, 0.000, 0.11],
    [0.250, 0.000, 0.11],
    [0.260, 0.000, 0.11],
    [0.270, 0.000, 0.11],
    [0.280, 0.000, 0.11],
    [0.290, 0.000, 0.11],
    [0.300, 0.000, 0.11],
    [0.310, 0.000, 0.11],
    [0.320, 0.000, 0.11],


]

MOVE_TIMES_S = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
DWELL_TIMES_S = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# ---- Impedance gains -------------------------------------------------
#
# UNITS: Dynamixel current units per metre. NOT N/m.
#
# To get N/m you need the torque constant Kt of the XM430-W350, which is
# not in this file because a wrong constant is worse than no constant.
# To find it: hang a known mass m at a known radius R from a joint, hold
# position, read Present Current I. Then
#     Kt [N.m per current unit] = m * g * R / I
# and                  K [N/m]  = KC * Kt / (lever arm terms).
# Until then treat KC as a tuning number, not a physical stiffness.
#
# # Order is [x, y, z]. Low x = soft along the approach direction.
# KC = [300.0, 900.0, 900.0]

# # Damping, current units per (metre per second).
# # Start near critical for your effective mass. Too low oscillates.
# DC = [12.0, 30.0, 30.0]

KC = [5000.0, 10000.0, 10000.0]
DC = [70.0, 100.0, 100.0]

# ---- Safety ----------------------------------------------------------

CONTROL_HZ = 100.0                # servo loop rate
MAX_GOAL_CURRENT = 250            # hard clamp per joint, current units
RAMP_TIME_S = 1.5                 # fade impedance in from zero at start
MAX_TIP_ERROR_M = 0.060           # abort if the tip strays this far
JOINT_LIMIT_MARGIN_RAD = 0.05     # abort this close to a joint limit

# Joint limits, radians. VERIFY against your URDF before trusting these.
# grep -n "limit" open_manipulator_x.urdf.xacro
JOINT_LIMITS_RAD = [
    (-2.80, 2.80),   # joint1
    (-1.75, 1.60),   # joint2
    (-1.60, 1.50),   # joint3
    (-1.70, 2.00),   # joint4
]


# ============================================================
# DYNAMIXEL CONTROL TABLE  (X series)
# VERIFY in the ROBOTIS e-Manual for XM430-W350 before trusting.
# ============================================================

ADDR_OPERATING_MODE = 11      # 1 byte.  0 = current, 3 = position
ADDR_CURRENT_LIMIT = 38       # 2 bytes, EEPROM
ADDR_TORQUE_ENABLE = 64       # 1 byte
ADDR_GOAL_CURRENT = 102       # 2 bytes, signed
ADDR_GOAL_POSITION = 116      # 4 bytes
ADDR_PRESENT_BLOCK = 126      # Present Current(2) Velocity(4) Position(4)
LEN_PRESENT_BLOCK = 10
ADDR_HW_ERROR = 70            # 1 byte

MODE_CURRENT = 0
MODE_POSITION = 3

# Scaling. VERIFY in the e-Manual.
POS_UNITS_PER_REV = 4096.0
POS_CENTER = 2048.0
VEL_UNIT_RPM = 0.229          # rev/min per unit
CUR_UNIT_MA = 2.69            # mA per unit


def to_signed(value: int, bits: int) -> int:
    limit = 1 << bits
    return value - limit if value >= limit // 2 else value


def pos_to_rad(raw: int) -> float:
    return (raw - POS_CENTER) * 2.0 * math.pi / POS_UNITS_PER_REV


def rad_to_pos(rad: float) -> int:
    return int(round(rad * POS_UNITS_PER_REV / (2.0 * math.pi) + POS_CENTER))


def vel_to_rad_s(raw: int) -> float:
    return raw * VEL_UNIT_RPM * 2.0 * math.pi / 60.0


# ============================================================
# GEOMETRY
# Values read from the joint origins in
#   open_manipulator_x_description/urdf/open_manipulator_x.urdf.xacro
# The sign convention is checked in _self_check() against a real MoveIt
# solution recorded from this arm, so it is verified on hardware.
# ============================================================

BASE_X = 0.012
BASE_Z = 0.017 + 0.0595
L1 = math.hypot(0.024, 0.128)         # 0.130225, offset upper link
DELTA = math.atan2(0.024, 0.128)      # 0.185348, its built-in tilt
L2 = 0.124
L3 = 0.126

MAX_WRIST_REACH = L1 + L2
MIN_WRIST_REACH = abs(L1 - L2)


def link_angles(q: List[float]) -> Tuple[float, float, float]:
    a = math.pi / 2.0 - DELTA - q[1]
    b = a - (math.pi / 2.0 - DELTA) - q[2]
    c = b - q[3]
    return a, b, c


def fk(q: List[float]) -> List[float]:
    """Tip position in metres, link1 frame."""
    a, b, c = link_angles(q)
    r = BASE_X + L1 * math.cos(a) + L2 * math.cos(b) + L3 * math.cos(c)
    z = BASE_Z + L1 * math.sin(a) + L2 * math.sin(b) + L3 * math.sin(c)
    return [r * math.cos(q[0]), r * math.sin(q[0]), z]


def jacobian(q: List[float]) -> List[List[float]]:
    """3x4 tip velocity Jacobian. Checked against finite differences."""
    a, b, c = link_angles(q)
    s1, c1 = math.sin(q[0]), math.cos(q[0])

    r = BASE_X + L1 * math.cos(a) + L2 * math.cos(b) + L3 * math.cos(c)

    dr = [L1 * math.sin(a) + L2 * math.sin(b) + L3 * math.sin(c),
          L2 * math.sin(b) + L3 * math.sin(c),
          L3 * math.sin(c)]

    dz = [-(L1 * math.cos(a) + L2 * math.cos(b) + L3 * math.cos(c)),
          -(L2 * math.cos(b) + L3 * math.cos(c)),
          -(L3 * math.cos(c))]

    j = [[0.0] * 4 for _ in range(3)]
    j[0][0] = -r * s1
    j[1][0] = r * c1
    j[2][0] = 0.0

    for i in range(3):
        j[0][i + 1] = c1 * dr[i]
        j[1][i + 1] = s1 * dr[i]
        j[2][i + 1] = dz[i]

    return j


def jt_times(j: List[List[float]], f: List[float]) -> List[float]:
    """J^T f  ->  joint torques from a tip force."""
    return [sum(j[row][col] * f[row] for row in range(3)) for col in range(4)]


def j_times(j: List[List[float]], qd: List[float]) -> List[float]:
    """J qdot  ->  tip velocity from joint velocity."""
    return [sum(j[row][col] * qd[col] for col in range(4)) for row in range(3)]


def ik_pitch(x: float, y: float, z: float, pitch: float) -> Optional[List[float]]:
    """Closed-form 4-DOF IK, elbow-up. None if unreachable."""
    j1 = math.atan2(y, x)
    r = math.hypot(x, y) - BASE_X
    zp = z - BASE_Z

    rw = r - L3 * math.cos(pitch)
    zw = zp - L3 * math.sin(pitch)
    d = math.hypot(rw, zw)

    if d > MAX_WRIST_REACH or d < MIN_WRIST_REACH:
        return None

    cos_beta = (L1 * L1 + L2 * L2 - d * d) / (2.0 * L1 * L2)
    cos_gamma = (d * d + L1 * L1 - L2 * L2) / (2.0 * d * L1)

    beta = math.acos(max(-1.0, min(1.0, cos_beta)))
    gamma = math.acos(max(-1.0, min(1.0, cos_gamma)))

    a = math.atan2(zw, rw) + gamma
    b = a - (math.pi - beta)

    return [j1,
            math.pi / 2.0 - DELTA - a,
            math.pi / 2.0 + DELTA - beta,
            b - pitch]


def _self_check():
    """The reference angles are a real MoveIt solution recorded from the
    physical arm, so this tests the sign convention against measured
    hardware, not against an assumption."""
    ref = [0.220, 0.0, 0.090]
    p = fk([0.0, -0.06321, 0.14415, 0.91478])
    assert all(abs(u - v) < 1e-3 for u, v in zip(p, ref)), f"fk drift: {p}"

    q = ik_pitch(0.220, 0.0, 0.090, 0.0)
    assert q is not None
    assert all(abs(u - v) < 1e-9 for u, v in zip(fk(q), ref))

    # Jacobian against finite differences.
    qt = [0.10, 0.30, 0.50, -0.70]
    jm = jacobian(qt)
    h = 1e-7
    for col in range(4):
        qp = list(qt); qp[col] += h
        qm = list(qt); qm[col] -= h
        num = [(u - v) / (2.0 * h) for u, v in zip(fk(qp), fk(qm))]
        for row in range(3):
            assert abs(num[row] - jm[row][col]) < 1e-6, "jacobian mismatch"


_self_check()


# ============================================================
# BUS
# ============================================================

class Bus:
    """Direct Dynamixel I/O. One sync read and one sync write per cycle."""

    def __init__(self):
        self.port = PortHandler(PORT_NAME)
        self.packet = PacketHandler(2.0)

        if not self.port.openPort():
            raise RuntimeError(
                f"Cannot open {PORT_NAME}. "
                "Is ros2_control still running? Stop the bringup launch."
            )

        if not self.port.setBaudRate(BAUD_RATE):
            raise RuntimeError(f"Cannot set baud rate {BAUD_RATE}.")

        self.reader = GroupSyncRead(
            self.port, self.packet, ADDR_PRESENT_BLOCK, LEN_PRESENT_BLOCK
        )
        for did in DXL_IDS:
            if not self.reader.addParam(did):
                raise RuntimeError(f"sync read addParam failed for id {did}")

        self.writer_current = GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_CURRENT, 2
        )

        self.read_failures = 0
        self.read_attempts = 0

    # -- single register helpers ------------------------------------

    def ping_all(self):
        for did in DXL_IDS:
            _, result, _ = self.packet.ping(self.port, did)
            if result != COMM_SUCCESS:
                raise RuntimeError(
                    f"id {did} did not answer a ping. Check power and cables."
                )

    def hardware_errors(self) -> List[int]:
        out = []
        for did in DXL_IDS:
            v, _, _ = self.packet.read1ByteTxRx(self.port, did, ADDR_HW_ERROR)
            out.append(v)
        return out

    def torque(self, on: bool):
        for did in DXL_IDS:
            self.packet.write1ByteTxRx(
                self.port, did, ADDR_TORQUE_ENABLE, 1 if on else 0
            )

    def set_mode(self, mode: int):
        """Operating Mode lives in EEPROM. Torque must be off to write it."""
        self.torque(False)
        time.sleep(0.05)
        for did in DXL_IDS:
            # SDK write calls return (comm_result, dxl_error). Two values.
            # SDK read calls return (data, comm_result, dxl_error). Three.
            result, error = self.packet.write1ByteTxRx(
                self.port, did, ADDR_OPERATING_MODE, mode
            )
            if result != COMM_SUCCESS:
                raise RuntimeError(
                    f"id {did}: could not set operating mode "
                    f"(comm result {result})"
                )
            if error != 0:
                raise RuntimeError(
                    f"id {did}: servo rejected the operating mode "
                    f"(dxl error {error}). Torque must be off to write EEPROM."
                )

        time.sleep(0.05)

        actual = self.get_mode()
        if any(m != mode for m in actual):
            raise RuntimeError(
                f"Operating mode did not stick. Wanted {mode}, read {actual}."
            )

    def get_mode(self) -> List[int]:
        return [self.packet.read1ByteTxRx(self.port, d, ADDR_OPERATING_MODE)[0]
                for d in DXL_IDS]

    def current_limits(self) -> List[int]:
        return [self.packet.read2ByteTxRx(self.port, d, ADDR_CURRENT_LIMIT)[0]
                for d in DXL_IDS]

    def write_goal_positions(self, q: List[float]):
        for did, angle in zip(DXL_IDS, q):
            result, error = self.packet.write4ByteTxRx(
                self.port, did, ADDR_GOAL_POSITION, rad_to_pos(angle)
            )
            if result != COMM_SUCCESS or error != 0:
                raise RuntimeError(
                    f"id {did}: goal position write failed "
                    f"(comm {result}, dxl error {error})"
                )

    # -- servo loop I/O ---------------------------------------------

    def read_state(self) -> Optional[Tuple[List[float], List[float], List[float]]]:
        """Returns (position rad, velocity rad/s, current units) or None."""
        self.read_attempts += 1

        if self.reader.txRxPacket() != COMM_SUCCESS:
            self.read_failures += 1
            return None

        pos, vel, cur = [], [], []

        for did in DXL_IDS:
            if not self.reader.isAvailable(
                did, ADDR_PRESENT_BLOCK, LEN_PRESENT_BLOCK
            ):
                self.read_failures += 1
                return None

            raw_cur = self.reader.getData(did, ADDR_PRESENT_BLOCK, 2)
            raw_vel = self.reader.getData(did, ADDR_PRESENT_BLOCK + 2, 4)
            raw_pos = self.reader.getData(did, ADDR_PRESENT_BLOCK + 6, 4)

            cur.append(float(to_signed(raw_cur, 16)))
            vel.append(vel_to_rad_s(to_signed(raw_vel, 32)))
            pos.append(pos_to_rad(to_signed(raw_pos, 32)))

        return pos, vel, cur

    def write_currents(self, currents: List[float]) -> bool:
        self.writer_current.clearParam()

        for did, value in zip(DXL_IDS, currents):
            clamped = int(round(max(-MAX_GOAL_CURRENT,
                                    min(MAX_GOAL_CURRENT, value))))
            raw = clamped & 0xFFFF
            if not self.writer_current.addParam(
                did, [raw & 0xFF, (raw >> 8) & 0xFF]
            ):
                return False

        return self.writer_current.txPacket() == COMM_SUCCESS

    def shutdown(self):
        try:
            self.write_currents([0.0] * len(DXL_IDS))
        except Exception:
            pass
        try:
            self.torque(False)
        except Exception:
            pass
        try:
            self.port.closePort()
        except Exception:
            pass


# ============================================================
# TRAJECTORY
# ============================================================

def build_plan() -> List[List[float]]:
    """Solve every waypoint before anything moves."""
    n = len(CARTESIAN_WAYPOINTS_M)

    if not (len(MOVE_TIMES_S) == len(DWELL_TIMES_S) == n):
        raise RuntimeError("Waypoint, move-time and dwell-time lists differ.")

    plan = []

    for i, xyz in enumerate(CARTESIAN_WAYPOINTS_M):
        q = ik_pitch(xyz[0], xyz[1], xyz[2], PITCH_RAD)

        if q is None:
            raise RuntimeError(
                f"Waypoint {i + 1} {xyz} is out of reach at pitch "
                f"{math.degrees(PITCH_RAD):.1f} deg."
            )

        for k, (angle, (lo, hi)) in enumerate(zip(q, JOINT_LIMITS_RAD)):
            if not (lo < angle < hi):
                raise RuntimeError(
                    f"Waypoint {i + 1}: joint{k + 1} = {angle:.4f} rad is "
                    f"outside the limit ({lo}, {hi}). "
                    "Check JOINT_LIMITS_RAD against your URDF."
                )

        plan.append(q)

    return plan


def lerp_xyz(a: List[float], b: List[float], s: float) -> List[float]:
    """Straight line in Cartesian space, smoothed at both ends."""
    s = max(0.0, min(1.0, s))
    smooth = 0.5 - 0.5 * math.cos(math.pi * s)     # cosine ease
    return [u + (v - u) * smooth for u, v in zip(a, b)]


# ============================================================
# NODE
# ============================================================

class ImpedanceNode(Node):

    def __init__(self, bus: Bus):
        super().__init__("cartesian_impedance")
        self.bus = bus
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

    def publish(self, q: List[float], qd: List[float], cur: List[float]):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = list(q)
        msg.velocity = list(qd)
        msg.effort = [c * CUR_UNIT_MA for c in cur]    # mA, not N.m
        self.pub.publish(msg)


# ============================================================
# PHASE 1 : CALIBRATE GRAVITY
# ============================================================

def calibrate(node: ImpedanceNode, bus: Bus, plan: List[List[float]]):
    log = node.get_logger()

    log.info("Calibration: position control, measuring hold current.")
    log.info(f"Current limits: {bus.current_limits()}")

    bus.set_mode(MODE_POSITION)
    bus.torque(True)
    time.sleep(0.2)

    samples = []

    for i, q in enumerate(plan):
        xyz = CARTESIAN_WAYPOINTS_M[i]
        log.info(f"Waypoint {i + 1}/{len(plan)} -> {xyz}")

        bus.write_goal_positions(q)

        # Let it arrive, then average the hold current over 1 second.
        time.sleep(MOVE_TIMES_S[i])

        acc = [0.0] * len(DXL_IDS)
        got = 0
        deadline = time.monotonic() + 1.0

        while time.monotonic() < deadline:
            state = bus.read_state()
            if state is not None:
                pos, vel, cur = state
                acc = [a + c for a, c in zip(acc, cur)]
                got += 1
                node.publish(pos, vel, cur)
            time.sleep(0.01)

        if got == 0:
            raise RuntimeError(f"No readings at waypoint {i + 1}.")

        mean = [a / got for a in acc]
        state = bus.read_state()
        held = state[0] if state else q

        samples.append({"q": held, "gravity_current": mean})
        log.info("  hold current: "
                 + ", ".join(f"{v:+7.1f}" for v in mean)
                 + f"   ({got} samples)")

    with open(CALIB_FILE, "w") as handle:
        json.dump({
            "pitch_rad": PITCH_RAD,
            "waypoints": CARTESIAN_WAYPOINTS_M,
            "samples": samples,
        }, handle, indent=2)

    log.info(f"Saved {CALIB_FILE}")
    log.info(f"Read failures during calibration: "
             f"{bus.read_failures}/{bus.read_attempts}")


def load_calibration() -> List[dict]:
    if not os.path.exists(CALIB_FILE):
        raise RuntimeError(
            f"No calibration at {CALIB_FILE}. "
            "Run with --calibrate first."
        )

    with open(CALIB_FILE) as handle:
        data = json.load(handle)

    if abs(data.get("pitch_rad", 1e9) - PITCH_RAD) > 1e-6:
        raise RuntimeError(
            "Calibration was recorded at a different PITCH_RAD. Re-calibrate."
        )

    if data.get("waypoints") != CARTESIAN_WAYPOINTS_M:
        raise RuntimeError(
            "Calibration was recorded for different waypoints. Re-calibrate."
        )

    return data["samples"]


def gravity_at(samples: List[dict], q: List[float]) -> List[float]:
    """Nearest-neighbour gravity feedforward in joint space.

    Crude on purpose. It is exact at the recorded poses and degrades
    smoothly between them. A proper model needs link masses and a torque
    constant, neither of which this script assumes.
    """
    best = None
    best_d = float("inf")

    for sample in samples:
        d = sum((a - b) ** 2 for a, b in zip(sample["q"], q))
        if d < best_d:
            best_d = d
            best = sample

    return list(best["gravity_current"])


# ============================================================
# PHASE 2 : IMPEDANCE RUN
# ============================================================

def run_impedance(node: ImpedanceNode, bus: Bus, plan: List[List[float]]):
    log = node.get_logger()
    samples = load_calibration()

    errors = bus.hardware_errors()
    if any(errors):
        raise RuntimeError(
            f"Hardware error latched: {list(zip(DXL_IDS, errors))}. "
            "Power cycle the arm."
        )

    log.info(f"Current limits: {bus.current_limits()}")
    log.info(f"KC = {KC}  DC = {DC}  (current units per m, per m/s)")
    log.info(f"Clamp = +/-{MAX_GOAL_CURRENT} units "
             f"({MAX_GOAL_CURRENT * CUR_UNIT_MA:.0f} mA)")

    # Move to the start pose in position control, so impedance starts from
    # a known place instead of wherever the arm happens to be.
    log.info("Moving to the start pose in position control...")
    bus.set_mode(MODE_POSITION)
    bus.torque(True)
    bus.write_goal_positions(plan[0])
    time.sleep(3.0)

    log.info("Switching to current control.")
    bus.set_mode(MODE_CURRENT)
    bus.torque(True)
    time.sleep(0.1)

    dt = 1.0 / CONTROL_HZ
    start = time.monotonic()

    last_state = bus.read_state()
    if last_state is None:
        raise RuntimeError("No reading at the start of the impedance loop.")

    stats_max_err = 0.0

    for i in range(len(plan)):
        x_from = fk(plan[i - 1]) if i > 0 else fk(plan[0])
        x_to = CARTESIAN_WAYPOINTS_M[i]

        move_t = MOVE_TIMES_S[i]
        dwell_t = DWELL_TIMES_S[i]
        segment_t = move_t + dwell_t

        log.info(f"Waypoint {i + 1}/{len(plan)} -> {x_to}")

        t0 = time.monotonic()

        while True:
            now = time.monotonic()
            elapsed = now - t0

            if elapsed > segment_t:
                break

            # ---- read ------------------------------------------------
            state = bus.read_state()
            if state is None:
                # Hold the previous command for one cycle rather than
                # commanding zero, which would drop the arm.
                time.sleep(dt)
                continue

            q, qd, cur = state
            last_state = state

            # ---- setpoint --------------------------------------------
            s = min(1.0, elapsed / move_t) if move_t > 0 else 1.0
            x_d = lerp_xyz(x_from, x_to, s)

            # ---- current tip state -----------------------------------
            x = fk(q)
            jm = jacobian(q)
            xd = j_times(jm, qd)

            err = [a - b for a, b in zip(x_d, x)]
            err_norm = math.sqrt(sum(e * e for e in err))
            stats_max_err = max(stats_max_err, err_norm)

            # ---- safety ----------------------------------------------
            if err_norm > MAX_TIP_ERROR_M:
                raise RuntimeError(
                    f"Tip error {err_norm * 1000:.0f} mm exceeds the "
                    f"{MAX_TIP_ERROR_M * 1000:.0f} mm limit. Stopping."
                )

            for k, (angle, (lo, hi)) in enumerate(zip(q, JOINT_LIMITS_RAD)):
                if angle < lo + JOINT_LIMIT_MARGIN_RAD or \
                   angle > hi - JOINT_LIMIT_MARGIN_RAD:
                    raise RuntimeError(
                        f"joint{k + 1} = {angle:.4f} rad is at its limit. "
                        "Stopping."
                    )

            # ---- impedance law ---------------------------------------
            #   f = Kc (x_d - x) - Dc xdot          (tip space)
            #   tau = J^T f  +  gravity             (joint space)
            #
            # Damping uses MEASURED joint velocity through the Jacobian,
            # not a numerical derivative of the error. A filtered
            # derivative lags by roughly one filter time constant, which
            # at these frequencies turns damping into energy injection.
            force = [KC[k] * err[k] - DC[k] * xd[k] for k in range(3)]

            tau = jt_times(jm, force)
            grav = gravity_at(samples, q)

            ramp = min(1.0, (now - start) / RAMP_TIME_S)
            command = [grav[k] + ramp * tau[k] for k in range(len(DXL_IDS))]

            if not bus.write_currents(command):
                log.warn("Goal current write failed on one cycle.")

            node.publish(q, qd, cur)

            # ---- pace ------------------------------------------------
            sleep_for = dt - (time.monotonic() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

        tip = fk(last_state[0])
        log.info(f"  measured tip: x={tip[0]:.4f} y={tip[1]:.4f} "
                 f"z={tip[2]:.4f} m  "
                 f"(error {math.dist(tip, x_to) * 1000:.1f} mm)")

    log.info(f"Largest tracking error: {stats_max_err * 1000:.1f} mm")
    log.info(f"Read failures: {bus.read_failures}/{bus.read_attempts}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true",
                       help="Phase 1. Position control, record gravity.")
    group.add_argument("--run", action="store_true",
                       help="Phase 2. Current control, Cartesian impedance.")
    args = parser.parse_args()

    plan = build_plan()

    rclpy.init()
    bus = None
    node = None
    code = 0

    def on_signal(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        bus = Bus()
        node = ImpedanceNode(bus)

        bus.ping_all()
        node.get_logger().info(f"All {len(DXL_IDS)} servos answered.")

        if args.calibrate:
            calibrate(node, bus, plan)
        else:
            run_impedance(node, bus, plan)

    except KeyboardInterrupt:
        if node:
            node.get_logger().warn("Interrupted. Turning torque off.")
        code = 130

    except Exception as exc:
        if node:
            node.get_logger().error(str(exc))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        code = 1

    finally:
        # Torque off on every exit path, including a crash.
        if bus:
            bus.shutdown()
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(code)


if __name__ == "__main__":
    main()