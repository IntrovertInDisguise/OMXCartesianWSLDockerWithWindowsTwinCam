#!/usr/bin/env python3
"""
ArmImpedance.py

Cartesian impedance control for OpenMANIPULATOR-X.
ROS 2 Humble. Single file. No new package.

    tau = J^T [ Kc (x_d - x) + Ki integral(x_d - x) - Dc xdot ]
          - Kd_joint qdot                      <-- NEW, damps all 4 DOF
          + n_hat [ Kp_n (p_d - p) - Kd_n pdot ]   <-- NEW, holds the pitch
          + gravity_feedforward

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


====================================================================
WHY THE ARM FELL AT WAYPOINT 2, AND WHAT CHANGED
====================================================================

Waypoint 1 is the only segment where the arm does not move: it is driven
there in position control, the mode is switched, and the gravity table is
exact at that exact pose. It holds. Waypoint 2 is the first segment that
asks the arm to MOVE, and three separate defects all bite at once.

(1) THE CONTROL LAW HAD NO AUTHORITY OVER ONE WHOLE DEGREE OF FREEDOM.

    jacobian() is 3x4. tau = J^T f can only ever produce torques inside
    the 3-dimensional row space of J. The orthogonal complement, null(J),
    is 1-dimensional and gets EXACTLY ZERO torque from the old law --
    measured, not argued: the nullspace component of J^T f came out at
    1e-16 units.

    At [0.230, 0, 0.110] that direction is

        n = [0.000, +0.396, +0.138, -0.908]

    Sliding 0.2 rad along it moves the tip 0.85 mm and changes the tip
    pitch by 0.075 rad. So the arm can collapse internally -- elbow down,
    wrist up -- while the tip barely moves, which means the tip error
    never grows enough to make the controller react, and MAX_TIP_ERROR_M
    never trips until the collapse is already large. Nothing damped it
    either: Dc also entered through J^T, so the nullspace velocity had
    zero stiffness AND zero damping. It is a free integrator driven by
    whatever gravity error exists. It runs away.

    In calibration this cannot happen, because position control closes a
    loop on all four joints individually.

    FIX: two additions, both outside J^T.
      * KD_JOINT: honest joint-space damping on all four joints. This
        alone removes the runaway.
      * KP_NULL / KD_NULL: a PD on the tip pitch, projected onto n_hat,
        the exact null direction computed per cycle from the 4D
        generalised cross product of the rows of J. Because it lies in
        null(J) it holds the posture without fighting the Cartesian task.

(2) THE GRAVITY TABLE WAS SHORT BY THE STICTION TERM, AND THE ARM IS TOO
    SOFT TO MAKE THE DIFFERENCE UP.

    Phase 1 approached every waypoint from the same side, moving outward.
    Friction opposes motion, so it was HELPING to hold the arm, and the
    recorded Present Current is therefore

        gravity - stiction

    Stationary, that is enough. Moving, stiction drops away or reverses
    and the feedforward is short. How much sag does a shortfall buy?
    At this pose the joint-2 lever arm in z is 0.218 m, so

        1 mm of z error -> 2.18 current units of joint-2 torque

    A 20-unit (54 mA) shortfall therefore parks the arm 9.2 mm low; 40
    units puts it 18.4 mm low. Breakaway friction on an XM430-W350 with
    its 353:1 gearbox is comfortably in that range. That is a large,
    structural, one-directional sag -- and it lands on top of (1).

    FIXES:
      * --calibrate now sweeps the waypoints FORWARDS and then BACKWARDS
        and averages the two. The stiction term is +f one way and -f the
        other, so it cancels and what is left is the gravity term.
      * KI_TIP, a bounded integral on the tip error, absorbs whatever
        bias survives. Anti-windup clamped, and held at zero until the
        ramp finishes so it cannot wind up during the mode switch.

(3) THE GRAVITY LOOKUP WAS A STAIRCASE.

    gravity_at() was nearest-neighbour over 11 samples, so the
    feedforward stepped discontinuously as the arm crossed each midpoint
    -- and the first midpoint it crosses is in the middle of segment 2.
    A step in feedforward is a step in torque, which is an impulse.

    FIX: the samples are the vertices of a piecewise-linear path in joint
    space. gravity_at() now projects q onto that polyline and blends the
    two bracketing samples. Continuous everywhere, and still exact at the
    recorded poses.

ALSO CHANGED

  * x_from is now the MEASURED tip position at the start of every
    segment, not just the first. It used to be CARTESIAN_WAYPOINTS_M[i-1]
    for i>0, which re-injected any standing offset as a step in x_d at
    each segment boundary.

  * Abort no longer drops the arm. The old shutdown() wrote zero current
    and cut torque, so every abort -- including the tip-error abort that
    is supposed to protect the arm -- ended in a free fall onto the
    table. park() now switches back to position control at the MEASURED
    pose and holds. Use --no-park for the old behaviour.

  * Profile Velocity and Acceleration are set explicitly, so the position
    moves in phase 1 are slow and repeatable instead of running at the
    default (0 = maximum speed). Calibration current recorded after a
    slam is not the same number as after a gentle approach.

  * Goal Current is read back after torque enable, to prove the seeding
    survived the transition rather than assuming it did.

  * Current Limit is checked against MAX_GOAL_CURRENT at startup.


--------------------------------------------------------------------
UNITS WARNING

Stiffness KC and damping DC below are in Dynamixel current units per
metre, NOT newtons per metre. Converting to N/m needs the motor torque
constant, which this script does not know and does not guess. See the
note next to KC.
--------------------------------------------------------------------


CSV OUTPUT

  --calibrate  writes CALIB_CSV
  --run        writes TRACE_FILE

Both carry time, the four joint currents, the four joint angles, and the
forward-kinematic tip pose. Plot them with

    python3 plot_arm.py /tmp/imp_run.csv
    python3 plot_arm.py /tmp/imp_calib.csv
"""

import argparse
import csv
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
NJ = len(DXL_IDS)

CALIB_FILE = os.path.expanduser("~/.omx_gravity_calib.json")
CALIB_CSV = "/tmp/imp_calib.csv"
TRACE_FILE = "/tmp/imp_run.csv"

# Tip pitch, radians. 0.0 = pointing front. -pi/2 = pointing down.
PITCH_RAD = 0.0

# Absolute Cartesian XYZ waypoints in metres, in the link1 frame.
CARTESIAN_WAYPOINTS_M: List[List[float]] = [
    [0.220, 0.000, 0.110],
    [0.230, 0.000, 0.110],
    [0.240, 0.000, 0.110],
    [0.250, 0.000, 0.110],
    [0.260, 0.000, 0.110],
    [0.270, 0.000, 0.110],
    [0.280, 0.000, 0.110],
    [0.290, 0.000, 0.110],
    [0.300, 0.000, 0.110],
    [0.310, 0.000, 0.110],
    [0.320, 0.000, 0.110],
]

MOVE_TIMES_S = [10.0] * 11
DWELL_TIMES_S = [5.0] * 11

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
# Order is [x, y, z]. Low x = soft along the approach direction.
#
# Scale check at [0.230, 0, 0.110]: the joint-2 z lever arm is 0.218 m,
# so KC[2] = 10000 gives 2.18 current units of joint-2 torque per mm of
# sag. The x lever arm is only 0.0335 m, so KC[0] = 5000 gives 0.17 units
# per mm along the direction of travel -- deliberately soft, but it does
# mean the arm will lag the setpoint in x. That lag is not a fault.
KC = [5000.0, 10000.0, 10000.0]

# Damping, current units per (metre per second).
DC = [70.0, 100.0, 100.0]

# Integral on tip error, current units per (metre * second).
# Absorbs the stiction bias that survives the two-way calibration.
# Zero this to turn the integral off.
KI_TIP = [1500.0, 3000.0, 3000.0]
I_CLAMP = 80.0                    # max integral FORCE per axis, current units

# ---- Joint-space terms.  These are what stop the nullspace runaway. ---
#
# KD_JOINT is plain viscous damping on each joint, applied OUTSIDE J^T so
# it reaches all four degrees of freedom. Present Velocity is quantised
# at 0.229 rpm = 0.024 rad/s, so KD_JOINT[k] * 0.024 is the torque ripple
# this term contributes at standstill. At 25 that is 0.6 units. Fine.
KD_JOINT = [10.0, 25.0, 25.0, 15.0]     # current units per (rad/s)

# PD on the tip pitch, projected into null(J). Units: current units per
# rad and per (rad/s). Set KP_NULL = 0 to disable the posture term (but
# leave KD_JOINT on, or the nullspace is undamped again).
KP_NULL = 250.0
KD_NULL = 25.0

# If |n . grad_pitch| falls below this the nullspace is nearly orthogonal
# to the pitch coordinate and the projection blows up. Skip the term.
NULL_MIN_ALIGN = 0.05

# ---- Safety ----------------------------------------------------------

CONTROL_HZ = 100.0                # servo loop rate
MAX_GOAL_CURRENT = 250            # hard clamp per joint, current units
RAMP_TIME_S = 1.5                 # fade impedance in from zero at start
MAX_TIP_ERROR_M = 0.060           # abort if the tip strays this far
MAX_PITCH_ERROR_RAD = 0.35        # abort if the posture collapses
JOINT_LIMIT_MARGIN_RAD = 0.05     # abort this close to a joint limit

# Profile for the position-control moves in phase 1 and the approach in
# phase 2. Units of Profile Velocity are 0.229 rpm, Acceleration
# 214.577 rev/min^2. 0 means "no limit", which is what we are avoiding.
PROFILE_VELOCITY = 40             # ~9.2 rpm
PROFILE_ACCELERATION = 20

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
ADDR_GOAL_CURRENT = 102       # 2 bytes, signed, RAM
ADDR_PROFILE_ACCEL = 108      # 4 bytes
ADDR_PROFILE_VELOCITY = 112   # 4 bytes
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

# Gradient of the tip pitch with respect to the joint angles. Constant,
# because pitch is exactly -(q2+q3+q4).
PITCH_GRAD = [0.0, -1.0, -1.0, -1.0]


def link_angles(q: List[float]) -> Tuple[float, float, float]:
    a = math.pi / 2.0 - DELTA - q[1]
    b = a - (math.pi / 2.0 - DELTA) - q[2]
    c = b - q[3]
    return a, b, c


def pitch_of(q: List[float]) -> float:
    """Absolute elevation of the last link.

    Algebraically identical to link_angles(q)[2], but written in the form
    that makes the point obvious: the pitch depends only on q2+q3+q4, and
    jacobian() below has no row for it.  _self_check() asserts the two
    forms agree.

    In the ORIGINAL script nothing constrained this quantity and it was
    free to drift.  It is now closed by the nullspace PD in
    run_impedance(), and _self_check() proves the projection direction
    really does lie in null(J).
    """
    return -(q[1] + q[2] + q[3])


def pitch_rate_of(qd: List[float]) -> float:
    return -(qd[1] + qd[2] + qd[3])


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


def null_of_jacobian(j: List[List[float]]) -> List[float]:
    """The 1-D null space of a 3x4 Jacobian, exactly.

    n_i = (-1)^i * det(J with column i deleted).  Then

        sum_i J[r][i] n_i = det( [J_r ; J_0 ; J_1 ; J_2] ) = 0

    for every row r, because that 4x4 has a repeated row.  So J n = 0 by
    construction, with no linear algebra library and no iteration.

    This is the joint-velocity direction that moves nothing at the tip.
    It is precisely the direction the old control law could not touch.
    Returns an unnormalised vector; magnitude goes to zero at a
    singularity, which the caller checks.
    """
    def det3(c0: int, c1: int, c2: int) -> float:
        return (j[0][c0] * (j[1][c1] * j[2][c2] - j[1][c2] * j[2][c1])
                - j[0][c1] * (j[1][c0] * j[2][c2] - j[1][c2] * j[2][c0])
                + j[0][c2] * (j[1][c0] * j[2][c1] - j[1][c1] * j[2][c0]))

    return [det3(1, 2, 3), -det3(0, 2, 3), det3(0, 1, 3), -det3(0, 1, 2)]


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

    # The short pitch form must agree with link_angles.
    for qc in ([0.1, 0.3, 0.5, -0.7], [-0.2, -0.4, 0.2, 0.9]):
        assert abs(link_angles(qc)[2] - pitch_of(qc)) < 1e-12, "pitch form"

    # And ik_pitch must deliver the pitch it was asked for.
    for target in (-0.9, -0.4, 0.0):
        qs = ik_pitch(0.250, 0.0, 0.110, target)
        assert qs is not None and abs(pitch_of(qs) - target) < 1e-9

    # ---- the nullspace claims the new control law depends on ----------
    for qc in ([0.0, -0.3, 0.4, -0.1], [0.2, 0.1, 0.5, -0.6], qt):
        jm = jacobian(qc)
        n = null_of_jacobian(jm)
        mag = math.sqrt(sum(v * v for v in n))
        assert mag > 1e-9, "nullspace vanished at a test pose"
        nh = [v / mag for v in n]

        # 1. It really is the null space: J n = 0.
        for row in range(3):
            assert abs(sum(jm[row][c] * nh[c] for c in range(4))) < 1e-9, \
                "null_of_jacobian is not in the null space"

        # 2. It really is the direction J^T cannot reach: for any tip
        #    force, the torque J^T f has no component along n.
        for f in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                  [0.3, -0.7, 0.5]):
            tau = jt_times(jm, f)
            assert abs(sum(a * b for a, b in zip(tau, nh))) < 1e-9, \
                "J^T reaches the null space, which is impossible"

        # 3. Moving along n changes the pitch but not the tip. This is
        #    what makes the pitch a valid coordinate for the nullspace PD.
        step = 0.05
        q2 = [a + step * b for a, b in zip(qc, nh)]
        assert math.dist(fk(qc), fk(q2)) < 1e-3, "null step moved the tip"
        assert abs(pitch_of(q2) - pitch_of(qc)) > 1e-3, \
            "null step did not move the pitch"


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
        self.park_on_exit = True

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
        """Operating Mode lives in EEPROM. Torque must be off to write it.

        This leaves torque OFF on return.  The caller re-enables it, so
        that a holding current can be seeded first.  See the note in the
        module docstring about the 67 mm drop.
        """
        self.torque(False)
        time.sleep(0.02)
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

        time.sleep(0.02)

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

    def read_goal_currents(self) -> List[int]:
        """Read back what the servo thinks its Goal Current is.

        Used to PROVE the seeding survived the torque-enable transition
        instead of assuming Goal Current is preserved across it.
        """
        out = []
        for did in DXL_IDS:
            raw, _, _ = self.packet.read2ByteTxRx(
                self.port, did, ADDR_GOAL_CURRENT
            )
            out.append(to_signed(raw, 16))
        return out

    def set_profile(self, velocity: int, accel: int):
        """Speed limits for position-control moves. Only meaningful in
        Operating Mode 3; harmless otherwise."""
        for did in DXL_IDS:
            self.packet.write4ByteTxRx(
                self.port, did, ADDR_PROFILE_ACCEL, accel)
            self.packet.write4ByteTxRx(
                self.port, did, ADDR_PROFILE_VELOCITY, velocity)

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

    def read_state_blocking(self, timeout_s: float = 0.5):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.read_state()
            if state is not None:
                return state
            time.sleep(0.005)
        raise RuntimeError("No servo reading within timeout.")

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

    # -- exit --------------------------------------------------------

    def park(self) -> bool:
        """Hold the arm where it is instead of dropping it.

        The old shutdown() wrote zero current and cut torque. In current
        mode that is a free fall, and it happened on EVERY abort path --
        including the tip-error abort whose whole purpose is to protect
        the arm. This switches back to position control at the pose the
        arm is actually in and leaves torque ON.
        """
        try:
            q = self.read_state_blocking(timeout_s=0.5)[0]
        except Exception:
            return False

        try:
            self.set_mode(MODE_POSITION)          # leaves torque off
            self.set_profile(PROFILE_VELOCITY, PROFILE_ACCELERATION)
            self.write_goal_positions(q)          # goal = where we are
            self.torque(True)
            time.sleep(0.3)
            return True
        except Exception:
            return False

    def release(self):
        try:
            self.write_currents([0.0] * NJ)
        except Exception:
            pass
        try:
            self.torque(False)
        except Exception:
            pass

    def shutdown(self) -> str:
        parked = False
        if self.park_on_exit:
            parked = self.park()
        if not parked:
            self.release()
        try:
            self.port.closePort()
        except Exception:
            pass
        return ("parked, torque left ON, arm is holding itself"
                if parked else "torque OFF, arm is limp")


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
# CSV
# ============================================================

def open_csv(path: str, header: List[str]):
    handle = open(path, "w", newline="")
    writer = csv.writer(handle)
    writer.writerow(header)
    return handle, writer


CORE_COLUMNS = (
    ["t", "phase", "wp", "s"]
    + ["x", "y", "z"]                            # FK tip pose, metres
    + ["pitch"]                                  # FK tip pitch, rad
    + [f"q{k + 1}" for k in range(NJ)]           # joint angles, rad
    + [f"qd{k + 1}" for k in range(NJ)]          # joint velocities, rad/s
    + [f"cur{k + 1}" for k in range(NJ)]         # MEASURED current, units
    + [f"cur{k + 1}_mA" for k in range(NJ)]      # same, milliamps
)


# ============================================================
# PHASE 1 : CALIBRATE GRAVITY
# ============================================================

def _sweep(node, bus, plan, order, csv_writer, t0, direction_tag, log):
    """One pass over the waypoints in position control, recording the
    hold current at each. Returns {waypoint_index: (q_held, mean_cur)}."""
    out = {}

    for step, i in enumerate(order):
        q = plan[i]
        xyz = CARTESIAN_WAYPOINTS_M[i]
        log.info(f"  [{direction_tag}] {step + 1}/{len(order)}  "
                 f"waypoint {i + 1} -> {xyz}")

        bus.write_goal_positions(q)
        time.sleep(MOVE_TIMES_S[i])

        acc = [0.0] * NJ
        got = 0
        held = q
        deadline = time.monotonic() + 1.0

        while time.monotonic() < deadline:
            state = bus.read_state()
            if state is not None:
                pos, vel, cur = state
                acc = [a + c for a, c in zip(acc, cur)]
                got += 1
                held = pos
                node.publish(pos, vel, cur)

                tip = fk(pos)
                csv_writer.writerow(
                    [f"{time.monotonic() - t0:.4f}", direction_tag, i + 1, "1.0"]
                    + [f"{v:.5f}" for v in tip]
                    + [f"{pitch_of(pos):.5f}"]
                    + [f"{v:.5f}" for v in pos]
                    + [f"{v:.5f}" for v in vel]
                    + [f"{v:.1f}" for v in cur]
                    + [f"{v * CUR_UNIT_MA:.1f}" for v in cur])
            time.sleep(0.01)

        if got == 0:
            raise RuntimeError(f"No readings at waypoint {i + 1}.")

        mean = [a / got for a in acc]
        out[i] = (held, mean)

        log.info("     hold current: " + ", ".join(f"{v:+7.1f}" for v in mean)
                 + f"   ({got} samples)")

    return out


def calibrate(node, bus: Bus, plan: List[List[float]]):
    log = node.get_logger()

    errors = bus.hardware_errors()
    if any(errors):
        raise RuntimeError(
            f"Hardware error latched: {list(zip(DXL_IDS, errors))}. "
            "Power cycle the arm before calibrating."
        )

    log.info("Calibration: position control, measuring hold current.")
    log.info(f"Current limits: {bus.current_limits()}")
    log.info("Two passes, forwards then backwards. Averaging them cancels "
             "the stiction bias -- friction helps hold the arm on the way "
             "out and hinders on the way back, so it enters the two passes "
             "with opposite sign.")

    bus.set_mode(MODE_POSITION)
    bus.set_profile(PROFILE_VELOCITY, PROFILE_ACCELERATION)
    bus.torque(True)
    time.sleep(0.2)

    handle, writer = open_csv(CALIB_CSV, CORE_COLUMNS)
    t0 = time.monotonic()

    try:
        n = len(plan)
        fwd = _sweep(node, bus, plan, list(range(n)), writer, t0, "fwd", log)
        log.info("Reversing.")
        rev = _sweep(node, bus, plan, list(range(n - 1, -1, -1)),
                     writer, t0, "rev", log)
    finally:
        handle.close()

    samples = []
    log.info("")
    log.info("Per-waypoint stiction estimate, (fwd - rev) / 2 in current "
             "units. Large numbers here are exactly the term that used to "
             "be baked into the feedforward:")

    for i in range(len(plan)):
        q_f, c_f = fwd[i]
        q_r, c_r = rev[i]
        mean = [(a + b) / 2.0 for a, b in zip(c_f, c_r)]
        half = [(a - b) / 2.0 for a, b in zip(c_f, c_r)]
        q_avg = [(a + b) / 2.0 for a, b in zip(q_f, q_r)]

        samples.append({
            "q": q_avg,
            "gravity_current": mean,
            "stiction_half_range": half,
            "fwd_current": c_f,
            "rev_current": c_r,
        })

        log.info(f"  wp {i + 1:2d}  gravity "
                 + ", ".join(f"{v:+7.1f}" for v in mean)
                 + "   stiction +/-"
                 + ", ".join(f"{abs(v):5.1f}" for v in half))
        log.info(f"          pitch {pitch_of(q_avg):+.4f} rad "
                 f"(commanded {PITCH_RAD:+.4f})")

    worst = max(abs(v) for s in samples for v in s["stiction_half_range"])
    log.info("")
    log.info(f"Largest stiction half-range: {worst:.1f} units "
             f"({worst * CUR_UNIT_MA:.0f} mA). At the joint-2 z lever arm "
             f"of ~0.218 m that is worth roughly {worst / 2.176:.1f} mm of "
             "sag if it is left in the feedforward, which is what the old "
             "single-pass calibration did.")

    with open(CALIB_FILE, "w") as fh:
        json.dump({
            "pitch_rad": PITCH_RAD,
            "waypoints": CARTESIAN_WAYPOINTS_M,
            "samples": samples,
            "bidirectional": True,
        }, fh, indent=2)

    log.info(f"Saved {CALIB_FILE}")
    log.info(f"CSV written to {CALIB_CSV}")
    log.info(f"Read failures during calibration: "
             f"{bus.read_failures}/{bus.read_attempts}")


def load_calibration() -> List[dict]:
    if not os.path.exists(CALIB_FILE):
        raise RuntimeError(
            f"No calibration at {CALIB_FILE}. Run with --calibrate first."
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

    if not data.get("bidirectional"):
        raise RuntimeError(
            "This calibration file is from the old single-pass sweep, so "
            "the stiction bias is still baked into it. Re-run --calibrate."
        )

    return data["samples"]


def gravity_at(samples: List[dict], q: List[float]) -> List[float]:
    """Gravity feedforward, interpolated along the joint-space path.

    The samples are the vertices of a piecewise-linear path in joint
    space. Project q onto that polyline, find the bracketing pair, and
    blend. Continuous everywhere, and still exact at the recorded poses.

    The old version was nearest-neighbour, which stepped discontinuously
    at each midpoint. With 11 samples 10 mm apart the first step landed
    in the middle of segment 2, which is the segment that failed.

    This is still not a gravity MODEL -- it is a measurement table. A
    model needs link masses and a torque constant, neither of which this
    script assumes. But after the two-pass calibration the table holds
    gravity, not "gravity minus whatever friction was helping".
    """
    if len(samples) == 1:
        return list(samples[0]["gravity_current"])

    best_d = float("inf")
    best_k = 0
    best_u = 0.0

    for k in range(len(samples) - 1):
        a = samples[k]["q"]
        b = samples[k + 1]["q"]
        ab = [y - x for x, y in zip(a, b)]
        den = sum(v * v for v in ab)

        if den < 1e-12:
            u = 0.0
        else:
            u = sum((qi - ai) * abi
                    for qi, ai, abi in zip(q, a, ab)) / den
            u = max(0.0, min(1.0, u))

        d = sum((qi - (ai + u * abi)) ** 2
                for qi, ai, abi in zip(q, a, ab))

        if d < best_d:
            best_d, best_k, best_u = d, k, u

    ga = samples[best_k]["gravity_current"]
    gb = samples[best_k + 1]["gravity_current"]
    return [(1.0 - best_u) * ga[j] + best_u * gb[j] for j in range(NJ)]


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
# PHASE 2 : IMPEDANCE RUN
# ============================================================

def mean_of(rows: List[List[float]]) -> List[float]:
    if not rows:
        return [float("nan")] * NJ
    n = len(rows)
    return [sum(r[k] for r in rows) / n for k in range(NJ)]


def fmt(values: List[float]) -> str:
    return ", ".join(f"{v:+7.1f}" for v in values)


def run_impedance(node, bus: Bus, plan: List[List[float]]):
    log = node.get_logger()
    samples = load_calibration()

    errors = bus.hardware_errors()
    if any(errors):
        raise RuntimeError(
            f"Hardware error latched: {list(zip(DXL_IDS, errors))}. "
            "Power cycle the arm."
        )

    limits = bus.current_limits()
    log.info(f"Current limits: {limits}")
    if any(l < MAX_GOAL_CURRENT for l in limits):
        log.warn(f"Current Limit is below MAX_GOAL_CURRENT "
                 f"({MAX_GOAL_CURRENT}). The servo will truncate the "
                 "command before the clamp in this script ever sees it.")

    log.info(f"KC = {KC}  DC = {DC}  KI = {KI_TIP}  "
             "(current units per m, m/s, m.s)")
    log.info(f"KD_JOINT = {KD_JOINT}  KP_NULL = {KP_NULL}  "
             f"KD_NULL = {KD_NULL}")
    log.info(f"Clamp = +/-{MAX_GOAL_CURRENT} units "
             f"({MAX_GOAL_CURRENT * CUR_UNIT_MA:.0f} mA)")

    # Move to the start pose in position control, so impedance starts from
    # a known place instead of wherever the arm happens to be.
    log.info("Moving to the start pose in position control...")
    bus.set_mode(MODE_POSITION)
    bus.set_profile(PROFILE_VELOCITY, PROFILE_ACCELERATION)
    bus.torque(True)
    bus.write_goal_positions(plan[0])
    time.sleep(4.0)

    # ---- Mode switch without free-fall -------------------------------
    # set_mode() must drop torque to write EEPROM, and Goal Current stays
    # 0 until something writes it.  In current mode zero goal current is
    # zero holding torque, so re-enabling torque does not help.  That gap
    # was ~240 ms and produced the 67 mm drop.  Goal Current (102) is
    # RAM, so seed it BEFORE enabling torque.
    q_before = bus.read_state_blocking()[0]
    hold = gravity_at(samples, q_before)

    log.info("Switching to current control, seeding " + fmt(hold))
    bus.set_mode(MODE_CURRENT)
    bus.write_currents(hold)
    bus.torque(True)

    # Prove the seed survived the transition rather than assuming it.
    seeded = bus.read_goal_currents()
    log.info("Goal Current read back after torque enable: " + fmt(seeded))
    if any(abs(a - b) > 3 for a, b in zip(seeded, hold)):
        raise RuntimeError(
            f"Goal Current did not survive the torque enable. Wanted "
            f"{[round(v) for v in hold]}, read {seeded}. The arm would "
            "drop. Not proceeding."
        )

    dt = 1.0 / CONTROL_HZ
    start = time.monotonic()

    last_state = bus.read_state_blocking()
    last_command = list(hold)

    log.info(f"Start tip {['%.4f' % v for v in fk(last_state[0])]}  "
             f"pitch {pitch_of(last_state[0]):+.4f} rad")

    stats_max_err = 0.0
    stats_max_pitch_err = 0.0
    periods = []
    integral = [0.0, 0.0, 0.0]        # tip-space integral FORCE, units

    columns = (CORE_COLUMNS
               + ["xd", "yd", "zd", "err_mm", "pitch_d", "pitch_err"]
               + [f"cmd{k + 1}" for k in range(NJ)]
               + [f"grav{k + 1}" for k in range(NJ)]
               + [f"imp{k + 1}" for k in range(NJ)]
               + [f"damp{k + 1}" for k in range(NJ)]
               + [f"null{k + 1}" for k in range(NJ)]
               + ["ix", "iy", "iz", "period"])

    trace, tracer = open_csv(TRACE_FILE, columns)
    prev_cycle = start

    try:
        for i in range(len(plan)):
            # Start EVERY segment from where the arm actually is. The old
            # code used CARTESIAN_WAYPOINTS_M[i-1] for i>0, which threw
            # away the standing offset and re-injected it as a step in
            # x_d at each segment boundary.
            x_from = fk(last_state[0])
            x_to = CARTESIAN_WAYPOINTS_M[i]

            move_t = MOVE_TIMES_S[i]
            dwell_t = DWELL_TIMES_S[i]
            segment_t = move_t + dwell_t

            log.info(f"Waypoint {i + 1}/{len(plan)} -> {x_to}")

            t0 = time.monotonic()
            dwell_cur, dwell_cmd, dwell_grav = [], [], []

            while True:
                now = time.monotonic()
                elapsed = now - t0

                if elapsed > segment_t:
                    break

                # ---- read --------------------------------------------
                state = bus.read_state()
                if state is None:
                    # Re-send the previous command rather than leaving it
                    # or commanding zero, which would drop the arm.
                    bus.write_currents(last_command)
                    time.sleep(dt)
                    continue

                q, qd, cur = state
                last_state = state

                # ---- setpoint ----------------------------------------
                s = min(1.0, elapsed / move_t) if move_t > 0 else 1.0
                x_d = lerp_xyz(x_from, x_to, s)

                # ---- current tip state -------------------------------
                x = fk(q)
                jm = jacobian(q)
                xd = j_times(jm, qd)

                err = [a - b for a, b in zip(x_d, x)]
                err_norm = math.sqrt(sum(e * e for e in err))
                stats_max_err = max(stats_max_err, err_norm)

                pitch = pitch_of(q)
                pitch_err = PITCH_RAD - pitch
                stats_max_pitch_err = max(stats_max_pitch_err, abs(pitch_err))

                # ---- safety ------------------------------------------
                if err_norm > MAX_TIP_ERROR_M:
                    raise RuntimeError(
                        f"Tip error {err_norm * 1000:.0f} mm exceeds the "
                        f"{MAX_TIP_ERROR_M * 1000:.0f} mm limit. Stopping."
                    )

                # The old script had no check on this at all, which is
                # why a nullspace collapse could grow unnoticed: it moves
                # the tip very little, so err_norm stays small.
                if abs(pitch_err) > MAX_PITCH_ERROR_RAD:
                    raise RuntimeError(
                        f"Tip pitch is {pitch:+.3f} rad against a commanded "
                        f"{PITCH_RAD:+.3f}. The arm is collapsing in the "
                        "null space. Stopping."
                    )

                for k, (angle, (lo, hi)) in enumerate(zip(q,
                                                          JOINT_LIMITS_RAD)):
                    if angle < lo + JOINT_LIMIT_MARGIN_RAD or \
                       angle > hi - JOINT_LIMIT_MARGIN_RAD:
                        raise RuntimeError(
                            f"joint{k + 1} = {angle:.4f} rad is at its "
                            "limit. Stopping."
                        )

                ramp = min(1.0, (now - start) / RAMP_TIME_S)

                # ---- integral ----------------------------------------
                # Held at zero until the ramp finishes, so it cannot wind
                # up during the mode switch. Clamped per axis.
                if ramp >= 1.0:
                    for k in range(3):
                        integral[k] = max(
                            -I_CLAMP,
                            min(I_CLAMP, integral[k] + KI_TIP[k] * err[k] * dt)
                        )

                # ---- Cartesian impedance -----------------------------
                #   f = Kc (x_d - x) + I - Dc xdot      (tip space)
                #   tau = J^T f                         (joint space)
                #
                # Damping uses MEASURED joint velocity through the
                # Jacobian, not a numerical derivative of the error. A
                # filtered derivative lags by roughly one filter time
                # constant, which at these frequencies turns damping into
                # energy injection.
                force = [KC[k] * err[k] + integral[k] - DC[k] * xd[k]
                         for k in range(3)]
                tau = jt_times(jm, force)

                # ---- joint damping, OUTSIDE J^T ----------------------
                # J^T has rank 3, so everything above is blind to
                # null(J). This term is not, and it is what stops the
                # internal collapse from running away.
                damp = [-KD_JOINT[k] * qd[k] for k in range(NJ)]

                # ---- nullspace posture PD ----------------------------
                # Hold the tip pitch using only torques that produce no
                # tip motion, so the Cartesian task is untouched.
                nvec = null_of_jacobian(jm)
                nmag = math.sqrt(sum(v * v for v in nvec))
                null_tau = [0.0] * NJ

                if KP_NULL != 0.0 and nmag > 1e-9:
                    nh = [v / nmag for v in nvec]
                    align = sum(a * b for a, b in zip(nh, PITCH_GRAD))
                    if abs(align) > NULL_MIN_ALIGN:
                        # Rescale so nh . grad_pitch == 1, i.e. one unit
                        # of scalar effort = one unit of pitch effort.
                        nh = [v / align for v in nh]
                        effort = (KP_NULL * pitch_err
                                  - KD_NULL * pitch_rate_of(qd))
                        null_tau = [v * effort for v in nh]

                grav = gravity_at(samples, q)

                imp = [ramp * tau[k] for k in range(NJ)]
                dmp = [ramp * damp[k] for k in range(NJ)]
                nll = [ramp * null_tau[k] for k in range(NJ)]

                command = [grav[k] + imp[k] + dmp[k] + nll[k]
                           for k in range(NJ)]

                if not bus.write_currents(command):
                    log.warn("Goal current write failed on one cycle.")
                last_command = command

                node.publish(q, qd, cur)

                # ---- logging -----------------------------------------
                if s >= 1.0:                    # dwell only: arm is still
                    dwell_cur.append(list(cur))
                    dwell_cmd.append(list(command))
                    dwell_grav.append(list(grav))

                period = now - prev_cycle
                prev_cycle = now
                periods.append(period)

                tracer.writerow(
                    [f"{now - start:.4f}", "run", i + 1, f"{s:.4f}"]
                    + [f"{v:.5f}" for v in x]
                    + [f"{pitch:.5f}"]
                    + [f"{v:.5f}" for v in q]
                    + [f"{v:.5f}" for v in qd]
                    + [f"{v:.1f}" for v in cur]
                    + [f"{v * CUR_UNIT_MA:.1f}" for v in cur]
                    + [f"{v:.5f}" for v in x_d]
                    + [f"{err_norm * 1000:.2f}",
                       f"{PITCH_RAD:.5f}", f"{pitch_err:.5f}"]
                    + [f"{v:.1f}" for v in command]
                    + [f"{v:.1f}" for v in grav]
                    + [f"{v:.1f}" for v in imp]
                    + [f"{v:.1f}" for v in dmp]
                    + [f"{v:.1f}" for v in nll]
                    + [f"{v:.1f}" for v in integral]
                    + [f"{period:.4f}"])

                # ---- pace --------------------------------------------
                sleep_for = dt - (time.monotonic() - now)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # ---- per-waypoint summary --------------------------------
            tip = fk(last_state[0])
            log.info(f"  measured tip: x={tip[0]:.4f} y={tip[1]:.4f} "
                     f"z={tip[2]:.4f} m  "
                     f"(error {math.dist(tip, x_to) * 1000:.1f} mm)")
            log.info(f"  pitch {pitch_of(last_state[0]):+.4f} rad "
                     f"(commanded {PITCH_RAD:+.4f})")
            log.info(f"  integral force: "
                     + ", ".join(f"{v:+6.1f}" for v in integral))

            m_cur = mean_of(dwell_cur)
            m_cmd = mean_of(dwell_cmd)
            m_grv = mean_of(dwell_grav)
            m_imp = [c - g for c, g in zip(m_cmd, m_grv)]
            log.info(f"  current over dwell ({len(dwell_cur)} samples), "
                     "units:")
            log.info("    measured  " + fmt(m_cur)
                     + f"   [{m_cur[1] * CUR_UNIT_MA:+.0f} mA on joint2]")
            log.info("    command   " + fmt(m_cmd))
            log.info("    gravity   " + fmt(m_grv))
            log.info("    residual  " + fmt(m_imp))

            saturated = [JOINT_NAMES[k] for k in range(NJ)
                         if abs(m_cmd[k]) > 0.9 * MAX_GOAL_CURRENT]
            if saturated:
                log.warn(f"    command near the +/-{MAX_GOAL_CURRENT} clamp "
                         f"on {saturated}. The clamp is shaping the "
                         "response, not the gains.")
    finally:
        trace.close()

    log.info(f"Largest tracking error: {stats_max_err * 1000:.1f} mm")
    log.info(f"Largest pitch error: {stats_max_pitch_err:.4f} rad")
    log.info(f"Read failures: {bus.read_failures}/{bus.read_attempts}")

    if periods:
        ordered = sorted(periods)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[int(0.95 * (len(ordered) - 1))]
        log.info(f"Loop period: median {p50 * 1000:.2f} ms, "
                 f"p95 {p95 * 1000:.2f} ms, target {1000.0 / CONTROL_HZ:.1f}")

    log.info(f"Trace written to {TRACE_FILE}")
    log.info(f"Plot it with:  python3 plot_arm.py {TRACE_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true",
                       help="Phase 1. Position control, record gravity "
                            "forwards and backwards.")
    group.add_argument("--run", action="store_true",
                       help="Phase 2. Current control, Cartesian impedance.")
    parser.add_argument("--no-park", action="store_true",
                        help="On exit, cut torque instead of holding "
                             "position. The arm will fall.")
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
        bus.park_on_exit = not args.no_park
        node = ImpedanceNode(bus)

        bus.ping_all()
        node.get_logger().info(f"All {NJ} servos answered.")

        if args.calibrate:
            calibrate(node, bus, plan)
        else:
            run_impedance(node, bus, plan)

    except KeyboardInterrupt:
        if node:
            node.get_logger().warn("Interrupted.")
        code = 130

    except Exception as exc:
        if node:
            node.get_logger().error(str(exc))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        code = 1

    finally:
        # Never leave the arm unsupported unless explicitly asked to.
        if bus:
            how = bus.shutdown()
            msg = f"Exit: {how}."
            if node:
                node.get_logger().warn(msg)
            else:
                print(msg, file=sys.stderr)
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(code)


if __name__ == "__main__":
    main()