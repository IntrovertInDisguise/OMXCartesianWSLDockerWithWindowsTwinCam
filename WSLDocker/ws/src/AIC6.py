#!/usr/bin/env python3
"""
AIC6.py

Cartesian impedance control for OpenMANIPULATOR-X, with joint-current based
tip-force estimation.  ROS 2 Humble.  Single file.  Talks to the Dynamixels
directly with dynamixel_sdk.  ros2_control must NOT be running.

    python3 AIC6.py --set-mode position  # PHYSICALLY SUPPORTED
    python3 AIC6.py --calibrate          # verifies position mode
    python3 AIC6.py --set-mode current   # PHYSICALLY SUPPORTED
    python3 AIC6.py --run                # verifies current mode; no EEPROM write
    python3 AIC6.py --kdl-check      # no hardware: compare KDL gravity to calibration
    python3 AIC6.py --replot DIR     # regenerate plots from a capture folder

Every run writes ./captures/<mode>_<timestamp>/ containing
    data.csv      one row per control cycle, every signal below
    meta.json     gains, geometry, constants, source of each
    plots/*.png   subplot figures, each titled with its source



====================================================================
AIC6 PERSISTENT-MODE CHANGE: NO OPERATING-MODE WRITE INSIDE --run
====================================================================

The AIC3/AIC4/AIC5 startup drop was caused by changing DYNAMIXEL Operating
Mode while the arm was unsupported. Operating Mode is EEPROM/non-volatile, so
AIC6 moves that write into an explicit maintenance command that must be run
with the arm physically supported:

    --set-mode position   # EEPROM write, leaves torque OFF
    --calibrate           # verifies position mode; no EEPROM write
    --set-mode current    # EEPROM write, leaves torque OFF
    --run                 # verifies current mode; no EEPROM write

AIC6 also separates restoring feedback from damping during the startup ramp:

    F = ramp * (K*e + integral) - D*v_ctrl

so damping is never ramped away. Coulomb-friction feedforward is disabled
during fast transients (max |qd_ctrl| > FRICTION_GATE_RAD_S), because friction
compensation acts in the direction of motion and would otherwise cancel some
of the passive friction needed to arrest a disturbance. Before the waypoint
clock starts, AIC6 holds KDL gravity plus full Cartesian/joint damping until
position-derived joint speed stays below STARTUP_SETTLE_QD_RAD_S for the
configured number of consecutive cycles.

AIC6 also removes the hidden EEPROM write from park(): a CURRENT-mode run never
switches back to POSITION mode in the air. On exit it leaves KDL gravity current
with torque ON as an interim support state; physically support the arm before
changing mode or removing power.

====================================================================
AIC4 CHANGE: KDL/URDF GRAVITY IN --run
====================================================================

AIC4 keeps the AIC3 waypoint schedule, impedance law, measured-velocity
damping, bounded integral, joint regularisation, Coulomb-friction
feedforward, joint-limit barrier, gravity-preserving clamp, current-based
force estimator, logging, plotting and abort/park behaviour.

The one controller-source change is gravity compensation during --run:

    AIC3: g(q) = interpolation of the bidirectional hold-current table
    AIC4: g(q) = KDL::ChainDynParam(URDF, [0,0,-9.81]) * TORQUE_SCALE

The KDL gravity vector is recomputed from the MEASURED q every servo cycle,
so recovery after the operating-mode drop is no longer an off-table
extrapolation/projection problem. --calibrate is still retained because its
forward/reverse half-difference supplies the Coulomb-friction table. The
calibration half-sum is kept only as a diagnostic comparison against KDL; it
is not used as gravity feedforward in --run.

The current-control mode-switch seeding is also evaluated at the MEASURED
post-switch q: after the EEPROM write returns with torque OFF, AIC4 reads q,
computes KDL gravity at that dropped posture, writes Goal Current, then
enables torque. This removes AIC3's stale pre-drop gravity seed.

Historical note: AIC5 tried to mitigate the in-air EEPROM drop with a
position-control clearance pose. AIC6 removes that mechanism entirely; --run
starts in already-configured CURRENT mode and the first Cartesian segment begins
from the measured current-mode entry pose.

====================================================================
CONTROLLER RELATION TO omx_variable_stiffness_controller.cpp
====================================================================

AIC4 intentionally borrows several controller ideas from the C++ controller
(J^T Cartesian impedance, torque_scale, joint-space regularisation,
gravity-preserving saturation and a joint-limit barrier), but it is NOT an
exact Python port of that controller. Important controller-level differences:

  * AIC4 controls XYZ translation (3x4 Jacobian) and carries pitch through
    IK/joint regularisation. The C++ controller uses a 6D Cartesian wrench
    with explicit rotational stiffness/damping.
  * AIC4 damping uses measured Cartesian velocity J*qdot. The C++ controller
    uses a low-pass filtered derivative of Cartesian pose error.
  * AIC4 has no manipulability-triggered DLS projection or singularity-escape
    joint-space mode. Instead it preflights IK paths and uses joint limits.
  * AIC4 gravity now matches the C++ source structurally: KDL/URDF
    ChainDynParam evaluated from measured q every cycle. AIC4 still adds the
    measured Coulomb-friction feedforward and bounded Cartesian integral,
    which the C++ normal impedance law does not.
  * AIC4's measured-current tip-force estimate is different from the C++
    deflection/commanded-wrench contact estimate.

Therefore: structurally similar in the ordinary, well-conditioned Cartesian
impedance path, but not controller-identical. The waypoint change below does
not otherwise replace AIC3's controller law.


====================================================================
WHAT CHANGED FROM AIC2, AND WHY
====================================================================

AIC2 stalled: the tip reached waypoint 1 and then refused to advance.
Four separate causes, all now addressed.

(1) THE UNIT SYSTEM WAS UNKNOWN, SO THE GAINS WERE UNCALIBRATED.

    AIC2's KC was "Dynamixel current units per metre" with no way to
    reach N/m.  The working C++ controller
    (omx_variable_stiffness_controller) computes force in newtons and
    multiplies the resulting joint torque by

        torque_scale_ = 180.0        # raw Dynamixel units per N.m

    That is the missing constant.  It implies 1/180 N.m per unit =
    5.56 mN.m/unit, and at 2.69 mA/unit that is about 2.07 N.m/A.  The
    XM430-W350 datasheet stall figures give roughly 1.8 N.m/A, so it is
    the right size.  Cross-check against this arm's own calibration:
    measured joint-2 hold current of -117.6 units / 180 = 0.65 N.m,
    which matches a rough hand estimate of the arm mass beyond joint 2
    (about 0.5 kg at 0.15 m -> 0.74 N.m).

    TORQUE_SCALE is therefore USED but NOT PROVEN.  See --fit-scale in
    the notes at the bottom of this docstring for how to measure it.

(2) THE STIFFNESS WAS AN ORDER OF MAGNITUDE BELOW THE STICTION.

    With TORQUE_SCALE known, AIC2's KC = [5000, 1000, 100] units/m is
    [27.8, 5.6, 0.56] N/m.  The C++ YAML, which ran on this arm, uses
    [300, 25.3, 60] N/m -- and its own comment records why:

        "at K_x=20 the impedance law could only pull with ~1 N at full
         5 cm error, below the measured joint-stiction stall force
         (~1.0-1.1 N at the EE)"

    K_x = 20 N/m is 3600 units/m.  AIC2 had 5000.  Same stall, twice.

    Independent check from AIC2's own calibration: joint-3 stiction is
    +/-13 to 29 current units.  Through the joint-3 x lever arm of
    0.101 m that is 197 units = 1.09 N of tip force needed just to
    break away.  The C++ number and this number agree.

(3) WAYPOINT STOPS ARE ENABLED AGAIN -- DELIBERATELY.

    This replacement restores the AIC2-style waypoint schedule because
    the experiment requires stops.  Every waypoint has an editable XYZ,
    move time and dwell time.  Each move still uses AIC3's quintic
    10s^3-15s^4+6s^5, with zero velocity and acceleration at both ends.

    Stopping means stiction can re-set at every dwell, so each restart can
    require another breakaway.  That tradeoff is intentional here; AIC4's
    friction feedforward, SI-scaled impedance, gravity-preserving clamp and
    joint regularisation remain active during every move and dwell.

(4) THE FRICTION WAS MEASURED AND THEN THROWN AWAY.

    AIC2's two-pass calibration computes (fwd - rev)/2 per joint per
    pose.  That IS the Coulomb friction magnitude.  AIC2 logged it and
    used nothing.  AIC4 feeds it forward:

        tau_friction = FRICTION_FF * f_c(q) * tanh(qdot / FRICTION_EPS)

    tanh, not sign: Present Velocity is quantised at 0.024 rad/s and a
    hard sign chatters.  FRICTION_FF < 1 on purpose -- over-compensating
    Coulomb friction produces a limit cycle.

ALSO NEW

  * GRAVITY-PRESERVING CLAMP.  AIC2 clamped the total command, so a
    large impedance term starved the gravity feedforward and the arm
    sagged.  From the C++: split tau = G + PD, clamp G, then scale only
    PD by a single global alpha so every joint stays coordinated.

  * JOINT-LIMIT BARRIER.  A restoring spring inside the limit margin,
    instead of AIC2's hard abort.  Abort is kept, but only well past
    the barrier.

  * JOINT-SPACE REGULARISATION replaces AIC2's nullspace pitch PD.
    q_ref comes from closed-form IK of the current Cartesian setpoint,
    so it constrains all four joints -- including the nullspace
    direction that J^T cannot reach -- and it brings the correct pitch
    with it for free.  Clamped per joint at REG_FRACTION of the limit
    so it can never starve the Cartesian task (also from the C++).

  * PERSISTENT CURRENT MODE. Operating Mode is never switched inside --run.
    The first waypoint move starts from the measured current-mode entry pose
    and ramps pitch to PITCH_RAD, so there is no designed-in EEPROM free-fall
    recovery segment.

  * TIP FORCE ESTIMATE FROM MEASURED CURRENT.  No force sensor.

        tau_ext = I_measured - g(q) - f_c(q)*tanh(qdot/eps)
        F_applied = pinv(J^T) . tau_ext / TORQUE_SCALE       [newtons]

    J^T is 4x3, so this is a least-squares solve of (J J^T) f = J tau.
    This is a MEASUREMENT: it reads motor current and is independent of
    the impedance gains.  The C++ published K*x_err instead, which is
    the commanded force, not a measured one.  Both are logged here.

    ACCURACY.  Computed from this arm's own stiction numbers, worst
    case over sign combinations, through the least-squares inverse:

        current quantisation (1 unit)   ~ +/-0.05 N in x
        residual joint friction         ~ +/-2.0 to 2.5 N in x

    The friction term dominates by a factor of 40.  Treat the absolute
    force as indicative only.  For anything quantitative, run the same
    waypoint schedule twice -- once loaded, once unloaded -- and subtract at
    matched q.  Gravity and friction are common-mode and largely
    cancel; what is left is the load.  Both raw and friction-
    compensated estimates are logged so the size of the correction is
    visible.


--------------------------------------------------------------------
MEASURING TORQUE_SCALE (recommended, not automated here)

The calibration capture holds the arm at the configured waypoint poses
against gravity alone.  Compute the model gravity torque in N.m at
each of those poses from the URDF link masses and COM offsets, then
regress measured current against modelled N.m.  The slope is
TORQUE_SCALE, measured on your hardware.  N_waypoints x 4 equations,
one unknown.  Until that is done, every newton in this file inherits
the uncertainty of the borrowed 180.
--------------------------------------------------------------------
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
# HARDWARE
# ============================================================

PORT_NAME = "/dev/ttyUSB0"
BAUD_RATE = 1000000
DXL_IDS = [11, 12, 13, 14]          # joint1..joint4. Gripper (15) untouched.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
NJ = len(DXL_IDS)

# Reuse the AIC3 calibration file. In AIC5 --run, only its
# friction_current field is required for control; gravity_current is diagnostic.
CALIB_FILE = os.path.expanduser("~/.omx_gravity_calib_v3.json")
CAPTURE_ROOT = os.path.abspath("./captures")

# KDL gravity model. Override the xacro path without editing this file with
#   export OMX_URDF_XACRO=/absolute/path/to/open_manipulator_x.urdf.xacro
KDL_DESCRIPTION_PACKAGE = "open_manipulator_x_description"
KDL_XACRO_BASENAME = "open_manipulator_x.urdf.xacro"
KDL_ROOT_LINK = os.environ.get("OMX_KDL_ROOT_LINK", "link1")
KDL_TIP_LINK = os.environ.get("OMX_KDL_TIP_LINK", "end_effector_link")
KDL_GRAVITY_MPS2 = [0.0, 0.0, -9.81]


# ============================================================
# UNITS
# ============================================================

# Dynamixel current units per N.m.
# SOURCE: torque_scale_ = 180.0 in omx_variable_stiffness_controller.cpp,
# which ran on this arm.  NOT a datasheet value.  See docstring.
TORQUE_SCALE = 180.0

POS_UNITS_PER_REV = 4096.0          # SOURCE: ROBOTIS e-Manual, X series
POS_CENTER = 2048.0
VEL_UNIT_RPM = 0.229                # rev/min per Present Velocity unit
CUR_UNIT_MA = 2.69                  # mA per current unit


# ============================================================
# TRAJECTORY
# ============================================================

# Tip pitch, radians. 0.0 = pointing front. -pi/2 = pointing down.
PITCH_RAD = 0.0

# Position-control clearance used ONLY before switching Operating Mode.
# The observed AIC3 torque-off window dropped the tip ~0.041 m. A 0.050 m
# pre-lift keeps the nominal waypoint schedule unchanged while giving the arm
# clearance during the unavoidable EEPROM torque-off window. Adjust if your
# measured mode-switch drop changes; this is not part of the spring stroke.
MODE_SWITCH_CLEARANCE_M = 0.050

# AIC2-style absolute Cartesian waypoints in metres, link1 frame.
#
# MOVE_TIMES_S[i] is the quintic move time FROM the measured tip pose at
# the start of segment i TO CARTESIAN_WAYPOINTS_M[i].
# DWELL_TIMES_S[i] is the stop/hold time at that waypoint.
#
# IMPORTANT: waypoint 1 is also the recovery target after the Operating
# Mode EEPROM write drops the arm. During that first move, pitch ramps from
# the measured dropped pitch to PITCH_RAD, preserving AIC3's joint-4 fix.
CARTESIAN_WAYPOINTS_M: List[List[float]] = [
    [0.220, 0.000, 0.10],
    [0.230, 0.000, 0.10],
    [0.240, 0.000, 0.10],
    [0.250, 0.000, 0.10],
    [0.260, 0.000, 0.10],
    [0.270, 0.000, 0.10],
    [0.280, 0.000, 0.10],
    [0.290, 0.000, 0.10],
    [0.300, 0.000, 0.10],
    [0.310, 0.000, 0.10],
    [0.320, 0.000, 0.10],
]

# One entry per waypoint. Edit these independently.
MOVE_TIMES_S = [10.0] * 11
DWELL_TIMES_S = [5.0] * 11

# NOTE ON WORKSPACE. The C++ runner actually worked over x = 0.165 to
# 0.265 at z = 0.09. At x = 0.320 the arm has less x authority. If it
# lags badly near the far end, move the waypoints back before raising Kx.

# Gravity/friction calibration uses the SAME waypoint locations. The table
# interpolates continuously between the calibrated joint poses. Calibration
# timing is intentionally separate from experimental move/dwell timing.
CALIB_MOVE_TIME_S = 10.0            # position-mode settle per sample
CALIB_HOLD_TIME_S = 1.0             # averaging window per sample

# Position-mode approach to waypoint 1 before switching Operating Mode.
APPROACH_TIME_S = 6.0


# ============================================================
# GAINS
# ============================================================
#
# Written in SI. Converted to current units at the bottom of this block
# by multiplying by TORQUE_SCALE, so the numbers here can be compared
# directly against the C++ YAML.

# Cartesian stiffness, N/m. [x, y, z].
# SOURCE: fixed_stiffness_x/y/z in robot1_variable_stiffness.yaml, which
# ran on this arm. x = 300 was itself raised from 20 after the arm
# stalled on stiction -- the same failure AIC2 hit.
K_CART_NPM = [300.0, 25.0, 60.0]

# Cartesian damping, N.s/m.
# SOURCE: fixed_damping_x/y/z in the same YAML.
# D_CART_NSPM = [15.0, 3.0, 15.0]
# Damping is delay-limited: its discrete stability margin scales as
# b*dt/I. The source YAML ran at update_rate: 500. This loop runs at
# CONTROL_HZ = 100, so damping is scaled by CONTROL_HZ/500. Stiffness is
# NOT scaled; it is not the delay-sensitive term.
SOURCE_CONTROL_HZ = 500.0
_DAMP_SCALE = 100.0 / SOURCE_CONTROL_HZ      # keep in step with CONTROL_HZ
D_CART_NSPM = [15.0 * _DAMP_SCALE, 3.0 * _DAMP_SCALE, 15.0 * _DAMP_SCALE]
# Integral on tip error, N per (m.s). The C++ has no integral because
# its gravity comes from a KDL model. This one patches a measured table,
# so a small bounded integral is kept to absorb table error.
KI_CART = [0.0, 0.0, 0.0]
# KI_CART = [8.0, 16.0, 16.0]
I_CLAMP_N = 1.5                     # max integral force per axis, newtons

# Joint-space regularisation toward IK(x_d). Replaces AIC2's nullspace
# pitch PD: it constrains all four joints, including the direction J^T
# cannot reach, and carries the commanded pitch with it.
# SOURCE: trajectory_joint_reg_K / _D in the same YAML.
KP_JOINT_NM_RAD = 1.0
KD_JOINT_NMS_RAD = 0.3 * _DAMP_SCALE
# Per-joint cap on the regularisation term, as a fraction of the torque
# limit. Without it one badly-tracking joint drags the global clamp
# alpha down and starves every joint of Cartesian authority.
# SOURCE: reg_limit = max_joint_torque_command_ * 0.4 in the C++.
REG_FRACTION = 0.4

# Coulomb friction feedforward.
# f_c(q) comes from the two-pass calibration: (fwd - rev)/2 per joint.
# Under 1.0 on purpose -- over-compensating Coulomb friction turns the
# joint into a limit cycle.
FRICTION_FF = 0.75
# tanh width, rad/s. Present Velocity is quantised at
# 0.229 rpm = 0.024 rad/s, so anything below that is indistinguishable
# from standstill. Set to twice the quantum.
FRICTION_EPS = 0.048
FRICTION_GATE_RAD_S = 0.60           # disable Coulomb FF during fast transients
STARTUP_SETTLE_QD_RAD_S = 0.15       # near-stillness before waypoint clock starts
STARTUP_SETTLE_CYCLES = 20           # consecutive quiet cycles
STARTUP_SETTLE_TIMEOUT_S = 5.0       # fail closed if current-mode start never settles
CALIB_REVERSE_OVERSHOOT_SPACINGS = 2.0 # reverse pass must genuinely approach wp11

# Joint-limit barrier spring, N.m/rad, active inside BARRIER_MARGIN_RAD.
# SOURCE: the joint-limit barrier in the C++ update loop.
K_BARRIER_NM_RAD = 8.0
BARRIER_MARGIN_RAD = 0.15
# check_path() refuses a path that comes inside BARRIER_MARGIN_RAD of a
# limit, and warns below this. AIC2's failing recover had 0.165 rad of
# joint-4 headroom -- legal, but not enough to absorb any overshoot.
PATH_HEADROOM_WARN_RAD = 0.30

# --- converted to Dynamixel current units ---------------------------
KC = [k * TORQUE_SCALE for k in K_CART_NPM]        # units per m
DC = [d * TORQUE_SCALE for d in D_CART_NSPM]       # units per (m/s)
KI = [k * TORQUE_SCALE for k in KI_CART]           # units per (m.s)
I_CLAMP = I_CLAMP_N * TORQUE_SCALE                 # units
KP_JOINT = KP_JOINT_NM_RAD * TORQUE_SCALE          # units per rad
KD_JOINT = KD_JOINT_NMS_RAD * TORQUE_SCALE         # units per (rad/s)
K_BARRIER = K_BARRIER_NM_RAD * TORQUE_SCALE        # units per rad


# ============================================================
# SAFETY
# ============================================================

CONTROL_HZ = 100.0
assert abs(CONTROL_HZ - 100.0) < 1e-9, \
    "_DAMP_SCALE hardcodes 100.0; update it if CONTROL_HZ changes."
# Per-joint command clamp, current units.
# AIC2 used 250 (0.67 A), which was below the force the task needs.
# The C++ used max_joint_torque_command = 900 (2.42 A). 600 is a
# deliberate middle: enough authority, less thermal exposure over a
# a long waypoint schedule. The servo Current Limit reads 1193, so this is legal.
# THERMAL HEADROOM IS NOT VERIFIED. Watch for hardware error 0x04.
MAX_GOAL_CURRENT = 600

RAMP_TIME_S = 2.0                   # fade impedance in from zero

# Controller velocity source. Dynamixel Present Velocity showed ~51.7 ms
# lag in the failed AIC4 capture at a 9.34 Hz vertical oscillation, i.e.
# ~174 deg of phase lag. AIC5 therefore differentiates measured Present
# Position causally and low-pass filters that derivative. Present Velocity
# remains logged as a diagnostic only.
VELOCITY_FILTER_CANDIDATES_HZ = [15.0, 20.0, 25.0, 30.0, 40.0]
CONTROL_VEL_CUTOFF_HZ = 30.0

# Fast motion aborts added after the AIC4 vertical-banging run. These use
# the selected position-derived control velocity, not delayed servo velocity.
MAX_CTRL_VZ_M_S = 0.35
MAX_CTRL_QD_RAD_S = 2.5
FAST_ABORT_CONSECUTIVE_CYCLES = 3
CLAMP_ABORT_ALPHA = 0.75
CLAMP_ABORT_CONSECUTIVE_CYCLES = 5
MAX_TIP_ERROR_M = 0.080             # abort if the tip strays this far
MAX_PITCH_ERROR_RAD = 0.40          # abort if the posture collapses
HARD_LIMIT_MARGIN_RAD = 0.03        # abort this close to a joint limit
HW_CHECK_INTERVAL_S = 2.0           # how often to read the error register

# Profile for position-control moves. Units: velocity 0.229 rpm,
# acceleration 214.577 rev/min^2. 0 means "no limit", which is what we
# are avoiding: calibration current after a slam is not the same number
# as after a gentle approach.
PROFILE_VELOCITY = 40               # about 9.2 rpm
PROFILE_ACCELERATION = 20

# Joint limits, radians. VERIFY against your URDF before trusting these.
#   grep -n "limit" open_manipulator_x.urdf.xacro
JOINT_LIMITS_RAD = [
    (-2.80, 2.80),   # joint1
    (-1.75, 1.60),   # joint2
    (-1.60, 1.50),   # joint3
    (-1.70, 2.00),   # joint4
]


# ============================================================
# DYNAMIXEL CONTROL TABLE (X series)
# VERIFY in the ROBOTIS e-Manual for XM430-W350 before trusting.
# ============================================================

ADDR_OPERATING_MODE = 11      # 1 byte.  0 = current, 3 = position
ADDR_CURRENT_LIMIT = 38       # 2 bytes, EEPROM
ADDR_TORQUE_ENABLE = 64       # 1 byte
ADDR_HW_ERROR = 70            # 1 byte
ADDR_GOAL_CURRENT = 102       # 2 bytes, signed, RAM
ADDR_PROFILE_ACCEL = 108      # 4 bytes
ADDR_PROFILE_VELOCITY = 112   # 4 bytes
ADDR_GOAL_POSITION = 116      # 4 bytes
ADDR_PRESENT_BLOCK = 126      # Present Current(2) Velocity(4) Position(4)
LEN_PRESENT_BLOCK = 10

MODE_CURRENT = 0
MODE_POSITION = 3

HW_ERROR_BITS = {
    0x01: "input voltage",
    0x04: "overheating",
    0x08: "motor encoder",
    0x10: "electrical shock",
    0x20: "overload",
}


def to_signed(value: int, bits: int) -> int:
    limit = 1 << bits
    return value - limit if value >= limit // 2 else value


def pos_to_rad(raw: int) -> float:
    return (raw - POS_CENTER) * 2.0 * math.pi / POS_UNITS_PER_REV


def rad_to_pos(rad: float) -> int:
    return int(round(rad * POS_UNITS_PER_REV / (2.0 * math.pi) + POS_CENTER))


def vel_to_rad_s(raw: int) -> float:
    return raw * VEL_UNIT_RPM * 2.0 * math.pi / 60.0


def decode_hw_error(value: int) -> str:
    if value == 0:
        return "none"
    names = [text for bit, text in HW_ERROR_BITS.items() if value & bit]
    return ", ".join(names) if names else f"unknown (0x{value:02X})"


# ============================================================
# GEOMETRY
# Values read from the joint origins in
#   open_manipulator_x_description/urdf/open_manipulator_x.urdf.xacro
# The sign convention is checked in _self_check() against a MoveIt
# solution recorded from this physical arm, so it is verified on
# hardware, not against an assumption.
# ============================================================

BASE_X = 0.012
BASE_Z = 0.017 + 0.0595
L1 = math.hypot(0.024, 0.128)         # 0.130225, offset upper link
DELTA = math.atan2(0.024, 0.128)      # 0.185348, its built-in tilt
L2 = 0.124
L3 = 0.126

MAX_WRIST_REACH = L1 + L2
MIN_WRIST_REACH = abs(L1 - L2)


def link_angles(q: Sequence[float]) -> Tuple[float, float, float]:
    a = math.pi / 2.0 - DELTA - q[1]
    b = a - (math.pi / 2.0 - DELTA) - q[2]
    c = b - q[3]
    return a, b, c


def pitch_of(q: Sequence[float]) -> float:
    """Absolute elevation of the last link. Depends only on q2+q3+q4, and
    jacobian() has no row for it. _self_check() asserts this agrees with
    link_angles()[2]."""
    return -(q[1] + q[2] + q[3])


def fk(q: Sequence[float]) -> List[float]:
    """Tip position in metres, link1 frame."""
    a, b, c = link_angles(q)
    r = BASE_X + L1 * math.cos(a) + L2 * math.cos(b) + L3 * math.cos(c)
    z = BASE_Z + L1 * math.sin(a) + L2 * math.sin(b) + L3 * math.sin(c)
    return [r * math.cos(q[0]), r * math.sin(q[0]), z]


def jacobian(q: Sequence[float]) -> List[List[float]]:
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


def jt_times(j: List[List[float]], f: Sequence[float]) -> List[float]:
    """J^T f -> joint torques from a tip force."""
    return [sum(j[row][col] * f[row] for row in range(3)) for col in range(4)]


def j_times(j: List[List[float]], qd: Sequence[float]) -> List[float]:
    """J qdot -> tip velocity from joint velocity."""
    return [sum(j[row][col] * qd[col] for col in range(4)) for row in range(3)]


def solve_jt_lstsq(j: List[List[float]], tau: Sequence[float],
                   ridge: float = 1e-9) -> List[float]:
    """Least-squares f minimising ||J^T f - tau||.

    J^T is 4x3, so the system is overdetermined and there is generally no
    exact f. The normal equations are (J J^T) f = J tau, a 3x3 solve.
    A tiny ridge keeps it defined if the arm ever nears a singularity.

    This is the inverse of jt_times() in the least-squares sense, and it
    is what turns measured joint current into an estimated tip force.
    """
    a = [[sum(j[r][k] * j[c][k] for k in range(4)) + (ridge if r == c else 0.0)
          for c in range(3)] for r in range(3)]
    b = [sum(j[r][k] * tau[k] for k in range(4)) for r in range(3)]

    m = [a[i][:] + [b[i]] for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        if abs(m[i][i]) < 1e-15:
            return [float("nan")] * 3
        for r in range(3):
            if r != i:
                f = m[r][i] / m[i][i]
                for c in range(i, 4):
                    m[r][c] -= f * m[i][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def ik_pitch(x: float, y: float, z: float,
             pitch: float) -> Optional[List[float]]:
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


def quintic(s: float) -> float:
    """Smoothstep 10s^3 - 15s^4 + 6s^5.

    Zero velocity AND zero acceleration at both ends. AIC2 used a cosine
    ease, which has zero velocity but a step in acceleration -- an
    impulse in commanded force at each end of a segment.
    SOURCE: the quintic in the C++ MOVE_FORWARD state.
    """
    s = max(0.0, min(1.0, s))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def lerp_xyz(a: Sequence[float], b: Sequence[float], s: float) -> List[float]:
    w = quintic(s)
    return [u + (v - u) * w for u, v in zip(a, b)]


def calibration_samples() -> List[List[float]]:
    """Calibration vertices: exactly the configured waypoint locations.

    This mirrors AIC2: if a waypoint location changes, the gravity/friction
    calibration for that path must change with it.  table_at() interpolates
    continuously between these vertices during the run.
    """
    return [list(xyz) for xyz in CARTESIAN_WAYPOINTS_M]


def validate_waypoint_settings():
    """Fail fast on malformed waypoint/timing configuration."""
    n = len(CARTESIAN_WAYPOINTS_M)
    if n < 1:
        raise RuntimeError("CARTESIAN_WAYPOINTS_M must contain at least one point.")
    if not (len(MOVE_TIMES_S) == len(DWELL_TIMES_S) == n):
        raise RuntimeError(
            "Waypoint, MOVE_TIMES_S and DWELL_TIMES_S lengths differ: "
            f"{n}, {len(MOVE_TIMES_S)}, {len(DWELL_TIMES_S)}.")
    for i, xyz in enumerate(CARTESIAN_WAYPOINTS_M):
        if len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz):
            raise RuntimeError(f"Waypoint {i + 1} must be three finite XYZ values: {xyz}")
        if MOVE_TIMES_S[i] < 0.0 or DWELL_TIMES_S[i] < 0.0:
            raise RuntimeError(
                f"Waypoint {i + 1} has a negative move/dwell time: "
                f"{MOVE_TIMES_S[i]}, {DWELL_TIMES_S[i]}.")


def _add_system_python_paths_for_ros_packages():
    """Best-effort compatibility for venvs that hide apt-installed ROS modules.

    This does not install anything. It only exposes the standard system/ROS
    Python paths when they exist and are not already visible.
    """
    candidates = ["/usr/lib/python3/dist-packages"]
    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates.append(f"/opt/ros/{ros_distro}/lib/{pyver}/site-packages")
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)


def _resolve_omx_xacro() -> str:
    """Locate the installed/source OpenMANIPULATOR-X xacro deterministically."""
    override = os.environ.get("OMX_URDF_XACRO", "").strip()
    if override:
        path = os.path.abspath(os.path.expanduser(override))
        if not os.path.isfile(path):
            raise RuntimeError(f"OMX_URDF_XACRO does not exist: {path}")
        return path

    candidates = []
    try:
        _add_system_python_paths_for_ros_packages()
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory(KDL_DESCRIPTION_PACKAGE)
        candidates.extend([
            os.path.join(share, "urdf", KDL_XACRO_BASENAME),
            os.path.join(share, KDL_XACRO_BASENAME),
        ])
    except Exception:
        pass

    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    relative = Path("open_manipulator") / KDL_DESCRIPTION_PACKAGE / "urdf" / KDL_XACRO_BASENAME
    relative_pkg = Path(KDL_DESCRIPTION_PACKAGE) / "urdf" / KDL_XACRO_BASENAME
    candidates.extend(str(base / rel) for base in (here, cwd, cwd / "src", here.parent)
                      for rel in (relative, relative_pkg))

    seen = set()
    for candidate in candidates:
        path = os.path.abspath(os.path.expanduser(candidate))
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    tried = "\n  ".join(sorted(seen))
    raise RuntimeError(
        "Could not locate open_manipulator_x.urdf.xacro for KDL gravity. "
        "Set OMX_URDF_XACRO to the absolute xacro path. Tried:\n  " + tried)



def _urdf_to_kdl_pose(kdl, pose):
    rpy = pose.rpy if pose and pose.rpy and len(pose.rpy) == 3 else [0.0, 0.0, 0.0]
    xyz = pose.xyz if pose and pose.xyz and len(pose.xyz) == 3 else [0.0, 0.0, 0.0]
    return kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))


def _urdf_to_kdl_inertia(kdl, inertial):
    origin = _urdf_to_kdl_pose(kdl, inertial.origin)
    inertia = inertial.inertia
    rot = kdl.RotationalInertia(
        inertia.ixx, inertia.iyy, inertia.izz,
        inertia.ixy, inertia.ixz, inertia.iyz)
    return origin.M * kdl.RigidBodyInertia(inertial.mass, origin.p, rot)


def _urdf_to_kdl_joint(kdl, joint):
    frame = _urdf_to_kdl_pose(kdl, joint.origin)
    if joint.type in ("revolute", "continuous"):
        return kdl.Joint(
            joint.name, frame.p, frame.M * kdl.Vector(*joint.axis),
            kdl.Joint.RotAxis)
    if joint.type == "prismatic":
        return kdl.Joint(
            joint.name, frame.p, frame.M * kdl.Vector(*joint.axis),
            kdl.Joint.TransAxis)
    return kdl.Joint(joint.name, kdl.Joint.Fixed)


def _add_urdf_children_to_kdl_tree(kdl, robot, link, tree):
    inertia = kdl.RigidBodyInertia(0.0)
    if link.inertial:
        inertia = _urdf_to_kdl_inertia(kdl, link.inertial)

    parent_joint_name, parent_link_name = robot.parent_map[link.name]
    parent_joint = robot.joint_map[parent_joint_name]
    segment = kdl.Segment(
        link.name,
        _urdf_to_kdl_joint(kdl, parent_joint),
        _urdf_to_kdl_pose(kdl, parent_joint.origin),
        inertia)
    if not tree.addSegment(segment, parent_link_name):
        return False

    for _, child_name in robot.child_map.get(link.name, []):
        if not _add_urdf_children_to_kdl_tree(
                kdl, robot, robot.link_map[child_name], tree):
            return False
    return True


def _chain_from_urdf_model(kdl, robot, root_link: str, tip_link: str):
    """Build only root_link -> tip_link in memory.

    This deliberately does NOT call URDF.get_root(). Some OpenMANIPULATOR-X
    xacro expansions used as component descriptions are not a single globally
    rooted robot, even though the manipulator chain itself is complete. KDL
    gravity only needs the requested serial chain. No source xacro/URDF file is
    edited; all normalization and chain construction is internal to AIC4.
    """
    if root_link not in robot.link_map:
        raise RuntimeError(
            f"KDL root link {root_link!r} is absent from expanded description. "
            f"Available links: {sorted(robot.link_map.keys())}")
    if tip_link not in robot.link_map:
        raise RuntimeError(
            f"KDL tip link {tip_link!r} is absent from expanded description. "
            f"Available links: {sorted(robot.link_map.keys())}")

    # Walk tip -> root through URDF parent_map, then reverse to root -> tip.
    edges = []
    child = tip_link
    visited = set()
    while child != root_link:
        if child in visited:
            raise RuntimeError(
                f"Cycle encountered while tracing {root_link}->{tip_link} at {child!r}.")
        visited.add(child)
        if child not in robot.parent_map:
            raise RuntimeError(
                f"No parent path from tip {tip_link!r} back to root {root_link!r}; "
                f"trace stopped at {child!r}. The expanded description may be a "
                "macro/component definition rather than the instantiated robot.")
        joint_name, parent_name = robot.parent_map[child]
        edges.append((joint_name, parent_name, child))
        child = parent_name
    edges.reverse()

    chain = kdl.Chain()
    for joint_name, parent_name, child_name in edges:
        joint = robot.joint_map[joint_name]
        link = robot.link_map[child_name]
        inertia = kdl.RigidBodyInertia(0.0)
        if link.inertial:
            inertia = _urdf_to_kdl_inertia(kdl, link.inertial)
        chain.addSegment(kdl.Segment(
            child_name,
            _urdf_to_kdl_joint(kdl, joint),
            _urdf_to_kdl_pose(kdl, joint.origin),
            inertia))

    return chain


class KDLGravityModel:
    """URDF/KDL gravity torque model matching the C++ ChainDynParam structure.

    Returns gravity in both N.m and raw Dynamixel-current command units.
    TORQUE_SCALE remains the same borrowed 180 raw units/N.m used by AIC3/C++.
    """

    def __init__(self, log=None):
        _add_system_python_paths_for_ros_packages()
        try:
            import PyKDL
            import xacro
            from urdf_parser_py.urdf import URDF
        except Exception as exc:
            raise RuntimeError(
                "AIC5 needs PyKDL, xacro and urdf_parser_py. ROS 2 Humble "
                "does not provide kdl_parser_py, so AIC5 carries the small "
                "URDF->PyKDL conversion internally. Source "
                "/opt/ros/humble/setup.bash and expose the ROS/system Python "
                f"packages to this venv. Original import error: {exc}") from exc

        self.PyKDL = PyKDL
        self.xacro_path = _resolve_omx_xacro()
        try:
            # This standard file is a COMPONENT xacro: it defines the
            # <xacro:macro name="open_manipulator_x" params="prefix=''"> macro
            # but intentionally instantiates no links when processed by itself.
            #
            # AIC5 therefore parses the source into an in-memory DOM, appends a
            # single in-memory macro invocation equivalent to the standard robot
            # wrapper's <xacro:open_manipulator_x prefix=""/>, and only then
            # asks xacro to expand that DOM.  Nothing is written to the xacro,
            # URDF, package share, workspace, or any other external file.
            doc = xacro.parse(None, self.xacro_path)
            root = doc.documentElement
            if root.tagName != "robot":
                raise RuntimeError(
                    f"Component xacro root is <{root.tagName}>, expected <robot>.")

            invocation = doc.createElementNS(
                "http://www.ros.org/wiki/xacro",
                "xacro:open_manipulator_x")
            invocation.setAttribute("prefix", "")
            root.appendChild(invocation)

            # process_doc mutates only this DOM.  The source file remains
            # byte-for-byte untouched.
            xacro.process_doc(doc)
            root = doc.documentElement

            # urdf_parser_py requires robot@name.  The component macro's outer
            # <robot> does not provide one, so normalize only this in-memory DOM.
            if not root.hasAttribute("name") or not root.getAttribute("name").strip():
                root.setAttribute("name", "open_manipulator_x_aic6")
                if log is not None:
                    log.info(
                        "In-memory expanded robot had no name; inserted parser-only "
                        "name='open_manipulator_x_aic6'.")

            xml = doc.toxml()
            robot = URDF.from_xml_string(xml)
            if log is not None:
                log.info(
                    "Instantiated open_manipulator_x(prefix='') in memory: "
                    f"{len(robot.links)} links, {len(robot.joints)} joints; "
                    "no external description file modified.")

            # Build only the requested manipulator serial chain.
            self.chain = _chain_from_urdf_model(
                PyKDL, robot, KDL_ROOT_LINK, KDL_TIP_LINK)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to expand/parse/build internal KDL chain from the "
                f"OpenMANIPULATOR-X xacro at {self.xacro_path}: {exc}") from exc

        if self.chain.getNrOfJoints() != NJ:
            raise RuntimeError(
                f"KDL chain {KDL_ROOT_LINK}->{KDL_TIP_LINK} has "
                f"{self.chain.getNrOfJoints()} movable joints, expected {NJ}. "
                "The gravity vector would not map safely to dxl1..dxl4.")

        moving_names = []
        for i in range(self.chain.getNrOfSegments()):
            joint = self.chain.getSegment(i).getJoint()
            if joint.getType() != PyKDL.Joint.Fixed:
                moving_names.append(joint.getName())
        if moving_names != JOINT_NAMES:
            raise RuntimeError(
                f"KDL moving-joint order is {moving_names}, expected {JOINT_NAMES}. "
                "Refusing to map gravity torques to the servos.")

        gravity = PyKDL.Vector(*KDL_GRAVITY_MPS2)
        self.solver = PyKDL.ChainDynParam(self.chain, gravity)
        self._q = PyKDL.JntArray(NJ)
        self._g = PyKDL.JntArray(NJ)

        if log is not None:
            log.info(
                f"KDL gravity ready: {KDL_ROOT_LINK}->{KDL_TIP_LINK}, "
                f"{self.chain.getNrOfSegments()} segments/{NJ} joints, "
                f"xacro={self.xacro_path}")

    def gravity_nm(self, q: Sequence[float]) -> List[float]:
        if len(q) != NJ:
            raise RuntimeError(f"KDL gravity expected {NJ} joints, got {len(q)}")
        for i, value in enumerate(q):
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite q[{i}] passed to KDL gravity: {value}")
            self._q[i] = float(value)
        rc = self.solver.JntToGravity(self._q, self._g)
        if rc < 0:
            raise RuntimeError(f"PyKDL ChainDynParam.JntToGravity failed with code {rc}")
        out = [float(self._g[i]) for i in range(NJ)]
        if not all(math.isfinite(v) for v in out):
            raise RuntimeError(f"KDL gravity returned non-finite torque: {out}")
        return out

    def gravity_current(self, q: Sequence[float]) -> List[float]:
        current = [tau * TORQUE_SCALE for tau in self.gravity_nm(q)]
        if any(abs(v) > MAX_GOAL_CURRENT for v in current):
            raise RuntimeError(
                "KDL gravity alone exceeds MAX_GOAL_CURRENT at this posture: "
                + fmt(current) + ". Refusing to let the feedback clamp hide a "
                "gravity-model/scale/URDF mismatch.")
        return current


def log_kdl_vs_calibration(log, gravity_model: KDLGravityModel,
                           samples: List[dict]) -> Dict[str, object]:
    """Diagnostic only: compare KDL gravity with measured calibration half-sum."""
    rows = []
    max_abs = 0.0
    sq = 0.0
    count = 0
    strong_sign_mismatches = []
    log.info("KDL-vs-calibration gravity diagnostic (calibration is NOT used as run gravity):")
    for i, sample in enumerate(samples):
        kdl = gravity_model.gravity_current(sample["q"])
        measured = list(sample.get("gravity_current", [float("nan")] * NJ))
        delta = [kdl[j] - measured[j] for j in range(NJ)]
        finite_delta = [d for d in delta if math.isfinite(d)]
        if finite_delta:
            max_abs = max(max_abs, max(abs(d) for d in finite_delta))
            sq += sum(d * d for d in finite_delta)
            count += len(finite_delta)
        for j in range(NJ):
            # Ignore near-zero channels. A strong opposite sign is not a tuning
            # disagreement; it indicates a frame/joint-order/model mismatch and
            # is unsafe for gravity feedforward.
            if (math.isfinite(measured[j]) and abs(measured[j]) >= 20.0
                    and abs(kdl[j]) >= 20.0 and measured[j] * kdl[j] < 0.0):
                strong_sign_mismatches.append({
                    "waypoint": i + 1, "joint": j + 1,
                    "kdl": kdl[j], "calib": measured[j]})
        rows.append({"waypoint": i + 1, "kdl_current": kdl,
                     "calib_half_sum_current": measured, "delta": delta})
        log.info(f"  wp{i+1:02d} KDL {fmt(kdl)} | calib {fmt(measured)} | "
                 f"delta {fmt(delta)}")
    rms = math.sqrt(sq / count) if count else float("nan")
    log.info(f"KDL-vs-calibration delta: RMS {rms:.1f} raw units, "
             f"max |delta| {max_abs:.1f} raw units. This is diagnostic, not a fit.")
    if strong_sign_mismatches:
        log.warn("KDL gravity has strong sign mismatches against the unloaded "
                 "hold-current calibration: " + str(strong_sign_mismatches))
    return {"rms_delta_units": rms, "max_abs_delta_units": max_abs,
            "strong_sign_mismatches": strong_sign_mismatches, "rows": rows}


def _self_check():
    ref = [0.220, 0.0, 0.090]
    p = fk([0.0, -0.06321, 0.14415, 0.91478])
    assert all(abs(u - v) < 1e-3 for u, v in zip(p, ref)), f"fk drift: {p}"

    q = ik_pitch(0.220, 0.0, 0.090, 0.0)
    assert q is not None
    assert all(abs(u - v) < 1e-9 for u, v in zip(fk(q), ref))

    qt = [0.10, 0.30, 0.50, -0.70]
    jm = jacobian(qt)
    h = 1e-7
    for col in range(4):
        qp = list(qt); qp[col] += h
        qm = list(qt); qm[col] -= h
        num = [(u - v) / (2.0 * h) for u, v in zip(fk(qp), fk(qm))]
        for row in range(3):
            assert abs(num[row] - jm[row][col]) < 1e-6, "jacobian mismatch"

    for qc in ([0.1, 0.3, 0.5, -0.7], [-0.2, -0.4, 0.2, 0.9]):
        assert abs(link_angles(qc)[2] - pitch_of(qc)) < 1e-12, "pitch form"

    for target in (-0.9, -0.4, 0.0):
        qs = ik_pitch(0.250, 0.0, 0.110, target)
        assert qs is not None and abs(pitch_of(qs) - target) < 1e-9

    # The force estimator must invert the torque map exactly for any
    # force that J^T can actually produce.
    for qc in ([0.0, -0.3, 0.4, -0.1], qt):
        jm = jacobian(qc)
        for f in ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.3, -0.7, 0.5]):
            tau = jt_times(jm, f)
            back = solve_jt_lstsq(jm, tau)
            assert all(abs(u - v) < 1e-6 for u, v in zip(f, back)), \
                "solve_jt_lstsq does not invert jt_times"

    # Quintic endpoints: value, slope and curvature all zero at s=0,1.
    assert abs(quintic(0.0)) < 1e-12 and abs(quintic(1.0) - 1.0) < 1e-12
    h = 1e-4
    for s0 in (0.0, 1.0):
        d1 = (quintic(s0 + h) - quintic(s0 - h)) / (2 * h)
        assert abs(d1) < 1e-6, "quintic endpoint velocity is not zero"

    assert len(calibration_samples()) == len(CARTESIAN_WAYPOINTS_M)


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
                f"Cannot open {PORT_NAME}. Is ros2_control still running? "
                "Stop the bringup launch first."
            )
        if not self.port.setBaudRate(BAUD_RATE):
            raise RuntimeError(f"Cannot set baud rate {BAUD_RATE}.")

        self.reader = GroupSyncRead(
            self.port, self.packet, ADDR_PRESENT_BLOCK, LEN_PRESENT_BLOCK)
        for did in DXL_IDS:
            if not self.reader.addParam(did):
                raise RuntimeError(f"sync read addParam failed for id {did}")

        self.writer_current = GroupSyncWrite(
            self.port, self.packet, ADDR_GOAL_CURRENT, 2)

        self.read_failures = 0
        self.read_attempts = 0
        self.park_on_exit = True
        # Last known gravity feedforward. park() falls back to holding
        # this in current mode if the servo will not accept a mode switch.
        self.hold_current: Optional[List[float]] = None
        self.parked_in_current_mode = False

    # -- single register helpers ------------------------------------

    def ping_all(self):
        for did in DXL_IDS:
            _, result, _ = self.packet.ping(self.port, did)
            if result != COMM_SUCCESS:
                raise RuntimeError(
                    f"id {did} did not answer a ping. Check power and cables.")

    def hardware_errors(self) -> List[int]:
        out = []
        for did in DXL_IDS:
            v, _, _ = self.packet.read1ByteTxRx(self.port, did, ADDR_HW_ERROR)
            out.append(v)
        return out

    def assert_no_hw_error(self, context: str):
        errors = self.hardware_errors()
        if any(errors):
            detail = ", ".join(
                f"id {d}: {decode_hw_error(e)}"
                for d, e in zip(DXL_IDS, errors) if e)
            raise RuntimeError(
                f"Hardware error latched ({context}): {detail}. "
                "Power cycle the arm.")

    def torque(self, on: bool):
        for did in DXL_IDS:
            self.packet.write1ByteTxRx(
                self.port, did, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def get_torque(self) -> List[int]:
        return [self.packet.read1ByteTxRx(self.port, d, ADDR_TORQUE_ENABLE)[0]
                for d in DXL_IDS]

    def set_mode(self, mode: int):
        """Write EEPROM Operating Mode and leave torque OFF.

        AIC6 calls this only from the explicit --set-mode command.
        Neither --calibrate nor --run changes Operating Mode in the air. If
        the requested EEPROM value is already present, do not rewrite it.
        """
        self.torque(False)
        time.sleep(0.02)
        before = self.get_mode()
        if all(m == mode for m in before):
            return
        for did, current_mode in zip(DXL_IDS, before):
            if current_mode == mode:
                continue
            result, error = self.packet.write1ByteTxRx(
                self.port, did, ADDR_OPERATING_MODE, mode)
            if result != COMM_SUCCESS:
                raise RuntimeError(
                    f"id {did}: could not set operating mode "
                    f"(comm result {result})")
            if error != 0:
                raise RuntimeError(
                    f"id {did}: servo rejected the operating mode "
                    f"(dxl error {error}).")
        time.sleep(0.02)
        actual = self.get_mode()
        if any(m != mode for m in actual):
            raise RuntimeError(
                f"Operating mode did not stick. Wanted {mode}, read {actual}.")

    def get_mode(self) -> List[int]:
        return [self.packet.read1ByteTxRx(self.port, d, ADDR_OPERATING_MODE)[0]
                for d in DXL_IDS]

    def current_limits(self) -> List[int]:
        return [self.packet.read2ByteTxRx(self.port, d, ADDR_CURRENT_LIMIT)[0]
                for d in DXL_IDS]

    def read_goal_currents(self) -> List[int]:
        out = []
        for did in DXL_IDS:
            raw, _, _ = self.packet.read2ByteTxRx(
                self.port, did, ADDR_GOAL_CURRENT)
            out.append(to_signed(raw, 16))
        return out

    def set_profile(self, velocity: int, accel: int):
        for did in DXL_IDS:
            self.packet.write4ByteTxRx(
                self.port, did, ADDR_PROFILE_ACCEL, accel)
            self.packet.write4ByteTxRx(
                self.port, did, ADDR_PROFILE_VELOCITY, velocity)

    def write_goal_positions(self, q: Sequence[float]):
        for did, angle in zip(DXL_IDS, q):
            result, error = self.packet.write4ByteTxRx(
                self.port, did, ADDR_GOAL_POSITION, rad_to_pos(angle))
            if result != COMM_SUCCESS or error != 0:
                raise RuntimeError(
                    f"id {did}: goal position write failed "
                    f"(comm {result}, dxl error {error})")

    # -- servo loop I/O ---------------------------------------------

    def read_state(self):
        """(position rad, velocity rad/s, current units) or None."""
        self.read_attempts += 1
        if self.reader.txRxPacket() != COMM_SUCCESS:
            self.read_failures += 1
            return None

        pos, vel, cur = [], [], []
        for did in DXL_IDS:
            if not self.reader.isAvailable(
                    did, ADDR_PRESENT_BLOCK, LEN_PRESENT_BLOCK):
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

    def write_currents(self, currents: Sequence[float]) -> bool:
        self.writer_current.clearParam()
        for did, value in zip(DXL_IDS, currents):
            clamped = int(round(max(-MAX_GOAL_CURRENT,
                                    min(MAX_GOAL_CURRENT, value))))
            raw = clamped & 0xFFFF
            if not self.writer_current.addParam(
                    did, [raw & 0xFF, (raw >> 8) & 0xFF]):
                return False
        return self.writer_current.txPacket() == COMM_SUCCESS

    # -- exit --------------------------------------------------------

    def park(self) -> bool:
        """Exit without an in-air EEPROM mode write.

        POSITION mode: hold measured q with the internal position loop.
        CURRENT mode: leave the last known KDL gravity current commanded and
        torque ON. This is only an interim support state; physically support
        the arm before running --set-mode position or removing power.
        """
        try:
            q = self.read_state_blocking(timeout_s=0.5)[0]
            modes = self.get_mode()
        except Exception:
            return False
        try:
            if all(m == MODE_POSITION for m in modes):
                self.set_profile(PROFILE_VELOCITY, PROFILE_ACCELERATION)
                self.write_goal_positions(q)
                self.torque(True)
                time.sleep(0.3)
                self.parked_in_current_mode = False
                return True
            if all(m == MODE_CURRENT for m in modes):
                if self.hold_current is None:
                    return False
                self.write_currents(self.hold_current)
                self.torque(True)
                self.parked_in_current_mode = True
                return True
            return False
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
        parked = self.park() if self.park_on_exit else False
        if not parked:
            self.release()
        try:
            self.port.closePort()
        except Exception:
            pass
        if parked and self.parked_in_current_mode:
            return ("holding KDL gravity current in CURRENT mode, torque ON; "
                    "no EEPROM mode switch was attempted. Physically support "
                    "the arm before --set-mode position or power removal")
        return ("parked in position mode, torque left ON, arm is holding "
                "itself" if parked else "torque OFF, arm is limp")


# ============================================================
# CAPTURE FOLDER
# ============================================================

LAST_CAPTURE_DIR: Optional[str] = None


class Capture:
    """One folder per run: data.csv, meta.json, plots/.

    The folder and its meta.json are created UP FRONT and the path is
    published in LAST_CAPTURE_DIR, so an aborted run still leaves a
    plottable capture behind. An abort is exactly when the data matters
    most; AIC2 lost it.
    """

    def __init__(self, mode: str, columns: List[str]):
        global LAST_CAPTURE_DIR
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(CAPTURE_ROOT, f"{mode}_{stamp}")
        self.plots_dir = os.path.join(self.dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.csv_path = os.path.join(self.dir, "data.csv")
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.columns = list(columns)
        self._handle = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(self.columns)
        self.rows = 0
        LAST_CAPTURE_DIR = self.dir
        self.meta(base_meta(mode))          # placeholder, rewritten at the end

    def write(self, row: Dict[str, object]):
        self._writer.writerow(
            [row.get(name, "") for name in self.columns])
        self.rows += 1

    def meta(self, payload: Dict[str, object]):
        with open(self.meta_path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def close(self):
        try:
            self._handle.close()
        except Exception:
            pass


def base_meta(mode: str) -> Dict[str, object]:
    return {
        "mode": mode,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "units": {
            "TORQUE_SCALE_units_per_Nm": TORQUE_SCALE,
            "TORQUE_SCALE_source":
                "torque_scale_ = 180.0 in omx_variable_stiffness_controller"
                ".cpp. Borrowed, NOT independently verified. Every newton "
                "in this capture inherits its uncertainty.",
            "CUR_UNIT_MA": CUR_UNIT_MA,
            "VEL_UNIT_RPM": VEL_UNIT_RPM,
            "POS_UNITS_PER_REV": POS_UNITS_PER_REV,
        },
        "gains_SI": {
            "K_CART_NPM": K_CART_NPM,
            "D_CART_NSPM": D_CART_NSPM,
            "KI_CART_N_per_ms": KI_CART,
            "I_CLAMP_N": I_CLAMP_N,
            "KP_JOINT_NM_RAD": KP_JOINT_NM_RAD,
            "KD_JOINT_NMS_RAD": KD_JOINT_NMS_RAD,
            "REG_FRACTION": REG_FRACTION,
            "K_BARRIER_NM_RAD": K_BARRIER_NM_RAD,
            "FRICTION_FF": FRICTION_FF,
            "FRICTION_EPS_rad_s": FRICTION_EPS,
            "FRICTION_GATE_RAD_S": FRICTION_GATE_RAD_S,
            "source": "fixed_stiffness_*/fixed_damping_*/"
                      "trajectory_joint_reg_* in robot1_variable_stiffness"
                      ".yaml, converted to current units by TORQUE_SCALE.",
        },
        "gains_current_units": {
            "KC": KC, "DC": DC, "KI": KI, "I_CLAMP": I_CLAMP,
            "KP_JOINT": KP_JOINT, "KD_JOINT": KD_JOINT,
            "K_BARRIER": K_BARRIER,
        },
        "trajectory": {
            "CARTESIAN_WAYPOINTS_M": CARTESIAN_WAYPOINTS_M,
            "MOVE_TIMES_S": MOVE_TIMES_S,
            "DWELL_TIMES_S": DWELL_TIMES_S,
            "PITCH_RAD": PITCH_RAD,
            "operating_mode_strategy": "persistent current mode; --run never writes Operating Mode EEPROM",
            "scheduled_run_time_s": sum(MOVE_TIMES_S) + sum(DWELL_TIMES_S),
            "profile": "piecewise quintic 10s^3-15s^4+6s^5 with a dwell at every waypoint",
            "segment_start": "measured tip pose at the start of each waypoint move",
            "first_segment_pitch": "ramps measured current-mode entry pitch to PITCH_RAD",
        },
        "safety": {
            "MAX_GOAL_CURRENT": MAX_GOAL_CURRENT,
            "MAX_GOAL_CURRENT_mA": MAX_GOAL_CURRENT * CUR_UNIT_MA,
            "MAX_TIP_ERROR_M": MAX_TIP_ERROR_M,
            "MAX_PITCH_ERROR_RAD": MAX_PITCH_ERROR_RAD,
            "MAX_CTRL_VZ_M_S": MAX_CTRL_VZ_M_S,
            "MAX_CTRL_QD_RAD_S": MAX_CTRL_QD_RAD_S,
            "FAST_ABORT_CONSECUTIVE_CYCLES": FAST_ABORT_CONSECUTIVE_CYCLES,
            "STARTUP_SETTLE_QD_RAD_S": STARTUP_SETTLE_QD_RAD_S,
            "STARTUP_SETTLE_CYCLES": STARTUP_SETTLE_CYCLES,
            "STARTUP_SETTLE_TIMEOUT_S": STARTUP_SETTLE_TIMEOUT_S,
            "CLAMP_ABORT_ALPHA": CLAMP_ABORT_ALPHA,
            "CLAMP_ABORT_CONSECUTIVE_CYCLES": CLAMP_ABORT_CONSECUTIVE_CYCLES,
            "JOINT_LIMITS_RAD": JOINT_LIMITS_RAD,
            "note": "Thermal headroom at MAX_GOAL_CURRENT over the "
                    f"{sum(MOVE_TIMES_S) + sum(DWELL_TIMES_S):.0f} s "
                    "configured waypoint schedule is NOT verified.",
        },
        "geometry": {
            "BASE_X": BASE_X, "BASE_Z": BASE_Z,
            "L1": L1, "DELTA": DELTA, "L2": L2, "L3": L3,
            "source": "joint origins in open_manipulator_x.urdf.xacro; "
                      "sign convention verified in _self_check() against a "
                      "MoveIt solution recorded from this arm.",
        },
        "gravity_compensation": {
            "source": "PyKDL ChainDynParam from OpenMANIPULATOR-X URDF/xacro",
            "evaluation": "recomputed from measured q every --run control cycle",
            "root_link": KDL_ROOT_LINK,
            "tip_link": KDL_TIP_LINK,
            "gravity_mps2": KDL_GRAVITY_MPS2,
            "conversion": "gravity_Nm * TORQUE_SCALE -> raw Dynamixel current units",
            "calibration_half_sum_usage": "diagnostic only; not used as --run gravity feedforward",
        },
        "velocity_control": {
            "source": "causal derivative of measured Present Position",
            "filter": "backward difference followed by exact-discrete first-order low-pass",
            "candidate_cutoffs_Hz": VELOCITY_FILTER_CANDIDATES_HZ,
            "selected_cutoff_Hz": CONTROL_VEL_CUTOFF_HZ,
            "servo_present_velocity_usage": "diagnostic/logging only; not used by damping, joint derivative regularisation, or friction direction",
            "failed_AIC4_evidence": {
                "vertical_oscillation_Hz": 9.34,
                "present_velocity_best_lag_ms": 51.7,
                "phase_at_oscillation_deg": 173.8,
                "mean_damping_power_against_position_derived_velocity_W": 2.97019
            }
        },
        "force_estimate": {
            "formula": "F_applied = pinv(J^T) . (I_meas - g_KDL(q) "
                       "- FRICTION_FF*f_c(q)*tanh(qdot/eps)) / TORQUE_SCALE",
            "sensor": "none. Estimated from Present Current only.",
            "caveat": "Residual joint friction dominates. Order +/-2 N in x "
                      "for this arm, against +/-0.05 N of current "
                      "quantisation. Absolute values are indicative only; "
                      "for anything quantitative, subtract a matched "
                      "unloaded run at the same q.",
        },
    }


# ============================================================
# COLUMNS
#
# Naming: what it is, then how it was obtained.
#   MEASURED    read straight off the servo
#   CALCULATED  derived here from measured values
#   COMMANDED   produced by this script and sent to the servo
# ============================================================

def joint_cols(prefix: str) -> List[str]:
    return [f"{prefix}{k + 1}" for k in range(NJ)]


def _filter_tag(fc: float) -> str:
    return f"f{int(round(fc))}"


COLUMNS_RUN = (
    ["t", "phase", "waypoint", "motion", "s", "period", "ramp", "clamp_alpha", "read_fail"]
    # MEASURED
    + joint_cols("q") + joint_cols("qd") + joint_cols("cur")
    + joint_cols("cur_mA") + joint_cols("cur_Nm")
    # CALCULATED state. Legacy qd/v columns remain servo-velocity-derived
    # diagnostics; qd_ctrl/v*_ctrl drive the controller.
    + ["x", "y", "z", "pitch", "vx", "vy", "vz"]
    + joint_cols("qd_ctrl") + ["vx_ctrl", "vy_ctrl", "vz_ctrl"]
    + sum((joint_cols(_filter_tag(fc) + "_qd")
           + [_filter_tag(fc) + "_vx", _filter_tag(fc) + "_vy",
              _filter_tag(fc) + "_vz"]
           for fc in VELOCITY_FILTER_CANDIDATES_HZ), [])
    # COMMANDED setpoint
    + ["xd", "yd", "zd", "pitch_d"] + joint_cols("qref")
    # CALCULATED error
    + ["ex", "ey", "ez", "err_mm", "pitch_err"]
    # COMMANDED impedance force
    + ["fx_cmd_N", "fy_cmd_N", "fz_cmd_N", "ix_N", "iy_N", "iz_N"]
    # COMMANDED torque breakdown, current units
    + joint_cols("grav") + joint_cols("fric") + joint_cols("imp")
    + joint_cols("reg") + joint_cols("bar") + joint_cols("cmd")
    + joint_cols("cmd_Nm")
    # CALCULATED external force estimate from MEASURED current
    + joint_cols("tau_ext")
    + ["fx_est_N", "fy_est_N", "fz_est_N", "f_est_N"]
    + ["fx_raw_N", "fy_raw_N", "fz_raw_N", "f_raw_N"]
)

COLUMNS_CALIB = (
    ["t", "phase", "pass", "sample", "s", "period"]
    + joint_cols("q") + joint_cols("qd") + joint_cols("cur")
    + joint_cols("cur_mA") + joint_cols("cur_Nm")
    + ["x", "y", "z", "pitch"]
    + ["xd", "yd", "zd", "pitch_d"]
    + ["ex", "ey", "ez", "err_mm", "pitch_err"]
)


def fmt_row(values: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in values.items():
        if isinstance(value, float):
            out[key] = f"{value:.6g}"
        else:
            out[key] = value
    return out


def pack_joints(row: Dict[str, object], prefix: str,
                values: Sequence[float]):
    for k in range(NJ):
        row[f"{prefix}{k + 1}"] = values[k]


# ============================================================
# CALIBRATION TABLE
# ============================================================

def table_at(samples: List[dict], q: Sequence[float],
             key: str) -> List[float]:
    """Interpolate a per-joint quantity along the calibrated joint path.

    The samples are the vertices of a piecewise-linear path in joint
    space. Project q onto that polyline, find the bracketing pair, blend.
    Continuous everywhere, exact at the recorded poses.

    AIC2's first version was nearest-neighbour, which stepped
    discontinuously at each midpoint -- a step in feedforward is an
    impulse in torque.

    This is a measurement table, not a model. It has no link masses in
    it and does not pretend to.
    """
    if len(samples) == 1:
        return list(samples[0][key])

    best_d = float("inf")
    best_k, best_u = 0, 0.0

    for k in range(len(samples) - 1):
        a = samples[k]["q"]
        b = samples[k + 1]["q"]
        ab = [y - x for x, y in zip(a, b)]
        den = sum(v * v for v in ab)
        if den < 1e-12:
            u = 0.0
        else:
            u = sum((qi - ai) * abi for qi, ai, abi in zip(q, a, ab)) / den
            u = max(0.0, min(1.0, u))
        d = sum((qi - (ai + u * abi)) ** 2
                for qi, ai, abi in zip(q, a, ab))
        if d < best_d:
            best_d, best_k, best_u = d, k, u

    ga = samples[best_k][key]
    gb = samples[best_k + 1][key]
    return [(1.0 - best_u) * ga[j] + best_u * gb[j] for j in range(NJ)]


class CausalVelocityBank:
    """Parallel causal position-derived velocity filters.

    For each candidate cutoff fc, first form the backward difference
        qdot_raw[k] = (q[k] - q[k-1]) / dt
    then apply an exact-discrete first-order low-pass
        qdot_f[k] = alpha*qdot_f[k-1] + (1-alpha)*qdot_raw[k]
        alpha = exp(-2*pi*fc*dt).

    Every candidate is computed on every control cycle so the run capture
    contains an apples-to-apples record. Only CONTROL_VEL_CUTOFF_HZ drives
    feedback. Dynamixel Present Velocity is not used by this class.
    """

    def __init__(self, cutoffs_hz: Sequence[float]):
        self.cutoffs = [float(v) for v in cutoffs_hz]
        self.prev_q: Optional[List[float]] = None
        self.state = {fc: [0.0] * NJ for fc in self.cutoffs}

    def reset(self, q: Optional[Sequence[float]] = None):
        # Do NOT latch a q measured before the loop starts. It is separated
        # from the first cycle by unknown setup time, and dividing that real
        # position change by the NOMINAL dt manufactures a velocity spike.
        self.prev_q = list(q) if q is not None else None
        for fc in self.cutoffs:
            self.state[fc] = [0.0] * NJ

    def update(self, q: Sequence[float], dt_s: float) -> Dict[float, List[float]]:
        dt_s = max(1e-4, min(float(dt_s), 5.0 / CONTROL_HZ))
        if self.prev_q is None:
            self.prev_q = list(q)
            return {fc: list(v) for fc, v in self.state.items()}
        raw = [(q[k] - self.prev_q[k]) / dt_s for k in range(NJ)]
        self.prev_q = list(q)
        out: Dict[float, List[float]] = {}
        for fc in self.cutoffs:
            alpha = math.exp(-2.0 * math.pi * fc * dt_s)
            old = self.state[fc]
            new = [alpha * old[k] + (1.0 - alpha) * raw[k]
                   for k in range(NJ)]
            self.state[fc] = new
            out[fc] = list(new)
        return out


def friction_torque(fc: Sequence[float],
                    qd: Sequence[float]) -> List[float]:
    """Coulomb friction feedforward.

    Friction opposes motion, so the motor must supply +f_c in the
    direction of travel. AIC5 feeds this function the position-derived
    filtered control velocity. tanh instead of sign avoids chatter around
    standstill and makes the feedforward continuous. FRICTION_FF < 1 because over-compensating
    Coulomb friction produces a limit cycle.
    """
    return [FRICTION_FF * abs(fc[k]) * math.tanh(qd[k] / FRICTION_EPS)
            for k in range(NJ)]


def estimate_tip_force(j: List[List[float]], cur: Sequence[float],
                       grav: Sequence[float],
                       fric: Sequence[float]) -> Tuple[List[float], List[float]]:
    """Tip force applied BY the arm TO the environment, in newtons.

    Static balance at the joints, all in current units:

        I_measured = g(q) + f_c.sign(qdot) + tau_applied

    where tau_applied = J^T F_applied is the extra motor torque doing
    work on the environment. Rearranged:

        tau_applied = I_measured - g(q) - f_c.sign(qdot)
        F_applied   = pinv(J^T) tau_applied / TORQUE_SCALE

    Returns (compensated, raw). The raw version omits the friction term
    so the size of that correction is visible in the log. Neither is a
    force-sensor reading; see meta.json force_estimate.caveat.
    """
    tau_c = [cur[k] - grav[k] - fric[k] for k in range(NJ)]
    tau_r = [cur[k] - grav[k] for k in range(NJ)]
    fc = [v / TORQUE_SCALE for v in solve_jt_lstsq(j, tau_c)]
    fr = [v / TORQUE_SCALE for v in solve_jt_lstsq(j, tau_r)]
    return fc, fr


def clamp_preserving(grav: Sequence[float],
                     extra: Sequence[float],
                     limit: float = MAX_GOAL_CURRENT
                     ) -> Tuple[List[float], float]:
    """Clamp to +/-limit by scaling ONLY the non-gravity part.

    AIC2 clamped the total, so a large impedance term starved the gravity
    feedforward and the arm sagged. One global alpha keeps the joints
    coordinated instead of distorting the commanded direction.
    SOURCE: the gravity-preserving clamp in the C++ update loop.
    """
    alpha = 1.0
    for g, p in zip(grav, extra):
        g = max(-limit, min(limit, g))
        if p > 1e-9 and g + p > limit:
            alpha = min(alpha, (limit - g) / p)
        elif p < -1e-9 and g + p < -limit:
            alpha = min(alpha, (-limit - g) / p)
    alpha = max(0.0, alpha)
    out = [max(-limit, min(limit, g)) + alpha * p
           for g, p in zip(grav, extra)]
    return out, alpha


def check_path(x_from: Sequence[float], x_to: Sequence[float],
               p_from: float, p_to: float,
               n: int = 101) -> List[float]:
    """Verify the whole planned path is reachable and inside the joint
    limits BEFORE commanding any of it. Returns the smallest headroom to
    a limit on each joint, in radians.

    AIC2 checked only its waypoints. It never checked the path between
    them, so it drove joint4 to -1.6506 rad against a -1.70 limit while
    every endpoint was legal. This is cheap: closed-form IK, 101 samples,
    once per phase.
    """
    worst = [float("inf")] * NJ
    for i in range(n):
        w = quintic(i / (n - 1.0))
        x = [a + (b - a) * w for a, b in zip(x_from, x_to)]
        pitch = p_from + (p_to - p_from) * w
        q = ik_pitch(x[0], x[1], x[2], pitch)
        if q is None:
            raise RuntimeError(
                f"Planned path leaves the workspace at s={i / (n - 1.0):.2f}: "
                f"[{x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f}] at pitch "
                f"{pitch:+.4f} rad is unreachable. Not moving.")
        for k, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            room = min(q[k] - lo, hi - q[k])
            worst[k] = min(worst[k], room)
            if room < BARRIER_MARGIN_RAD:
                raise RuntimeError(
                    f"Planned path drives joint{k + 1} to {q[k]:+.4f} rad at "
                    f"s={i / (n - 1.0):.2f}, only {room:.3f} rad from its "
                    f"limit ({lo}, {hi}) and inside the "
                    f"{BARRIER_MARGIN_RAD} rad barrier. Not moving. "
                    "Lower the pitch demand, shorten that waypoint segment, or move "
                    "the workspace in.")
    return worst


def barrier_torque(q: Sequence[float]) -> List[float]:
    """Restoring spring inside the joint-limit margin.

    AIC2 aborted here. Aborting is the right last resort but the wrong
    first response: a soft push back costs nothing and usually avoids the
    abort entirely. SOURCE: the joint-limit barrier in the C++.
    """
    out = [0.0] * NJ
    for k, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
        if q[k] < lo + BARRIER_MARGIN_RAD:
            out[k] = K_BARRIER * (lo + BARRIER_MARGIN_RAD - q[k])
        elif q[k] > hi - BARRIER_MARGIN_RAD:
            out[k] = K_BARRIER * (hi - BARRIER_MARGIN_RAD - q[k])
    return out


# ============================================================
# NODE
# ============================================================

class ImpedanceNode(Node):

    def __init__(self):
        super().__init__("cartesian_impedance_v6_persistent")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

    def publish(self, q, qd, cur):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = list(q)
        msg.velocity = list(qd)
        # N.m via TORQUE_SCALE, which is borrowed. See meta.json.
        msg.effort = [c / TORQUE_SCALE for c in cur]
        self.pub.publish(msg)


# ============================================================
# PHASE 1 : CALIBRATE
# ============================================================

def _sweep(node, bus, plan, order, cap, t0, tag, log):
    """One pass over the samples in position control, recording the hold
    current at each. Returns {index: (q_held, mean_current)}."""
    out = {}
    prev = time.monotonic()

    for step, i in enumerate(order):
        q_goal = plan[i]
        xyz = CALIB_XYZ[i]
        log.info(f"  [{tag}] {step + 1}/{len(order)}  sample {i + 1} -> "
                 f"[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]")

        bus.write_goal_positions(q_goal)
        time.sleep(CALIB_MOVE_TIME_S)

        acc = [0.0] * NJ
        got = 0
        held = list(q_goal)
        deadline = time.monotonic() + CALIB_HOLD_TIME_S

        while time.monotonic() < deadline:
            state = bus.read_state()
            now = time.monotonic()
            if state is not None:
                q, qd, cur = state
                acc = [a + c for a, c in zip(acc, cur)]
                got += 1
                held = q
                node.publish(q, qd, cur)

                tip = fk(q)
                err = [a - b for a, b in zip(xyz, tip)]
                row = {
                    "t": now - t0, "phase": "calib", "pass": tag,
                    "sample": i + 1, "s": 1.0, "period": now - prev,
                    "x": tip[0], "y": tip[1], "z": tip[2],
                    "pitch": pitch_of(q),
                    "xd": xyz[0], "yd": xyz[1], "zd": xyz[2],
                    "pitch_d": PITCH_RAD,
                    "ex": err[0], "ey": err[1], "ez": err[2],
                    "err_mm": math.sqrt(sum(e * e for e in err)) * 1000.0,
                    "pitch_err": PITCH_RAD - pitch_of(q),
                }
                pack_joints(row, "q", q)
                pack_joints(row, "qd", qd)
                pack_joints(row, "cur", cur)
                pack_joints(row, "cur_mA", [c * CUR_UNIT_MA for c in cur])
                pack_joints(row, "cur_Nm", [c / TORQUE_SCALE for c in cur])
                cap.write(fmt_row(row))
            prev = now
            time.sleep(0.01)

        if got == 0:
            raise RuntimeError(f"No readings at sample {i + 1}.")

        mean = [a / got for a in acc]
        out[i] = (held, mean)
        log.info("     hold current: "
                 + ", ".join(f"{v:+7.1f}" for v in mean)
                 + f"   ({got} samples)")

    return out


def calibrate(node, bus: Bus, plan: List[List[float]]) -> str:
    log = node.get_logger()
    bus.assert_no_hw_error("before calibration")

    log.info("Calibration: position control, measuring bidirectional hold current.")
    log.info("AIC4 --run uses KDL/URDF gravity, not the calibration half-sum. "
             "This calibration is retained for Coulomb-friction feedforward; "
             "the half-sum is stored only as a KDL diagnostic.")
    log.info(f"Current limits: {bus.current_limits()}")
    log.info("Two passes, forwards then backwards. Averaging cancels the "
             "Coulomb friction: it helps hold the arm on the way out and "
             "hinders on the way back, so it enters the two passes with "
             "opposite sign. The half-difference is retained as the friction "
             "feedforward table. The half-sum is diagnostic only in AIC4.")

    cap = Capture("calib", COLUMNS_CALIB)
    modes = bus.get_mode()
    if any(m != MODE_POSITION for m in modes):
        cap.close()
        raise RuntimeError(
            f"--calibrate requires all servos already in Position Control "
            f"Mode ({MODE_POSITION}); read {modes}. Physically support the "
            "arm and run: python3 AIC6.py --set-mode position")
    bus.set_profile(PROFILE_VELOCITY, PROFILE_ACCELERATION)
    bus.torque(True)
    time.sleep(0.2)

    t0 = time.monotonic()
    try:
        n = len(plan)
        fwd = _sweep(node, bus, plan, list(range(n)), cap, t0, "fwd", log)

        # Reverse-direction calibration must actually ARRIVE at the final
        # waypoint from the far side. Starting the reverse sweep while already
        # sitting at waypoint n makes its first hold sample a duplicate of the
        # forward measurement and falsely drives the wp-n friction estimate to 0.
        if n < 2:
            raise RuntimeError("Calibration reversal needs at least two waypoints.")

        step_xyz = [
            CALIB_XYZ[-1][k] - CALIB_XYZ[-2][k]
            for k in range(3)
        ]
        over = [
            CALIB_XYZ[-1][k]
            + CALIB_REVERSE_OVERSHOOT_SPACINGS * step_xyz[k]
            for k in range(3)
        ]

        q_over = ik_pitch(over[0], over[1], over[2], PITCH_RAD)
        if q_over is None:
            raise RuntimeError(
                f"Calibration overshoot pose {over} is unreachable. Move the "
                "last waypoint in or reduce CALIB_REVERSE_OVERSHOOT_SPACINGS."
            )

        for k, (angle, (lo, hi)) in enumerate(
                zip(q_over, JOINT_LIMITS_RAD)):
            if not (
                lo + BARRIER_MARGIN_RAD
                < angle
                < hi - BARRIER_MARGIN_RAD
            ):
                raise RuntimeError(
                    f"Calibration overshoot puts joint{k + 1} at "
                    f"{angle:+.4f} rad, too close to ({lo}, {hi})."
                )

        over_headroom = check_path(
            CALIB_XYZ[-1], over, PITCH_RAD, PITCH_RAD
        )

        log.info(
            f"Overshooting to {[round(v, 3) for v in over]} before reversing; "
            + ", ".join(
                f"j{k + 1} headroom {over_headroom[k]:.3f} rad"
                for k in range(NJ)
            )
        )

        bus.write_goal_positions(q_over)
        time.sleep(CALIB_MOVE_TIME_S)

        log.info("Reversing after overshoot.")
        rev = _sweep(
            node, bus, plan, list(range(n - 1, -1, -1)),
            cap, t0, "rev", log
        )
    finally:
        cap.close()

    samples = []
    log.info("")
    log.info("Per-sample split. 'hold_half_sum'=(fwd+rev)/2 is diagnostic; "
             "'friction'=|fwd-rev|/2 is used by AIC4 --run:")

    for i in range(len(plan)):
        q_f, c_f = fwd[i]
        q_r, c_r = rev[i]
        mean = [(a + b) / 2.0 for a, b in zip(c_f, c_r)]
        half = [abs(a - b) / 2.0 for a, b in zip(c_f, c_r)]
        q_avg = [(a + b) / 2.0 for a, b in zip(q_f, q_r)]

        samples.append({
            "q": q_avg,
            "xyz": CALIB_XYZ[i],
            "gravity_current": mean,
            "friction_current": half,
            "fwd_current": c_f,
            "rev_current": c_r,
        })
        log.info(f"  {i + 1:2d}  hold_half_sum "
                 + ", ".join(f"{v:+7.1f}" for v in mean)
                 + "   friction "
                 + ", ".join(f"{v:5.1f}" for v in half))

    worst = max(v for s in samples for v in s["friction_current"])
    log.info("")
    log.info(f"Largest Coulomb term: {worst:.1f} units "
             f"({worst * CUR_UNIT_MA:.0f} mA, {worst / TORQUE_SCALE:.3f} N.m "
             f"using the borrowed TORQUE_SCALE).")

    with open(CALIB_FILE, "w") as fh:
        json.dump({
            "version": 4,
            "pitch_rad": PITCH_RAD,
            "waypoints": CARTESIAN_WAYPOINTS_M,
            "n_points": len(CALIB_XYZ),
            "torque_scale": TORQUE_SCALE,
            "samples": samples,
        }, fh, indent=2)

    meta = base_meta("calibrate")
    meta["calibration"] = {
        "file": CALIB_FILE,
        "points": len(CALIB_XYZ),
        "method": "bidirectional position-control sweep; hold-current half-sum = "
                  "(fwd+rev)/2 retained only for KDL diagnostics; "
                  "Coulomb friction = |fwd-rev|/2 used by --run",
        "largest_friction_units": worst,
    }
    meta["read_failures"] = f"{bus.read_failures}/{bus.read_attempts}"
    cap.meta(meta)

    log.info(f"Saved {CALIB_FILE}")
    log.info(f"Capture: {cap.dir}")
    log.info(f"Read failures: {bus.read_failures}/{bus.read_attempts}")
    return cap.dir


def load_calibration() -> List[dict]:
    if not os.path.exists(CALIB_FILE):
        raise RuntimeError(
            f"No calibration at {CALIB_FILE}. Run --calibrate first.")

    with open(CALIB_FILE) as handle:
        data = json.load(handle)

    if data.get("version") not in (3, 4):
        raise RuntimeError(
            "Calibration file is from an unsupported script version. "
            "Re-run --calibrate.")
    if abs(data.get("pitch_rad", 1e9) - PITCH_RAD) > 1e-6:
        raise RuntimeError(
            "Calibration was recorded at a different PITCH_RAD.")

    samples = data.get("samples", [])
    recorded_xyz = [sample.get("xyz") for sample in samples]
    if len(recorded_xyz) != len(CALIB_XYZ) or any(
            xyz is None or len(xyz) != 3 or math.dist(xyz, target) > 1e-9
            for xyz, target in zip(recorded_xyz, CALIB_XYZ)):
        raise RuntimeError(
            "Calibration was recorded for different waypoint locations. "
            "Re-run --calibrate after changing CARTESIAN_WAYPOINTS_M.")
    return samples


# ============================================================
# PHASE 2 : IMPEDANCE RUN
# ============================================================

def fmt(values: Sequence[float]) -> str:
    return ", ".join(f"{v:+7.1f}" for v in values)


def run_impedance(node, bus: Bus) -> str:
    log = node.get_logger()
    validate_waypoint_settings()
    samples = load_calibration()
    gravity_model = KDLGravityModel(log)
    gravity_diagnostic = log_kdl_vs_calibration(log, gravity_model, samples)
    if gravity_diagnostic["strong_sign_mismatches"]:
        raise RuntimeError(
            "KDL gravity failed the sign sanity check against the unloaded "
            "calibration. Check KDL root/tip, joint order and URDF before "
            "enabling current control. See the mismatch list above.")
    bus.assert_no_hw_error("before run")

    limits = bus.current_limits()
    log.info(f"Current limits: {limits}")
    if any(l < MAX_GOAL_CURRENT for l in limits):
        log.warn(f"Current Limit is below MAX_GOAL_CURRENT "
                 f"({MAX_GOAL_CURRENT}). The servo truncates before this "
                 "script's clamp ever sees it.")

    log.info(f"K = {K_CART_NPM} N/m   D = {D_CART_NSPM} N.s/m   "
             f"KI = {KI_CART} N/(m.s)")
    log.info(f"-> current units: KC = {[round(v) for v in KC]}  "
             f"DC = {[round(v) for v in DC]}")
    log.info(f"Joint reg K = {KP_JOINT_NM_RAD} N.m/rad, "
             f"D = {KD_JOINT_NMS_RAD} N.m.s/rad, capped at "
             f"{REG_FRACTION:.0%} of the clamp")
    log.info(f"Friction feedforward {FRICTION_FF:.0%} of measured Coulomb, "
             f"tanh width {FRICTION_EPS:.3f} rad/s; disabled above "
             f"{FRICTION_GATE_RAD_S:.2f} rad/s during fast transients")
    log.info(f"Clamp +/-{MAX_GOAL_CURRENT} units "
             f"({MAX_GOAL_CURRENT * CUR_UNIT_MA:.0f} mA, "
             f"{MAX_GOAL_CURRENT / TORQUE_SCALE:.2f} N.m)")
    log.info(f"Waypoint schedule: {len(CARTESIAN_WAYPOINTS_M)} stops, "
             f"{sum(MOVE_TIMES_S):.1f} s moving + "
             f"{sum(DWELL_TIMES_S):.1f} s dwelling")
    log.info("Controller velocity: position-derived causal LP derivative; "
             f"selected {CONTROL_VEL_CUTOFF_HZ:.0f} Hz, candidates "
             f"{VELOCITY_FILTER_CANDIDATES_HZ}; Dynamixel Present Velocity "
             "is diagnostic only")
    log.info("Operating Mode strategy: persistent CURRENT mode; --run never writes EEPROM")
    log.info(f"Fast aborts: |vz_ctrl|>{MAX_CTRL_VZ_M_S:.2f} m/s or "
             f"|qd_ctrl|>{MAX_CTRL_QD_RAD_S:.2f} rad/s for "
             f"{FAST_ABORT_CONSECUTIVE_CYCLES} consecutive cycles")

    # Solve all waypoint endpoints and preflight every nominal inter-waypoint
    # Cartesian segment before the schedule starts. The first segment is
    # checked separately because its start pose/pitch is the measured
    # persistent-current-mode entry pose.
    plan = build_calib_plan()
    for i in range(1, len(CARTESIAN_WAYPOINTS_M)):
        worst = check_path(CARTESIAN_WAYPOINTS_M[i - 1],
                           CARTESIAN_WAYPOINTS_M[i],
                           PITCH_RAD, PITCH_RAD)
        tight = [f"joint{k + 1} ({worst[k]:.3f} rad)" for k in range(NJ)
                 if worst[k] < PATH_HEADROOM_WARN_RAD]
        if tight:
            log.warn(f"Nominal path to waypoint {i + 1} has little room: "
                     + ", ".join(tight))

    q_start = plan[0]

    # -- persistent-current-mode entry ------------------------------
    # Operating Mode is EEPROM and survives power cycles. --run NEVER writes
    # it. Configure CURRENT mode beforehand while the arm is physically
    # supported; this removes the all-joints-unpowered EEPROM window entirely.
    modes = bus.get_mode()
    if any(m != MODE_CURRENT for m in modes):
        raise RuntimeError(
            f"--run requires all servos already in Current Control Mode "
            f"({MODE_CURRENT}); read {modes}. Physically support the arm and "
            "run: python3 AIC6.py --set-mode current")

    torque_states = bus.get_torque()

    torque_all_off = all(t == 0 for t in torque_states)
    torque_all_on = all(t != 0 for t in torque_states)

    if not (torque_all_off or torque_all_on):
        raise RuntimeError(
            f"Mixed Torque Enable state at --run entry: {torque_states}. "
            "Refusing a partial take-over. Physically support the arm and "
            "normalise the servo state before retrying."
        )

    if torque_all_on:
        log.warn(
            f"Torque already ON at --run entry ({torque_states}). Taking over "
            "in place in CURRENT mode; no torque-off or Operating Mode write "
            "will be performed."
        )

    q_seed = bus.read_state_blocking(timeout_s=0.5)[0]
    x_seed = fk(q_seed)
    hold = gravity_model.gravity_current(q_seed)
    bus.hold_current = list(hold)
    log.info("Persistent-current-mode start. Supported pose: "
             f"[{x_seed[0]:.4f}, {x_seed[1]:.4f}, {x_seed[2]:.4f}], "
             f"pitch {pitch_of(q_seed):+.4f}; KDL gravity seed " + fmt(hold))

    # # Seed KDL hold current before enabling torque on a fresh start.
    # # If a previous --run parked with torque already ON, this same write is
    # # a smooth in-place take-over: never drop torque just to satisfy an entry
    # # condition.
    # if not bus.write_currents(hold):
    #     raise RuntimeError("Could not seed Goal Current at --run entry.")

    # seeded = bus.read_goal_currents()
    # if any(abs(a - b) > 3 for a, b in zip(seeded, hold)):
    #     raise RuntimeError(
    #         f"Goal Current seed did not stick. Wanted "
    #         f"{[round(v) for v in hold]}, read {seeded}. Refusing to continue."
    #     )

    # if torque_all_off:
    #     bus.torque(True)
    #     log.info(
    #         "Torque enabled in already-configured CURRENT mode; --run did "
    #         "not touch Operating Mode."
    #     )
    # else:
    #     log.info(
    #         "Continuing with torque already ON in CURRENT mode; --run did "
    #         "not release the arm or touch Operating Mode."
    #     )
        # Goal Current is RAM, but the servo holds it at 0 while Torque Enable
    # is 0: AIC6's first two --run attempts wrote the seed with torque off
    # and read back [0,0,0,0]. Enable torque FIRST. In CURRENT mode a Goal
    # Current of 0 is zero torque -- the same limp state the supported arm
    # is already in -- and the window is one sync write, ~1 ms at 1 Mbaud.
    if torque_all_off:
        bus.torque(True)
        log.info("Torque enabled in already-configured CURRENT mode at Goal "
                 "Current 0; seeding KDL gravity now.")
    else:
        log.info("Torque already ON; writing the KDL gravity seed in place.")

    seeded = [0] * NJ
    for _ in range(5):
        if not bus.write_currents(hold):
            raise RuntimeError("Could not write Goal Current at --run entry.")
        seeded = bus.read_goal_currents()
        if all(abs(a - b) <= 3 for a, b in zip(seeded, hold)):
            break
        time.sleep(0.002)
    if any(abs(a - b) > 3 for a, b in zip(seeded, hold)):
        raise RuntimeError(
            f"Goal Current will not stick. Wanted {[round(v) for v in hold]}, "
            f"read {seeded}. Torque {bus.get_torque()}, mode {bus.get_mode()}.")

    # Wait for stillness before starting the waypoint clock. Damping is FULL
    # strength here; stiffness, integral and Coulomb-friction FF are absent.
    dt = 1.0 / CONTROL_HZ
    settle_bank = CausalVelocityBank(VELOCITY_FILTER_CANDIDATES_HZ)
    settle_bank.reset()
    quiet = 0
    deadline = time.monotonic() + STARTUP_SETTLE_TIMEOUT_S
    last_state = (q_seed, [0.0] * NJ, [0.0] * NJ)
    last_command = list(hold)
    settle_peak_qd = 0.0
    settle_peak_vz = 0.0
    settle_fast_count = 0
    settle_prev = time.monotonic()
    while time.monotonic() < deadline:
        loop_t = time.monotonic()
        state = bus.read_state_blocking(timeout_s=0.2)
        q, qd_servo, cur = state
        settle_step = min(max(loop_t - settle_prev, 1e-4), 5.0 * dt)
        settle_prev = loop_t
        qd_candidates = settle_bank.update(q, settle_step)
        # qd_candidates = settle_bank.update(q, dt)
        qd_ctrl = qd_candidates[CONTROL_VEL_CUTOFF_HZ]
        jm = jacobian(q)
        v_ctrl = j_times(jm, qd_ctrl)
        too_fast = (
            abs(v_ctrl[2]) > MAX_CTRL_VZ_M_S
            or max(abs(u) for u in qd_ctrl) > MAX_CTRL_QD_RAD_S
        )

        settle_fast_count = settle_fast_count + 1 if too_fast else 0

        if settle_fast_count >= FAST_ABORT_CONSECUTIVE_CYCLES:
            raise RuntimeError(
                f"Startup fast-motion abort: "
                f"vz_ctrl={v_ctrl[2]:+.3f} m/s, "
                f"max|qd_ctrl|="
                f"{max(abs(u) for u in qd_ctrl):.3f} rad/s.")
        settle_peak_qd = max(settle_peak_qd, max(abs(u) for u in qd_ctrl))
        settle_peak_vz = max(settle_peak_vz, abs(v_ctrl[2]))

        grav = gravity_model.gravity_current(q)
        bus.hold_current = list(grav)
        damp_force = [-DC[k] * v_ctrl[k] for k in range(3)]
        imp_d = jt_times(jm, damp_force)
        reg_d = [-KD_JOINT * qd_ctrl[k] for k in range(NJ)]
        bar = barrier_torque(q)
        extra = [imp_d[k] + reg_d[k] + bar[k] for k in range(NJ)]
        command, _ = clamp_preserving(grav, extra)
        if max(abs(u) for u in qd_ctrl) > 0.2:
            log.warn(f"settle dt={settle_step*1000:.1f}ms "
                     f"q={[round(u, 4) for u in q]} "
                     f"qd_ctrl={[round(u, 3) for u in qd_ctrl]} "
                     f"cmd={[round(u) for u in command]}")
        if not bus.write_currents(command):
            raise RuntimeError("Goal current write failed during startup settle.")
        last_command = command
        last_state = state

        if (max(abs(u) for u in qd_ctrl) < STARTUP_SETTLE_QD_RAD_S) and (abs(v_ctrl[2]) < 0.03):
            quiet += 1
            if quiet >= STARTUP_SETTLE_CYCLES:
                break
        else:
            quiet = 0

        sleep_for = dt - (time.monotonic() - loop_t)
        if sleep_for > 0:
            time.sleep(sleep_for)
    else:
        raise RuntimeError(
            f"Arm did not settle in CURRENT mode within "
            f"{STARTUP_SETTLE_TIMEOUT_S:.1f} s. Peak |vz_ctrl|="
            f"{settle_peak_vz:.3f} m/s, peak |qd_ctrl|="
            f"{settle_peak_qd:.3f} rad/s.")

    log.info(f"Startup settled before schedule: {quiet} quiet cycles; "
             f"peak |vz_ctrl|={settle_peak_vz:.3f} m/s, "
             f"peak |qd_ctrl|={settle_peak_qd:.3f} rad/s")

    cap = Capture("run", COLUMNS_RUN)
    dt = 1.0 / CONTROL_HZ

    last_state = bus.read_state_blocking()
    x_entry = fk(last_state[0])
    pitch_entry = pitch_of(last_state[0])
    log.info(f"Current-mode schedule entry pose: "
             f"[{x_entry[0]:.4f}, {x_entry[1]:.4f}, {x_entry[2]:.4f}]  "
             f"pitch {pitch_entry:+.4f}  "
             f"({math.dist(x_entry, CARTESIAN_WAYPOINTS_M[0]) * 1000:.0f} mm "
             "from waypoint 1)")

    worst = check_path(x_entry, CARTESIAN_WAYPOINTS_M[0],
                       pitch_entry, PITCH_RAD)
    log.info("Current-mode approach-to-waypoint-1 joint headroom: "
             + ", ".join(f"j{k + 1} {worst[k]:.3f} rad" for k in range(NJ)))
    tight = [f"joint{k + 1} ({worst[k]:.3f} rad)" for k in range(NJ)
             if worst[k] < PATH_HEADROOM_WARN_RAD]
    if tight:
        log.warn("Current-mode approach has little room to spare on "
                 + ", ".join(tight))

    start = time.monotonic()
    prev_cycle = start
    integral = [0.0, 0.0, 0.0]
    last_hw_check = start

    stats = {"max_err": 0.0, "max_pitch_err": 0.0,
             "max_cmd": 0.0, "min_alpha": 1.0}
    periods: List[float] = []
    q_ref = list(last_state[0])
    velocity_bank = CausalVelocityBank(VELOCITY_FILTER_CANDIDATES_HZ)
    velocity_bank.reset()
    fast_motion_count = 0
    clamp_low_count = 0

    try:
        for i, x_to_cfg in enumerate(CARTESIAN_WAYPOINTS_M):
            # AIC2-style segment start: use where the arm ACTUALLY is, not
            # the previous nominal waypoint. This avoids assuming away any
            # standing tracking offset at a stop.
            x_from = fk(last_state[0])
            x_to = list(x_to_cfg)
            p_from = pitch_of(last_state[0]) if i == 0 else PITCH_RAD
            p_to = PITCH_RAD
            move_t = float(MOVE_TIMES_S[i])
            dwell_t = float(DWELL_TIMES_S[i])
            segment_t = move_t + dwell_t
            name = f"waypoint_{i + 1:02d}"

            log.info(f"--- {name}/{len(CARTESIAN_WAYPOINTS_M)}: "
                     f"[{x_from[0]:.4f}, {x_from[1]:.4f}, {x_from[2]:.4f}] "
                     f"-> [{x_to[0]:.4f}, {x_to[1]:.4f}, {x_to[2]:.4f}], "
                     f"pitch {p_from:+.4f} -> {p_to:+.4f}, "
                     f"move {move_t:.2f} s, dwell {dwell_t:.2f} s")

            worst = check_path(x_from, x_to, p_from, p_to)
            log.info("    joint headroom over this segment: "
                     + ", ".join(f"j{k + 1} {worst[k]:.3f} rad"
                                 for k in range(NJ)))
            tight = [f"joint{k + 1} ({worst[k]:.3f} rad)" for k in range(NJ)
                     if worst[k] < PATH_HEADROOM_WARN_RAD]
            if tight:
                log.warn("    little room to spare on " + ", ".join(tight)
                         + f". Under {PATH_HEADROOM_WARN_RAD} rad the barrier "
                         "and impedance can fight; a small overshoot reaches "
                         "the hard limit.")

            t_phase = time.monotonic()

            while True:
                now = time.monotonic()
                elapsed = now - t_phase
                if elapsed > segment_t:
                    break

                if now - last_hw_check > HW_CHECK_INTERVAL_S:
                    last_hw_check = now
                    bus.assert_no_hw_error("during run")

                state = bus.read_state()
                if state is None:
                    # Re-send the previous command. Commanding zero in
                    # current mode is a free fall.
                    bus.write_currents(last_command)
                    time.sleep(dt)
                    continue

                q, qd, cur = state
                last_state = state
                period = now - prev_cycle
                prev_cycle = now
                periods.append(period)
                step = min(max(period, 1e-4), 5.0 * dt)

                # ---- setpoint --------------------------------------
                s = min(1.0, elapsed / move_t) if move_t > 0 else 1.0
                x_d = lerp_xyz(x_from, x_to, s)
                pitch_d = p_from + (p_to - p_from) * quintic(s)
                motion = "move" if elapsed < move_t else "dwell"

                q_try = ik_pitch(x_d[0], x_d[1], x_d[2], pitch_d)
                if q_try is not None:
                    q_ref = q_try           # else keep the last good one

                # ---- measured state --------------------------------
                x = fk(q)
                jm = jacobian(q)
                # Dynamixel Present Velocity remains a diagnostic only.
                v_servo = j_times(jm, qd)
                qd_candidates = velocity_bank.update(q, step)
                qd_ctrl = qd_candidates[CONTROL_VEL_CUTOFF_HZ]
                v_ctrl = j_times(jm, qd_ctrl)
                v_candidates = {fc: j_times(jm, qdf)
                                for fc, qdf in qd_candidates.items()}
                pitch = pitch_of(q)

                err = [a - b for a, b in zip(x_d, x)]
                err_norm = math.sqrt(sum(e * e for e in err))
                pitch_err = pitch_d - pitch
                stats["max_err"] = max(stats["max_err"], err_norm)
                stats["max_pitch_err"] = max(stats["max_pitch_err"],
                                             abs(pitch_err))

                # ---- safety ----------------------------------------
                if err_norm > MAX_TIP_ERROR_M:
                    raise RuntimeError(
                        f"Tip error {err_norm * 1000:.0f} mm exceeds the "
                        f"{MAX_TIP_ERROR_M * 1000:.0f} mm limit. Stopping.")
                if abs(pitch_err) > MAX_PITCH_ERROR_RAD:
                    raise RuntimeError(
                        f"Tip pitch {pitch:+.3f} rad against a commanded "
                        f"{pitch_d:+.3f}. Posture collapse. Stopping.")
                for k, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
                    if (q[k] < lo + HARD_LIMIT_MARGIN_RAD
                            or q[k] > hi - HARD_LIMIT_MARGIN_RAD):
                        raise RuntimeError(
                            f"joint{k + 1} = {q[k]:.4f} rad is past the "
                            "barrier and at its hard limit. Stopping.")

                if (abs(v_ctrl[2]) > MAX_CTRL_VZ_M_S
                        or max(abs(u) for u in qd_ctrl) > MAX_CTRL_QD_RAD_S):
                    fast_motion_count += 1
                else:
                    fast_motion_count = 0
                if fast_motion_count >= FAST_ABORT_CONSECUTIVE_CYCLES:
                    raise RuntimeError(
                        f"Fast-motion abort: vz_ctrl={v_ctrl[2]:+.3f} m/s, "
                        f"max|qd_ctrl|={max(abs(u) for u in qd_ctrl):.3f} "
                        "rad/s. Stopping before the AIC4 banging regime.")

                ramp = min(1.0, (now - start) / RAMP_TIME_S)

                # ---- integral --------------------------------------
                # Held at zero until the ramp finishes so it cannot wind
                # up during the mode switch. Clamped per axis.
                if ramp >= 1.0:
                    for k in range(3):
                        integral[k] = max(-I_CLAMP, min(
                            I_CLAMP, integral[k] + KI[k] * err[k] * step))

                # ---- Cartesian impedance ---------------------------
                # Only restoring action ramps. Damping is full-strength from
                # the first cycle because it removes kinetic energy.
                force_restore = [KC[k] * err[k] + integral[k]
                                 for k in range(3)]
                force_damp = [-DC[k] * v_ctrl[k] for k in range(3)]
                force = [ramp * force_restore[k] + force_damp[k]
                         for k in range(3)]
                imp = jt_times(jm, force)

                # ---- joint-space regularisation --------------------
                reg_limit = REG_FRACTION * MAX_GOAL_CURRENT
                reg_p = [KP_JOINT * (q_ref[k] - q[k]) for k in range(NJ)]
                reg_d = [-KD_JOINT * qd_ctrl[k] for k in range(NJ)]
                reg = [max(-reg_limit, min(
                           reg_limit, ramp * reg_p[k] + reg_d[k]))
                       for k in range(NJ)]

                # ---- feedforwards ----------------------------------
                grav = gravity_model.gravity_current(q)
                bus.hold_current = list(grav)      # for park()'s fallback
                fc = table_at(samples, q, "friction_current")
                # Coulomb compensation acts in the direction of motion. Gate
                # it during fast transients so it does not cancel the passive
                # friction that helps the damping arrest the arm.
                settling_fast = (max(abs(u) for u in qd_ctrl)
                                 > FRICTION_GATE_RAD_S)
                fric = ([0.0] * NJ if settling_fast
                        else friction_torque(fc, qd_ctrl))
                bar = barrier_torque(q)

                # ---- assemble --------------------------------------
                extra = [imp[k] + reg[k] + fric[k] + bar[k]
                         for k in range(NJ)]
                command, alpha = clamp_preserving(grav, extra)
                stats["min_alpha"] = min(stats["min_alpha"], alpha)
                stats["max_cmd"] = max(stats["max_cmd"],
                                       max(abs(c) for c in command))
                if alpha < CLAMP_ABORT_ALPHA:
                    clamp_low_count += 1
                else:
                    clamp_low_count = 0
                if clamp_low_count >= CLAMP_ABORT_CONSECUTIVE_CYCLES:
                    raise RuntimeError(
                        f"Clamp abort: alpha={alpha:.3f} stayed below "
                        f"{CLAMP_ABORT_ALPHA:.2f} for "
                        f"{CLAMP_ABORT_CONSECUTIVE_CYCLES} cycles. Stopping.")

                if not bus.write_currents(command):
                    log.warn("Goal current write failed on one cycle.")
                last_command = command
                node.publish(q, qd, cur)

                # ---- force estimate from MEASURED current ----------
                f_est, f_raw = estimate_tip_force(jm, cur, grav, fric)
                tau_ext = [cur[k] - grav[k] - fric[k] for k in range(NJ)]

                # ---- log -------------------------------------------
                row = {
                    "t": now - start, "phase": name, "waypoint": i + 1,
                    "motion": motion, "s": s,
                    "period": period, "ramp": ramp, "clamp_alpha": alpha,
                    "read_fail": bus.read_failures,
                    "x": x[0], "y": x[1], "z": x[2], "pitch": pitch,
                    "vx": v_servo[0], "vy": v_servo[1], "vz": v_servo[2],
                    "vx_ctrl": v_ctrl[0], "vy_ctrl": v_ctrl[1],
                    "vz_ctrl": v_ctrl[2],
                    "xd": x_d[0], "yd": x_d[1], "zd": x_d[2],
                    "pitch_d": pitch_d,
                    "ex": err[0], "ey": err[1], "ez": err[2],
                    "err_mm": err_norm * 1000.0, "pitch_err": pitch_err,
                    "fx_cmd_N": force[0] / TORQUE_SCALE,
                    "fy_cmd_N": force[1] / TORQUE_SCALE,
                    "fz_cmd_N": force[2] / TORQUE_SCALE,
                    "ix_N": integral[0] / TORQUE_SCALE,
                    "iy_N": integral[1] / TORQUE_SCALE,
                    "iz_N": integral[2] / TORQUE_SCALE,
                    "fx_est_N": f_est[0], "fy_est_N": f_est[1],
                    "fz_est_N": f_est[2],
                    "f_est_N": math.sqrt(sum(u * u for u in f_est)),
                    "fx_raw_N": f_raw[0], "fy_raw_N": f_raw[1],
                    "fz_raw_N": f_raw[2],
                    "f_raw_N": math.sqrt(sum(u * u for u in f_raw)),
                }
                pack_joints(row, "q", q)
                pack_joints(row, "qd", qd)
                pack_joints(row, "qd_ctrl", qd_ctrl)

                for cutoff in VELOCITY_FILTER_CANDIDATES_HZ:
                    tg = _filter_tag(cutoff)
                    pack_joints(
                        row, tg + "_qd", qd_candidates[cutoff]
                    )
                    vc = v_candidates[cutoff]
                    row[tg + "_vx"] = vc[0]
                    row[tg + "_vy"] = vc[1]
                    row[tg + "_vz"] = vc[2]

                pack_joints(row, "cur", cur)
                pack_joints(row, "cur_mA", [c * CUR_UNIT_MA for c in cur])
                pack_joints(row, "cur_Nm", [c / TORQUE_SCALE for c in cur])
                pack_joints(row, "qref", q_ref)
                pack_joints(row, "grav", grav)
                pack_joints(row, "fric", fric)
                pack_joints(row, "imp", imp)
                pack_joints(row, "reg", reg)
                pack_joints(row, "bar", bar)
                pack_joints(row, "cmd", command)
                pack_joints(row, "cmd_Nm",
                            [c / TORQUE_SCALE for c in command])
                pack_joints(row, "tau_ext", tau_ext)
                cap.write(fmt_row(row))

                sleep_for = dt - (time.monotonic() - now)
                if sleep_for > 0:
                    time.sleep(sleep_for)

            tip = fk(last_state[0])
            log.info(f"  end of {name}: tip "
                     f"[{tip[0]:.4f}, {tip[1]:.4f}, {tip[2]:.4f}]  "
                     f"({math.dist(tip, x_to) * 1000:.1f} mm from target)  "
                     f"pitch {pitch_of(last_state[0]):+.4f}")
    finally:
        cap.close()

    ordered = sorted(periods) if periods else [0.0]
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]

    log.info(f"Largest tip error: {stats['max_err'] * 1000:.1f} mm")
    log.info(f"Largest pitch error: {stats['max_pitch_err']:.4f} rad")
    log.info(f"Peak command: {stats['max_cmd']:.0f} units "
             f"({stats['max_cmd'] * CUR_UNIT_MA:.0f} mA)")
    log.info(f"Smallest clamp alpha: {stats['min_alpha']:.3f}"
             + ("  (clamp shaped the response, not the gains)"
                if stats["min_alpha"] < 0.95 else ""))
    log.info(f"Loop period: median {p50 * 1000:.2f} ms, "
             f"p95 {p95 * 1000:.2f} ms, target {1000.0 / CONTROL_HZ:.1f} ms")
    log.info(f"Read failures: {bus.read_failures}/{bus.read_attempts}")

    meta = base_meta("run")
    meta["results"] = {
        "rows": cap.rows,
        "max_tip_error_mm": stats["max_err"] * 1000.0,
        "max_pitch_error_rad": stats["max_pitch_err"],
        "peak_command_units": stats["max_cmd"],
        "min_clamp_alpha": stats["min_alpha"],
        "loop_period_median_ms": p50 * 1000.0,
        "loop_period_p95_ms": p95 * 1000.0,
        "read_failures": f"{bus.read_failures}/{bus.read_attempts}",
    }
    meta["calibration_file"] = CALIB_FILE
    meta["gravity_compensation"]["xacro_path"] = gravity_model.xacro_path
    meta["gravity_compensation"]["kdl_vs_calibration"] = gravity_diagnostic
    cap.meta(meta)
    log.info(f"Capture: {cap.dir}")
    return cap.dir


# ============================================================
# PLOTS
#
# Every title states WHERE the signal came from:
#   MEASURED    read off the servo
#   CALCULATED  derived here, with the formula
#   COMMANDED   sent to the servo by this script
# ============================================================

FIGURES_RUN = [
    ("01_joint_measured",
     "Joint state  |  MEASURED (Dynamixel sync read of Present "
     "Position/Velocity/Current)",
     [("Position  [rad]  = (raw - 2048)*2pi/4096",
       [(f"q{k+1}", f"joint{k+1}") for k in range(NJ)]),
      ("Velocity  [rad/s]  = raw*0.229 rpm*2pi/60  (quantum 0.024)",
       [(f"qd{k+1}", f"joint{k+1}") for k in range(NJ)]),
      ("Present Current  [mA]  = raw*2.69",
       [(f"cur_mA{k+1}", f"joint{k+1}") for k in range(NJ)])]),

    ("02_joint_command_breakdown",
     "Commanded joint torque, split by term  |  COMMANDED by this script"
     "\nCALCULATED: cmd = clamp_preserving(g_KDL(q), ramp*(J^T f + reg) "
     "+ fric + barrier),  N.m = units / TORQUE_SCALE (=180, borrowed)",
     [(f"joint{k+1}  [current units]",
       [(f"grav{k+1}", "gravity FF (KDL/URDF)"),
        (f"fric{k+1}", "Coulomb FF (calib table x tanh)"),
        (f"imp{k+1}", "impedance J^T f"),
        (f"reg{k+1}", "joint reg to IK(x_d)"),
        (f"bar{k+1}", "limit barrier"),
        (f"cmd{k+1}", "TOTAL commanded"),
        (f"cur{k+1}", "MEASURED current")])
      for k in range(NJ)]),

    ("03_cartesian_tracking",
     "Tip pose  |  measured trace is CALCULATED: forward kinematics of "
     "MEASURED q (this script's fk(), URDF link lengths)"
     "\ndesired trace is COMMANDED: piecewise quintic 10s^3-15s^4+6s^5 between "
     "configured waypoints, with explicit dwells",
     [("x  [m]", [("xd", "COMMANDED x_d"), ("x", "CALCULATED fk(q)")]),
      ("y  [m]", [("yd", "COMMANDED y_d"), ("y", "CALCULATED fk(q)")]),
      ("z  [m]", [("zd", "COMMANDED z_d"), ("z", "CALCULATED fk(q)")]),
      ("pitch  [rad]  = -(q2+q3+q4)",
       [("pitch_d", "COMMANDED"), ("pitch", "CALCULATED from MEASURED q")])]),

    ("04_cartesian_error",
     "Tip tracking error  |  CALCULATED: e = x_d - fk(q_measured)",
     [("component error  [m]",
       [("ex", "e_x"), ("ey", "e_y"), ("ez", "e_z")]),
      ("error norm  [mm]", [("err_mm", "|e|")]),
      ("pitch error  [rad]  = PITCH_RAD - pitch(q)",
       [("pitch_err", "pitch error")])]),

    ("05_cartesian_velocity",
     "Tip velocity  |  servo Present Velocity is DIAGNOSTIC; damping uses "
     "causal position-derived filtered velocity"
     "\nAIC4 failure: Present Velocity lagged position-derived motion by ~51.7 ms "
     "at 9.34 Hz (~174 deg)",
     [("selected controller velocity  [m/s]",
       [("vx_ctrl", "v_x ctrl"), ("vy_ctrl", "v_y ctrl"),
        ("vz_ctrl", "v_z ctrl")]),
      ("vertical velocity source comparison  [m/s]",
       [("vz", "servo Present Velocity-derived v_z"),
        ("vz_ctrl", "SELECTED control v_z")]
       + [(_filter_tag(fc) + "_vz", f"candidate {fc:.0f} Hz")
          for fc in VELOCITY_FILTER_CANDIDATES_HZ])]),

    ("06_commanded_impedance_force",
     "Commanded impedance force  |  COMMANDED, CALCULATED as "
     "f = K.e + I - D.v_ctrl, then converted by / TORQUE_SCALE (=180, borrowed)"
     "\nThis is what the controller ASKED for. It is not a measurement.",
     [("commanded force  [N]",
       [("fx_cmd_N", "f_x"), ("fy_cmd_N", "f_y"), ("fz_cmd_N", "f_z")]),
      ("integral term  [N]",
       [("ix_N", "I_x"), ("iy_N", "I_y"), ("iz_N", "I_z")])]),

    ("07_estimated_tip_force",
     "Estimated tip force applied to the environment  |  NO FORCE SENSOR"
     "\nCALCULATED from MEASURED current: "
     "F = pinv(J^T)(I_meas - g_KDL(q) - f_c(q)tanh(qdot_ctrl/eps)) / TORQUE_SCALE"
     "\nCAVEAT: residual joint friction dominates, order +/-2 N in x. "
     "Absolute values indicative only; subtract a matched unloaded run.",
     [("friction-compensated  [N]",
       [("fx_est_N", "F_x"), ("fy_est_N", "F_y"), ("fz_est_N", "F_z"),
        ("f_est_N", "|F|")]),
      ("raw, no friction term  [N]  (difference = size of the correction)",
       [("fx_raw_N", "F_x raw"), ("fy_raw_N", "F_y raw"),
        ("fz_raw_N", "F_z raw"), ("f_raw_N", "|F| raw")]),
      ("residual joint torque  [current units]  "
       "tau_ext = I_meas - g(q) - f_c",
       [(f"tau_ext{k+1}", f"joint{k+1}") for k in range(NJ)])]),

    ("08_joint_reference",
     "Joint reference vs measured  |  reference is CALCULATED: closed-form "
     "IK of the COMMANDED tip setpoint at PITCH_RAD"
     "\nThis is what the joint-space regularisation pulls toward. It "
     "constrains all four joints, including the null(J) direction J^T "
     "cannot reach.",
     [(f"joint{k+1}  [rad]",
       [(f"qref{k+1}", "CALCULATED IK(x_d)"),
        (f"q{k+1}", "MEASURED")]) for k in range(NJ)]),

    ("09_loop_health",
     "Control loop health  |  MEASURED wall clock and internal state",
     [("cycle period  [s]  (target 1/CONTROL_HZ)", [("period", "period")]),
      ("impedance ramp and clamp scale  [-]  "
       "alpha<1 means the clamp is shaping the response, not the gains",
       [("ramp", "startup ramp"), ("clamp_alpha", "clamp alpha")]),
      ("cumulative sync-read failures  [-]", [("read_fail", "failures")])]),
]

FIGURES_CALIB = [
    ("01_joint_measured",
     "Joint state during calibration  |  MEASURED (Dynamixel sync read)",
     [("Position  [rad]", [(f"q{k+1}", f"joint{k+1}") for k in range(NJ)]),
      ("Velocity  [rad/s]", [(f"qd{k+1}", f"joint{k+1}") for k in range(NJ)]),
      ("Present Current  [mA]",
       [(f"cur_mA{k+1}", f"joint{k+1}") for k in range(NJ)])]),

    ("02_hold_current",
     "Hold current per joint  |  MEASURED Present Current in position "
     "control\nThe forward and reverse passes differ by twice the Coulomb "
     "friction. gravity=(fwd+rev)/2, friction=|fwd-rev|/2.",
     [(f"joint{k+1}  [current units]", [(f"cur{k+1}", f"joint{k+1}")])
      for k in range(NJ)]),

    ("03_cartesian_tracking",
     "Tip pose during calibration  |  CALCULATED: fk(MEASURED q) against "
     "the COMMANDED position-control target",
     [("x  [m]", [("xd", "COMMANDED"), ("x", "CALCULATED fk(q)")]),
      ("z  [m]", [("zd", "COMMANDED"), ("z", "CALCULATED fk(q)")]),
      ("pitch  [rad]",
       [("pitch_d", "COMMANDED"), ("pitch", "CALCULATED")]),
      ("error norm  [mm]", [("err_mm", "|x_d - fk(q)|")])]),
]


def read_capture_csv(path: str) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for name in header:
            columns[name] = []
        for raw in reader:
            for name, value in zip(header, raw):
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(float("nan"))
    return columns


def make_plots(capture_dir: str, figures, log=None) -> int:
    """Render every figure spec into capture_dir/plots. Returns the count."""
    def say(text):
        if log:
            log.info(text)
        else:
            print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        say("matplotlib is not installed; skipping plots. "
            "pip install matplotlib, then: AIC6.py --replot " + capture_dir)
        return 0

    csv_path = os.path.join(capture_dir, "data.csv")
    if not os.path.exists(csv_path):
        say(f"No data.csv in {capture_dir}")
        return 0

    data = read_capture_csv(csv_path)
    if not data.get("t"):
        say("data.csv has no rows; nothing to plot.")
        return 0

    t = data["t"]
    plots_dir = os.path.join(capture_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    made = 0

    for name, title, panels in figures:
        panels = [(label, [(col, lbl) for col, lbl in series
                           if col in data and any(
                               v == v for v in data[col])])
                  for label, series in panels]
        panels = [p for p in panels if p[1]]
        if not panels:
            continue

        rows = len(panels)
        fig, axes = plt.subplots(rows, 1, figsize=(14, 3.1 * rows + 1.6),
                                 sharex=True, squeeze=False)
        fig.suptitle(title, fontsize=11, fontweight="bold", ha="left", x=0.01)

        for ax, (label, series) in zip(axes[:, 0], panels):
            for col, lbl in series:
                ax.plot(t, data[col], linewidth=1.2, label=lbl)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper left",
                      bbox_to_anchor=(1.005, 1.0), borderaxespad=0.0)
        axes[-1, 0].set_xlabel("time  [s]", fontsize=9)

        fig.tight_layout(rect=(0, 0, 0.83, 0.94))
        out = os.path.join(plots_dir, f"{name}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        made += 1

    say(f"Wrote {made} figures to {plots_dir}")
    return made


def figures_for(capture_dir: str):
    meta_path = os.path.join(capture_dir, "meta.json")
    mode = ""
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as handle:
                mode = json.load(handle).get("mode", "")
        except Exception:
            mode = ""
    if not mode:
        mode = "calibrate" if "calib" in os.path.basename(capture_dir) else "run"
    return FIGURES_CALIB if mode.startswith("calib") else FIGURES_RUN


# ============================================================
# MAIN
# ============================================================

CALIB_XYZ = calibration_samples()


def build_calib_plan() -> List[List[float]]:
    """Validate settings and solve every waypoint/calibration pose."""
    validate_waypoint_settings()
    plan = []
    for i, xyz in enumerate(CALIB_XYZ):
        q = ik_pitch(xyz[0], xyz[1], xyz[2], PITCH_RAD)
        if q is None:
            raise RuntimeError(
                f"Waypoint {i + 1} {xyz} is out of reach at pitch "
                f"{math.degrees(PITCH_RAD):.1f} deg.")
        for k, (angle, (lo, hi)) in enumerate(zip(q, JOINT_LIMITS_RAD)):
            if not (lo < angle < hi):
                raise RuntimeError(
                    f"Waypoint {i + 1}: joint{k + 1} = "
                    f"{angle:.4f} rad is outside ({lo}, {hi}). Check "
                    "JOINT_LIMITS_RAD against your URDF.")
        plan.append(q)
    return plan


def velocity_filter_sweep(capture_dir: str) -> str:
    """Offline replay of every candidate using measured q from a capture.

    Creates one sub-capture per cutoff plus summary.csv. Touches no hardware.
    The central-difference velocity is used only as an offline reference, never
    as a causal controller signal.
    """
    src = os.path.join(capture_dir, "data.csv")
    if not os.path.exists(src):
        raise RuntimeError(f"No data.csv in {capture_dir}")
    with open(src, newline="") as h:
        rows = list(csv.DictReader(h))
    if len(rows) < 3:
        raise RuntimeError("Capture has too few rows for a filter sweep.")

    def num(r, key):
        try:
            return float(r[key])
        except Exception:
            return float("nan")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = os.path.join(capture_dir, f"velocity_filter_sweep_{stamp}")
    os.makedirs(root, exist_ok=True)

    qd_fd = [[float("nan")] * NJ for _ in rows]
    vz_fd = [float("nan")] * len(rows)
    for i in range(1, len(rows) - 1):
        dt2 = num(rows[i + 1], "t") - num(rows[i - 1], "t")
        if not math.isfinite(dt2) or dt2 <= 0:
            continue
        qdm = [(num(rows[i + 1], f"q{k+1}") -
                num(rows[i - 1], f"q{k+1}")) / dt2 for k in range(NJ)]
        qd_fd[i] = qdm
        q = [num(rows[i], f"q{k+1}") for k in range(NJ)]
        vz_fd[i] = j_times(jacobian(q), qdm)[2]

    summary = []
    for fc in VELOCITY_FILTER_CANDIDATES_HZ:
        outdir = os.path.join(root, f"candidate_{int(round(fc)):02d}Hz")
        os.makedirs(outdir, exist_ok=True)
        bank = CausalVelocityBank([fc])
        q0 = [num(rows[0], f"q{k+1}") for k in range(NJ)]
        bank.reset(q0)
        records = []
        powers = []
        errs = []
        quiet_vals = []
        prev_t = num(rows[0], "t")
        for i, r in enumerate(rows):
            t = num(r, "t")
            q = [num(r, f"q{k+1}") for k in range(NJ)]
            dt1 = t - prev_t if i else 1.0 / CONTROL_HZ
            prev_t = t
            qdc = bank.update(q, dt1)[fc]
            vc = j_times(jacobian(q), qdc)
            ref = vz_fd[i]
            fD = -D_CART_NSPM[2] * vc[2]
            power = fD * ref if math.isfinite(ref) else float("nan")
            if math.isfinite(power) and 4.5 <= t <= 6.3:
                powers.append(power)
            if math.isfinite(ref) and 4.5 <= t <= 6.3:
                errs.append((vc[2] - ref) ** 2)
            if 0.5 <= t <= 2.0:
                quiet_vals.append(vc[2])
            rec = {
                "t": t, "vz_fd": ref,
                "vx_ctrl": vc[0], "vy_ctrl": vc[1], "vz_ctrl": vc[2],
                "damping_force_z_N": fD,
                "damping_power_vs_vz_fd_W": power,
            }
            for k in range(NJ):
                rec[f"qd_ctrl{k+1}"] = qdc[k]
                rec[f"qd_fd{k+1}"] = qd_fd[i][k]
            records.append(rec)
        fields = list(records[0].keys())
        with open(os.path.join(outdir, "data.csv"), "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            w.writerows(records)
        mean_power = sum(powers) / len(powers) if powers else float("nan")
        rms = math.sqrt(sum(errs) / len(errs)) if errs else float("nan")
        qmean = sum(quiet_vals) / len(quiet_vals) if quiet_vals else float("nan")
        qstd = (math.sqrt(sum((v - qmean) ** 2 for v in quiet_vals) /
                          len(quiet_vals)) if quiet_vals else float("nan"))
        meta = {
            "source_capture": os.path.abspath(capture_dir),
            "candidate_cutoff_Hz": fc,
            "filter": "causal backward difference + first-order LP",
            "violent_window_s": [4.5, 6.3],
            "mean_damping_power_vs_position_derived_vz_W": mean_power,
            "rms_vz_error_vs_central_difference_mps": rms,
            "quiet_vz_std_mps": qstd,
        }
        with open(os.path.join(outdir, "meta.json"), "w") as h:
            json.dump(meta, h, indent=2)
        summary.append(meta)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            tt = [x["t"] for x in records]
            plt.figure(figsize=(10, 5))
            plt.plot(tt, [x["vz_fd"] for x in records],
                     label="offline central-difference vz")
            plt.plot(tt, [x["vz_ctrl"] for x in records],
                     label=f"causal {fc:.0f} Hz vz")
            plt.xlabel("time [s]")
            plt.ylabel("z velocity [m/s]")
            plt.title(f"Velocity filter candidate {fc:.0f} Hz")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "velocity_comparison.png"), dpi=160)
            plt.close()
        except Exception:
            pass

    with open(os.path.join(root, "summary.csv"), "w", newline="") as h:
        fields = [
            "candidate_cutoff_Hz",
            "mean_damping_power_vs_position_derived_vz_W",
            "rms_vz_error_vs_central_difference_mps",
            "quiet_vz_std_mps",
            "source_capture", "filter", "violent_window_s",
        ]
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"Wrote {len(summary)} filter candidate captures to {root}")
    for m in summary:
        print(f"  {m['candidate_cutoff_Hz']:>4.0f} Hz: "
              f"mean damping power="
              f"{m['mean_damping_power_vs_position_derived_vz_W']:+.4f} W, "
              f"RMS vz err={m['rms_vz_error_vs_central_difference_mps']:.4f} m/s, "
              f"quiet std={m['quiet_vz_std_mps']:.4f} m/s")
    return root


def main():
    parser = argparse.ArgumentParser(
        description="Cartesian impedance control with current-based tip "
                    "force estimation for OpenMANIPULATOR-X.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--calibrate", action="store_true",
                       help="Phase 1. Position control. Record unloaded bidirectional "
                            "hold currents. AIC5 uses the half-difference for "
                            "friction; KDL supplies --run gravity.")
    group.add_argument("--run", action="store_true",
                       help="Phase 2. Current control. Segmented Cartesian "
                            "impedance motion with per-waypoint dwell times "
                            "and KDL/URDF gravity compensation.")
    group.add_argument("--set-mode", choices=["position", "current"],
                       help="Write EEPROM Operating Mode and exit with torque OFF. "
                            "ONLY use while the arm is physically supported. "
                            "--calibrate and --run never change Operating Mode.")
    group.add_argument("--kdl-check", action="store_true",
                       help="No hardware. Build the KDL/URDF model and compare "
                            "its gravity against the existing unloaded "
                            "calibration half-sum.")
    group.add_argument("--replot", metavar="DIR",
                       help="Regenerate plots from an existing capture "
                            "folder. Touches no hardware.")
    group.add_argument("--filter-sweep", metavar="DIR",
                       help="No hardware. Replay measured q from an existing "
                            "capture through every velocity-filter candidate; "
                            "write one candidate capture plus summary.csv.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plotting after the run.")
    parser.add_argument("--no-park", action="store_true",
                        help="On exit, cut torque instead of holding "
                             "position. The arm will fall.")
    args = parser.parse_args()

    if args.filter_sweep:
        directory = os.path.abspath(args.filter_sweep)
        if not os.path.isdir(directory):
            print(f"No such capture folder: {directory}", file=sys.stderr)
            return 1
        velocity_filter_sweep(directory)
        return 0

    if args.replot:
        directory = os.path.abspath(args.replot)
        if not os.path.isdir(directory):
            print(f"No such capture folder: {directory}", file=sys.stderr)
            return 1
        make_plots(directory, figures_for(directory))
        return 0

    if args.kdl_check:
        class _StdoutLog:
            @staticmethod
            def info(message):
                print("[INFO] " + str(message))
            @staticmethod
            def warn(message):
                print("[WARN] " + str(message))

        log = _StdoutLog()
        samples = load_calibration()
        model = KDLGravityModel(log)
        diag = log_kdl_vs_calibration(log, model, samples)
        if diag["strong_sign_mismatches"]:
            print("[FAIL] Strong gravity sign mismatch. Do NOT run --run.")
            return 2
        print("[PASS] KDL chain/joint order and gravity signs passed the "
              "unloaded-calibration sanity check.")
        return 0

    plan = build_calib_plan()

    rclpy.init()
    bus = None
    node = None
    capture_dir = None
    code = 0

    def on_signal(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        bus = Bus()
        bus.park_on_exit = not args.no_park
        node = ImpedanceNode()

        bus.ping_all()
        node.get_logger().info(f"All {NJ} servos answered.")
        node.get_logger().info(
            f"TORQUE_SCALE = {TORQUE_SCALE} units per N.m. Borrowed from "
            "the C++ controller, NOT independently verified. Every newton "
            "reported below inherits that uncertainty.")

        if args.set_mode:
            bus.park_on_exit = False
            target_mode = (MODE_POSITION if args.set_mode == "position"
                           else MODE_CURRENT)
            before = bus.get_mode()
            torque_before = bus.get_torque()
            node.get_logger().warn(
                "--set-mode writes EEPROM with torque OFF. The arm MUST be "
                "physically supported for this command.")
            node.get_logger().info(
                f"Operating Mode before: {before}; Torque Enable before: "
                f"{torque_before}")
            bus.set_mode(target_mode)
            after = bus.get_mode()
            torque_after = bus.get_torque()
            node.get_logger().info(
                f"Operating Mode after: {after}; Torque Enable after: "
                f"{torque_after}")
            if any(m != target_mode for m in after):
                raise RuntimeError(
                    f"Mode verification failed: wanted {target_mode}, read {after}.")
            if any(torque_after):
                raise RuntimeError(
                    f"--set-mode must leave torque OFF; read {torque_after}.")
            node.get_logger().warn(
                "Mode is stored in EEPROM and torque remains OFF. Keep the arm "
                "supported until a later command explicitly enables torque.")
        elif args.calibrate:
            capture_dir = calibrate(node, bus, plan)
        else:
            capture_dir = run_impedance(node, bus)

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
            message = f"Exit: {how}."
            if node:
                node.get_logger().warn(message)
            else:
                print(message, file=sys.stderr)
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # Fall back to the folder the Capture published, so an aborted run is
    # still plotted.
    if capture_dir is None:
        capture_dir = LAST_CAPTURE_DIR

    # Plot outside the ROS lifecycle so a plotting failure cannot leave
    # the arm in a bad state.
    if capture_dir and not args.no_plots:
        try:
            make_plots(capture_dir, figures_for(capture_dir))
        except Exception as exc:
            print(f"Plotting failed: {exc}", file=sys.stderr)
    if capture_dir:
        print(f"\nCapture folder: {capture_dir}")

    return code


if __name__ == "__main__":
    sys.exit(main())