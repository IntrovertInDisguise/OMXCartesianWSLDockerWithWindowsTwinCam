#!/usr/bin/env python3
"""
MeasureFriction.py

Measures the breakaway current of each OpenMANIPULATOR-X joint.

Breakaway current is the smallest Goal Current that makes a joint start to
move from rest. Below it, nothing happens at all. The XM430-W350 has a
353:1 gearbox, so this number is large and it decides whether torque-mode
impedance control is possible on this arm.

Method, per joint:
  1. Hold the arm at the test pose in position control.
  2. Record the hold current. That is the gravity term.
  3. Switch to current control and command the gravity term.
  4. Add a slowly rising extra current, in both directions.
  5. Record the extra current at which the joint first moves.

Nothing here is assumed. The output is measured on your hardware.

ros2_control must NOT be running.

    python3 MeasureFriction.py
"""

import math
import sys
import time

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

PORT_NAME = "/dev/ttyUSB0"
BAUD_RATE = 1000000
DXL_IDS = [11, 12, 13, 14]

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

MODE_CURRENT = 0
MODE_POSITION = 3

POS_UNITS_PER_REV = 4096.0
POS_CENTER = 2048.0
CUR_UNIT_MA = 2.69

# Test pose. Mid-range, away from limits. Roughly x=0.25, z=0.09, pitch 0.
TEST_POSE_RAD = [0.0, 0.20338, 0.83784, -1.04122]

RAMP_RATE = 8.0            # extra current units per second
MAX_EXTRA = 220            # give up above this
MOVE_THRESHOLD_RAD = 0.010 # about 0.6 degrees, well above encoder noise
SETTLE_S = 1.0


def to_signed(value, bits):
    limit = 1 << bits
    return value - limit if value >= limit // 2 else value


def pos_to_rad(raw):
    return (raw - POS_CENTER) * 2.0 * math.pi / POS_UNITS_PER_REV


def rad_to_pos(rad):
    return int(round(rad * POS_UNITS_PER_REV / (2.0 * math.pi) + POS_CENTER))


class Arm:
    def __init__(self):
        self.port = PortHandler(PORT_NAME)
        self.packet = PacketHandler(2.0)

        if not self.port.openPort():
            raise RuntimeError(
                f"Cannot open {PORT_NAME}. Is ros2_control still running?"
            )
        if not self.port.setBaudRate(BAUD_RATE):
            raise RuntimeError("Cannot set baud rate.")

    def torque(self, on, ids=None):
        for did in (ids or DXL_IDS):
            self.packet.write1ByteTxRx(
                self.port, did, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def set_mode(self, mode, ids=None):
        ids = ids or DXL_IDS
        self.torque(False, ids)
        time.sleep(0.05)
        for did in ids:
            result, error = self.packet.write1ByteTxRx(
                self.port, did, ADDR_OPERATING_MODE, mode)
            if result != COMM_SUCCESS or error != 0:
                raise RuntimeError(f"id {did}: mode write failed")
        time.sleep(0.05)

    def position(self, did):
        raw, result, _ = self.packet.read4ByteTxRx(
            self.port, did, ADDR_PRESENT_POSITION)
        if result != COMM_SUCCESS:
            return None
        return pos_to_rad(to_signed(raw, 32))

    def current(self, did):
        raw, result, _ = self.packet.read2ByteTxRx(self.port, did, 126)
        if result != COMM_SUCCESS:
            return None
        return float(to_signed(raw, 16))

    def goal_position(self, did, rad):
        self.packet.write4ByteTxRx(
            self.port, did, ADDR_GOAL_POSITION, rad_to_pos(rad))

    def goal_current(self, did, value):
        raw = int(round(value)) & 0xFFFF
        self.packet.write2ByteTxRx(self.port, did, ADDR_GOAL_CURRENT,
                                   to_signed(raw, 16))

    def close(self):
        try:
            for did in DXL_IDS:
                self.goal_current(did, 0)
        except Exception:
            pass
        try:
            self.torque(False)
        except Exception:
            pass
        self.port.closePort()


def hold_pose(arm):
    """Put the whole arm at the test pose in position control."""
    arm.set_mode(MODE_POSITION)
    arm.torque(True)
    for did, angle in zip(DXL_IDS, TEST_POSE_RAD):
        arm.goal_position(did, angle)
    time.sleep(3.0)


def measure_gravity(arm, did):
    acc, n = 0.0, 0
    deadline = time.monotonic() + SETTLE_S
    while time.monotonic() < deadline:
        c = arm.current(did)
        if c is not None:
            acc += c
            n += 1
        time.sleep(0.02)
    return acc / max(1, n)


def breakaway(arm, did, gravity, direction):
    """Ramp extra current until the joint moves. Returns extra current."""
    # Everything except this joint stays in position control and holds.
    arm.set_mode(MODE_CURRENT, [did])
    arm.torque(True, [did])
    arm.goal_current(did, gravity)
    time.sleep(0.5)

    start_pos = arm.position(did)
    if start_pos is None:
        raise RuntimeError(f"id {did}: cannot read position")

    extra = 0.0
    t0 = time.monotonic()
    result = None

    while extra < MAX_EXTRA:
        extra = RAMP_RATE * (time.monotonic() - t0)
        arm.goal_current(did, gravity + direction * extra)

        pos = arm.position(did)
        if pos is not None and abs(pos - start_pos) > MOVE_THRESHOLD_RAD:
            result = extra
            break

        time.sleep(0.01)

    # Release and put this joint back under position control at the pose.
    arm.goal_current(did, 0)
    arm.set_mode(MODE_POSITION, [did])
    arm.torque(True, [did])
    arm.goal_position(did, TEST_POSE_RAD[DXL_IDS.index(did)])
    time.sleep(2.0)

    return result


def main():
    arm = None
    try:
        arm = Arm()

        print("Moving to the test pose...")
        hold_pose(arm)

        print()
        print("joint   gravity     breakaway +    breakaway -")
        print("------------------------------------------------")

        results = {}

        for idx, did in enumerate(DXL_IDS):
            grav = measure_gravity(arm, did)

            up = breakaway(arm, did, grav, +1.0)
            down = breakaway(arm, did, grav, -1.0)

            results[did] = (grav, up, down)

            def show(v):
                return f"{v:7.1f}" if v is not None else "  >%3d" % MAX_EXTRA

            print(f"joint{idx + 1}  {grav:8.1f}    {show(up)}        {show(down)}")

        print()
        print("Breakaway is in Dynamixel current units.")
        print(f"One unit is about {CUR_UNIT_MA} mA. Verify in the e-Manual.")
        print()

        worst = max(
            (v for _, u, d in results.values() for v in (u, d) if v is not None),
            default=None,
        )
        if worst is None:
            print("No joint moved below the ramp ceiling. Friction is very")
            print("high, or the ramp ceiling is too low.")
        else:
            print(f"Largest measured breakaway: {worst:.1f} units.")
            print()
            print("Use this to size KC. The impedance term must exceed")
            print("breakaway at the error you are willing to tolerate:")
            print()
            print("    KC  =  breakaway / (jacobian_factor * tolerated_error)")
            print()
            print("At the poses in your trajectory the x-direction jacobian")
            print("factor is about 0.112, so for a 10 mm tolerated error:")
            print(f"    KC_x  =  {worst:.0f} / (0.112 * 0.010)"
                  f"  =  {worst / (0.112 * 0.010):.0f}")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    finally:
        if arm:
            arm.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())