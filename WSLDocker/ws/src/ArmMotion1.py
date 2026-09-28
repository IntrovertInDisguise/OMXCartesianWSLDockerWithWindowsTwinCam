#!/usr/bin/env python3
"""
ArmMotion.py

ROS 2 Humble Cartesian waypoint script for OpenMANIPULATOR-X.

IK is solved in this file, not by MoveIt.

Why: the stock OMX MoveIt config uses position_only_ik: true. It matches
XYZ and ignores orientation, so the same XYZ gives a different wrist angle
on each call. This file fixes the tip pitch instead.

The arm has 4 joints. XYZ uses 3 of them. The one remaining freedom is the
tip pitch. Roll does not exist. Yaw is locked to joint1.

    PITCH_RAD =  0.0        tip points forward (horizontal)
    PITCH_RAD = -pi/2       tip points straight down
    PITCH_RAD = -pi/4       tip points down at 45 deg

Architecture:
    [x, y, z] + pitch
        -> ik_pitch()  (closed form, this file)
        -> [joint1, joint2, joint3, joint4]
        -> ros2_control FollowJointTrajectory action
        -> dwell
        -> next waypoint

Expected concurrent processes:
    1. Physical OpenManipulator-X bringup / ros2_control
    2. This script
    (move_group is no longer needed)
"""

import math
import sys
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


# ============================================================
# USER SETTINGS
# ============================================================

JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
]

# Tip pitch, radians. 0.0 = pointing front. -pi/2 = pointing down.
PITCH_RAD = 0.0

# Absolute Cartesian XYZ waypoints in metres, in the link1 frame.
CARTESIAN_WAYPOINTS_M: List[List[float]] = [
    [0.220, 0.000, 0.090],
    [0.250, 0.000, 0.090],
    [0.280, 0.000, 0.090],
    [0.300, 0.000, 0.090],
    [0.280, 0.000, 0.120],
     [0.250, 0.000, 0.090],
     [0.220, 0.000, 0.090],
    
]

# Time given to the trajectory controller to reach each point.
MOVE_TIMES_S = [
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
    3.0,
]

# Additional stop/hold time after each point.
DWELL_TIMES_S = [
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
    5.0,
]

# Candidate names cover the root and the /robot1 namespace.
TRAJECTORY_ACTION_CANDIDATES = [
    "/robot1/arm_controller/follow_joint_trajectory",
    "/arm_controller/follow_joint_trajectory",
]

JOINT_STATE_TOPIC_CANDIDATES = [
    "/robot1/joint_states",
    "/joint_states",
]


# ============================================================
# GEOMETRY
# ============================================================
#
# Values read from the joint origins in:
#   open_manipulator_x_description/urdf/open_manipulator_x.urdf.xacro
#
#   joint1  origin xyz="0.012 0.0 0.017"     base offset
#   joint2  origin xyz="0.0   0.0 0.0595"    shoulder height
#   joint3  origin xyz="0.024 0.0 0.128"     offset upper link
#   joint4  origin xyz="0.124 0.0 0.0"       forearm
#   end_effector_link origin xyz="0.126 0.0 0.0"
#
# Link 3 is an offset link. Its length is the hypotenuse, and DELTA is the
# angle it already carries when joint2 reads zero.

BASE_X = 0.012                        # m, radial offset of the shoulder
BASE_Z = 0.017 + 0.0595               # m, height of the shoulder
L1 = math.hypot(0.024, 0.128)         # m, shoulder to elbow  = 0.130225
DELTA = math.atan2(0.024, 0.128)      # rad, built-in tilt of L1 = 0.185348
L2 = 0.124                            # m, elbow to wrist
L3 = 0.126                            # m, wrist to tip

MAX_WRIST_REACH = L1 + L2             # m, 0.254225
MIN_WRIST_REACH = abs(L1 - L2)        # m, 0.006225


def ik_pitch(x: float, y: float, z: float, pitch: float) -> Optional[List[float]]:
    """Closed-form 4-DOF inverse kinematics, elbow-up branch.

    x, y, z : tip position in metres, link1 frame
    pitch   : tip elevation in radians. 0 = forward, -pi/2 = straight down.

    Returns [j1, j2, j3, j4] in radians, or None if the point is out of
    reach for that pitch.
    """
    j1 = math.atan2(y, x)

    # Reduce to the 2D plane that contains the arm.
    r = math.hypot(x, y) - BASE_X
    zp = z - BASE_Z

    # Step back along the last link to find the wrist.
    rw = r - L3 * math.cos(pitch)
    zw = zp - L3 * math.sin(pitch)

    d = math.hypot(rw, zw)

    if d > MAX_WRIST_REACH or d < MIN_WRIST_REACH:
        return None

    # Law of cosines on the shoulder-elbow-wrist triangle.
    cos_beta = (L1 * L1 + L2 * L2 - d * d) / (2.0 * L1 * L2)
    cos_gamma = (d * d + L1 * L1 - L2 * L2) / (2.0 * d * L1)

    beta = math.acos(max(-1.0, min(1.0, cos_beta)))
    gamma = math.acos(max(-1.0, min(1.0, cos_gamma)))

    # Absolute elevation of each link.
    a = math.atan2(zw, rw) + gamma      # upper link
    b = a - (math.pi - beta)            # forearm

    j2 = math.pi / 2.0 - DELTA - a
    j3 = math.pi / 2.0 + DELTA - beta
    j4 = b - pitch

    return [j1, j2, j3, j4]


def fk(q: List[float]) -> List[float]:
    """Tip XYZ in metres from joint angles. Used to check ik_pitch."""
    a = math.pi / 2.0 - DELTA - q[1]
    b = a - (math.pi / 2.0 - DELTA) - q[2]
    c = b - q[3]

    r = (BASE_X
         + L1 * math.cos(a)
         + L2 * math.cos(b)
         + L3 * math.cos(c))

    z = (BASE_Z
         + L1 * math.sin(a)
         + L2 * math.sin(b)
         + L3 * math.sin(c))

    return [r * math.cos(q[0]), r * math.sin(q[0]), z]


def max_reach_at(z: float, pitch: float) -> float:
    """Largest x reachable at this height and pitch, on the y = 0 line."""
    zw = (z - BASE_Z) - L3 * math.sin(pitch)

    if abs(zw) > MAX_WRIST_REACH:
        return float("nan")

    rw = math.sqrt(MAX_WRIST_REACH ** 2 - zw * zw)
    return BASE_X + rw + L3 * math.cos(pitch)


def _self_check():
    """Runnable check. The reference angles are a real MoveIt solution
    recorded from the physical arm, so this tests the sign convention
    against measured hardware, not against an assumption."""
    p = fk([0.0, -0.06321, 0.14415, 0.91478])
    ref = [0.220, 0.0, 0.090]
    assert all(abs(u - v) < 1e-3 for u, v in zip(p, ref)), f"fk drift: {p}"

    q = ik_pitch(0.220, 0.0, 0.090, 0.0)
    assert q is not None, "ik_pitch failed on a reachable point"
    assert all(abs(u - v) < 1e-9 for u, v in zip(fk(q), ref)), "round trip"

    # Straight down at x = 0.30 must be reported as unreachable.
    assert ik_pitch(0.300, 0.0, 0.090, -math.pi / 2.0) is None


_self_check()


# ============================================================
# HELPERS
# ============================================================

def seconds_to_duration(seconds: float) -> Duration:
    if seconds <= 0.0:
        raise ValueError("Duration must be > 0.")

    whole = int(seconds)
    nanos = int(round((seconds - whole) * 1e9))

    if nanos >= 1_000_000_000:
        whole += 1
        nanos -= 1_000_000_000

    msg = Duration()
    msg.sec = whole
    msg.nanosec = nanos
    return msg


# ============================================================
# NODE
# ============================================================

class CartesianWaypointMover(Node):

    def __init__(self):
        super().__init__("cartesian_waypoint_mover")

        self.trajectory_client: Optional[ActionClient] = None
        self.trajectory_action_name: Optional[str] = None

        self.latest_joint_state: Optional[JointState] = None
        self.joint_state_subscription = None
        self.joint_state_topic: Optional[str] = None

    # --------------------------------------------------------
    # Configuration checks
    # --------------------------------------------------------

    def validate_configuration(self):
        n = len(CARTESIAN_WAYPOINTS_M)

        if n == 0:
            raise RuntimeError("CARTESIAN_WAYPOINTS_M is empty.")

        if len(MOVE_TIMES_S) != n:
            raise RuntimeError(
                "MOVE_TIMES_S must contain one value per waypoint."
            )

        if len(DWELL_TIMES_S) != n:
            raise RuntimeError(
                "DWELL_TIMES_S must contain one value per waypoint."
            )

        for i, xyz in enumerate(CARTESIAN_WAYPOINTS_M):
            if len(xyz) != 3:
                raise RuntimeError(
                    f"Waypoint {i + 1} must be [x, y, z]. Got: {xyz}"
                )

        # Solve every point before moving anything, so an unreachable
        # point stops the run at the start and not half way through.
        self.get_logger().info(
            f"Checking {n} waypoints at pitch "
            f"{math.degrees(PITCH_RAD):.1f} deg..."
        )

        self.solutions: List[List[float]] = []

        for i, xyz in enumerate(CARTESIAN_WAYPOINTS_M):
            q = ik_pitch(xyz[0], xyz[1], xyz[2], PITCH_RAD)

            if q is None:
                limit = max_reach_at(xyz[2], PITCH_RAD)
                raise RuntimeError(
                    f"Waypoint {i + 1} {xyz} is out of reach at pitch "
                    f"{math.degrees(PITCH_RAD):.1f} deg. "
                    f"Largest x at z={xyz[2]:.3f} is about "
                    f"{limit:.4f} m. Reduce x, or use a shallower pitch."
                )

            self.solutions.append(q)

            self.get_logger().info(
                f"  wp {i + 1} {xyz} -> "
                + ", ".join(
                    f"{name}={value:.5f}"
                    for name, value in zip(JOINT_NAMES, q)
                )
            )

    # --------------------------------------------------------
    # Discover current robot joint-state topic
    # --------------------------------------------------------

    def discover_joint_state_topic(self):
        self.get_logger().info("Searching for JointState topic...")

        deadline = time.monotonic() + 10.0

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.2)

            graph: Dict[str, List[str]] = dict(
                self.get_topic_names_and_types()
            )

            for candidate in JOINT_STATE_TOPIC_CANDIDATES:
                types = graph.get(candidate, [])
                if "sensor_msgs/msg/JointState" in types:
                    self.joint_state_topic = candidate
                    self.joint_state_subscription = self.create_subscription(
                        JointState,
                        candidate,
                        self._joint_state_callback,
                        10,
                    )
                    self.get_logger().info(
                        f"Using JointState topic: {candidate}"
                    )
                    return

        raise RuntimeError(
            "No JointState topic found at "
            f"{JOINT_STATE_TOPIC_CANDIDATES}. "
            "Run: ros2 topic list -t | grep joint_states"
        )

    def _joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def wait_for_joint_state(self):
        self.get_logger().info(
            f"Waiting for measured state on {self.joint_state_topic}..."
        )

        deadline = time.monotonic() + 10.0

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.latest_joint_state is not None:
                names = set(self.latest_joint_state.name)

                missing = [
                    name for name in JOINT_NAMES
                    if name not in names
                ]

                if not missing:
                    self.get_logger().info(
                        "Measured joint state received."
                    )
                    return

        raise RuntimeError(
            "JointState messages did not contain joint1..joint4."
        )

    # --------------------------------------------------------
    # Discover active FollowJointTrajectory action
    # --------------------------------------------------------

    def connect_trajectory_controller(self):
        self.get_logger().info(
            "Searching for active arm_controller trajectory action..."
        )

        while rclpy.ok():
            for action_name in TRAJECTORY_ACTION_CANDIDATES:
                client = ActionClient(
                    self,
                    FollowJointTrajectory,
                    action_name,
                )

                if client.wait_for_server(timeout_sec=0.5):
                    self.trajectory_client = client
                    self.trajectory_action_name = action_name

                    print()
                    print("========================================")
                    print(" TRAJECTORY ACTION CONNECTION ESTABLISHED")
                    print(f" {action_name}")
                    print("========================================")
                    print()
                    return

                client.destroy()

            self.get_logger().warn(
                "No active arm trajectory action found at "
                f"{TRAJECTORY_ACTION_CANDIDATES}. "
                "The bringup arm_controller must be ACTIVE."
            )

    # --------------------------------------------------------
    # Send joint target to physical arm controller
    # --------------------------------------------------------

    def send_joint_target(
        self,
        q: List[float],
        move_time_s: float,
        waypoint_number: int,
    ) -> bool:

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(q)
        point.time_from_start = seconds_to_duration(move_time_s)

        goal.trajectory.points = [point]

        self.get_logger().info(
            f"Sending waypoint {waypoint_number} to "
            f"{self.trajectory_action_name}"
        )

        future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error(
                "Trajectory controller returned no goal handle."
            )
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                f"Waypoint {waypoint_number} was rejected."
            )
            return False

        self.get_logger().info(
            f"Waypoint {waypoint_number} accepted."
        )

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        wrapped = result_future.result()

        if wrapped is None:
            self.get_logger().error(
                f"No result for waypoint {waypoint_number}."
            )
            return False

        result = wrapped.result

        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(
                f"Waypoint {waypoint_number} failed: "
                f"error_code={result.error_code}, "
                f"error_string='{result.error_string}'"
            )
            return False

        self.get_logger().info(
            f"Waypoint {waypoint_number} reached."
        )
        return True

    # --------------------------------------------------------
    # Report the true tip position after each move
    # --------------------------------------------------------

    def report_measured_tip(self, waypoint_number: int, target: List[float]):
        """The controller runs open loop, so 'Goal reached' does not prove
        the arm arrived. This reads the encoders and reports the error."""
        if self.latest_joint_state is None:
            return

        by_name = dict(
            zip(self.latest_joint_state.name,
                self.latest_joint_state.position)
        )

        if any(name not in by_name for name in JOINT_NAMES):
            return

        measured = [float(by_name[name]) for name in JOINT_NAMES]
        tip = fk(measured)

        err = math.dist(tip, target)

        self.get_logger().info(
            f"Waypoint {waypoint_number} measured tip: "
            f"x={tip[0]:.4f}, y={tip[1]:.4f}, z={tip[2]:.4f} m "
            f"(error {err * 1000.0:.1f} mm)"
        )

    # --------------------------------------------------------
    # Execute Cartesian sequence
    # --------------------------------------------------------

    def execute(self):
        self.validate_configuration()

        self.discover_joint_state_topic()
        self.wait_for_joint_state()
        self.connect_trajectory_controller()

        for i, xyz in enumerate(CARTESIAN_WAYPOINTS_M):
            waypoint_number = i + 1
            q = self.solutions[i]

            if not self.send_joint_target(
                q=q,
                move_time_s=MOVE_TIMES_S[i],
                waypoint_number=waypoint_number,
            ):
                raise RuntimeError(
                    f"Stopping: trajectory execution failed at "
                    f"waypoint {waypoint_number}."
                )

            rclpy.spin_once(self, timeout_sec=0.1)
            self.report_measured_tip(waypoint_number, xyz)

            dwell = DWELL_TIMES_S[i]

            if dwell > 0.0:
                self.get_logger().info(
                    f"Holding waypoint {waypoint_number} "
                    f"for {dwell:.3f} s"
                )

                deadline = time.monotonic() + dwell

                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.05)

        print()
        print("========================================")
        print(" ALL CARTESIAN WAYPOINTS COMPLETED")
        print("========================================")
        print()


def main(args=None):
    rclpy.init(args=args)
    node = CartesianWaypointMover()

    try:
        node.execute()

    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user.")

    except Exception as exc:
        node.get_logger().error(str(exc))
        sys.exit(1)

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()