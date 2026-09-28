#!/usr/bin/env python3
"""
ArmMotion_moveit.py

Standalone ROS 2 Humble Cartesian waypoint script using MoveIt ONLY for IK.

No custom ROS package is required.
No open_manipulator_msgs import is used.

Architecture:
    [x, y, z]
        -> MoveIt /compute_ik
        -> [joint1, joint2, joint3, joint4]
        -> existing ros2_control FollowJointTrajectory action
        -> dwell
        -> next waypoint

Expected concurrent processes:
    1. Physical OpenManipulator-X bringup / ros2_control
    2. MoveIt move_group
    3. This script

The official OpenManipulator-X Humble kinematics configuration uses:
    planning group: arm
    position_only_ik: true
"""

import sys
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


# ============================================================
# USER SETTINGS
# ============================================================

MOVE_GROUP = "arm"

# Express Cartesian points relative to this TF/model frame.
BASE_FRAME = "link1"

# Position-only IK is enabled in the stock OMX Humble MoveIt config.
# Leaving IK_LINK_NAME empty asks MoveIt to use the group's solver tip.
IK_LINK_NAME = ""

IK_SERVICE = "/compute_ik"

JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
]

# Absolute Cartesian XYZ waypoints in metres.
CARTESIAN_WAYPOINTS_M: List[List[float]] = [
    [0.220, 0.000, 0.090],
    [0.250, 0.000, 0.090],
    [0.280, 0.000, 0.090],
    [0.30, 0.000, 0.090],
    [0.22, 0.000, 0.090],

]

# Time given to the ros2_control trajectory controller to reach each point.
MOVE_TIMES_S = [
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
]

IK_TIMEOUT_S = 1.0
AVOID_COLLISIONS = False

# Candidate names cover the root and your /robot1 namespace.
TRAJECTORY_ACTION_CANDIDATES = [
    "/robot1/arm_controller/follow_joint_trajectory",
    "/arm_controller/follow_joint_trajectory",
]

JOINT_STATE_TOPIC_CANDIDATES = [
    "/robot1/joint_states",
    "/joint_states",
]


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

class MoveItCartesianWaypointMover(Node):

    def __init__(self):
        super().__init__("moveit_cartesian_waypoint_mover")

        self.ik_client = self.create_client(
            GetPositionIK,
            IK_SERVICE,
        )

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
    # Connect to MoveIt /compute_ik
    # --------------------------------------------------------

    def connect_moveit(self):
        self.get_logger().info(
            f"Connecting to MoveIt IK service {IK_SERVICE}..."
        )

        while rclpy.ok():
            if self.ik_client.wait_for_service(timeout_sec=1.0):
                print()
                print("========================================")
                print(" MOVEIT /compute_ik CONNECTION ESTABLISHED")
                print("========================================")
                print()
                return

            self.get_logger().warn(
                f"{IK_SERVICE} is not available. "
                "Is move_group running?"
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
    # MoveIt IK
    # --------------------------------------------------------

    def compute_ik(self, xyz: List[float]) -> Optional[List[float]]:
        request = GetPositionIK.Request()

        ik = request.ik_request
        ik.group_name = MOVE_GROUP
        ik.ik_link_name = IK_LINK_NAME
        ik.avoid_collisions = AVOID_COLLISIONS
        ik.timeout = seconds_to_duration(IK_TIMEOUT_S)

        # Explicit seed from the physical robot.
        ik.robot_state.joint_state = self.latest_joint_state

        ik.pose_stamped.header.frame_id = BASE_FRAME
        ik.pose_stamped.header.stamp = self.get_clock().now().to_msg()

        pose = ik.pose_stamped.pose

        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])

        # Stock OMX Humble MoveIt config uses position_only_ik: true,
        # so orientation is not part of the IK constraint. Still provide
        # a valid normalized quaternion.
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        self.get_logger().info(
            "MoveIt IK request: "
            f"x={xyz[0]:.4f}, "
            f"y={xyz[1]:.4f}, "
            f"z={xyz[2]:.4f} m"
        )

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()

        if response is None:
            self.get_logger().error(
                "MoveIt /compute_ik returned no response."
            )
            return None

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                "MoveIt IK failed. "
                f"MoveIt error code={response.error_code.val}"
            )
            return None

        solution = response.solution.joint_state

        positions_by_name = {
            name: position
            for name, position in zip(
                solution.name,
                solution.position,
            )
        }

        missing = [
            name for name in JOINT_NAMES
            if name not in positions_by_name
        ]

        if missing:
            self.get_logger().error(
                f"IK response is missing joints: {missing}"
            )
            return None

        q = [
            float(positions_by_name[name])
            for name in JOINT_NAMES
        ]

        self.get_logger().info(
            "MoveIt IK solution [rad]: "
            + ", ".join(
                f"{name}={value:.5f}"
                for name, value in zip(JOINT_NAMES, q)
            )
        )

        return q

    # --------------------------------------------------------
    # Send IK result to physical arm controller
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
    # Execute Cartesian sequence
    # --------------------------------------------------------

    def execute(self):
        self.validate_configuration()

        # Connect all required layers before commanding anything.
        self.discover_joint_state_topic()
        self.wait_for_joint_state()
        self.connect_moveit()
        self.connect_trajectory_controller()

        for i, xyz in enumerate(CARTESIAN_WAYPOINTS_M):
            waypoint_number = i + 1

            # Refresh the physical-state IK seed.
            rclpy.spin_once(self, timeout_sec=0.05)

            q = self.compute_ik(xyz)

            if q is None:
                raise RuntimeError(
                    f"Stopping: IK failed at Cartesian waypoint "
                    f"{waypoint_number}."
                )

            if not self.send_joint_target(
                q=q,
                move_time_s=MOVE_TIMES_S[i],
                waypoint_number=waypoint_number,
            ):
                raise RuntimeError(
                    f"Stopping: trajectory execution failed at "
                    f"waypoint {waypoint_number}."
                )

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
    node = MoveItCartesianWaypointMover()

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
