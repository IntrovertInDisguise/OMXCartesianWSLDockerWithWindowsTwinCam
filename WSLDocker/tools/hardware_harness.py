#!/usr/bin/env python3
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


class HardwareHarness(Node):
    # ARCHITECTURAL LIMITATION (added 2026-06-02):
    # v1 approaches the spring along the z-axis (vertical drops / lifts).
    # Horizontal spring specimens (e.g. S1: k=49.4 N/m, L0=128 mm; S2: k=87.1 N/m)
    # require x-axis compression after bilateral contact.  This harness cannot
    # compress those springs and should NOT be used for spring-stiffness experiments.
    # Use hardware_harness_contact_gated_load.py (the contact-gated ablation harness)
    # or hardware_harness_v3.py for horizontal spring compression work.
    def __init__(self) -> None:
        super().__init__("hw_harness")

        self.pub1 = self.create_publisher(
            PoseStamped, "/robot1/robot1_variable_stiffness/waypoint_command", 10
        )
        self.pub2 = self.create_publisher(
            PoseStamped, "/robot2/robot2_variable_stiffness/waypoint_command", 10
        )

        self.sub1 = self.create_subscription(
            JointState, "/robot1/joint_states", self.cb1, 10
        )
        self.sub2 = self.create_subscription(
            JointState, "/robot2/joint_states", self.cb2, 10
        )

        self.wp_sub1 = self.create_subscription(
            Bool,
            "/robot1/robot1_variable_stiffness/waypoint_active",
            self.wp_cb1,
            10,
        )
        self.wp_sub2 = self.create_subscription(
            Bool,
            "/robot2/robot2_variable_stiffness/waypoint_active",
            self.wp_cb2,
            10,
        )
        self.ee_sub1 = self.create_subscription(
            Point,
            "/robot1/robot1_variable_stiffness/end_effector_position",
            self.ee_cb1,
            10,
        )
        self.ee_sub2 = self.create_subscription(
            Point,
            "/robot2/robot2_variable_stiffness/end_effector_position",
            self.ee_cb2,
            10,
        )
        self.des_sub1 = self.create_subscription(
            Pose,
            "/robot1/robot1_variable_stiffness/cartesian_pose_desired",
            self.des_cb1,
            10,
        )
        self.des_sub2 = self.create_subscription(
            Pose,
            "/robot2/robot2_variable_stiffness/cartesian_pose_desired",
            self.des_cb2,
            10,
        )

        self.js1: Optional[JointState] = None
        self.js2: Optional[JointState] = None
        self.js1_last_time: Optional[float] = None
        self.js2_last_time: Optional[float] = None
        self.wp1_active: Optional[bool] = None
        self.wp2_active: Optional[bool] = None
        self.ee1: Optional[Point] = None
        self.ee2: Optional[Point] = None
        self.des1: Optional[Pose] = None
        self.des2: Optional[Pose] = None

        self.command_frame = "absolute"
        self.command_repeats = 1
        self.command_dt = 0.1
        self.command_timeout = 6.0
        self.command_sample_delay = 0.75
        self.command_default_x = 0.22
        self.command_default_y = 0.0
        self.move_velocity_limit = 1.5
        self.hold_velocity_limit = 0.8
        self.stage3_z_offsets = [0.0, -0.02, -0.04]
        self.stage4_hold_offset = -0.04
        self.min_command_z = 0.16

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def cb1(self, msg: JointState) -> None:
        self.js1 = msg
        self.js1_last_time = self.now_s()

    def cb2(self, msg: JointState) -> None:
        self.js2 = msg
        self.js2_last_time = self.now_s()

    def wp_cb1(self, msg: Bool) -> None:
        self.wp1_active = bool(msg.data)

    def wp_cb2(self, msg: Bool) -> None:
        self.wp2_active = bool(msg.data)

    def ee_cb1(self, msg: Point) -> None:
        self.ee1 = msg

    def ee_cb2(self, msg: Point) -> None:
        self.ee2 = msg

    def des_cb1(self, msg: Pose) -> None:
        self.des1 = msg

    def des_cb2(self, msg: Pose) -> None:
        self.des2 = msg

    # ---------- helpers ----------

    def spin_for(self, duration: float, step: float = 0.05) -> None:
        end_t = self.now_s() + duration
        while self.now_s() < end_t:
            rclpy.spin_once(self, timeout_sec=step)

    def make_pose(self, x: float, y: float, z: float, frame_id: Optional[str] = None) -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.command_frame if frame_id is None else frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def current_xy(self, point: Optional[Point]) -> Tuple[float, float]:
        if point is None:
            return self.command_default_x, self.command_default_y
        return point.x, point.y

    def publish_pose_pair(self, z: float, repeats: Optional[int] = None, dt: Optional[float] = None) -> None:
        self.publish_pose_targets(
            z1=z,
            z2=z,
            repeats=repeats,
            dt=dt,
        )

    def publish_pose_targets(
        self,
        z1: float,
        z2: float,
        x1: Optional[float] = None,
        y1: Optional[float] = None,
        x2: Optional[float] = None,
        y2: Optional[float] = None,
        frame1: Optional[str] = None,
        frame2: Optional[str] = None,
        repeats: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> None:
        repeat_count = self.command_repeats if repeats is None else repeats
        step_dt = self.command_dt if dt is None else dt
        target_x1, target_y1 = self.current_xy(self.ee1) if x1 is None or y1 is None else (x1, y1)
        target_x2, target_y2 = self.current_xy(self.ee2) if x2 is None or y2 is None else (x2, y2)
        msg1 = self.make_pose(target_x1, target_y1, z1, frame_id=frame1)
        msg2 = self.make_pose(target_x2, target_y2, z2, frame_id=frame2)
        for _ in range(repeat_count):
            stamp = self.get_clock().now().to_msg()
            msg1.header.stamp = stamp
            msg2.header.stamp = stamp
            self.pub1.publish(msg1)
            self.pub2.publish(msg2)
            rclpy.spin_once(self, timeout_sec=step_dt)

    def wait_for_ee_positions(self, timeout: float = 3.0) -> bool:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            if self.ee1 is not None and self.ee2 is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def wait_for_desired_poses(self, timeout: float = 3.0) -> bool:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            if self.des1 is not None and self.des2 is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def wait_for_waypoint_idle(self, timeout: Optional[float] = None) -> bool:
        limit = self.command_timeout if timeout is None else timeout
        t0 = self.now_s()
        saw_status = False
        saw_active = False
        while self.now_s() - t0 < limit:
            statuses = []
            if self.wp1_active is not None:
                statuses.append(self.wp1_active)
            if self.wp2_active is not None:
                statuses.append(self.wp2_active)

            if statuses:
                saw_status = True
                if any(statuses):
                    saw_active = True
                if saw_active and not any(statuses):
                    return True
            rclpy.spin_once(self, timeout_sec=0.05)

        if not saw_status:
            self.spin_for(2.25)
            return True
        return False

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

    def max_abs_velocity(self, js: Optional[JointState]) -> Optional[float]:
        if js is None or js.velocity is None or len(js.velocity) == 0:
            return None
        return max(abs(v) for v in js.velocity)

    def check_limits(self, js: Optional[JointState], limit: float = 3.0) -> bool:
        vmax = self.max_abs_velocity(js)
        return vmax is not None and vmax < limit

    def snapshot_velocities(self) -> Dict[str, Optional[float]]:
        return {
            "robot1_max_vel": self.max_abs_velocity(self.js1),
            "robot2_max_vel": self.max_abs_velocity(self.js2),
        }

    def visible_topics(self, token: str) -> List[str]:
        topics = self.get_topic_names_and_types()
        return sorted(name for name, _ in topics if token in name)

    def safe_abort(self) -> None:
        if self.pub1.get_subscription_count() == 0 and self.pub2.get_subscription_count() == 0:
            self.get_logger().error(
                "ABORT: no waypoint subscribers; dual hardware bringup is not active or not visible in this shell"
            )
            return
        self.get_logger().error("ABORT: sending neutral pose")
        self.publish_pose_pair(0.25)
        self.wait_for_waypoint_idle(timeout=self.command_timeout)

    # ---------- stages ----------

    def stage1_liveness(self) -> Tuple[bool, str, Dict[str, Any]]:
        pubs_ok = self.wait_for_subscribers(timeout=3.0)
        js_ok = self.wait_joint_states(timeout=5.0)
        ee_ok = self.wait_for_ee_positions(timeout=3.0)
        des_ok = self.wait_for_desired_poses(timeout=3.0)
        ok = pubs_ok and js_ok and ee_ok and des_ok
        info = {
            "pub1_subs": self.pub1.get_subscription_count(),
            "pub2_subs": self.pub2.get_subscription_count(),
            "js1_seen": self.js1 is not None,
            "js2_seen": self.js2 is not None,
            "wp1_status_seen": self.wp1_active is not None,
            "wp2_status_seen": self.wp2_active is not None,
            "ee1_seen": self.ee1 is not None,
            "ee2_seen": self.ee2 is not None,
            "des1_seen": self.des1 is not None,
            "des2_seen": self.des2 is not None,
            "visible_joint_topics": self.visible_topics("joint_states"),
            "visible_waypoint_topics": self.visible_topics("waypoint"),
        }
        if ok:
            message = "publishers connected and joint_states present"
        elif (
            info["pub1_subs"] == 0
            and info["pub2_subs"] == 0
            and not info["js1_seen"]
            and not info["js2_seen"]
        ):
            message = "bringup not running or ROS graph not visible in this shell"
        elif not pubs_ok:
            message = "joint_states visible but waypoint subscribers not connected"
        elif not ee_ok:
            message = "joint_states visible but end_effector_position topics missing"
        elif not des_ok:
            message = "joint_states visible but cartesian_pose_desired topics missing"
        else:
            message = "waypoint subscribers connected but joint_states missing"
        return ok, message, info

    def compute_sync_waypoint_targets(self, z_offset: float) -> Optional[Dict[str, float]]:
        if self.des1 is None or self.des2 is None:
            return None

        common_x = 0.5 * (self.des1.position.x + self.des2.position.x)
        common_y = 0.5 * (self.des1.position.y + self.des2.position.y)
        common_z = max(0.5 * (self.des1.position.z + self.des2.position.z) + z_offset, self.min_command_z)

        return {
            "common_target_x": common_x,
            "common_target_y": common_y,
            "common_target_z": common_z,
            "robot1_offset_x": common_x - self.des1.position.x,
            "robot1_offset_y": common_y - self.des1.position.y,
            "robot1_offset_z": common_z - self.des1.position.z,
            "robot2_offset_x": common_x - self.des2.position.x,
            "robot2_offset_y": common_y - self.des2.position.y,
            "robot2_offset_z": common_z - self.des2.position.z,
        }

    def stage2_idle(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.spin_for(0.5)
        info = self.snapshot_velocities()
        ok = self.check_limits(self.js1) and self.check_limits(self.js2)
        return ok, "low velocity idle", info

    def stage3_sync_move(self) -> Tuple[bool, str, Dict[str, Any]]:
        stage_info: List[Dict[str, Any]] = []

        for z_offset in self.stage3_z_offsets:
            target_info = self.compute_sync_waypoint_targets(z_offset)
            if target_info is None:
                return False, "desired pose topics unavailable", {"samples": stage_info}

            self.publish_pose_targets(
                x1=target_info["robot1_offset_x"],
                y1=target_info["robot1_offset_y"],
                z1=target_info["robot1_offset_z"],
                x2=target_info["robot2_offset_x"],
                y2=target_info["robot2_offset_y"],
                z2=target_info["robot2_offset_z"],
                frame1="offset",
                frame2="offset",
            )
            self.spin_for(self.command_sample_delay)

            snap = {
                "z_offset": z_offset,
                **target_info,
                **self.snapshot_velocities(),
            }
            stage_info.append(snap)

            if not self.check_limits(self.js1, self.move_velocity_limit) or not self.check_limits(self.js2, self.move_velocity_limit):
                return False, "velocity spike", {"samples": stage_info}
        return True, "synchronous motion ok", {"samples": stage_info}

    def stage4_hold(self) -> Tuple[bool, str, Dict[str, Any]]:
        target_info = self.compute_sync_waypoint_targets(self.stage4_hold_offset)
        if target_info is None:
            return False, "desired pose topics unavailable", self.snapshot_velocities()

        self.publish_pose_targets(
            x1=target_info["robot1_offset_x"],
            y1=target_info["robot1_offset_y"],
            z1=target_info["robot1_offset_z"],
            x2=target_info["robot2_offset_x"],
            y2=target_info["robot2_offset_y"],
            z2=target_info["robot2_offset_z"],
            frame1="offset",
            frame2="offset",
        )
        self.spin_for(1.0)
        info = {
            **target_info,
            **self.snapshot_velocities(),
        }
        ok = self.check_limits(self.js1, self.hold_velocity_limit) and self.check_limits(self.js2, self.hold_velocity_limit)
        return ok, "hold stable", info

    # ---------- run ----------

    def run(self) -> List[Dict[str, Any]]:
        stages = [
            self.stage1_liveness,
            self.stage2_idle,
            self.stage3_sync_move,
            self.stage4_hold,
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
            self.get_logger().info(f"Stage {idx}: {msg} -> {ok} | {info}")

            if not ok:
                self.safe_abort()
                return results

        self.get_logger().info("ALL STAGES PASSED")
        return results


def main() -> None:
    rclpy.init()
    node = HardwareHarness()
    try:
        out = node.run()
        print(json.dumps(out, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
