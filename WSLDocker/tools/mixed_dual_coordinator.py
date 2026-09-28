#!/usr/bin/env python3
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


@dataclass
class StageRecord:
    stage: int
    name: str
    passed: bool
    message: str
    info: Dict[str, Any]


class MixedDualCoordinator(Node):
    """
    Mixed hardware + Gazebo coordination harness.

    Assumptions:
    - robot1 = hardware
    - robot2 = Gazebo
    - both accept PoseStamped on waypoint_command
    - both publish JointState
    """

    def __init__(self) -> None:
        super().__init__("mixed_dual_coordinator")

        # Topics
        self.hw_wp_topic = "/robot1/robot1_variable_stiffness/waypoint_command"
        self.gz_wp_topic = "/robot2/robot2_variable_stiffness/waypoint_command"
        self.hw_js_topic = "/robot1/joint_states"
        self.gz_js_topic = "/robot2/joint_states"

        # Publishers
        self.hw_pub = self.create_publisher(PoseStamped, self.hw_wp_topic, 10)
        self.gz_pub = self.create_publisher(PoseStamped, self.gz_wp_topic, 10)

        # Subscribers
        self.hw_js_sub = self.create_subscription(JointState, self.hw_js_topic, self._hw_js_cb, 10)
        self.gz_js_sub = self.create_subscription(JointState, self.gz_js_topic, self._gz_js_cb, 10)

        # State
        self.hw_js: Optional[JointState] = None
        self.gz_js: Optional[JointState] = None
        self.hw_js_t: Optional[float] = None
        self.gz_js_t: Optional[float] = None

        self.abort_reason: Optional[str] = None
        self.records: List[StageRecord] = []

        # Limits
        self.hardware_velocity_limit = 2.5
        self.gazebo_velocity_limit = 4.0
        self.js_stale_timeout = 0.75
        self.phase_timeout = 3.0

        # Nominal poses
        self.neutral_z = 0.25
        self.free_space_sequence = [0.25, 0.20, 0.15]
        self.hardware_only_press_sequence = [0.15, 0.12, 0.10]

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _hw_js_cb(self, msg: JointState) -> None:
        self.hw_js = msg
        self.hw_js_t = self.now_s()

    def _gz_js_cb(self, msg: JointState) -> None:
        self.gz_js = msg
        self.gz_js_t = self.now_s()

    def spin_for(self, duration: float, step: float = 0.05) -> None:
        end_t = self.now_s() + duration
        while self.now_s() < end_t:
            rclpy.spin_once(self, timeout_sec=step)
            if self.abort_reason is not None:
                break

    def make_pose(self, z: float, frame_id: str = "offset") -> PoseStamped:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = 0.03
        msg.pose.position.y = 0.0
        msg.pose.position.z = z
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        return msg

    def publish_pair(self, hw_z: float, gz_z: float, repeats: int = 3, dt: float = 0.05) -> None:
        hw_msg = self.make_pose(hw_z)
        gz_msg = self.make_pose(gz_z)
        for _ in range(repeats):
            hw_msg.header.stamp = self.get_clock().now().to_msg()
            gz_msg.header.stamp = self.get_clock().now().to_msg()
            self.hw_pub.publish(hw_msg)
            self.gz_pub.publish(gz_msg)
            rclpy.spin_once(self, timeout_sec=dt)

    def max_abs_vel(self, msg: Optional[JointState]) -> Optional[float]:
        if msg is None or msg.velocity is None or len(msg.velocity) == 0:
            return None
        return max(abs(v) for v in msg.velocity)

    def js_fresh(self) -> Tuple[bool, Dict[str, Any]]:
        now = self.now_s()
        hw_age = None if self.hw_js_t is None else now - self.hw_js_t
        gz_age = None if self.gz_js_t is None else now - self.gz_js_t
        ok = (
            hw_age is not None and hw_age < self.js_stale_timeout and
            gz_age is not None and gz_age < self.js_stale_timeout
        )
        return ok, {"hw_js_age_s": hw_age, "gz_js_age_s": gz_age}

    def velocity_ok(self) -> Tuple[bool, Dict[str, Any]]:
        hw_v = self.max_abs_vel(self.hw_js)
        gz_v = self.max_abs_vel(self.gz_js)
        ok = (
            hw_v is not None and hw_v < self.hardware_velocity_limit and
            gz_v is not None and gz_v < self.gazebo_velocity_limit
        )
        return ok, {"hw_max_vel": hw_v, "gz_max_vel": gz_v}

    def subscriptions_ready(self) -> bool:
        return self.hw_pub.get_subscription_count() > 0 and self.gz_pub.get_subscription_count() > 0

    def wait_ready(self, timeout: float = 5.0) -> Tuple[bool, Dict[str, Any]]:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.subscriptions_ready():
                fresh_ok, fresh_info = self.js_fresh()
                if fresh_ok:
                    return True, {
                        "hw_pub_subs": self.hw_pub.get_subscription_count(),
                        "gz_pub_subs": self.gz_pub.get_subscription_count(),
                        **fresh_info,
                    }
        fresh_ok, fresh_info = self.js_fresh()
        return False, {
            "hw_pub_subs": self.hw_pub.get_subscription_count(),
            "gz_pub_subs": self.gz_pub.get_subscription_count(),
            **fresh_info,
        }

    def wait_phase_settle(self, timeout: float = 2.0) -> Tuple[bool, Dict[str, Any]]:
        t0 = self.now_s()
        while self.now_s() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            fresh_ok, fresh_info = self.js_fresh()
            vel_ok, vel_info = self.velocity_ok()
            if fresh_ok and vel_ok:
                return True, {**fresh_info, **vel_info, "phase_wait_s": self.now_s() - t0}
            if not fresh_ok:
                self.abort_reason = "joint_states stale"
                return False, {**fresh_info, **vel_info}
        return False, {"phase_wait_s": self.now_s() - t0, **self.velocity_ok()[1]}

    def abort_to_neutral(self) -> None:
        self.get_logger().error(f"ABORT: {self.abort_reason}")
        self.publish_pair(self.neutral_z, self.neutral_z, repeats=8, dt=0.05)
        self.spin_for(1.0)

    def run_stage(self, stage: int, name: str, fn) -> bool:
        passed, message, info = fn()
        record = StageRecord(stage=stage, name=name, passed=passed, message=message, info=info)
        self.records.append(record)
        self.get_logger().info(f"Stage {stage} [{name}] -> {passed} | {message} | {info}")
        if not passed and self.abort_reason is None:
            self.abort_reason = f"stage_{stage}_{name}_failed"
        return passed

    # ---------- stages ----------

    def stage1_bringup_ready(self) -> Tuple[bool, str, Dict[str, Any]]:
        ok, info = self.wait_ready(timeout=6.0)
        return ok, "publishers connected and joint_states fresh", info

    def stage2_idle_sanity(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.spin_for(0.5)
        fresh_ok, fresh_info = self.js_fresh()
        vel_ok, vel_info = self.velocity_ok()
        ok = fresh_ok and vel_ok
        return ok, "idle state sane", {**fresh_info, **vel_info}

    def stage3_synchronized_free_space(self) -> Tuple[bool, str, Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for z in self.free_space_sequence:
            self.publish_pair(z, z, repeats=4, dt=0.05)
            ok, info = self.wait_phase_settle(timeout=self.phase_timeout)
            info["commanded_z"] = z
            samples.append(info)
            if not ok:
                return False, "free-space synchronized move failed", {"samples": samples}
        return True, "free-space synchronized move ok", {"samples": samples}

    def stage4_hardware_only_press_debug(self) -> Tuple[bool, str, Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        gz_hold_z = self.free_space_sequence[-1]
        for hw_z in self.hardware_only_press_sequence:
            self.publish_pair(hw_z, gz_hold_z, repeats=4, dt=0.05)
            ok, info = self.wait_phase_settle(timeout=self.phase_timeout)
            info["hw_commanded_z"] = hw_z
            info["gz_commanded_z"] = gz_hold_z
            samples.append(info)
            if not ok:
                return False, "hardware-only press debug failed", {"samples": samples}
        return True, "hardware-only press debug ok", {"samples": samples}

    def stage5_return_hold(self) -> Tuple[bool, str, Dict[str, Any]]:
        self.publish_pair(self.neutral_z, self.neutral_z, repeats=6, dt=0.05)
        ok, info = self.wait_phase_settle(timeout=self.phase_timeout)
        return ok, "return to neutral and hold", info

    def run(self) -> List[Dict[str, Any]]:
        stages = [
            (1, "bringup_ready", self.stage1_bringup_ready),
            (2, "idle_sanity", self.stage2_idle_sanity),
            (3, "synchronized_free_space", self.stage3_synchronized_free_space),
            (4, "hardware_only_press_debug", self.stage4_hardware_only_press_debug),
            (5, "return_hold", self.stage5_return_hold),
        ]

        for stage_id, name, fn in stages:
            if not self.run_stage(stage_id, name, fn):
                self.abort_to_neutral()
                break

        return [asdict(r) for r in self.records]


def main() -> None:
    rclpy.init()
    node = MixedDualCoordinator()
    try:
        out = node.run()
        print(json.dumps(out, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
