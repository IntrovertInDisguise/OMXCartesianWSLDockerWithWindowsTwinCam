#!/usr/bin/env python3
"""
Single-Arm Mixed-Run CSV Logger
================================

Writes timestamped joint_states (pos, vel, effort) for the single robot
to a CSV file.  This is the single-arm counterpart to ``mixed_logger.py``
(which logs both robot1 and robot2 for dual-arm experiments).

The output directory is ``logs/single_arm_mixed/<timestamp>/`` and contains
a single ``robot1_joint_states.csv`` file.

Usage
-----
  # Source ROS envs, then run alongside the single-arm harness:
  python3 tools/single_arm_mixed_logger.py

  # Custom output root:
  python3 tools/single_arm_mixed_logger.py --out-dir /tmp/my_logs

Environment variables:
  OMX_LOG_DIR  – log root (default ``logs/single_arm_mixed``)
"""

import argparse
import csv
import os
from datetime import datetime
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class SingleArmMixedLogger(Node):
    """ROS2 node that records joint_states for a single arm to CSV."""

    def __init__(
        self,
        out_dir: str = "logs/single_arm_mixed",
        robot_ns: str = "robot1",
    ) -> None:
        super().__init__("single_arm_mixed_logger")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(out_dir, ts)
        os.makedirs(self.out_dir, exist_ok=True)

        self.csv_path = os.path.join(self.out_dir, "robot1_joint_states.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        header: List[str] = ["timestamp", "positions", "velocities", "efforts"]
        self.csv_writer.writerow(header)

        topic = f"/{robot_ns}/joint_states"
        self.create_subscription(JointState, topic, self._cb, 50)

        self.get_logger().info(
            f"SingleArmMixedLogger writing to {self.csv_path} "
            f"(subscribed to {topic})"
        )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    @staticmethod
    def _row_from_msg(msg: JointState) -> List[str]:
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = ";".join(f"{p:.6f}" for p in msg.position) if msg.position else ""
        vel = ";".join(f"{v:.6f}" for v in msg.velocity) if msg.velocity else ""
        eff = ";".join(f"{e:.6f}" for e in msg.effort) if msg.effort else ""
        return [f"{ts:.9f}", pos, vel, eff]

    def _cb(self, msg: JointState) -> None:
        try:
            self.csv_writer.writerow(self._row_from_msg(msg))
        except Exception as exc:
            self.get_logger().error(f"Failed to write row: {exc}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self) -> None:
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-arm mixed-run CSV logger for joint_states.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get("OMX_LOG_DIR", "logs/single_arm_mixed"),
        help="Root output directory (default: $OMX_LOG_DIR or logs/single_arm_mixed).",
    )
    p.add_argument(
        "--robot-ns",
        default="robot1",
        help="Robot namespace for topic subscription (default: robot1).",
    )
    return p


def main() -> None:
    args = build_cli().parse_args()
    rclpy.init()
    node = SingleArmMixedLogger(out_dir=args.out_dir, robot_ns=args.robot_ns)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
