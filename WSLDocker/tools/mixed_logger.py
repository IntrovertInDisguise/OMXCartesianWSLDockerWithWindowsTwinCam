#!/usr/bin/env python3
"""Simple mixed-run CSV logger for robot1 (hardware) and robot2 (Gazebo).

Writes two CSV files with timestamped joint_states (pos, vel, effort).
Usage: source ROS envs then run this node alongside the mixed coordinator.
"""
import os
import csv
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class MixedLogger(Node):
    def __init__(self, out_dir: str = "logs/mixed") -> None:
        super().__init__("mixed_logger")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(out_dir, ts)
        os.makedirs(self.out_dir, exist_ok=True)

        self.r1_path = os.path.join(self.out_dir, "robot1_joint_states.csv")
        self.r2_path = os.path.join(self.out_dir, "robot2_joint_states.csv")

        self.r1_file = open(self.r1_path, "w", newline="")
        self.r2_file = open(self.r2_path, "w", newline="")

        self.r1_writer = csv.writer(self.r1_file)
        self.r2_writer = csv.writer(self.r2_file)

        header = ["timestamp", "positions", "velocities", "efforts"]
        self.r1_writer.writerow(header)
        self.r2_writer.writerow(header)

        self.sub1 = self.create_subscription(JointState, "/robot1/joint_states", self.cb1, 50)
        self.sub2 = self.create_subscription(JointState, "/robot2/joint_states", self.cb2, 50)

        self.get_logger().info(f"MixedLogger writing to {self.out_dir}")

    def _row_from_msg(self, msg: JointState):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = ";".join([f"{p:.6f}" for p in msg.position]) if msg.position else ""
        vel = ";".join([f"{v:.6f}" for v in msg.velocity]) if msg.velocity else ""
        eff = ";".join([f"{e:.6f}" for e in msg.effort]) if msg.effort else ""
        return [f"{ts:.9f}", pos, vel, eff]

    def cb1(self, msg: JointState) -> None:
        try:
            self.r1_writer.writerow(self._row_from_msg(msg))
        except Exception as e:
            self.get_logger().error(f"Failed to write robot1 row: {e}")

    def cb2(self, msg: JointState) -> None:
        try:
            self.r2_writer.writerow(self._row_from_msg(msg))
        except Exception as e:
            self.get_logger().error(f"Failed to write robot2 row: {e}")

    def destroy_node(self):
        try:
            self.r1_file.flush(); self.r2_file.flush()
            self.r1_file.close(); self.r2_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = MixedLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
