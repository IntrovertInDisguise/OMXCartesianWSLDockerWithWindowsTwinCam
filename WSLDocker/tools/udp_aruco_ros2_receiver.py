#!/usr/bin/env python3
"""
udp_aruco_ros2_receiver.py

Run this inside WSL with ROS2 sourced.

Receives UDP JSON packets from the Windows ArUco sender and republishes:

    /single_arm_aruco/detections_json    std_msgs/String
    /single_arm_aruco/poses              geometry_msgs/PoseArray
    /single_arm_aruco/marker_order       std_msgs/String

The PoseArray order matches /single_arm_aruco/marker_order.

Examples:
    source /opt/ros/humble/setup.bash
    python3 udp_aruco_ros2_receiver.py

    # Or override port using ROS2 parameters
    python3 udp_aruco_ros2_receiver.py --ros-args -p udp_port:=5005

Check topics:
    ros2 topic echo /single_arm_aruco/detections_json
    ros2 topic echo /single_arm_aruco/poses
    ros2 topic echo /single_arm_aruco/marker_order
"""

from __future__ import annotations

import json
import select
import socket
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseArray


class UdpArucoRos2Receiver(Node):
    def __init__(self) -> None:
        super().__init__("udp_aruco_ros2_receiver")

        self.declare_parameter("bind_ip", "0.0.0.0")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("frame_id", "camera_color_optical_frame")

        self.declare_parameter("detections_topic", "/single_arm_aruco/detections_json")
        self.declare_parameter("poses_topic", "/single_arm_aruco/poses")
        self.declare_parameter("marker_order_topic", "/single_arm_aruco/marker_order")

        bind_ip = self.get_parameter("bind_ip").value
        udp_port = int(self.get_parameter("udp_port").value)

        detections_topic = self.get_parameter("detections_topic").value
        poses_topic = self.get_parameter("poses_topic").value
        marker_order_topic = self.get_parameter("marker_order_topic").value

        self.default_frame_id = self.get_parameter("frame_id").value

        self.json_pub = self.create_publisher(String, detections_topic, 10)
        self.pose_pub = self.create_publisher(PoseArray, poses_topic, 10)
        self.marker_order_pub = self.create_publisher(String, marker_order_topic, 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_ip, udp_port))
        self.sock.setblocking(False)

        self.rx_count = 0
        self.last_frame: Optional[int] = None

        # 100 Hz polling. UDP receive itself is non-blocking.
        self.timer = self.create_timer(0.01, self.poll_udp)

        self.get_logger().info(f"Listening for UDP ArUco packets on {bind_ip}:{udp_port}")
        self.get_logger().info(f"Publishing JSON:         {detections_topic}")
        self.get_logger().info(f"Publishing PoseArray:    {poses_topic}")
        self.get_logger().info(f"Publishing marker order: {marker_order_topic}")

    def poll_udp(self) -> None:
        """
        Drain all available UDP packets each timer cycle.
        This prevents buildup if the sender runs at high FPS.
        """
        while True:
            readable, _, _ = select.select([self.sock], [], [], 0.0)

            if not readable:
                break

            try:
                data, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception as exc:
                self.get_logger().warning(f"UDP receive error: {exc}")
                break

            try:
                text = data.decode("utf-8")
                packet = json.loads(text)
            except Exception as exc:
                self.get_logger().warning(f"Bad UDP JSON packet: {exc}")
                continue

            self.rx_count += 1
            self.publish_packet(packet)

            if self.rx_count <= 5 or self.rx_count % 100 == 0:
                markers = packet.get("markers", [])
                frame = packet.get("frame", None)
                ids = [m.get("id") for m in markers]
                self.get_logger().info(
                    f"RX {self.rx_count}: frame={frame}, markers={len(markers)}, ids={ids}, from={addr}"
                )

    def publish_packet(self, packet: Dict[str, Any]) -> None:
        # Publish original JSON unchanged
        json_msg = String()
        json_msg.data = json.dumps(packet)
        self.json_pub.publish(json_msg)

        # Convert available marker poses to PoseArray
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = packet.get("camera_frame", self.default_frame_id)

        marker_order: List[Dict[str, Any]] = []

        for marker in packet.get("markers", []):
            pos = marker.get("position_camera_m", None)
            quat = marker.get("orientation_xyzw_camera", None)

            # PoseArray only includes markers for which metric pose exists.
            # If Windows sender has no camera intrinsics, these fields will be absent.
            if pos is None or quat is None:
                continue

            if len(pos) != 3 or len(quat) != 4:
                continue

            pose = Pose()

            pose.position.x = float(pos[0])
            pose.position.y = float(pos[1])
            pose.position.z = float(pos[2])

            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            pose_array.poses.append(pose)

            marker_order.append(
                {
                    "id": marker.get("id"),
                    "label": marker.get("label"),
                    "pose_array_index": len(pose_array.poses) - 1,
                }
            )

        self.pose_pub.publish(pose_array)

        order_msg = String()
        order_msg.data = json.dumps(
            {
                "timestamp": packet.get("timestamp"),
                "frame": packet.get("frame"),
                "camera_frame": pose_array.header.frame_id,
                "markers": marker_order,
            }
        )
        self.marker_order_pub.publish(order_msg)

    def destroy_node(self) -> bool:
        try:
            self.sock.close()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = UdpArucoRos2Receiver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
