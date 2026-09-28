#!/usr/bin/env python3
"""
udp_aruco_to_calibration.py

Bridge the Windows→WSL/Docker UDP ArUco stream into the existing calibration
pipeline WITHOUT transporting heavy camera images.

The Windows sender (`windows_aruco_udp_sender.py`) already runs OpenCV/RealSense
ArUco detection and ships 6-DoF marker poses as JSON UDP packets. The receiver
(`udp_aruco_ros2_receiver.py`) republishes them as:
    /single_arm_aruco/detections_json   (std_msgs/String, raw JSON)
    /single_arm_aruco/poses             (geometry_msgs/PoseArray, camera frame)
    /single_arm_aruco/marker_order      (std_msgs/String)

This node consumes those topics and:
  1. Records every detection to a JSONL file in the SAME format that
     `tools/aruco_fit_extrinsics.py` consumes (one line per packet:
     {"ts_ns":..., "detections":[{"marker_id":int,"rvec":[x,y,z],"tvec":[x,y,z]}]}).
     That lets you FIT camera→world extrinsics offline, image-free.
  2. Re-publishes the raw JSON onto /camera_aruco/detections_json so the existing
     in-container consumers (compute_start_position_from_aruco.py,
     update_sdf_from_aruco_detection.py, single_arm_aruco_visual.py, etc.) work
     unchanged — they already read that topic.

No image topics are subscribed, so zero camera bandwidth is used.

Usage
-----
  # Default: record JSONL + relay to /camera_aruco/detections_json
  python3 tools/udp_aruco_to_calibration.py

  # Only record (no relay):
  python3 tools/udp_aruco_to_calibration.py --no-relay

  # Only relay (no JSONL):
  python3 tools/udp_aruco_to_calibration.py --no-record

  # Custom output dir / topic:
  python3 tools/udp_aruco_to_calibration.py \
      --output-dir logs/aruco_udp_run \
      --relay-topic /camera_aruco/detections_json

Downstream (image-free calibration)
-----------------------------------
  # 1) Fit camera->world extrinsics from the recorded JSONL:
  python3 tools/aruco_fit_extrinsics.py \
      --input-jsonl logs/aruco_udp_run/detections.jsonl \
      --marker-map tools/marker_map.json \
      --marker-length 0.038 \
      --output logs/aruco_udp_run/camera_extrinsics.json

  # 2) Compute start positions / update SDF using those extrinsics:
  python3 tools/compute_start_position_from_aruco.py \
      --camera-extrinsics logs/aruco_udp_run/camera_extrinsics.json
  python3 tools/update_sdf_from_aruco_detection.py \
      --camera-extrinsics logs/aruco_udp_run/camera_extrinsics.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class UdpArucoToCalibration(Node):
    def __init__(
        self,
        input_topic: str = "/single_arm_aruco/detections_json",
        relay_topic: str = "/camera_aruco/detections_json",
        output_dir: str = "logs/arx_udp_run",
        do_record: bool = True,
        do_relay: bool = True,
    ) -> None:
        super().__init__("udp_aruco_to_calibration")

        # Parameters are declared for visibility/override but the actual values
        # are taken from the constructor args (set before any file/socket opens).
        self.declare_parameter("input_topic", input_topic)
        self.declare_parameter("relay_topic", relay_topic)
        self.declare_parameter("output_dir", output_dir)
        self.declare_parameter("record", do_record)
        self.declare_parameter("relay", do_relay)

        self.do_record = do_record
        self.do_relay = do_relay

        self.json_pub: Optional[rclpy.publisher.Publisher] = None
        if self.do_relay:
            self.json_pub = self.create_publisher(String, relay_topic, 10)

        self.sub = self.create_subscription(
            String, input_topic, self._on_detections, 10
        )

        self.jsonl_path: Optional[str] = None
        self._jsonl_handle = None
        if self.do_record:
            os.makedirs(output_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.jsonl_path = os.path.join(output_dir, f"detections_{ts}.jsonl")
            self._jsonl_handle = open(self.jsonl_path, "w", encoding="utf-8")
            self.get_logger().info(f"Recording detections to {self.jsonl_path}")

        self.rx_count = 0
        self.get_logger().info(f"Subscribed to {input_topic}")
        if self.do_relay:
            self.get_logger().info(f"Relaying to {relay_topic}")
        self.get_logger().info(
            "Image-free calibration bridge active (no camera topics subscribed)."
        )

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _to_fit_record(packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a Windows-sender packet into the aruco_fit_extrinsics JSONL
        shape: {"ts_ns": int, "detections":[{"marker_id","rvec","tvec"}]}.

        The sender emits per marker:
            position_camera_m   -> tvec (meters, camera frame)
            rotation_vec        -> rvec (axis-angle, camera frame)
            orientation_xyzw_camera -> ignored here (extrinsics fit uses rvec/tvec)
        """
        markers = packet.get("markers", [])
        detections: List[Dict[str, Any]] = []
        for m in markers:
            rvec = m.get("rotation_vec")
            tvec = m.get("position_camera_m")
            if rvec is None or tvec is None:
                continue
            try:
                rvec = [float(x) for x in rvec]
                tvec = [float(x) for x in tvec]
            except (TypeError, ValueError):
                continue
            detections.append(
                {
                    "marker_id": int(m.get("id")),
                    "rvec": rvec,
                    "tvec": tvec,
                }
            )
        if not detections:
            return None
        ts = packet.get("timestamp")
        ts_ns = int(ts * 1e9) if isinstance(ts, (int, float)) else int(time.time() * 1e9)
        return {"ts_ns": ts_ns, "detections": detections}

    # ── callbacks ───────────────────────────────────────────────────────
    def _on_detections(self, msg: String) -> None:
        try:
            packet = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"Bad detections JSON: {exc}")
            return

        self.rx_count += 1

        if self.do_relay and self.json_pub is not None:
            # Re-publish the ORIGINAL packet unchanged so existing consumers
            # (which expect /camera_aruco/detections_json) work as-is.
            self.json_pub.publish(msg)

        if self.do_record and self._jsonl_handle is not None:
            rec = self._to_fit_record(packet)
            if rec is not None:
                # ROS receive time (same clock as the robot controller) is the
                # alignment anchor for offline time-sync with the harness
                # trial_start_ros_time_s. Recorded alongside the Windows sender
                # timestamp (ts_ns) so post-processing can choose the source.
                rec["ros_recv_time_s"] = self.get_clock().now().nanoseconds / 1e9
                self._jsonl_handle.write(json.dumps(rec) + "\n")
                # Flush periodically so data survives a Ctrl-C without close.
                if self.rx_count % 10 == 0:
                    self._jsonl_handle.flush()

        if self.rx_count <= 5 or self.rx_count % 100 == 0:
            n = len(packet.get("markers", []))
            self.get_logger().info(f"RX {self.rx_count}: markers={n}")

    def destroy_node(self) -> bool:
        if self._jsonl_handle is not None:
            try:
                self._jsonl_handle.flush()
                self._jsonl_handle.close()
            except Exception:
                pass
            self._jsonl_handle = None
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bridge UDP ArUco stream into calibration pipeline (no images).")
    p.add_argument("--input-topic", default="/single_arm_aruco/detections_json")
    p.add_argument("--relay-topic", default="/camera_aruco/detections_json")
    p.add_argument("--output-dir", default="logs/arx_udp_run")
    p.add_argument("--no-record", action="store_true", help="Do not write JSONL.")
    p.add_argument("--no-relay", action="store_true", help="Do not re-publish to relay topic.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = UdpArucoToCalibration(
        input_topic=args.input_topic,
        relay_topic=args.relay_topic,
        output_dir=args.output_dir,
        do_record=not args.no_record,
        do_relay=not args.no_relay,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
