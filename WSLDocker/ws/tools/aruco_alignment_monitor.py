#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64, Int32

try:
    from tools.ablation_config import (
        DEFAULT_ARUCO_DICTIONARY_NAME,
        DEFAULT_ARUCO_MARKER_LENGTH_M,
        DEFAULT_ARUCO_METAL_PLATFORM_MARKER_ID,
        DEFAULT_ARUCO_ROBOT1_MARKER_ID,
        DEFAULT_ARUCO_ROBOT2_MARKER_ID,
        DEFAULT_ARUCO_TARGET_ROW_FRACTION,
    )
except ImportError:
    from ablation_config import (
        DEFAULT_ARUCO_DICTIONARY_NAME,
        DEFAULT_ARUCO_MARKER_LENGTH_M,
        DEFAULT_ARUCO_METAL_PLATFORM_MARKER_ID,
        DEFAULT_ARUCO_ROBOT1_MARKER_ID,
        DEFAULT_ARUCO_ROBOT2_MARKER_ID,
        DEFAULT_ARUCO_TARGET_ROW_FRACTION,
    )

try:
    from tools.aruco_alignment_utils import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_MARKERS_DETECTED_TOPIC,
        ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROLL_DEG_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
        detect_aruco_alignment,
    )
except ImportError:
    from aruco_alignment_utils import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_MARKERS_DETECTED_TOPIC,
        ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROLL_DEG_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
        detect_aruco_alignment,
    )


DEFAULT_COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_COLOR_CAMERA_INFO_TOPIC = "/camera/color/camera_info"


class ArucoAlignmentMonitor(Node):
    def __init__(
        self,
        image_topic: str,
        camera_info_topic: str,
        robot1_marker_id: int,
        robot2_marker_id: int,
        marker_length_m: float,
        dictionary_name: str,
        target_row_fraction: float,
        metal_platform_marker_id: Optional[int],
        publish_period_s: float,
    ) -> None:
        super().__init__("aruco_alignment_monitor")
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.robot1_marker_id = int(robot1_marker_id)
        self.robot2_marker_id = int(robot2_marker_id)
        self.marker_length_m = float(marker_length_m)
        self.dictionary_name = dictionary_name
        self.target_row_fraction = float(target_row_fraction)
        self.metal_platform_marker_id = int(metal_platform_marker_id) if metal_platform_marker_id is not None else None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.color_image: Optional[Image] = None
        self.color_camera_info: Optional[CameraInfo] = None

        self.create_subscription(Image, self.image_topic, self._cb_color_image, qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._cb_color_camera_info, qos)

        self.pub_valid = self.create_publisher(Bool, ARUCO_ALIGNMENT_VALID_TOPIC, 1)
        self.pub_robot1_trim = self.create_publisher(Float64, ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC, 1)
        self.pub_robot2_trim = self.create_publisher(Float64, ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC, 1)
        self.pub_roll_deg = self.create_publisher(Float64, ARUCO_ALIGNMENT_ROLL_DEG_TOPIC, 1)
        self.pub_center_y_error = self.create_publisher(Float64, ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC, 1)
        self.pub_markers_detected = self.create_publisher(Int32, ARUCO_ALIGNMENT_MARKERS_DETECTED_TOPIC, 1)

        self.create_timer(float(publish_period_s), self._publish_alignment)

    def _cb_color_image(self, msg: Image) -> None:
        self.color_image = msg

    def _cb_color_camera_info(self, msg: CameraInfo) -> None:
        self.color_camera_info = msg

    def _publish_alignment(self) -> None:
        result = detect_aruco_alignment(
            self.color_image,
            self.color_camera_info,
            robot1_marker_id=self.robot1_marker_id,
            robot2_marker_id=self.robot2_marker_id,
            marker_length_m=self.marker_length_m,
            dictionary_name=self.dictionary_name,
            target_row_fraction=self.target_row_fraction,
            metal_platform_marker_id=self.metal_platform_marker_id,
        )

        valid_msg = Bool()
        valid_msg.data = bool(result.valid)
        self.pub_valid.publish(valid_msg)

        markers_msg = Int32()
        markers_msg.data = int(result.markers_detected)
        self.pub_markers_detected.publish(markers_msg)

        robot1_trim_msg = Float64()
        robot1_trim_msg.data = float(result.recommended_robot1_z_trim_m)
        self.pub_robot1_trim.publish(robot1_trim_msg)

        robot2_trim_msg = Float64()
        robot2_trim_msg.data = float(result.recommended_robot2_z_trim_m)
        self.pub_robot2_trim.publish(robot2_trim_msg)

        roll_msg = Float64()
        roll_msg.data = float(result.roll_deg)
        self.pub_roll_deg.publish(roll_msg)

        center_msg = Float64()
        center_msg.data = float(result.center_y_error_m)
        self.pub_center_y_error.publish(center_msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish ArUco-based alignment trims for the dual-arm spring rig")
    parser.add_argument("--image-topic", default=DEFAULT_COLOR_IMAGE_TOPIC,
                        help="Color image topic to monitor")
    parser.add_argument("--camera-info-topic", default=DEFAULT_COLOR_CAMERA_INFO_TOPIC,
                        help="CameraInfo topic paired with the color image")
    parser.add_argument("--robot1-marker-id", type=int, default=DEFAULT_ARUCO_ROBOT1_MARKER_ID,
                        help="ArUco marker ID associated with the robot1 side of the rig")
    parser.add_argument("--robot2-marker-id", type=int, default=DEFAULT_ARUCO_ROBOT2_MARKER_ID,
                        help="ArUco marker ID associated with the robot2 side of the rig")
    parser.add_argument("--marker-length-m", type=float, default=DEFAULT_ARUCO_MARKER_LENGTH_M,
                        help="Physical ArUco marker length in meters")
    parser.add_argument("--dictionary", default=DEFAULT_ARUCO_DICTIONARY_NAME,
                        help="OpenCV ArUco dictionary name, e.g. DICT_4X4_50")
    parser.add_argument("--target-row-fraction", type=float, default=DEFAULT_ARUCO_TARGET_ROW_FRACTION,
                        help="Desired normalized image row for the rig markers (0=top, 1=bottom); used as fallback when platform marker is not visible")
    parser.add_argument("--metal-platform-marker-id", type=int, default=DEFAULT_ARUCO_METAL_PLATFORM_MARKER_ID,
                        help="ArUco marker ID for the metal platform (used to compute midpoint z reference; set to -1 to disable)")
    parser.add_argument("--publish-period-s", type=float, default=0.5,
                        help="How often to recompute and publish alignment trims")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    metal_platform_id = args.metal_platform_marker_id if args.metal_platform_marker_id >= 0 else None
    node = ArucoAlignmentMonitor(
        image_topic=args.image_topic,
        camera_info_topic=args.camera_info_topic,
        robot1_marker_id=args.robot1_marker_id,
        robot2_marker_id=args.robot2_marker_id,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary,
        target_row_fraction=args.target_row_fraction,
        metal_platform_marker_id=metal_platform_id,
        publish_period_s=args.publish_period_s,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()