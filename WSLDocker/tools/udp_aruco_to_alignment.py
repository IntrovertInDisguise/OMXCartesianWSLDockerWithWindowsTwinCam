#!/usr/bin/env python3
"""udp_aruco_to_alignment.py

Image-free live ArUco alignment bridge for the single-arm harness.

Consumes the UDP-delivered ArUco detections on ``/single_arm_aruco/detections_json``
(and optionally ``/camera/color/camera_info`` for intrinsics) and republishes the
SAME single-arm alignment topics that ``hardware_harness_single_arm_v3.py``
subscribes to for live drift correction:

    /single_arm/aruco_alignment/valid                (Bool)
    /single_arm/aruco_alignment/center_y_error_m     (Float64)
    /single_arm/aruco_alignment/recommended_z_trim_m (Float64)

This lets the hardware run perform live alignment entirely over the UDP ArUco
route (Windows camera -> poses only, no raw image frames in the container),
mirroring the computation in ``tools/aruco_alignment_utils_single.py``
``detect_single_arm_aruco_alignment()`` so the harness sees identical values.

The metric error is computed with the pinhole relation used by the monitor:

    pixel_error   = cap_row - effective_target_row
    metric_error  = pixel_error * depth_m / fy
    z_trim        = -metric_error
    center_y_error = metric_error

where ``cap_row`` is the spring-cap marker's pixel y, ``depth_m`` is its camera
z (``position_camera_m[2]``), and ``fy`` comes from camera intrinsics (or a
parameter fallback). When the metal-platform marker is visible, the effective
target row is the midpoint between the cap row and the platform row; otherwise
it is ``image_height * target_row_fraction``.

Usage:
    python3 tools/udp_aruco_to_alignment.py
    python3 tools/udp_aruco_to_alignment.py --robot-marker-id 1 --platform-marker-id 0 \
        --fy 600.0 --image-height 480.0 --publish-period-s 0.5
    python3 tools/udp_aruco_to_alignment.py --use-camera-info false --fy 600.0 --image-height 480.0
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool, String
from std_msgs.msg import Float64
from sensor_msgs.msg import CameraInfo


try:
    from tools.aruco_alignment_utils_single import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
        ARUCO_ALIGNMENT_Z_TRIM_TOPIC,
    )
except ImportError:
    from aruco_alignment_utils_single import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
        ARUCO_ALIGNMENT_Z_TRIM_TOPIC,
    )


def _center_y_px(marker: Dict[str, Any]) -> Optional[float]:
    """Return the marker's pixel-row (y) center, from center_px or corners_px."""
    cp = marker.get("center_px")
    if isinstance(cp, (list, tuple)) and len(cp) >= 2:
        try:
            if all(math.isfinite(float(v)) for v in cp[:2]):
                return float(cp[1])
        except (TypeError, ValueError):
            pass
    corners = marker.get("corners_px")
    if isinstance(corners, (list, tuple)) and len(corners) >= 1:
        ys: List[float] = []
        for c in corners:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                try:
                    ys.append(float(c[1]))
                except (TypeError, ValueError):
                    pass
        if ys:
            return sum(ys) / len(ys)
    return None


class UdpArucoToAlignment(Node):
    def __init__(
        self,
        input_topic: str = "/single_arm_aruco/detections_json",
        valid_topic: str = ARUCO_ALIGNMENT_VALID_TOPIC,
        center_y_error_topic: str = ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        z_trim_topic: str = ARUCO_ALIGNMENT_Z_TRIM_TOPIC,
        robot_marker_id: int = 1,
        platform_marker_id: int = 0,
        target_row_fraction: float = 0.5,
        fy: float = float("nan"),
        image_height: float = float("nan"),
        camera_info_topic: str = "/camera/color/camera_info",
        use_camera_info: bool = True,
        publish_period_s: float = 0.5,
    ) -> None:
        super().__init__("udp_aruco_to_alignment")

        self.robot_marker_id = int(robot_marker_id)
        self.platform_marker_id = int(platform_marker_id) if platform_marker_id >= 0 else None
        self.target_row_fraction = float(target_row_fraction)
        self.fy = float(fy)
        self.image_height = float(image_height)
        self.use_camera_info = bool(use_camera_info)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.pub_valid = self.create_publisher(Bool, valid_topic, qos)
        self.pub_center_y_error = self.create_publisher(Float64, center_y_error_topic, qos)
        self.pub_z_trim = self.create_publisher(Float64, z_trim_topic, qos)

        self.latest_packet: Optional[Dict[str, Any]] = None
        self.create_subscription(String, input_topic, self._on_detections, qos)

        if self.use_camera_info:
            self.create_subscription(CameraInfo, camera_info_topic, self._on_camera_info, qos)

        self.create_timer(float(publish_period_s), self._publish_alignment)

        self.get_logger().info(
            f"UDP ArUco -> alignment bridge: input={input_topic}, robot_marker={self.robot_marker_id}, "
            f"platform_marker={self.platform_marker_id}, target_row_fraction={self.target_row_fraction}, "
            f"publish_period={publish_period_s}s, use_camera_info={self.use_camera_info}"
        )

    # ------------------------------------------------------------------
    def _on_detections(self, msg: String) -> None:
        try:
            self.latest_packet = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"Bad detections_json: {exc}")
            self.latest_packet = None

    def _on_camera_info(self, msg: CameraInfo) -> None:
        # K is row-major 3x3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        if len(msg.k) >= 6 and math.isfinite(msg.k[4]) and msg.k[4] > 0.0:
            self.fy = float(msg.k[4])
        if msg.height and msg.height > 0:
            self.image_height = float(msg.height)

    # ------------------------------------------------------------------
    def _compute_alignment(self, packet: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not packet:
            return None
        markers = packet.get("markers", [])
        if not isinstance(markers, list) or len(markers) == 0:
            return None
        if not (math.isfinite(self.fy) and self.fy > 0.0):
            return None
        if not (math.isfinite(self.image_height) and self.image_height > 0.0):
            return None

        by_id: Dict[int, Dict[str, Any]] = {}
        for m in markers:
            mid = m.get("id")
            if mid is None:
                continue
            by_id[int(mid)] = m

        # Metal-platform marker row (for midpoint z reference).
        platform_row_px: Optional[float] = None
        if self.platform_marker_id is not None and self.platform_marker_id in by_id:
            platform_row_px = _center_y_px(by_id[self.platform_marker_id])

        cap = by_id.get(self.robot_marker_id)
        if cap is None:
            return None
        cap_row = _center_y_px(cap)
        tvec = cap.get("position_camera_m")
        if cap_row is None or not isinstance(tvec, (list, tuple)) or len(tvec) < 3:
            return None
        depth = float(tvec[2])
        if not (math.isfinite(depth) and depth > 0.0):
            return None

        if platform_row_px is not None:
            effective_target_row = (cap_row + platform_row_px) / 2.0
        else:
            effective_target_row = self.image_height * self.target_row_fraction

        pixel_error = cap_row - effective_target_row
        metric_error = pixel_error * depth / self.fy
        trim_m = -metric_error

        return {
            "valid": True,
            "center_y_error_m": float(metric_error),
            "recommended_z_trim_m": float(trim_m),
        }

    def _publish_alignment(self) -> None:
        result = self._compute_alignment(self.latest_packet)
        valid = bool(result and result.get("valid"))

        valid_msg = Bool()
        valid_msg.data = valid
        self.pub_valid.publish(valid_msg)

        cy_msg = Float64()
        cy_msg.data = float(result["center_y_error_m"]) if result else float("nan")
        self.pub_center_y_error.publish(cy_msg)

        zt_msg = Float64()
        zt_msg.data = float(result["recommended_z_trim_m"]) if result else float("nan")
        self.pub_z_trim.publish(zt_msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Image-free UDP ArUco -> single-arm alignment bridge")
    p.add_argument("--input-topic", default="/single_arm_aruco/detections_json")
    p.add_argument("--valid-topic", default=ARUCO_ALIGNMENT_VALID_TOPIC)
    p.add_argument("--center-y-error-topic", default=ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC)
    p.add_argument("--z-trim-topic", default=ARUCO_ALIGNMENT_Z_TRIM_TOPIC)
    p.add_argument("--robot-marker-id", type=int, default=1)
    p.add_argument("--platform-marker-id", type=int, default=0,
                   help="Metal-platform marker id (-1 to disable midpoint reference)")
    p.add_argument("--target-row-fraction", type=float, default=0.5)
    p.add_argument("--fy", type=float, default=float("nan"),
                   help="Camera y focal length in pixels (used if camera_info unavailable)")
    p.add_argument("--image-height", type=float, default=float("nan"),
                   help="Image height in pixels (used if camera_info unavailable)")
    p.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    p.add_argument("--use-camera-info", action="store_true", default=True,
                   help="Subscribe to camera_info for fy/height (default on)")
    p.add_argument("--no-camera-info", dest="use_camera_info", action="store_false",
                   help="Disable camera_info subscription; rely on --fy/--image-height")
    p.add_argument("--publish-period-s", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = UdpArucoToAlignment(
        input_topic=args.input_topic,
        valid_topic=args.valid_topic,
        center_y_error_topic=args.center_y_error_topic,
        z_trim_topic=args.z_trim_topic,
        robot_marker_id=args.robot_marker_id,
        platform_marker_id=args.platform_marker_id,
        target_row_fraction=args.target_row_fraction,
        fy=args.fy,
        image_height=args.image_height,
        camera_info_topic=args.camera_info_topic,
        use_camera_info=args.use_camera_info,
        publish_period_s=args.publish_period_s,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
