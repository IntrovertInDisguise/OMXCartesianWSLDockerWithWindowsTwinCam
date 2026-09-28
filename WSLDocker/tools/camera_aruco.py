#!/usr/bin/env python3
"""
camera_aruco.py
────────────────
ROS 2 ArUco tracker for the dual-OMX spring rig.

Tracked layout
──────────────
- Two OMX robots face each other, bases at the centre of the extreme opposite
  sides of their respective green platforms, approximately 78 cm apart.
- Each green platform is a 0.30 m by 0.30 m square, 1 cm thick, resting on the
  ground. Robot bases sit at the outer far-edge centre of each green platform.
- A central metallic square platform (0.30 m by 0.30 m, solid 5 cm thick
  block) sits rigidly on the ground with its upper surface 5 cm above ground.
  It has a low-friction top surface and overlaps the inner edges of the two
  green platforms.
- The spring, with two white spring caps of height 6 cm each, sits on the
  raised metallic platform and is free to slide on that low-friction surface.
- The metallic platform is rigid and the spring caps are also rigid (they
  cannot sink into each other).
- The spring is attached only axially to the inner face centres of the caps
  and does not touch the metallic platform.
- Each square platform has its own ArUco marker:
  - green platform under robot 1 — at the bottom-far corner (robot base edge)
  - central metallic platform — at the bottom-left corner
  - green platform under robot 2 — at the bottom-far corner (robot base edge)
- Each spring cap has its own ArUco marker, located on top of the cap.
- A separate ground ArUco marker (world reference) adjoins the bottom edge
  centre of the right-side green platform, just off the platform on the ground.
- All ArUco markers in this setup are 38 mm by 38 mm (0.038 m), not 40 mm.
- A back wall / curtain is placed behind the scene (positive-y side) as a
  visual backdrop.

Camera position
───────────────
- The RealSense (or Gazebo user camera in simulation) is mounted front-top of
  the setup at approximately (x=0, y=−1.5, z=1.5) with yaw=π/2 facing the
  scene center and pitch=−0.8 looking downward at ~46°. Wide horizontal FOV
  (90°) captures the full 78 cm span plus the back wall backdrop. In Gazebo
  the initial viewpoint is set via the ``<gui>`` element in the world file.

Behavior
────────
- Subscribes to the color image and camera-info topics.
- Subscribes to the aligned depth image and aligned depth camera-info topics.
- Detects all configured ArUco markers in the color stream.
- Publishes each tracked marker pose in the camera frame.
- When the ground/world marker is visible, also publishes every tracked marker
  pose in the world-marker frame.
- Reports additional relative poses in the metal-platform workspace frame and
    the two green-platform manipulator frames when those reference markers are
    visible, so sliding platforms do not get conflated with the ground frame.
- Publishes per-marker visibility flags, total detected marker count, and a
  JSON summary that is convenient for logging or ad-hoc debugging.
- Optional `--log-marker-visibility-transitions` logging reports when
    configured marker IDs first go missing and when they recover.
- Can capture one RGBD frame on demand, verify that the aligned depth stream is
    decodable and non-corrupt, and print the current marker poses in the world
    frame with all 6 DoF (x, y, z, roll, pitch, yaw).

Default marker roles
────────────────────
The script provides a default six-marker rig layout:
- `ground_world`
- `green_platform_robot1`
- `metal_platform_center`
- `green_platform_robot2`
- `spring_cap_robot1`
- `spring_cap_robot2`

Default IDs are placeholders only. Override them with repeated `--marker`
arguments if your printed tags use different IDs.
`--world-marker-id` retargets the ground/world reference tag while preserving
the six-marker default layout.

Verified hardware mapping
─────────────────────────
The currently verified physical tag assignment for this rig is:
- `ground_world:6`
- `green_platform_robot1:2`
- `metal_platform_center:0`
- `green_platform_robot2:4`
- `spring_cap_robot1:1`
- `spring_cap_robot2:3`

Example
───────
  source /opt/ros/humble/setup.bash
  source /workspaces/omx_ros2/ws/install/setup.bash
    /usr/bin/python3 tools/camera_aruco.py \
        --world-marker-id 6 \
        --marker ground_world:6 \
        --marker green_platform_robot1:2 \
        --marker metal_platform_center:0 \
        --marker green_platform_robot2:4 \
        --marker spring_cap_robot1:1 \
        --marker spring_cap_robot2:3

    source /opt/ros/humble/setup.bash && source /workspaces/omx_ros2/ws/install/setup.bash && ros2 topic echo --once /camera_aruco/detections_json --field data
    source /opt/ros/humble/setup.bash && source /workspaces/omx_ros2/ws/install/setup.bash && ros2 run image_tools showimage --ros-args -r image:=/spring/visual_bowing/overlay
    /usr/bin/python3 tools/camera_aruco.py --log-marker-visibility-transitions
    /usr/bin/python3 tools/camera_aruco.py --publish-period-s 0.1 --fast
    /usr/bin/python3 tools/camera_aruco.py --publish-period-s 0.1 --fast --processing-resolution 640x480
    /usr/bin/python3 tools/camera_aruco.py --fast --reference-marker-refresh-frames 100
    /usr/bin/python3 tools/camera_aruco.py --print-once --capture-timeout-s 5.0
    
    Rate test:
    ros2 topic hz /camera_aruco/detections_json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64MultiArray, Int32, String

try:
    from tools.ablation_config import DEFAULT_ARUCO_DICTIONARY_NAME
except ImportError:
    from ablation_config import DEFAULT_ARUCO_DICTIONARY_NAME

try:
    from tools import aruco_alignment_utils as aruco_utils
except ImportError:
    import aruco_alignment_utils as aruco_utils

try:
    import cv2
except ImportError:
    cv2 = getattr(aruco_utils, "cv2", None)

try:
    from tools.depth_frame_utils import camera_info_to_dict, decode_sensor_image
except ImportError:
    from depth_frame_utils import camera_info_to_dict, decode_sensor_image


DEFAULT_COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_COLOR_CAMERA_INFO_TOPIC = "/camera/color/camera_info"
DEFAULT_DEPTH_IMAGE_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEFAULT_DEPTH_CAMERA_INFO_TOPIC = "/camera/aligned_depth_to_color/camera_info"
DEFAULT_MARKER_LENGTH_M = 0.038
DEFAULT_PUBLISH_PERIOD_S = 0.25
DEFAULT_CAPTURE_TIMEOUT_S = 5.0
DEFAULT_SUMMARY_TOPIC = "/camera_aruco/detections_json"
DEFAULT_COUNT_TOPIC = "/camera_aruco/markers_detected"
DEFAULT_WORLD_VISIBLE_TOPIC = "/camera_aruco/world_visible"
DEFAULT_WORLD_MARKER_ID = 0
DEFAULT_VISUAL_BOWING_TOPIC = "/spring/visual_bowing"
DEFAULT_VISUAL_BOWING_OVERLAY_TOPIC = "/spring/visual_bowing/overlay"
DEFAULT_OVERLAY_STALE_GRACE_FRAMES = 2
ARUCO_DETECTOR_PROFILE_DEFAULT = "default"
ARUCO_DETECTOR_PROFILE_CHOICES = ("default", "fast")
WORKSPACE_MARKER_LABEL = "metal_platform_center"
MANIPULATOR_ROBOT1_MARKER_LABEL = "green_platform_robot1"
MANIPULATOR_ROBOT2_MARKER_LABEL = "green_platform_robot2"
SPRING_CAP_MARKER_LABELS = frozenset({"spring_cap_robot1", "spring_cap_robot2"})
STATIC_MARKER_OVERLAY_STALE_GRACE_FRAMES = 3
STATIC_OVERLAY_MARKER_LABELS = frozenset(
    {
        "ground_world",
        MANIPULATOR_ROBOT1_MARKER_LABEL,
        WORKSPACE_MARKER_LABEL,
        MANIPULATOR_ROBOT2_MARKER_LABEL,
    }
)

LEFT_CAP_ID = 4
RIGHT_CAP_ID = 5
ROI_HALF_WIDTH_PX = 40
ROI_END_MARGIN_PX = 16
BASELINE_FRAMES = 20
MIN_SEGMENT_AREA = 24
MORPH_KERNEL_SIZE = 3
SPRING_BOX_AXIAL_MARGIN_PX = 4.0
SPRING_BOX_LATERAL_MARGIN_PX = 2.0
BENT_CENTER_OFFSET_MIN_PX = 2.0
BENT_CENTER_OFFSET_RATIO = 0.35
CAP_TRACK_MIN_ROI_HALF_SIZE_PX = 56.0
CAP_TRACK_ROI_HALF_SIZE_SCALE = 4.0
CAP_TRACK_ROI_EXTRA_MARGIN_PX = 24.0


@dataclass(frozen=True)
class MarkerSpec:
    label: str
    marker_id: int


@dataclass
class MarkerDetection:
    spec: MarkerSpec
    center_px: Tuple[float, float]
    rvec: np.ndarray
    tvec: np.ndarray
    corners_px: Optional[np.ndarray] = None

    def transform_camera_marker(self) -> np.ndarray:
        rotation_matrix, _ = aruco_utils.cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64).reshape(3, 1))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = np.asarray(self.tvec, dtype=np.float64).reshape(3)
        return transform


DEFAULT_MARKER_SPECS: List[MarkerSpec] = [
    MarkerSpec("ground_world", DEFAULT_WORLD_MARKER_ID),
    MarkerSpec("green_platform_robot1", 1),
    MarkerSpec("metal_platform_center", 2),
    MarkerSpec("green_platform_robot2", 3),
    MarkerSpec("spring_cap_robot1", 4),
    MarkerSpec("spring_cap_robot2", 5),
]


def get_marker_center(corners) -> Tuple[float, float]:
    corners_array = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    center_xy = np.mean(corners_array, axis=0)
    return float(center_xy[0]), float(center_xy[1])


def _sanitize_label(label: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", label.strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "marker"


def _parse_marker_arg(value: str) -> MarkerSpec:
    if ":" not in value:
        raise argparse.ArgumentTypeError("marker must use LABEL:ID format")
    label, marker_id = value.split(":", 1)
    label = _sanitize_label(label)
    if not label:
        raise argparse.ArgumentTypeError("marker label cannot be empty")
    try:
        marker_id_int = int(marker_id)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"marker id must be an integer: {marker_id}") from exc
    return MarkerSpec(label=label, marker_id=marker_id_int)


def _parse_processing_resolution(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", str(value))
    if match is None:
        raise argparse.ArgumentTypeError("processing resolution must use WIDTHxHEIGHT format, for example 640x480")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("processing resolution must be positive")
    return (width, height)


def _resolve_marker_specs(marker_args: Optional[List[MarkerSpec]], world_marker_id: int) -> List[MarkerSpec]:
    specs = list(marker_args) if marker_args else list(DEFAULT_MARKER_SPECS)

    labels_seen = set()
    ids_seen = set()
    resolved: List[MarkerSpec] = []
    ground_world_index: Optional[int] = None
    for spec in specs:
        label = _sanitize_label(spec.label)
        if label in labels_seen:
            raise ValueError(f"duplicate marker label: {label}")
        if spec.marker_id in ids_seen:
            raise ValueError(f"duplicate marker id: {spec.marker_id}")
        labels_seen.add(label)
        ids_seen.add(spec.marker_id)
        if label == "ground_world":
            ground_world_index = len(resolved)
        resolved.append(MarkerSpec(label=label, marker_id=int(spec.marker_id)))

    if world_marker_id in ids_seen:
        return resolved

    if ground_world_index is not None:
        resolved[ground_world_index] = MarkerSpec(
            label=resolved[ground_world_index].label,
            marker_id=int(world_marker_id),
        )
        return resolved

    resolved.insert(0, MarkerSpec(label="ground_world", marker_id=int(world_marker_id)))

    return resolved


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / scale
        qx = (rotation[2, 1] - rotation[1, 2]) * scale
        qy = (rotation[0, 2] - rotation[2, 0]) * scale
        qz = (rotation[1, 0] - rotation[0, 1]) * scale
        return qx, qy, qz, qw

    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0, 1] + rotation[1, 0]) / scale
        qz = (rotation[0, 2] + rotation[2, 0]) / scale
        return qx, qy, qz, qw

    if rotation[1, 1] > rotation[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / scale
        qx = (rotation[0, 1] + rotation[1, 0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1, 2] + rotation[2, 1]) / scale
        return qx, qy, qz, qw

    scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
    qw = (rotation[1, 0] - rotation[0, 1]) / scale
    qx = (rotation[0, 2] + rotation[2, 0]) / scale
    qy = (rotation[1, 2] + rotation[2, 1]) / scale
    qz = 0.25 * scale
    return qx, qy, qz, qw


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> Tuple[float, float, float]:
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0

    return tuple(float(np.degrees(angle)) for angle in (roll, pitch, yaw))


def _wrap_angle_rad(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3:
        if array.shape[2] > 3:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = _normalize_to_uint8(array)
        gray = np.tensordot(array.astype(np.float32), np.array([0.299, 0.587, 0.114], dtype=np.float32), axes=([-1], [0]))
        return np.clip(gray, 0.0, 255.0).astype(np.uint8)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    return _normalize_to_uint8(array)


def _to_rgb_u8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        mono = _to_gray_u8(array)
        return np.repeat(mono[:, :, None], 3, axis=2)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = _normalize_to_uint8(array)
    return np.ascontiguousarray(array)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array_f = np.asarray(array, dtype=np.float32)
    finite = array_f[np.isfinite(array_f)]
    if finite.size == 0:
        return np.zeros(array_f.shape, dtype=np.uint8)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo + 1e-12:
        return np.zeros(array_f.shape, dtype=np.uint8)
    scaled = (array_f - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _resize_array(array: np.ndarray, target_width: int, target_height: int, *, interpolation: str = "area") -> np.ndarray:
    target_width = int(target_width)
    target_height = int(target_height)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target resize dimensions must be positive")

    array_np = np.asarray(array)
    if array_np.shape[1] == target_width and array_np.shape[0] == target_height:
        return np.ascontiguousarray(array_np)

    if cv2 is not None and hasattr(cv2, "resize"):
        if interpolation == "nearest":
            interpolation_flag = getattr(cv2, "INTER_NEAREST", 0)
        else:
            interpolation_flag = getattr(cv2, "INTER_AREA", getattr(cv2, "INTER_LINEAR", 1))
        return np.ascontiguousarray(cv2.resize(array_np, (target_width, target_height), interpolation=interpolation_flag))

    y_indices = np.clip(np.round(np.linspace(0, array_np.shape[0] - 1, target_height)).astype(int), 0, array_np.shape[0] - 1)
    x_indices = np.clip(np.round(np.linspace(0, array_np.shape[1] - 1, target_width)).astype(int), 0, array_np.shape[1] - 1)
    if array_np.ndim == 2:
        return np.ascontiguousarray(array_np[np.ix_(y_indices, x_indices)])
    return np.ascontiguousarray(array_np[y_indices][:, x_indices, ...])


def _scale_camera_matrix(
    camera_matrix: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    scaled = np.asarray(camera_matrix, dtype=np.float64).copy()
    sx = float(target_width) / float(source_width)
    sy = float(target_height) / float(source_height)
    scaled[0, 0] *= sx
    scaled[0, 2] *= sx
    scaled[1, 1] *= sy
    scaled[1, 2] *= sy
    return scaled


def _video_writer_candidates(output_path: str) -> List[Tuple[str, str]]:
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".avi":
        return [(output_path, "MJPG"), (base + ".mp4", "mp4v")]
    if ext.lower() == ".mp4":
        return [(output_path, "mp4v"), (base + ".avi", "MJPG")]
    return [(output_path + ".mp4", "mp4v"), (output_path + ".avi", "MJPG")]


def _camera_info_json_path_for_video(video_path: str) -> str:
    return f"{video_path}.camera_info.json"


def _default_post_process_jsonl_path(video_path: str) -> str:
    base, _ext = os.path.splitext(video_path)
    return base + ".aruco.jsonl"


def _default_post_process_csv_path(video_path: str) -> str:
    base, _ext = os.path.splitext(video_path)
    return base + ".aruco.csv"


def _default_post_process_overlay_video_path(video_path: str) -> str:
    base, _ext = os.path.splitext(video_path)
    return base + ".aruco_overlay.mp4"


def _csv_pose_fieldnames(prefix: str) -> List[str]:
    return [
        f"{prefix}_tx_m",
        f"{prefix}_ty_m",
        f"{prefix}_tz_m",
        f"{prefix}_qx",
        f"{prefix}_qy",
        f"{prefix}_qz",
        f"{prefix}_qw",
    ]


def _summary_csv_fieldnames(marker_specs: List[MarkerSpec]) -> List[str]:
    fieldnames = [
        "record_type",
        "source",
        "sequence",
        "video_path",
        "frame_index",
        "video_timestamp_s",
        "wall_time_unix_ns",
        "color_image_stamp_ns",
        "color_camera_info_stamp_ns",
        "depth_image_stamp_ns",
        "depth_camera_info_stamp_ns",
        "valid",
        "reason",
        "camera_frame_id",
        "world_frame_id",
        "world_marker_id",
        "world_label",
        "image_width",
        "image_height",
        "markers_detected",
        "world_marker_visible",
        "coordinate_frame_camera",
        "coordinate_frame_world",
        "coordinate_frame_workspace",
        "coordinate_frame_manipulator_robot1",
        "coordinate_frame_manipulator_robot2",
        "depth_available",
        "depth_camera_info_available",
        "depth_valid",
        "depth_reason",
        "depth_encoding",
        "visual_valid_detection",
        "visual_reason",
        "visual_L_px",
        "visual_axis_angle_rad",
        "visual_theta_cap_rad",
        "visual_theta_cap_workspace_rad",
        "visual_cap_span_workspace_m",
        "visual_cap_axis_workspace_rad",
        "visual_w_vis_raw_px",
        "visual_w_vis_delta_px",
        "visual_bent",
        "visual_bend_score",
        "visual_spring_box_center_offset_px",
        "visual_baseline_px",
        "visual_baseline_frames_collected",
        "visual_left_cap_id",
        "visual_right_cap_id",
    ]
    for spec in marker_specs:
        prefix = spec.label
        fieldnames.extend(
            [
                f"{prefix}_visible",
                f"{prefix}_marker_id",
                f"{prefix}_center_px_x",
                f"{prefix}_center_px_y",
            ]
        )
        for pose_prefix in (
            f"{prefix}_pose_camera",
            f"{prefix}_pose_world",
            f"{prefix}_pose_workspace",
            f"{prefix}_pose_manipulator_robot1",
            f"{prefix}_pose_manipulator_robot2",
        ):
            fieldnames.extend(_csv_pose_fieldnames(pose_prefix))
        fieldnames.extend(
            [
                f"{prefix}_depth_center_m",
                f"{prefix}_depth_minus_camera_z_m",
            ]
        )
    return fieldnames


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(value) else float(value)
    return value


def _set_pose_csv_values(row: Dict[str, Any], prefix: str, pose: Optional[Dict[str, Any]]) -> None:
    if pose is None:
        return
    translation = list(pose.get("translation_m", []))
    quaternion = list(pose.get("quaternion_xyzw", []))
    if len(translation) >= 3:
        row[f"{prefix}_tx_m"] = _csv_scalar(translation[0])
        row[f"{prefix}_ty_m"] = _csv_scalar(translation[1])
        row[f"{prefix}_tz_m"] = _csv_scalar(translation[2])
    if len(quaternion) >= 4:
        row[f"{prefix}_qx"] = _csv_scalar(quaternion[0])
        row[f"{prefix}_qy"] = _csv_scalar(quaternion[1])
        row[f"{prefix}_qz"] = _csv_scalar(quaternion[2])
        row[f"{prefix}_qw"] = _csv_scalar(quaternion[3])


def _summary_csv_row(
    summary: Dict[str, Any],
    marker_specs: List[MarkerSpec],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {fieldname: "" for fieldname in _summary_csv_fieldnames(marker_specs)}
    for key, value in (metadata or {}).items():
        if key in row:
            row[key] = _csv_scalar(value)

    coordinate_frames = summary.get("coordinate_frames") or {}
    depth_stream = summary.get("depth_stream") or {}
    visual_bowing = summary.get("visual_bowing") or {}
    observed_labels = {
        str(label)
        for label in summary.get("observed_labels", [])
        if label is not None
    }
    row.update(
        {
            "valid": _csv_scalar(summary.get("valid")),
            "reason": _csv_scalar(summary.get("reason")),
            "camera_frame_id": _csv_scalar(summary.get("camera_frame_id")),
            "world_frame_id": _csv_scalar(summary.get("world_frame_id")),
            "world_marker_id": _csv_scalar(summary.get("world_marker_id")),
            "world_label": _csv_scalar(summary.get("world_label")),
            "image_width": _csv_scalar(summary.get("image_width")),
            "image_height": _csv_scalar(summary.get("image_height")),
            "markers_detected": _csv_scalar(summary.get("markers_detected")),
            "world_marker_visible": _csv_scalar(summary.get("world_marker_visible")),
            "coordinate_frame_camera": _csv_scalar((coordinate_frames.get("camera") or {}).get("frame_id")),
            "coordinate_frame_world": _csv_scalar((coordinate_frames.get("world") or {}).get("frame_id")),
            "coordinate_frame_workspace": _csv_scalar((coordinate_frames.get("workspace") or {}).get("frame_id")),
            "coordinate_frame_manipulator_robot1": _csv_scalar((coordinate_frames.get("manipulator_robot1") or {}).get("frame_id")),
            "coordinate_frame_manipulator_robot2": _csv_scalar((coordinate_frames.get("manipulator_robot2") or {}).get("frame_id")),
            "depth_available": _csv_scalar(depth_stream.get("available")),
            "depth_camera_info_available": _csv_scalar(depth_stream.get("camera_info_available")),
            "depth_valid": _csv_scalar(depth_stream.get("valid")),
            "depth_reason": _csv_scalar(depth_stream.get("reason")),
            "depth_encoding": _csv_scalar(depth_stream.get("encoding")),
            "visual_valid_detection": _csv_scalar(visual_bowing.get("valid_detection")),
            "visual_reason": _csv_scalar(visual_bowing.get("reason")),
            "visual_L_px": _csv_scalar(visual_bowing.get("L_px")),
            "visual_axis_angle_rad": _csv_scalar(visual_bowing.get("axis_angle_rad")),
            "visual_theta_cap_rad": _csv_scalar(visual_bowing.get("theta_cap_rad")),
            "visual_theta_cap_workspace_rad": _csv_scalar(visual_bowing.get("theta_cap_workspace_rad")),
            "visual_cap_span_workspace_m": _csv_scalar(visual_bowing.get("cap_span_workspace_m")),
            "visual_cap_axis_workspace_rad": _csv_scalar(visual_bowing.get("cap_axis_workspace_rad")),
            "visual_w_vis_raw_px": _csv_scalar(visual_bowing.get("w_vis_raw_px")),
            "visual_w_vis_delta_px": _csv_scalar(visual_bowing.get("w_vis_delta_px")),
            "visual_bent": _csv_scalar(visual_bowing.get("bent")),
            "visual_bend_score": _csv_scalar(visual_bowing.get("bend_score")),
            "visual_spring_box_center_offset_px": _csv_scalar(visual_bowing.get("spring_box_center_offset_px")),
            "visual_baseline_px": _csv_scalar(visual_bowing.get("baseline_px")),
            "visual_baseline_frames_collected": _csv_scalar(visual_bowing.get("baseline_frames_collected")),
            "visual_left_cap_id": _csv_scalar(visual_bowing.get("left_cap_id")),
            "visual_right_cap_id": _csv_scalar(visual_bowing.get("right_cap_id")),
        }
    )

    detections_by_label = {
        str(entry.get("label")): entry
        for entry in summary.get("detections", [])
        if isinstance(entry, dict) and entry.get("label") is not None
    }
    for spec in marker_specs:
        prefix = spec.label
        detection = detections_by_label.get(spec.label)
        row[f"{prefix}_visible"] = 1 if (detection is not None or spec.label in observed_labels) else 0
        row[f"{prefix}_marker_id"] = int(spec.marker_id)
        if detection is None:
            continue

        center_px = list(detection.get("center_px", []))
        if len(center_px) >= 2:
            row[f"{prefix}_center_px_x"] = _csv_scalar(center_px[0])
            row[f"{prefix}_center_px_y"] = _csv_scalar(center_px[1])
        _set_pose_csv_values(row, f"{prefix}_pose_camera", detection.get("pose_camera"))
        _set_pose_csv_values(row, f"{prefix}_pose_world", detection.get("pose_world"))
        _set_pose_csv_values(row, f"{prefix}_pose_workspace", detection.get("pose_workspace"))
        _set_pose_csv_values(row, f"{prefix}_pose_manipulator_robot1", detection.get("pose_manipulator_robot1"))
        _set_pose_csv_values(row, f"{prefix}_pose_manipulator_robot2", detection.get("pose_manipulator_robot2"))
        row[f"{prefix}_depth_center_m"] = _csv_scalar(detection.get("depth_center_m"))
        row[f"{prefix}_depth_minus_camera_z_m"] = _csv_scalar(detection.get("depth_minus_camera_z_m"))
    return row


class _SummaryCsvLogger:
    def __init__(self, csv_path: Optional[str], marker_specs: List[MarkerSpec]) -> None:
        self.csv_path = str(csv_path).strip() if csv_path else None
        self._marker_specs = list(marker_specs)
        self._fieldnames = _summary_csv_fieldnames(self._marker_specs) if self.csv_path else []
        self._header_written = False
        if self.csv_path:
            csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
            self._header_written = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0

    def append(self, summary: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.csv_path:
            return
        row = _summary_csv_row(summary, marker_specs=self._marker_specs, metadata=metadata)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


def _camera_info_from_dict(payload: Dict[str, Any]):
    frame_id = str(payload.get("frame_id", ""))
    stamp_ns = int(payload.get("stamp_ns", 0) or 0)
    return types.SimpleNamespace(
        header=types.SimpleNamespace(
            frame_id=frame_id,
            stamp=types.SimpleNamespace(
                sec=stamp_ns // 1_000_000_000,
                nanosec=stamp_ns % 1_000_000_000,
            ),
        ),
        width=int(payload.get("width", 0) or 0),
        height=int(payload.get("height", 0) or 0),
        distortion_model=str(payload.get("distortion_model", "")),
        d=list(payload.get("d", [])),
        k=list(payload.get("k", [])),
        r=list(payload.get("r", [])),
        p=list(payload.get("p", [])),
        binning_x=int(payload.get("binning_x", 0) or 0),
        binning_y=int(payload.get("binning_y", 0) or 0),
    )


def _load_camera_info_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as handle:
        return _camera_info_from_dict(json.load(handle))


def _write_json_file(file_path: str, payload: Dict[str, Any]) -> None:
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _make_rgb8_image_message(frame_rgb: np.ndarray, frame_id: str, stamp_ns: int):
    frame = _to_rgb_u8(frame_rgb)
    return types.SimpleNamespace(
        header=types.SimpleNamespace(
            frame_id=str(frame_id),
            stamp=types.SimpleNamespace(
                sec=int(stamp_ns) // 1_000_000_000,
                nanosec=int(stamp_ns) % 1_000_000_000,
            ),
        ),
        width=int(frame.shape[1]),
        height=int(frame.shape[0]),
        encoding="rgb8",
        is_bigendian=0,
        step=int(frame.shape[1] * 3),
        data=np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
    )


def _scale_camera_info_message(camera_info_msg: Any, target_width: int, target_height: int):
    payload = camera_info_to_dict(camera_info_msg)
    source_width = int(payload.get("width", 0) or 0)
    source_height = int(payload.get("height", 0) or 0)
    payload["width"] = int(target_width)
    payload["height"] = int(target_height)

    k = list(payload.get("k", []))
    if source_width > 0 and source_height > 0 and len(k) == 9:
        scaled_k = _scale_camera_matrix(
            np.asarray(k, dtype=np.float64).reshape(3, 3),
            source_width,
            source_height,
            int(target_width),
            int(target_height),
        )
        payload["k"] = scaled_k.reshape(-1).tolist()

    p = list(payload.get("p", []))
    if source_width > 0 and source_height > 0 and len(p) == 12:
        projection = np.asarray(p, dtype=np.float64).reshape(3, 4)
        sx = float(target_width) / float(source_width)
        sy = float(target_height) / float(source_height)
        projection[0, 0] *= sx
        projection[0, 2] *= sx
        projection[0, 3] *= sx
        projection[1, 1] *= sy
        projection[1, 2] *= sy
        projection[1, 3] *= sy
        payload["p"] = projection.reshape(-1).tolist()

    return _camera_info_from_dict(payload)


class _LazyRgbVideoWriter:
    def __init__(self, output_path: str, fps: float) -> None:
        self.requested_output_path = str(output_path)
        self.fps = float(fps)
        self.output_path = str(output_path)
        self.codec_name: Optional[str] = None
        self.writer = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.frame_count = 0

    def _ensure_writer(self, frame_width: int, frame_height: int) -> None:
        if self.writer is not None:
            return
        if cv2 is None or not hasattr(cv2, "VideoWriter") or not hasattr(cv2, "VideoWriter_fourcc"):
            raise RuntimeError("OpenCV video I/O is unavailable in the active interpreter")

        for candidate_path, codec_name in _video_writer_candidates(self.requested_output_path):
            candidate_dir = os.path.dirname(os.path.abspath(candidate_path))
            if candidate_dir:
                os.makedirs(candidate_dir, exist_ok=True)
            writer = cv2.VideoWriter(
                candidate_path,
                cv2.VideoWriter_fourcc(*codec_name),
                float(self.fps),
                (int(frame_width), int(frame_height)),
            )
            if writer is None or not writer.isOpened():
                if writer is not None:
                    writer.release()
                continue
            self.writer = writer
            self.output_path = candidate_path
            self.codec_name = codec_name
            self.frame_width = int(frame_width)
            self.frame_height = int(frame_height)
            return

        raise RuntimeError("Unable to open OpenCV VideoWriter for either MP4 or AVI output")

    def write_rgb_frame(self, frame_rgb: np.ndarray) -> None:
        frame = _to_rgb_u8(frame_rgb)
        frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
        self._ensure_writer(frame_width, frame_height)
        if self.frame_width != frame_width or self.frame_height != frame_height:
            raise RuntimeError(
                f"Video frame size changed from {self.frame_width}x{self.frame_height} to {frame_width}x{frame_height}"
            )
        self.writer.write(np.ascontiguousarray(frame[:, :, ::-1], dtype=np.uint8))
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def build_oriented_roi_mask(
    p_left,
    p_right,
    image_shape,
    half_width,
    end_margin,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    height, width = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)

    left = np.asarray(p_left, dtype=np.float64).reshape(2)
    right = np.asarray(p_right, dtype=np.float64).reshape(2)
    axis = right - left
    axis_length = float(np.linalg.norm(axis))
    if not np.isfinite(axis_length) or axis_length <= 1e-6:
        return mask, None

    axis_unit = axis / axis_length
    normal_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
    start = left + axis_unit * float(end_margin)
    end = right - axis_unit * float(end_margin)
    trimmed_length = float(np.linalg.norm(end - start))
    if trimmed_length <= 1e-6:
        return mask, None

    polygon = np.array(
        [
            start + normal_unit * float(half_width),
            end + normal_unit * float(half_width),
            end - normal_unit * float(half_width),
            start - normal_unit * float(half_width),
        ],
        dtype=np.float64,
    )

    yy, xx = np.indices((height, width), dtype=np.float64)
    rel_x = xx + 0.5 - start[0]
    rel_y = yy + 0.5 - start[1]
    proj = rel_x * axis_unit[0] + rel_y * axis_unit[1]
    perp = rel_x * normal_unit[0] + rel_y * normal_unit[1]
    inside = (proj >= 0.0) & (proj <= trimmed_length) & (np.abs(perp) <= float(half_width))
    mask[inside] = 255
    return mask, polygon


def segment_spring_in_roi(
    frame,
    roi_mask,
    min_segment_area: int = MIN_SEGMENT_AREA,
    morph_kernel_size: int = MORPH_KERNEL_SIZE,
) -> np.ndarray:
    mask = np.asarray(roi_mask, dtype=np.uint8)
    gray = _to_gray_u8(np.asarray(frame))
    if gray.shape[:2] != mask.shape[:2] or not np.any(mask > 0):
        return np.zeros(mask.shape[:2], dtype=np.uint8)

    roi_values = gray[mask > 0]
    if roi_values.size == 0:
        return np.zeros(mask.shape[:2], dtype=np.uint8)

    deviation = np.abs(gray.astype(np.float32) - float(np.median(roi_values)))
    if cv2 is not None and all(hasattr(cv2, name) for name in ("GaussianBlur", "Sobel", "magnitude")):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
    else:
        grad_y, grad_x = np.gradient(gray.astype(np.float32))
        gradient = np.hypot(grad_x, grad_y)

    combined = np.maximum(deviation, gradient)
    combined[mask == 0] = 0.0
    masked_values = combined[mask > 0]
    if masked_values.size == 0 or float(np.max(masked_values)) <= 0.0:
        return np.zeros(mask.shape[:2], dtype=np.uint8)

    threshold_value = max(1.0, float(np.percentile(masked_values, 75.0)))
    binary = np.zeros(mask.shape[:2], dtype=np.uint8)
    binary[(combined >= threshold_value) & (mask > 0)] = 255

    kernel_size = max(1, int(morph_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if cv2 is not None and all(hasattr(cv2, name) for name in ("morphologyEx", "connectedComponentsWithStats", "MORPH_CLOSE", "MORPH_OPEN", "CC_STAT_AREA")):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        filtered = np.zeros_like(binary)
        for label_index in range(1, int(num_labels)):
            if int(stats[label_index, cv2.CC_STAT_AREA]) >= int(min_segment_area):
                filtered[labels == label_index] = 255
        binary = filtered
    elif np.count_nonzero(binary) < int(min_segment_area):
        binary.fill(0)

    binary[mask == 0] = 0
    return binary


def point_line_distances(points, p_left, p_right) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    left = np.asarray(p_left, dtype=np.float64).reshape(2)
    right = np.asarray(p_right, dtype=np.float64).reshape(2)
    axis = right - left
    axis_length = float(np.linalg.norm(axis))
    if not np.isfinite(axis_length) or axis_length <= 1e-6 or points_array.size == 0:
        return np.asarray([], dtype=np.float64)
    rel = points_array - left
    return np.abs(rel[:, 0] * axis[1] - rel[:, 1] * axis[0]) / axis_length


def build_segment_box_polygon(
    segment_points,
    p_left,
    p_right,
    axial_margin_px: float = SPRING_BOX_AXIAL_MARGIN_PX,
    lateral_margin_px: float = SPRING_BOX_LATERAL_MARGIN_PX,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    points_array = np.asarray(segment_points, dtype=np.float64).reshape(-1, 2)
    left = np.asarray(p_left, dtype=np.float64).reshape(2)
    right = np.asarray(p_right, dtype=np.float64).reshape(2)
    axis = right - left
    axis_length = float(np.linalg.norm(axis))
    defaults = {
        "spring_box_center_offset_px": float("nan"),
        "spring_box_signed_center_offset_px": float("nan"),
        "spring_box_half_width_px": float("nan"),
        "spring_box_axial_span_px": float("nan"),
        "bend_score": float("nan"),
        "bend_threshold_px": float("nan"),
        "bent": False,
    }
    if not np.isfinite(axis_length) or axis_length <= 1e-6 or points_array.size == 0:
        return None, defaults

    axis_unit = axis / axis_length
    normal_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
    rel = points_array - left
    proj = rel[:, 0] * axis_unit[0] + rel[:, 1] * axis_unit[1]
    perp = rel[:, 0] * normal_unit[0] + rel[:, 1] * normal_unit[1]

    min_proj = float(np.min(proj) - float(axial_margin_px))
    max_proj = float(np.max(proj) + float(axial_margin_px))
    min_perp = float(np.min(perp) - float(lateral_margin_px))
    max_perp = float(np.max(perp) + float(lateral_margin_px))

    start = left + axis_unit * min_proj
    end = left + axis_unit * max_proj
    polygon = np.array(
        [
            start + normal_unit * max_perp,
            end + normal_unit * max_perp,
            end + normal_unit * min_perp,
            start + normal_unit * min_perp,
        ],
        dtype=np.float64,
    )

    center_offset_px = 0.5 * (max_perp + min_perp)
    half_width_px = 0.5 * (max_perp - min_perp)
    axial_span_px = max_proj - min_proj
    bend_threshold_px = max(float(BENT_CENTER_OFFSET_MIN_PX), float(half_width_px * BENT_CENTER_OFFSET_RATIO))
    bend_score = float(abs(center_offset_px) / max(half_width_px, 1.0))
    return polygon, {
        "spring_box_center_offset_px": float(abs(center_offset_px)),
        "spring_box_signed_center_offset_px": float(center_offset_px),
        "spring_box_half_width_px": float(half_width_px),
        "spring_box_axial_span_px": float(axial_span_px),
        "bend_score": bend_score,
        "bend_threshold_px": float(bend_threshold_px),
        "bent": bool(abs(center_offset_px) > bend_threshold_px),
    }


def compute_w_vis(
    frame,
    p_left,
    p_right,
    roi_half_width_px: int = ROI_HALF_WIDTH_PX,
    roi_end_margin_px: int = ROI_END_MARGIN_PX,
    min_segment_area: int = MIN_SEGMENT_AREA,
    morph_kernel_size: int = MORPH_KERNEL_SIZE,
) -> Dict[str, Any]:
    frame_array = np.asarray(frame)
    frame_height = int(frame_array.shape[0]) if frame_array.ndim >= 2 else 0
    frame_width = int(frame_array.shape[1]) if frame_array.ndim >= 2 else 0
    full_empty_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    _, default_box_metrics = build_segment_box_polygon(np.empty((0, 2), dtype=np.float64), p_left, p_right)
    p_left_array = np.asarray(p_left, dtype=np.float64).reshape(2)
    p_right_array = np.asarray(p_right, dtype=np.float64).reshape(2)
    axis = p_right_array - p_left_array
    axis_length_px = float(np.linalg.norm(axis))
    axis_angle_rad = float(np.arctan2(axis[1], axis[0])) if axis_length_px > 1e-6 else float("nan")
    if frame_height <= 0 or frame_width <= 0:
        return {
            "valid_detection": False,
            "reason": "frame is empty",
            "L_px": axis_length_px,
            "axis_angle_rad": axis_angle_rad,
            "w_vis_raw_px": float("nan"),
            "roi_mask": full_empty_mask,
            "roi_polygon": None,
            "segment_mask": full_empty_mask.copy(),
            "segment_points": np.empty((0, 2), dtype=np.float64),
            "spring_box_polygon": None,
            **default_box_metrics,
        }

    crop_pad_px = max(float(roi_half_width_px), 0.0) + max(float(roi_end_margin_px), 0.0) + 4.0
    x0 = max(0, int(np.floor(min(p_left_array[0], p_right_array[0]) - crop_pad_px)))
    y0 = max(0, int(np.floor(min(p_left_array[1], p_right_array[1]) - crop_pad_px)))
    x1 = min(frame_width, int(np.ceil(max(p_left_array[0], p_right_array[0]) + crop_pad_px)))
    y1 = min(frame_height, int(np.ceil(max(p_left_array[1], p_right_array[1]) + crop_pad_px)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return {
            "valid_detection": False,
            "reason": "cap markers are too close to build a spring ROI",
            "L_px": axis_length_px,
            "axis_angle_rad": axis_angle_rad,
            "w_vis_raw_px": float("nan"),
            "roi_mask": full_empty_mask,
            "roi_polygon": None,
            "segment_mask": full_empty_mask.copy(),
            "segment_points": np.empty((0, 2), dtype=np.float64),
            "spring_box_polygon": None,
            **default_box_metrics,
        }

    frame_crop = frame_array[y0:y1, x0:x1, ...] if frame_array.ndim >= 3 else frame_array[y0:y1, x0:x1]
    crop_offset = np.array([float(x0), float(y0)], dtype=np.float64)
    roi_mask_crop, roi_polygon_crop = build_oriented_roi_mask(
        p_left_array - crop_offset,
        p_right_array - crop_offset,
        frame_crop.shape,
        roi_half_width_px,
        roi_end_margin_px,
    )
    roi_mask = np.zeros_like(full_empty_mask)
    roi_mask[y0:y1, x0:x1] = roi_mask_crop
    roi_polygon = None if roi_polygon_crop is None else np.asarray(roi_polygon_crop, dtype=np.float64) + crop_offset
    if roi_polygon is None:
        return {
            "valid_detection": False,
            "reason": "cap markers are too close to build a spring ROI",
            "L_px": axis_length_px,
            "axis_angle_rad": axis_angle_rad,
            "w_vis_raw_px": float("nan"),
            "roi_mask": roi_mask,
            "roi_polygon": None,
            "segment_mask": full_empty_mask.copy(),
            "segment_points": np.empty((0, 2), dtype=np.float64),
            "spring_box_polygon": None,
            **default_box_metrics,
        }

    segment_mask_crop = segment_spring_in_roi(
        frame_crop,
        roi_mask_crop,
        min_segment_area=min_segment_area,
        morph_kernel_size=morph_kernel_size,
    )
    segment_mask = np.zeros_like(full_empty_mask)
    segment_mask[y0:y1, x0:x1] = segment_mask_crop
    segment_indices = np.column_stack(np.nonzero(segment_mask_crop > 0))
    if segment_indices.size == 0:
        return {
            "valid_detection": False,
            "reason": "no spring pixels segmented inside the oriented ROI",
            "L_px": axis_length_px,
            "axis_angle_rad": axis_angle_rad,
            "w_vis_raw_px": float("nan"),
            "roi_mask": roi_mask,
            "roi_polygon": roi_polygon,
            "segment_mask": segment_mask,
            "segment_points": np.empty((0, 2), dtype=np.float64),
            "spring_box_polygon": None,
            **default_box_metrics,
        }

    segment_points = segment_indices[:, ::-1].astype(np.float64)
    segment_points[:, 0] += float(x0)
    segment_points[:, 1] += float(y0)
    distances = point_line_distances(segment_points, p_left_array, p_right_array)
    w_vis_raw_px = float(np.max(distances)) if distances.size else float("nan")
    spring_box_polygon, spring_box_metrics = build_segment_box_polygon(segment_points, p_left_array, p_right_array)
    return {
        "valid_detection": bool(np.isfinite(w_vis_raw_px)),
        "reason": "ok" if np.isfinite(w_vis_raw_px) else "distance computation failed",
        "L_px": axis_length_px,
        "axis_angle_rad": axis_angle_rad,
        "w_vis_raw_px": w_vis_raw_px,
        "roi_mask": roi_mask,
        "roi_polygon": roi_polygon,
        "segment_mask": segment_mask,
        "segment_points": segment_points,
        "spring_box_polygon": spring_box_polygon,
        **spring_box_metrics,
    }


def _transform_to_pose_message(transform: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(transform[0, 3])
    msg.pose.position.y = float(transform[1, 3])
    msg.pose.position.z = float(transform[2, 3])
    qx, qy, qz, qw = _rotation_matrix_to_quaternion(transform[:3, :3])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def _pose_dict_from_transform(transform: np.ndarray) -> Dict[str, List[float]]:
    qx, qy, qz, qw = _rotation_matrix_to_quaternion(transform[:3, :3])
    return {
        "translation_m": [float(transform[0, 3]), float(transform[1, 3]), float(transform[2, 3])],
        "quaternion_xyzw": [float(qx), float(qy), float(qz), float(qw)],
        "rpy_deg": list(_rotation_matrix_to_rpy_deg(transform[:3, :3])),
    }


def _frame_id_for_marker_label(label: str) -> str:
    return f"aruco_{label}"


def _visual_bowing_summary(reason: str = "unavailable", **overrides: Any) -> Dict[str, Any]:
    summary = {
        "valid_detection": False,
        "reason": reason,
        "L_px": float("nan"),
        "axis_angle_rad": float("nan"),
        "axis_angle_frame": "camera_image_px",
        "theta_cap_rad": float("nan"),
        "theta_cap_rad_frame": "camera_marker",
        "theta_cap_camera_rad": float("nan"),
        "theta_cap_workspace_rad": float("nan"),
        "w_vis_px": float("nan"),
        "w_vis_raw_px": float("nan"),
        "w_vis_delta_px": float("nan"),
        "segmentation_frame": "camera_image_px",
        "baseline_px": float("nan"),
        "baseline_frames_collected": 0,
        "spring_box_center_offset_px": float("nan"),
        "spring_box_signed_center_offset_px": float("nan"),
        "spring_box_half_width_px": float("nan"),
        "spring_box_axial_span_px": float("nan"),
        "bend_score": float("nan"),
        "bend_threshold_px": float("nan"),
        "bent": False,
        "spring_box_polygon_px": [],
        "spring_box_polygon_frame": "camera_image_px",
        "cap_span_workspace_m": float("nan"),
        "cap_axis_workspace_rad": float("nan"),
        "left_cap_pose_workspace": None,
        "right_cap_pose_workspace": None,
        "left_cap_pose_manipulator": None,
        "right_cap_pose_manipulator": None,
        "coordinate_frames": {
            "workspace": None,
            "manipulator_robot1": None,
            "manipulator_robot2": None,
        },
        "left_cap_id": LEFT_CAP_ID,
        "right_cap_id": RIGHT_CAP_ID,
    }
    summary.update(overrides)
    return summary


def _visual_bowing_message_data(summary: Dict[str, Any]) -> List[float]:
    return [
        1.0 if bool(summary.get("valid_detection")) else 0.0,
        float(summary.get("L_px", float("nan"))),
        float(summary.get("axis_angle_rad", float("nan"))),
        float(summary.get("theta_cap_rad", float("nan"))),
        float(summary.get("w_vis_raw_px", float("nan"))),
        float(summary.get("w_vis_delta_px", float("nan"))),
        1.0 if bool(summary.get("bent")) else 0.0,
        float(summary.get("bend_score", float("nan"))),
        float(summary.get("spring_box_center_offset_px", float("nan"))),
    ]


def _overlay_image_message(overlay_rgb: np.ndarray, template_image: Optional[Image]) -> Image:
    overlay_msg = Image()
    overlay_msg.header = getattr(template_image, "header", None)
    overlay_msg.height = int(overlay_rgb.shape[0])
    overlay_msg.width = int(overlay_rgb.shape[1])
    overlay_msg.encoding = "rgb8"
    overlay_msg.is_bigendian = 0
    overlay_msg.step = int(overlay_rgb.shape[1] * 3)
    overlay_msg.data = np.ascontiguousarray(overlay_rgb, dtype=np.uint8).tobytes()
    return overlay_msg


def _depth_raw_to_meters(raw_value: float, encoding: str) -> float:
    normalized = str(encoding).lower()
    if normalized in {"mono16", "16uc1"}:
        return float(raw_value) * 1.0e-3
    return float(raw_value)


def _sample_aligned_depth_m(depth_array: np.ndarray, center_px: Tuple[float, float], encoding: str) -> Optional[float]:
    if depth_array.ndim != 2 or depth_array.size == 0:
        return None

    px = int(round(center_px[0]))
    py = int(round(center_px[1]))
    px = max(0, min(px, depth_array.shape[1] - 1))
    py = max(0, min(py, depth_array.shape[0] - 1))

    y0 = max(0, py - 1)
    y1 = min(depth_array.shape[0], py + 2)
    x0 = max(0, px - 1)
    x1 = min(depth_array.shape[1], px + 2)
    window = np.asarray(depth_array[y0:y1, x0:x1]).reshape(-1)
    nonzero = window[window > 0]
    if nonzero.size == 0:
        return None
    return _depth_raw_to_meters(float(np.median(nonzero)), encoding)


class CameraArucoTracker(Node):
    def __init__(
        self,
        image_topic: str,
        camera_info_topic: str,
        depth_image_topic: str,
        depth_camera_info_topic: str,
        marker_specs: List[MarkerSpec],
        world_marker_id: int,
        dictionary_name: str,
        marker_length_m: float,
        publish_period_s: float,
        enable_visual_bowing: bool = True,
        enable_depth_processing: bool = True,
        processing_resolution: Optional[Tuple[int, int]] = None,
        reference_marker_refresh_frames: int = 1,
        aruco_detector_profile: str = ARUCO_DETECTOR_PROFILE_DEFAULT,
        publish_spring_caps_only: bool = False,
        capture_color_video_path: Optional[str] = None,
        capture_video_fps: float = 30.0,
        capture_only: bool = False,
        left_cap_id: Optional[int] = None,
        right_cap_id: Optional[int] = None,
        roi_half_width_px: int = ROI_HALF_WIDTH_PX,
        roi_end_margin_px: int = ROI_END_MARGIN_PX,
        baseline_frames: int = BASELINE_FRAMES,
        min_segment_area: int = MIN_SEGMENT_AREA,
        morph_kernel_size: int = MORPH_KERNEL_SIZE,
        overlay_stale_grace_frames: int = DEFAULT_OVERLAY_STALE_GRACE_FRAMES,
        log_marker_visibility_transitions: bool = False,
        raw_signals_jsonl_path: Optional[str] = None,
        raw_signals_csv_path: Optional[str] = None,
    ) -> None:
        super().__init__("camera_aruco_tracker")
        self.image_topic = image_topic
        self.camera_info_topic = camera_info_topic
        self.depth_image_topic = depth_image_topic
        self.depth_camera_info_topic = depth_camera_info_topic
        self.marker_specs = list(marker_specs)
        self.marker_by_id = {spec.marker_id: spec for spec in self.marker_specs}
        self.marker_by_label = {spec.label: spec for spec in self.marker_specs}
        self.world_marker_id = int(world_marker_id)
        self.world_label = self.marker_by_id[self.world_marker_id].label
        self.world_frame_id = f"aruco_{self.world_label}"
        self.dictionary_name = dictionary_name
        self.marker_length_m = float(marker_length_m)
        self.left_cap_id = int(self._resolve_cap_marker_id("spring_cap_robot1", left_cap_id, LEFT_CAP_ID))
        self.right_cap_id = int(self._resolve_cap_marker_id("spring_cap_robot2", right_cap_id, RIGHT_CAP_ID))
        self.reference_marker_refresh_frames = max(1, int(reference_marker_refresh_frames))
        self.aruco_detector_profile = str(aruco_detector_profile or ARUCO_DETECTOR_PROFILE_DEFAULT).strip().lower()
        if self.aruco_detector_profile not in ARUCO_DETECTOR_PROFILE_CHOICES:
            raise ValueError(
                "aruco_detector_profile must be one of "
                + ", ".join(ARUCO_DETECTOR_PROFILE_CHOICES)
            )
        self.publish_spring_caps_only = bool(publish_spring_caps_only)
        self.processing_resolution = None if processing_resolution is None else (int(processing_resolution[0]), int(processing_resolution[1]))
        self.capture_color_video_path = str(capture_color_video_path).strip() if capture_color_video_path else None
        self.capture_video_fps = float(capture_video_fps)
        self.capture_only = bool(capture_only)
        self.roi_half_width_px = max(1, int(roi_half_width_px))
        self.roi_end_margin_px = max(0, int(roi_end_margin_px))
        self.baseline_frames = max(0, int(baseline_frames))
        self.min_segment_area = max(1, int(min_segment_area))
        self.morph_kernel_size = max(1, int(morph_kernel_size))
        self.enable_visual_bowing = bool(enable_visual_bowing)
        self.enable_depth_processing = bool(enable_depth_processing)

        if self.capture_color_video_path is not None and self.capture_video_fps <= 0.0:
            raise ValueError("capture_video_fps must be > 0 when color video capture is enabled")
        if self.capture_color_video_path is not None and (cv2 is None or not hasattr(cv2, "VideoWriter") or not hasattr(cv2, "VideoWriter_fourcc")):
            raise RuntimeError("Color video capture requires OpenCV video I/O in the active interpreter")

        self.color_image: Optional[Image] = None
        self.color_camera_info: Optional[CameraInfo] = None
        self.depth_image: Optional[Image] = None
        self.depth_camera_info: Optional[CameraInfo] = None
        self._logged_detection_error = False
        self._capture_color_video_writer = None if self.capture_color_video_path is None else _LazyRgbVideoWriter(self.capture_color_video_path, self.capture_video_fps)
        self._capture_color_video_info_saved = False
        self._logged_capture_video_error = False

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.image_topic, self._cb_color_image, qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._cb_color_camera_info, qos)
        self.create_subscription(Image, self.depth_image_topic, self._cb_depth_image, qos)
        self.create_subscription(CameraInfo, self.depth_camera_info_topic, self._cb_depth_camera_info, qos)

        self.pose_camera_publishers = {}
        self.pose_world_publishers = {}
        self.visible_publishers = {}
        for spec in self.marker_specs:
            self.pose_camera_publishers[spec.label] = self.create_publisher(
                PoseStamped,
                f"/camera_aruco/{spec.label}/pose_camera",
                1,
            )
            self.pose_world_publishers[spec.label] = self.create_publisher(
                PoseStamped,
                f"/camera_aruco/{spec.label}/pose_world",
                1,
            )
            self.visible_publishers[spec.label] = self.create_publisher(
                Bool,
                f"/camera_aruco/{spec.label}/visible",
                1,
            )

        self.markers_detected_publisher = self.create_publisher(Int32, DEFAULT_COUNT_TOPIC, 1)
        self.world_visible_publisher = self.create_publisher(Bool, DEFAULT_WORLD_VISIBLE_TOPIC, 1)
        self.summary_publisher = self.create_publisher(String, DEFAULT_SUMMARY_TOPIC, 1)
        self.visual_bowing_publisher = self.create_publisher(Float64MultiArray, DEFAULT_VISUAL_BOWING_TOPIC, 1)
        self.visual_bowing_overlay_publisher = self.create_publisher(Image, DEFAULT_VISUAL_BOWING_OVERLAY_TOPIC, 1)

        self._visual_bowing_baseline_samples: List[float] = []
        self._visual_bowing_baseline_px = float("nan")
        self._latest_visual_bowing_overlay_rgb: Optional[np.ndarray] = None
        self.overlay_stale_grace_frames = max(0, int(overlay_stale_grace_frames))
        self.log_marker_visibility_transitions = bool(log_marker_visibility_transitions)
        self.raw_signals_jsonl_path = str(raw_signals_jsonl_path).strip() if raw_signals_jsonl_path else None
        self.raw_signals_csv_path = str(raw_signals_csv_path).strip() if raw_signals_csv_path else None
        self._raw_signals_csv_logger = _SummaryCsvLogger(self.raw_signals_csv_path, self.marker_specs)
        self._raw_signal_sequence = 0
        self._logged_raw_signal_write_error = False
        self._logged_raw_signal_csv_write_error = False
        self.cap_marker_labels = self._resolve_cap_marker_labels()
        self.reference_marker_labels = frozenset(
            spec.label for spec in self.marker_specs if spec.label not in self.cap_marker_labels
        )
        self._processed_color_frame_cache_key: Optional[int] = None
        self._processed_color_frame_rgb: Optional[np.ndarray] = None
        self._processed_color_frame_gray: Optional[np.ndarray] = None
        self._processed_color_frame_source_width = 0
        self._processed_color_frame_source_height = 0
        self._cap_detection_cache: Dict[str, MarkerDetection] = {}
        self._reference_marker_cache: Dict[str, MarkerDetection] = {}
        self._analysis_frame_index = 0
        self._overlay_detection_cache: Dict[str, MarkerDetection] = {}
        self._overlay_detection_cache_age: Dict[str, int] = {}
        self._overlay_bowing_cache: Optional[Dict[str, Optional[np.ndarray]]] = None
        self._overlay_bowing_cache_age = 0
        self._overlay_image_age_frames = 0
        self._marker_visibility_state = {spec.label: None for spec in self.marker_specs}
        self._marker_missing_streaks = {spec.label: 0 for spec in self.marker_specs}

        self.get_logger().info(
            "Tracking ArUco markers: "
            + ", ".join(f"{spec.label}:{spec.marker_id}" for spec in self.marker_specs)
            + f" | world={self.world_label}:{self.world_marker_id}"
            + f" | marker_length={self.marker_length_m:.3f} m"
        )
        if self.processing_resolution is not None:
            self.get_logger().info(
                f"Processing color/depth frames at {self.processing_resolution[0]}x{self.processing_resolution[1]} before live detection"
            )
        if self.aruco_detector_profile != ARUCO_DETECTOR_PROFILE_DEFAULT:
            self.get_logger().info(
                f"Using OpenCV ArUco detector profile '{self.aruco_detector_profile}'"
            )
        if self.reference_marker_refresh_frames > 1 and self.cap_marker_labels:
            self.get_logger().info(
                "Spring-cap priority mode enabled; non-cap registration markers refresh every "
                f"{self.reference_marker_refresh_frames} publish cycles"
            )
        if self.publish_spring_caps_only and self.cap_marker_labels:
            self.get_logger().info(
                "Publishing only spring-cap pose entries/topics at high rate; non-cap markers remain internal for registration"
            )
        if self.capture_color_video_path is not None:
            self.get_logger().info(
                f"Raw color video capture enabled at {self.capture_video_fps:.2f} FPS -> {self.capture_color_video_path}"
            )
        if self.capture_only:
            self.get_logger().info("Capture-only mode enabled; live ArUco publishing timer is disabled")
        if not self.enable_visual_bowing:
            self.get_logger().info("Visual bowing summary and overlay are disabled for higher-rate pose publishing")
        if not self.enable_depth_processing:
            self.get_logger().info("Aligned-depth decoding is disabled during live publishing for higher-rate pose publishing")
        if self.raw_signals_jsonl_path:
            raw_dir = os.path.dirname(os.path.abspath(self.raw_signals_jsonl_path))
            if raw_dir:
                os.makedirs(raw_dir, exist_ok=True)
            self.get_logger().info(
                f"Raw camera_aruco summaries will be appended to {self.raw_signals_jsonl_path}"
            )
        if self.raw_signals_csv_path:
            self.get_logger().info(
                f"Flat camera_aruco CSV summaries will be appended to {self.raw_signals_csv_path}"
            )

        if not self.capture_only:
            self.create_timer(float(publish_period_s), self._publish_detections)

    def close_resources(self) -> None:
        if self._capture_color_video_writer is not None:
            self._capture_color_video_writer.close()

    def destroy_node(self):
        self.close_resources()
        return super().destroy_node()

    def _append_raw_signal_record(self, summary: Dict[str, Any], source: str) -> None:
        if not self.raw_signals_jsonl_path and not self.raw_signals_csv_path:
            return
        record = {
            "record_type": "camera_aruco_raw_signal",
            "source": str(source),
            "sequence": int(self._raw_signal_sequence),
            "wall_time_unix_ns": time.time_ns(),
            "color_image_stamp_ns": _message_stamp_ns(self.color_image),
            "color_camera_info_stamp_ns": _message_stamp_ns(self.color_camera_info),
            "depth_image_stamp_ns": _message_stamp_ns(self.depth_image),
            "depth_camera_info_stamp_ns": _message_stamp_ns(self.depth_camera_info),
            "summary": summary,
        }
        wrote_any_record = False
        if self.raw_signals_jsonl_path:
            try:
                with open(self.raw_signals_jsonl_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, separators=(",", ":")))
                    handle.write("\n")
                wrote_any_record = True
                self._logged_raw_signal_write_error = False
            except OSError as exc:
                if not self._logged_raw_signal_write_error:
                    self.get_logger().warning(
                        f"camera_aruco raw-signal logging failed for {self.raw_signals_jsonl_path}: {exc}"
                    )
                    self._logged_raw_signal_write_error = True
        if self.raw_signals_csv_path:
            try:
                self._raw_signals_csv_logger.append(summary, metadata={key: value for key, value in record.items() if key != "summary"})
                wrote_any_record = True
                self._logged_raw_signal_csv_write_error = False
            except OSError as exc:
                if not self._logged_raw_signal_csv_write_error:
                    self.get_logger().warning(
                        f"camera_aruco raw-signal CSV logging failed for {self.raw_signals_csv_path}: {exc}"
                    )
                    self._logged_raw_signal_csv_write_error = True
        if wrote_any_record:
            self._raw_signal_sequence += 1

    def _resolve_cap_marker_id(self, preferred_label: str, configured_id: Optional[int], fallback_id: int) -> int:
        if configured_id is not None:
            return int(configured_id)
        spec = self.marker_by_label.get(preferred_label)
        if spec is not None:
            return int(spec.marker_id)
        return int(fallback_id)

    def _resolve_cap_marker_labels(self) -> Set[str]:
        preferred_ids = {int(self.left_cap_id), int(self.right_cap_id)}
        return {
            spec.label
            for spec in self.marker_specs
            if spec.label in SPRING_CAP_MARKER_LABELS or int(spec.marker_id) in preferred_ids
        }

    def _should_refresh_reference_markers(self, require_world: bool = False) -> bool:
        if require_world:
            return True
        if self.reference_marker_refresh_frames <= 1:
            return True
        if not self.reference_marker_labels or not self.cap_marker_labels:
            return True
        if self._analysis_frame_index % self.reference_marker_refresh_frames == 0:
            return True
        return not self._reference_marker_cache

    def _refresh_reference_marker_cache(self, detections: Dict[str, MarkerDetection]) -> None:
        self._reference_marker_cache = {
            label: detection
            for label, detection in detections.items()
            if label in self.reference_marker_labels
        }

    def _published_marker_labels(self) -> Set[str]:
        if self.publish_spring_caps_only and self.cap_marker_labels:
            return set(self.cap_marker_labels)
        return {spec.label for spec in self.marker_specs}

    def _update_cap_detection_cache(self, detections: Dict[str, MarkerDetection]) -> None:
        for label, detection in detections.items():
            if label in self.cap_marker_labels:
                self._cap_detection_cache[label] = detection

    def _cap_tracking_roi_bounds(
        self,
        image_shape: Tuple[int, int],
        detection: MarkerDetection,
    ) -> Optional[Tuple[int, int, int, int]]:
        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            return None

        corners = detection.corners_px
        if corners is not None and np.asarray(corners).size >= 8:
            corners_array = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            center_x = float(np.mean(corners_array[:, 0]))
            center_y = float(np.mean(corners_array[:, 1]))
            marker_span_px = max(
                float(np.max(corners_array[:, 0]) - np.min(corners_array[:, 0])),
                float(np.max(corners_array[:, 1]) - np.min(corners_array[:, 1])),
                8.0,
            )
            half_span_px = 0.5 * marker_span_px
        else:
            center_x = float(detection.center_px[0])
            center_y = float(detection.center_px[1])
            half_span_px = 4.0

        roi_half_size_px = max(
            float(CAP_TRACK_MIN_ROI_HALF_SIZE_PX),
            float(half_span_px * CAP_TRACK_ROI_HALF_SIZE_SCALE + CAP_TRACK_ROI_EXTRA_MARGIN_PX),
        )
        x0 = max(0, int(np.floor(center_x - roi_half_size_px)))
        y0 = max(0, int(np.floor(center_y - roi_half_size_px)))
        x1 = min(width, int(np.ceil(center_x + roi_half_size_px)))
        y1 = min(height, int(np.ceil(center_y + roi_half_size_px)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return x0, y0, x1, y1

    def _estimate_marker_detections(
        self,
        marker_corners: List[np.ndarray],
        marker_specs: List[MarkerSpec],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Dict[str, MarkerDetection]:
        if not marker_corners:
            return {}

        rvecs, tvecs, _ = aruco_utils.cv2.aruco.estimatePoseSingleMarkers(
            marker_corners,
            self.marker_length_m,
            camera_matrix,
            dist_coeffs,
        )

        detections: Dict[str, MarkerDetection] = {}
        for corners, spec, rvec, tvec in zip(marker_corners, marker_specs, np.asarray(rvecs), np.asarray(tvecs)):
            detections[spec.label] = MarkerDetection(
                spec=spec,
                center_px=get_marker_center(corners),
                rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
                tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
                corners_px=np.asarray(corners, dtype=np.float64).reshape(-1, 2),
            )
        self._update_cap_detection_cache(detections)
        return detections

    def _detect_cap_markers_from_cached_rois(
        self,
        gray: np.ndarray,
        dictionary: Any,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tracked_label_set: Set[str],
    ) -> Optional[Dict[str, MarkerDetection]]:
        if not tracked_label_set:
            return {}

        detector_parameters = aruco_utils._get_detector_parameters(self.aruco_detector_profile)
        selected_corners: List[np.ndarray] = []
        selected_specs: List[MarkerSpec] = []
        for label in sorted(tracked_label_set):
            cached_detection = self._cap_detection_cache.get(label)
            if cached_detection is None:
                return None

            bounds = self._cap_tracking_roi_bounds(gray.shape[:2], cached_detection)
            if bounds is None:
                return None
            x0, y0, x1, y1 = bounds
            roi_gray = np.ascontiguousarray(gray[y0:y1, x0:x1])
            if roi_gray.size == 0:
                return None

            corners, ids, _ = aruco_utils.cv2.aruco.detectMarkers(
                roi_gray,
                dictionary,
                parameters=detector_parameters,
            )
            if ids is None or len(ids) == 0:
                return None

            matched_label = False
            ids_flat = [int(value) for value in np.asarray(ids).reshape(-1)]
            for marker_corners, marker_id in zip(corners, ids_flat):
                spec = self.marker_by_id.get(marker_id)
                if spec is None or spec.label != label:
                    continue
                corners_full = np.asarray(marker_corners, dtype=np.float64).reshape(1, -1, 2)
                corners_full[..., 0] += float(x0)
                corners_full[..., 1] += float(y0)
                selected_corners.append(corners_full)
                selected_specs.append(spec)
                matched_label = True
                break
            if not matched_label:
                return None

        return self._estimate_marker_detections(selected_corners, selected_specs, camera_matrix, dist_coeffs)

    def _cb_color_image(self, msg: Image) -> None:
        self.color_image = msg
        self._processed_color_frame_cache_key = None
        self._processed_color_frame_rgb = None
        self._processed_color_frame_gray = None
        self._processed_color_frame_source_width = 0
        self._processed_color_frame_source_height = 0
        self._record_color_video_frame(msg)

    def _cb_color_camera_info(self, msg: CameraInfo) -> None:
        self.color_camera_info = msg
        self._maybe_write_capture_camera_info_sidecar()

    def _cb_depth_image(self, msg: Image) -> None:
        self.depth_image = msg

    def _cb_depth_camera_info(self, msg: CameraInfo) -> None:
        self.depth_camera_info = msg

    def _capture_camera_info_json_path(self) -> Optional[str]:
        if self._capture_color_video_writer is None:
            return None
        output_path = self._capture_color_video_writer.output_path or self.capture_color_video_path
        if not output_path:
            return None
        return _camera_info_json_path_for_video(output_path)

    def _maybe_write_capture_camera_info_sidecar(self) -> None:
        if self._capture_color_video_writer is None or self.color_camera_info is None or self._capture_color_video_info_saved:
            return
        if self._capture_color_video_writer.frame_count <= 0:
            return
        info_path = self._capture_camera_info_json_path()
        if not info_path:
            return
        _write_json_file(info_path, camera_info_to_dict(self.color_camera_info))
        self._capture_color_video_info_saved = True

    def _record_color_video_frame(self, image_msg: Image) -> None:
        if self._capture_color_video_writer is None:
            return
        try:
            frame_rgb = _to_rgb_u8(decode_sensor_image(image_msg))
            self._capture_color_video_writer.write_rgb_frame(frame_rgb)
            self._maybe_write_capture_camera_info_sidecar()
            self._logged_capture_video_error = False
        except Exception as exc:
            if not self._logged_capture_video_error:
                self.get_logger().warning(f"camera_aruco color-video capture failed: {exc}")
                self._logged_capture_video_error = True

    def _current_processed_color_frame(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        if self.color_image is None:
            raise RuntimeError("image unavailable")
        cache_key = id(self.color_image)
        if (
            self._processed_color_frame_cache_key == cache_key
            and self._processed_color_frame_rgb is not None
            and self._processed_color_frame_gray is not None
        ):
            return (
                self._processed_color_frame_rgb,
                self._processed_color_frame_gray,
                int(self._processed_color_frame_source_width),
                int(self._processed_color_frame_source_height),
            )

        color_frame = decode_sensor_image(self.color_image)
        gray_source = _to_gray_u8(color_frame)
        source_height, source_width = int(gray_source.shape[0]), int(gray_source.shape[1])
        frame_rgb = _to_rgb_u8(color_frame)
        if self.processing_resolution is not None:
            frame_rgb = _resize_array(
                frame_rgb,
                self.processing_resolution[0],
                self.processing_resolution[1],
                interpolation="area",
            )
            gray = _resize_array(
                gray_source,
                self.processing_resolution[0],
                self.processing_resolution[1],
                interpolation="area",
            )
        else:
            gray = gray_source

        self._processed_color_frame_cache_key = cache_key
        self._processed_color_frame_rgb = frame_rgb
        self._processed_color_frame_gray = gray
        self._processed_color_frame_source_width = source_width
        self._processed_color_frame_source_height = source_height
        return frame_rgb, gray, source_width, source_height

    def _aruco_runtime_error(self) -> Optional[str]:
        if aruco_utils._aruco_available():
            return None

        interpreter = sys.executable
        message = f"OpenCV ArUco support is unavailable in interpreter {interpreter}."
        if "/.venv/" in interpreter or interpreter.endswith("/.venv/bin/python3"):
            message += (
                " The active repository virtualenv does not currently provide cv2/aruco. "
                "Run this script with /usr/bin/python3 after sourcing ROS, or install an "
                "OpenCV contrib build into the virtualenv."
            )
        else:
            message += " Install an OpenCV build that includes the aruco module."
        return message

    def _available_topic_names(self) -> List[str]:
        get_topics = getattr(self, "get_topic_names_and_types", None)
        if get_topics is None:
            return []
        try:
            topics = get_topics()
        except Exception:
            return []
        return sorted(name for name, _types in topics)

    def _expected_frame_topics(self) -> List[str]:
        return [
            self.image_topic,
            self.camera_info_topic,
            self.depth_image_topic,
            self.depth_camera_info_topic,
        ]

    def self_check_summary(self) -> Dict[str, Any]:
        runtime_error = self._aruco_runtime_error()
        available_topics = self._available_topic_names()
        relevant_topics = [
            name for name in available_topics if name.startswith("/camera/") or name.startswith("/spring_monitor/")
        ]
        expected_topics = self._expected_frame_topics()
        missing_topics = [topic for topic in expected_topics if topic not in available_topics]

        hints = []
        if runtime_error is not None:
            hints.append(runtime_error)
        if missing_topics:
            hints.append(
                "Start the RealSense node or launch dual_hardware_variable_stiffness.launch.py "
                "with enable_depth_camera:=true."
            )

        return {
            "ok": runtime_error is None and not missing_topics,
            "interpreter": sys.executable,
            "aruco_available": runtime_error is None,
            "aruco_error": runtime_error,
            "expected_topics": expected_topics,
            "missing_topics": missing_topics,
            "visible_relevant_topics": relevant_topics,
            "visible_topics_sample": available_topics[:20],
            "hints": hints,
        }

    def _publish_visibility(self, label: str, visible: bool) -> None:
        visible_msg = Bool()
        visible_msg.data = bool(visible)
        self.visible_publishers[label].publish(visible_msg)

    def _log_marker_visibility_transitions(
        self,
        visible_labels,
        failure_reason: Optional[str] = None,
    ) -> None:
        if not self.log_marker_visibility_transitions:
            return

        visible_set = set(visible_labels)
        for spec in self.marker_specs:
            visible = spec.label in visible_set
            was_visible = self._marker_visibility_state.get(spec.label)
            if visible:
                missed_cycles = int(self._marker_missing_streaks.get(spec.label, 0))
                if was_visible is False:
                    self.get_logger().info(
                        f"camera_aruco marker recovered: {spec.label}:{spec.marker_id} "
                        f"after {missed_cycles} missed publish cycle(s)"
                    )
                self._marker_missing_streaks[spec.label] = 0
            else:
                self._marker_missing_streaks[spec.label] = int(self._marker_missing_streaks.get(spec.label, 0)) + 1
                if was_visible is not False:
                    message = (
                        f"camera_aruco marker missing: {spec.label}:{spec.marker_id} "
                        f"streak={self._marker_missing_streaks[spec.label]}"
                    )
                    if failure_reason:
                        message += f" reason={failure_reason}"
                    self.get_logger().warning(message)
            self._marker_visibility_state[spec.label] = visible

    def _publish_empty_state(self, reason: str) -> None:
        visual_bowing = _visual_bowing_summary(
            reason=reason,
            left_cap_id=self.left_cap_id,
            right_cap_id=self.right_cap_id,
            baseline_px=self._visual_bowing_baseline_px,
            baseline_frames_collected=len(self._visual_bowing_baseline_samples),
        )
        self._log_marker_visibility_transitions(set(), failure_reason=reason)
        for spec in self.marker_specs:
            self._publish_visibility(spec.label, False)

        count_msg = Int32()
        count_msg.data = 0
        self.markers_detected_publisher.publish(count_msg)

        world_visible_msg = Bool()
        world_visible_msg.data = False
        self.world_visible_publisher.publish(world_visible_msg)

        bowing_msg = Float64MultiArray()
        bowing_msg.data = _visual_bowing_message_data(visual_bowing)
        self.visual_bowing_publisher.publish(bowing_msg)
        self._publish_visual_bowing_overlay()

        summary = {
            "valid": False,
            "reason": reason,
            "world_marker_id": self.world_marker_id,
            "world_label": self.world_label,
            "markers_detected": 0,
            "observed_labels": [],
            "cached_marker_labels": [],
            "registration_cache_active": False,
            "detections": [],
            "world_marker_visible": False,
            "world_registration_available": False,
            "reference_marker_refresh_frames": int(self.reference_marker_refresh_frames),
            "visual_bowing": visual_bowing,
        }

        summary_msg = String()
        summary_msg.data = json.dumps(summary, separators=(",", ":"))
        self.summary_publisher.publish(summary_msg)
        self._append_raw_signal_record(summary, source="publish_empty_state")

    def _detect_markers(self, tracked_labels: Optional[Set[str]] = None) -> Tuple[str, Dict[str, MarkerDetection], int, int]:
        if not aruco_utils._aruco_available():
            raise RuntimeError("OpenCV ArUco support is unavailable")
        if self.color_image is None or self.color_camera_info is None:
            raise RuntimeError("image or camera_info unavailable")

        source_width = int(getattr(self.color_image, "width", 0) or 0)
        source_height = int(getattr(self.color_image, "height", 0) or 0)
        if self.processing_resolution is None:
            gray = aruco_utils._gray_image(self.color_image)
        else:
            _frame_rgb, gray, source_width, source_height = self._current_processed_color_frame()
        if source_width <= 0 or source_height <= 0:
            source_height, source_width = int(gray.shape[0]), int(gray.shape[1])

        dictionary = aruco_utils._get_aruco_dictionary(self.dictionary_name)
        camera_matrix, dist_coeffs = aruco_utils._camera_matrix_from_info(self.color_camera_info)
        if camera_matrix is None:
            raise RuntimeError("camera_info is missing intrinsics")
        if self.processing_resolution is not None and (int(gray.shape[1]) != source_width or int(gray.shape[0]) != source_height):
            camera_matrix = _scale_camera_matrix(
                camera_matrix,
                source_width,
                source_height,
                int(gray.shape[1]),
                int(gray.shape[0]),
            )

        tracked_label_set = None if tracked_labels is None else set(tracked_labels)
        frame_id = self.color_camera_info.header.frame_id or self.color_image.header.frame_id or "camera"
        if tracked_label_set is not None and tracked_label_set and tracked_label_set.issubset(self.cap_marker_labels):
            roi_detections = self._detect_cap_markers_from_cached_rois(
                gray,
                dictionary,
                camera_matrix,
                dist_coeffs,
                tracked_label_set,
            )
            if roi_detections is not None:
                return frame_id, roi_detections, int(gray.shape[1]), int(gray.shape[0])

        corners, ids, _ = aruco_utils.cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=aruco_utils._get_detector_parameters(self.aruco_detector_profile),
        )
        if ids is None or len(ids) == 0:
            return frame_id, {}, int(gray.shape[1]), int(gray.shape[0])

        selected_corners = []
        selected_specs: List[MarkerSpec] = []
        ids_flat = [int(value) for value in np.asarray(ids).reshape(-1)]
        for marker_corners, marker_id in zip(corners, ids_flat):
            spec = self.marker_by_id.get(marker_id)
            if spec is None:
                continue
            if tracked_label_set is not None and spec.label not in tracked_label_set:
                continue
            selected_corners.append(np.asarray(marker_corners, dtype=np.float64).reshape(1, -1, 2))
            selected_specs.append(spec)

        if not selected_corners:
            return frame_id, {}, int(gray.shape[1]), int(gray.shape[0])

        detections = self._estimate_marker_detections(selected_corners, selected_specs, camera_matrix, dist_coeffs)
        return frame_id, detections, int(gray.shape[1]), int(gray.shape[0])

    def _decode_depth_stream(self, color_width: int, color_height: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        depth_status: Dict[str, Any] = {
            "available": self.depth_image is not None,
            "camera_info_available": self.depth_camera_info is not None,
            "valid": False,
            "reason": "depth image unavailable",
        }
        if self.depth_image is None:
            return None, depth_status

        depth_status.update(
            {
                "frame_id": getattr(getattr(self.depth_image, "header", None), "frame_id", ""),
                "encoding": str(getattr(self.depth_image, "encoding", "")),
                "width": int(getattr(self.depth_image, "width", 0) or 0),
                "height": int(getattr(self.depth_image, "height", 0) or 0),
            }
        )

        if self.depth_camera_info is None:
            depth_status["reason"] = "depth camera_info unavailable"
            return None, depth_status

        try:
            depth_array = decode_sensor_image(self.depth_image)
        except Exception as exc:
            depth_status["reason"] = f"depth decode failed: {exc}"
            return None, depth_status

        if depth_array.ndim != 2:
            depth_status["reason"] = f"depth image expected 2-D array, got shape {depth_array.shape}"
            return None, depth_status

        if depth_array.shape[1] != color_width or depth_array.shape[0] != color_height:
            if self.processing_resolution is not None:
                depth_array = _resize_array(depth_array, color_width, color_height, interpolation="nearest")
                depth_status["processed_width"] = int(color_width)
                depth_status["processed_height"] = int(color_height)
            else:
                depth_status["reason"] = (
                    f"aligned depth resolution mismatch: depth={depth_array.shape[1]}x{depth_array.shape[0]} "
                    f"color={color_width}x{color_height}"
                )
                return None, depth_status

        if depth_array.shape[1] != color_width or depth_array.shape[0] != color_height:
            depth_status["reason"] = (
                f"aligned depth resolution mismatch: depth={depth_array.shape[1]}x{depth_array.shape[0]} "
                f"color={color_width}x{color_height}"
            )
            return None, depth_status

        nonzero = depth_array[depth_array > 0]
        nonzero_ratio = float(nonzero.size / depth_array.size) if depth_array.size else 0.0
        depth_status["nonzero_ratio"] = nonzero_ratio
        if nonzero.size == 0:
            depth_status["reason"] = "depth frame decoded but all samples are zero"
            return None, depth_status

        depth_status.update(
            {
                "valid": True,
                "reason": "ok",
                "min_depth_m": _depth_raw_to_meters(float(np.min(nonzero)), depth_status["encoding"]),
                "max_depth_m": _depth_raw_to_meters(float(np.max(nonzero)), depth_status["encoding"]),
                "camera_info_frame_id": getattr(getattr(self.depth_camera_info, "header", None), "frame_id", ""),
            }
        )
        return depth_array, depth_status

    def _find_cap_detection(
        self,
        detections: Dict[str, MarkerDetection],
        preferred_label: str,
        fallback_id: int,
    ) -> Optional[MarkerDetection]:
        detection = detections.get(preferred_label)
        if detection is not None:
            return detection
        for candidate in detections.values():
            if candidate.spec.marker_id == int(fallback_id):
                return candidate
        return None

    def _marker_yaw_rad(self, detection: MarkerDetection) -> float:
        rotation = detection.transform_camera_marker()[:3, :3]
        return float(np.radians(_rotation_matrix_to_rpy_deg(rotation)[2]))

    def _relative_pose_dict(
        self,
        target_detection: MarkerDetection,
        reference_detection: Optional[MarkerDetection],
    ) -> Optional[Dict[str, List[float]]]:
        if reference_detection is None:
            return None
        target_transform = target_detection.transform_camera_marker()
        reference_transform_inv = _invert_transform(reference_detection.transform_camera_marker())
        return _pose_dict_from_transform(reference_transform_inv @ target_transform)

    def _relative_pose_dict_from_transform(
        self,
        target_transform: np.ndarray,
        reference_detection: Optional[MarkerDetection],
    ) -> Optional[Dict[str, List[float]]]:
        if reference_detection is None:
            return None
        reference_transform_inv = _invert_transform(reference_detection.transform_camera_marker())
        return _pose_dict_from_transform(reference_transform_inv @ target_transform)

    def _coordinate_frame_summary(self, detections: Dict[str, MarkerDetection]) -> Dict[str, Dict[str, Optional[str]]]:
        workspace_detection = detections.get(WORKSPACE_MARKER_LABEL)
        manipulator_robot1_detection = detections.get(MANIPULATOR_ROBOT1_MARKER_LABEL)
        manipulator_robot2_detection = detections.get(MANIPULATOR_ROBOT2_MARKER_LABEL)
        return {
            "camera": {
                "frame_id": str(self.color_camera_info.header.frame_id or self.image_topic) if self.color_camera_info is not None else None,
                "marker_label": None,
                "semantic": "camera optical frame",
            },
            "world": {
                "frame_id": self.world_frame_id if detections.get(self.world_label) is not None else None,
                "marker_label": self.world_label,
                "semantic": "ground reference frame",
            },
            "workspace": {
                "frame_id": _frame_id_for_marker_label(WORKSPACE_MARKER_LABEL) if workspace_detection is not None else None,
                "marker_label": WORKSPACE_MARKER_LABEL,
                "semantic": "metal platform workspace frame",
            },
            "manipulator_robot1": {
                "frame_id": _frame_id_for_marker_label(MANIPULATOR_ROBOT1_MARKER_LABEL) if manipulator_robot1_detection is not None else None,
                "marker_label": MANIPULATOR_ROBOT1_MARKER_LABEL,
                "semantic": "robot1 manipulator frame",
            },
            "manipulator_robot2": {
                "frame_id": _frame_id_for_marker_label(MANIPULATOR_ROBOT2_MARKER_LABEL) if manipulator_robot2_detection is not None else None,
                "marker_label": MANIPULATOR_ROBOT2_MARKER_LABEL,
                "semantic": "robot2 manipulator frame",
            },
        }

    def _update_visual_bowing_baseline(self, w_vis_raw_px: float) -> float:
        if not np.isfinite(w_vis_raw_px):
            return self._visual_bowing_baseline_px
        if self.baseline_frames > 0 and len(self._visual_bowing_baseline_samples) < self.baseline_frames:
            self._visual_bowing_baseline_samples.append(float(w_vis_raw_px))
            self._visual_bowing_baseline_px = float(np.median(self._visual_bowing_baseline_samples))
        return self._visual_bowing_baseline_px

    def _set_latest_visual_bowing_overlay(self, overlay_rgb: np.ndarray) -> None:
        self._latest_visual_bowing_overlay_rgb = np.ascontiguousarray(overlay_rgb, dtype=np.uint8)
        self._overlay_image_age_frames = 0

    def _overlay_detection_grace_frames(self, label: str) -> int:
        if label in STATIC_OVERLAY_MARKER_LABELS:
            return max(self.overlay_stale_grace_frames, int(STATIC_MARKER_OVERLAY_STALE_GRACE_FRAMES))
        return self.overlay_stale_grace_frames

    def _overlay_detections_with_grace(
        self,
        detections: Dict[str, MarkerDetection],
    ) -> Tuple[Dict[str, MarkerDetection], Set[str]]:
        for label, detection in detections.items():
            self._overlay_detection_cache[label] = detection
            self._overlay_detection_cache_age[label] = 0

        overlay_detections = dict(detections)
        stale_detection_labels: Set[str] = set()
        for label, cached_detection in list(self._overlay_detection_cache.items()):
            if label in detections:
                continue
            age_frames = self._overlay_detection_cache_age.get(label, 0) + 1
            if age_frames > self._overlay_detection_grace_frames(label):
                self._overlay_detection_cache.pop(label, None)
                self._overlay_detection_cache_age.pop(label, None)
                continue
            self._overlay_detection_cache_age[label] = age_frames
            overlay_detections[label] = cached_detection
            stale_detection_labels.add(label)

        return overlay_detections, stale_detection_labels

    def _overlay_bowing_result_with_grace(
        self,
        bowing_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        overlay_result = None if bowing_result is None else dict(bowing_result)
        spring_box_polygon = None if overlay_result is None else overlay_result.get("spring_box_polygon")
        if spring_box_polygon is not None:
            self._overlay_bowing_cache = {
                "roi_polygon": None
                if overlay_result.get("roi_polygon") is None
                else np.asarray(overlay_result["roi_polygon"], dtype=np.float64).copy(),
                "spring_box_polygon": np.asarray(spring_box_polygon, dtype=np.float64).copy(),
            }
            self._overlay_bowing_cache_age = 0
            return overlay_result

        if self._overlay_bowing_cache is None:
            return overlay_result

        age_frames = self._overlay_bowing_cache_age + 1
        if age_frames > self.overlay_stale_grace_frames:
            self._overlay_bowing_cache = None
            self._overlay_bowing_cache_age = 0
            return overlay_result

        self._overlay_bowing_cache_age = age_frames
        if overlay_result is None:
            overlay_result = {}
        if overlay_result.get("roi_polygon") is None:
            overlay_result["roi_polygon"] = self._overlay_bowing_cache.get("roi_polygon")
        if overlay_result.get("spring_box_polygon") is None:
            overlay_result["spring_box_polygon"] = self._overlay_bowing_cache.get("spring_box_polygon")
        return overlay_result

    def _publish_visual_bowing_overlay(self) -> None:
        if self._latest_visual_bowing_overlay_rgb is None or self.color_image is None:
            return
        if self._overlay_image_age_frames > self.overlay_stale_grace_frames:
            self._latest_visual_bowing_overlay_rgb = None
            self._overlay_image_age_frames = 0
            return
        self.visual_bowing_overlay_publisher.publish(
            _overlay_image_message(self._latest_visual_bowing_overlay_rgb, self.color_image)
        )
        self._overlay_image_age_frames += 1

    def _draw_visual_bowing_overlay(
        self,
        frame_rgb: np.ndarray,
        detections: Dict[str, MarkerDetection],
        left_detection: Optional[MarkerDetection],
        right_detection: Optional[MarkerDetection],
        bowing_result: Optional[Dict[str, Any]],
        summary: Dict[str, Any],
        stale_detection_labels: Optional[Set[str]] = None,
    ) -> np.ndarray:
        overlay = _to_rgb_u8(frame_rgb).copy()
        stale_labels = set() if stale_detection_labels is None else set(stale_detection_labels)
        segment_mask = None if bowing_result is None else bowing_result.get("segment_mask")
        if segment_mask is not None and np.any(segment_mask > 0):
            highlight = np.zeros_like(overlay)
            highlight[:, :, 1] = 255
            keep = segment_mask > 0
            overlay[keep] = ((0.35 * overlay[keep]) + (0.65 * highlight[keep])).astype(np.uint8)

        if cv2 is not None and all(hasattr(cv2, name) for name in ("circle", "line", "polylines", "putText", "FONT_HERSHEY_SIMPLEX")):
            left_label = None if left_detection is None else left_detection.spec.label
            right_label = None if right_detection is None else right_detection.spec.label
            for detection in sorted(detections.values(), key=lambda value: value.spec.label):
                color = (200, 200, 200)
                if detection.spec.label == self.world_label:
                    color = (255, 165, 0)
                elif detection.spec.label == left_label:
                    color = (255, 0, 0)
                elif detection.spec.label == right_label:
                    color = (0, 255, 255)
                is_stale = detection.spec.label in stale_labels
                if is_stale:
                    color = tuple(min(255, int(round(component * 0.45 + 110))) for component in color)
                center = tuple(int(round(value)) for value in detection.center_px)
                cv2.circle(overlay, center, 4 if is_stale else 5, color, 1 if is_stale else 2)
                label = f"{detection.spec.label}:{detection.spec.marker_id}"
                if is_stale:
                    label += " [held]"
                cv2.putText(overlay, label, (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            if left_detection is not None and right_detection is not None:
                p_left = tuple(int(round(value)) for value in left_detection.center_px)
                p_right = tuple(int(round(value)) for value in right_detection.center_px)
                line_is_stale = left_label in stale_labels or right_label in stale_labels
                cv2.line(
                    overlay,
                    p_left,
                    p_right,
                    (180, 180, 80) if line_is_stale else (255, 255, 0),
                    1 if line_is_stale else 2,
                )

            roi_polygon = None if bowing_result is None else bowing_result.get("roi_polygon")
            if roi_polygon is not None:
                polygon_i32 = np.round(np.asarray(roi_polygon, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [polygon_i32], True, (255, 0, 255), 1)

            spring_box_polygon = None if bowing_result is None else bowing_result.get("spring_box_polygon")
            if spring_box_polygon is not None:
                polygon_i32 = np.round(np.asarray(spring_box_polygon, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [polygon_i32], True, (0, 255, 0), 2)

            text = "w_vis_delta_px=nan"
            bend_text = "bend unavailable"
            if bool(summary.get("valid_detection")) and np.isfinite(summary.get("w_vis_delta_px", float("nan"))):
                text = f"w_vis_delta_px={summary['w_vis_delta_px']:.2f}"
                if np.isfinite(summary.get("spring_box_center_offset_px", float("nan"))):
                    bend_state = "yes" if bool(summary.get("bent")) else "no"
                    bend_text = (
                        f"bent={bend_state} offset_px={summary['spring_box_center_offset_px']:.2f} "
                        f"score={summary.get('bend_score', float('nan')):.2f}"
                    )
            elif summary.get("reason"):
                text = f"w_vis invalid: {summary['reason']}"
            cv2.putText(overlay, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(overlay, bend_text, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return overlay

    def _compute_visual_bowing_summary(self, detections: Dict[str, MarkerDetection]) -> Dict[str, Any]:
        if self.color_image is None:
            return _visual_bowing_summary(
                reason="color image unavailable",
                coordinate_frames=self._coordinate_frame_summary(detections),
                left_cap_id=self.left_cap_id,
                right_cap_id=self.right_cap_id,
                baseline_px=self._visual_bowing_baseline_px,
                baseline_frames_collected=len(self._visual_bowing_baseline_samples),
            )

        try:
            if self.processing_resolution is None:
                color_frame = decode_sensor_image(self.color_image)
                frame_rgb = _to_rgb_u8(color_frame)
            else:
                frame_rgb, _gray, _source_width, _source_height = self._current_processed_color_frame()
        except Exception as exc:
            return _visual_bowing_summary(
                reason=f"color image decode failed: {exc}",
                coordinate_frames=self._coordinate_frame_summary(detections),
                left_cap_id=self.left_cap_id,
                right_cap_id=self.right_cap_id,
                baseline_px=self._visual_bowing_baseline_px,
                baseline_frames_collected=len(self._visual_bowing_baseline_samples),
            )
        overlay_detections, stale_detection_labels = self._overlay_detections_with_grace(detections)
        workspace_detection = detections.get(WORKSPACE_MARKER_LABEL)
        manipulator_robot1_detection = detections.get(MANIPULATOR_ROBOT1_MARKER_LABEL)
        manipulator_robot2_detection = detections.get(MANIPULATOR_ROBOT2_MARKER_LABEL)
        coordinate_frames = self._coordinate_frame_summary(detections)
        left_detection = self._find_cap_detection(detections, "spring_cap_robot1", self.left_cap_id)
        right_detection = self._find_cap_detection(detections, "spring_cap_robot2", self.right_cap_id)
        left_overlay_detection = self._find_cap_detection(overlay_detections, "spring_cap_robot1", self.left_cap_id)
        right_overlay_detection = self._find_cap_detection(overlay_detections, "spring_cap_robot2", self.right_cap_id)
        if left_detection is None or right_detection is None:
            summary = _visual_bowing_summary(
                reason="one or both cap markers are missing",
                coordinate_frames=coordinate_frames,
                left_cap_id=self.left_cap_id,
                right_cap_id=self.right_cap_id,
                baseline_px=self._visual_bowing_baseline_px,
                baseline_frames_collected=len(self._visual_bowing_baseline_samples),
            )
            overlay_bowing_result = None
            if left_overlay_detection is not None and right_overlay_detection is not None:
                overlay_bowing_result = compute_w_vis(
                    frame_rgb,
                    np.asarray(left_overlay_detection.center_px, dtype=np.float64),
                    np.asarray(right_overlay_detection.center_px, dtype=np.float64),
                    roi_half_width_px=self.roi_half_width_px,
                    roi_end_margin_px=self.roi_end_margin_px,
                    min_segment_area=self.min_segment_area,
                    morph_kernel_size=self.morph_kernel_size,
                )
            overlay_bowing_result = self._overlay_bowing_result_with_grace(overlay_bowing_result)
            self._set_latest_visual_bowing_overlay(self._draw_visual_bowing_overlay(
                frame_rgb,
                overlay_detections,
                left_overlay_detection,
                right_overlay_detection,
                overlay_bowing_result,
                summary,
                stale_detection_labels=stale_detection_labels,
            ))
            return summary

        p_left = np.asarray(left_detection.center_px, dtype=np.float64)
        p_right = np.asarray(right_detection.center_px, dtype=np.float64)
        bowing_result = compute_w_vis(
            frame_rgb,
            p_left,
            p_right,
            roi_half_width_px=self.roi_half_width_px,
            roi_end_margin_px=self.roi_end_margin_px,
            min_segment_area=self.min_segment_area,
            morph_kernel_size=self.morph_kernel_size,
        )
        try:
            theta_cap_camera_rad = _wrap_angle_rad(self._marker_yaw_rad(right_detection) - self._marker_yaw_rad(left_detection))
        except Exception:
            theta_cap_camera_rad = float("nan")

        left_cap_pose_workspace = self._relative_pose_dict(left_detection, workspace_detection)
        right_cap_pose_workspace = self._relative_pose_dict(right_detection, workspace_detection)
        left_cap_pose_manipulator = self._relative_pose_dict(left_detection, manipulator_robot1_detection)
        right_cap_pose_manipulator = self._relative_pose_dict(right_detection, manipulator_robot2_detection)

        theta_cap_workspace_rad = float("nan")
        theta_cap_rad = theta_cap_camera_rad
        theta_cap_rad_frame = "camera_marker"
        cap_span_workspace_m = float("nan")
        cap_axis_workspace_rad = float("nan")
        if left_cap_pose_workspace is not None and right_cap_pose_workspace is not None:
            try:
                left_transform_workspace = _transform_from_pose_dict(left_cap_pose_workspace)
                right_transform_workspace = _transform_from_pose_dict(right_cap_pose_workspace)
                theta_cap_workspace_rad = _wrap_angle_rad(
                    float(np.radians(_rotation_matrix_to_rpy_deg(right_transform_workspace[:3, :3])[2]))
                    - float(np.radians(_rotation_matrix_to_rpy_deg(left_transform_workspace[:3, :3])[2]))
                )
                cap_delta_workspace = right_transform_workspace[:3, 3] - left_transform_workspace[:3, 3]
                cap_span_workspace_m = float(np.linalg.norm(cap_delta_workspace))
                cap_axis_workspace_rad = float(np.arctan2(cap_delta_workspace[1], cap_delta_workspace[0]))
                theta_cap_rad = theta_cap_workspace_rad
                theta_cap_rad_frame = _frame_id_for_marker_label(WORKSPACE_MARKER_LABEL)
            except Exception:
                theta_cap_workspace_rad = float("nan")

        baseline_px = self._visual_bowing_baseline_px
        w_vis_delta_px = float("nan")
        if bool(bowing_result.get("valid_detection")) and np.isfinite(bowing_result.get("w_vis_raw_px", float("nan"))):
            baseline_px = self._update_visual_bowing_baseline(float(bowing_result["w_vis_raw_px"]))
            if np.isfinite(baseline_px):
                w_vis_delta_px = float(bowing_result["w_vis_raw_px"] - baseline_px)

        spring_box_polygon = bowing_result.get("spring_box_polygon")
        spring_box_polygon_px = []
        if spring_box_polygon is not None:
            spring_box_polygon_px = [
                [float(point[0]), float(point[1])]
                for point in np.asarray(spring_box_polygon, dtype=np.float64).reshape(-1, 2)
            ]

        summary = _visual_bowing_summary(
            reason=str(bowing_result.get("reason", "ok")),
            valid_detection=bool(bowing_result.get("valid_detection")),
            L_px=float(bowing_result.get("L_px", float("nan"))),
            axis_angle_rad=float(bowing_result.get("axis_angle_rad", float("nan"))),
            axis_angle_frame="camera_image_px",
            theta_cap_rad=theta_cap_rad,
            theta_cap_rad_frame=theta_cap_rad_frame,
            theta_cap_camera_rad=theta_cap_camera_rad,
            theta_cap_workspace_rad=theta_cap_workspace_rad,
            w_vis_px=float(bowing_result.get("w_vis_raw_px", float("nan"))),
            w_vis_raw_px=float(bowing_result.get("w_vis_raw_px", float("nan"))),
            w_vis_delta_px=w_vis_delta_px,
            segmentation_frame="camera_image_px",
            baseline_px=baseline_px,
            baseline_frames_collected=len(self._visual_bowing_baseline_samples),
            spring_box_center_offset_px=float(bowing_result.get("spring_box_center_offset_px", float("nan"))),
            spring_box_signed_center_offset_px=float(bowing_result.get("spring_box_signed_center_offset_px", float("nan"))),
            spring_box_half_width_px=float(bowing_result.get("spring_box_half_width_px", float("nan"))),
            spring_box_axial_span_px=float(bowing_result.get("spring_box_axial_span_px", float("nan"))),
            bend_score=float(bowing_result.get("bend_score", float("nan"))),
            bend_threshold_px=float(bowing_result.get("bend_threshold_px", float("nan"))),
            bent=bool(bowing_result.get("bent")),
            spring_box_polygon_px=spring_box_polygon_px,
            spring_box_polygon_frame="camera_image_px",
            cap_span_workspace_m=cap_span_workspace_m,
            cap_axis_workspace_rad=cap_axis_workspace_rad,
            left_cap_pose_workspace=left_cap_pose_workspace,
            right_cap_pose_workspace=right_cap_pose_workspace,
            left_cap_pose_manipulator=left_cap_pose_manipulator,
            right_cap_pose_manipulator=right_cap_pose_manipulator,
            coordinate_frames={
                "workspace": coordinate_frames["workspace"],
                "manipulator_robot1": coordinate_frames["manipulator_robot1"],
                "manipulator_robot2": coordinate_frames["manipulator_robot2"],
            },
            left_cap_id=int(left_detection.spec.marker_id),
            right_cap_id=int(right_detection.spec.marker_id),
        )
        overlay_bowing_result = self._overlay_bowing_result_with_grace(bowing_result)
        self._set_latest_visual_bowing_overlay(self._draw_visual_bowing_overlay(
            frame_rgb,
            overlay_detections,
            left_overlay_detection,
            right_overlay_detection,
            overlay_bowing_result,
            summary,
            stale_detection_labels=stale_detection_labels,
        ))
        return summary

    def analyze_current_frame(
        self,
        require_world: bool = False,
        require_depth: bool = False,
    ) -> Dict[str, Any]:
        refresh_reference_markers = self._should_refresh_reference_markers(require_world=require_world)
        self._analysis_frame_index += 1
        if refresh_reference_markers:
            camera_frame_id, observed_detections, image_width, image_height = self._detect_markers()
            self._refresh_reference_marker_cache(observed_detections)
        else:
            camera_frame_id, observed_detections, image_width, image_height = self._detect_markers(
                tracked_labels=set(self.cap_marker_labels)
            )

        detections = dict(observed_detections)
        cached_marker_labels: List[str] = []
        if not refresh_reference_markers:
            for label, detection in self._reference_marker_cache.items():
                if label in detections:
                    continue
                detections[label] = detection
                cached_marker_labels.append(label)
        cached_marker_labels.sort()
        cached_marker_label_set = set(cached_marker_labels)
        observed_labels = sorted(observed_detections.keys())

        depth_array = None
        if self.enable_depth_processing or require_depth:
            depth_array, depth_status = self._decode_depth_stream(image_width, image_height)
        else:
            depth_status = {
                "available": self.depth_image is not None,
                "camera_info_available": self.depth_camera_info is not None,
                "valid": False,
                "reason": "depth processing disabled for performance",
            }
        coordinate_frames = self._coordinate_frame_summary(detections)

        if require_depth and not depth_status["valid"]:
            raise RuntimeError(depth_status["reason"])

        world_detection = detections.get(self.world_label)
        world_visible = self.world_label in observed_detections
        world_registration_available = world_detection is not None
        if require_world and not world_visible:
            raise RuntimeError("world marker not visible in current frame")
        world_transform_inv = (
            _invert_transform(world_detection.transform_camera_marker())
            if world_detection is not None
            else None
        )

        summary_entries: List[Dict[str, Any]] = []
        workspace_detection = detections.get(WORKSPACE_MARKER_LABEL)
        manipulator_robot1_detection = detections.get(MANIPULATOR_ROBOT1_MARKER_LABEL)
        manipulator_robot2_detection = detections.get(MANIPULATOR_ROBOT2_MARKER_LABEL)
        published_marker_labels = self._published_marker_labels()
        for spec in self.marker_specs:
            if spec.label not in published_marker_labels:
                continue
            detection = detections.get(spec.label)
            if detection is None:
                continue
            held_from_cache = spec.label in cached_marker_label_set

            camera_transform = detection.transform_camera_marker()
            pose_camera = _pose_dict_from_transform(camera_transform)
            pose_world = None
            if world_transform_inv is not None:
                pose_world = _pose_dict_from_transform(world_transform_inv @ camera_transform)

            pose_workspace = self._relative_pose_dict_from_transform(camera_transform, workspace_detection)
            pose_manipulator_robot1 = self._relative_pose_dict_from_transform(camera_transform, manipulator_robot1_detection)
            pose_manipulator_robot2 = self._relative_pose_dict_from_transform(camera_transform, manipulator_robot2_detection)
            poses_relative = {}
            if pose_world is not None:
                poses_relative[self.world_label] = pose_world
            if pose_workspace is not None:
                poses_relative[WORKSPACE_MARKER_LABEL] = pose_workspace
            if pose_manipulator_robot1 is not None:
                poses_relative[MANIPULATOR_ROBOT1_MARKER_LABEL] = pose_manipulator_robot1
            if pose_manipulator_robot2 is not None:
                poses_relative[MANIPULATOR_ROBOT2_MARKER_LABEL] = pose_manipulator_robot2

            depth_center_m = None
            depth_minus_camera_z_m = None
            if depth_array is not None and not held_from_cache:
                depth_center_m = _sample_aligned_depth_m(
                    depth_array,
                    detection.center_px,
                    depth_status["encoding"],
                )
                if depth_center_m is not None:
                    depth_minus_camera_z_m = depth_center_m - pose_camera["translation_m"][2]

            summary_entries.append(
                {
                    "label": spec.label,
                    "marker_id": spec.marker_id,
                    "center_px": [float(detection.center_px[0]), float(detection.center_px[1])],
                    "pose_camera": pose_camera,
                    "pose_world": pose_world,
                    "pose_workspace": pose_workspace,
                    "pose_manipulator_robot1": pose_manipulator_robot1,
                    "pose_manipulator_robot2": pose_manipulator_robot2,
                    "poses_relative": poses_relative,
                    "held_from_cache": bool(held_from_cache),
                    "depth_center_m": depth_center_m,
                    "depth_minus_camera_z_m": depth_minus_camera_z_m,
                }
            )

        summary_entries.sort(key=lambda entry: entry["marker_id"])
        if self.enable_visual_bowing:
            visual_bowing = self._compute_visual_bowing_summary(detections)
        else:
            self._latest_visual_bowing_overlay_rgb = None
            self._overlay_image_age_frames = 0
            visual_bowing = _visual_bowing_summary(
                reason="disabled for performance",
                baseline_px=self._visual_bowing_baseline_px,
                baseline_frames_collected=len(self._visual_bowing_baseline_samples),
                coordinate_frames={
                    "workspace": coordinate_frames["workspace"],
                    "manipulator_robot1": coordinate_frames["manipulator_robot1"],
                    "manipulator_robot2": coordinate_frames["manipulator_robot2"],
                },
                left_cap_id=self.left_cap_id,
                right_cap_id=self.right_cap_id,
            )
        return {
            "valid": True,
            "camera_frame_id": camera_frame_id,
            "world_frame_id": self.world_frame_id if world_visible else None,
            "world_marker_id": self.world_marker_id,
            "world_label": self.world_label,
            "image_width": image_width,
            "image_height": image_height,
            "markers_detected": len(observed_detections),
            "observed_labels": observed_labels,
            "published_detection_labels": sorted(entry["label"] for entry in summary_entries),
            "cached_marker_labels": cached_marker_labels,
            "registration_cache_active": bool(cached_marker_labels),
            "world_marker_visible": bool(world_visible),
            "world_registration_available": bool(world_registration_available),
            "reference_marker_refresh_frames": int(self.reference_marker_refresh_frames),
            "aruco_detector_profile": self.aruco_detector_profile,
            "publish_spring_caps_only": bool(self.publish_spring_caps_only),
            "coordinate_frames": coordinate_frames,
            "depth_stream": depth_status,
            "detections": summary_entries,
            "visual_bowing": visual_bowing,
        }

    def wait_for_frame_bundle(self, timeout_s: float, require_depth: bool = False) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            have_color = self.color_image is not None and self.color_camera_info is not None
            have_depth = self.depth_image is not None and self.depth_camera_info is not None
            if have_color and ((not require_depth) or have_depth):
                return
            rclpy.spin_once(self, timeout_sec=0.05)
        missing = []
        if self.color_image is None:
            missing.append("color image")
        if self.color_camera_info is None:
            missing.append("color camera_info")
        if require_depth and self.depth_image is None:
            missing.append("aligned depth image")
        if require_depth and self.depth_camera_info is None:
            missing.append("aligned depth camera_info")
        available_topics = self._available_topic_names()
        relevant_topics = [
            name for name in available_topics if name.startswith("/camera/") or name.startswith("/spring_monitor/")
        ]
        message = "timed out waiting for frame bundle: " + ", ".join(missing)
        message += ". Expected topics include " + ", ".join(self._expected_frame_topics()) + "."
        if relevant_topics:
            message += " Visible camera/spring topics: " + _format_topic_list(relevant_topics) + "."
        elif available_topics:
            message += " No /camera/* topics are visible right now. Current ROS graph: "
            message += _format_topic_list(available_topics) + "."
        else:
            message += " No ROS topics are visible beyond the base graph."
        message += (
            " Start the RealSense node or launch dual_hardware_variable_stiffness.launch.py "
            "with enable_depth_camera:=true."
        )
        raise TimeoutError(message)

    def capture_one_frame_analysis(self, timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S) -> Dict[str, Any]:
        runtime_error = self._aruco_runtime_error()
        if runtime_error is not None:
            raise RuntimeError(runtime_error)
        self.wait_for_frame_bundle(timeout_s=timeout_s, require_depth=True)
        summary = self.analyze_current_frame(require_world=True, require_depth=True)
        self._append_raw_signal_record(summary, source="capture_one_frame_analysis")
        print(json.dumps(summary, indent=2, sort_keys=False))
        return summary

    def calibration_check_summary(self, timeout_s: float = DEFAULT_CAPTURE_TIMEOUT_S) -> Dict[str, Any]:
        """
        Perform a quick calibration check: wait for a single RGB-D frame bundle,
        analyze detections and return a compact summary including camera intrinsics
        and 4x4 transforms for detected markers (camera->marker and world->marker when available).
        """
        runtime_error = self._aruco_runtime_error()
        if runtime_error is not None:
            raise RuntimeError(runtime_error)
        # Require at least depth+color so transforms include metric depth
        self.wait_for_frame_bundle(timeout_s=timeout_s, require_depth=True)
        summary = self.analyze_current_frame(require_world=False, require_depth=True)

        # Camera intrinsics
        camera_info = self.color_camera_info
        camera_matrix, dist_coeffs = aruco_utils._camera_matrix_from_info(camera_info)
        intrinsics = None
        if camera_matrix is not None:
            intrinsics = {
                "camera_matrix": camera_matrix.tolist(),
                "dist_coeffs": None if dist_coeffs is None else [float(x) for x in dist_coeffs.reshape(-1).tolist()],
                "frame_id": camera_info.header.frame_id if camera_info is not None else None,
            }

        # Compose transforms: for each detection, include camera->marker (4x4)
        detections = summary.get("detections", [])
        transforms = {}
        world_detection = None
        for entry in detections:
            label = entry.get("label")
            # pose_camera is present
            pose_camera = entry.get("pose_camera")
            if pose_camera is not None:
                cam_tf = _transform_from_pose_dict(pose_camera)
                transforms[label] = {
                    "camera_to_marker": cam_tf.tolist(),
                }
            else:
                transforms[label] = {"camera_to_marker": None}
            if entry.get("pose_world") is not None and world_detection is None:
                world_detection = entry

        # If a world marker is visible, include camera->world and world->marker transforms
        if world_detection is not None:
            world_label = world_detection.get("label")
            world_pose = world_detection.get("pose_camera")
            if world_pose is not None:
                world_cam_tf = _transform_from_pose_dict(world_pose)
                # camera -> world = inv(world_cam_tf)
                try:
                    cam_to_world = np.linalg.inv(world_cam_tf)
                except Exception:
                    cam_to_world = None
                if cam_to_world is not None:
                    for label, info in transforms.items():
                        if info.get("camera_to_marker") is not None:
                            cam_to_marker = np.asarray(info["camera_to_marker"], dtype=np.float64)
                            world_to_marker = None
                            try:
                                world_to_marker = cam_to_world @ cam_to_marker
                            except Exception:
                                world_to_marker = None
                            info["world_to_marker"] = None if world_to_marker is None else world_to_marker.tolist()

        out = {
            "ok": True,
            "intrinsics": intrinsics,
            "transforms": transforms,
            "summary": summary,
        }
        # Print a compact JSON to stdout for quick operator consumption
        print(json.dumps(out, indent=2, sort_keys=False))
        return out

    def _publish_detections(self) -> None:
        try:
            summary = self.analyze_current_frame(require_world=False, require_depth=False)
            self._logged_detection_error = False
        except Exception as exc:
            if not self._logged_detection_error:
                self.get_logger().warning(f"camera_aruco detection unavailable: {exc}")
                self._logged_detection_error = True
            self._publish_empty_state(str(exc))
            return

        self._log_marker_visibility_transitions(
            set(summary.get("observed_labels") or [
                entry["label"] for entry in summary["detections"] if not entry.get("held_from_cache")
            ]),
            failure_reason=None,
        )

        stamp = self.get_clock().now().to_msg()
        observed_labels = set(summary.get("observed_labels") or [
            entry["label"] for entry in summary["detections"] if not entry.get("held_from_cache")
        ])
        for spec in self.marker_specs:
            detection_summary = next(
                (entry for entry in summary["detections"] if entry["label"] == spec.label),
                None,
            )
            self._publish_visibility(spec.label, spec.label in observed_labels)
            if detection_summary is None:
                continue

            self.pose_camera_publishers[spec.label].publish(
                _transform_to_pose_message(
                    _transform_from_pose_dict(detection_summary["pose_camera"]),
                    summary["camera_frame_id"],
                    stamp,
                )
            )

            if detection_summary["pose_world"] is not None:
                self.pose_world_publishers[spec.label].publish(
                    _transform_to_pose_message(
                        _transform_from_pose_dict(detection_summary["pose_world"]),
                        self.world_frame_id,
                        stamp,
                    )
                )

        count_msg = Int32()
        count_msg.data = int(summary["markers_detected"])
        self.markers_detected_publisher.publish(count_msg)

        world_visible_msg = Bool()
        world_visible_msg.data = bool(summary["world_marker_visible"])
        self.world_visible_publisher.publish(world_visible_msg)

        if self.enable_visual_bowing:
            bowing_msg = Float64MultiArray()
            bowing_msg.data = _visual_bowing_message_data(summary.get("visual_bowing", _visual_bowing_summary()))
            self.visual_bowing_publisher.publish(bowing_msg)
            self._publish_visual_bowing_overlay()

        summary_msg = String()
        summary_msg.data = json.dumps(summary, separators=(",", ":"))
        self.summary_publisher.publish(summary_msg)
        self._append_raw_signal_record(summary, source="publish_detections")


def _transform_from_pose_dict(pose_dict: Dict[str, Any]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    tx, ty, tz = pose_dict["translation_m"]
    qx, qy, qz, qw = pose_dict["quaternion_xyzw"]
    transform[:3, 3] = [float(tx), float(ty), float(tz)]

    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        return transform
    qx, qy, qz, qw = [component / norm for component in (qx, qy, qz, qw)]
    transform[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return transform


def _format_topic_list(topic_names: List[str], limit: int = 12) -> str:
    if not topic_names:
        return ""
    if len(topic_names) <= limit:
        return ", ".join(topic_names)
    return ", ".join(topic_names[:limit]) + f", ... (+{len(topic_names) - limit} more)"


def _message_stamp_ns(message: Any) -> Optional[int]:
    if message is None:
        return None
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, "sec", 0) or 0)
    nanosec = int(getattr(stamp, "nanosec", 0) or 0)
    return sec * 1_000_000_000 + nanosec


def _post_process_video_file(
    video_path: str,
    camera_info_json_path: str,
    marker_specs: List[MarkerSpec],
    world_marker_id: int,
    dictionary_name: str,
    marker_length_m: float,
    processing_resolution: Optional[Tuple[int, int]] = None,
    reference_marker_refresh_frames: int = 1,
    aruco_detector_profile: str = ARUCO_DETECTOR_PROFILE_DEFAULT,
    publish_spring_caps_only: bool = False,
    enable_visual_bowing: bool = True,
    jsonl_output_path: Optional[str] = None,
    csv_output_path: Optional[str] = None,
    overlay_video_output_path: Optional[str] = None,
    left_cap_id: Optional[int] = None,
    right_cap_id: Optional[int] = None,
    roi_half_width_px: int = ROI_HALF_WIDTH_PX,
    roi_end_margin_px: int = ROI_END_MARGIN_PX,
    baseline_frames: int = BASELINE_FRAMES,
    min_segment_area: int = MIN_SEGMENT_AREA,
    morph_kernel_size: int = MORPH_KERNEL_SIZE,
    overlay_stale_grace_frames: int = DEFAULT_OVERLAY_STALE_GRACE_FRAMES,
) -> Dict[str, Any]:
    if cv2 is None or not hasattr(cv2, "VideoCapture"):
        raise RuntimeError("OpenCV VideoCapture is unavailable in the active interpreter")

    capture = cv2.VideoCapture(str(video_path))
    if capture is None or not capture.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    camera_info_msg = _load_camera_info_json(camera_info_json_path)
    jsonl_path = str(jsonl_output_path or _default_post_process_jsonl_path(video_path))
    csv_path = str(csv_output_path or _default_post_process_csv_path(video_path))
    overlay_path = overlay_video_output_path
    if overlay_path is None and enable_visual_bowing:
        overlay_path = _default_post_process_overlay_video_path(video_path)

    cap_prop_fps = getattr(cv2, "CAP_PROP_FPS", None)
    cap_prop_pos_msec = getattr(cv2, "CAP_PROP_POS_MSEC", None)
    video_fps = 0.0
    if cap_prop_fps is not None:
        try:
            video_fps = float(capture.get(cap_prop_fps) or 0.0)
        except Exception:
            video_fps = 0.0
    fallback_fps = video_fps if video_fps > 0.0 else 30.0

    overlay_writer = None if overlay_path is None else _LazyRgbVideoWriter(str(overlay_path), fallback_fps)
    jsonl_parent = os.path.dirname(os.path.abspath(jsonl_path))
    if jsonl_parent:
        os.makedirs(jsonl_parent, exist_ok=True)
    csv_logger = _SummaryCsvLogger(csv_path, marker_specs)

    rclpy.init()
    tracker = CameraArucoTracker(
        image_topic=DEFAULT_COLOR_IMAGE_TOPIC,
        camera_info_topic=DEFAULT_COLOR_CAMERA_INFO_TOPIC,
        depth_image_topic=DEFAULT_DEPTH_IMAGE_TOPIC,
        depth_camera_info_topic=DEFAULT_DEPTH_CAMERA_INFO_TOPIC,
        marker_specs=marker_specs,
        world_marker_id=world_marker_id,
        dictionary_name=dictionary_name,
        marker_length_m=marker_length_m,
        publish_period_s=max(DEFAULT_PUBLISH_PERIOD_S, 1.0 / fallback_fps),
        enable_visual_bowing=enable_visual_bowing,
        enable_depth_processing=False,
        processing_resolution=processing_resolution,
        reference_marker_refresh_frames=reference_marker_refresh_frames,
        aruco_detector_profile=aruco_detector_profile,
        publish_spring_caps_only=publish_spring_caps_only,
        left_cap_id=left_cap_id,
        right_cap_id=right_cap_id,
        roi_half_width_px=roi_half_width_px,
        roi_end_margin_px=roi_end_margin_px,
        baseline_frames=baseline_frames,
        min_segment_area=min_segment_area,
        morph_kernel_size=morph_kernel_size,
        overlay_stale_grace_frames=overlay_stale_grace_frames,
        log_marker_visibility_transitions=False,
        raw_signals_jsonl_path=None,
    )

    frames_processed = 0
    frames_failed = 0
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                timestamp_ms = 0.0
                if cap_prop_pos_msec is not None:
                    try:
                        timestamp_ms = float(capture.get(cap_prop_pos_msec) or 0.0)
                    except Exception:
                        timestamp_ms = 0.0
                if timestamp_ms <= 0.0 and fallback_fps > 0.0:
                    timestamp_ms = float(frames_processed) * (1000.0 / fallback_fps)

                frame_rgb = np.ascontiguousarray(np.asarray(frame_bgr, dtype=np.uint8)[..., ::-1])
                frame_camera_info = camera_info_msg
                if int(getattr(frame_camera_info, "width", 0) or 0) != int(frame_rgb.shape[1]) or int(getattr(frame_camera_info, "height", 0) or 0) != int(frame_rgb.shape[0]):
                    frame_camera_info = _scale_camera_info_message(frame_camera_info, int(frame_rgb.shape[1]), int(frame_rgb.shape[0]))

                frame_id = str(getattr(getattr(frame_camera_info, "header", None), "frame_id", "video_frame")) or "video_frame"
                tracker.color_image = _make_rgb8_image_message(frame_rgb, frame_id=frame_id, stamp_ns=int(timestamp_ms * 1_000_000.0))
                tracker.color_camera_info = frame_camera_info
                tracker.depth_image = None
                tracker.depth_camera_info = None

                try:
                    summary = tracker.analyze_current_frame(require_world=False, require_depth=False)
                except Exception as exc:
                    frames_failed += 1
                    summary = {
                        "valid": False,
                        "reason": str(exc),
                        "camera_frame_id": frame_id,
                        "world_frame_id": None,
                        "world_marker_id": tracker.world_marker_id,
                        "world_label": tracker.world_label,
                        "image_width": int(frame_rgb.shape[1]),
                        "image_height": int(frame_rgb.shape[0]),
                        "markers_detected": 0,
                        "world_marker_visible": False,
                        "coordinate_frames": tracker._coordinate_frame_summary({}),
                        "depth_stream": {
                            "available": False,
                            "camera_info_available": False,
                            "valid": False,
                            "reason": "offline video has no depth stream",
                        },
                        "detections": [],
                        "visual_bowing": _visual_bowing_summary(
                            reason=str(exc),
                            baseline_px=tracker._visual_bowing_baseline_px,
                            baseline_frames_collected=len(tracker._visual_bowing_baseline_samples),
                            coordinate_frames={
                                "workspace": None,
                                "manipulator_robot1": None,
                                "manipulator_robot2": None,
                            },
                            left_cap_id=tracker.left_cap_id,
                            right_cap_id=tracker.right_cap_id,
                        ),
                    }
                    tracker._latest_visual_bowing_overlay_rgb = None

                record = {
                    "record_type": "camera_aruco_video_frame",
                    "video_path": str(video_path),
                    "frame_index": frames_processed,
                    "video_timestamp_s": float(timestamp_ms / 1000.0),
                    "summary": summary,
                }
                handle.write(json.dumps(record, separators=(",", ":")))
                handle.write("\n")
                csv_logger.append(
                    summary,
                    metadata={
                        "record_type": "camera_aruco_video_frame",
                        "video_path": str(video_path),
                        "frame_index": int(frames_processed),
                        "video_timestamp_s": float(timestamp_ms / 1000.0),
                    },
                )

                if overlay_writer is not None:
                    overlay_frame = tracker._latest_visual_bowing_overlay_rgb
                    overlay_writer.write_rgb_frame(frame_rgb if overlay_frame is None else overlay_frame)

                frames_processed += 1
        finally:
            capture.release()
            if overlay_writer is not None:
                overlay_writer.close()
            tracker.destroy_node()
            rclpy.shutdown()

    return {
        "video_path": str(video_path),
        "camera_info_json": str(camera_info_json_path),
        "jsonl_output_path": jsonl_path,
        "csv_output_path": csv_path,
        "overlay_video_output_path": None if overlay_writer is None else overlay_writer.output_path,
        "overlay_video_codec": None if overlay_writer is None else overlay_writer.codec_name,
        "frames_processed": int(frames_processed),
        "frames_failed": int(frames_failed),
        "video_fps": float(fallback_fps),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track all configured ArUco markers for the dual-OMX spring rig")
    parser.add_argument(
        "--image-topic",
        default=DEFAULT_COLOR_IMAGE_TOPIC,
        help="Color image topic to monitor",
    )
    parser.add_argument(
        "--camera-info-topic",
        default=DEFAULT_COLOR_CAMERA_INFO_TOPIC,
        help="CameraInfo topic paired with the color image",
    )
    parser.add_argument(
        "--depth-image-topic",
        default=DEFAULT_DEPTH_IMAGE_TOPIC,
        help="Aligned depth image topic paired with the color stream",
    )
    parser.add_argument(
        "--depth-camera-info-topic",
        default=DEFAULT_DEPTH_CAMERA_INFO_TOPIC,
        help="CameraInfo topic paired with the aligned depth image",
    )
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_ARUCO_DICTIONARY_NAME,
        help="OpenCV ArUco dictionary name, for example DICT_4X4_50",
    )
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=DEFAULT_MARKER_LENGTH_M,
        help="Physical marker size in meters (default 0.038 for 38 mm markers)",
    )
    parser.add_argument(
        "--world-marker-id",
        type=int,
        default=DEFAULT_WORLD_MARKER_ID,
        help="Marker ID used as the world-reference tag",
    )
    parser.add_argument(
        "--marker",
        action="append",
        type=_parse_marker_arg,
        default=None,
        help="Tracked marker in LABEL:ID format; repeat to override the default six-marker rig map",
    )
    parser.add_argument(
        "--publish-period-s",
        type=float,
        default=DEFAULT_PUBLISH_PERIOD_S,
        help="How often to publish tracked marker poses and JSON summary",
    )
    parser.add_argument(
        "--capture-color-video",
        default=None,
        help="Optional raw color video file to record from the incoming image topic",
    )
    parser.add_argument(
        "--capture-video-fps",
        type=float,
        default=30.0,
        help="Nominal FPS stored in --capture-color-video",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Subscribe and record raw color video without running the live ArUco publish loop",
    )
    parser.add_argument(
        "--post-process-video",
        default=None,
        help="Offline color video to replay through the ArUco tracker",
    )
    parser.add_argument(
        "--post-process-camera-info-json",
        default=None,
        help="Camera-info JSON used with --post-process-video; defaults to VIDEO.camera_info.json",
    )
    parser.add_argument(
        "--post-process-jsonl",
        default=None,
        help="Optional JSONL output path for per-frame summaries during --post-process-video",
    )
    parser.add_argument(
        "--post-process-csv",
        default=None,
        help="Optional CSV output path for flat per-frame ArUco variables during --post-process-video; defaults to VIDEO basename + .aruco.csv",
    )
    parser.add_argument(
        "--post-process-overlay-video",
        default=None,
        help="Optional overlay video path for --post-process-video; defaults to VIDEO basename + .aruco_overlay.mp4 when overlays are enabled",
    )
    parser.add_argument(
        "--processing-resolution",
        type=_parse_processing_resolution,
        default=None,
        help="Optional WIDTHxHEIGHT processing resolution for live detection and overlay generation, for example 640x480",
    )
    parser.add_argument(
        "--reference-marker-refresh-frames",
        type=int,
        default=1,
        help=(
            "When >1, pose-estimate only the spring-cap markers on intermediate frames and "
            "reuse the most recent non-cap marker poses for registration until the next full refresh"
        ),
    )
    parser.add_argument(
        "--aruco-detector-profile",
        choices=ARUCO_DETECTOR_PROFILE_CHOICES,
        default=ARUCO_DETECTOR_PROFILE_DEFAULT,
        help="OpenCV ArUco detector tuning profile; 'fast' uses a speed-tuned legacy parameter preset",
    )
    parser.add_argument(
        "--publish-spring-caps-only",
        action="store_true",
        help="Publish only spring-cap pose entries and pose topics while still using non-cap markers internally for registration",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Preset for higher-rate live pose publishing; equivalent to --disable-visual-bowing --disable-depth-processing",
    )
    parser.add_argument(
        "--disable-visual-bowing",
        action="store_true",
        help="Skip spring bowing summary and overlay generation to reduce per-frame CPU cost",
    )
    parser.add_argument(
        "--disable-depth-processing",
        action="store_true",
        help="Skip aligned-depth decoding during live publishing to reduce per-frame CPU cost; --print-once still forces depth validation",
    )
    parser.add_argument(
        "--print-once",
        action="store_true",
        help="Wait for one RGBD frame, verify the depth stream, and print current world-frame marker poses",
    )
    parser.add_argument(
        "--calibration-check",
        action="store_true",
        help=(
            "Perform a one-shot calibration check: capture one RGBD frame and print "
            "camera intrinsics plus 4x4 camera->marker and world->marker transforms"
        ),
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Print interpreter, ArUco runtime, and expected camera topic readiness without waiting for frames",
    )
    parser.add_argument(
        "--capture-timeout-s",
        type=float,
        default=DEFAULT_CAPTURE_TIMEOUT_S,
        help="Timeout for --print-once while waiting for color/depth frames",
    )
    parser.add_argument(
        "--left-cap-id",
        type=int,
        default=None,
        help="Marker ID used for the left spring cap in the visual bowing proxy (defaults to spring_cap_robot1 from --marker)",
    )
    parser.add_argument(
        "--right-cap-id",
        type=int,
        default=None,
        help="Marker ID used for the right spring cap in the visual bowing proxy (defaults to spring_cap_robot2 from --marker)",
    )
    parser.add_argument(
        "--roi-half-width-px",
        type=int,
        default=ROI_HALF_WIDTH_PX,
        help="Half-width of the oriented bowing ROI in pixels",
    )
    parser.add_argument(
        "--roi-end-margin-px",
        type=int,
        default=ROI_END_MARGIN_PX,
        help="Pixels trimmed from each cap end before building the bowing ROI",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=BASELINE_FRAMES,
        help="Number of initial valid bowing frames used to estimate the baseline median",
    )
    parser.add_argument(
        "--min-segment-area",
        type=int,
        default=MIN_SEGMENT_AREA,
        help="Minimum connected spring segment area retained inside the bowing ROI",
    )
    parser.add_argument(
        "--morph-kernel-size",
        type=int,
        default=MORPH_KERNEL_SIZE,
        help="Morphology kernel size used during spring segmentation inside the bowing ROI",
    )
    parser.add_argument(
        "--overlay-stale-grace-frames",
        type=int,
        default=DEFAULT_OVERLAY_STALE_GRACE_FRAMES,
        help="How many publish cycles the overlay may reuse the last marker labels and spring box after a missed detection",
    )
    parser.add_argument(
        "--log-marker-visibility-transitions",
        action="store_true",
        help="Log when configured marker IDs first go missing or recover in the publish stream",
    )
    parser.add_argument(
        "--raw-signals-jsonl",
        default=None,
        help="Optional JSONL file path that appends one raw per-frame summary record, including timestamps and failure summaries",
    )
    parser.add_argument(
        "--raw-signals-csv",
        default=None,
        help="Optional CSV file path that appends flat per-frame ArUco variables alongside the raw summary stream",
    )
    return parser.parse_args()


def _resolve_live_processing_flags(args: argparse.Namespace) -> Tuple[bool, bool]:
    disable_visual_bowing = bool(getattr(args, "disable_visual_bowing", False)) or bool(getattr(args, "fast", False))
    disable_depth_processing = bool(getattr(args, "disable_depth_processing", False)) or bool(getattr(args, "fast", False))
    return (not disable_visual_bowing, not disable_depth_processing)


def main() -> int:
    args = _parse_args()
    marker_specs = _resolve_marker_specs(args.marker, args.world_marker_id)
    enable_visual_bowing, enable_depth_processing = _resolve_live_processing_flags(args)

    if args.capture_only and args.capture_color_video is None:
        raise SystemExit("--capture-only requires --capture-color-video")
    if args.capture_only and (args.self_check or args.print_once or args.post_process_video or args.calibration_check):
        raise SystemExit("--capture-only cannot be combined with --self-check, --print-once, --calibration-check, or --post-process-video")
    if args.post_process_video is not None and args.capture_color_video is not None:
        raise SystemExit("--post-process-video cannot be combined with --capture-color-video")
    if args.post_process_video is not None:
        camera_info_json_path = str(args.post_process_camera_info_json or _camera_info_json_path_for_video(args.post_process_video))
        result = _post_process_video_file(
            video_path=str(args.post_process_video),
            camera_info_json_path=camera_info_json_path,
            marker_specs=marker_specs,
            world_marker_id=args.world_marker_id,
            dictionary_name=args.dictionary,
            marker_length_m=args.marker_length_m,
            processing_resolution=args.processing_resolution,
            reference_marker_refresh_frames=args.reference_marker_refresh_frames,
            aruco_detector_profile=args.aruco_detector_profile,
            publish_spring_caps_only=args.publish_spring_caps_only,
            enable_visual_bowing=enable_visual_bowing,
            jsonl_output_path=args.post_process_jsonl,
            csv_output_path=args.post_process_csv,
            overlay_video_output_path=args.post_process_overlay_video,
            left_cap_id=args.left_cap_id,
            right_cap_id=args.right_cap_id,
            roi_half_width_px=args.roi_half_width_px,
            roi_end_margin_px=args.roi_end_margin_px,
            baseline_frames=args.baseline_frames,
            min_segment_area=args.min_segment_area,
            morph_kernel_size=args.morph_kernel_size,
            overlay_stale_grace_frames=args.overlay_stale_grace_frames,
        )
        print(json.dumps(result, indent=2, sort_keys=False))
        return 0

    rclpy.init()
    node = CameraArucoTracker(
        image_topic=args.image_topic,
        camera_info_topic=args.camera_info_topic,
        depth_image_topic=args.depth_image_topic,
        depth_camera_info_topic=args.depth_camera_info_topic,
        marker_specs=marker_specs,
        world_marker_id=args.world_marker_id,
        dictionary_name=args.dictionary,
        marker_length_m=args.marker_length_m,
        publish_period_s=args.publish_period_s,
        enable_visual_bowing=enable_visual_bowing,
        enable_depth_processing=enable_depth_processing,
        processing_resolution=args.processing_resolution,
        reference_marker_refresh_frames=args.reference_marker_refresh_frames,
        aruco_detector_profile=args.aruco_detector_profile,
        publish_spring_caps_only=args.publish_spring_caps_only,
        capture_color_video_path=args.capture_color_video,
        capture_video_fps=args.capture_video_fps,
        capture_only=args.capture_only,
        left_cap_id=args.left_cap_id,
        right_cap_id=args.right_cap_id,
        roi_half_width_px=args.roi_half_width_px,
        roi_end_margin_px=args.roi_end_margin_px,
        baseline_frames=args.baseline_frames,
        min_segment_area=args.min_segment_area,
        morph_kernel_size=args.morph_kernel_size,
        overlay_stale_grace_frames=args.overlay_stale_grace_frames,
        log_marker_visibility_transitions=args.log_marker_visibility_transitions,
        raw_signals_jsonl_path=args.raw_signals_jsonl,
        raw_signals_csv_path=args.raw_signals_csv,
    )
    try:
        if args.self_check:
            summary = node.self_check_summary()
            print(json.dumps(summary, indent=2, sort_keys=False))
            return 0 if summary["ok"] else 1
        if args.print_once:
            node.capture_one_frame_analysis(timeout_s=args.capture_timeout_s)
            return 0
        if args.calibration_check:
            node.calibration_check_summary(timeout_s=args.capture_timeout_s)
            return 0
        rclpy.spin(node)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())