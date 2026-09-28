#!/usr/bin/env python3
"""Compute safe start_position from ArUco metal platform marker detection.

Takes a single snapshot from the depth camera, detects the metal platform
ArUco marker (ID=0), and computes a safe ``start_position`` that places each
end-effector outside the metallic platform boundary.

Camera-to-world extrinsics are loaded from ``camera_extrinsics.json`` (produced
by :mod:`aruco_fit_extrinsics`).  When the extrinsics file is absent, the tool
falls back to the **known SDF geometry**: the metal platform is 0.30×0.30 m
centred at world origin and the marker sits at (-0.12, -0.12, 0.0505) in the
model frame.

Output is JSON with ``robot1`` and ``robot2`` start positions in robot-local
coordinates (from each robot's base link).

Usage::

    # With camera extrinsics (hardware)
    python3 tools/compute_start_position_from_aruco.py \\
        --camera-extrinsics logs/aruco_hw_run_host/camera_extrinsics.json

    # Fallback to known geometry (simulation / no extrinsics)
    python3 tools/compute_start_position_from_aruco.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# ── Platform geometry (from SDF) ──────────────────────────────────────────
METAL_PLATFORM_SIZE_M = 0.30          # 0.30 × 0.30 m platform
ARUCO_MARKER_LENGTH_M = 0.05          # 5 cm marker size
ARUCO_MARKER_ID_METAL = 0             # Marker on metal platform

# Marker position relative to platform centre (from SDF)
MARKER_OFFSET_IN_PLATFORM_FRAME = (-0.12, -0.12, 0.0505)
# Platform spans ±0.15 m from centre, marker is 3 cm from the left/bottom edge
MARKER_TO_LEFT_EDGE_X = 0.03   # platform_left_x = marker_x - 0.03
MARKER_TO_RIGHT_EDGE_X = 0.27  # platform_right_x = marker_x + 0.27

# ── Robot geometry (from world file) ──────────────────────────────────────
ROBOT1_BASE_X = -0.39  # metres
ROBOT2_BASE_X = 0.39   # metres
BASE_HALF_SPACING = 0.39

# ── Safety / defaults ─────────────────────────────────────────────────────
SAFETY_MARGIN_M = 0.02  # 2 cm outside platform edge
DEFAULT_START_Z = 0.08  # Spring cap center height (platform at 0.05 + half cap height 0.03 = 0.08m)
DEFAULT_START_Y = 0.0

# ── Camera topics ─────────────────────────────────────────────────────────
DEFAULT_COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_COLOR_CAMERA_INFO_TOPIC = "/camera/color/camera_info"


# ─────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────

def platform_edges_from_marker(marker_world_x: float) -> Tuple[float, float]:
    """Return (left_edge_x, right_edge_x) of the platform from marker world x."""
    left = marker_world_x - MARKER_TO_LEFT_EDGE_X
    right = marker_world_x + MARKER_TO_RIGHT_EDGE_X
    return left, right


def safe_start_positions(
    platform_left_x: float,
    platform_right_x: float,
    safety_margin: float = SAFETY_MARGIN_M,
    start_z: float = DEFAULT_START_Z,
    start_y: float = DEFAULT_START_Y,
) -> Dict[str, list]:
    """Compute safe start_position [x, y, z] and retract_x for both robots.
    
    Returns dict with 'robot1', 'robot2' (start positions) and 'robot1_retract_x', 'robot2_retract_x'.
    
    Robot1 (base at -0.39): EE must be at world_x < platform_left_x - margin
    Robot2 (base at +0.39): EE must be at world_x > platform_right_x + margin
    
    Retract X is closer to base to avoid plate during lift phase.
    """
    # Robot1: world_x = base_x + local_x  →  local_x = world_x - base_x
    r1_world_x = platform_left_x - safety_margin
    r1_local_x = r1_world_x - ROBOT1_BASE_X  # e.g. -0.17 - (-0.39) = 0.22
    # Retract X must stay at/above MIN_X (0.16) so the homing Phase A never
    # drives the arm backward into the base/singularity zone. 0.18 leaves a
    # 2 cm safety margin above the floor (matches the controller clamp + YAML).
    r1_retract_x = 0.18

    # Robot2: world_x = base_x - local_x  →  local_x = base_x - world_x
    r2_world_x = platform_right_x + safety_margin
    r2_local_x = ROBOT2_BASE_X - r2_world_x  # e.g. 0.39 - 0.17 = 0.22
    r2_retract_x = 0.18

    return {
        "robot1": [round(r1_local_x, 4), round(start_y, 4), round(start_z, 4)],
        "robot2": [round(r2_local_x, 4), round(start_y, 4), round(start_z, 4)],
        "robot1_retract_x": r1_retract_x,
        "robot2_retract_x": r2_retract_x,
        "_meta": {
            "platform_left_x": round(platform_left_x, 4),
            "platform_right_x": round(platform_right_x, 4),
            "platform_center_y": round(start_y, 4),
            "robot1_world_x": round(r1_world_x, 4),
            "robot2_world_x": round(r2_world_x, 4),
            "safety_margin": safety_margin,
            "robot1_base_x": ROBOT1_BASE_X,
            "robot2_base_x": ROBOT2_BASE_X,
        },
    }


def start_from_known_geometry(
    safety_margin: float = SAFETY_MARGIN_M,
    start_z: float = DEFAULT_START_Z,
) -> Dict[str, list]:
    """Compute start_position using the known SDF geometry (no camera needed)."""
    # Marker is at (-0.12, -0.12) in platform frame → world x = -0.12
    marker_x = MARKER_OFFSET_IN_PLATFORM_FRAME[0]
    left, right = platform_edges_from_marker(marker_x)
    return safe_start_positions(left, right, safety_margin, start_z, start_y=DEFAULT_START_Y)


# ─────────────────────────────────────────────────────────────────────────
#  Camera-based detection
# ─────────────────────────────────────────────────────────────────────────

def load_camera_extrinsics(path: str) -> Optional[np.ndarray]:
    """Load the 4×4 camera-to-world matrix from a JSON file."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    mat = data.get("camera_to_world")
    if mat is None:
        return None
    return np.asarray(mat, dtype=float).reshape(4, 4)


def detect_marker_in_image(
    image_msg, camera_info_msg, marker_length: float, marker_id: int
) -> Optional[np.ndarray]:
    """Detect ArUco marker in a ROS Image message.  Returns tvec in camera frame."""
    try:
        import cv2
        from cv_bridge import CvBridge
    except ImportError:
        print("ERROR: cv2 / cv_bridge not available", file=sys.stderr)
        return None

    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    except Exception as exc:
        print(f"ERROR: failed to convert image: {exc}", file=sys.stderr)
        return None

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    if not hasattr(cv2, "aruco"):
        print("ERROR: OpenCV ArUco module not available", file=sys.stderr)
        return None

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is None:
        return None

    ids_flat = [int(v) for v in np.asarray(ids).reshape(-1)]
    if marker_id not in ids_flat:
        return None

    idx = ids_flat.index(marker_id)
    camera_matrix = np.asarray(camera_info_msg.k, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.asarray(camera_info_msg.d or [0.0] * 5, dtype=np.float64)

    _, rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(
        corners[idx:idx + 1], marker_length, camera_matrix, dist_coeffs
    )
    return np.asarray(tvecs[0][0], dtype=float)


def marker_camera_to_world(
    tvec_cam: np.ndarray, T_cam_to_world: np.ndarray
) -> np.ndarray:
    """Transform a point from camera frame to world frame."""
    p_cam = np.array([tvec_cam[0], tvec_cam[1], tvec_cam[2], 1.0])
    p_world = T_cam_to_world @ p_cam
    return p_world[:3]


def read_last_detection_from_jsonl(path: str, marker_id: int) -> Optional[np.ndarray]:
    """Read the most recent camera-frame tvec for ``marker_id`` from a UDP
    calibration JSONL (produced by ``udp_aruco_to_calibration.py``).

    Each line is ``{"ts_ns": int, "detections": [{"marker_id", "rvec", "tvec"}, ...]}``.
    Returns the tvec (3,) in the camera frame, or None if not found.
    """
    if not os.path.isfile(path):
        return None
    last: Optional[np.ndarray] = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for det in rec.get("detections", []):
                    if int(det.get("marker_id")) != marker_id:
                        continue
                    tvec = det.get("tvec")
                    if isinstance(tvec, (list, tuple)) and len(tvec) >= 3:
                        last = np.asarray(tvec, dtype=float)[:3]
    except OSError:
        return None
    return last


# ─────────────────────────────────────────────────────────────────────────
#  ROS-based snapshot acquisition
# ─────────────────────────────────────────────────────────────────────────

class SnapshotNode:
    """Minimal ROS2 node that grabs a single image + camera_info pair."""

    def __init__(self, image_topic: str, camera_info_topic: str, timeout: float = 5.0):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image

        rclpy.init(args=[])
        self._node = Node("aruco_start_position_snapshot")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._image = None
        self._camera_info = None
        self._node.create_subscription(Image, image_topic, self._cb_image, qos)
        self._node.create_subscription(CameraInfo, camera_info_topic, self._cb_camera_info, qos)

        self._timeout = timeout

    def _cb_image(self, msg):
        self._image = msg

    def _cb_camera_info(self, msg):
        self._camera_info = msg

    def acquire(self):
        """Spin until we have both image and camera_info, or timeout."""
        import rclpy
        t0 = time.time()
        while (self._image is None or self._camera_info is None) and (time.time() - t0 < self._timeout):
            rclpy.spin_once(self._node, timeout_sec=0.1)
        return self._image, self._camera_info

    def destroy(self):
        self._node.destroy_node()
        import rclpy
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute safe start_position from ArUco marker detection or known geometry"
    )
    ap.add_argument(
        "--camera-extrinsics",
        default=os.environ.get("OMX_CAMERA_EXTRINSICS", ""),
        help="Path to camera_extrinsics.json (camera→world 4×4 matrix)",
    )
    ap.add_argument("--image-topic", default=DEFAULT_COLOR_IMAGE_TOPIC)
    ap.add_argument("--camera-info-topic", default=DEFAULT_COLOR_CAMERA_INFO_TOPIC)
    ap.add_argument("--marker-length", type=float, default=ARUCO_MARKER_LENGTH_M)
    ap.add_argument("--marker-id", type=int, default=ARUCO_MARKER_ID_METAL)
    ap.add_argument("--safety-margin", type=float, default=SAFETY_MARGIN_M)
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z)
    ap.add_argument("--timeout", type=float, default=5.0, help="Seconds to wait for camera topics")
    ap.add_argument("--output", default="", help="Write JSON result to this file path")
    ap.add_argument(
        "--force-geometry-fallback", action="store_true",
        help="Skip camera detection; use known SDF geometry instead",
    )
    ap.add_argument(
        "--detections-jsonl",
        default="",
        help="Image-free mode: read the latest marker tvec (camera frame) from a "
             "UDP calibration JSONL (tools/udp_aruco_to_calibration.py output) "
             "instead of subscribing to a camera image. Requires --camera-extrinsics.",
    )
    args = ap.parse_args()

    result: Dict
    method = "known_geometry"

    # Try camera-based detection if extrinsics are available
    T_cam_to_world = None
    if args.camera_extrinsics and not args.force_geometry_fallback:
        T_cam_to_world = load_camera_extrinsics(args.camera_extrinsics)
        if T_cam_to_world is None:
            print(f"WARNING: Could not load extrinsics from {args.camera_extrinsics}", file=sys.stderr)

    if T_cam_to_world is not None and not args.force_geometry_fallback:
        # Image-free mode: read the latest marker tvec from a UDP calibration
        # JSONL (no camera image subscription needed).
        if args.detections_jsonl:
            tvec_cam = read_last_detection_from_jsonl(args.detections_jsonl, args.marker_id)
            if tvec_cam is not None:
                tvec_world = marker_camera_to_world(tvec_cam, T_cam_to_world)
                left, right = platform_edges_from_marker(tvec_world[0])
                platform_center_y = float(tvec_world[1] - MARKER_OFFSET_IN_PLATFORM_FRAME[1])
                result = safe_start_positions(
                    left,
                    right,
                    args.safety_margin,
                    args.start_z,
                    start_y=platform_center_y,
                )
                method = "aruco_detection_jsonl"
                result["_meta"]["marker_world_x"] = round(float(tvec_world[0]), 6)
                result["_meta"]["marker_world_y"] = round(float(tvec_world[1]), 6)
                result["_meta"]["marker_world_z"] = round(float(tvec_world[2]), 6)
                result["_meta"]["platform_center_y"] = round(platform_center_y, 6)
                print(
                    f"[OK] Detected marker {args.marker_id} (image-free JSONL) at world "
                    f"({tvec_world[0]:.4f}, {tvec_world[1]:.4f}, {tvec_world[2]:.4f})",
                    file=sys.stderr,
                )
            else:
                print(
                    f"WARNING: Marker {args.marker_id} not found in {args.detections_jsonl}; "
                    f"falling back to known geometry",
                    file=sys.stderr,
                )
                result = start_from_known_geometry(args.safety_margin, args.start_z)
        else:
            # Camera-based detection
            try:
                snap = SnapshotNode(args.image_topic, args.camera_info_topic, args.timeout)
                image, camera_info = snap.acquire()
                snap.destroy()

                if image is not None and camera_info is not None:
                    tvec_cam = detect_marker_in_image(image, camera_info, args.marker_length, args.marker_id)
                    if tvec_cam is not None:
                        tvec_world = marker_camera_to_world(tvec_cam, T_cam_to_world)
                        left, right = platform_edges_from_marker(tvec_world[0])
                        platform_center_y = float(tvec_world[1] - MARKER_OFFSET_IN_PLATFORM_FRAME[1])
                        result = safe_start_positions(
                            left,
                            right,
                            args.safety_margin,
                            args.start_z,
                            start_y=platform_center_y,
                        )
                        method = "aruco_detection"
                        # Store detected marker world position for SDF update
                        result["_meta"]["marker_world_x"] = round(float(tvec_world[0]), 6)
                        result["_meta"]["marker_world_y"] = round(float(tvec_world[1]), 6)
                        result["_meta"]["marker_world_z"] = round(float(tvec_world[2]), 6)
                        result["_meta"]["platform_center_y"] = round(platform_center_y, 6)
                        print(
                            f"[OK] Detected marker {args.marker_id} at world "
                            f"({tvec_world[0]:.4f}, {tvec_world[1]:.4f}, {tvec_world[2]:.4f})",
                            file=sys.stderr,
                        )
                    else:
                        print(f"WARNING: Marker {args.marker_id} not detected; falling back to known geometry", file=sys.stderr)
                        result = start_from_known_geometry(args.safety_margin, args.start_z)
                else:
                    print("WARNING: No camera data received; falling back to known geometry", file=sys.stderr)
                    result = start_from_known_geometry(args.safety_margin, args.start_z)
            except Exception as exc:
                print(f"WARNING: Camera detection failed ({exc}); falling back to known geometry", file=sys.stderr)
                result = start_from_known_geometry(args.safety_margin, args.start_z)
    else:
        # Known geometry fallback
        result = start_from_known_geometry(args.safety_margin, args.start_z)

    # Add metadata
    result["_meta"]["method"] = method
    result["_meta"]["safety_margin"] = args.safety_margin
    result["_meta"]["start_z"] = args.start_z
    
    # If using known geometry, also store the expected marker position for consistency
    if method == "known_geometry":
        result["_meta"]["marker_world_x"] = float(MARKER_OFFSET_IN_PLATFORM_FRAME[0])
        result["_meta"]["marker_world_y"] = float(MARKER_OFFSET_IN_PLATFORM_FRAME[1])
        result["_meta"]["marker_world_z"] = float(MARKER_OFFSET_IN_PLATFORM_FRAME[2])
        result["_meta"]["platform_center_y"] = 0.0

    output_json = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output_json + "\n")
        print(f"Wrote result to {args.output}", file=sys.stderr)

    print(output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
