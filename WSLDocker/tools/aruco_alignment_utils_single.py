#!/usr/bin/env python3
"""
Single-Arm ArUco Alignment Utilities
======================================

ArUco detection and alignment computation for the **single-robot + wall**
configuration.  This is the single-arm counterpart to
``aruco_alignment_utils.py`` (which is dual-arm only).

Key differences from the dual-arm version
------------------------------------------
- Only one robot marker (no robot2 fields)
- No roll computation from a marker pair (roll requires two markers)
- No ``evaluate_fallback_alignment()`` (psi_left / psi_right are dual-arm
  spring angles that do not exist in the single-arm+wall setup)
- Topic names omit the "robot1" qualifier (e.g.
  ``/single_arm/aruco_alignment/recommended_z_trim_m`` instead of
  ``/spring_monitor/aruco_alignment/recommended_robot1_z_trim_m``)
- Shared helpers are imported from ``aruco_alignment_utils.py`` so there
  is no code duplication for OpenCV setup, detector profiles, etc.

Consumers
----------
- ``single_arm_aruco_visual.py``  — live camera overlay (imports helpers
  directly; may optionally use ``detect_single_arm_aruco_alignment``)
- ``hardware_harness_single_arm_v3.py`` — subscribes to the single-arm
  alignment topics for z-trim guidance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

# Re-export scope-agnostic helpers from the dual-arm module so that
# single-arm consumers can import everything from this single file.
try:
    from tools.aruco_alignment_utils import (  # noqa: F401 – re-exported
        ARUCO_DETECTOR_PROFILE_CHOICES,
        ARUCO_DETECTOR_PROFILE_DEFAULT,
        ARUCO_DETECTOR_PROFILE_FAST,
        _aruco_available,
        _camera_matrix_from_info,
        _configure_detector_parameters,
        _finite,
        _get_aruco_dictionary,
        _get_detector_parameters,
        _gray_image,
        _normalize_detector_profile,
    )
except ImportError:
    from aruco_alignment_utils import (  # noqa: F401 – re-exported
        ARUCO_DETECTOR_PROFILE_CHOICES,
        ARUCO_DETECTOR_PROFILE_DEFAULT,
        ARUCO_DETECTOR_PROFILE_FAST,
        _aruco_available,
        _camera_matrix_from_info,
        _configure_detector_parameters,
        _finite,
        _get_aruco_dictionary,
        _get_detector_parameters,
        _gray_image,
        _normalize_detector_profile,
    )

# ── Single-arm topic constants ──────────────────────────────────────────────
# These use a separate namespace (/single_arm/) so they never clash with the
# dual-arm topics (/spring_monitor/) when both nodes run on the same ROS
# graph (e.g. during mixed testing).

ARUCO_ALIGNMENT_VALID_TOPIC = "/single_arm/aruco_alignment/valid"
ARUCO_ALIGNMENT_Z_TRIM_TOPIC = "/single_arm/aruco_alignment/recommended_z_trim_m"
ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC = "/single_arm/aruco_alignment/center_y_error_m"
ARUCO_ALIGNMENT_MARKERS_DETECTED_TOPIC = "/single_arm/aruco_alignment/markers_detected"


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SingleArmArucoAlignmentResult:
    """Alignment result for the single-arm (one robot + wall) configuration.

    Compared to the dual-arm ``ArucoAlignmentResult``:
      - No ``robot2_marker_seen`` / ``recommended_robot2_z_trim_m``
      - No ``roll_deg`` (roll requires two markers)
      - ``marker_seen`` replaces ``robot1_marker_seen``
      - ``recommended_z_trim_m`` replaces ``recommended_robot1_z_trim_m``
    """

    valid: bool
    markers_detected: int = 0
    marker_ids: List[int] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    marker_seen: bool = False
    recommended_z_trim_m: float = float("nan")
    center_y_error_m: float = float("nan")
    message: str = ""


# ── Core detection function ─────────────────────────────────────────────────

def detect_single_arm_aruco_alignment(
    image_msg: Any,
    camera_info_msg: Any,
    marker_id: int,
    marker_length_m: float,
    dictionary_name: str,
    target_row_fraction: float = 0.5,
    metal_platform_marker_id: Optional[int] = None,
    detector_profile: str = ARUCO_DETECTOR_PROFILE_DEFAULT,
) -> SingleArmArucoAlignmentResult:
    """Detect a single ArUco marker and compute the z-trim for one robot.

    Parameters
    ----------
    image_msg
        ROS2 ``Image`` message (color or mono).
    camera_info_msg
        ROS2 ``CameraInfo`` message with intrinsics.
    marker_id
        The ArUco marker ID attached to the robot's spring cap.
    marker_length_m
        Side length of the ArUco marker in metres.
    dictionary_name
        OpenCV ArUco dictionary name (e.g. ``"DICT_4X4_50"``).
    target_row_fraction
        Fraction of image height used as the target row when the metal
        platform marker is not visible (default 0.5 = centre).
    metal_platform_marker_id
        Optional marker ID on the metal platform; when visible the
        effective target row is the midpoint between the cap and the
        platform (same logic as the dual-arm version).
    detector_profile
        ``"default"`` or ``"fast"`` — see ``aruco_alignment_utils``.

    Returns
    -------
    SingleArmArucoAlignmentResult
        Alignment result with ``recommended_z_trim_m`` for the single
        robot.  ``marker_seen`` is ``True`` only when the target marker
        was detected with a valid pose.
    """
    if not _aruco_available():
        return SingleArmArucoAlignmentResult(
            valid=False, message="OpenCV ArUco support is unavailable"
        )
    if image_msg is None or camera_info_msg is None:
        return SingleArmArucoAlignmentResult(
            valid=False, message="image or camera_info unavailable"
        )

    try:
        gray = _gray_image(image_msg)
        dictionary = _get_aruco_dictionary(dictionary_name)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=_get_detector_parameters(detector_profile),
        )
    except Exception as exc:
        return SingleArmArucoAlignmentResult(
            valid=False, message=f"ArUco detection failed: {exc}"
        )

    if ids is None or len(ids) == 0:
        return SingleArmArucoAlignmentResult(
            valid=False,
            image_width=int(gray.shape[1]),
            image_height=int(gray.shape[0]),
            message="no ArUco markers detected",
        )

    camera_matrix, dist_coeffs = _camera_matrix_from_info(camera_info_msg)
    if camera_matrix is None:
        return SingleArmArucoAlignmentResult(
            valid=False,
            markers_detected=len(ids),
            marker_ids=[int(v) for v in np.asarray(ids).reshape(-1)],
            image_width=int(gray.shape[1]),
            image_height=int(gray.shape[0]),
            message="camera_info is missing intrinsics",
        )

    try:
        _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            float(marker_length_m),
            camera_matrix,
            dist_coeffs,
        )
    except Exception as exc:
        return SingleArmArucoAlignmentResult(
            valid=False,
            markers_detected=len(ids),
            marker_ids=[int(v) for v in np.asarray(ids).reshape(-1)],
            image_width=int(gray.shape[1]),
            image_height=int(gray.shape[0]),
            message=f"pose estimation failed: {exc}",
        )

    ids_flat = [int(v) for v in np.asarray(ids).reshape(-1)]
    fy = float(camera_matrix[1, 1])
    target_row_px = float(gray.shape[0]) * float(target_row_fraction)

    # First pass: locate the metal platform marker row (for midpoint z reference).
    platform_row_px: Optional[float] = None
    if metal_platform_marker_id is not None:
        platform_id = int(metal_platform_marker_id)
        for marker_corners, mid, marker_tvec in zip(corners, ids_flat, np.asarray(tvecs)):
            if mid != platform_id:
                continue
            platform_center_xy = np.mean(
                np.asarray(marker_corners, dtype=np.float64).reshape(-1, 2), axis=0
            )
            platform_depth_m = float(np.asarray(marker_tvec, dtype=np.float64).reshape(-1)[2])
            if _finite(platform_depth_m) and platform_depth_m > 0.0:
                platform_row_px = float(platform_center_xy[1])
            break

    # Second pass: find the robot's spring-cap marker.
    marker_seen = False
    trim_m = float("nan")
    metric_error = float("nan")

    for marker_corners, mid, marker_tvec in zip(corners, ids_flat, np.asarray(tvecs)):
        if mid != int(marker_id):
            continue
        center_xy = np.mean(
            np.asarray(marker_corners, dtype=np.float64).reshape(-1, 2), axis=0
        )
        marker_depth_m = float(np.asarray(marker_tvec, dtype=np.float64).reshape(-1)[2])
        if not (_finite(marker_depth_m) and marker_depth_m > 0.0 and _finite(fy) and fy > 0.0):
            continue
        cap_row = float(center_xy[1])
        # Use midpoint between spring cap row and metal platform row as z
        # reference when the platform marker is visible; fall back to the
        # fixed target_row_px otherwise.
        if platform_row_px is not None:
            effective_target_row = (cap_row + platform_row_px) / 2.0
        else:
            effective_target_row = target_row_px
        pixel_error = cap_row - effective_target_row
        metric_error = pixel_error * marker_depth_m / fy
        trim_m = -metric_error
        marker_seen = True
        break  # only one marker of interest

    if not marker_seen:
        return SingleArmArucoAlignmentResult(
            valid=False,
            markers_detected=len(ids_flat),
            marker_ids=ids_flat,
            image_width=int(gray.shape[1]),
            image_height=int(gray.shape[0]),
            message="target ArUco marker not detected",
        )

    message = f"marker={marker_id} z_trim={trim_m:+.4f} m"

    return SingleArmArucoAlignmentResult(
        valid=True,
        markers_detected=len(ids_flat),
        marker_ids=ids_flat,
        image_width=int(gray.shape[1]),
        image_height=int(gray.shape[0]),
        marker_seen=True,
        recommended_z_trim_m=trim_m,
        center_y_error_m=metric_error,
        message=message,
    )
