#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    from tools.depth_frame_utils import decode_sensor_image
except ImportError:
    from depth_frame_utils import decode_sensor_image


ARUCO_ALIGNMENT_VALID_TOPIC = "/spring_monitor/aruco_alignment/valid"
ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC = "/spring_monitor/aruco_alignment/recommended_robot1_z_trim_m"
ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC = "/spring_monitor/aruco_alignment/recommended_robot2_z_trim_m"
ARUCO_ALIGNMENT_ROLL_DEG_TOPIC = "/spring_monitor/aruco_alignment/roll_deg"
ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC = "/spring_monitor/aruco_alignment/center_y_error_m"
ARUCO_ALIGNMENT_MARKERS_DETECTED_TOPIC = "/spring_monitor/aruco_alignment/markers_detected"

ARUCO_DETECTOR_PROFILE_DEFAULT = "default"
ARUCO_DETECTOR_PROFILE_FAST = "fast"
ARUCO_DETECTOR_PROFILE_CHOICES = (
    ARUCO_DETECTOR_PROFILE_DEFAULT,
    ARUCO_DETECTOR_PROFILE_FAST,
)


@dataclass
class ArucoAlignmentResult:
    valid: bool
    markers_detected: int = 0
    marker_ids: List[int] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    robot1_marker_seen: bool = False
    robot2_marker_seen: bool = False
    recommended_robot1_z_trim_m: float = float("nan")
    recommended_robot2_z_trim_m: float = float("nan")
    center_y_error_m: float = float("nan")
    roll_deg: float = float("nan")
    message: str = ""


@dataclass
class FallbackAlignmentResult:
    valid: bool
    alignment_ok: bool
    yaw_deg: float = float("nan")
    lateral_deflection_m: float = float("nan")
    psi_diff_deg: float = float("nan")
    message: str = ""


def _finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def _aruco_available() -> bool:
    return cv2 is not None and hasattr(cv2, "aruco")


def _get_aruco_dictionary(dictionary_name: str):
    if not _aruco_available():
        raise RuntimeError("OpenCV ArUco support is unavailable")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def _normalize_detector_profile(detector_profile: str) -> str:
    normalized = str(detector_profile or ARUCO_DETECTOR_PROFILE_DEFAULT).strip().lower()
    if normalized not in ARUCO_DETECTOR_PROFILE_CHOICES:
        raise ValueError(
            f"Unknown ArUco detector profile: {detector_profile}. "
            f"Expected one of {', '.join(ARUCO_DETECTOR_PROFILE_CHOICES)}"
        )
    return normalized


def _configure_detector_parameters(detector_parameters: Any, detector_profile: str) -> Any:
    profile = _normalize_detector_profile(detector_profile)
    if profile == ARUCO_DETECTOR_PROFILE_DEFAULT:
        return detector_parameters

    aruco_module = getattr(cv2, "aruco", None)
    if aruco_module is None:
        return detector_parameters

    fast_overrides = {
        "cornerRefinementMethod": getattr(aruco_module, "CORNER_REFINE_NONE", 0),
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 13,
        "adaptiveThreshWinSizeStep": 10,
        "minMarkerPerimeterRate": 0.05,
        "maxMarkerPerimeterRate": 2.0,
        "polygonalApproxAccuracyRate": 0.05,
    }
    for attribute_name, attribute_value in fast_overrides.items():
        if hasattr(detector_parameters, attribute_name):
            setattr(detector_parameters, attribute_name, attribute_value)
    return detector_parameters


def _get_detector_parameters(detector_profile: str = ARUCO_DETECTOR_PROFILE_DEFAULT):
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        detector_parameters = cv2.aruco.DetectorParameters_create()
    else:
        detector_parameters = cv2.aruco.DetectorParameters()
    return _configure_detector_parameters(detector_parameters, detector_profile)


def _camera_matrix_from_info(camera_info_msg: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if camera_info_msg is None:
        return None, None
    k = list(getattr(camera_info_msg, "k", []))
    if len(k) != 9:
        return None, None
    camera_matrix = np.asarray(k, dtype=np.float64).reshape(3, 3)
    d = list(getattr(camera_info_msg, "d", []))
    dist_coeffs = np.asarray(d if d else [0.0] * 5, dtype=np.float64)
    return camera_matrix, dist_coeffs


def _gray_image(image_msg: Any) -> np.ndarray:
    array = decode_sensor_image(image_msg)
    if array.ndim == 2:
        return array
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def detect_aruco_alignment(
    image_msg: Any,
    camera_info_msg: Any,
    robot1_marker_id: int,
    robot2_marker_id: int,
    marker_length_m: float,
    dictionary_name: str,
    target_row_fraction: float = 0.5,
    metal_platform_marker_id: Optional[int] = None,
    detector_profile: str = ARUCO_DETECTOR_PROFILE_DEFAULT,
) -> ArucoAlignmentResult:
    if not _aruco_available():
        return ArucoAlignmentResult(valid=False, message="OpenCV ArUco support is unavailable")
    if image_msg is None or camera_info_msg is None:
        return ArucoAlignmentResult(valid=False, message="image or camera_info unavailable")

    try:
        gray = _gray_image(image_msg)
        dictionary = _get_aruco_dictionary(dictionary_name)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=_get_detector_parameters(detector_profile),
        )
    except Exception as exc:
        return ArucoAlignmentResult(valid=False, message=f"ArUco detection failed: {exc}")

    if ids is None or len(ids) == 0:
        return ArucoAlignmentResult(
            valid=False,
            image_width=int(gray.shape[1]),
            image_height=int(gray.shape[0]),
            message="no ArUco markers detected",
        )

    camera_matrix, dist_coeffs = _camera_matrix_from_info(camera_info_msg)
    if camera_matrix is None:
        return ArucoAlignmentResult(
            valid=False,
            markers_detected=len(ids),
            marker_ids=[int(value) for value in np.asarray(ids).reshape(-1)],
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
        return ArucoAlignmentResult(
            valid=False,
            markers_detected=len(ids),
            marker_ids=[int(value) for value in np.asarray(ids).reshape(-1)],
            image_width=int(gray.shape[1]),
            image_height=int(gray.shape[0]),
            message=f"pose estimation failed: {exc}",
        )

    ids_flat = [int(value) for value in np.asarray(ids).reshape(-1)]
    fy = float(camera_matrix[1, 1])
    target_row_px = float(gray.shape[0]) * float(target_row_fraction)

    slot_trims = {}
    slot_errors = {}
    slot_centers = {}
    slot_seen = {"robot1": False, "robot2": False}

    # First pass: locate the metal platform marker row (for midpoint z reference).
    platform_row_px: Optional[float] = None
    if metal_platform_marker_id is not None:
        platform_id = int(metal_platform_marker_id)
        for marker_corners, marker_id, marker_tvec in zip(corners, ids_flat, np.asarray(tvecs)):
            if marker_id != platform_id:
                continue
            platform_center_xy = np.mean(
                np.asarray(marker_corners, dtype=np.float64).reshape(-1, 2), axis=0
            )
            platform_depth_m = float(np.asarray(marker_tvec, dtype=np.float64).reshape(-1)[2])
            if _finite(platform_depth_m) and platform_depth_m > 0.0:
                platform_row_px = float(platform_center_xy[1])
            break

    for marker_corners, marker_id, marker_tvec in zip(corners, ids_flat, np.asarray(tvecs)):
        if marker_id not in {int(robot1_marker_id), int(robot2_marker_id)}:
            continue
        center_xy = np.mean(np.asarray(marker_corners, dtype=np.float64).reshape(-1, 2), axis=0)
        marker_depth_m = float(np.asarray(marker_tvec, dtype=np.float64).reshape(-1)[2])
        if not (_finite(marker_depth_m) and marker_depth_m > 0.0 and _finite(fy) and fy > 0.0):
            continue
        cap_row = float(center_xy[1])
        # Use midpoint between spring cap row and metal platform row as z reference when
        # the platform marker is visible; fall back to the fixed target_row_px otherwise.
        if platform_row_px is not None:
            effective_target_row = (cap_row + platform_row_px) / 2.0
        else:
            effective_target_row = target_row_px
        pixel_error = cap_row - effective_target_row
        metric_error = pixel_error * marker_depth_m / fy
        trim_m = -metric_error
        slot_name = "robot1" if marker_id == int(robot1_marker_id) else "robot2"
        slot_seen[slot_name] = True
        slot_trims[slot_name] = trim_m
        slot_errors[slot_name] = metric_error
        slot_centers[slot_name] = (float(center_xy[0]), cap_row)

    if not slot_trims:
        return ArucoAlignmentResult(
            valid=False,
            markers_detected=len(ids_flat),
            marker_ids=ids_flat,
            image_width=int(gray.shape[1]),
            image_height=int(gray.shape[0]),
            message="target ArUco markers not detected",
        )

    if "robot1" not in slot_trims and "robot2" in slot_trims:
        slot_trims["robot1"] = slot_trims["robot2"]
        slot_errors["robot1"] = slot_errors["robot2"]
    if "robot2" not in slot_trims and "robot1" in slot_trims:
        slot_trims["robot2"] = slot_trims["robot1"]
        slot_errors["robot2"] = slot_errors["robot1"]

    roll_deg = float("nan")
    if len(slot_centers) >= 2:
        points = sorted(slot_centers.values(), key=lambda item: item[0])
        left_pt, right_pt = points[0], points[-1]
        roll_deg = math.degrees(math.atan2(right_pt[1] - left_pt[1], right_pt[0] - left_pt[0]))

    center_y_error_m = float(np.mean(list(slot_errors.values()))) if slot_errors else float("nan")
    detected_slots = sorted(slot_name for slot_name, seen in slot_seen.items() if seen)
    message = (
        f"markers={detected_slots} "
        f"r1_trim={slot_trims['robot1']:+.4f} m "
        f"r2_trim={slot_trims['robot2']:+.4f} m"
    )
    if _finite(roll_deg):
        message += f" roll={roll_deg:+.2f} deg"

    return ArucoAlignmentResult(
        valid=True,
        markers_detected=len(ids_flat),
        marker_ids=ids_flat,
        image_width=int(gray.shape[1]),
        image_height=int(gray.shape[0]),
        robot1_marker_seen=slot_seen["robot1"],
        robot2_marker_seen=slot_seen["robot2"],
        recommended_robot1_z_trim_m=slot_trims["robot1"],
        recommended_robot2_z_trim_m=slot_trims["robot2"],
        center_y_error_m=center_y_error_m,
        roll_deg=roll_deg,
        message=message,
    )


def evaluate_fallback_alignment(
    lateral_deflection_m: float,
    yaw_deg: float,
    psi_left_rad: float,
    psi_right_rad: float,
    w_threshold_m: float,
    yaw_threshold_deg: float,
    psi_diff_threshold_deg: float,
) -> FallbackAlignmentResult:
    psi_diff_deg = float("nan")
    if _finite(psi_left_rad) and _finite(psi_right_rad):
        psi_diff_deg = math.degrees(psi_left_rad - psi_right_rad)

    valid = any(_finite(value) for value in (lateral_deflection_m, yaw_deg, psi_diff_deg))
    if not valid:
        return FallbackAlignmentResult(
            valid=False,
            alignment_ok=False,
            yaw_deg=yaw_deg,
            lateral_deflection_m=lateral_deflection_m,
            psi_diff_deg=psi_diff_deg,
            message="fallback alignment topics unavailable",
        )

    violations: List[str] = []
    if _finite(lateral_deflection_m) and abs(lateral_deflection_m) > w_threshold_m:
        violations.append(f"|w|={abs(lateral_deflection_m):.4f} m > {w_threshold_m:.4f} m")
    if _finite(yaw_deg) and abs(yaw_deg) > yaw_threshold_deg:
        violations.append(f"|yaw|={abs(yaw_deg):.2f} deg > {yaw_threshold_deg:.2f} deg")
    if _finite(psi_diff_deg) and abs(psi_diff_deg) > psi_diff_threshold_deg:
        violations.append(
            f"|psi_L-psi_R|={abs(psi_diff_deg):.2f} deg > {psi_diff_threshold_deg:.2f} deg"
        )

    if violations:
        return FallbackAlignmentResult(
            valid=True,
            alignment_ok=False,
            yaw_deg=yaw_deg,
            lateral_deflection_m=lateral_deflection_m,
            psi_diff_deg=psi_diff_deg,
            message="fallback misalignment: " + "; ".join(violations),
        )

    return FallbackAlignmentResult(
        valid=True,
        alignment_ok=True,
        yaw_deg=yaw_deg,
        lateral_deflection_m=lateral_deflection_m,
        psi_diff_deg=psi_diff_deg,
        message="fallback alignment within thresholds",
    )