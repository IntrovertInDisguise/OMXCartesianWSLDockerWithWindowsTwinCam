#!/usr/bin/env python3
"""
extract_aruco_from_clips.py
────────────────────────────
Post-process 30-FPS AVI clips captured by the host-side recorder
(realsense_text_trigger_capture.py) to extract per-frame spring-cap tilt
angles and lateral offsets from ArUco marker poses.

What this measures
──────────────────
For each video clip and each detected spring-cap ArUco marker the script
computes two physically meaningful, directly sensor-observable quantities:

  theta_deg  — tilt angle of the spring-cap face (degrees).
               Derived from the rvec returned by estimatePoseSingleMarkers:
               R = cv2.Rodrigues(rvec); theta = arccos(R[2,2]).
               R[2,2] is the cosine of the angle between the marker's face
               normal and the camera's optical axis. When the cap is
               perfectly perpendicular to the camera, theta = 0.

  lat_offset_m — lateral position of the cap centre in the camera frame
                 (metres), measured as sqrt(tvec_x² + tvec_y²).  This is
                 the distance of the cap centre from the camera optical
                 axis projected onto the sensor plane (at the cap's depth).
                 It quantifies off-axis drift of the contact point.

These are NOT derived from any spring model; they come directly from the
solvePnP solution for each detected marker.

Intrinsics format
─────────────────
Pass a JSON file produced in one of two ways:

  1. ROS-style (from camera_info_to_dict in depth_frame_utils.py):
     {"k": [fx, 0, cx, 0, fy, cy, 0, 0, 1], "d": [d0, d1, d2, d3, d4]}

  2. RealSense SDK-style (from rs2_intrinsics serialisation):
     {"fx": 615.3, "fy": 615.3, "ppx": 320.8, "ppy": 240.5,
      "coeffs": [d0, d1, d2, d3, d4]}

  If no intrinsics file is given, zero-distortion with fx=fy=800 and
  principal point at the image centre is assumed (suitable for quick
  checks; NOT for paper-quality output).

Clip-to-harness matching
─────────────────────────
The script can optionally join the per-frame output to the harness sync CSV
(--harness-csv).  Matching is done by window index:

  * Clips are sorted alphabetically.  The i-th clip corresponds to the
    i-th mutual-contact window.
  * Windows in the harness CSV are identified by rising edges of the
    `bag_recording_active` column.
  * The merged output adds harness columns (case_name, case_k_lateral_npm,
    load_step_index, shared_load_n, mean_load_n, timestamp) to every frame
    in the clip, joining on window_index.

Usage
─────
  python3 tools/extract_aruco_from_clips.py \\
      --clips-dir /path/to/captures \\
      --intrinsics /path/to/camera_info.json \\
      --output per_frame_aruco.csv

  # With harness CSV join:
  python3 tools/extract_aruco_from_clips.py \\
      --clips-dir /path/to/captures \\
      --intrinsics /path/to/camera_info.json \\
      --output per_frame_aruco.csv \\
      --harness-csv logs/dual_variable_stiffness/sync_log.csv \\
      --merged-csv merged_aruco_harness.csv

  # Override spring-cap marker IDs (default 4 and 5):
  python3 tools/extract_aruco_from_clips.py \\
      --clips-dir /path/to/captures \\
      --cap1-id 1 --cap2-id 3 \\
      --output per_frame_aruco.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2 as _cv2_module
    _CV2_AVAILABLE = hasattr(_cv2_module, "aruco")
except ImportError:
    _cv2_module = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

# cv2 is referenced by name throughout; alias for cleaner code.
cv2 = _cv2_module  # may be None when running pure-Python tests

try:
    from tools.ablation_config import DEFAULT_ARUCO_DICTIONARY_NAME
except ImportError:
    from ablation_config import DEFAULT_ARUCO_DICTIONARY_NAME

try:
    from tools.aruco_alignment_utils import (
        _get_aruco_dictionary,
        _get_detector_parameters,
    )
except ImportError:
    from aruco_alignment_utils import (
        _get_aruco_dictionary,
        _get_detector_parameters,
    )

# Default spring-cap marker IDs from camera_aruco.py DEFAULT_MARKER_SPECS
DEFAULT_CAP1_MARKER_ID = 4  # spring_cap_robot1
DEFAULT_CAP2_MARKER_ID = 5  # spring_cap_robot2
DEFAULT_MARKER_LENGTH_M = 0.038  # 38 mm — the physical tag size on the rig
DEFAULT_DICTIONARY_NAME = DEFAULT_ARUCO_DICTIONARY_NAME

# Harness CSV columns used for the optional join
_HARNESS_JOIN_COLUMNS = [
    "timestamp",
    "case_name",
    "case_k_lateral_npm",
    "load_step_index",
    "load_step_state",
    "shared_load_n",
    "mean_load_n",
    "bag_recording_active",
]

_VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov"}


# ──────────────────────────────────────────────────────────────────────────────
# Intrinsics helpers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # shape (5,) or (4,) or (8,)

    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def load_intrinsics(json_path: Optional[str], image_width: int = 0, image_height: int = 0) -> CameraIntrinsics:
    """Load camera intrinsics from a JSON file.

    Accepts ROS-style (``k`` array) or RealSense SDK-style (``fx``/``fy``/``ppx``/``ppy``) JSON.
    Falls back to synthetic defaults when *json_path* is ``None``.
    """
    if json_path is None:
        cx = image_width / 2.0 if image_width else 320.0
        cy = image_height / 2.0 if image_height else 240.0
        return CameraIntrinsics(fx=800.0, fy=800.0, cx=cx, cy=cy, dist_coeffs=np.zeros(5, dtype=np.float64))

    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    # ROS-style: {"k": [fx, 0, cx, 0, fy, cy, 0, 0, 1], "d": [...]}
    if "k" in data:
        k = list(data["k"])
        if len(k) != 9:
            raise ValueError(f"Intrinsics JSON 'k' must have 9 elements, got {len(k)}")
        fx, cx = float(k[0]), float(k[2])
        fy, cy = float(k[4]), float(k[5])
        raw_d = list(data.get("d", []))
        dist_coeffs = np.asarray(raw_d if raw_d else [0.0] * 5, dtype=np.float64)
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, dist_coeffs=dist_coeffs)

    # RealSense SDK-style: {"fx":..., "fy":..., "ppx":..., "ppy":..., "coeffs":[...]}
    if "fx" in data and "fy" in data:
        fx = float(data["fx"])
        fy = float(data["fy"])
        cx = float(data.get("ppx", data.get("cx", 0.0)))
        cy = float(data.get("ppy", data.get("cy", 0.0)))
        raw_d = list(data.get("coeffs", data.get("d", [])))
        dist_coeffs = np.asarray(raw_d if raw_d else [0.0] * 5, dtype=np.float64)
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, dist_coeffs=dist_coeffs)

    raise ValueError(
        f"Unrecognised intrinsics JSON format in {json_path}. "
        "Expected 'k' array (ROS-style) or 'fx'/'fy'/'ppx'/'ppy' (RealSense SDK-style)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-frame result
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CapPose:
    detected: bool = False
    theta_deg: float = float("nan")
    lat_offset_m: float = float("nan")
    depth_m: float = float("nan")
    tvec_x: float = float("nan")
    tvec_y: float = float("nan")
    tvec_z: float = float("nan")
    rvec_r0: float = float("nan")
    rvec_r1: float = float("nan")
    rvec_r2: float = float("nan")


@dataclass
class FrameResult:
    clip_name: str
    window_index: int
    frame_idx: int
    frame_ts_s: float
    cap1: CapPose = field(default_factory=CapPose)
    cap2: CapPose = field(default_factory=CapPose)

    def to_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "clip_name": self.clip_name,
            "window_index": self.window_index,
            "frame_idx": self.frame_idx,
            "frame_ts_s": self.frame_ts_s,
        }
        for prefix, cap in (("cap1", self.cap1), ("cap2", self.cap2)):
            row[f"{prefix}_detected"] = int(cap.detected)
            row[f"{prefix}_theta_deg"] = cap.theta_deg
            row[f"{prefix}_lat_offset_m"] = cap.lat_offset_m
            row[f"{prefix}_depth_m"] = cap.depth_m
            row[f"{prefix}_tvec_x"] = cap.tvec_x
            row[f"{prefix}_tvec_y"] = cap.tvec_y
            row[f"{prefix}_tvec_z"] = cap.tvec_z
            row[f"{prefix}_rvec_r0"] = cap.rvec_r0
            row[f"{prefix}_rvec_r1"] = cap.rvec_r1
            row[f"{prefix}_rvec_r2"] = cap.rvec_r2
        return row

    @staticmethod
    def csv_columns() -> List[str]:
        return [
            "clip_name",
            "window_index",
            "frame_idx",
            "frame_ts_s",
            "cap1_detected",
            "cap1_theta_deg",
            "cap1_lat_offset_m",
            "cap1_depth_m",
            "cap1_tvec_x",
            "cap1_tvec_y",
            "cap1_tvec_z",
            "cap1_rvec_r0",
            "cap1_rvec_r1",
            "cap1_rvec_r2",
            "cap2_detected",
            "cap2_theta_deg",
            "cap2_lat_offset_m",
            "cap2_depth_m",
            "cap2_tvec_x",
            "cap2_tvec_y",
            "cap2_tvec_z",
            "cap2_rvec_r0",
            "cap2_rvec_r1",
            "cap2_rvec_r2",
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Core pose extraction
# ──────────────────────────────────────────────────────────────────────────────


def _rvec_to_theta_deg(rvec: np.ndarray) -> float:
    """Return the tilt angle (degrees) of a marker face from the camera Z-axis.

    Converts the Rodrigues vector *rvec* to a rotation matrix R.  The marker's
    face normal in camera space is R[:,2].  The tilt angle is
    arccos(R[2,2]) = arccos(dot(R[:,2], [0,0,1])).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for _rvec_to_theta_deg")
    r_vec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    rotation_matrix, _ = cv2.Rodrigues(r_vec)
    cos_theta = float(np.clip(rotation_matrix[2, 2], -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def _tvec_to_lat_offset(tvec: np.ndarray) -> Tuple[float, float, float]:
    """Return (lat_offset_m, depth_m, tvec_x, tvec_y, tvec_z) from a pose tvec."""
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    lat_offset = float(math.sqrt(t[0] ** 2 + t[1] ** 2))
    return lat_offset, float(t[2]), float(t[0]), float(t[1]), float(t[2])


def _detect_cap_poses(
    gray: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    cap1_id: int,
    cap2_id: int,
    marker_length_m: float,
    dictionary_name: str,
) -> Tuple[CapPose, CapPose]:
    """Detect spring-cap markers in *gray* and return poses for cap1 and cap2."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for _detect_cap_poses")
    aruco_dict = _get_aruco_dictionary(dictionary_name)
    detector_params = _get_detector_parameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

    cap1 = CapPose()
    cap2 = CapPose()

    if ids is None or len(ids) == 0:
        return cap1, cap2

    ids_flat = np.asarray(ids, dtype=int).reshape(-1)
    target_ids = {cap1_id, cap2_id}
    detected_indices = [i for i, mid in enumerate(ids_flat) if int(mid) in target_ids]

    if not detected_indices:
        return cap1, cap2

    detected_corners = [corners[i] for i in detected_indices]
    detected_ids = [int(ids_flat[i]) for i in detected_indices]

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        detected_corners,
        float(marker_length_m),
        camera_matrix,
        dist_coeffs,
    )

    for rvec, tvec, mid in zip(np.asarray(rvecs), np.asarray(tvecs), detected_ids):
        rvec_flat = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
        theta = _rvec_to_theta_deg(rvec_flat)
        lat_offset, depth, tx, ty, tz = _tvec_to_lat_offset(tvec_flat)
        pose = CapPose(
            detected=True,
            theta_deg=theta,
            lat_offset_m=lat_offset,
            depth_m=depth,
            tvec_x=tx,
            tvec_y=ty,
            tvec_z=tz,
            rvec_r0=float(rvec_flat[0]),
            rvec_r1=float(rvec_flat[1]),
            rvec_r2=float(rvec_flat[2]),
        )
        if mid == cap1_id:
            cap1 = pose
        else:
            cap2 = pose

    return cap1, cap2


# ──────────────────────────────────────────────────────────────────────────────
# Clip-level processing
# ──────────────────────────────────────────────────────────────────────────────


def process_clip(
    clip_path: str,
    window_index: int,
    intrinsics: CameraIntrinsics,
    cap1_id: int,
    cap2_id: int,
    marker_length_m: float,
    dictionary_name: str,
) -> List[FrameResult]:
    """Process a single AVI clip and return a list of per-frame results."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for process_clip")
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {clip_path}", file=sys.stderr)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0:
        fps = 30.0

    camera_matrix = intrinsics.camera_matrix()
    dist_coeffs = intrinsics.dist_coeffs
    clip_name = Path(clip_path).name
    results: List[FrameResult] = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_ts_s = frame_idx / fps

        cap1_pose, cap2_pose = _detect_cap_poses(
            gray,
            camera_matrix,
            dist_coeffs,
            cap1_id,
            cap2_id,
            marker_length_m,
            dictionary_name,
        )

        results.append(
            FrameResult(
                clip_name=clip_name,
                window_index=window_index,
                frame_idx=frame_idx,
                frame_ts_s=frame_ts_s,
                cap1=cap1_pose,
                cap2=cap2_pose,
            )
        )
        frame_idx += 1

    cap.release()
    return results


def find_clips(clips_dir: str) -> List[str]:
    """Return sorted list of video file paths in *clips_dir*."""
    return sorted(
        str(p)
        for p in Path(clips_dir).iterdir()
        if p.suffix.lower() in _VIDEO_EXTENSIONS
    )


def process_all_clips(
    clips_dir: str,
    intrinsics: CameraIntrinsics,
    cap1_id: int,
    cap2_id: int,
    marker_length_m: float,
    dictionary_name: str,
) -> List[FrameResult]:
    """Process every clip in *clips_dir* and return all frame results."""
    clip_paths = find_clips(clips_dir)
    if not clip_paths:
        print(f"No video files found in {clips_dir}", file=sys.stderr)
        return []

    all_results: List[FrameResult] = []
    for window_index, clip_path in enumerate(clip_paths):
        print(f"  [{window_index + 1}/{len(clip_paths)}] {Path(clip_path).name}", file=sys.stderr)
        results = process_clip(
            clip_path,
            window_index,
            intrinsics,
            cap1_id,
            cap2_id,
            marker_length_m,
            dictionary_name,
        )
        all_results.extend(results)
        detected_frames = sum(1 for r in results if r.cap1.detected or r.cap2.detected)
        print(
            f"      {len(results)} frames, {detected_frames} with ≥1 cap detected",
            file=sys.stderr,
        )

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CSV I/O
# ──────────────────────────────────────────────────────────────────────────────


def write_per_frame_csv(results: List[FrameResult], output_path: str) -> None:
    columns = FrameResult.csv_columns()
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_row())
    print(f"Wrote {len(results)} rows → {output_path}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Optional harness CSV join
# ──────────────────────────────────────────────────────────────────────────────


def _detect_windows_from_harness(harness_rows: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """Group harness rows into mutual-contact windows by rising edges of bag_recording_active.

    Returns a dict: window_index → list of harness row dicts within that window.
    """
    windows: Dict[int, List[Dict[str, str]]] = {}
    current_window: Optional[int] = None
    prev_active = False

    for row in harness_rows:
        raw = row.get("bag_recording_active", "").strip().lower()
        active = raw in {"true", "1", "yes"}

        if active and not prev_active:
            current_window = len(windows)
            windows[current_window] = []

        if active and current_window is not None:
            windows[current_window].append(row)

        prev_active = active

    return windows


def _pick_representative_harness_row(window_rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Pick the best single harness row to represent a window for the merge.

    Prefers the first row where load_step_state indicates active loading.
    Falls back to the last row in the window.
    """
    for row in window_rows:
        if row.get("load_step_state", "").strip().lower() in {"loading", "hold", "active"}:
            return row
    return window_rows[-1] if window_rows else {}


def merge_with_harness_csv(
    frame_results: List[FrameResult],
    harness_csv_path: str,
    merged_output_path: str,
) -> None:
    """Join per-frame ArUco results to harness sync CSV by window_index.

    For each frame result, attaches the set of harness scalar columns
    (case_name, K_lat, load step, forces, timestamp) from the corresponding
    mutual-contact window in the harness CSV.
    """
    with open(harness_csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        harness_rows = list(reader)
        harness_fieldnames = list(reader.fieldnames or [])

    if not harness_rows:
        print(f"WARNING: harness CSV {harness_csv_path} is empty — skipping merge", file=sys.stderr)
        return

    windows = _detect_windows_from_harness(harness_rows)
    if not windows:
        print(
            "WARNING: no bag_recording_active transitions found in harness CSV — "
            "merge will attach empty harness columns",
            file=sys.stderr,
        )

    # Determine which harness columns actually exist
    available_join_cols = [c for c in _HARNESS_JOIN_COLUMNS if c in (harness_rows[0] if harness_rows else {})]

    frame_columns = FrameResult.csv_columns()
    merged_columns = frame_columns + [c for c in available_join_cols if c not in frame_columns]

    with open(merged_output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_columns, extrasaction="ignore")
        writer.writeheader()
        for result in frame_results:
            row = result.to_row()
            window_rows = windows.get(result.window_index, [])
            if window_rows:
                rep = _pick_representative_harness_row(window_rows)
                for col in available_join_cols:
                    row[col] = rep.get(col, "")
            else:
                for col in available_join_cols:
                    row[col] = ""
            writer.writerow(row)

    print(
        f"Merged {len(frame_results)} frames across {len(windows)} windows → {merged_output_path}",
        file=sys.stderr,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract spring-cap tilt and lateral offset from 30-FPS AVI clips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--clips-dir",
        required=True,
        help="Directory containing AVI clip files captured by realsense_text_trigger_capture.py.",
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        metavar="JSON",
        help="Camera intrinsics JSON (ROS-style 'k' array or RealSense SDK fx/fy/ppx/ppy). "
        "Omit to use synthetic defaults (fx=fy=800, zero distortion).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="CSV",
        help="Output path for the per-frame ArUco CSV.",
    )
    parser.add_argument(
        "--cap1-id",
        type=int,
        default=DEFAULT_CAP1_MARKER_ID,
        help=f"ArUco marker ID for spring_cap_robot1 (default: {DEFAULT_CAP1_MARKER_ID}).",
    )
    parser.add_argument(
        "--cap2-id",
        type=int,
        default=DEFAULT_CAP2_MARKER_ID,
        help=f"ArUco marker ID for spring_cap_robot2 (default: {DEFAULT_CAP2_MARKER_ID}).",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=DEFAULT_MARKER_LENGTH_M,
        metavar="M",
        help=f"Physical ArUco marker side length in metres (default: {DEFAULT_MARKER_LENGTH_M}).",
    )
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY_NAME,
        help=f"ArUco dictionary name (default: {DEFAULT_DICTIONARY_NAME}).",
    )
    parser.add_argument(
        "--harness-csv",
        default=None,
        metavar="CSV",
        help="Optional: harness sync CSV to join to the per-frame output.",
    )
    parser.add_argument(
        "--merged-csv",
        default=None,
        metavar="CSV",
        help="Output path for the merged ArUco+harness CSV. "
        "Required when --harness-csv is supplied.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.harness_csv and not args.merged_csv:
        parser.error("--merged-csv is required when --harness-csv is supplied")

    if not _CV2_AVAILABLE:
        print("ERROR: OpenCV with ArUco support is required. "
              "Install opencv-contrib-python or run with the system Python "
              "that has cv2 available.", file=sys.stderr)
        return 1

    intrinsics = load_intrinsics(args.intrinsics)

    print(
        f"Intrinsics: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
        f"cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f}",
        file=sys.stderr,
    )
    print(
        f"Cap markers: cap1={args.cap1_id} cap2={args.cap2_id} "
        f"length={args.marker_length:.4f} m  dict={args.dictionary}",
        file=sys.stderr,
    )

    frame_results = process_all_clips(
        clips_dir=args.clips_dir,
        intrinsics=intrinsics,
        cap1_id=args.cap1_id,
        cap2_id=args.cap2_id,
        marker_length_m=args.marker_length,
        dictionary_name=args.dictionary,
    )

    if not frame_results:
        print("No frames processed — check --clips-dir.", file=sys.stderr)
        return 1

    write_per_frame_csv(frame_results, args.output)

    if args.harness_csv:
        merge_with_harness_csv(frame_results, args.harness_csv, args.merged_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
