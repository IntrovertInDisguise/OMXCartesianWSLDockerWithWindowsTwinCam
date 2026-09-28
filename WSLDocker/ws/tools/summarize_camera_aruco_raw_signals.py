#!/usr/bin/env python3
"""
Summarize raw JSONL traces written by tools/camera_aruco.py --raw-signals-jsonl.

The summary is intentionally offline-only and stdlib-only so it can be used on
saved traces without ROS or OpenCV installed.

Typical usage:
  python3 tools/summarize_camera_aruco_raw_signals.py /tmp/camera_aruco_raw.jsonl
  python3 tools/summarize_camera_aruco_raw_signals.py /tmp/camera_aruco_raw.jsonl --output /tmp/camera_aruco_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EXPECTED_LABELS = [
    "ground_world",
    "green_platform_robot1",
    "metal_platform_center",
    "green_platform_robot2",
    "spring_cap_robot1",
    "spring_cap_robot2",
]

CSV_EXPORT_BASENAMES = {
    "overview": "overview.csv",
    "dropout_runs": "dropout_runs.csv",
    "marker_miss_streaks": "marker_miss_streaks.csv",
    "angle_flip_candidates": "angle_flip_candidates.csv",
}


def _to_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return float("nan")
    return float(text)


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _wrap_angle_deg(angle_deg: float) -> float:
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


def load_raw_signal_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def _record_sequence(record: Dict[str, Any], fallback_index: int) -> int:
    try:
        return int(record.get("sequence", fallback_index))
    except (TypeError, ValueError):
        return int(fallback_index)


def _summary_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    summary = record.get("summary")
    return summary if isinstance(summary, dict) else {}


def _detected_labels(summary: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for entry in summary.get("detections", []):
        if isinstance(entry, dict) and entry.get("label"):
            labels.append(str(entry["label"]))
    return labels


def _missing_labels(summary: Dict[str, Any], expected_labels: Sequence[str]) -> List[str]:
    expected = set(str(label) for label in expected_labels)
    seen = set(_detected_labels(summary))
    return sorted(expected - seen)


def _extract_angle_deg(summary: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    visual = summary.get("visual_bowing", {})
    if not isinstance(visual, dict) or not bool(visual.get("valid_detection")):
        return None, None
    for field in ("theta_cap_workspace_rad", "theta_cap_camera_rad", "theta_cap_rad"):
        value_rad = _to_float(visual.get(field))
        if _is_finite(value_rad):
            return math.degrees(value_rad), field
    return None, None


def _build_dropout_runs(records: Sequence[Dict[str, Any]], expected_labels: Sequence[str]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    active: Optional[Dict[str, Any]] = None
    for index, record in enumerate(records):
        sequence = _record_sequence(record, index)
        summary = _summary_dict(record)
        missing = _missing_labels(summary, expected_labels)
        is_valid = bool(summary.get("valid", False))
        is_dropout = (not is_valid) or bool(missing)
        if not is_dropout:
            if active is not None:
                active["missing_labels_union"] = sorted(active["missing_labels_union"])
                active["source_counts"] = dict(active["source_counts"])
                active["invalid_reason_counts"] = dict(active["invalid_reason_counts"])
                runs.append(active)
                active = None
            continue

        markers_detected = int(summary.get("markers_detected", 0) or 0)
        if active is None:
            active = {
                "start_sequence": sequence,
                "end_sequence": sequence,
                "length": 0,
                "source_counts": Counter(),
                "invalid_reason_counts": Counter(),
                "missing_labels_union": set(),
                "markers_detected_min": markers_detected,
                "markers_detected_max": markers_detected,
                "first_color_image_stamp_ns": record.get("color_image_stamp_ns"),
                "last_color_image_stamp_ns": record.get("color_image_stamp_ns"),
            }

        active["end_sequence"] = sequence
        active["length"] += 1
        active["source_counts"][str(record.get("source", "unknown"))] += 1
        active["markers_detected_min"] = min(active["markers_detected_min"], markers_detected)
        active["markers_detected_max"] = max(active["markers_detected_max"], markers_detected)
        active["last_color_image_stamp_ns"] = record.get("color_image_stamp_ns")
        active["missing_labels_union"].update(missing)
        if not is_valid:
            active["invalid_reason_counts"][str(summary.get("reason", ""))] += 1

    if active is not None:
        active["missing_labels_union"] = sorted(active["missing_labels_union"])
        active["source_counts"] = dict(active["source_counts"])
        active["invalid_reason_counts"] = dict(active["invalid_reason_counts"])
        runs.append(active)
    return runs


def _build_marker_miss_streaks(records: Sequence[Dict[str, Any]], expected_labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for label in expected_labels:
        runs: List[Dict[str, Any]] = []
        active: Optional[Dict[str, Any]] = None
        for index, record in enumerate(records):
            sequence = _record_sequence(record, index)
            summary = _summary_dict(record)
            missing = label in _missing_labels(summary, expected_labels)
            if not missing:
                if active is not None:
                    active["source_counts"] = dict(active["source_counts"])
                    active["invalid_reason_counts"] = dict(active["invalid_reason_counts"])
                    runs.append(active)
                    active = None
                continue
            if active is None:
                active = {
                    "start_sequence": sequence,
                    "end_sequence": sequence,
                    "length": 0,
                    "source_counts": Counter(),
                    "invalid_reason_counts": Counter(),
                }
            active["end_sequence"] = sequence
            active["length"] += 1
            active["source_counts"][str(record.get("source", "unknown"))] += 1
            if not bool(summary.get("valid", False)):
                active["invalid_reason_counts"][str(summary.get("reason", ""))] += 1
        if active is not None:
            active["source_counts"] = dict(active["source_counts"])
            active["invalid_reason_counts"] = dict(active["invalid_reason_counts"])
            runs.append(active)
        if runs:
            result[str(label)] = {
                "run_count": len(runs),
                "max_run_length": max(run["length"] for run in runs),
                "runs": runs,
            }
    return result


def _build_angle_flip_candidates(
    records: Sequence[Dict[str, Any]],
    expected_labels: Sequence[str],
    angle_flip_threshold_deg: float,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None
    for index, record in enumerate(records):
        sequence = _record_sequence(record, index)
        summary = _summary_dict(record)
        angle_deg, angle_field = _extract_angle_deg(summary)
        if angle_deg is None or angle_field is None:
            continue
        current = {
            "sequence": sequence,
            "angle_deg": float(angle_deg),
            "angle_field": str(angle_field),
            "markers_detected": int(summary.get("markers_detected", 0) or 0),
            "missing_labels": _missing_labels(summary, expected_labels),
            "detected_labels": _detected_labels(summary),
        }
        if previous is not None:
            delta_deg = _wrap_angle_deg(current["angle_deg"] - previous["angle_deg"])
            if abs(delta_deg) >= float(angle_flip_threshold_deg):
                candidates.append(
                    {
                        "previous_sequence": previous["sequence"],
                        "sequence": current["sequence"],
                        "previous_angle_deg": previous["angle_deg"],
                        "angle_deg": current["angle_deg"],
                        "delta_deg": float(delta_deg),
                        "previous_angle_field": previous["angle_field"],
                        "angle_field": current["angle_field"],
                        "previous_missing_labels": previous["missing_labels"],
                        "missing_labels": current["missing_labels"],
                        "previous_markers_detected": previous["markers_detected"],
                        "markers_detected": current["markers_detected"],
                        "detected_labels": current["detected_labels"],
                    }
                )
        previous = current
    return candidates


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _string_cell(value: Any) -> str:
    return "" if value is None else str(value)


def _float_or_nan(value: Any) -> float:
    try:
        return _to_float(value)
    except (TypeError, ValueError):
        return float("nan")


def _bool_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    return 1.0 if bool(value) else 0.0


def _pose_translation_component(pose: Any, index: int) -> float:
    if not isinstance(pose, dict):
        return float("nan")
    translation = pose.get("translation_m")
    if not isinstance(translation, (list, tuple)) or len(translation) <= index:
        return float("nan")
    return _float_or_nan(translation[index])


HARNESS_CAMERA_ARUCO_SPRING_LABELS = (
    "spring_cap_robot1",
    "spring_cap_robot2",
)


def harness_camera_aruco_log_columns(expected_labels: Sequence[str] = DEFAULT_EXPECTED_LABELS) -> List[str]:
    _ = expected_labels
    return [
        "aruco_valid",
        "aruco_reason",
        "aruco_markers_detected",
        "aruco_world_marker_visible",
        "aruco_coordinate_frame_workspace",
        "aruco_visual_valid_detection",
        "aruco_visual_reason",
        "aruco_visual_theta_cap_workspace_rad",
        "aruco_visual_cap_span_workspace_m",
        "aruco_visual_w_vis_delta_px",
        "aruco_visual_bent",
        "aruco_spring_cap_robot1_visible",
        "aruco_spring_cap_robot1_pose_workspace_tx_m",
        "aruco_spring_cap_robot1_pose_workspace_ty_m",
        "aruco_spring_cap_robot1_pose_workspace_tz_m",
        "aruco_spring_cap_robot2_visible",
        "aruco_spring_cap_robot2_pose_workspace_tx_m",
        "aruco_spring_cap_robot2_pose_workspace_ty_m",
        "aruco_spring_cap_robot2_pose_workspace_tz_m",
    ]


def harness_camera_aruco_log_row(
    summary: Optional[Dict[str, Any]],
    expected_labels: Sequence[str] = DEFAULT_EXPECTED_LABELS,
) -> Dict[str, Any]:
    row = {fieldname: float("nan") for fieldname in harness_camera_aruco_log_columns(expected_labels)}
    for fieldname in (
        "aruco_reason",
        "aruco_coordinate_frame_workspace",
        "aruco_visual_reason",
    ):
        row[fieldname] = ""

    if not isinstance(summary, dict):
        return row

    coordinate_frames = summary.get("coordinate_frames")
    if not isinstance(coordinate_frames, dict):
        coordinate_frames = {}
    depth_stream = summary.get("depth_stream")
    if not isinstance(depth_stream, dict):
        depth_stream = {}
    visual_bowing = summary.get("visual_bowing")
    if not isinstance(visual_bowing, dict):
        visual_bowing = {}

    row.update(
        {
            "aruco_valid": _bool_or_nan(summary.get("valid")),
            "aruco_reason": _string_cell(summary.get("reason")),
            "aruco_markers_detected": _float_or_nan(summary.get("markers_detected")),
            "aruco_world_marker_visible": _bool_or_nan(summary.get("world_marker_visible")),
            "aruco_coordinate_frame_workspace": _string_cell((coordinate_frames.get("workspace") or {}).get("frame_id")),
            "aruco_visual_valid_detection": _bool_or_nan(visual_bowing.get("valid_detection")),
            "aruco_visual_reason": _string_cell(visual_bowing.get("reason")),
            "aruco_visual_theta_cap_workspace_rad": _float_or_nan(visual_bowing.get("theta_cap_workspace_rad")),
            "aruco_visual_cap_span_workspace_m": _float_or_nan(visual_bowing.get("cap_span_workspace_m")),
            "aruco_visual_w_vis_delta_px": _float_or_nan(visual_bowing.get("w_vis_delta_px")),
            "aruco_visual_bent": _bool_or_nan(visual_bowing.get("bent")),
        }
    )

    detections_by_label: Dict[str, Dict[str, Any]] = {}
    for detection in summary.get("detections", []):
        if isinstance(detection, dict) and detection.get("label"):
            detections_by_label[str(detection["label"])] = detection

    for label in HARNESS_CAMERA_ARUCO_SPRING_LABELS:
        prefix = f"aruco_{label}"
        detection = detections_by_label.get(str(label))
        row[f"{prefix}_visible"] = 1.0 if detection is not None else 0.0
        if detection is None:
            continue
        row[f"{prefix}_pose_workspace_tx_m"] = _pose_translation_component(detection.get("pose_workspace"), 0)
        row[f"{prefix}_pose_workspace_ty_m"] = _pose_translation_component(detection.get("pose_workspace"), 1)
        row[f"{prefix}_pose_workspace_tz_m"] = _pose_translation_component(detection.get("pose_workspace"), 2)
    return row


def _write_csv_rows(path: str, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv_exports(summary: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    overview_path = os.path.join(output_dir, CSV_EXPORT_BASENAMES["overview"])
    _write_csv_rows(
        overview_path,
        [
            "records",
            "valid_record_count",
            "invalid_record_count",
            "records_with_all_expected_labels",
            "records_with_missing_expected_labels",
            "valid_records_with_all_expected_labels",
            "dropout_run_count",
            "angle_flip_candidate_count",
            "expected_labels_json",
            "source_counts_json",
            "markers_detected_histogram_json",
            "marker_missing_frame_counts_json",
        ],
        [
            {
                "records": int(summary.get("records", 0) or 0),
                "valid_record_count": int(summary.get("valid_record_count", 0) or 0),
                "invalid_record_count": int(summary.get("invalid_record_count", 0) or 0),
                "records_with_all_expected_labels": int(
                    summary.get("records_with_all_expected_labels", 0) or 0
                ),
                "records_with_missing_expected_labels": int(
                    summary.get("records_with_missing_expected_labels", 0) or 0
                ),
                "valid_records_with_all_expected_labels": int(
                    summary.get("valid_records_with_all_expected_labels", 0) or 0
                ),
                "dropout_run_count": int(summary.get("dropout_run_count", 0) or 0),
                "angle_flip_candidate_count": int(
                    summary.get("angle_flip_candidate_count", 0) or 0
                ),
                "expected_labels_json": _json_cell(summary.get("expected_labels", [])),
                "source_counts_json": _json_cell(summary.get("source_counts", {})),
                "markers_detected_histogram_json": _json_cell(
                    summary.get("markers_detected_histogram", {})
                ),
                "marker_missing_frame_counts_json": _json_cell(
                    summary.get("marker_missing_frame_counts", {})
                ),
            }
        ],
    )

    dropout_runs_path = os.path.join(output_dir, CSV_EXPORT_BASENAMES["dropout_runs"])
    dropout_rows = []
    for index, run in enumerate(summary.get("dropout_runs", []), start=1):
        dropout_rows.append(
            {
                "run_index": index,
                "start_sequence": run.get("start_sequence"),
                "end_sequence": run.get("end_sequence"),
                "length": run.get("length"),
                "markers_detected_min": run.get("markers_detected_min"),
                "markers_detected_max": run.get("markers_detected_max"),
                "first_color_image_stamp_ns": run.get("first_color_image_stamp_ns"),
                "last_color_image_stamp_ns": run.get("last_color_image_stamp_ns"),
                "missing_labels_union_json": _json_cell(run.get("missing_labels_union", [])),
                "source_counts_json": _json_cell(run.get("source_counts", {})),
                "invalid_reason_counts_json": _json_cell(run.get("invalid_reason_counts", {})),
            }
        )
    _write_csv_rows(
        dropout_runs_path,
        [
            "run_index",
            "start_sequence",
            "end_sequence",
            "length",
            "markers_detected_min",
            "markers_detected_max",
            "first_color_image_stamp_ns",
            "last_color_image_stamp_ns",
            "missing_labels_union_json",
            "source_counts_json",
            "invalid_reason_counts_json",
        ],
        dropout_rows,
    )

    marker_miss_path = os.path.join(output_dir, CSV_EXPORT_BASENAMES["marker_miss_streaks"])
    marker_rows = []
    marker_miss_streaks = summary.get("marker_miss_streaks", {})
    for label in sorted(marker_miss_streaks):
        for run_index, run in enumerate(marker_miss_streaks[label].get("runs", []), start=1):
            marker_rows.append(
                {
                    "label": label,
                    "run_index": run_index,
                    "start_sequence": run.get("start_sequence"),
                    "end_sequence": run.get("end_sequence"),
                    "length": run.get("length"),
                    "source_counts_json": _json_cell(run.get("source_counts", {})),
                    "invalid_reason_counts_json": _json_cell(run.get("invalid_reason_counts", {})),
                }
            )
    _write_csv_rows(
        marker_miss_path,
        [
            "label",
            "run_index",
            "start_sequence",
            "end_sequence",
            "length",
            "source_counts_json",
            "invalid_reason_counts_json",
        ],
        marker_rows,
    )

    angle_flip_path = os.path.join(output_dir, CSV_EXPORT_BASENAMES["angle_flip_candidates"])
    angle_rows = []
    for index, candidate in enumerate(summary.get("angle_flip_candidates", []), start=1):
        angle_rows.append(
            {
                "candidate_index": index,
                "previous_sequence": candidate.get("previous_sequence"),
                "sequence": candidate.get("sequence"),
                "previous_angle_deg": candidate.get("previous_angle_deg"),
                "angle_deg": candidate.get("angle_deg"),
                "delta_deg": candidate.get("delta_deg"),
                "previous_angle_field": candidate.get("previous_angle_field"),
                "angle_field": candidate.get("angle_field"),
                "previous_markers_detected": candidate.get("previous_markers_detected"),
                "markers_detected": candidate.get("markers_detected"),
                "previous_missing_labels_json": _json_cell(candidate.get("previous_missing_labels", [])),
                "missing_labels_json": _json_cell(candidate.get("missing_labels", [])),
                "detected_labels_json": _json_cell(candidate.get("detected_labels", [])),
            }
        )
    _write_csv_rows(
        angle_flip_path,
        [
            "candidate_index",
            "previous_sequence",
            "sequence",
            "previous_angle_deg",
            "angle_deg",
            "delta_deg",
            "previous_angle_field",
            "angle_field",
            "previous_markers_detected",
            "markers_detected",
            "previous_missing_labels_json",
            "missing_labels_json",
            "detected_labels_json",
        ],
        angle_rows,
    )

    return {
        "overview_csv_path": overview_path,
        "dropout_runs_csv_path": dropout_runs_path,
        "marker_miss_streaks_csv_path": marker_miss_path,
        "angle_flip_candidates_csv_path": angle_flip_path,
    }


def summarize_records(
    records: Sequence[Dict[str, Any]],
    expected_labels: Optional[Sequence[str]] = None,
    angle_flip_threshold_deg: float = 10.0,
) -> Dict[str, Any]:
    expected_labels = list(DEFAULT_EXPECTED_LABELS if expected_labels is None else expected_labels)
    source_counts = Counter()
    valid_counts = Counter()
    markers_detected_histogram = Counter()
    marker_missing_frame_counts = Counter()
    unique_color_stamps = set()
    unique_depth_stamps = set()
    valid_record_count = 0
    invalid_record_count = 0
    records_with_all_expected_labels = 0
    records_with_missing_expected_labels = 0
    valid_records_with_all_expected_labels = 0

    for index, record in enumerate(records):
        summary = _summary_dict(record)
        source_counts[str(record.get("source", "unknown"))] += 1
        is_valid = bool(summary.get("valid", False))
        valid_counts[is_valid] += 1
        if is_valid:
            valid_record_count += 1
        else:
            invalid_record_count += 1
        markers_detected_histogram[int(summary.get("markers_detected", 0) or 0)] += 1
        missing = _missing_labels(summary, expected_labels)
        if missing:
            records_with_missing_expected_labels += 1
            for label in missing:
                marker_missing_frame_counts[str(label)] += 1
        else:
            records_with_all_expected_labels += 1
            if is_valid:
                valid_records_with_all_expected_labels += 1
        if record.get("color_image_stamp_ns") is not None:
            unique_color_stamps.add(record.get("color_image_stamp_ns"))
        if record.get("depth_image_stamp_ns") is not None:
            unique_depth_stamps.add(record.get("depth_image_stamp_ns"))
        _record_sequence(record, index)

    dropout_runs = _build_dropout_runs(records, expected_labels)
    marker_miss_streaks = _build_marker_miss_streaks(records, expected_labels)
    angle_flip_candidates = _build_angle_flip_candidates(records, expected_labels, angle_flip_threshold_deg)

    return {
        "records": len(records),
        "expected_labels": list(expected_labels),
        "source_counts": dict(source_counts),
        "valid_counts": {str(key): count for key, count in valid_counts.items()},
        "valid_record_count": valid_record_count,
        "invalid_record_count": invalid_record_count,
        "markers_detected_histogram": dict(sorted(markers_detected_histogram.items())),
        "marker_missing_frame_counts": dict(sorted(marker_missing_frame_counts.items())),
        "records_with_all_expected_labels": records_with_all_expected_labels,
        "records_with_missing_expected_labels": records_with_missing_expected_labels,
        "valid_records_with_all_expected_labels": valid_records_with_all_expected_labels,
        "unique_color_image_stamps": len(unique_color_stamps),
        "unique_depth_image_stamps": len(unique_depth_stamps),
        "dropout_runs": dropout_runs,
        "dropout_run_count": len(dropout_runs),
        "marker_miss_streaks": marker_miss_streaks,
        "angle_flip_threshold_deg": float(angle_flip_threshold_deg),
        "angle_flip_candidates": angle_flip_candidates,
        "angle_flip_candidate_count": len(angle_flip_candidates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize camera_aruco raw JSONL traces")
    parser.add_argument("input", help="Path to camera_aruco raw JSONL file")
    parser.add_argument(
        "--expected-label",
        action="append",
        default=None,
        help="Expected marker label; repeat to override the default six-marker rig layout",
    )
    parser.add_argument(
        "--angle-flip-threshold-deg",
        type=float,
        default=10.0,
        help="Absolute wrapped angle jump in degrees above which a candidate flip is reported",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument(
        "--csv-output-dir",
        default=None,
        help="Optional directory in which to write overview/dropout/marker-miss/angle-flip CSV tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_raw_signal_records(args.input)
    summary = summarize_records(
        records,
        expected_labels=args.expected_label,
        angle_flip_threshold_deg=args.angle_flip_threshold_deg,
    )
    summary["input_path"] = os.path.abspath(args.input)
    rendered = json.dumps(summary, indent=2, sort_keys=False)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
    if args.csv_output_dir:
        write_summary_csv_exports(summary, args.csv_output_dir)


if __name__ == "__main__":
    main()