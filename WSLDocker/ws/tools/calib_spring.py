#!/usr/bin/env python3
"""
Interpret spring-calibration artifacts emitted by hardware_harness_ablated.py.

This script is intentionally standalone: it does not depend on ROS 2 and it
does not attempt to recalibrate from raw depth images. Instead, it reads the
calibration JSON written by the ablation harness, optionally reads the adjacent
depth_camera_calibration.json metadata, and produces a paper-friendly report
describing the measured spring parameters and how the current K_lat case grid
relates to the measured critical stiffness K_lat*.

Typical usage:
  python3 tools/calib_spring.py /path/to/calib_result_20260515_101500.json
  python3 tools/calib_spring.py /path/to/calib_dir --write-report
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from tools.ablation_config import CARTESIAN_STIFFNESS_LIMIT_NPM as HW_K_LAT_CEILING_NPM
    from tools.ablation_config import HW_K_LAT_CASES
except ImportError:
    from ablation_config import CARTESIAN_STIFFNESS_LIMIT_NPM as HW_K_LAT_CEILING_NPM
    from ablation_config import HW_K_LAT_CASES


@dataclass
class CaseAssessment:
    case: str
    K_lat_Npm: float
    fraction_of_K_lat_star: float
    classification: str


@dataclass
class CalibrationReport:
    calibration_json_path: str
    depth_camera_calibration_path: Optional[str]
    depth_camera_source: Optional[str]
    depth_camera_is_approximate: Optional[bool]
    L0_m: float
    k_a_Npm: float
    P_b_hat_N: float
    B_eff_Nm2: float
    k_theta_Nm_per_rad: float
    K_lat_star_Npm: float
    K_lat_meas_Npm: float
    K_lat_meas_fraction_of_ceiling: float
    default_case_assessment: List[CaseAssessment]
    notes: List[str]


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as handle:
        return json.load(handle)


def resolve_calibration_json(path: str) -> str:
    if os.path.isfile(path):
        return path
    candidates: List[str] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.startswith("calib_result_") and name.endswith(".json"):
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise FileNotFoundError(f"No calib_result_*.json found under {path}")
    candidates.sort()
    return candidates[-1]


def resolve_depth_camera_json(calibration_json_path: str) -> Optional[str]:
    candidate = os.path.join(os.path.dirname(calibration_json_path), "depth_camera_calibration.json")
    return candidate if os.path.exists(candidate) else None


def classify_fraction_of_critical(fraction: float) -> str:
    if not math.isfinite(fraction):
        return "unknown"
    if fraction < 0.10:
        return "far_subcritical"
    if fraction < 0.50:
        return "subcritical"
    if fraction < 0.90:
        return "near_critical"
    if fraction <= 1.05:
        return "critical_band"
    return "supercritical"


def build_case_assessment(k_lat_star: float) -> List[CaseAssessment]:
    assessments: List[CaseAssessment] = []
    for case, values in HW_K_LAT_CASES.items():
        for value in values:
            fraction = value / k_lat_star if _is_finite(k_lat_star) and k_lat_star > 0 else float("nan")
            assessments.append(
                CaseAssessment(
                    case=case,
                    K_lat_Npm=value,
                    fraction_of_K_lat_star=fraction,
                    classification=classify_fraction_of_critical(fraction),
                )
            )
    return assessments


def build_notes(
    calibration_payload: Dict[str, Any],
    depth_payload: Optional[Dict[str, Any]],
    k_lat_star: float,
    k_lat_meas: float,
    case_assessment: List[CaseAssessment],
) -> List[str]:
    notes: List[str] = []
    if depth_payload is None:
        notes.append("No depth_camera_calibration.json was found alongside the calibration JSON.")
    elif depth_payload.get("is_approximate"):
        notes.append(
            "Depth-camera extrinsics are marked approximate; use an external camera calibration when reporting geometry-sensitive results."
        )

    if _is_finite(k_lat_meas):
        fraction_of_ceiling = k_lat_meas / HW_K_LAT_CEILING_NPM
        if fraction_of_ceiling < 0.85:
            notes.append(
                f"Measured closed-loop K_lat is materially below the {HW_K_LAT_CEILING_NPM:.0f} N/m hardware ceiling; "
                "controller saturation or contact compliance may be reducing the realized stiffness."
            )

    if _is_finite(k_lat_star):
        supercritical = [item for item in case_assessment if item.classification == "supercritical"]
        if supercritical:
            notes.append(
                "One or more default case values exceed the measured K_lat* estimate; treat those cases as post-critical for this spring configuration."
            )
    else:
        notes.append(
            "K_lat* could not be validated from the calibration JSON; check P_b_hat_N and k_theta_Nm_per_rad."
        )

    required = ["L0_m", "k_a_Npm", "P_b_hat_N", "k_theta_Nm_per_rad"]
    missing = [name for name in required if not _is_finite(calibration_payload.get(name))]
    if missing:
        notes.append(
            "Some calibration quantities are missing or NaN: " + ", ".join(missing)
        )

    return notes


def build_report(calibration_json_path: str) -> CalibrationReport:
    calibration_payload = _load_json(calibration_json_path)
    depth_camera_path = resolve_depth_camera_json(calibration_json_path)
    depth_payload = _load_json(depth_camera_path) if depth_camera_path else None

    k_lat_star = calibration_payload.get("K_lat_star_Npm")
    if not _is_finite(k_lat_star):
        p_b_hat = calibration_payload.get("P_b_hat_N")
        k_theta = calibration_payload.get("k_theta_Nm_per_rad")
        if _is_finite(p_b_hat) and _is_finite(k_theta) and k_theta > 0:
            k_lat_star = (float(p_b_hat) ** 2) / float(k_theta)
        else:
            k_lat_star = float("nan")

    k_lat_meas = float(calibration_payload.get("K_lat_meas_Npm", float("nan")))
    case_assessment = build_case_assessment(float(k_lat_star))
    notes = build_notes(calibration_payload, depth_payload, float(k_lat_star), k_lat_meas, case_assessment)

    return CalibrationReport(
        calibration_json_path=calibration_json_path,
        depth_camera_calibration_path=depth_camera_path,
        depth_camera_source=(depth_payload or {}).get("source"),
        depth_camera_is_approximate=(depth_payload or {}).get("is_approximate"),
        L0_m=float(calibration_payload.get("L0_m", float("nan"))),
        k_a_Npm=float(calibration_payload.get("k_a_Npm", float("nan"))),
        P_b_hat_N=float(calibration_payload.get("P_b_hat_N", float("nan"))),
        B_eff_Nm2=float(calibration_payload.get("B_eff_Nm2", float("nan"))),
        k_theta_Nm_per_rad=float(calibration_payload.get("k_theta_Nm_per_rad", float("nan"))),
        K_lat_star_Npm=float(k_lat_star),
        K_lat_meas_Npm=k_lat_meas,
        K_lat_meas_fraction_of_ceiling=(
            k_lat_meas / HW_K_LAT_CEILING_NPM if _is_finite(k_lat_meas) else float("nan")
        ),
        default_case_assessment=case_assessment,
        notes=notes,
    )


def report_to_dict(report: CalibrationReport) -> Dict[str, Any]:
    payload = asdict(report)
    payload["default_case_assessment"] = [asdict(item) for item in report.default_case_assessment]
    return payload


def write_report(report: CalibrationReport, output_path: str) -> None:
    with open(output_path, "w") as handle:
        json.dump(report_to_dict(report), handle, indent=2)


def render_text_report(report: CalibrationReport) -> str:
    lines = [
        f"Calibration JSON : {report.calibration_json_path}",
        f"Depth extrinsics : {report.depth_camera_calibration_path or 'missing'}",
        f"Depth source     : {report.depth_camera_source or 'unknown'}",
        f"L0               : {report.L0_m:.6f} m",
        f"k_a              : {report.k_a_Npm:.3f} N/m",
        f"P_b_hat          : {report.P_b_hat_N:.3f} N",
        f"B_eff            : {report.B_eff_Nm2:.6f} N.m^2",
        f"k_theta          : {report.k_theta_Nm_per_rad:.6f} N.m/rad",
        f"K_lat*           : {report.K_lat_star_Npm:.3f} N/m",
        f"K_lat^meas       : {report.K_lat_meas_Npm:.3f} N/m",
        "",
        "Default case grid versus K_lat*:",
    ]
    for item in report.default_case_assessment:
        lines.append(
            f"  {item.case:5s}  K={item.K_lat_Npm:6.1f}  fraction={item.fraction_of_K_lat_star:7.3f}  {item.classification}"
        )
    if report.notes:
        lines.append("")
        lines.append("Notes:")
        for note in report.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret spring calibration artifacts")
    parser.add_argument("input", help="Calibration JSON file or directory containing calib_result_*.json")
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write spring_calibration_report.json next to the calibration JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path for --write-report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_json_path = resolve_calibration_json(args.input)
    report = build_report(calibration_json_path)
    print(render_text_report(report))
    if args.write_report:
        output_path = args.output or os.path.join(
            os.path.dirname(calibration_json_path), "spring_calibration_report.json"
        )
        write_report(report, output_path)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()