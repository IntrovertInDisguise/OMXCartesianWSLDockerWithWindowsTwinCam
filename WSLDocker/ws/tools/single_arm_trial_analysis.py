#!/usr/bin/env python3
"""Post-process a single-arm trial into online stiffness and beam-column validation.

The harness writes a per-trial sync CSV with lateral probe rows. This module
fits a small linearized stiffness model from those rows, then derives a reduced
beam-column validation summary. If a post-processed ArUco/video trace is
available, it is folded in as an optional geometry check rather than a hard
requirement.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is part of the workspace env
    np = None  # type: ignore[assignment]


DEFAULT_BOUNDARY_CONDITION = "cantilever"
DEFAULT_BOUNDARY_FACTORS = {
    "cantilever": 3.0,
    "clamped_clamped": 192.0,
    "pinned_pinned": 12.0,
}


@dataclass
class ProbeFit:
    sample_count: int
    displacement_axis: str
    force_axis: str
    raw_slope_npm: float
    online_lateral_eigenvalue_npm: float
    intercept_n: float
    residual_rms_n: float
    residual_std_n: float


@dataclass
class BeamColumnFit:
    boundary_condition: str
    boundary_factor: float
    effective_length_m: float
    effective_bending_stiffness_nm2: float
    critical_load_n: float
    predicted_lateral_stiffness_npm: float
    cubic_softening_npm3: float
    camera_trace_path: Optional[str]
    camera_frame_count: int
    camera_mean_lat_offset_m: float
    camera_mean_theta_deg: float
    camera_validation_note: str


@dataclass
class TrialAnalysis:
    sync_csv_path: str
    probe_fit: ProbeFit
    beam_column_fit: BeamColumnFit
    notes: List[str]



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



def _load_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))



def _probe_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    probe_rows = [row for row in rows if str(row.get("contact_mode", "")).strip() == "lateral_probe"]
    if probe_rows:
        return probe_rows
    return [row for row in rows if _is_finite(_to_float(row.get("probe_lateral_offset_m"))) and _is_finite(_to_float(row.get("probe_lateral_force_n")))]



def _linear_fit(x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float, float, float]:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if len(x_values) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    if np is not None:
        x = np.asarray(x_values, dtype=float)
        y = np.asarray(y_values, dtype=float)
        coeffs = np.polyfit(x, y, deg=1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        residuals = y - (slope * x + intercept)
        rms = float(np.sqrt(np.mean(residuals ** 2)))
        std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        return slope, intercept, rms, std

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    slope = numerator / denominator if denominator > 1e-12 else float("nan")
    intercept = y_mean - slope * x_mean if _is_finite(slope) else float("nan")
    residuals = [y - (slope * x + intercept) for x, y in zip(x_values, y_values)]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    std = math.sqrt(sum((value - (sum(residuals) / len(residuals))) ** 2 for value in residuals) / max(len(residuals) - 1, 1)) if residuals else float("nan")
    return slope, intercept, rms, std



def _fit_cubic_stiffness(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) < 3:
        return float("nan")
    if np is None:
        return float("nan")
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    design = np.column_stack([x, x ** 3])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeffs[1])



def _resolve_boundary_factor(boundary_condition: str) -> float:
    key = boundary_condition.strip().lower().replace("-", "_")
    return float(DEFAULT_BOUNDARY_FACTORS.get(key, DEFAULT_BOUNDARY_FACTORS[DEFAULT_BOUNDARY_CONDITION]))



def _load_camera_trace_rows(camera_trace_path: Optional[str]) -> List[Dict[str, Any]]:
    if not camera_trace_path:
        return []
    path = Path(camera_trace_path)
    if not path.exists():
        return []

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if rows:
        return rows

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []



def _camera_trace_statistics(rows: Sequence[Dict[str, Any]]) -> Tuple[float, float, int, str]:
    lat_offsets: List[float] = []
    theta_degs: List[float] = []
    for row in rows:
        for prefix in ("cap1", "cap2"):
            lat_value = _to_float(row.get(f"{prefix}_lat_offset_m"))
            theta_value = _to_float(row.get(f"{prefix}_theta_deg"))
            if _is_finite(lat_value):
                lat_offsets.append(lat_value)
            if _is_finite(theta_value):
                theta_degs.append(theta_value)
    if not lat_offsets and not theta_degs:
        return float("nan"), float("nan"), 0, "camera trace present but no usable cap pose fields were found"
    mean_lat = sum(lat_offsets) / len(lat_offsets) if lat_offsets else float("nan")
    mean_theta = sum(theta_degs) / len(theta_degs) if theta_degs else float("nan")
    return mean_lat, mean_theta, max(len(lat_offsets), len(theta_degs)), "camera trace parsed successfully"



def analyze_trial(
    sync_csv_path: str,
    camera_trace_path: Optional[str] = None,
    boundary_condition: str = DEFAULT_BOUNDARY_CONDITION,
    effective_length_m: Optional[float] = None,
) -> TrialAnalysis:
    rows = _load_csv_rows(sync_csv_path)
    probe_rows = _probe_rows(rows)
    if not probe_rows:
        raise ValueError(f"No lateral probe rows were found in {sync_csv_path}")

    displacement_axis = "offset_y"
    force_axis = "contact_fy"
    displacements = [_to_float(row.get(displacement_axis)) for row in probe_rows]
    forces = [_to_float(row.get(force_axis)) for row in probe_rows]

    slope, intercept, rms, std = _linear_fit(displacements, forces)
    if _is_finite(slope):
        online_eigenvalue = abs(slope)
    else:
        online_eigenvalue = float("nan")

    boundary_factor = _resolve_boundary_factor(boundary_condition)
    if effective_length_m is None:
        candidate = os.environ.get("OMX_HARNESS_SINGLE_ARM_BEAM_EFFECTIVE_LENGTH_M")
        if candidate is not None and candidate.strip():
            effective_length_m = _to_float(candidate)
    if effective_length_m is None or not _is_finite(float(effective_length_m)):
        effective_length_m = 0.128

    if _is_finite(online_eigenvalue):
        effective_bending_stiffness = online_eigenvalue * (float(effective_length_m) ** 3) / boundary_factor
        predicted_lateral_stiffness = boundary_factor * effective_bending_stiffness / (float(effective_length_m) ** 3)
        critical_load = (math.pi ** 2) * effective_bending_stiffness / (float(effective_length_m) ** 2)
    else:
        effective_bending_stiffness = float("nan")
        predicted_lateral_stiffness = float("nan")
        critical_load = float("nan")

    cubic_softening = _fit_cubic_stiffness(displacements, forces)

    camera_rows = _load_camera_trace_rows(camera_trace_path)
    camera_mean_lat, camera_mean_theta, camera_frame_count, camera_note = _camera_trace_statistics(camera_rows)
    if camera_rows and not _is_finite(camera_mean_lat):
        camera_note = "camera trace was found, but no usable cap pose columns were present"

    probe_fit = ProbeFit(
        sample_count=len(probe_rows),
        displacement_axis=displacement_axis,
        force_axis=force_axis,
        raw_slope_npm=slope,
        online_lateral_eigenvalue_npm=online_eigenvalue,
        intercept_n=intercept,
        residual_rms_n=rms,
        residual_std_n=std,
    )
    beam_fit = BeamColumnFit(
        boundary_condition=boundary_condition,
        boundary_factor=boundary_factor,
        effective_length_m=float(effective_length_m),
        effective_bending_stiffness_nm2=effective_bending_stiffness,
        critical_load_n=critical_load,
        predicted_lateral_stiffness_npm=predicted_lateral_stiffness,
        cubic_softening_npm3=cubic_softening,
        camera_trace_path=camera_trace_path,
        camera_frame_count=camera_frame_count,
        camera_mean_lat_offset_m=camera_mean_lat,
        camera_mean_theta_deg=camera_mean_theta,
        camera_validation_note=camera_note,
    )

    notes = [
        "online eigenvalue is reported as the absolute lateral stiffness slope from the probe window",
        "beam-column fit is a reduced-order validation on the same probe rows",
    ]
    if camera_rows:
        notes.append("optional ArUco/video trace was incorporated as a geometry validation input")
    else:
        notes.append("no optional camera trace was supplied; geometry validation is robot-only")

    return TrialAnalysis(
        sync_csv_path=sync_csv_path,
        probe_fit=probe_fit,
        beam_column_fit=beam_fit,
        notes=notes,
    )



def analysis_to_dict(analysis: TrialAnalysis) -> Dict[str, Any]:
    payload = asdict(analysis)
    payload["probe_fit"] = asdict(analysis.probe_fit)
    payload["beam_column_fit"] = asdict(analysis.beam_column_fit)
    return payload



def write_analysis_json(analysis: TrialAnalysis, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(analysis_to_dict(analysis), handle, indent=2)



def render_text_report(analysis: TrialAnalysis) -> str:
    probe = analysis.probe_fit
    beam = analysis.beam_column_fit
    lines = [
        f"Sync CSV             : {analysis.sync_csv_path}",
        f"Probe samples        : {probe.sample_count}",
        f"Probe slope          : {probe.raw_slope_npm:.6f} N/m",
        f"Online eigenvalue    : {probe.online_lateral_eigenvalue_npm:.6f} N/m",
        f"Residual RMS         : {probe.residual_rms_n:.6f} N",
        f"Boundary condition   : {beam.boundary_condition}",
        f"Effective length     : {beam.effective_length_m:.6f} m",
        f"EI_eff               : {beam.effective_bending_stiffness_nm2:.6f} N.m^2",
        f"Critical load        : {beam.critical_load_n:.6f} N",
        f"Cubic softening      : {beam.cubic_softening_npm3:.6e} N/m^3",
        f"Camera trace         : {beam.camera_trace_path or 'missing'}",
        f"Camera note          : {beam.camera_validation_note}",
    ]
    if analysis.notes:
        lines.append("")
        lines.append("Notes:")
        for note in analysis.notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)



def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Post-process a single-arm probe trial")
    parser.add_argument("sync_csv_path", help="Single-arm harness sync CSV path")
    parser.add_argument("--camera-trace", default=None, help="Optional ArUco/video trace CSV or JSONL")
    parser.add_argument("--boundary-condition", default=DEFAULT_BOUNDARY_CONDITION, help="Boundary condition label used for the reduced beam fit")
    parser.add_argument("--effective-length-m", type=float, default=None, help="Override the effective beam length used by the reduced beam fit")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()



def main() -> int:
    args = _parse_args()
    analysis = analyze_trial(
        args.sync_csv_path,
        camera_trace_path=args.camera_trace,
        boundary_condition=args.boundary_condition,
        effective_length_m=args.effective_length_m,
    )
    print(render_text_report(analysis))
    if args.output:
        write_analysis_json(analysis, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
