#!/usr/bin/env python3
"""
Compare hardware and simulation ablation summary CSV files.

The ablated hardware harness writes a summary.csv intended to match the main
simulation summary contract closely enough for direct comparison. This script
loads one hardware summary and one simulation summary, groups rows by
(case, K_lat_Npm), computes simple aggregate statistics, and writes a
comparison-friendly CSV if requested.

Typical usage:
  python3 tools/compare_sim_hw.py --hardware-summary hw/summary.csv --sim-summary sim/summary.csv
  python3 tools/compare_sim_hw.py --hardware-summary hw_dir --sim-summary sim_dir --write-csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_METRICS = [
    "t_contact_s",
    "t_mb_zero_s",
    "t_mc_zero_s",
    "t_w_3mm_s",
    "t_phi_4deg_s",
    "P_s_at_onset_N",
    "P_s_max_N",
]


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


def resolve_summary_csv(path: str) -> str:
    if os.path.isfile(path):
        return path
    candidates: List[str] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name == "summary.csv":
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise FileNotFoundError(f"No summary.csv found under {path}")
    candidates.sort()
    return candidates[-1]


def load_summary_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows: Iterable[Dict[str, str]], metrics: List[str]) -> Dict[Tuple[str, float], Dict[str, float]]:
    grouped: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for row in rows:
        case = row.get("case", "")
        k_lat = _to_float(row.get("K_lat_Npm"))
        key = (case, k_lat)
        bucket = grouped.setdefault(
            key,
            {
                "count": 0,
                "passed_count": 0,
                "K_lat_meas_values": [],
                **{metric: [] for metric in metrics},
            },
        )
        bucket["count"] += 1
        passed_text = str(row.get("passed", "")).strip().lower()
        if passed_text in {"1", "true", "yes"}:
            bucket["passed_count"] += 1
        k_lat_meas = _to_float(row.get("K_lat_meas_Npm"))
        if _is_finite(k_lat_meas):
            bucket["K_lat_meas_values"].append(k_lat_meas)
        for metric in metrics:
            value = _to_float(row.get(metric))
            if _is_finite(value):
                bucket[metric].append(value)

    aggregated: Dict[Tuple[str, float], Dict[str, float]] = {}
    for key, bucket in grouped.items():
        output: Dict[str, float] = {
            "count": float(bucket["count"]),
            "pass_rate": (bucket["passed_count"] / bucket["count"]) if bucket["count"] else float("nan"),
            "K_lat_meas_mean": mean_or_nan(bucket["K_lat_meas_values"]),
        }
        for metric in metrics:
            output[f"{metric}_mean"] = mean_or_nan(bucket[metric])
        aggregated[key] = output
    return aggregated


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def build_comparison_rows(
    hardware_rows: Iterable[Dict[str, str]],
    sim_rows: Iterable[Dict[str, str]],
    metrics: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    metrics = DEFAULT_METRICS if metrics is None else metrics
    hardware = aggregate_rows(hardware_rows, metrics)
    sim = aggregate_rows(sim_rows, metrics)
    keys = sorted(set(hardware) | set(sim))
    result: List[Dict[str, Any]] = []
    for case, k_lat in keys:
        hw = hardware.get((case, k_lat), {})
        sim_bucket = sim.get((case, k_lat), {})
        row: Dict[str, Any] = {
            "case": case,
            "K_lat_Npm": k_lat,
            "hardware_count": hw.get("count", float("nan")),
            "simulation_count": sim_bucket.get("count", float("nan")),
            "hardware_pass_rate": hw.get("pass_rate", float("nan")),
            "simulation_pass_rate": sim_bucket.get("pass_rate", float("nan")),
            "hardware_K_lat_meas_mean": hw.get("K_lat_meas_mean", float("nan")),
            "simulation_K_lat_meas_mean": sim_bucket.get("K_lat_meas_mean", float("nan")),
        }
        for metric in metrics:
            hw_mean = hw.get(f"{metric}_mean", float("nan"))
            sim_mean = sim_bucket.get(f"{metric}_mean", float("nan"))
            row[f"hardware_{metric}_mean"] = hw_mean
            row[f"simulation_{metric}_mean"] = sim_mean
            row[f"delta_{metric}"] = (
                hw_mean - sim_mean if _is_finite(hw_mean) and _is_finite(sim_mean) else float("nan")
            )
        result.append(row)
    return result


def write_comparison_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not rows:
        raise ValueError("No comparison rows to write")
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_rows(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"{row['case']} K={row['K_lat_Npm']:.1f}  pass(hw/sim)={row['hardware_pass_rate']:.3f}/{row['simulation_pass_rate']:.3f}  "
            f"P_s_max delta={row['delta_P_s_max_N']:.3f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare hardware and simulation summary CSV files")
    parser.add_argument("--hardware-summary", required=True, help="Hardware summary.csv file or directory")
    parser.add_argument("--sim-summary", required=True, help="Simulation summary.csv file or directory")
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Write sim_hw_comparison.csv alongside the hardware summary unless --output is set",
    )
    parser.add_argument("--output", default=None, help="Optional explicit output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hardware_path = resolve_summary_csv(args.hardware_summary)
    sim_path = resolve_summary_csv(args.sim_summary)
    rows = build_comparison_rows(load_summary_rows(hardware_path), load_summary_rows(sim_path))
    print(render_rows(rows))
    if args.write_csv:
        output_path = args.output or os.path.join(os.path.dirname(hardware_path), "sim_hw_comparison.csv")
        write_comparison_csv(rows, output_path)
        print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()