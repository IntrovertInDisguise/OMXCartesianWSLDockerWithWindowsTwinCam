#!/usr/bin/env python3
"""Fit Windows-camera monotonic time to Linux/ROS wall time from SYNC beacons."""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")

def parse_payload(payload: str) -> Dict[str, str]:
    return {k: v for k, v in KV_RE.findall(payload or "")}

def load_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    residual = y - (a * x + b)
    return float(a), float(b), residual

def robust_fit(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        raise ValueError("Need at least two SYNC events for an affine clock fit")
    a0, b0, r0 = fit_affine(x, y)
    med = float(np.median(r0))
    mad = float(np.median(np.abs(r0 - med)))
    if mad <= 1e-12:
        keep = np.ones(len(x), dtype=bool)
    else:
        sigma = 1.4826 * mad
        keep = np.abs(r0 - med) <= 4.0 * sigma
        if keep.sum() < 2:
            keep = np.ones(len(x), dtype=bool)
    a, b, _ = fit_affine(x[keep], y[keep])
    residual_all = y - (a * x + b)
    return a, b, residual_all, keep

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sync_csv", type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    rows = load_rows(args.sync_csv)
    sync_x, sync_y = [], []
    start_ros: Optional[float] = None
    end_ros: Optional[float] = None
    for row in rows:
        if row.get("record_type") != "event":
            continue
        event = (row.get("event") or "").upper()
        kv = parse_payload(row.get("payload") or "")
        if event == "SYNC":
            try:
                sync_x.append(float(row["host_monotonic_ns"]) * 1e-9)
                sync_y.append(float(kv["linux_wall_time_s"]))
            except (KeyError, TypeError, ValueError):
                pass
        elif event == "MOVE_FORWARD_START" and start_ros is None:
            try: start_ros = float(kv["ros_log_time_s"])
            except (KeyError, ValueError): pass
        elif event == "MOVE_FORWARD_END" and end_ros is None:
            try: end_ros = float(kv["ros_log_time_s"])
            except (KeyError, ValueError): pass
    x = np.asarray(sync_x, dtype=float)
    y = np.asarray(sync_y, dtype=float)
    a, b, residual, keep = robust_fit(x, y)
    output = args.output or args.sync_csv.with_name(args.sync_csv.stem + "_robot_aligned.csv")
    frame_rows = [r for r in rows if r.get("record_type") == "frame"]
    if not frame_rows:
        raise ValueError("No written frame rows found in sync CSV")
    fieldnames = list(frame_rows[0].keys()) + ["robot_time_est_s", "t_from_move_forward_start_s", "t_from_move_forward_end_s"]
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in frame_rows:
            win_s = float(row["host_monotonic_ns"]) * 1e-9
            robot_s = a * win_s + b
            out = dict(row)
            out["robot_time_est_s"] = f"{robot_s:.9f}"
            out["t_from_move_forward_start_s"] = f"{robot_s - start_ros:.9f}" if start_ros is not None else ""
            out["t_from_move_forward_end_s"] = f"{robot_s - end_ros:.9f}" if end_ros is not None else ""
            w.writerow(out)
    abs_r_ms = np.abs(residual) * 1e3
    print(f"SYNC anchors: {len(x)} total, {int(keep.sum())} used")
    print(f"Clock map: robot_time_s = {a:.12f} * windows_monotonic_s + {b:.9f}")
    print(f"SYNC fit |residual|: p95={float(np.percentile(abs_r_ms,95)):.3f} ms, max={float(np.max(abs_r_ms)):.3f} ms")
    print(f"Wrote: {output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
