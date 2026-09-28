#!/usr/bin/env python3
"""Align Linux motor telemetry rows to written Windows RealSense video frames.

The Windows recorder CSV contains periodic SYNC events. Each SYNC row has:
  - Windows receive ``host_monotonic_ns``
  - Linux sender ``linux_monotonic_ns`` embedded in the payload

We robustly fit the affine cross-host clock map

    t_windows = a * t_linux + b

and apply it to ``motor_current_torque_telemetry.csv`` rows, whose logger stores
Linux ``host_monotonic_ns``. Each telemetry row is then assigned the nearest
*written* video frame and a signed alignment residual.

The UDP path introduces receive-latency jitter; the reported SYNC residuals are
therefore an empirical bound/diagnostic, not a hardware-trigger guarantee.
"""

from __future__ import annotations

import argparse
import csv
import re
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


def kv(payload: str) -> Dict[str, str]:
    return {k: v for k, v in KV_RE.findall(payload or "")}


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    # Center x to avoid conditioning loss from large monotonic timestamps.
    x0 = float(np.mean(x))
    xc = x - x0
    A = np.column_stack([xc, np.ones_like(xc)])
    a, c = np.linalg.lstsq(A, y, rcond=None)[0]
    b = float(c - a * x0)
    r = y - (a * x + b)
    return float(a), b, r


def robust_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if len(x) < 3:
        raise ValueError("Need at least 3 SYNC anchors")
    a0, b0, r0 = fit_affine(x, y)
    med = float(np.median(r0))
    mad = float(np.median(np.abs(r0 - med)))
    if mad <= 1e-12:
        keep = np.ones(len(x), dtype=bool)
    else:
        sigma = 1.4826 * mad
        keep = np.abs(r0 - med) <= 4.0 * sigma
        if int(keep.sum()) < 3:
            keep = np.ones(len(x), dtype=bool)
    a, b, _ = fit_affine(x[keep], y[keep])
    residual = y - (a * x + b)
    return a, b, residual, keep


def nearest_frame(frame_times_ns: List[int], frame_indices: List[int], target_ns: int):
    i = bisect_left(frame_times_ns, target_ns)
    candidates = []
    if i < len(frame_times_ns):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    if not candidates:
        return None, None
    best = min(candidates, key=lambda k: abs(frame_times_ns[k] - target_ns))
    return frame_indices[best], frame_times_ns[best] - target_ns


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("camera_sync_csv", type=Path)
    p.add_argument("motor_telemetry_csv", type=Path)
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()

    cam = read_csv(a.camera_sync_csv)
    motor = read_csv(a.motor_telemetry_csv)

    lx: List[float] = []
    wy: List[float] = []
    event_windows: Dict[str, float] = {}
    for row in cam:
        if row.get("record_type") != "event":
            continue
        event = (row.get("event") or "").upper()
        payload = row.get("payload") or ""
        data = kv(payload)
        try:
            win_s = float(row["host_monotonic_ns"]) * 1e-9
        except (KeyError, ValueError):
            continue
        event_windows.setdefault(event, win_s)
        if event == "SYNC":
            try:
                lx.append(float(data["linux_monotonic_ns"]) * 1e-9)
                wy.append(win_s)
            except (KeyError, ValueError):
                pass

    x = np.asarray(lx, dtype=float)
    y = np.asarray(wy, dtype=float)
    slope, intercept, residual, keep = robust_fit(x, y)

    frame_rows = [r for r in cam if r.get("record_type") == "frame"]
    if not frame_rows:
        raise ValueError("No written video frame rows found")
    frame_rows.sort(key=lambda r: int(r["host_monotonic_ns"]))
    frame_times = [int(r["host_monotonic_ns"]) for r in frame_rows]
    frame_indices = [int(r["video_frame_index"]) for r in frame_rows]

    out = a.output or a.motor_telemetry_csv.with_name(a.motor_telemetry_csv.stem + "_video_aligned.csv")
    extra = [
        "windows_monotonic_est_ns",
        "nearest_video_frame_index",
        "nearest_video_frame_dt_ms",
        "t_from_move_forward_start_s",
        "t_from_contact_s",
        "t_from_abort_s",
    ]
    fields = list(motor[0].keys()) + extra if motor else extra
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in motor:
            try:
                linux_s = float(row["host_monotonic_ns"]) * 1e-9
            except (KeyError, ValueError):
                w.writerow({**row, **{k: "" for k in extra}})
                continue
            win_s = slope * linux_s + intercept
            win_ns = int(round(win_s * 1e9))
            frame_idx, frame_dt_ns = nearest_frame(frame_times, frame_indices, win_ns)
            enriched = dict(row)
            enriched["windows_monotonic_est_ns"] = win_ns
            enriched["nearest_video_frame_index"] = frame_idx if frame_idx is not None else ""
            enriched["nearest_video_frame_dt_ms"] = (
                f"{frame_dt_ns / 1e6:.6f}" if frame_dt_ns is not None else ""
            )
            for ev, col in [
                ("MOVE_FORWARD_START", "t_from_move_forward_start_s"),
                ("CONTACT", "t_from_contact_s"),
                ("ABORT", "t_from_abort_s"),
            ]:
                enriched[col] = f"{win_s - event_windows[ev]:.9f}" if ev in event_windows else ""
            w.writerow(enriched)

    abs_ms = np.abs(residual) * 1e3
    print(f"SYNC anchors: {len(x)} total, {int(keep.sum())} used")
    print(f"Clock map: t_windows = {slope:.12f} * t_linux + {intercept:.9f}")
    print(f"Relative clock-rate offset: {(slope - 1.0) * 1e6:.3f} ppm")
    print(f"SYNC |residual| p95: {np.percentile(abs_ms, 95):.3f} ms")
    print(f"SYNC |residual| max: {np.max(abs_ms):.3f} ms")
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
