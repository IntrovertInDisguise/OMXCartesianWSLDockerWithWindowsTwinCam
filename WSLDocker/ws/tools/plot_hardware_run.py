#!/usr/bin/env python3
"""
Exhaustive post-run plotter for single-arm OMX hardware trials.

This is the hardware-run backend for tools/plot_logs.py.

Data sources
------------
1) motor_current_torque_telemetry.csv
   Used DIRECTLY for high-rate joint/motor plots:
   - measured joint position / velocity
   - measured Present Current (raw and A)
   - controller pre-safety current request
   - Dynamixel Goal Current register readback
   - controller request model-equivalent Nm
   - calibrated current-derived torque estimate, only when finite

2) single_arm_harness_v3_snapshot_enriched.csv
   Preferred for Cartesian / contact / compression / event plots.
   Falls back to the newest raw single_arm_harness_v3_snapshot.csv.

Scientific semantics
--------------------
- Present Current is measured motor current, NOT direct measured torque.
- Goal Current is command-register readback.
- pre_safety_request is the controller-side request before later safety/ramp
  processing.
- request_model_equivalent_Nm is a controller-model quantity.
- torque_est_Nm is plotted only when calibration has produced finite values.
- contact_wrench is controller-estimated unless an independent force sensor is
  introduced elsewhere.

The plotter writes PNG + vector PDF and a plot_manifest.csv audit. Numeric
time-series columns not covered by a named figure are automatically plotted
under raw/unclassified so logged numeric data are not silently ignored.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
matplotlib.rcParams.update(PUB_RC)


TIME_OR_METADATA = {
    # Generic / motor timebases
    "host_monotonic_ns",
    "host_wall_time_s",
    "rx_ros_time_s",
    "message_time_s",
    "timestamp",
    "ros_time_s",
    "telemetry_match_ros_time_s",
    "elapsed_compression_s",
    "t_from_move_forward_start_s",
    "t_from_contact_s",
    "t_from_abort_s",
    # Camera identifiers / absolute timestamps: selected diagnostics are handled
    # explicitly below; indices themselves are metadata.
    "camera_video_frame_index",
    "camera_capture_index",
    "camera_top_timestamp_ms",
    "camera_side_timestamp_ms",
}

STRING_OR_CATEGORICAL = {
    "record_type",
    "phase",
    "contact_mode",
    "run_status",
    "move_forward_start_time_source",
    "contact_time_source",
    "abort_time_source",
    "abort_reason",
    "stop_reason",
}

JOINTS = ("joint1", "joint2", "joint3", "joint4")


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def has_finite(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and finite_numeric(df[col]).notna().any()


def first_finite(series: pd.Series) -> Optional[float]:
    s = finite_numeric(series).dropna()
    return float(s.iloc[0]) if len(s) else None


def numeric_time(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Return relative seconds, absolute seconds if available, and source column."""
    for col in candidates:
        if col not in df.columns:
            continue
        s = finite_numeric(df[col])
        good = s.notna()
        if good.any():
            t_abs = s.to_numpy(dtype=float)
            t0 = float(s[good].iloc[0])
            return t_abs - t0, t_abs, col
    # Last-resort row index, not physical time.
    t = np.arange(len(df), dtype=float)
    return t, None, "row_index"


def discover_snapshot(run_folder: Path) -> Optional[Path]:
    root_enriched = run_folder / "single_arm_harness_v3_snapshot_enriched.csv"
    if root_enriched.exists():
        return root_enriched

    enriched = sorted(
        run_folder.rglob("single_arm_harness_v3_snapshot_enriched.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if enriched:
        return enriched[0]

    raw = sorted(
        run_folder.rglob("single_arm_harness_v3_snapshot.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return raw[0] if raw else None


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


class PlotAudit:
    def __init__(self):
        self.rows: List[Dict[str, str]] = []
        self.used: Dict[str, Set[str]] = {"motor": set(), "harness": set()}

    def mark_used(self, source: str, cols: Iterable[str]):
        self.used.setdefault(source, set()).update(c for c in cols if c)

    def add(self, source_file: str, column: str, semantic_name: str, unit: str,
            category: str, plotted: bool, plot_file: str = "", reason: str = ""):
        self.rows.append({
            "source_file": source_file,
            "column": column,
            "semantic_name": semantic_name,
            "unit": unit,
            "category": category,
            "plotted": "yes" if plotted else "no",
            "plot_file": plot_file,
            "reason_if_not_plotted": reason,
        })

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "source_file", "column", "semantic_name", "unit",
            "category", "plotted", "plot_file", "reason_if_not_plotted",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(self.rows)


def save_figure(fig, stem: Path) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    plt.close(fig)
    return out


def _contiguous_true_spans(t_abs: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    if t_abs is None or len(t_abs) == 0 or len(mask) != len(t_abs):
        return spans
    start = None
    last = None
    for t, m in zip(t_abs, mask):
        if not np.isfinite(t):
            continue
        if bool(m):
            if start is None:
                start = float(t)
            last = float(t)
        elif start is not None:
            spans.append((start, last if last is not None else start))
            start = None
            last = None
    if start is not None:
        spans.append((start, last if last is not None else start))
    return spans


def derive_hold_spans_abs(hdf: Optional[pd.DataFrame], h_abs: Optional[np.ndarray]) -> List[Tuple[float, float]]:
    if hdf is None or h_abs is None:
        return []
    if "quasistatic_phase" in hdf.columns:
        phase = hdf["quasistatic_phase"].astype(str).str.upper()
        return _contiguous_true_spans(h_abs, phase.eq("HOLD").to_numpy())
    # Backward-compatible fallback; future staircase runs should use the explicit field.
    if "phase" in hdf.columns:
        phase = hdf["phase"].astype(str).str.lower()
        return _contiguous_true_spans(h_abs, phase.eq("hold").to_numpy())
    return []


def derive_equilibrium_spans_abs(hdf: Optional[pd.DataFrame], h_abs: Optional[np.ndarray]) -> List[Tuple[float, float]]:
    if hdf is None or h_abs is None or "equilibrium_window" not in hdf.columns:
        return []
    flag = finite_numeric(hdf["equilibrium_window"]).fillna(0).to_numpy() >= 0.5
    return _contiguous_true_spans(h_abs, flag)


def event_times_abs(hdf: Optional[pd.DataFrame], h_abs: Optional[np.ndarray]) -> List[Tuple[float, str]]:
    if hdf is None or h_abs is None:
        return []
    events = []
    for col, label in (
        ("move_forward_started", "motion start"),
        ("contact_established", "contact"),
        ("abort_detected", "abort"),
        ("normal_move_end", "motion end"),
        ("u2d2_loss_after_contact", "U2D2 loss"),
    ):
        if col not in hdf.columns:
            continue
        s = finite_numeric(hdf[col]).fillna(0).to_numpy()
        prev = 0.0
        for i, v in enumerate(s):
            if v >= 0.5 and prev < 0.5 and i < len(h_abs) and np.isfinite(h_abs[i]):
                events.append((float(h_abs[i]), label))
                break
            prev = v
    return events


def decorate_axes(ax, time_abs: Optional[np.ndarray], time_rel: np.ndarray,
                  hold_spans_abs: Sequence[Tuple[float, float]],
                  equilibrium_spans_abs: Sequence[Tuple[float, float]],
                  events_abs: Sequence[Tuple[float, str]]):
    if time_abs is None:
        ax.grid(True, which="major", alpha=0.35)
        ax.minorticks_on()
        return

    finite = time_abs[np.isfinite(time_abs)]
    if len(finite) == 0:
        return
    t0 = float(finite[0])

    # HOLD windows: low-opacity shading.
    for a, b in hold_spans_abs:
        ax.axvspan(a - t0, b - t0, alpha=0.08)

    # Settled equilibrium sub-window: slightly stronger shading.
    for a, b in equilibrium_spans_abs:
        ax.axvspan(a - t0, b - t0, alpha=0.14)

    # Event lines.
    seen = set()
    for t, label in events_abs:
        ax.axvline(t - t0, linestyle="--", linewidth=1.0,
                   label=label if label not in seen else None)
        seen.add(label)

    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()


def plot_subplots(
    df: pd.DataFrame,
    t_rel: np.ndarray,
    t_abs: Optional[np.ndarray],
    series: Sequence[Tuple[str, str, str]],
    title: str,
    stem: Path,
    audit: PlotAudit,
    source_key: str,
    source_file: str,
    hold_spans_abs=(),
    equilibrium_spans_abs=(),
    events_abs=(),
) -> Optional[List[Path]]:
    valid = [(c, lab, unit) for c, lab, unit in series if has_finite(df, c)]
    if not valid:
        return None

    fig, axes = plt.subplots(len(valid), 1, figsize=(15, max(4.0 * len(valid), 6.5)),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for ax, (col, label, unit) in zip(axes, valid):
        y = finite_numeric(df[col]).to_numpy(dtype=float)
        ax.plot(t_rel, y, label=label)
        ax.set_ylabel(f"{label}\n({unit})" if unit else label)
        decorate_axes(ax, t_abs, t_rel, hold_spans_abs, equilibrium_spans_abs, events_abs)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    paths = save_figure(fig, stem)

    rel = str(paths[0])
    for col, label, unit in valid:
        audit.mark_used(source_key, [col])
        audit.add(source_file, col, label, unit, title, True, rel)
    return paths


def plot_overlays(
    df: pd.DataFrame,
    t_rel: np.ndarray,
    t_abs: Optional[np.ndarray],
    groups: Sequence[Tuple[str, Sequence[Tuple[str, str]], str]],
    title: str,
    stem: Path,
    audit: PlotAudit,
    source_key: str,
    source_file: str,
    hold_spans_abs=(),
    equilibrium_spans_abs=(),
    events_abs=(),
) -> Optional[List[Path]]:
    valid_groups = []
    for axis_label, curves, unit in groups:
        valid_curves = [(c, lab) for c, lab in curves if has_finite(df, c)]
        if valid_curves:
            valid_groups.append((axis_label, valid_curves, unit))
    if not valid_groups:
        return None

    fig, axes = plt.subplots(len(valid_groups), 1,
                             figsize=(15, max(4.2 * len(valid_groups), 6.5)),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for ax, (axis_label, curves, unit) in zip(axes, valid_groups):
        for col, label in curves:
            ax.plot(t_rel, finite_numeric(df[col]).to_numpy(dtype=float), label=label)
            audit.mark_used(source_key, [col])
            audit.add(source_file, col, label, unit, title, True, str(stem.with_suffix(".png")))
        ax.set_ylabel(f"{axis_label}\n({unit})" if unit else axis_label)
        decorate_axes(ax, t_abs, t_rel, hold_spans_abs, equilibrium_spans_abs, events_abs)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_figure(fig, stem)


def plot_motor(mdf: pd.DataFrame, run_folder: Path, save_root: Path,
               audit: PlotAudit, hdf: Optional[pd.DataFrame],
               h_abs: Optional[np.ndarray]):
    source = "motor_current_torque_telemetry.csv"
    if "record_type" in mdf.columns:
        mdf = mdf[mdf["record_type"].astype(str).eq("feedback")].copy()
    if len(mdf) == 0:
        return

    t, t_abs, _ = numeric_time(mdf, ("rx_ros_time_s", "message_time_s", "host_wall_time_s"))
    hold_spans = derive_hold_spans_abs(hdf, h_abs)
    eq_spans = derive_equilibrium_spans_abs(hdf, h_abs)
    events = event_times_abs(hdf, h_abs)
    joint_dir = save_root / "raw" / "joints"
    diag_dir = save_root / "raw" / "diagnostics"

    plot_subplots(
        mdf, t, t_abs,
        [(f"{j}_position_rad", f"Joint {i} measured position", "rad")
         for i, j in enumerate(JOINTS, 1)],
        "Measured joint positions", joint_dir / "joint_positions",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        mdf, t, t_abs,
        [(f"{j}_velocity_rad_s", f"Joint {i} measured velocity", "rad/s")
         for i, j in enumerate(JOINTS, 1)],
        "Measured joint velocities", joint_dir / "joint_velocities",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        mdf, t, t_abs,
        [(f"{j}_present_current_A", f"Joint {i} Present Current", "A")
         for i, j in enumerate(JOINTS, 1)],
        "Measured motor current — Present Current", joint_dir / "present_current_all_joints_A",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    # Per-joint current chain: controller request -> register -> measured current.
    for i, j in enumerate(JOINTS, 1):
        plot_overlays(
            mdf, t, t_abs,
            [(
                f"Joint {i}",
                [
                    (f"{j}_pre_safety_request_raw_latest", "Pre-safety request"),
                    (f"{j}_goal_current_register_raw_latest", "Goal Current register"),
                    (f"{j}_present_current_raw", "Present Current"),
                ],
                "raw current units",
            )],
            f"Joint {i} current chain — raw units",
            joint_dir / f"joint{i}_current_chain_raw",
            audit, "motor", source, hold_spans, eq_spans, events,
        )

        plot_overlays(
            mdf, t, t_abs,
            [(
                f"Joint {i}",
                [
                    (f"{j}_pre_safety_request_A_latest", "Pre-safety request"),
                    (f"{j}_goal_current_register_A_latest", "Goal Current register"),
                    (f"{j}_present_current_A", "Present Current"),
                ],
                "A",
            )],
            f"Joint {i} current chain",
            joint_dir / f"joint{i}_current_chain_A",
            audit, "motor", source, hold_spans, eq_spans, events,
        )

        # Derived electrical tracking differences.
        req = f"{j}_pre_safety_request_A_latest"
        goal = f"{j}_goal_current_register_A_latest"
        pres = f"{j}_present_current_A"
        if all(c in mdf.columns for c in (req, goal, pres)):
            d = mdf.copy()
            d[f"{j}_request_minus_goal_A"] = finite_numeric(d[req]) - finite_numeric(d[goal])
            d[f"{j}_goal_minus_present_A"] = finite_numeric(d[goal]) - finite_numeric(d[pres])
            d[f"{j}_request_minus_present_A"] = finite_numeric(d[req]) - finite_numeric(d[pres])
            plot_overlays(
                d, t, t_abs,
                [(
                    f"Joint {i}",
                    [
                        (f"{j}_request_minus_goal_A", "Request − Goal register"),
                        (f"{j}_goal_minus_present_A", "Goal register − Present"),
                        (f"{j}_request_minus_present_A", "Request − Present"),
                    ],
                    "A",
                )],
                f"Joint {i} current tracking differences",
                joint_dir / f"joint{i}_current_tracking_error_A",
                audit, "motor", source, hold_spans, eq_spans, events,
            )

    plot_subplots(
        mdf, t, t_abs,
        [(f"{j}_request_model_equivalent_Nm_latest",
          f"Joint {i} controller request model-equivalent torque", "N·m")
         for i, j in enumerate(JOINTS, 1)],
        "Controller request model-equivalent torque (not measured torque)",
        joint_dir / "controller_request_model_equivalent_torque",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    # Plot calibrated torque only when actually populated.
    torque_cols = [(f"{j}_torque_est_Nm", f"Joint {i} calibrated current-derived torque", "N·m")
                   for i, j in enumerate(JOINTS, 1)]
    if any(has_finite(mdf, c) for c, _, _ in torque_cols):
        plot_subplots(
            mdf, t, t_abs, torque_cols,
            "Calibrated current-derived torque estimate (not a direct torque sensor)",
            joint_dir / "calibrated_current_derived_torque",
            audit, "motor", source, hold_spans, eq_spans, events,
        )
    else:
        for c, label, unit in torque_cols:
            if c in mdf.columns:
                audit.mark_used("motor", [c])
                audit.add(source, c, label, unit, "torque", False, "",
                          "No finite calibrated values; torque calibration coefficients were not supplied.")

    plot_subplots(
        mdf, t, t_abs,
        [
            ("pre_safety_request_age_s", "Pre-safety request age", "s"),
            ("goal_register_age_s", "Goal Current register age", "s"),
        ],
        "Motor telemetry age diagnostics", diag_dir / "motor_telemetry_age",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    audit.mark_used("motor", ["record_type", "host_monotonic_ns", "host_wall_time_s",
                              "rx_ros_time_s", "message_time_s"])


def plot_harness(hdf: pd.DataFrame, save_root: Path, audit: PlotAudit):
    source = "single_arm_harness_v3_snapshot_enriched.csv"
    t, t_abs, _ = numeric_time(hdf, ("ros_time_s", "timestamp"))
    hold_spans = derive_hold_spans_abs(hdf, t_abs)
    eq_spans = derive_equilibrium_spans_abs(hdf, t_abs)
    events = event_times_abs(hdf, t_abs)

    cart_dir = save_root / "raw" / "cartesian"
    ctrl_dir = save_root / "raw" / "controller"
    diag_dir = save_root / "raw" / "diagnostics"

    plot_overlays(
        hdf, t, t_abs,
        [
            ("X", [("desired_x", "Commanded X"), ("ee_x", "Measured X")], "m"),
            ("Y", [("desired_y", "Commanded Y"), ("ee_y", "Measured Y")], "m"),
            ("Z", [("desired_z", "Commanded Z"), ("ee_z", "Measured Z")], "m"),
        ],
        "Cartesian command versus measured position",
        cart_dir / "xyz_commanded_vs_measured",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    # Derived Cartesian tracking error.
    if all(c in hdf.columns for c in ("desired_x", "desired_y", "desired_z", "ee_x", "ee_y", "ee_z")):
        d = hdf.copy()
        d["derived_err_x_m"] = finite_numeric(d["desired_x"]) - finite_numeric(d["ee_x"])
        d["derived_err_y_m"] = finite_numeric(d["desired_y"]) - finite_numeric(d["ee_y"])
        d["derived_err_z_m"] = finite_numeric(d["desired_z"]) - finite_numeric(d["ee_z"])
        d["derived_err_norm_m"] = np.sqrt(
            d["derived_err_x_m"] ** 2 + d["derived_err_y_m"] ** 2 + d["derived_err_z_m"] ** 2
        )
        plot_subplots(
            d, t, t_abs,
            [
                ("derived_err_x_m", "X tracking error", "m"),
                ("derived_err_y_m", "Y tracking error", "m"),
                ("derived_err_z_m", "Z tracking error", "m"),
                ("derived_err_norm_m", "Cartesian error norm", "m"),
            ],
            "Cartesian tracking error",
            cart_dir / "xyz_tracking_error",
            audit, "harness", source, hold_spans, eq_spans, events,
        )

    plot_overlays(
        hdf, t, t_abs,
        [(
            "Compression depth",
            [
                ("compression_depth_commanded_m", "Commanded"),
                ("compression_depth_measured_m", "Measured"),
            ],
            "m",
        )],
        "Compression depth — commanded versus measured",
        cart_dir / "compression_commanded_vs_measured",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        hdf, t, t_abs,
        [
            ("lateral_y_m", "Lateral Y", "m"),
            ("lateral_z_m", "Lateral Z", "m"),
            ("r_perp_m", "Lateral resultant r⊥", "m"),
        ],
        "Lateral response",
        cart_dir / "lateral_response",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        hdf, t, t_abs,
        [
            ("contact_fx", "Estimated contact force X", "N"),
            ("contact_fy", "Estimated contact force Y", "N"),
            ("contact_fz", "Estimated contact force Z", "N"),
            ("contact_force_norm", "Estimated contact force norm", "N"),
        ],
        "Controller-estimated contact force",
        cart_dir / "contact_force_estimated",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        hdf, t, t_abs,
        [
            ("contact_tx", "Estimated contact moment X", "N·m"),
            ("contact_ty", "Estimated contact moment Y", "N·m"),
            ("contact_tz", "Estimated contact moment Z", "N·m"),
        ],
        "Controller-estimated contact moment",
        cart_dir / "contact_moment_estimated",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        hdf, t, t_abs,
        [
            ("k_x_n_m", "Kx", "N/m"),
            ("k_y_n_m", "Ky", "N/m"),
            ("k_z_n_m", "Kz", "N/m"),
            ("k_lat_commanded_n_m", "Commanded Klat", "N/m"),
        ],
        "Cartesian stiffness state",
        ctrl_dir / "cartesian_stiffness",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    # Joint values on harness timebase are useful for cross-modal alignment,
    # although high-rate raw plots come from motor telemetry.
    plot_subplots(
        hdf, t, t_abs,
        [(f"joint{i}_pos", f"Joint {i} position on harness timebase", "rad") for i in range(1, 5)],
        "Joint position — harness synchronized timebase",
        ctrl_dir / "joint_positions_harness_timebase",
        audit, "harness", source, hold_spans, eq_spans, events,
    )
    plot_subplots(
        hdf, t, t_abs,
        [(f"joint{i}_vel", f"Joint {i} velocity on harness timebase", "rad/s") for i in range(1, 5)],
        "Joint velocity — harness synchronized timebase",
        ctrl_dir / "joint_velocities_harness_timebase",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        hdf, t, t_abs,
        [
            ("motor_telem_dt_ms", "Nearest motor sample Δt", "ms"),
            ("pre_safety_request_age_s", "Pre-safety request age", "s"),
            ("goal_register_age_s", "Goal register age", "s"),
        ],
        "Cross-stream telemetry synchronization",
        diag_dir / "cross_stream_telemetry_sync",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        hdf, t, t_abs,
        [
            ("camera_frame_dt_ms", "Nearest camera frame Δt", "ms"),
            ("camera_clock_fit_abs_residual_p95_ms", "Camera clock fit p95 residual", "ms"),
            ("camera_clock_fit_abs_residual_max_ms", "Camera clock fit max residual", "ms"),
        ],
        "Camera synchronization diagnostics",
        diag_dir / "camera_sync_diagnostics",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    # Camera top/side hardware timestamp skew, if available.
    if "camera_top_timestamp_ms" in hdf.columns and "camera_side_timestamp_ms" in hdf.columns:
        d = hdf.copy()
        d["derived_camera_top_minus_side_ms"] = (
            finite_numeric(d["camera_top_timestamp_ms"]) -
            finite_numeric(d["camera_side_timestamp_ms"])
        )
        plot_subplots(
            d, t, t_abs,
            [("derived_camera_top_minus_side_ms", "Top − side camera timestamp", "ms")],
            "Dual-camera timestamp skew",
            diag_dir / "camera_top_side_timestamp_skew",
            audit, "harness", source, hold_spans, eq_spans, events,
        )

    # Binary/event state.
    plot_subplots(
        hdf, t, t_abs,
        [
            ("contact_active", "Harness contact active", "0/1"),
            ("contact_valid", "Controller contact valid", "0/1"),
            ("move_forward_started", "Move started", "0/1"),
            ("contact_established", "Contact established", "0/1"),
            ("abort_detected", "Abort detected", "0/1"),
            ("u2d2_loss_after_contact", "U2D2 loss after contact", "0/1"),
            ("normal_move_end", "Normal move end", "0/1"),
            ("quasistatic_stage", "Quasistatic stage", "index"),
            ("equilibrium_window", "Equilibrium analysis window", "0/1"),
        ],
        "Experiment state and event flags",
        diag_dir / "experiment_state",
        audit, "harness", source, hold_spans, eq_spans, events,
    )

    audit.mark_used("harness", list(TIME_OR_METADATA | STRING_OR_CATEGORICAL))


def auto_plot_unassigned(df: pd.DataFrame, t: np.ndarray, t_abs: Optional[np.ndarray],
                         source_key: str, source_file: str, save_root: Path,
                         audit: PlotAudit, hdf_for_decor: Optional[pd.DataFrame],
                         h_abs_for_decor: Optional[np.ndarray]):
    """Plot every remaining numeric, non-time column so nothing numeric vanishes silently."""
    used = audit.used.setdefault(source_key, set())
    out_dir = save_root / "raw" / "unclassified"
    hold_spans = derive_hold_spans_abs(hdf_for_decor, h_abs_for_decor)
    eq_spans = derive_equilibrium_spans_abs(hdf_for_decor, h_abs_for_decor)
    events = event_times_abs(hdf_for_decor, h_abs_for_decor)

    for col in df.columns:
        if col in used:
            continue
        if col in TIME_OR_METADATA:
            audit.add(source_file, col, col, "", "metadata/timebase", False, "",
                      "Used as timebase/identifier or intentionally not treated as a dependent time-series.")
            used.add(col)
            continue
        if col in STRING_OR_CATEGORICAL:
            audit.add(source_file, col, col, "", "categorical", False, "",
                      "Categorical/text field; not a numeric time-series.")
            used.add(col)
            continue

        s = finite_numeric(df[col])
        if not s.notna().any():
            audit.add(source_file, col, col, "", "unclassified", False, "",
                      "No finite numeric values.")
            used.add(col)
            continue

        vals = s.dropna()
        if len(vals) and np.nanmax(vals.to_numpy()) == np.nanmin(vals.to_numpy()):
            # Still audit constants, but do not make dozens of flat figures.
            audit.add(source_file, col, col, "", "constant", False, "",
                      f"Finite but constant over run: {vals.iloc[0]}")
            used.add(col)
            continue

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(t, s.to_numpy(dtype=float), label=col)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(col)
        ax.set_title(f"Unclassified logged time-series: {col}")
        decorate_axes(ax, t_abs, t, hold_spans, eq_spans, events)
        ax.legend(loc="best")
        fig.tight_layout()
        stem = out_dir / safe_name(col)
        paths = save_figure(fig, stem)
        audit.add(source_file, col, col, "", "unclassified", True, str(paths[0]), "")
        used.add(col)


def build_quasistatic_plateau_csv(hdf: pd.DataFrame, out_path: Path):
    """
    If future staircase columns are present, write one row per stage using the
    equilibrium_window rows. This is intentionally generic and only summarizes
    columns that are available.
    """
    needed = {"quasistatic_stage", "equilibrium_window"}
    if not needed.issubset(hdf.columns):
        return

    stage = finite_numeric(hdf["quasistatic_stage"])
    eq = finite_numeric(hdf["equilibrium_window"]).fillna(0) >= 0.5
    work = hdf.loc[eq & stage.notna()].copy()
    if len(work) == 0:
        return
    work["quasistatic_stage"] = stage.loc[work.index].astype(int)

    candidates = [
        "stage_target_x_m", "stage_compression_m",
        "compression_depth_commanded_m", "compression_depth_measured_m",
        "ee_x", "ee_y", "ee_z",
        "lateral_y_m", "lateral_z_m", "r_perp_m",
        "contact_fx", "contact_fy", "contact_fz", "contact_force_norm",
        "k_x_n_m", "k_y_n_m", "k_z_n_m", "k_lat_commanded_n_m",
    ]
    numeric = [c for c in candidates if has_finite(work, c)]
    rows = []
    for st, g in work.groupby("quasistatic_stage", sort=True):
        row = {"quasistatic_stage": int(st), "n_equilibrium_samples": len(g)}
        for c in numeric:
            x = finite_numeric(g[c]).dropna()
            if len(x):
                row[f"{c}_median"] = float(x.median())
                row[f"{c}_mean"] = float(x.mean())
                row[f"{c}_std"] = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Exhaustive OMX single-arm hardware run plotter")
    p.add_argument("--run-folder", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=None)
    args = p.parse_args(argv)

    run = args.run_folder.resolve()
    if not run.is_dir():
        print(f"[PLOT] ERROR: run folder not found: {run}")
        return 2

    save_root = (args.save_dir.resolve() if args.save_dir else (run / "plots").resolve())
    save_root.mkdir(parents=True, exist_ok=True)

    motor_path = run / "motor_current_torque_telemetry.csv"
    snap_path = discover_snapshot(run)

    print(f"[PLOT] run_folder={run}")
    print(f"[PLOT] save_root={save_root}")
    print(f"[PLOT] motor_csv={motor_path if motor_path.exists() else 'MISSING'}")
    print(f"[PLOT] harness_snapshot={snap_path if snap_path else 'MISSING'}")

    audit = PlotAudit()

    hdf = None
    h_abs = None
    h_t = None
    if snap_path is not None:
        hdf = pd.read_csv(snap_path, low_memory=False)
        h_t, h_abs, _ = numeric_time(hdf, ("ros_time_s", "timestamp"))
        plot_harness(hdf, save_root, audit)

    mdf = None
    if motor_path.exists():
        mdf = pd.read_csv(motor_path, low_memory=False)
        plot_motor(mdf, run, save_root, audit, hdf, h_abs)

    # Exhaustiveness audit / fallback plots.
    if mdf is not None and len(mdf):
        feedback = mdf
        if "record_type" in feedback.columns:
            feedback = feedback[feedback["record_type"].astype(str).eq("feedback")].copy()
        mt, mabs, _ = numeric_time(feedback, ("rx_ros_time_s", "message_time_s", "host_wall_time_s"))
        auto_plot_unassigned(
            feedback, mt, mabs, "motor", motor_path.name,
            save_root, audit, hdf, h_abs
        )

    if hdf is not None and len(hdf):
        auto_plot_unassigned(
            hdf, h_t, h_abs, "harness", snap_path.name,
            save_root, audit, hdf, h_abs
        )
        build_quasistatic_plateau_csv(
            hdf, save_root / "quasistatic" / "equilibrium_plateaus.csv"
        )

    # Explicit missing-source audit.
    if not motor_path.exists():
        audit.add(motor_path.name, "*", "High-rate motor telemetry", "", "source", False, "",
                  "motor_current_torque_telemetry.csv is missing.")
    if snap_path is None:
        audit.add("harness snapshot", "*", "Harness/controller telemetry", "", "source", False, "",
                  "No enriched or raw harness snapshot was found.")

    manifest = save_root / "plot_manifest.csv"
    audit.write(manifest)

    print(f"[PLOT] manifest={manifest}")
    print(f"[PLOT] plots complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
