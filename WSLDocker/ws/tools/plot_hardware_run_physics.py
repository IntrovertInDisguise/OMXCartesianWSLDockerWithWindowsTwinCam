#!/usr/bin/env python3
"""
Physics-grounded post-run time-series plotter for single-arm OMX hardware trials.

Primary data streams
--------------------
1) motor_current_torque_telemetry.csv
   Direct high-rate motor measurements and current commands:
   - measured joint angle q [rad]
   - measured joint angular velocity dq/dt [rad/s]
   - measured motor current I_meas [A]
   - commanded current before robot limits [A]
   - commanded motor current after robot limits [A]
   - calculated command-scale torque quantity [N·m-equivalent]
   - calibrated current-derived torque [N·m], only when calibration exists

2) the newest enriched experiment snapshot CSV
   Cartesian and experiment quantities:
   - commanded/measured Cartesian position [m]
   - commanded/measured compression [m]
   - lateral displacement [m]
   - calculated contact force [N] and moment [N·m]
   - commanded Cartesian stiffness [N/m]
   - synchronization and event diagnostics

Physics/measurement rules
-------------------------
- Every plot label says commanded, measured, or calculated.
- Present motor current is measured current, not measured torque.
- Current-derived torque is only quantitative when calibration coefficients exist.
- Calculated contact wrench is shown with its impedance-law/filter formula.
- Commanded current plots show the robot current-command inequality.
- PNG and vector PDF are written for each named figure.
- plot_manifest.csv records the physical label, unit, and disposition of every column.
"""


from __future__ import annotations

import argparse
import csv
import json
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
from cycler import cycler


PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 17,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.1,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.08,
    "figure.constrained_layout.w_pad": 0.08,
    "axes.prop_cycle": cycler(color=[
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
    ]),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.20,
}
matplotlib.rcParams.update(PUB_RC)

LINE_STYLES = ("-", "--", "-.", ":")


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
    """Prefer an explicitly repaired historical snapshot, then newest enriched snapshot."""
    repaired = run_folder / "snapshot_enriched_joint_names_repaired.csv"

    if repaired.is_file():
        return repaired

    enriched = sorted(
        run_folder.rglob("*snapshot_enriched.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if enriched:
        return enriched[0]

    raw = sorted(
        [p for p in run_folder.rglob("*snapshot.csv") if "enriched" not in p.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return raw[0] if raw else None


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


class PlotAudit:
    def __init__(self):
        self.rows: List[Dict[str, str]] = []
        self.used: Dict[str, Set[str]] = {"motor": set(), "state": set()}

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


def decorate_axes(
    ax,
    t_abs,
    t_rel,
    hold_spans_abs=(),
    equilibrium_spans_abs=(),
    events_abs=(),
):
    # Time-zero for absolute annotations.
    t0 = None

    if t_abs is not None:
        t_abs_arr = np.asarray(
            t_abs,
            dtype=float,
        )

        finite_abs = t_abs_arr[
            np.isfinite(t_abs_arr)
        ]

        if len(finite_abs):
            t0 = float(
                finite_abs[0]
            )

    if t0 is None:
        t0 = 0.0

    # Hold/equilibrium windows are context annotations only.
    for a, b in hold_spans_abs:
        ax.axvspan(
            a - t0,
            b - t0,
            alpha=0.08,
        )

    for a, b in equilibrium_spans_abs:
        ax.axvspan(
            a - t0,
            b - t0,
            alpha=0.14,
        )

    # Event lines are annotations only.
    seen = set()

    for event_t, label in events_abs:
        ax.axvline(
            event_t - t0,
            linestyle="--",
            linewidth=1.0,
            label=label if label not in seen else None,
        )

        seen.add(
            label
        )

    # Scale y to 120% of actual plotted non-vertical time-series data.
    # Reference/safety/event/start/stop lines must not flatten trends.
    data_y_parts = []

    for line in ax.lines:
        x_line = np.asarray(
            line.get_xdata(),
            dtype=float,
        ).reshape(-1)

        y_line = np.asarray(
            line.get_ydata(),
            dtype=float,
        ).reshape(-1)

        is_vertical = (
            len(x_line) == 2
            and np.all(np.isfinite(x_line))
            and np.isclose(
                x_line[0],
                x_line[1],
            )
        )

        if is_vertical:
            continue

        y_line = y_line[
            np.isfinite(y_line)
        ]

        if len(y_line):
            data_y_parts.append(
                y_line
            )

    if data_y_parts:
        y_all = np.concatenate(
            data_y_parts
        )

        y_min = float(
            np.min(y_all)
        )

        y_max = float(
            np.max(y_all)
        )

        y_range = y_max - y_min

        if y_range > 0.0:
            y_pad = 0.10 * y_range
        else:
            y_pad = 0.10 * max(
                abs(y_min),
                1e-6,
            )

        ax.set_ylim(
            y_min - y_pad,
            y_max + y_pad,
        )

    # Mark explicit plotted start/stop.
    t_rel_arr = np.asarray(
        t_rel,
        dtype=float,
    )

    finite_t = t_rel_arr[
        np.isfinite(t_rel_arr)
    ]

    if len(finite_t):
        t_start = float(
            np.min(finite_t)
        )

        t_stop = float(
            np.max(finite_t)
        )

        t_range = t_stop - t_start

        t_pad = (
            0.01 * t_range
            if t_range > 0.0
            else 0.01
        )

        ax.axvline(
            t_start,
            linestyle="-",
            linewidth=1.8,
            label="START",
        )

        ax.axvline(
            t_stop,
            linestyle="-",
            linewidth=1.8,
            label="STOP",
        )

        ax.set_xlim(
            t_start - t_pad,
            t_stop + t_pad,
        )

    ax.grid(
        True,
        which="major",
        alpha=0.35,
    )

    ax.minorticks_on()





ENCODER_COUNTS_PER_REV = 4096.0
ENCODER_DQ_RAD = 2.0 * np.pi / ENCODER_COUNTS_PER_REV

_OMX_JOINT_ORIGINS_M = np.array([
    [0.0120, 0.0, 0.0170],
    [0.0000, 0.0, 0.0595],
    [0.0240, 0.0, 0.1280],
    [0.1240, 0.0, 0.0000],
], dtype=float)

_OMX_JOINT_AXES = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=float)

_OMX_TIP_OFFSET_M = np.array(
    [0.1260, 0.0, 0.0],
    dtype=float,
)


def _axis_angle_rotation(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = np.cos(theta)
    sn = np.sin(theta)
    C = 1.0 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*sn, x*z*C + y*sn],
        [y*x*C + z*sn,  c + y*y*C,    y*z*C - x*sn],
        [z*x*C - y*sn,  z*y*C + x*sn, c + z*z*C],
    ], dtype=float)


def omx_fk_and_position_jacobian(q):
    q = np.asarray(q, dtype=float).reshape(4)

    R = np.eye(3, dtype=float)
    p0 = np.zeros(3, dtype=float)

    origins_world = []
    axes_world = []

    for origin_local, axis_local, qi in zip(
        _OMX_JOINT_ORIGINS_M,
        _OMX_JOINT_AXES,
        q,
    ):
        p0 = p0 + R @ origin_local

        origins_world.append(p0.copy())
        axes_world.append((R @ axis_local).copy())

        R = R @ _axis_angle_rotation(
            axis_local,
            qi,
        )

    p_ee = p0 + R @ _OMX_TIP_OFFSET_M

    Jv = np.zeros((3, 4), dtype=float)

    for i in range(4):
        Jv[:, i] = np.cross(
            axes_world[i],
            p_ee - origins_world[i],
        )

    return p_ee, Jv


def add_proprioceptive_resolution(df):
    r"""
    Joint encoder least count:
        Delta q_LC = 2*pi/N

    Logged relative Cartesian quantity:
        Delta x = x_FK(q_k) - x_FK(q_ref)

    First-order conservative encoder quantization bound:
        B_Delta_x =
            (Delta q_LC/2)
            sum_i (
                |J_v,ji(q_k)|
                +
                |J_v,ji(q_ref)|
            )

    Therefore:
        |epsilon_Delta_x| <= B_Delta_x

    The reference Cartesian position is recovered from the exact
    definitions used when the CSV was generated:

        x_ref = x_FK - compression_depth_measured
        y_ref = y_FK - lateral_y
        z_ref = z_FK - lateral_z

    A logged joint-state row whose FK is nearest that recorded
    reference position supplies q_ref.

    This is an encoder-quantization bound only.
    Mechanical compliance, backlash, structural deflection, gearbox
    effects, calibration error, and model error can make the actual
    Cartesian uncertainty larger.
    """
    out = df.copy()

    qcols = [
        "joint1_pos",
        "joint2_pos",
        "joint3_pos",
        "joint4_pos",
    ]

    required = qcols + [
        "ee_x",
        "ee_y",
        "ee_z",
        "compression_depth_measured_m",
        "lateral_y_m",
        "lateral_z_m",
    ]

    if not all(c in out.columns for c in required):
        return out

    def num(col):
        return pd.to_numeric(
            out[col],
            errors="coerce",
        ).to_numpy(dtype=float)

    qmat = np.column_stack([
        num(c)
        for c in qcols
    ])

    ee = np.column_stack([
        num("ee_x"),
        num("ee_y"),
        num("ee_z"),
    ])

    x_ref_samples = (
        num("ee_x")
        - num("compression_depth_measured_m")
    )

    y_ref_samples = (
        num("ee_y")
        - num("lateral_y_m")
    )

    z_ref_samples = (
        num("ee_z")
        - num("lateral_z_m")
    )

    refs = np.array([
        np.nanmedian(x_ref_samples),
        np.nanmedian(y_ref_samples),
        np.nanmedian(z_ref_samples),
    ])

    valid_ref_rows = (
        np.all(np.isfinite(ee), axis=1)
        & np.all(np.isfinite(qmat), axis=1)
    )

    if not np.any(valid_ref_rows):
        return out

    dist = np.full(
        len(out),
        np.inf,
        dtype=float,
    )

    dist[valid_ref_rows] = np.linalg.norm(
        ee[valid_ref_rows]
        - refs[None, :],
        axis=1,
    )

    ref_index = int(
        np.argmin(dist)
    )

    q_ref = qmat[ref_index]

    p_ref_fk, Jv_ref = (
        omx_fk_and_position_jacobian(
            q_ref
        )
    )

    ref_fk_mismatch = float(
        np.linalg.norm(
            p_ref_fk - refs
        )
    )

    n = len(out)

    fk = np.full(
        (n, 3),
        np.nan,
        dtype=float,
    )

    relative_bound = np.full(
        (n, 3),
        np.nan,
        dtype=float,
    )

    ref_half_bound = (
        0.5
        * ENCODER_DQ_RAD
        * np.sum(
            np.abs(Jv_ref),
            axis=1,
        )
    )

    for k, q in enumerate(qmat):

        if not np.all(np.isfinite(q)):
            continue

        p_ee, Jv = (
            omx_fk_and_position_jacobian(
                q
            )
        )

        fk[k, :] = p_ee

        current_half_bound = (
            0.5
            * ENCODER_DQ_RAD
            * np.sum(
                np.abs(Jv),
                axis=1,
            )
        )

        relative_bound[k, :] = (
            current_half_bound
            + ref_half_bound
        )

    out["fk_check_x_m"] = fk[:, 0]
    out["fk_check_y_m"] = fk[:, 1]
    out["fk_check_z_m"] = fk[:, 2]

    out["encoder_relative_x_bound_m"] = (
        relative_bound[:, 0]
    )

    out["encoder_relative_y_bound_m"] = (
        relative_bound[:, 1]
    )

    out["encoder_relative_z_bound_m"] = (
        relative_bound[:, 2]
    )

    out["encoder_relative_r_bound_m"] = np.sqrt(
        relative_bound[:, 1]**2
        + relative_bound[:, 2]**2
    )

    out.attrs["reference_xyz_m"] = refs
    out.attrs["reference_row_index"] = ref_index
    out.attrs["reference_q_rad"] = q_ref
    out.attrs["reference_fk_mismatch_m"] = (
        ref_fk_mismatch
    )

    return out

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
    reference_lines=(),
) -> Optional[List[Path]]:
    valid = [(c, lab, unit) for c, lab, unit in series if has_finite(df, c)]
    if not valid:
        return None

    fig, axes = plt.subplots(len(valid), 1, figsize=(18, max(4.0 * len(valid), 6.5)),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for ax, (col, label, unit) in zip(axes, valid):
        y = finite_numeric(df[col]).to_numpy(dtype=float)
        ax.plot(t_rel, y, label=label)
        ax.set_ylabel(f"{label}\n({unit})" if unit else label, labelpad=10)
        decorate_axes(ax, t_abs, t_rel, hold_spans_abs, equilibrium_spans_abs, events_abs)
        for y_ref, ref_label in reference_lines:
            ax.axhline(y_ref, linestyle=":", linewidth=1.6, color="0.15", label=ref_label)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
        )
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, wrap=True)

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
    reference_lines=(),
) -> Optional[List[Path]]:
    valid_groups = []
    for axis_label, curves, unit in groups:
        valid_curves = [(c, lab) for c, lab in curves if has_finite(df, c)]
        if valid_curves:
            valid_groups.append((axis_label, valid_curves, unit))
    if not valid_groups:
        return None

    fig, axes = plt.subplots(len(valid_groups), 1,
                             figsize=(18, max(4.2 * len(valid_groups), 6.5)),
                             sharex=True, squeeze=False)
    axes = axes.flatten()
    for ax, (axis_label, curves, unit) in zip(axes, valid_groups):
        for curve_i, (col, label) in enumerate(curves):
            ax.plot(
                t_rel,
                finite_numeric(df[col]).to_numpy(dtype=float),
                label=label,
                linestyle=LINE_STYLES[curve_i % len(LINE_STYLES)],
            )
            audit.mark_used(source_key, [col])
            audit.add(source_file, col, label, unit, title, True, str(stem.with_suffix(".png")))
        ax.set_ylabel(f"{axis_label}\n({unit})" if unit else axis_label, labelpad=10)
        decorate_axes(ax, t_abs, t_rel, hold_spans_abs, equilibrium_spans_abs, events_abs)
        for y_ref, ref_label in reference_lines:
            ax.axhline(y_ref, linestyle=":", linewidth=1.6, color="0.15", label=ref_label)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
        )
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, wrap=True)

    return save_figure(fig, stem)


def plot_motor(mdf: pd.DataFrame, run_folder: Path, save_root: Path,
               audit: PlotAudit, sdf: Optional[pd.DataFrame],
               s_abs: Optional[np.ndarray],
               current_limit_count: float,
               current_a_per_count: float,
               command_scale_count_per_nm: float):
    source = "motor_current_torque_telemetry.csv"
    if "record_type" in mdf.columns:
        mdf = mdf[mdf["record_type"].astype(str).eq("feedback")].copy()
    if len(mdf) == 0:
        return

    t, t_abs, _ = numeric_time(mdf, ("rx_ros_time_s", "message_time_s", "host_wall_time_s"))
    hold_spans = derive_hold_spans_abs(sdf, s_abs)
    eq_spans = derive_equilibrium_spans_abs(sdf, s_abs)
    events = event_times_abs(sdf, s_abs)
    joint_dir = save_root / "raw" / "joints"
    diag_dir = save_root / "raw" / "diagnostics"

    limit_a = current_limit_count * current_a_per_count
    ref_a = [
        (+limit_a, rf"$+I_{{lim}}$ ({limit_a:.3f} A)"),
        (-limit_a, rf"$-I_{{lim}}$ ({-limit_a:.3f} A)"),
    ]
    ref_count = [
        (+current_limit_count, rf"$+u_{{lim}}$ ({current_limit_count:.0f} counts)"),
        (-current_limit_count, rf"$-u_{{lim}}$ ({-current_limit_count:.0f} counts)"),
    ]

    plot_subplots(
        mdf, t, t_abs,
        [(f"{j}_position_rad", rf"Measured $q_{i}$", "rad")
         for i, j in enumerate(JOINTS, 1)],
        "Measured joint angle $q_i$ [rad]",
        joint_dir / "measured_joint_angle",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    plot_subplots(
        mdf, t, t_abs,
        [(f"{j}_velocity_rad_s", rf"Measured $\dot q_{i}$", "rad/s")
         for i, j in enumerate(JOINTS, 1)],
        r"Measured joint angular velocity $\dot q_i$ [rad/s]",
        joint_dir / "measured_joint_angular_velocity",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    plot_overlays(
        mdf,
        t,
        t_abs,
        [
            (
                f"Joint {i}",
                [
                    (
                        f"{j}_goal_current_register_A_latest",
                        "Safety-limited robot command",
                    ),
                    (
                        f"{j}_present_current_A",
                        "Measured motor current",
                    ),
                ],
                "A",
            )
            for i, j in enumerate(
                JOINTS,
                1,
            )
        ],
        (
            "Motor-current summary: safety-limited robot command "
            "and measured motor current"
            "\n"
            r"$I_{cmd}=s_I c_{goal}$ and "
            r"$I_{meas}=s_I c_{present}$"
            "\n"
            f"Parameter: s_I={current_a_per_count*1000.0:.2f} mA/count; "
            f"safety command limit={limit_a:.3f} A "
            "(reported in title only; excluded from y-axis scaling)"
            "\n"
            "Sources: "
            "I_cmd = safety-limited robot Goal Current command; "
            "I_meas = actuator Present Current sensing feedback"
        ),
        joint_dir
        / "motor_current_safety_limited_command_and_measured_A",
        audit,
        "motor",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    for i, j in enumerate(JOINTS, 1):
        plot_overlays(
            mdf, t, t_abs,
            [(
                f"Joint {i}",
                [
                    (f"{j}_pre_safety_request_raw_latest", "Computer current request"),
                    (f"{j}_goal_current_register_raw_latest", "Safety-limited robot command"),
                    (f"{j}_present_current_raw", "Measured motor current"),
                ],
                f"current count (1 count = {current_a_per_count*1000.0:.2f} mA)",
            )],
            rf"Joint {i} motor-current signal path; "
            rf"$|u_{{cmd}}|\leq u_{{lim}}$; "
            rf"parameter: $u_{{lim}}={current_limit_count:.0f}$ counts",
            joint_dir / f"joint{i}_current_counts",
            audit, "motor", source, hold_spans, eq_spans, events,
            reference_lines=ref_count,
        )

        plot_overlays(
            mdf, t, t_abs,
            [(
                f"Joint {i}",
                [
                    (f"{j}_pre_safety_request_A_latest", "Computer current request"),
                    (f"{j}_goal_current_register_A_latest", "Safety-limited robot command"),
                    (f"{j}_present_current_A", "Measured motor current"),
                ],
                "A",
            )],
            rf"Joint {i} motor-current signal path: computer request, safety-limited robot command, and measured feedback; "
            rf"$|I_{{cmd}}|\leq I_{{lim}}$; "
            rf"parameter: $I_{{lim}}={limit_a:.3f}$ A",
            joint_dir / f"joint{i}_motor_current_signals_A",
            audit, "motor", source, hold_spans, eq_spans, events,
            reference_lines=ref_a,
        )

        before = f"{j}_pre_safety_request_A_latest"
        commanded = f"{j}_goal_current_register_A_latest"
        measured = f"{j}_present_current_A"
        if all(c in mdf.columns for c in (before, commanded, measured)):
            d = mdf.copy()
            d[f"{j}_before_minus_commanded_A"] = finite_numeric(d[before]) - finite_numeric(d[commanded])
            d[f"{j}_commanded_minus_measured_A"] = finite_numeric(d[commanded]) - finite_numeric(d[measured])
            d[f"{j}_before_minus_measured_A"] = finite_numeric(d[before]) - finite_numeric(d[measured])
            plot_overlays(
                d, t, t_abs,
                [(
                    f"Joint {i}",
                    [
                        (f"{j}_before_minus_commanded_A", "Computer request − safety-limited robot command"),
                        (f"{j}_commanded_minus_measured_A", "Safety-limited robot command − measured motor current"),
                        (f"{j}_before_minus_measured_A", "Computer request − measured motor current"),
                    ],
                    "A",
                )],
                f"Joint {i} motor-current differences [A]",
                joint_dir / f"joint{i}_current_difference_A",
                audit, "motor", source, hold_spans, eq_spans, events,
            )

    plot_subplots(
        mdf, t, t_abs,
        [(f"{j}_request_model_equivalent_Nm_latest",
          rf"Joint {i} $u^*_{{cmd}}$", "software unit")
         for i, j in enumerate(JOINTS, 1)],
        rf"Calculated software command-scale quantity: "
        rf"$u^*=u_{{before}}/s_u$; "
        rf"parameter: $s_u={command_scale_count_per_nm:.0f}$ counts/software-unit; "
        rf"not torque",
        joint_dir / "calculated_software_command_scale",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    torque_cols = [
        (f"{j}_torque_est_Nm", rf"Joint {i} calculated $\tau_I$", "N·m")
        for i, j in enumerate(JOINTS, 1)
    ]
    if any(has_finite(mdf, c) for c, _, _ in torque_cols):
        plot_subplots(
            mdf, t, t_abs, torque_cols,
            r"Calculated torque from measured current: "
            r"$\tau_I=\mathrm{sgn}(I)\max(0,a|I|+b)$ [N·m]",
            joint_dir / "calculated_current_derived_torque",
            audit, "motor", source, hold_spans, eq_spans, events,
        )
    else:
        for c, label, unit in torque_cols:
            if c in mdf.columns:
                audit.mark_used("motor", [c])
                audit.add(
                    source, c, label, unit, "torque", False, "",
                    "No finite calibrated values; current-to-torque coefficients were not supplied."
                )

    plot_subplots(
        mdf, t, t_abs,
        [
            ("pre_safety_request_age_s", "Age of computer current request", "s"),
            ("goal_register_age_s", "Age of safety-limited robot command", "s"),
        ],
        "Motor-current timing diagnostics [s]",
        diag_dir / "motor_current_timing",
        audit, "motor", source, hold_spans, eq_spans, events,
    )

    audit.mark_used(
        "motor",
        ["record_type", "host_monotonic_ns", "host_wall_time_s",
         "rx_ros_time_s", "message_time_s"]
    )



def summarize_run_dls(run_folder: Path):
    """
    Read the run log and summarize the lambda_sq values that were
    actually reported during that run.
    """
    log_path = (
        run_folder
        / "single_arm_hardware_bringup.log"
    )

    if not log_path.exists():
        return 0, 0, float("nan")

    rx = re.compile(
        r"lambda_sq\s*:\s*"
        r"([-+0-9.eE]+)"
    )

    vals = []

    for line in log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():

        m = rx.search(line)

        if m is None:
            continue

        try:
            vals.append(
                float(m.group(1))
            )
        except ValueError:
            pass

    if not vals:
        return 0, 0, float("nan")

    a = np.asarray(
        vals,
        dtype=float,
    )

    nonzero = np.abs(a) > 1e-12

    return (
        int(len(a)),
        int(np.sum(nonzero)),
        float(np.max(np.abs(a))),
    )


def plot_state(
    sdf: pd.DataFrame,
    run_folder: Path,
    save_root: Path,
    audit: PlotAudit,
    contact_filter_alpha: float,
    velocity_filter_alpha: float,
    damping_x: float,
    damping_y: float,
    damping_z: float,
):
    source = "experiment_state_log.csv"

    t, t_abs, _ = numeric_time(
        sdf,
        ("ros_time_s", "timestamp"),
    )

    hold_spans = derive_hold_spans_abs(
        sdf,
        t_abs,
    )

    eq_spans = derive_equilibrium_spans_abs(
        sdf,
        t_abs,
    )

    events = event_times_abs(
        sdf,
        t_abs,
    )

    cart_dir = save_root / "raw" / "cartesian"
    cmd_dir = save_root / "raw" / "commands"
    diag_dir = save_root / "raw" / "diagnostics"

    # ========================================================
    # CARTESIAN POSITION
    # ========================================================

    plot_overlays(
        sdf,
        t,
        t_abs,
        [
            (
                "X position",
                [
                    ("desired_x", "Commanded X"),
                    ("ee_x", "Proprioceptive FK X"),
                ],
                "m",
            ),
            (
                "Y position",
                [
                    ("desired_y", "Commanded Y"),
                    ("ee_y", "Proprioceptive FK Y"),
                ],
                "m",
            ),
            (
                "Z position",
                [
                    ("desired_z", "Commanded Z"),
                    ("ee_z", "Proprioceptive FK Z"),
                ],
                "m",
            ),
        ],
        r"Cartesian position: computer-commanded target and proprioceptive FK from robot sensing feedback"
        "\n"
        r"$\mathbf{x}_{FK}=\mathbf{p}\!\left(T_{URDF/KDL}(\mathbf{q}_{meas})\right)$"
        "\n"
        r"Sources: $\mathbf{x}_{cmd}$ = computer/controller command; "
        r"$\mathbf{q}_{meas}$ = robot joint-encoder feedback; "
        r"$\mathbf{x}_{FK}$ = kinematic transformation of that feedback",
        cart_dir / "cartesian_position_commanded_and_fk",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    # ========================================================
    # CARTESIAN ERROR
    # ========================================================

    xyz_cols = (
        "desired_x",
        "desired_y",
        "desired_z",
        "ee_x",
        "ee_y",
        "ee_z",
    )

    if all(c in sdf.columns for c in xyz_cols):

        d = sdf.copy()

        d["derived_err_x_m"] = (
            finite_numeric(d["desired_x"])
            - finite_numeric(d["ee_x"])
        )

        d["derived_err_y_m"] = (
            finite_numeric(d["desired_y"])
            - finite_numeric(d["ee_y"])
        )

        d["derived_err_z_m"] = (
            finite_numeric(d["desired_z"])
            - finite_numeric(d["ee_z"])
        )

        d["derived_err_norm_m"] = np.sqrt(
            d["derived_err_x_m"]**2
            + d["derived_err_y_m"]**2
            + d["derived_err_z_m"]**2
        )

        plot_subplots(
            d,
            t,
            t_abs,
            [
                (
                    "derived_err_x_m",
                    r"$e_x=x_{cmd}-x_{FK}$",
                    "m",
                ),
                (
                    "derived_err_y_m",
                    r"$e_y=y_{cmd}-y_{FK}$",
                    "m",
                ),
                (
                    "derived_err_z_m",
                    r"$e_z=z_{cmd}-z_{FK}$",
                    "m",
                ),
                (
                    "derived_err_norm_m",
                    r"$\|\mathbf{e}\|$",
                    "m",
                ),
            ],
            r"Cartesian tracking error: "
            r"$\mathbf{e}=\mathbf{x}_{cmd}-\mathbf{x}_{FK}$; "
            r"$\mathbf{x}_{FK}=\mathbf{p}(T_{URDF/KDL}(\mathbf{q}_{meas}))$"
            "\n"
            r"Sources: $\mathbf{x}_{cmd}$ = computer/controller command; "
            r"$\mathbf{q}_{meas}$ = robot joint-encoder feedback; "
            r"$\mathbf{x}_{FK}$ = transformed sensing feedback",
            cart_dir / "cartesian_tracking_error",
            audit,
            "state",
            source,
            hold_spans,
            eq_spans,
            events,
        )

    # ========================================================
    # COMPRESSION
    # ========================================================

    def constant_reference_from_difference(
        source_col,
        difference_col,
    ):
        if (
            source_col not in sdf.columns
            or difference_col not in sdf.columns
        ):
            return float("nan")

        source_vals = finite_numeric(
            sdf[source_col]
        ).to_numpy(dtype=float)

        difference_vals = finite_numeric(
            sdf[difference_col]
        ).to_numpy(dtype=float)

        mask = (
            np.isfinite(source_vals)
            & np.isfinite(difference_vals)
        )

        if not np.any(mask):
            return float("nan")

        return float(
            np.median(
                source_vals[mask]
                - difference_vals[mask]
            )
        )

    x_ref_fk = constant_reference_from_difference(
        "ee_x",
        "compression_depth_measured_m",
    )

    x_ref_cmd = constant_reference_from_difference(
        "desired_x",
        "compression_depth_commanded_m",
    )

    compression_title = (
        r"Compression depth: "
        r"$\delta_{FK}=x_{FK}(\mathbf{q}_{meas})-x_{ref,FK}$ and "
        r"$\delta_{cmd}=x_{cmd}-x_{ref,cmd}$"
        "\n"
        r"Sources: $x_{cmd}$ = computer/controller command; "
        r"$\mathbf{q}_{meas}$ = robot joint-encoder feedback; "
        r"$x_{FK}=p_x(T_{URDF/KDL}(\mathbf{q}_{meas}))$"
        "\n"
        f"Reference parameters from the recorded run: "
        f"x_ref,FK={x_ref_fk:.6f} m; "
        f"x_ref,cmd={x_ref_cmd:.6f} m"
    )

    plot_overlays(
        sdf,
        t,
        t_abs,
        [
            (
                "Compression depth",
                [
                    (
                        "compression_depth_commanded_m",
                        "Commanded compression",
                    ),
                    (
                        "compression_depth_measured_m",
                        "Proprioceptive FK compression",
                    ),
                ],
                "m",
            ),
        ],
        compression_title,
        cart_dir
        / "compression_commanded_and_fk",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    # ========================================================
    # LATERAL DISPLACEMENT + JOINT-COUNT QUANTIZATION
    # ========================================================

    dr = add_proprioceptive_resolution(
        sdf
    )

    required_resolution_cols = (
        "fk_check_x_m",
        "fk_check_y_m",
        "fk_check_z_m",
        "encoder_relative_y_bound_m",
        "encoder_relative_z_bound_m",
        "encoder_relative_r_bound_m",
    )

    missing_resolution_cols = [
        c
        for c in required_resolution_cols
        if c not in dr.columns
    ]

    if missing_resolution_cols:

        raise RuntimeError(
            "Cannot verify Cartesian encoder quantization; "
            "missing columns: "
            + ", ".join(
                missing_resolution_cols
            )
        )

    fk_error = np.sqrt(
        (
            finite_numeric(
                dr["fk_check_x_m"]
            )
            - finite_numeric(
                dr["ee_x"]
            )
        )**2
        + (
            finite_numeric(
                dr["fk_check_y_m"]
            )
            - finite_numeric(
                dr["ee_y"]
            )
        )**2
        + (
            finite_numeric(
                dr["fk_check_z_m"]
            )
            - finite_numeric(
                dr["ee_z"]
            )
        )**2
    )

    fk_error = fk_error[
        np.isfinite(fk_error)
    ]

    if len(fk_error) == 0:

        raise RuntimeError(
            "No finite FK reproduction check available."
        )

    fk_error_median = float(
        np.median(fk_error)
    )

    fk_error_p95 = float(
        np.percentile(
            fk_error,
            95.0,
        )
    )

    fk_error_p99 = float(
        np.percentile(
            fk_error,
            99.0,
        )
    )

    fk_error_max = float(
        np.max(fk_error)
    )

    fk_error_outlier_fraction = float(
        np.mean(
            fk_error > 1e-4
        )
    )

    print(
        "[PLOT] FK reproduction check: "
        f"median={fk_error_median*1000.0:.9f} mm, "
        f"p95={fk_error_p95*1000.0:.9f} mm, "
        f"p99={fk_error_p99*1000.0:.9f} mm, "
        f"max={fk_error_max*1000.0:.9f} mm, "
        f">0.1 mm={100.0*fk_error_outlier_fraction:.3f}%"
    )

    # JointState and Cartesian pose are separate ROS topics and may not
    # correspond to exactly the same control cycle in an offline snapshot.
    # The FK/Jacobian implementation has been independently validated against
    # the controller's KDL chain. Reject systematic disagreement, while
    # tolerating sparse asynchronous samples.
    fk_systematic_mismatch = (
        fk_error_median > 1e-6
        or fk_error_p95 > 1e-5
        or fk_error_outlier_fraction > 0.05
    )

    if fk_systematic_mismatch:

        raise RuntimeError(
            "Systematic FK reproduction mismatch detected; "
            "encoder propagation is therefore not trusted."
        )

    refs = dr.attrs.get(
        "reference_xyz_m",
        np.array(
            [np.nan, np.nan, np.nan],
            dtype=float,
        ),
    )

    ref_row = dr.attrs.get(
        "reference_row_index",
        None,
    )

    ref_mismatch = float(
        dr.attrs.get(
            "reference_fk_mismatch_m",
            np.nan,
        )
    )

    by = finite_numeric(
        dr["encoder_relative_y_bound_m"]
    ).to_numpy(dtype=float)

    bz = finite_numeric(
        dr["encoder_relative_z_bound_m"]
    ).to_numpy(dtype=float)

    br = finite_numeric(
        dr["encoder_relative_r_bound_m"]
    ).to_numpy(dtype=float)

    y = finite_numeric(
        dr["lateral_y_m"]
    ).to_numpy(dtype=float)

    z = finite_numeric(
        dr["lateral_z_m"]
    ).to_numpy(dtype=float)

    rr = finite_numeric(
        dr["r_perp_m"]
    ).to_numpy(dtype=float)

    y_mm = y * 1000.0
    z_mm = z * 1000.0
    r_mm = rr * 1000.0

    by_mm = by * 1000.0
    bz_mm = bz * 1000.0
    br_mm = br * 1000.0

    finite_by = by_mm[
        np.isfinite(by_mm)
    ]

    finite_bz = bz_mm[
        np.isfinite(bz_mm)
    ]

    finite_br = br_mm[
        np.isfinite(br_mm)
    ]

    by_med = (
        float(np.median(finite_by))
        if len(finite_by)
        else float("nan")
    )

    bz_med = (
        float(np.median(finite_bz))
        if len(finite_bz)
        else float("nan")
    )

    br_med = (
        float(np.median(finite_br))
        if len(finite_br)
        else float("nan")
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(18, 15),
        sharex=True,
        squeeze=False,
    )

    axes = axes.flatten()

    axes[0].plot(
        t,
        y_mm,
        label="Proprioceptive FK lateral Y",
        linestyle="-",
    )

    axes[0].fill_between(
        t,
        y_mm - by_mm,
        y_mm + by_mm,
        alpha=0.24,
        label="Encoder-only quantization bound",
    )

    axes[0].set_ylabel(
        "Lateral Y\n(mm)",
        labelpad=10,
    )

    axes[1].plot(
        t,
        z_mm,
        label="Proprioceptive FK lateral Z",
        linestyle="-",
    )

    axes[1].fill_between(
        t,
        z_mm - bz_mm,
        z_mm + bz_mm,
        alpha=0.24,
        label="Encoder-only quantization bound",
    )

    axes[1].set_ylabel(
        "Lateral Z\n(mm)",
        labelpad=10,
    )

    axes[2].plot(
        t,
        r_mm,
        label=r"Proprioceptive FK $r_\perp$",
        linestyle="-",
    )

    axes[2].fill_between(
        t,
        np.maximum(
            0.0,
            r_mm - br_mm,
        ),
        r_mm + br_mm,
        alpha=0.24,
        label="Encoder-only propagated bound",
    )

    axes[2].set_ylabel(
        r"$r_\perp$"
        + "\n(mm)",
        labelpad=10,
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    for ax in axes:

        decorate_axes(
            ax,
            t_abs,
            t,
            hold_spans,
            eq_spans,
            events,
        )

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
        )

    lateral_title = (
        r"Proprioceptive lateral displacement: "
        r"$\Delta y=y_{FK}-y_{ref}$ and "
        r"$\Delta z=z_{FK}-z_{ref}$; "
        r"$r_\perp=\sqrt{\Delta y^2+\Delta z^2}$"
        "\n"
        r"Encoder propagation for each Cartesian component: "
        r"$B_{\Delta x_j}="
        r"\frac{\Delta q_{LC}}{2}"
        r"\sum_i("
        r"|J_{v,ji}(\mathbf{q}_k)|"
        r"+|J_{v,ji}(\mathbf{q}_{ref})|"
        r")$, "
        r"$|\epsilon_{\Delta x_j}|\leq B_{\Delta x_j}$"
        "\n"
        f"Parameters: "
        f"N={int(ENCODER_COUNTS_PER_REV)} counts/rev; "
        f"Delta q_LC={ENCODER_DQ_RAD:.9f} rad/count; "
        f"y_ref={refs[1]:.6f} m; "
        f"z_ref={refs[2]:.6f} m; "
        f"reference row={ref_row}"
        "\n"
        f"Median encoder-only bounds: "
        f"Y=±{by_med:.3f} mm; "
        f"Z=±{bz_med:.3f} mm; "
        f"radial=±{br_med:.3f} mm. "
        f"Reference-FK mismatch="
        f"{ref_mismatch*1000.0:.6f} mm. "
        f"Mechanical uncertainty can be larger."
    )

    fig.suptitle(
        lateral_title,
        wrap=True,
    )

    lateral_paths = save_figure(
        fig,
        cart_dir
        / "lateral_displacement_with_encoder_quantization",
    )

    for col, label in (
        (
            "lateral_y_m",
            "Proprioceptive FK lateral Y",
        ),
        (
            "lateral_z_m",
            "Proprioceptive FK lateral Z",
        ),
        (
            "r_perp_m",
            "Proprioceptive FK radial lateral displacement",
        ),
    ):

        if col in dr.columns:

            audit.mark_used(
                "state",
                [col],
            )

            audit.add(
                source,
                col,
                label,
                "m",
                lateral_title,
                True,
                str(lateral_paths[0]),
            )

    # ========================================================
    # CALCULATED IMPEDANCE-FORCE SIGNAL
    # ========================================================

    def gain_text(col):

        if col not in sdf.columns:
            return "not logged"

        a = finite_numeric(
            sdf[col]
        ).to_numpy(dtype=float)

        a = a[
            np.isfinite(a)
        ]

        if len(a) == 0:
            return "not logged"

        lo = float(
            np.min(a)
        )

        hi = float(
            np.max(a)
        )

        if abs(hi - lo) < 1e-9:
            return f"{lo:.4g}"

        return (
            f"{lo:.4g} to {hi:.4g}"
        )

    kx = gain_text(
        "k_x_n_m"
    )

    ky = gain_text(
        "k_y_n_m"
    )

    kz = gain_text(
        "k_z_n_m"
    )

    dls_samples, dls_nonzero, dls_max = (
        summarize_run_dls(
            run_folder
        )
    )

    if (
        dls_samples > 0
        and dls_nonzero == 0
    ):

        mapping_line = (
            r"Run-verified mapping: "
            r"$\boldsymbol{\tau}_{imp}"
            r"=J^T\mathbf{F}_{imp}$; "
            f"lambda-squared was zero in "
            f"{dls_samples}/{dls_samples} "
            f"logged diagnostics"
        )

    elif dls_samples > 0:

        mapping_line = (
            "Run includes DLS-active diagnostics; "
            f"{dls_nonzero}/{dls_samples} "
            "logged lambda-squared values were nonzero"
        )

    else:

        mapping_line = (
            "Run-log Jacobian mapping could not be "
            "independently verified"
        )

    force_title = (
        "Calculated Cartesian impedance-force signal — "
        "not a measured or torque-reconstructed contact force"
        "\n"
        r"$\mathbf{F}_{imp}="
        r"\mathbf{K}_{cmd}\mathbf{e}"
        r"+\mathbf{D}_{cmd}\dot{\mathbf{e}}_f$"
        "\n"
        + mapping_line
        + "\n"
        r"Published/stored calculated signal: "
        r"$\mathbf{F}_{stored,k}="
        r"\alpha\mathbf{F}_{imp,k}"
        r"+(1-\alpha)\mathbf{F}_{stored,k-1}$, "
        r"$\mathbf{F}_{stored,0}=\mathbf{0}$"
        "\n"
        r"Sources: $\mathbf{K}_{cmd},\mathbf{D}_{cmd},\mathbf{x}_{cmd}$ = computer/controller commands; "
        r"$\mathbf{x}_{FK}(\mathbf{q}_{meas})$ = robot encoder feedback transformed through URDF/KDL"
        "\n"
        f"Parameters: "
        f"Kx={kx}, Ky={ky}, Kz={kz} N/m; "
        f"Dx={damping_x:g}, "
        f"Dy={damping_y:g}, "
        f"Dz={damping_z:g} N s/m; "
        f"derivative-filter beta="
        f"{velocity_filter_alpha:g}; "
        f"storage-filter alpha="
        f"{contact_filter_alpha:g}"
    )

    plot_subplots(
        sdf,
        t,
        t_abs,
        [
            (
                "contact_fx",
                r"Stored $F_{plot,x}$",
                "N",
            ),
            (
                "contact_fy",
                r"Stored $F_{plot,y}$",
                "N",
            ),
            (
                "contact_fz",
                r"Stored $F_{plot,z}$",
                "N",
            ),
            (
                "contact_force_norm",
                r"$\|\mathbf{F}_{plot}\|$",
                "N",
            ),
        ],
        force_title,
        cart_dir
        / "calculated_cartesian_impedance_force_signal",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    # These logged rotational entries are deliberately excluded from
    # physical-moment plotting.
    for c in (
        "contact_tx",
        "contact_ty",
        "contact_tz",
    ):

        if c in sdf.columns:

            audit.mark_used(
                "state",
                [c],
            )

            audit.add(
                source,
                c,
                "Rotational impedance entry",
                "N m",
                "not plotted",
                False,
                "",
                "No physical contact-moment calculation is claimed.",
            )

    # ========================================================
    # COMMANDED CARTESIAN STIFFNESS
    # ========================================================

    plot_subplots(
        sdf,
        t,
        t_abs,
        [
            (
                "k_x_n_m",
                r"Commanded $K_x$",
                "N/m",
            ),
            (
                "k_y_n_m",
                r"Commanded $K_y$",
                "N/m",
            ),
            (
                "k_z_n_m",
                r"Commanded $K_z$",
                "N/m",
            ),
            (
                "k_lat_commanded_n_m",
                r"Commanded $K_{lat}$",
                "N/m",
            ),
        ],
        r"Commanded Cartesian stiffness "
        r"$\mathbf{K}_{cmd}$ [N/m] — "
        r"not a stiffness measurement",
        cmd_dir
        / "commanded_cartesian_stiffness",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    # ========================================================
    # JOINT STATE
    # ========================================================

    plot_subplots(
        sdf,
        t,
        t_abs,
        [
            (
                f"joint{i}_pos",
                rf"Measured $q_{i}$",
                "rad",
            )
            for i in range(1, 5)
        ],
        "Measured joint angle [rad]",
        cmd_dir
        / "measured_joint_angle_synchronized",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    plot_subplots(
        sdf,
        t,
        t_abs,
        [
            (
                f"joint{i}_vel",
                rf"Measured $\dot q_{i}$",
                "rad/s",
            )
            for i in range(1, 5)
        ],
        r"Measured joint angular velocity [rad/s]",
        cmd_dir
        / "measured_joint_angular_velocity_synchronized",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    # ========================================================
    # TIMING
    # ========================================================

    plot_subplots(
        sdf,
        t,
        t_abs,
        [
            (
                "motor_telem_dt_ms",
                "Time offset to nearest motor-current sample",
                "ms",
            ),
            (
                "pre_safety_request_age_s",
                "Age of computer current request",
                "s",
            ),
            (
                "goal_register_age_s",
                "Age of safety-limited robot command",
                "s",
            ),
        ],
        "Motor-current synchronization timing",
        diag_dir
        / "motor_current_synchronization",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    plot_subplots(
        sdf,
        t,
        t_abs,
        [
            (
                "camera_frame_dt_ms",
                "Time offset to nearest camera frame",
                "ms",
            ),
            (
                "camera_clock_fit_abs_residual_p95_ms",
                "Clock-fit 95th-percentile absolute error",
                "ms",
            ),
            (
                "camera_clock_fit_abs_residual_max_ms",
                "Clock-fit maximum absolute error",
                "ms",
            ),
        ],
        "Camera synchronization error [ms]",
        diag_dir
        / "camera_synchronization_error",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    if (
        "camera_top_timestamp_ms" in sdf.columns
        and "camera_side_timestamp_ms" in sdf.columns
    ):

        d = sdf.copy()

        d["derived_camera_top_minus_side_ms"] = (
            finite_numeric(
                d["camera_top_timestamp_ms"]
            )
            - finite_numeric(
                d["camera_side_timestamp_ms"]
            )
        )

        plot_subplots(
            d,
            t,
            t_abs,
            [
                (
                    "derived_camera_top_minus_side_ms",
                    "Top camera - side camera timestamp",
                    "ms",
                ),
            ],
            r"Dual-camera timing difference: "
            r"$\Delta t=t_{top}-t_{side}$ [ms]",
            diag_dir
            / "dual_camera_timing_difference",
            audit,
            "state",
            source,
            hold_spans,
            eq_spans,
            events,
        )

    plot_subplots(
        sdf,
        t,
        t_abs,
        [
            (
                "contact_active",
                "Physical-contact gate",
                "dimensionless",
            ),
            (
                "contact_valid",
                "Impedance-force signal valid",
                "dimensionless",
            ),
            (
                "move_forward_started",
                "Compression motion started",
                "dimensionless",
            ),
            (
                "contact_established",
                "Physical contact established",
                "dimensionless",
            ),
            (
                "abort_detected",
                "Run stopped by fault",
                "dimensionless",
            ),
            (
                "u2d2_loss_after_contact",
                "Communication loss after contact",
                "dimensionless",
            ),
            (
                "normal_move_end",
                "Compression motion completed",
                "dimensionless",
            ),
            (
                "quasistatic_stage",
                "Quasistatic compression stage",
                "index",
            ),
            (
                "equilibrium_window",
                "Settled equilibrium window",
                "dimensionless",
            ),
        ],
        "Experiment phase and event indicators",
        diag_dir
        / "experiment_phase_events",
        audit,
        "state",
        source,
        hold_spans,
        eq_spans,
        events,
    )

    audit.mark_used(
        "state",
        list(
            TIME_OR_METADATA
            | STRING_OR_CATEGORICAL
        ),
    )


def physical_label_and_unit(col: str) -> Tuple[str, str]:
    """Human-readable physical label and unit for remaining logged numeric columns."""
    explicit = {
        "offset_x": ("X offset", "m"),
        "offset_y": ("Y offset", "m"),
        "offset_z": ("Z offset", "m"),
        "press_offset_x": ("Compression X offset", "m"),
        "press_offset_y": ("Compression Y offset", "m"),
        "press_offset_z": ("Compression Z offset", "m"),
        "press_distance": ("Compression distance", "m"),
        "press_dir_x": ("Compression-direction X component", "dimensionless"),
        "press_dir_y": ("Compression-direction Y component", "dimensionless"),
        "press_dir_z": ("Compression-direction Z component", "dimensionless"),
        "projected_force": ("Force projected on compression axis", "N"),
        "bag_recording_active": ("Data recording active", "dimensionless"),
        "arm_max_vel": ("Joint-speed limit", "rad/s"),
        "contact_fx_mag": ("Absolute calculated axial contact force", "N"),
        "baseline_fx": ("Axial force baseline", "N"),
        "contact_threshold": ("Physical-contact force threshold", "N"),
        "camera_sync_available": ("Camera synchronization available", "dimensionless"),
        "ignored_precontact_dxl_lines": ("Ignored pre-contact communication warnings", "count"),
    }
    if col in explicit:
        return explicit[col]

    m = re.match(r"joint([1-4])_cmd$", col)
    if m:
        return (f"Joint {m.group(1)} measured motor current (duplicate field)",
                "current count (2.69 mA/count)")

    m = re.match(r"joint([1-4])_present_current_raw(?:_highrate)?$", col)
    if m:
        return (f"Joint {m.group(1)} measured motor current",
                "current count (2.69 mA/count)")

    m = re.match(r"joint([1-4])_present_current_A(?:_highrate)?$", col)
    if m:
        return (f"Joint {m.group(1)} measured motor current", "A")

    m = re.match(r"joint([1-4])_pre_safety_request_raw(?:_latest)?$", col)
    if m:
        return (f"Joint {m.group(1)} computer current request",
                "current count (2.69 mA/count)")

    m = re.match(r"joint([1-4])_pre_safety_request_A(?:_latest)?$", col)
    if m:
        return (f"Joint {m.group(1)} computer current request", "A")

    m = re.match(r"joint([1-4])_goal_current_register_raw(?:_latest)?$", col)
    if m:
        return (f"Joint {m.group(1)} safety-limited robot command",
                "current count (2.69 mA/count)")

    m = re.match(r"joint([1-4])_goal_current_register_A(?:_latest)?$", col)
    if m:
        return (f"Joint {m.group(1)} safety-limited robot command", "A")

    m = re.match(r"joint([1-4])_request_model_equivalent_Nm(?:_latest)?$", col)
    if m:
        return (f"Joint {m.group(1)} calculated software command-scale quantity", "software unit")

    m = re.match(r"joint([1-4])_torque_est_Nm(?:_highrate)?$", col)
    if m:
        return (f"Joint {m.group(1)} calculated current-derived torque", "N·m")

    if col.endswith("_n_m"):
        return (col.replace("_", " "), "N/m")
    if col.endswith("_rad_s"):
        return (col.replace("_", " "), "rad/s")
    if col.endswith("_rad"):
        return (col.replace("_", " "), "rad")
    if col.endswith("_ms"):
        return (col.replace("_", " "), "ms")
    if col.endswith("_s"):
        return (col.replace("_", " "), "s")
    if col.endswith("_A"):
        return (col.replace("_", " "), "A")
    if col.endswith("_Nm"):
        return (col.replace("_", " "), "N·m")
    if col.endswith("_m"):
        return (col.replace("_", " "), "m")
    if col.endswith("_raw"):
        return (col.replace("_", " "), "count")
    if col.endswith("_index"):
        return (col.replace("_", " "), "index")
    if col.endswith("_active") or col.endswith("_valid") or col.endswith("_available"):
        return (col.replace("_", " "), "dimensionless")

    return (col.replace("_", " "), "dimensionless")


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

        physical_label, physical_unit = physical_label_and_unit(col)
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(t, s.to_numpy(dtype=float), label=physical_label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{physical_label} ({physical_unit})")
        if col == "arm_max_vel":
            finite_limit = s[np.isfinite(s)]
            if len(finite_limit):
                lim_text = (
                    f"{float(np.nanmedian(finite_limit)):.4g}"
                )
                ax.set_title(
                    r"Joint-speed limit: "
                    r"$|\dot q_i|\leq \dot q_{lim}$; "
                    + f"parameter: median "
                    + r"$\dot q_{lim}$="
                    + lim_text
                    + " rad/s"
                )
            else:
                ax.set_title(
                    r"Joint-speed limit: "
                    r"$|\dot q_i|\leq \dot q_{lim}$"
                )
        else:
            ax.set_title(
                f"{physical_label} [{physical_unit}]"
            )
        decorate_axes(ax, t_abs, t, hold_spans, eq_spans, events)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
        )

        stem = out_dir / (
            safe_name(physical_label)
            + "__"
            + f"{len(used):03d}"
        )
        paths = save_figure(fig, stem)
        audit.add(source_file, col, physical_label, physical_unit,
                  "additional logged quantity", True, str(paths[0]), "")
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
    p.add_argument("--current-limit-counts", type=float, default=900.0,
                   help="Robot motor-current command limit in raw counts.")
    p.add_argument("--current-a-per-count", type=float, default=2.69e-3,
                   help="Motor-current conversion [A/count].")
    p.add_argument("--command-scale-count-per-nm", type=float, default=180.0,
                   help="Software command scaling used only for the calculated N·m-equivalent plot.")
    p.add_argument("--contact-filter-alpha", type=float, default=0.02,
                   help="Stored impedance-force low-pass coefficient.")
    p.add_argument("--velocity-filter-alpha", type=float, default=0.1,
                   help="Error-derivative low-pass coefficient.")
    p.add_argument("--damping-x", type=float, default=28.0,
                   help="Commanded Cartesian X damping [N s/m].")
    p.add_argument("--damping-y", type=float, default=3.0,
                   help="Commanded Cartesian Y damping [N s/m].")
    p.add_argument("--damping-z", type=float, default=18.0,
                   help="Commanded Cartesian Z damping [N s/m].")
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
    print(
        "[PLOT] experiment snapshot: "
        + ("FOUND" if snap_path else "MISSING")
    )

    audit = PlotAudit()

    hdf = None
    h_abs = None
    h_t = None
    if snap_path is not None:
        hdf = pd.read_csv(snap_path, low_memory=False)
        h_t, h_abs, _ = numeric_time(hdf, ("ros_time_s", "timestamp"))
        plot_state(
            hdf,
            run,
            save_root,
            audit,
            args.contact_filter_alpha,
            args.velocity_filter_alpha,
            args.damping_x,
            args.damping_y,
            args.damping_z,
        )

    mdf = None
    if motor_path.exists():
        mdf = pd.read_csv(motor_path, low_memory=False)
        plot_motor(
            mdf, run, save_root, audit, hdf, h_abs,
            args.current_limit_counts,
            args.current_a_per_count,
            args.command_scale_count_per_nm,
        )

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
            hdf, h_t, h_abs, "state", "experiment_state_log.csv",
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
        audit.add("experiment_state_log.csv", "*", "Cartesian and experiment telemetry", "", "source", False, "",
                  "No experiment snapshot CSV was found.")

    manifest = save_root / "plot_manifest.csv"
    audit.write(manifest)

    print(f"[PLOT] manifest={manifest}")
    print(f"[PLOT] plots complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
