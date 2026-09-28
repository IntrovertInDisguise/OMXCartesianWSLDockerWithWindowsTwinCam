#!/usr/bin/env python3
"""
Publication-quality plotter for OMX Variable Stiffness / Gravity Compensation logs.

Requirements (install on YOUR PC, not inside the container):
    pip install matplotlib pandas numpy

Usage examples:
    # 1) Full timeseries overview of the latest run
    python plot_logs.py --log-dir /tmp/variable_stiffness_logs

    # 2) Specify a particular timestamp
    python plot_logs.py --log-dir /tmp/variable_stiffness_logs --timestamp 20260309_124611

    # 3) Phase-space / comparative plot between two variables
    python plot_logs.py --log-dir /tmp/variable_stiffness_logs --phase ee_x tau1

    # 4) Compare current run against a baseline run
    python plot_logs.py --log-dir /tmp/variable_stiffness_logs \\
        --timestamp 20260309_124611 --baseline 20260309_070855

    # 5) Save figures instead of showing
    python plot_logs.py --log-dir /tmp/variable_stiffness_logs --save-dir ./figures

    # 6) List available runs and their detected modes
    python plot_logs.py --log-dir /tmp/variable_stiffness_logs --list
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None         # type: ignore[assignment]
    np = None          # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None          # type: ignore[assignment]

_HAS_DEPS = matplotlib is not None and np is not None and pd is not None

# ---------------------------------------------------------------------------
# Default log directory — override with env var OMX_LOG_DIR or --log-dir flag
# ---------------------------------------------------------------------------
DEFAULT_LOG_DIR = os.environ.get("OMX_LOG_DIR", "/tmp/variable_stiffness_logs")

# ---------------------------------------------------------------------------
# Publication-quality rcParams
# ---------------------------------------------------------------------------
PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia", "serif"],
    "font.size": 22,
    "font.weight": "bold",
    "axes.labelsize": 24,
    "axes.labelweight": "bold",
    "axes.titlesize": 26,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.8,
    "xtick.labelsize": 22,
    "xtick.major.width": 1.6,
    "xtick.major.size": 7,
    "xtick.minor.width": 1.0,
    "xtick.minor.size": 4,
    "xtick.minor.visible": True,
    "ytick.labelsize": 22,
    "ytick.major.width": 1.6,
    "ytick.major.size": 7,
    "ytick.minor.width": 1.0,
    "ytick.minor.size": 4,
    "ytick.minor.visible": True,
    "legend.fontsize": 22,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.2",
    "lines.linewidth": 2.8,
    "lines.markersize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.figsize": (20, 12),
}

# ---------------------------------------------------------------------------
# Colour palettes  (robot1=blue family, robot2=red family, baseline=grey)
# ---------------------------------------------------------------------------
C_R1 = "#00429d"  # robot 1 — deep blue
C_R2 = "#b5002a"  # robot 2 — dark crimson
C_BL = "#555555"  # baseline — dark grey
ROBOT_COLORS = {1: C_R1, 2: C_R2}
BASELINE_COLORS = {1: "#7a9cc6", 2: "#d46a6a"}  # muted for baseline overlay

# High-contrast dark palette for multi-line subplots (up to 8 variables)
_HC_PALETTE = [
    "#00429d",  # deep blue
    "#b5002a",  # dark crimson
    "#2d6a2e",  # forest green
    "#8b4513",  # saddle brown
    "#6a0dad",  # dark violet
    "#c45800",  # burnt orange
    "#1a7a7a",  # dark teal
    "#333333",  # near-black
]

# ---------------------------------------------------------------------------
# Column groups — human-readable labels with SI units
# ---------------------------------------------------------------------------

JOINT_POS_COLS = {
    "actual_x": "Actual Position X (m)",
    "actual_y": "Actual Position Y (m)",
    "actual_z": "Actual Position Z (m)",
}

JOINT_VEL_COLS = {
    "jv1": "Joint 1 Velocity (rad/s)",
    "jv2": "Joint 2 Velocity (rad/s)",
    "jv3": "Joint 3 Velocity (rad/s)",
    "jv4": "Joint 4 Velocity (rad/s)",
}

JOINT_TORQUE_COLS = {
    "tau1": "Joint 1 Torque (raw)",
    "tau2": "Joint 2 Torque (raw)",
    "tau3": "Joint 3 Torque (raw)",
    "tau4": "Joint 4 Torque (raw)",
}

JOINT_GAINS_COLS = {
    "Ktx": "Translational Stiffness X (N/m)",
    "Kty": "Translational Stiffness Y (N/m)",
    "Ktz": "Translational Stiffness Z (N/m)",
    "Dtx": "Translational Damping X (Ns/m)",
    "Dty": "Translational Damping Y (Ns/m)",
    "Dtz": "Translational Damping Z (Ns/m)",
}

JOINT_ROT_GAINS_COLS = {
    "Krx": "Rotational Stiffness X (Nm/rad)",
    "Kry": "Rotational Stiffness Y (Nm/rad)",
    "Krz": "Rotational Stiffness Z (Nm/rad)",
    "Drx": "Rotational Damping X (Nms/rad)",
    "Dry": "Rotational Damping Y (Nms/rad)",
    "Drz": "Rotational Damping Z (Nms/rad)",
}

JOINT_REF_COLS = {
    "desired_x": "Desired Position X (m)",
    "desired_y": "Desired Position Y (m)",
    "desired_z": "Desired Position Z (m)",
}

EE_POS_COLS = {
    "ee_x": "End-Effector Position X (m)",
    "ee_y": "End-Effector Position Y (m)",
    "ee_z": "End-Effector Position Z (m)",
}

EE_ORIENT_COLS = {
    "ee_roll": "End-Effector Roll (rad)",
    "ee_pitch": "End-Effector Pitch (rad)",
    "ee_yaw": "End-Effector Yaw (rad)",
}

EE_VEL_COLS = {
    "ee_vx": "End-Effector Velocity X (m/s)",
    "ee_vy": "End-Effector Velocity Y (m/s)",
    "ee_vz": "End-Effector Velocity Z (m/s)",
    "ee_wx": "End-Effector Angular Velocity X (rad/s)",
    "ee_wy": "End-Effector Angular Velocity Y (rad/s)",
    "ee_wz": "End-Effector Angular Velocity Z (rad/s)",
}

EE_FORCE_COLS = {
    "contact_fx": "Contact Force X (N)",
    "contact_fy": "Contact Force Y (N)",
    "contact_fz": "Contact Force Z (N)",
}

EE_TORQUE_EST_COLS = {
    "contact_tx": "Contact Torque X (Nm)",
    "contact_ty": "Contact Torque Y (Nm)",
    "contact_tz": "Contact Torque Z (Nm)",
}

MANIP_COLS = {
    "manip_yoshikawa": "Yoshikawa Manipulability",
    "manip_sigma_min": "Minimum Singular Value",
    "manip_sigma_max": "Maximum Singular Value",
    "manip_condition_number": "Condition Number",
}

WAYPOINT_COLS = {
    "waypoint_active": "Waypoint Active (bool)",
    "contact_valid": "Contact Valid (bool)",
}

# Master label lookup for any column
ALL_LABELS: Dict[str, str] = {}
for _d in [
    JOINT_POS_COLS, JOINT_VEL_COLS, JOINT_TORQUE_COLS, JOINT_GAINS_COLS,
    JOINT_ROT_GAINS_COLS, JOINT_REF_COLS, EE_POS_COLS, EE_ORIENT_COLS,
    EE_VEL_COLS, EE_FORCE_COLS, EE_TORQUE_EST_COLS, MANIP_COLS,
    WAYPOINT_COLS,
]:
    ALL_LABELS.update(_d)

# The subplot groups for the full timeseries overview
TIMESERIES_GROUPS = [
    ("Joint Actual Cartesian Positions", JOINT_POS_COLS),
    ("Joint Desired Cartesian Positions (Reference)", JOINT_REF_COLS),
    ("Joint Velocities", JOINT_VEL_COLS),
    ("Joint Torques (Commanded)", JOINT_TORQUE_COLS),
    ("Translational Stiffness and Damping", JOINT_GAINS_COLS),
    ("Rotational Stiffness and Damping", JOINT_ROT_GAINS_COLS),
    ("End-Effector Position", EE_POS_COLS),
    ("End-Effector Orientation", EE_ORIENT_COLS),
    ("End-Effector Velocities", EE_VEL_COLS),
    ("Contact Forces (Estimated)", EE_FORCE_COLS),
    ("Contact Torques (Estimated)", EE_TORQUE_EST_COLS),
    ("Manipulability Metrics", MANIP_COLS),
    ("Waypoint / Contact Flags", WAYPOINT_COLS),
]


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------
@dataclass
class RunMode:
    robot_count: int = 1          # 1 = single, 2 = dual
    platform: str = "hardware"    # "hardware" or "gazebo"
    controller: str = "variable_stiffness"  # or "gravity_compensation"
    timestamp: str = ""
    robot_dirs: Dict[int, str] = field(default_factory=dict)  # {1: path, 2: path}

    def label(self) -> str:
        rc = "Dual-Robot" if self.robot_count == 2 else "Single-Robot"
        plat = self.platform.capitalize()
        ctrl = self.controller.replace("_", " ").title()
        return f"{rc} | {plat} | {ctrl}"

    def short(self) -> str:
        return f"{'dual' if self.robot_count == 2 else 'single'}_{self.platform}_{self.controller}"


def detect_mode(log_dir: str, timestamp: Optional[str] = None) -> RunMode:
    """Inspect log_dir structure and infer the run mode."""
    log_dir = str(log_dir)
    mode = RunMode()

    # Detect robot subdirectories
    r1_dir = os.path.join(log_dir, "robot1")
    r2_dir = os.path.join(log_dir, "robot2")
    has_r1 = os.path.isdir(r1_dir)
    has_r2 = os.path.isdir(r2_dir)

    # If no robot subdirs, logs may be flat in log_dir
    if not has_r1 and not has_r2:
        mode.robot_count = 1
        mode.robot_dirs = {1: log_dir}
    elif has_r1 and has_r2:
        mode.robot_count = 2
        mode.robot_dirs = {1: r1_dir, 2: r2_dir}
    elif has_r1:
        mode.robot_count = 1
        mode.robot_dirs = {1: r1_dir}
    else:
        mode.robot_count = 1
        mode.robot_dirs = {1: r2_dir}

    # Pick timestamp (latest or specified)
    if timestamp:
        mode.timestamp = timestamp
    else:
        # find latest snapshot file
        all_snaps = []
        for rd in mode.robot_dirs.values():
            all_snaps.extend(glob.glob(os.path.join(rd, "*snapshot*.csv")))
        if all_snaps:
            all_snaps.sort(key=os.path.getmtime, reverse=True)
            m = re.search(r"(\d{8}_\d{6})", os.path.basename(all_snaps[0]))
            if m:
                mode.timestamp = m.group(1)

    # Detect controller type from filename
    sample_files = glob.glob(os.path.join(
        list(mode.robot_dirs.values())[0], f"*snapshot*{mode.timestamp}*"
    ))
    if sample_files:
        fname = os.path.basename(sample_files[0]).lower()
        if "gravity" in fname:
            mode.controller = "gravity_compensation"
        elif "variable_stiffness" in fname:
            mode.controller = "variable_stiffness"

    # Detect platform from path
    path_lower = log_dir.lower()
    if "gazebo" in path_lower or "sim" in path_lower:
        mode.platform = "gazebo"
    else:
        mode.platform = "hardware"

    return mode


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_snapshot(directory: str, timestamp: str) -> Optional[pd.DataFrame]:
    """Load a snapshot CSV for a given directory and timestamp."""
    pattern = os.path.join(directory, f"*snapshot*{timestamp}*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    if "timestamp" in df.columns:
        # Convert to relative seconds
        t0 = df["timestamp"].iloc[0]
        df["time_s"] = df["timestamp"] - t0
        # Drop rows where all data columns are NaN (initial empty rows)
        data_cols = [c for c in df.columns if c not in ("timestamp", "time_s", "contact_frame_id")]
        df = df.dropna(subset=data_cols, how="all").reset_index(drop=True)
        if len(df) > 0:
            t0 = df["timestamp"].iloc[0]
            df["time_s"] = df["timestamp"] - t0
    return df


def load_run(mode: RunMode) -> Dict[int, pd.DataFrame]:
    """Load DataFrames for all robots in a run. Key = robot number."""
    data = {}
    for rn, rdir in mode.robot_dirs.items():
        df = load_snapshot(rdir, mode.timestamp)
        if df is not None and len(df) > 0:
            data[rn] = df
    return data


def list_available_runs(log_dir: str):
    """Print available runs with detected modes."""
    timestamps = set()
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            m = re.search(r"snapshot_(\d{8}_\d{6})\.csv", f)
            if m:
                timestamps.add(m.group(1))

    if not timestamps:
        print(f"No snapshot CSV files found under {log_dir}")
        return

    print(f"\n{'Timestamp':<22} {'Mode':<50} {'Robots'}")
    print("-" * 90)
    for ts in sorted(timestamps, reverse=True):
        mode = detect_mode(log_dir, ts)
        data = load_run(mode)
        robots_str = ", ".join(
            f"robot{r} ({len(df)} rows)" for r, df in sorted(data.items())
        )
        print(f"  {ts:<20} {mode.label():<50} {robots_str}")
    print()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def apply_pub_style():
    """Apply publication-quality matplotlib rcParams."""
    matplotlib.rcParams.update(PUB_RC)


def _robot_label(robot_num: int, mode: RunMode, is_baseline: bool = False) -> str:
    prefix = "Baseline " if is_baseline else ""
    if mode.robot_count == 1:
        return f"{prefix}Robot"
    return f"{prefix}Robot {robot_num}"


def _add_time_xlabel(ax):
    ax.set_xlabel("Time (s)", fontsize=24, fontweight="bold")


def _enable_minor_grid(ax):
    """Turn on both major and minor grid lines."""
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", alpha=0.2, linewidth=0.6, linestyle=":")


def _style_legend(ax, ncol: int = 1):
    """Place legend outside the axes so it never occludes data."""
    leg = ax.legend(
        ncol=ncol,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
        edgecolor="0.2",
        fontsize=22,
    )
    if leg:
        for text in leg.get_texts():
            text.set_fontweight("bold")


def _replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN for clean plotting."""
    return df.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# 1) Full timeseries subplots
# ---------------------------------------------------------------------------

def plot_timeseries(
    data: Dict[int, pd.DataFrame],
    mode: RunMode,
    baseline_data: Optional[Dict[int, pd.DataFrame]] = None,
    baseline_mode: Optional[RunMode] = None,
    save_dir: Optional[str] = None,
):
    """
    Plot all logged info with separate subplots per variable group.
    If dual robot, overlay on same subplot with legend.
    If baseline provided, overlay with dashed style.
    """
    apply_pub_style()

    for group_title, col_map in TIMESERIES_GROUPS:
        # Filter to columns that actually exist in the data
        valid_cols = {}
        for col, label in col_map.items():
            for rn, df in data.items():
                if col in df.columns and df[col].notna().any():
                    valid_cols[col] = label
                    break

        if not valid_cols:
            continue

        n_sub = len(valid_cols)
        fig, axes = plt.subplots(n_sub, 1, figsize=(20, max(5.5 * n_sub, 8)),
                                 sharex=True, squeeze=False)
        axes = axes.flatten()

        fig.suptitle(
            f"{group_title}\n{mode.label()} — Run {mode.timestamp}",
            fontsize=28, fontweight="bold", y=0.99,
        )

        for idx, (col, label) in enumerate(valid_cols.items()):
            ax = axes[idx]
            plotted_anything = False

            # Current run
            for rn, df in sorted(data.items()):
                df_clean = _replace_inf(df)
                if col in df_clean.columns and df_clean[col].notna().any():
                    ax.plot(
                        df_clean["time_s"], df_clean[col],
                        color=ROBOT_COLORS.get(rn, C_R1),
                        linewidth=2.8,
                        label=_robot_label(rn, mode),
                    )
                    plotted_anything = True

            # Baseline overlay
            if baseline_data:
                bmode = baseline_mode or mode
                for rn, df in sorted(baseline_data.items()):
                    df_clean = _replace_inf(df)
                    if col in df_clean.columns and df_clean[col].notna().any():
                        ax.plot(
                            df_clean["time_s"], df_clean[col],
                            color=BASELINE_COLORS.get(rn, C_BL),
                            linewidth=2.0, linestyle="--", alpha=0.7,
                            label=_robot_label(rn, bmode, is_baseline=True),
                        )
                        plotted_anything = True

            ax.set_ylabel(label, fontsize=24, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=22)
            ax.tick_params(axis="both", which="minor", labelsize=18)
            _enable_minor_grid(ax)
            if plotted_anything:
                _style_legend(ax, ncol=2 if (mode.robot_count == 2 or baseline_data) else 1)

        _add_time_xlabel(axes[-1])
        # Leave right margin for external legend, top gap for suptitle
        fig.tight_layout(rect=[0, 0, 0.82, 0.95])

        if save_dir:
            safe_title = re.sub(r"[^a-zA-Z0-9_]", "_", group_title.lower())
            fpath = os.path.join(save_dir, f"timeseries_{safe_title}_{mode.timestamp}.png")
            fig.savefig(fpath)
            print(f"  Saved: {fpath}")
            plt.close(fig)

    if not save_dir:
        plt.show()


# ---------------------------------------------------------------------------
# 2) Phase-space / comparative analysis
# ---------------------------------------------------------------------------

def plot_phase_space(
    data: Dict[int, pd.DataFrame],
    mode: RunMode,
    var_x: str,
    var_y: str,
    baseline_data: Optional[Dict[int, pd.DataFrame]] = None,
    baseline_mode: Optional[RunMode] = None,
    save_dir: Optional[str] = None,
):
    """
    Plot var_y vs var_x as a single scatter/line plot (phase-space style).
    Overlay robots and baseline when available.
    """
    apply_pub_style()

    label_x = ALL_LABELS.get(var_x, var_x)
    label_y = ALL_LABELS.get(var_y, var_y)

    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    fig.suptitle(
        f"Phase-Space Analysis: {label_y} vs {label_x}\n"
        f"{mode.label()} — Run {mode.timestamp}",
        fontsize=28, fontweight="bold",
    )

    for rn, df in sorted(data.items()):
        df_clean = _replace_inf(df)
        if var_x in df_clean.columns and var_y in df_clean.columns:
            mask = df_clean[var_x].notna() & df_clean[var_y].notna()
            ax.plot(
                df_clean.loc[mask, var_x],
                df_clean.loc[mask, var_y],
                color=ROBOT_COLORS.get(rn, C_R1),
                linewidth=2.8, alpha=0.85,
                label=_robot_label(rn, mode),
            )
            # Mark start and end
            if mask.any():
                ax.plot(
                    df_clean.loc[mask, var_x].iloc[0],
                    df_clean.loc[mask, var_y].iloc[0],
                    "o", color=ROBOT_COLORS.get(rn, C_R1),
                    markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                    label=f"{_robot_label(rn, mode)} Start",
                )
                ax.plot(
                    df_clean.loc[mask, var_x].iloc[-1],
                    df_clean.loc[mask, var_y].iloc[-1],
                    "s", color=ROBOT_COLORS.get(rn, C_R1),
                    markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                    label=f"{_robot_label(rn, mode)} End",
                )

    if baseline_data:
        bmode = baseline_mode or mode
        for rn, df in sorted(baseline_data.items()):
            df_clean = _replace_inf(df)
            if var_x in df_clean.columns and var_y in df_clean.columns:
                mask = df_clean[var_x].notna() & df_clean[var_y].notna()
                ax.plot(
                    df_clean.loc[mask, var_x],
                    df_clean.loc[mask, var_y],
                    color=BASELINE_COLORS.get(rn, C_BL),
                    linewidth=2.0, linestyle="--", alpha=0.6,
                    label=_robot_label(rn, bmode, is_baseline=True),
                )

    ax.set_xlabel(label_x, fontsize=24, fontweight="bold")
    ax.set_ylabel(label_y, fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.tick_params(axis="both", which="minor", labelsize=18)
    _enable_minor_grid(ax)
    _style_legend(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 0.80, 0.93])

    if save_dir:
        fpath = os.path.join(save_dir, f"phase_{var_x}_vs_{var_y}_{mode.timestamp}.png")
        fig.savefig(fpath)
        print(f"  Saved: {fpath}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3) Baseline comparison — clones all group 1 + 2 plots with overlay
# ---------------------------------------------------------------------------

def plot_baseline_comparison(
    data: Dict[int, pd.DataFrame],
    mode: RunMode,
    baseline_data: Dict[int, pd.DataFrame],
    baseline_mode: RunMode,
    phase_pairs: Optional[List[Tuple[str, str]]] = None,
    save_dir: Optional[str] = None,
):
    """
    Re-create all timeseries plots and optional phase-space plots
    with the baseline overlaid.
    """
    print(f"\n=== Baseline Comparison ===")
    print(f"  Current:  {mode.label()} — {mode.timestamp}")
    print(f"  Baseline: {baseline_mode.label()} — {baseline_mode.timestamp}\n")

    plot_timeseries(data, mode, baseline_data, baseline_mode, save_dir)

    if phase_pairs:
        for vx, vy in phase_pairs:
            plot_phase_space(data, mode, vx, vy, baseline_data, baseline_mode, save_dir)


# ---------------------------------------------------------------------------
# Variable listing helper
# ---------------------------------------------------------------------------

def print_available_variables(data: Dict[int, pd.DataFrame]):
    """Print all plottable variable names from the loaded data."""
    all_vars = set()
    for df in data.values():
        for c in df.columns:
            if c not in ("timestamp", "time_s", "contact_frame_id"):
                if df[c].notna().any():
                    all_vars.add(c)

    print(f"\nAvailable variables ({len(all_vars)}):")
    print("-" * 60)
    for v in sorted(all_vars):
        label = ALL_LABELS.get(v, v)
        print(f"  {v:<30s}  {label}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Publication-quality plotter for OMX controller logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--log-dir", default=DEFAULT_LOG_DIR,
        help=f"Root log directory (default: {DEFAULT_LOG_DIR}). "
             "Override with OMX_LOG_DIR env var.",
    )
    p.add_argument(
        "--timestamp", "-t", default=None,
        help="Run timestamp (YYYYMMDD_HHMMSS). Defaults to latest.",
    )
    p.add_argument(
        "--list", "-l", action="store_true",
        help="List available runs and exit.",
    )
    p.add_argument(
        "--vars", action="store_true",
        help="Print available variable names and exit.",
    )
    p.add_argument(
        "--phase", nargs=2, metavar=("VAR_X", "VAR_Y"), default=None,
        help="Phase-space plot: specify X and Y variable names.",
    )
    p.add_argument(
        "--baseline", "-b", default=None,
        help="Baseline run timestamp to compare against.",
    )
    p.add_argument(
        "--save-dir", "-s", default=None,
        help="Directory to save figures (PNG 300 dpi). If omitted, shows interactively.",
    )
    p.add_argument(
        "--no-timeseries", action="store_true",
        help="Skip timeseries plots (useful with --phase).",
    )
    p.add_argument(
        "--downsample", type=int, default=None,
        help="Plot every Nth sample to speed up rendering of large logs.",
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run built-in unit tests and exit.",
    )
    return p


# ---------------------------------------------------------------------------
# Built-in unit tests
# ---------------------------------------------------------------------------

def run_self_tests():
    """Run built-in unit tests. Returns (passed, failed) counts."""
    import tempfile
    import shutil
    import io

    # Try to get heavy deps; if missing, data-dependent tests will be skipped
    global matplotlib, plt, np, pd, _HAS_DEPS
    if not _HAS_DEPS:
        try:
            import matplotlib as _mpl
            import matplotlib.pyplot as _plt
            import numpy as _np
            import pandas as _pd
            matplotlib, plt, np, pd = _mpl, _plt, _np, _pd
            _HAS_DEPS = True
        except ImportError:
            pass  # will skip data-dependent tests below

    passed = 0
    failed = 0
    skipped = 0
    errors: List[str] = []

    def _ok(name: str):
        nonlocal passed
        passed += 1
        print(f"  PASS: {name}")

    def _skip(name: str, reason: str = "missing deps"):
        nonlocal skipped
        skipped += 1
        print(f"  SKIP: {name} ({reason})")

    def _fail(name: str, msg: str):
        nonlocal failed
        failed += 1
        errors.append(f"{name}: {msg}")
        print(f"  FAIL: {name} — {msg}")

    def _test(name: str, fn, needs_deps: bool = False):
        if needs_deps and not _HAS_DEPS:
            _skip(name)
            return
        try:
            fn()
            _ok(name)
        except Exception as e:
            _fail(name, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("RUNNING BUILT-IN UNIT TESTS")
    if not _HAS_DEPS:
        print("  (pandas/numpy/matplotlib not installed — data tests will be skipped)")
    print("=" * 60 + "\n")

    # --- 1) RunMode dataclass ---
    def test_runmode_defaults():
        m = RunMode()
        assert m.robot_count == 1
        assert m.platform == "hardware"
        assert m.controller == "variable_stiffness"
        assert m.timestamp == ""
        assert m.robot_dirs == {}
    _test("RunMode defaults", test_runmode_defaults)

    def test_runmode_label_single():
        m = RunMode(robot_count=1, platform="hardware", controller="variable_stiffness")
        label = m.label()
        assert "Single-Robot" in label
        assert "Hardware" in label
        assert "Variable Stiffness" in label
    _test("RunMode.label() single hardware", test_runmode_label_single)

    def test_runmode_label_dual():
        m = RunMode(robot_count=2, platform="gazebo", controller="gravity_compensation")
        label = m.label()
        assert "Dual-Robot" in label
        assert "Gazebo" in label
        assert "Gravity Compensation" in label
    _test("RunMode.label() dual gazebo", test_runmode_label_dual)

    def test_runmode_short():
        m = RunMode(robot_count=2, platform="hardware", controller="variable_stiffness")
        assert m.short() == "dual_hardware_variable_stiffness"
        m2 = RunMode(robot_count=1, platform="gazebo", controller="gravity_compensation")
        assert m2.short() == "single_gazebo_gravity_compensation"
    _test("RunMode.short()", test_runmode_short)

    # --- 2) Column groups completeness ---
    def test_all_labels_populated():
        assert len(ALL_LABELS) > 0, "ALL_LABELS is empty"
        group_dicts = [
            JOINT_POS_COLS, JOINT_VEL_COLS, JOINT_TORQUE_COLS, JOINT_GAINS_COLS,
            JOINT_ROT_GAINS_COLS, JOINT_REF_COLS, EE_POS_COLS, EE_ORIENT_COLS,
            EE_VEL_COLS, EE_FORCE_COLS, EE_TORQUE_EST_COLS, MANIP_COLS,
            WAYPOINT_COLS,
        ]
        expected = 0
        for d in group_dicts:
            expected += len(d)
        assert len(ALL_LABELS) == expected, (
            f"ALL_LABELS has {len(ALL_LABELS)} entries, expected {expected}"
        )
    _test("ALL_LABELS covers all column groups", test_all_labels_populated)

    def test_timeseries_groups_valid():
        assert len(TIMESERIES_GROUPS) == 13, f"Expected 13 groups, got {len(TIMESERIES_GROUPS)}"
        for title, col_map in TIMESERIES_GROUPS:
            assert isinstance(title, str) and len(title) > 0
            assert isinstance(col_map, dict) and len(col_map) > 0
            for col, label in col_map.items():
                assert col in ALL_LABELS, f"Column {col!r} from group {title!r} not in ALL_LABELS"
    _test("TIMESERIES_GROUPS structural integrity", test_timeseries_groups_valid)

    # --- 3) _replace_inf ---
    def test_replace_inf():
        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0], "b": [np.inf, 2, 3, np.nan]})
        result = _replace_inf(df)
        assert result["a"].isna().sum() == 2, "Inf values should become NaN"
        assert result["b"].isna().sum() == 2, "Inf + existing NaN"
        assert result["a"].iloc[0] == 1.0
        assert result["a"].iloc[3] == 4.0
    _test("_replace_inf", test_replace_inf, needs_deps=True)

    # --- 4) _robot_label ---
    def test_robot_label_single():
        m = RunMode(robot_count=1)
        assert _robot_label(1, m) == "Robot"
        assert _robot_label(1, m, is_baseline=True) == "Baseline Robot"
    _test("_robot_label single", test_robot_label_single)

    def test_robot_label_dual():
        m = RunMode(robot_count=2)
        assert _robot_label(1, m) == "Robot 1"
        assert _robot_label(2, m) == "Robot 2"
        assert _robot_label(1, m, is_baseline=True) == "Baseline Robot 1"
    _test("_robot_label dual", test_robot_label_dual)

    # --- 5) detect_mode with temp dir structure ---
    tmpdir = tempfile.mkdtemp(prefix="plot_logs_test_")
    try:
        # Build dual-robot log structure
        r1 = os.path.join(tmpdir, "robot1")
        r2 = os.path.join(tmpdir, "robot2")
        os.makedirs(r1)
        os.makedirs(r2)

        ts = "20260309_999999"
        # Create minimal CSV files
        csv_header = "timestamp,actual_x,actual_y,actual_z,jv1\n"
        csv_row1 = "100.0,0.1,0.2,0.3,1.5\n"
        csv_row2 = "100.01,0.11,0.21,0.31,1.6\n"
        csv_row3 = "100.02,0.12,0.22,0.32,1.7\n"
        csv_content = csv_header + csv_row1 + csv_row2 + csv_row3

        f1 = os.path.join(r1, f"variable_stiffness_snapshot_{ts}.csv")
        f2 = os.path.join(r2, f"variable_stiffness_snapshot_{ts}.csv")
        for fpath in (f1, f2):
            with open(fpath, "w") as fh:
                fh.write(csv_content)

        def test_detect_mode_dual():
            mode = detect_mode(tmpdir, ts)
            assert mode.robot_count == 2, f"Expected 2 robots, got {mode.robot_count}"
            assert 1 in mode.robot_dirs and 2 in mode.robot_dirs
            assert mode.timestamp == ts
            assert mode.controller == "variable_stiffness"
            assert mode.platform == "hardware"
        _test("detect_mode dual-robot dir", test_detect_mode_dual)

        def test_detect_mode_latest():
            mode = detect_mode(tmpdir)  # no timestamp — should pick latest
            assert mode.timestamp == ts
        _test("detect_mode auto-latest timestamp", test_detect_mode_latest)

        # --- 6) load_snapshot ---
        def test_load_snapshot():
            df = load_snapshot(r1, ts)
            assert df is not None, "load_snapshot returned None"
            assert "time_s" in df.columns, "time_s column missing"
            assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
            assert abs(df["time_s"].iloc[0]) < 1e-6, "First time_s should be ~0"
            assert abs(df["time_s"].iloc[2] - 0.02) < 1e-4
            assert "actual_x" in df.columns
        _test("load_snapshot", test_load_snapshot, needs_deps=True)

        # --- 7) load_run ---
        def test_load_run():
            mode = detect_mode(tmpdir, ts)
            data = load_run(mode)
            assert len(data) == 2, f"Expected 2 robots, got {len(data)}"
            assert 1 in data and 2 in data
            for rn in (1, 2):
                assert len(data[rn]) == 3
                assert "time_s" in data[rn].columns
        _test("load_run dual", test_load_run, needs_deps=True)

        # --- 8) Single-robot directory detection ---
        tmpdir_single = tempfile.mkdtemp(prefix="plot_logs_test_single_")
        try:
            csv_flat = os.path.join(tmpdir_single, f"variable_stiffness_snapshot_{ts}.csv")
            with open(csv_flat, "w") as fh:
                fh.write(csv_content)

            def test_detect_mode_single_flat():
                mode = detect_mode(tmpdir_single, ts)
                assert mode.robot_count == 1
                assert 1 in mode.robot_dirs
                assert mode.robot_dirs[1] == tmpdir_single
            _test("detect_mode single (flat dir)", test_detect_mode_single_flat)

            def test_load_run_single():
                mode = detect_mode(tmpdir_single, ts)
                data = load_run(mode)
                assert len(data) == 1
                assert 1 in data and len(data[1]) == 3
            _test("load_run single", test_load_run_single, needs_deps=True)
        finally:
            shutil.rmtree(tmpdir_single, ignore_errors=True)

        # --- 9) Gravity compensation detection ---
        tmpdir_gc = tempfile.mkdtemp(prefix="plot_logs_test_gc_")
        try:
            gc_file = os.path.join(tmpdir_gc, f"gravity_compensation_snapshot_{ts}.csv")
            with open(gc_file, "w") as fh:
                fh.write(csv_content)

            def test_detect_gravity_comp():
                mode = detect_mode(tmpdir_gc, ts)
                assert mode.controller == "gravity_compensation", (
                    f"Expected gravity_compensation, got {mode.controller}"
                )
            _test("detect_mode gravity_compensation", test_detect_gravity_comp)
        finally:
            shutil.rmtree(tmpdir_gc, ignore_errors=True)

        # --- 10) Gazebo platform detection ---
        def test_detect_gazebo_platform():
            tmpdir_gz = tempfile.mkdtemp(prefix="plot_logs_gazebo_test_")
            gz_sub = os.path.join(tmpdir_gz, "gazebo_logs")
            os.makedirs(gz_sub)
            gz_file = os.path.join(gz_sub, f"variable_stiffness_snapshot_{ts}.csv")
            with open(gz_file, "w") as fh:
                fh.write(csv_content)
            try:
                mode = detect_mode(gz_sub, ts)
                assert mode.platform == "gazebo", f"Expected gazebo, got {mode.platform}"
            finally:
                shutil.rmtree(tmpdir_gz, ignore_errors=True)
        _test("detect_mode gazebo platform", test_detect_gazebo_platform)

        # --- 11) list_available_runs (smoke test) ---
        def test_list_available_runs():
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                list_available_runs(tmpdir)
            finally:
                sys.stdout = old_stdout
            output = buf.getvalue()
            assert ts in output, f"Timestamp {ts} not in list output"
            assert "Dual-Robot" in output
        _test("list_available_runs", test_list_available_runs, needs_deps=True)

        # --- 12) print_available_variables (smoke test) ---
        def test_print_available_variables():
            mode = detect_mode(tmpdir, ts)
            data = load_run(mode)
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()
            try:
                print_available_variables(data)
            finally:
                sys.stdout = old_stdout
            output = buf.getvalue()
            assert "actual_x" in output
            assert "jv1" in output
        _test("print_available_variables", test_print_available_variables, needs_deps=True)

        # --- 13) PUB_RC completeness ---
        def test_pub_rc():
            required_keys = [
                "font.family", "font.weight", "font.size",
                "axes.labelsize", "axes.labelweight", "axes.titlesize",
                "lines.linewidth", "lines.markersize",
                "savefig.dpi", "savefig.bbox",
                "xtick.minor.visible", "ytick.minor.visible",
            ]
            for k in required_keys:
                assert k in PUB_RC, f"PUB_RC missing key {k!r}"
            assert PUB_RC["font.family"] == "serif"
            assert PUB_RC["font.weight"] == "bold"
            assert PUB_RC["font.size"] >= 20, "base font.size must be >= 20"
            assert PUB_RC["axes.labelsize"] >= 22
            assert PUB_RC["axes.titlesize"] >= 24
            assert PUB_RC["xtick.labelsize"] >= 20
            assert PUB_RC["ytick.labelsize"] >= 20
            assert PUB_RC["legend.fontsize"] >= 20
            assert PUB_RC["lines.linewidth"] >= 2.5
            assert PUB_RC["savefig.dpi"] >= 300
            assert PUB_RC["xtick.minor.visible"] is True
            assert PUB_RC["ytick.minor.visible"] is True
        _test("PUB_RC completeness", test_pub_rc)

        # --- 14) Colour constants ---
        def test_colours():
            assert len(ROBOT_COLORS) == 2
            assert 1 in ROBOT_COLORS and 2 in ROBOT_COLORS
            assert len(BASELINE_COLORS) == 2
            assert C_R1 != C_R2 != C_BL
            assert len(_HC_PALETTE) >= 6, "Need at least 6 high-contrast colors"
        _test("Colour palette constants", test_colours)

        # --- 15) CLI parser ---
        def test_cli_parser():
            p = build_cli()
            # Basic parse
            ns = p.parse_args(["--log-dir", "/tmp/test"])
            assert ns.log_dir == "/tmp/test"
            assert ns.list is False
            assert ns.vars is False
            assert ns.phase is None
            assert ns.baseline is None
            assert ns.downsample is None
            # Parse with many flags
            ns2 = p.parse_args([
                "--log-dir", "/tmp/x",
                "--timestamp", "20260101_000000",
                "--phase", "ee_x", "tau1",
                "--baseline", "20260101_111111",
                "--save-dir", "/tmp/figs",
                "--downsample", "10",
                "--no-timeseries",
            ])
            assert ns2.timestamp == "20260101_000000"
            assert ns2.phase == ["ee_x", "tau1"]
            assert ns2.baseline == "20260101_111111"
            assert ns2.save_dir == "/tmp/figs"
            assert ns2.downsample == 10
            assert ns2.no_timeseries is True
        _test("CLI parser", test_cli_parser)

        # --- 16) load_snapshot with NaN-only rows are dropped ---
        def test_load_snapshot_nan_rows():
            csv_with_nan = (
                "timestamp,actual_x,actual_y\n"
                "100.0,,\n"   # all NaN data — should be dropped
                "100.01,0.1,0.2\n"
                "100.02,0.11,0.21\n"
            )
            tmpdir_nan = tempfile.mkdtemp(prefix="plot_logs_test_nan_")
            try:
                fpath = os.path.join(tmpdir_nan, f"variable_stiffness_snapshot_{ts}.csv")
                with open(fpath, "w") as fh:
                    fh.write(csv_with_nan)
                df = load_snapshot(tmpdir_nan, ts)
                assert df is not None
                assert len(df) == 2, f"Expected 2 rows after NaN drop, got {len(df)}"
                assert abs(df["time_s"].iloc[0]) < 1e-6
            finally:
                shutil.rmtree(tmpdir_nan, ignore_errors=True)
        _test("load_snapshot drops all-NaN rows", test_load_snapshot_nan_rows, needs_deps=True)

        # --- 17) DEFAULT_LOG_DIR / OMX_LOG_DIR env var ---
        def test_default_log_dir_env():
            p = build_cli()
            ns = p.parse_args([])
            assert ns.log_dir == DEFAULT_LOG_DIR, (
                f"CLI default {ns.log_dir!r} != DEFAULT_LOG_DIR {DEFAULT_LOG_DIR!r}"
            )
        _test("DEFAULT_LOG_DIR used by CLI", test_default_log_dir_env)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # --- Summary ---
    print("\n" + "=" * 60)
    total = passed + failed + skipped
    if failed == 0:
        msg = f"ALL {passed} TESTS PASSED"
        if skipped:
            msg += f" ({skipped} skipped — install pandas/numpy/matplotlib to run all)"
        print(msg)
    else:
        print(f"{failed}/{total} TESTS FAILED ({skipped} skipped)")
        for e in errors:
            print(f"  - {e}")
    print("=" * 60 + "\n")
    return passed, failed


def main():
    parser = build_cli()
    args = parser.parse_args()

    # --test mode: run self-tests and exit
    if args.test:
        _, failed = run_self_tests()
        sys.exit(1 if failed else 0)

    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        print(f"Error: log directory not found: {log_dir}")
        sys.exit(1)

    # --list mode
    if args.list:
        list_available_runs(log_dir)
        sys.exit(0)

    # Detect mode
    mode = detect_mode(log_dir, args.timestamp)
    if not mode.timestamp:
        print("Error: no log files found. Use --list to see available runs.")
        sys.exit(1)

    print(f"\nDetected mode: {mode.label()}")
    print(f"Run timestamp: {mode.timestamp}")
    print(f"Robot count:   {mode.robot_count}")
    print(f"Platform:      {mode.platform}")
    print(f"Controller:    {mode.controller}")

    # Load data
    data = load_run(mode)
    if not data:
        print("Error: could not load any data for the selected run.")
        sys.exit(1)

    for rn, df in data.items():
        print(f"  Robot {rn}: {len(df)} samples, "
              f"{df['time_s'].iloc[-1]:.1f}s duration")

    # Downsample if requested
    if args.downsample and args.downsample > 1:
        data = {rn: df.iloc[::args.downsample].reset_index(drop=True)
                for rn, df in data.items()}
        print(f"  Downsampled to every {args.downsample}th sample")

    # --vars mode
    if args.vars:
        print_available_variables(data)
        sys.exit(0)

    # Load baseline if specified
    baseline_data = None
    baseline_mode = None
    if args.baseline:
        baseline_mode = detect_mode(log_dir, args.baseline)
        baseline_data = load_run(baseline_mode)
        if not baseline_data:
            print(f"Warning: no data for baseline {args.baseline}, ignoring.")
            baseline_data = None
            baseline_mode = None
        else:
            if args.downsample and args.downsample > 1:
                baseline_data = {
                    rn: df.iloc[::args.downsample].reset_index(drop=True)
                    for rn, df in baseline_data.items()
                }
            for rn, df in baseline_data.items():
                print(f"  Baseline Robot {rn}: {len(df)} samples")

    # Create save directory
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"\nSaving figures to: {args.save_dir}")

    # Plot
    if args.phase:
        var_x, var_y = args.phase
        # Validate variables exist
        all_available = set()
        for df in data.values():
            all_available.update(df.columns)
        for v in (var_x, var_y):
            if v not in all_available:
                print(f"Error: variable '{v}' not found in data. Use --vars to list.")
                sys.exit(1)
        plot_phase_space(data, mode, var_x, var_y, baseline_data, baseline_mode,
                         args.save_dir)

    if not args.no_timeseries:
        if baseline_data:
            plot_baseline_comparison(
                data, mode, baseline_data, baseline_mode,
                phase_pairs=[tuple(args.phase)] if args.phase else None,
                save_dir=args.save_dir,
            )
        else:
            plot_timeseries(data, mode, save_dir=args.save_dir)

    if not args.save_dir and not args.phase and args.no_timeseries:
        print("Nothing to plot. Use --phase or remove --no-timeseries.")


if __name__ == "__main__":
    main()
