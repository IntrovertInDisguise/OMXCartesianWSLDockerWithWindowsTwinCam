#!/usr/bin/env python3
"""
Publication-quality plotter for single-arm OMX Variable Stiffness logs.

This is the single-arm counterpart to ``plot_logs.py``.  It reads the
snapshot CSV written by ``hardware_harness_single_arm_v3.py`` and produces
the same publication-quality matplotlib figures, but for a single robot
only (no robot2 overlay, no bilateral comparison).

The column schema is identical to the dual-arm snapshot CSV, so the same
column groups and label maps are reused from ``plot_logs.py``.

Requirements (install on YOUR PC, not inside the container):
    pip install matplotlib pandas numpy

Usage examples:
    # 1) Full timeseries overview of the latest single-arm run
    python3 tools/single_arm_plot_logs.py --log-dir /tmp/variable_stiffness_logs

    # 2) Specify a particular timestamp
    python3 tools/single_arm_plot_logs.py --log-dir /tmp/variable_stiffness_logs --timestamp 20260626_143000

    # 3) Phase-space / comparative plot between two variables
    python3 tools/single_arm_plot_logs.py --log-dir /tmp/variable_stiffness_logs --phase ee_x contact_fx

    # 4) Compare current run against a baseline run
    python3 tools/single_arm_plot_logs.py --log-dir /tmp/variable_stiffness_logs \\
        --timestamp 20260626_143000 --baseline 20260626_120000

    # 5) Save figures instead of showing
    python3 tools/single_arm_plot_logs.py --log-dir /tmp/variable_stiffness_logs --save-dir ./figures

    # 6) List available single-arm runs and their detected modes
    python3 tools/single_arm_plot_logs.py --log-dir /tmp/variable_stiffness_logs --list
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
# Import publication-quality style from the shared plot_logs module
# ---------------------------------------------------------------------------
_tools_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _tools_dir)
try:
    import plot_logs as pl
    from plot_logs import (
        PUB_RC,
        apply_pub_style,
        _enable_minor_grid,
        _style_legend,
        _add_time_xlabel,
        _replace_inf,
        ALL_LABELS,
        TIMESERIES_GROUPS,
        ROBOT_COLORS,
        C_R1,
        C_BL,
        BASELINE_COLORS,
        RunMode,
        detect_mode,
        load_snapshot,
        load_run,
        list_available_runs,
    )
except ImportError as exc:
    print(f"Error: could not import plot_logs.py: {exc}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Single-arm colour
# ---------------------------------------------------------------------------
C_SINGLE = "#00429d"  # same deep blue as robot1 in dual-arm plots


# ---------------------------------------------------------------------------
# Single-arm run detection
# ---------------------------------------------------------------------------

@dataclass
class SingleArmRunMode:
    """Mode descriptor for a single-arm run."""
    platform: str = "hardware"    # "hardware" or "gazebo"
    controller: str = "variable_stiffness"
    timestamp: str = ""
    run_dir: str = ""

    def label(self) -> str:
        plat = self.platform.capitalize()
        ctrl = self.controller.replace("_", " ").title()
        return f"Single-Robot | {plat} | {ctrl}"

    def short(self) -> str:
        return f"single_{self.platform}_{self.controller}"


def detect_single_arm_mode(log_dir: str, timestamp: Optional[str] = None) -> SingleArmRunMode:
    """Inspect log_dir and infer the single-arm run mode.

    Looks for directories matching ``single_arm_harness_v3_*`` and reads
    the manifest JSON if available.
    """
    mode = SingleArmRunMode()
    mode.run_dir = log_dir

    # Find matching subdirectories
    candidates: List[str] = []
    if os.path.isdir(log_dir):
        for entry in os.listdir(log_dir):
            full = os.path.join(log_dir, entry)
            if os.path.isdir(full) and "single_arm" in entry.lower():
                candidates.append(full)

    if not candidates:
        # Maybe log_dir itself is the run directory
        if os.path.isdir(log_dir) and any(
            f.endswith(".csv") for f in os.listdir(log_dir)
        ):
            candidates = [log_dir]

    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        mode.run_dir = candidates[0]

    # Pick timestamp
    if timestamp:
        mode.timestamp = timestamp
    else:
        # Extract from directory name or CSV filename
        m = re.search(r"(\d{8}_\d{6})", os.path.basename(mode.run_dir))
        if not m:
            # Try CSV files
            for f in os.listdir(mode.run_dir):
                m2 = re.search(r"(\d{8}_\d{6})", f)
                if m2:
                    m = m2
                    break
        if m:
            mode.timestamp = m.group(1)

    # Detect platform from path
    path_lower = log_dir.lower()
    if "gazebo" in path_lower or "sim" in path_lower:
        mode.platform = "gazebo"

    # Detect controller from manifest or filenames
    manifest_path = os.path.join(mode.run_dir, "single_arm_harness_v3_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            import json
            with open(manifest_path, "r") as fh:
                manifest = json.load(fh)
            # Manifest doesn't store controller type, but the filenames do
        except Exception:
            pass

    for f in os.listdir(mode.run_dir):
        if "gravity" in f.lower():
            mode.controller = "gravity_compensation"
            break

    return mode


def load_single_arm_snapshot(run_dir: str, timestamp: str) -> Optional["pd.DataFrame"]:
    """Load a snapshot CSV from a single-arm run directory."""
    if pd is None:
        return None

    # Try exact match first
    patterns = [
        os.path.join(run_dir, f"*snapshot*{timestamp}*.csv"),
        os.path.join(run_dir, f"single_arm_harness_v3_snapshot*.csv"),
        os.path.join(run_dir, "*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            # Prefer files with "snapshot" in the name
            snap_matches = [m for m in matches if "snapshot" in m.lower()]
            target = snap_matches[0] if snap_matches else matches[0]
            df = pd.read_csv(target)
            if "timestamp" in df.columns:
                t0 = df["timestamp"].iloc[0]
                df["time_s"] = df["timestamp"] - t0
                data_cols = [c for c in df.columns if c not in ("timestamp", "time_s", "contact_frame_id")]
                df = df.dropna(subset=data_cols, how="all").reset_index(drop=True)
                if len(df) > 0:
                    t0 = df["timestamp"].iloc[0]
                    df["time_s"] = df["timestamp"] - t0
            return df
    return None


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_single_arm_timeseries(
    df: "pd.DataFrame",
    mode: SingleArmRunMode,
    baseline_df: Optional["pd.DataFrame"] = None,
    baseline_mode: Optional[SingleArmRunMode] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Plot all logged info with separate subplots per variable group (single robot)."""
    apply_pub_style()

    for group_title, col_map in TIMESERIES_GROUPS:
        valid_cols = {}
        for col, label in col_map.items():
            if col in df.columns and df[col].notna().any():
                valid_cols[col] = label

        if not valid_cols:
            continue

        n_sub = len(valid_cols)
        fig, axes = plt.subplots(
            n_sub, 1, figsize=(20, max(5.5 * n_sub, 8)),
            sharex=True, squeeze=False,
        )
        axes = axes.flatten()

        fig.suptitle(
            f"{group_title}\n{mode.label()} — Run {mode.timestamp}",
            fontsize=28, fontweight="bold", y=0.99,
        )

        for idx, (col, label) in enumerate(valid_cols.items()):
            ax = axes[idx]
            plotted = False

            df_clean = _replace_inf(df)
            if col in df_clean.columns and df_clean[col].notna().any():
                ax.plot(
                    df_clean["time_s"], df_clean[col],
                    color=C_SINGLE, linewidth=2.8,
                    label="Robot",
                )
                plotted = True

            if baseline_df is not None:
                bdf = _replace_inf(baseline_df)
                if col in bdf.columns and bdf[col].notna().any():
                    ax.plot(
                        bdf["time_s"], bdf[col],
                        color=C_BL, linewidth=2.0, linestyle="--", alpha=0.7,
                        label="Baseline",
                    )
                    plotted = True

            ax.set_ylabel(label, fontsize=24, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=22)
            ax.tick_params(axis="both", which="minor", labelsize=18)
            _enable_minor_grid(ax)
            if plotted:
                _style_legend(ax, ncol=2 if baseline_df is not None else 1)

        _add_time_xlabel(axes[-1])
        fig.tight_layout(rect=[0, 0, 0.82, 0.95])

        if save_dir:
            safe_title = re.sub(r"[^a-zA-Z0-9_]", "_", group_title.lower())
            fpath = os.path.join(save_dir, f"single_arm_timeseries_{safe_title}_{mode.timestamp}.png")
            fig.savefig(fpath)
            print(f"  Saved: {fpath}")
            plt.close(fig)

    if not save_dir:
        plt.show()


def plot_single_arm_phase_space(
    df: "pd.DataFrame",
    mode: SingleArmRunMode,
    var_x: str,
    var_y: str,
    baseline_df: Optional["pd.DataFrame"] = None,
    baseline_mode: Optional[SingleArmRunMode] = None,
    save_dir: Optional[str] = None,
) -> None:
    """Plot var_y vs var_x as a single scatter/line plot (phase-space style)."""
    apply_pub_style()

    label_x = ALL_LABELS.get(var_x, var_x)
    label_y = ALL_LABELS.get(var_y, var_y)

    fig, ax = plt.subplots(1, 1, figsize=(16, 11))
    fig.suptitle(
        f"Phase-Space Analysis: {label_y} vs {label_x}\n"
        f"{mode.label()} — Run {mode.timestamp}",
        fontsize=28, fontweight="bold",
    )

    df_clean = _replace_inf(df)
    if var_x in df_clean.columns and var_y in df_clean.columns:
        mask = df_clean[var_x].notna() & df_clean[var_y].notna()
        ax.plot(
            df_clean.loc[mask, var_x],
            df_clean.loc[mask, var_y],
            color=C_SINGLE, linewidth=2.8, alpha=0.85,
            label="Robot",
        )
        if mask.any():
            ax.plot(
                df_clean.loc[mask, var_x].iloc[0],
                df_clean.loc[mask, var_y].iloc[0],
                "o", color=C_SINGLE,
                markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                label="Start",
            )
            ax.plot(
                df_clean.loc[mask, var_x].iloc[-1],
                df_clean.loc[mask, var_y].iloc[-1],
                "s", color=C_SINGLE,
                markersize=10, markeredgecolor="black", markeredgewidth=1.5,
                label="End",
            )

    if baseline_df is not None:
        bdf = _replace_inf(baseline_df)
        if var_x in bdf.columns and var_y in bdf.columns:
            mask = bdf[var_x].notna() & bdf[var_y].notna()
            ax.plot(
                bdf.loc[mask, var_x],
                bdf.loc[mask, var_y],
                color=C_BL, linewidth=2.0, linestyle="--", alpha=0.6,
                label="Baseline",
            )

    ax.set_xlabel(label_x, fontsize=24, fontweight="bold")
    ax.set_ylabel(label_y, fontsize=24, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.tick_params(axis="both", which="minor", labelsize=18)
    _enable_minor_grid(ax)
    _style_legend(ax, ncol=2)
    fig.tight_layout(rect=[0, 0, 0.80, 0.93])

    if save_dir:
        fpath = os.path.join(save_dir, f"single_arm_phase_{var_x}_vs_{var_y}_{mode.timestamp}.png")
        fig.savefig(fpath)
        print(f"  Saved: {fpath}")
        plt.close(fig)
    else:
        plt.show()


def list_single_arm_runs(log_dir: str) -> None:
    """Print available single-arm runs with detected modes."""
    timestamps: set = set()
    for root, _dirs, files in os.walk(log_dir):
        for f in files:
            m = re.search(r"snapshot_(\d{8}_\d{6})\.csv", f)
            if m:
                timestamps.add(m.group(1))
            # Also check single-arm harness naming
            m2 = re.search(r"single_arm_harness_v3_(\d{8}_\d{6})", root)
            if m2:
                timestamps.add(m2.group(1))

    if not timestamps:
        print(f"No single-arm snapshot CSV files found under {log_dir}")
        return

    print(f"\n{'Timestamp':<22} {'Mode':<50}")
    print("-" * 72)
    for ts in sorted(timestamps, reverse=True):
        mode = detect_single_arm_mode(log_dir, ts)
        print(f"  {ts:<20} {mode.label()}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Publication-quality plotter for single-arm OMX controller logs.",
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
        help="List available single-arm runs and exit.",
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
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not _HAS_DEPS:
        print("Error: matplotlib, numpy, and pandas are required.")
        print("  pip install matplotlib numpy pandas")
        sys.exit(1)

    args = build_cli().parse_args()

    # List mode
    if args.list:
        list_single_arm_runs(args.log_dir)
        return

    # Detect mode and load data
    mode = detect_single_arm_mode(args.log_dir, args.timestamp)
    if not mode.timestamp:
        print(f"No single-arm runs found in {args.log_dir}")
        print("Use --list to see available runs.")
        sys.exit(1)

    df = load_single_arm_snapshot(mode.run_dir, mode.timestamp)
    if df is None or len(df) == 0:
        print(f"No snapshot data found for timestamp {mode.timestamp}")
        sys.exit(1)

    if args.downsample and args.downsample > 1:
        df = df.iloc[:: args.downsample].reset_index(drop=True)

    # Print available variables
    if args.vars:
        pl.print_available_variables({1: df})
        return

    print(f"\n{'=' * 60}")
    print(f"  Single-Arm Plotter")
    print(f"  Mode:      {mode.label()}")
    print(f"  Timestamp: {mode.timestamp}")
    print(f"  Run dir:   {mode.run_dir}")
    print(f"  Rows:      {len(df)}")
    print(f"{'=' * 60}\n")

    # Load baseline if requested
    baseline_df = None
    baseline_mode = None
    if args.baseline:
        baseline_mode = detect_single_arm_mode(args.log_dir, args.baseline)
        baseline_df = load_single_arm_snapshot(baseline_mode.run_dir, args.baseline)
        if baseline_df is not None:
            print(f"  Baseline:  {baseline_mode.timestamp} ({len(baseline_df)} rows)")
        else:
            print(f"  Warning: baseline data not found for {args.baseline}")

    # Timeseries
    if not args.no_timeseries:
        plot_single_arm_timeseries(df, mode, baseline_df, baseline_mode, args.save_dir)

    # Phase-space
    if args.phase:
        var_x, var_y = args.phase
        plot_single_arm_phase_space(df, mode, var_x, var_y, baseline_df, baseline_mode, args.save_dir)


if __name__ == "__main__":
    main()
