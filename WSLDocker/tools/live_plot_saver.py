#!/usr/bin/env python3
"""
live_plot_saver.py — crash-resilient publication-quality plot generator.

This runs as a *background* process launched by run_single_arm_test_gazebo.sh.
It periodically (every --interval seconds) regenerates the publication-quality
PNG figures from the *latest* single-arm harness snapshot CSV, even if that CSV
is only partially written (the harness appends rows incrementally).

Why this exists:
  The harness only writes its final publication plot after a run completes.
  If gzserver / WSL / the harness crashes mid-run, no figure is ever produced.
  By regenerating from the partial CSV on a timer, the most recent PNG on disk
  always reflects the data captured up to the crash, and survives the crash.

Requirements: matplotlib, pandas, numpy (use the workspace .venv python).

Usage:
  python3 tools/live_plot_saver.py \
      --log-dir /workspaces/omx_ros2/logs/single_arm \
      --save-dir /workspaces/omx_ros2/logs/single_arm/figures \
      --interval 60 --downsample 4

The process runs forever until killed (the shell script kills it on cleanup).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Force a non-interactive backend BEFORE importing any matplotlib-using module,
# so this works headless (no DISPLAY) and never tries to open a GUI window.
import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402

# Make tools/ importable so we can reuse the publication-quality plotter.
_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import single_arm_plot_logs as sapl  # noqa: E402


def find_latest_run_dir(log_dir: str) -> "tuple[str, str] | None":
    """Return (run_dir, timestamp) for the most recently modified run, or None."""
    mode = sapl.detect_single_arm_mode(log_dir)
    if not mode or not mode.run_dir or not os.path.isdir(mode.run_dir):
        return None
    snap = os.path.join(mode.run_dir, "single_arm_harness_v3_snapshot.csv")
    if not os.path.isfile(snap):
        return None
    return mode.run_dir, (mode.timestamp or "")


def save_once(log_dir: str, save_dir: str, downsample: int | None) -> bool:
    """Generate figures from the latest partial CSV. Returns True if it plotted."""
    found = find_latest_run_dir(log_dir)
    if found is None:
        print(f"[live_plot_saver] no snapshot CSV yet under {log_dir}", flush=True)
        return False

    run_dir, timestamp = found
    snap = os.path.join(run_dir, "single_arm_harness_v3_snapshot.csv")
    try:
        df = sapl.load_single_arm_snapshot(run_dir, timestamp)
    except Exception as exc:  # noqa: BLE001
        print(f"[live_plot_saver] failed to load {snap}: {exc}", flush=True)
        return False

    if df is None or len(df) == 0:
        print(f"[live_plot_saver] {snap} has no rows yet", flush=True)
        return False

    if downsample and downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)

    mode = sapl.SingleArmRunMode()
    mode.run_dir = run_dir
    mode.timestamp = timestamp

    os.makedirs(save_dir, exist_ok=True)
    try:
        sapl.plot_single_arm_timeseries(df, mode, save_dir=save_dir)
        # A couple of high-value phase-space plots for the contact dynamics.
        for vx, vy in (("ee_x", "contact_fz"), ("contact_fx", "contact_fz")):
            if vx in df.columns and vy in df.columns:
                sapl.plot_single_arm_phase_space(df, mode, vx, vy, save_dir=save_dir)
        print(
            f"[live_plot_saver] saved figures for {timestamp} "
            f"({len(df)} rows) -> {save_dir}",
            flush=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[live_plot_saver] plot failed: {exc}", flush=True)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-dir", required=True,
                   help="Root log directory containing single_arm_harness_v3_* runs.")
    p.add_argument("--save-dir", required=True,
                   help="Directory to write publication-quality PNGs.")
    p.add_argument("--interval", type=float, default=60.0,
                   help="Seconds between regeneration passes.")
    p.add_argument("--downsample", type=int, default=None,
                   help="Plot every Nth sample to speed up large logs.")
    p.add_argument("--once", action="store_true",
                   help="Generate once and exit (used for the final save on cleanup).")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.once:
        save_once(args.log_dir, args.save_dir, args.downsample)
        return

    print(f"[live_plot_saver] started: log-dir={args.log_dir} "
          f"save-dir={args.save_dir} interval={args.interval}s", flush=True)
    while True:
        try:
            save_once(args.log_dir, args.save_dir, args.downsample)
        except Exception as exc:  # noqa: BLE001 - never let the saver die
            print(f"[live_plot_saver] unexpected error: {exc}", flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
