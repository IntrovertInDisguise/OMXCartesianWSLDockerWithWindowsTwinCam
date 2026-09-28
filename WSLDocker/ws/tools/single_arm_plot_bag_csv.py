#!/usr/bin/env python3
"""
Single-Arm Bag CSV Plotter
===========================

Generates publication-quality plots from the CSV files extracted from
ROS2 bag recordings of single-arm runs.  This is the single-arm counterpart
to ``plot_bag_csv.py`` (which plots both robot1 and robot2 for dual-arm).

Reads the per-topic CSV files written by ``ros2 bag play`` + CSV extraction
and produces:

- End-effector desired vs actual position (x, y, z)
- Euclidean position error norm over time
- Contact wrench (fx, fy, fz) over time

Usage
-----
  # Default: reads from logs/run_single_arm/csv_extract/
  python3 tools/single_arm_plot_bag_csv.py

  # Custom input/output directories:
  python3 tools/single_arm_plot_bag_csv.py --input-dir /tmp/my_run/csv --output-dir /tmp/my_run/plots

  # Specify robot namespace explicitly:
  python3 tools/single_arm_plot_bag_csv.py --robot-ns robot1

Requirements:
  pip install matplotlib numpy
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path: str):
    """Load a numeric CSV file.  Returns (header_list, numpy_array)."""
    import csv as _csv
    with open(path, "r") as f:
        reader = _csv.reader(f)
        header = next(reader)
        data = []
        for row in reader:
            if not row:
                continue
            try:
                data.append([float(x) for x in row])
            except ValueError:
                continue
    if not data:
        return header, np.zeros((0, len(header)))
    return header, np.array(data)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ee_single_arm(
    robot_ns: str,
    base_dir: str,
    out_dir: str,
) -> None:
    """Plot end-effector position and contact wrench for a single arm.

    Parameters
    ----------
    robot_ns : str
        Robot namespace (e.g. ``"robot1"``).  Used to locate the CSV files
        whose names follow the pattern
        ``{robot_ns}_{robot_ns}_variable_stiffness_*.csv``.
    base_dir : str
        Directory containing the extracted CSV files.
    out_dir : str
        Directory to write the output PNG files.
    """
    desired_f = os.path.join(
        base_dir,
        f"{robot_ns}_{robot_ns}_variable_stiffness_cartesian_pose_desired.csv",
    )
    actual_f = os.path.join(
        base_dir,
        f"{robot_ns}_{robot_ns}_variable_stiffness_end_effector_position.csv",
    )
    contact_f = os.path.join(
        base_dir,
        f"{robot_ns}_{robot_ns}_variable_stiffness_contact_wrench.csv",
    )

    h_d, d = load_csv(desired_f)
    h_a, a = load_csv(actual_f)
    h_c, c = load_csv(contact_f)

    if d.size == 0 or a.size == 0:
        print(f"Missing data for {robot_ns} — skipping EE plot")
        return

    t0 = min(d[0, 0], a[0, 0])
    td = d[:, 0] - t0
    ta = a[:, 0] - t0

    dx, dy, dz = d[:, 1], d[:, 2], d[:, 3]
    ax_x, ax_y, ax_z = a[:, 1], a[:, 2], a[:, 3]

    ensure_dir(out_dir)

    # --- Desired vs actual X ---
    plt.figure(figsize=(8, 3))
    plt.plot(td, dx, label="EE_des_x")
    plt.plot(ta, ax_x, label="EE_act_x")
    plt.xlabel("time (s)")
    plt.ylabel("x (m)")
    plt.title(f"{robot_ns} EE x desired vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{robot_ns}_ee_x.png"))
    plt.close()

    # --- Desired vs actual Y ---
    plt.figure(figsize=(8, 3))
    plt.plot(td, dy, label="EE_des_y")
    plt.plot(ta, ax_y, label="EE_act_y")
    plt.xlabel("time (s)")
    plt.ylabel("y (m)")
    plt.title(f"{robot_ns} EE y desired vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{robot_ns}_ee_y.png"))
    plt.close()

    # --- Desired vs actual Z ---
    plt.figure(figsize=(8, 3))
    plt.plot(td, dz, label="EE_des_z")
    plt.plot(ta, ax_z, label="EE_act_z")
    plt.xlabel("time (s)")
    plt.ylabel("z (m)")
    plt.title(f"{robot_ns} EE z desired vs actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{robot_ns}_ee_z.png"))
    plt.close()

    # --- Euclidean position error norm ---
    order = np.argsort(ta)
    ta_sorted = ta[order]
    ax_x_sorted = ax_x[order]
    ax_y_sorted = ax_y[order]
    ax_z_sorted = ax_z[order]
    ax_x_r = np.interp(td, ta_sorted, ax_x_sorted)
    ax_y_r = np.interp(td, ta_sorted, ax_y_sorted)
    ax_z_r = np.interp(td, ta_sorted, ax_z_sorted)
    err = np.sqrt((dx - ax_x_r) ** 2 + (dy - ax_y_r) ** 2 + (dz - ax_z_r) ** 2)

    plt.figure(figsize=(8, 3))
    plt.plot(td, err)
    plt.xlabel("time (s)")
    plt.ylabel("pos error (m)")
    plt.title(f"{robot_ns} EE position error norm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{robot_ns}_ee_error.png"))
    plt.close()

    # --- Contact forces (if present) ---
    if c.size != 0:
        tc = c[:, 0] - t0
        fx_c, fy_c, fz_c = c[:, 1], c[:, 2], c[:, 3]
        plt.figure(figsize=(8, 3))
        plt.plot(tc, fx_c, label="fx")
        plt.plot(tc, fy_c, label="fy")
        plt.plot(tc, fz_c, label="fz")
        plt.xlabel("time (s)")
        plt.ylabel("force (N)")
        plt.title(f"{robot_ns} contact wrench")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{robot_ns}_contact_wrench.png"))
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-arm bag CSV plotter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input-dir",
        default="logs/run_single_arm/csv_extract",
        help="Directory containing extracted CSV files (default: logs/run_single_arm/csv_extract).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots. Defaults to <input-dir>/../plots.",
    )
    p.add_argument(
        "--robot-ns",
        default="robot1",
        help="Robot namespace used in CSV filenames (default: robot1).",
    )
    return p


def main() -> None:
    args = build_cli().parse_args()
    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input_dir), "plots")

    print(f"Single-arm bag CSV plotter")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Robot:  {args.robot_ns}")

    try:
        plot_ee_single_arm(args.robot_ns, args.input_dir, out_dir)
        print(f"Plots written to {out_dir}")
    except Exception as exc:
        print(f"Failed to plot for {args.robot_ns}: {exc}")


if __name__ == "__main__":
    main()
