#!/usr/bin/env python3
"""Small helper to plot or summarize sync metrics CSVs produced by the harness.

Usage examples:
  python3 tools/plot_sync_metrics.py /tmp/variable_stiffness_logs/sync_metrics_20260402_123456.csv

This script is intentionally lightweight and depends only on matplotlib/numpy if plotting.
"""

import argparse
import csv
import os
from typing import List

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def read_sync_csv(path: str) -> List[dict]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r]


def numeric(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def summarize(rows):
    if not rows:
        print("No rows")
        return
    t = [numeric(r, "timestamp") for r in rows]
    dz = [numeric(r, "ee_z_diff") for r in rows]
    fz = [numeric(r, "contact_fz_diff") for r in rows]
    print(f"Rows: {len(rows)}")
    print(f"ee_z_diff: mean={np.nanmean(dz):.6f}, std={np.nanstd(dz):.6f}")
    print(f"contact_fz_diff: mean={np.nanmean(fz):.6f}, std={np.nanstd(fz):.6f}")


def plot(rows, out=None):
    if plt is None:
        print("matplotlib not available; cannot plot")
        return
    t = [numeric(r, "timestamp") for r in rows]
    dz = [numeric(r, "ee_z_diff") for r in rows]
    fz = [numeric(r, "contact_fz_diff") for r in rows]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(t, dz, label="ee_z_diff (m)")
    ax[0].legend()
    ax[1].plot(t, fz, label="contact_fz_diff (N)")
    ax[1].legend()
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    if out:
        fig.savefig(out)
        print(f"Saved plot to {out}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="sync_metrics CSV path")
    p.add_argument("--plot-out", default=None, help="save plot to file instead of displaying")
    args = p.parse_args()

    rows = read_sync_csv(args.csv)
    summarize(rows)
    plot(rows, args.plot_out)


if __name__ == "__main__":
    main()
