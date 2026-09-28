#!/usr/bin/env python3
"""Minimal pretrial marker update helper (smoke-test friendly).

This script is a lightweight stub used for CI and smoke-testing the harness
when actual camera hardware is not attached. It accepts the same CLI args as
the real helper and writes placeholder output files under the requested
`--out-dir` so downstream steps have expected artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime


def build_parser():
    p = argparse.ArgumentParser(description="Pretrial marker update (stub for smoke tests)")
    p.add_argument("--out-dir", default="logs/aruco_hw_run_host")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--max-frames", type=int, default=40)
    p.add_argument("--rate", type=float, default=4.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = os.path.abspath(args.out_dir)
    if args.clean and os.path.exists(out):
        print(f"[pretrial_stub] cleaning {out}")
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    # Create a minimal numeric-key marker map suitable for smoke tests.
    # Use common marker ids for the rig: spring caps are typically 4 and 5.
    # tvec format: [x, y, z] in metres; rvec: Rodrigues axis-angle vector.
    marker_map_numeric = {
        "4": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.05, 0.0, 0.12]},
        "5": {"rvec": [0.0, 0.0, 0.0], "tvec": [-0.05, 0.0, 0.12]},
        "0": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]},
        "1": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.15, 0.0, 0.0]},
        "2": {"rvec": [0.0, 0.0, 0.0], "tvec": [-0.15, 0.0, 0.0]},
    }
    with open(os.path.join(out, "marker_map.json"), "w") as fh:
        json.dump(marker_map_numeric, fh, indent=2)

    # Computed map mirrors the same numeric mapping for smoke tests.
    with open(os.path.join(out, "marker_map_computed.json"), "w") as fh:
        json.dump(marker_map_numeric, fh, indent=2)

    # Minimal camera extrinsics placeholder (identity)
    extr = {"camera_extrinsics": {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}}
    with open(os.path.join(out, "camera_extrinsics.json"), "w") as fh:
        json.dump(extr, fh, indent=2)

    # depth camera calibration placeholder
    depth_calib = {"depth_to_color_extrinsics": extr["camera_extrinsics"]}
    with open(os.path.join(out, "depth_camera_calibration.json"), "w") as fh:
        json.dump(depth_calib, fh, indent=2)

    # Create a small synthetic aruco_records.jsonl to allow extrinsics fitting in smoke tests.
    records_path = os.path.join(out, "aruco_records.jsonl")
    try:
        with open(records_path, "w", encoding="utf-8") as fh:
            # Create a few frames with detections for markers 4 and 5.
            ts_base = 1_600_000_000_000_000_000
            for i in range(3):
                ts = ts_base + i * 100_000_000
                rec = {
                    "ts_ns": int(ts),
                    "color_path": "",
                    "depth_path": "",
                    "camera_info": {},
                    "detections": [],
                }
                # use the same rvec/tvec values as the marker map so camera->world fit is identity
                for mid in ("4", "5"):
                    mm = marker_map_numeric[mid]
                    det = {
                        "marker_id": int(mid),
                        "center_px": [320.0 + (1 if mid == "4" else -1) * i, 240.0],
                        "rvec": list(mm["rvec"]),
                        "tvec": list(mm["tvec"]),
                        "depth_center_m": float(mm["tvec"][2]),
                    }
                    rec["detections"].append(det)
                fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass

    print(f"[pretrial_stub] wrote placeholders to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
