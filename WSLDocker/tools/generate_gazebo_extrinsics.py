#!/usr/bin/env python3
"""Generate synthetic camera extrinsics from a Gazebo world camera pose.

Reads the ``<pose>`` of the ``user_camera`` from the Gazebo world file and
produces a ``camera_extrinsics.json`` compatible with
``compute_start_position_from_aruco.py``.

This is used for simulation where no real camera image is available.

Usage::

    python3 tools/generate_gazebo_extrinsics.py \\
        --world ws/src/omx_variable_stiffness_controller/worlds/empty.world \\
        --output ws/src/omx_variable_stiffness_controller/config/gazebo_camera_extrinsics.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np


def euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert SDF Euler angles (roll, pitch, yaw) to 3×3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,      cp * sr,               cp * cr],
    ])


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def parse_world_camera_pose(world_path: str) -> tuple:
    """Extract the user_camera pose from a Gazebo world file.

    Returns (x, y, z, roll, pitch, yaw).
    """
    with open(world_path, "r") as fh:
        content = fh.read()

    # Try XML parsing first
    try:
        root = ET.fromstring(content)
        # Search for gui/camera/pose
        for gui in root.iter("gui"):
            for cam in gui.iter("camera"):
                if cam.get("name") == "user_camera":
                    pose_elem = cam.find("pose")
                    if pose_elem is not None and pose_elem.text:
                        parts = [float(v) for v in pose_elem.text.strip().split()]
                        if len(parts) == 6:
                            return tuple(parts)
    except ET.ParseError:
        pass

    # Fallback: regex search for <pose> inside <camera name='user_camera'>
    m = re.search(
        r"<camera\s+name=['\"]user_camera['\"]>.*?<pose>([^<]+)</pose>",
        content, re.DOTALL,
    )
    if m:
        parts = [float(v) for v in m.group(1).strip().split()]
        if len(parts) == 6:
            return tuple(parts)

    raise RuntimeError(f"Could not find user_camera pose in {world_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world",
        default="ws/src/omx_variable_stiffness_controller/worlds/empty.world",
        help="Path to Gazebo world file",
    )
    parser.add_argument(
        "--output",
        default="ws/src/omx_variable_stiffness_controller/config/gazebo_camera_extrinsics.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.world):
        print(f"ERROR: World file not found: {args.world}", file=sys.stderr)
        return 1

    x, y, z, roll, pitch, yaw = parse_world_camera_pose(args.world)
    print(f"Camera pose from world: x={x}, y={y}, z={z}, roll={roll}, pitch={pitch}, yaw={yaw}")

    R = euler_to_rotation(roll, pitch, yaw)
    T_cam_to_world = make_homogeneous(R, np.array([x, y, z]))
    T_world_to_cam = invert_homogeneous(T_cam_to_world)

    out = {
        "source": "gazebo_world_camera_pose",
        "world_file": args.world,
        "pose_sdf": [x, y, z, roll, pitch, yaw],
        "description": "Synthetic extrinsics from Gazebo empty.world user_camera pose",
        "camera_to_world": T_cam_to_world.tolist(),
        "world_to_camera": T_world_to_cam.tolist(),
        "rotation_matrix": R.tolist(),
        "translation": [x, y, z],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")

    print(f"Wrote extrinsics to {args.output}")
    print(f"T_cam_to_world:\n{np.array2string(T_cam_to_world, precision=4, suppress_small=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
