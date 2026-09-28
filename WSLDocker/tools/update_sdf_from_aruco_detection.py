#!/usr/bin/env python3
"""Update spring_assembly SDF model with detected ArUco marker position.

Reads the detected marker world position (from hardware camera) and updates
the spring_assembly/model.sdf so Gazebo simulation uses the actual detected
geometry instead of the hardcoded SDF values.

This creates a feedback loop:
1. Hardware run detects ArUco marker position from camera
2. This script updates spring_assembly/model.sdf with detected position
3. Gazebo simulation uses the updated SDF geometry

Usage::

    # After hardware detection, update SDF for next Gazebo run
    python3 tools/update_sdf_from_aruco_detection.py \\
        --detection logs/hardware_run/detected_marker.json \\
        --sdf ws/src/open_manipulator/open_manipulator_x_description/models/spring_assembly/model.sdf
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Tuple


# Expected marker pose in spring_assembly model frame (from SDF)
EXPECTED_MARKER_POSE = (-0.12, -0.12, 0.0505, 0.0, 0.0, 0.0)


def parse_detected_marker(detection_path: str) -> Tuple[float, float, float]:
    """Read detected marker world position from JSON.
    
    Expected JSON format:
    {
        "marker_world_x": -0.118,
        "marker_world_y": -0.121,
        "marker_world_z": 0.0508,
        ...
    }
    """
    if not os.path.isfile(detection_path):
        raise FileNotFoundError(f"Detection file not found: {detection_path}")
    
    with open(detection_path, "r") as fh:
        data = json.load(fh)
    
    # Try to extract marker world position from _meta (compute_start_position_from_aruco.py format)
    meta = data.get("_meta", {})
    marker_x = meta.get("marker_world_x") or data.get("marker_world_x")
    marker_y = meta.get("marker_world_y") or data.get("marker_world_y")
    marker_z = meta.get("marker_world_z") or data.get("marker_world_z")
    
    if marker_x is None or marker_y is None or marker_z is None:
        # Try alternative format: detected_position as [x, y, z]
        detected = data.get("detected_position") or data.get("marker_position")
        if detected and len(detected) >= 3:
            marker_x, marker_y, marker_z = detected[0], detected[1], detected[2]
        else:
            raise ValueError(f"Could not find marker_world_x/y/z in {detection_path}")
    
    return float(marker_x), float(marker_y), float(marker_z)


def update_sdf_marker_pose(sdf_path: str, marker_x: float, marker_y: float, marker_z: float) -> None:
    """Update the aruco_metal link pose in spring_assembly/model.sdf."""
    
    with open(sdf_path, "r") as fh:
        content = fh.read()
    
    # Find and replace the aruco_metal pose
    # Pattern: <link name="aruco_metal">\n      <pose>X Y Z R P Y</pose>
    pattern = r'(<link name="aruco_metal">\s*<pose>)([^<]+)(</pose>)'
    
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find aruco_metal pose in {sdf_path}")
    
    old_pose = match.group(2).strip()
    new_pose = f"{marker_x} {marker_y} {marker_z} 0 0 0"
    
    # Replace the pose
    updated_content = content[:match.start(2)] + new_pose + content[match.end(2):]
    
    with open(sdf_path, "w") as fh:
        fh.write(updated_content)
    
    print(f"Updated aruco_metal pose in {sdf_path}")
    print(f"  Old: {old_pose}")
    print(f"  New: {new_pose}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detection",
        required=True,
        help="Path to detected marker JSON (from compute_start_position_from_aruco.py)",
    )
    parser.add_argument(
        "--sdf",
        default="ws/src/open_manipulator/open_manipulator_x_description/models/spring_assembly/model.sdf",
        help="Path to spring_assembly/model.sdf",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without modifying files",
    )
    args = parser.parse_args()
    
    if not os.path.isfile(args.sdf):
        print(f"ERROR: SDF file not found: {args.sdf}", file=sys.stderr)
        return 1
    
    try:
        marker_x, marker_y, marker_z = parse_detected_marker(args.detection)
    except Exception as exc:
        print(f"ERROR: Failed to parse detection: {exc}", file=sys.stderr)
        return 1
    
    print(f"Detected marker position: x={marker_x:.4f}, y={marker_y:.4f}, z={marker_z:.4f}")
    print(f"Expected (SDF default):   x={EXPECTED_MARKER_POSE[0]:.4f}, y={EXPECTED_MARKER_POSE[1]:.4f}, z={EXPECTED_MARKER_POSE[2]:.4f}")
    
    dx = marker_x - EXPECTED_MARKER_POSE[0]
    dy = marker_y - EXPECTED_MARKER_POSE[1]
    dz = marker_z - EXPECTED_MARKER_POSE[2]
    print(f"Offset from expected:     dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would update SDF with detected position")
        return 0
    
    try:
        update_sdf_marker_pose(args.sdf, marker_x, marker_y, marker_z)
    except Exception as exc:
        print(f"ERROR: Failed to update SDF: {exc}", file=sys.stderr)
        return 1
    
    print("\nSDF updated successfully. Gazebo will use the detected marker position on next launch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
