#!/usr/bin/env python3
"""Calculate and update controller YAML start_position from platform geometry.

Reads platform dimensions and marker position, calculates safe start positions
outside the platform edges, and updates the robot controller YAMLs.

Usage::

    # With detected marker position
    python3 tools/update_start_positions_from_geometry.py \\
        --marker-x -0.118 \\
        --marker-y -0.121 \\
        --marker-z 0.0508 \\
        --safety-margin 0.02 \\
        --robot1-yaml ws/src/.../robot1_variable_stiffness.yaml \\
        --robot2-yaml ws/src/.../robot2_variable_stiffness.yaml

    # From known geometry (no detection)
    python3 tools/update_start_positions_from_geometry.py \\
        --robot1-yaml ws/src/.../robot1_variable_stiffness.yaml \\
        --robot2-yaml ws/src/.../robot2_variable_stiffness.yaml

    # Dry-run (show calculation without writing)
    python3 tools/update_start_positions_from_geometry.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Tuple

# Add tools to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from compute_start_position_from_aruco import (
    METAL_PLATFORM_SIZE_M,
    MARKER_OFFSET_IN_PLATFORM_FRAME,
    MARKER_TO_LEFT_EDGE_X,
    MARKER_TO_RIGHT_EDGE_X,
    ROBOT1_BASE_X,
    ROBOT2_BASE_X,
    SAFETY_MARGIN_M,
    DEFAULT_START_Z,
    platform_edges_from_marker,
    safe_start_positions,
)


def calculate_safe_positions(
    marker_x: float = MARKER_OFFSET_IN_PLATFORM_FRAME[0],
    marker_y: float = MARKER_OFFSET_IN_PLATFORM_FRAME[1],
    safety_margin: float = SAFETY_MARGIN_M,
    start_z: float = DEFAULT_START_Z,
) -> Tuple[dict, dict]:
    """Calculate safe start positions for both robots.
    
    Returns:
        (robot1_config, robot2_config) where each is a dict with:
        - local: [x, y, z] in robot-local coordinates
        - world: [x, y, z] in world coordinates
    
    Note: local_x is capped at 0.10 to prevent platform collision during homing.
    The homing routine moves the arms to start_position, and if local_x is too large,
    the arms extend into the platform footprint.
    """
    platform_left, platform_right = platform_edges_from_marker(marker_x)
    platform_center_y = marker_y - MARKER_OFFSET_IN_PLATFORM_FRAME[1]
    positions = safe_start_positions(
        platform_left,
        platform_right,
        safety_margin,
        start_z,
        start_y=platform_center_y,
    )
    
    robot1_local = positions["robot1"]
    robot2_local = positions["robot2"]
    
    # Cap local_x to prevent platform collision during homing
    max_local_x = 0.10  # Keeps EE at world_x = ±0.29 (outside platform at ±0.15)
    robot1_local[0] = min(robot1_local[0], max_local_x)
    robot2_local[0] = min(robot2_local[0], max_local_x)
    
    robot1_world_x = ROBOT1_BASE_X + robot1_local[0]
    robot2_world_x = ROBOT2_BASE_X - robot2_local[0]
    
    return {
        "local": robot1_local,
        "world": [robot1_world_x, robot1_local[1], robot1_local[2]],
    }, {
        "local": robot2_local,
        "world": [robot2_world_x, robot2_local[1], robot2_local[2]],
    }


def update_yaml_start_position(yaml_path: str, new_position: list, dry_run: bool = False) -> bool:
    """Update start_position in a YAML file.
    
    Args:
        yaml_path: Path to YAML file
        new_position: New [x, y, z] position
        dry_run: If True, show what would be changed without writing
    
    Returns:
        True if updated (or would be updated in dry-run mode)
    """
    if not os.path.isfile(yaml_path):
        print(f"ERROR: YAML file not found: {yaml_path}", file=sys.stderr)
        return False
    
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Find start_position line
    pattern = r'^(\s*start_position:\s*)\[([^\]]+)\]'
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        print(f"WARNING: No start_position found in {yaml_path}", file=sys.stderr)
        return False
    
    old_position_str = match.group(2)
    new_position_str = f"{new_position[0]}, {new_position[1]}, {new_position[2]}"
    
    if dry_run:
        print(f"  Would update {yaml_path}:")
        print(f"    Old: start_position: [{old_position_str}]")
        print(f"    New: start_position: [{new_position_str}]")
        return True
    
    # Replace the position
    updated_content = content[:match.start(2)] + new_position_str + content[match.end(2):]
    
    with open(yaml_path, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {yaml_path}:")
    print(f"  Old: start_position: [{old_position_str}]")
    print(f"  New: start_position: [{new_position_str}]")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-x", type=float, default=MARKER_OFFSET_IN_PLATFORM_FRAME[0],
                        help=f"Marker world X position (default: {MARKER_OFFSET_IN_PLATFORM_FRAME[0]})")
    parser.add_argument("--marker-y", type=float, default=MARKER_OFFSET_IN_PLATFORM_FRAME[1],
                        help=f"Marker world Y position (default: {MARKER_OFFSET_IN_PLATFORM_FRAME[1]})")
    parser.add_argument("--marker-z", type=float, default=MARKER_OFFSET_IN_PLATFORM_FRAME[2],
                        help=f"Marker world Z position (default: {MARKER_OFFSET_IN_PLATFORM_FRAME[2]})")
    parser.add_argument("--safety-margin", type=float, default=SAFETY_MARGIN_M,
                        help=f"Safety margin outside platform edge in meters (default: {SAFETY_MARGIN_M})")
    parser.add_argument("--start-z", type=float, default=DEFAULT_START_Z,
                        help=f"Start height Z in meters (default: {DEFAULT_START_Z})")
    parser.add_argument("--robot1-yaml", type=str,
                        help="Path to robot1_variable_stiffness.yaml")
    parser.add_argument("--robot2-yaml", type=str,
                        help="Path to robot2_variable_stiffness.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show calculations without writing files")
    parser.add_argument("--output-json", type=str,
                        help="Output calculated positions to JSON file")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CALCULATING SAFE START POSITIONS FROM PLATFORM GEOMETRY")
    print("=" * 70)
    print()
    print(f"Platform size: {METAL_PLATFORM_SIZE_M:.3f}m")
    print(f"Marker position: ({args.marker_x:.3f}, {args.marker_y:.3f}, {args.marker_z:.3f})")
    print(f"Safety margin: {args.safety_margin:.3f}m")
    print(f"Start height: {args.start_z:.3f}m")
    print()
    
    # Calculate safe positions
    robot1, robot2 = calculate_safe_positions(
        marker_x=args.marker_x,
        marker_y=args.marker_y,
        safety_margin=args.safety_margin,
        start_z=args.start_z,
    )
    
    print("Calculated safe positions:")
    print(f"  Robot1 local: [{robot1['local'][0]:.3f}, {robot1['local'][1]:.3f}, {robot1['local'][2]:.3f}]")
    print(f"  Robot1 world: [{robot1['world'][0]:.3f}, {robot1['world'][1]:.3f}, {robot1['world'][2]:.3f}]")
    print()
    print(f"  Robot2 local: [{robot2['local'][0]:.3f}, {robot2['local'][1]:.3f}, {robot2['local'][2]:.3f}]")
    print(f"  Robot2 world: [{robot2['world'][0]:.3f}, {robot2['world'][1]:.3f}, {robot2['world'][2]:.3f}]")
    print()
    
    # Calculate platform edges for verification
    platform_left, platform_right = platform_edges_from_marker(args.marker_x)
    print(f"Platform edges: left={platform_left:.3f}m, right={platform_right:.3f}m")
    print(f"Robot1 world_x: {robot1['world'][0]:.3f}m (margin: {abs(robot1['world'][0] - platform_left):.3f}m)")
    print(f"Robot2 world_x: {robot2['world'][0]:.3f}m (margin: {abs(robot2['world'][0] - platform_right):.3f}m)")
    print()
    
    # Output JSON if requested
    if args.output_json:
        output = {
            "robot1": robot1["local"],
            "robot2": robot2["local"],
            "_meta": {
                "marker_x": args.marker_x,
                "marker_y": args.marker_y,
                "marker_z": args.marker_z,
                "platform_center_y": args.marker_y - MARKER_OFFSET_IN_PLATFORM_FRAME[1],
                "safety_margin": args.safety_margin,
                "start_z": args.start_z,
                "platform_left": platform_left,
                "platform_right": platform_right,
                "robot1_world": robot1["world"],
                "robot2_world": robot2["world"],
            }
        }
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Wrote positions to {args.output_json}")
        print()
    
    # Update YAML files if provided
    if args.robot1_yaml or args.robot2_yaml:
        print("Updating controller YAMLs:")
        if args.robot1_yaml:
            success = update_yaml_start_position(args.robot1_yaml, robot1["local"], args.dry_run)
            if not success:
                sys.exit(1)
            print()
        
        if args.robot2_yaml:
            success = update_yaml_start_position(args.robot2_yaml, robot2["local"], args.dry_run)
            if not success:
                sys.exit(1)
            print()
    elif not args.output_json:
        print("No output requested. Use --robot1-yaml/--robot2-yaml to update YAMLs")
        print("or --output-json to save to JSON, or --dry-run to preview.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
