#!/usr/bin/env python3
"""
Update spring_assembly_wall SDF based on detected ArUco marker positions.

Uses the detected positions of metal_platform, spring_cap1, and spring_cap2
to compute relative positions and update the SDF model.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Detected marker positions in camera frame (from ArUco snapshot)
# These should be updated with actual detected values
DETECTED_POSITIONS = {
    'metal_platform': (-0.0030, -0.1628, 0.6644),
    'spring_cap1': (0.2365, -0.0352, 0.5350),
    'spring_cap2': (0.0586, -0.0369, 0.5625),
}

def compute_relative_positions():
    """Compute positions relative to metal_platform (the model origin)."""
    metal = DETECTED_POSITIONS['metal_platform']
    
    relative = {}
    for name, pos in DETECTED_POSITIONS.items():
        if name == 'metal_platform':
            relative[name] = (0.0, 0.0, 0.0)  # Origin
        else:
            relative[name] = (
                pos[0] - metal[0],
                pos[1] - metal[1],
                pos[2] - metal[2],
            )
    
    return relative

def compute_wall_position(cap1_pos):
    """
    Compute rigid wall position.
    Wall should be 3mm past spring_cap1's outer face in the -x direction.
    Cap is 45mm thick, so outer face is at cap_x - 0.0225.
    Wall should be at outer_face - 0.003 (3mm gap).
    """
    cap1_x = cap1_pos[0]
    cap_outer_face = cap1_x - 0.0225  # Half of 45mm thickness
    wall_x = cap_outer_face - 0.003   # 3mm past outer face
    return (wall_x, cap1_pos[1], cap1_pos[2])

def update_sdf(sdf_path: str, relative_positions: dict, wall_pos: tuple):
    """Update the SDF file with computed positions."""
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    
    # Find the model element
    model = root.find('.//model[@name="spring_assembly_wall"]')
    if model is None:
        print(f"Error: Could not find model 'spring_assembly_wall' in {sdf_path}")
        return False
    
    # Update spring_cap_robot1 pose
    cap1_link = model.find('.//link[@name="spring_cap_robot1"]')
    if cap1_link is not None:
        pose = cap1_link.find('pose')
        if pose is not None:
            pos = relative_positions['spring_cap1']
            pose.text = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} 0 0 0"
            print(f"Updated spring_cap_robot1: {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
    
    # Update spring_cap_robot2 pose
    cap2_link = model.find('.//link[@name="spring_cap_robot2"]')
    if cap2_link is not None:
        pose = cap2_link.find('pose')
        if pose is not None:
            pos = relative_positions['spring_cap2']
            pose.text = f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f} 0 0 0"
            print(f"Updated spring_cap_robot2: {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
    
    # Update rigid_wall pose
    wall_link = model.find('.//link[@name="rigid_wall"]')
    if wall_link is not None:
        pose = wall_link.find('pose')
        if pose is not None:
            pose.text = f"{wall_pos[0]:.4f} {wall_pos[1]:.4f} {wall_pos[2]:.4f} 0 0 0"
            print(f"Updated rigid_wall: {wall_pos[0]:.4f} {wall_pos[1]:.4f} {wall_pos[2]:.4f}")
    
    # Write updated SDF
    tree.write(sdf_path, encoding='unicode', xml_declaration=True)
    print(f"\nSDF updated: {sdf_path}")
    return True

def main():
    print("=== Computing relative positions ===")
    relative = compute_relative_positions()
    
    print("\nPositions relative to metal_platform:")
    for name, pos in relative.items():
        print(f"  {name}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    
    # Compute wall position
    wall_pos = compute_wall_position(relative['spring_cap1'])
    print(f"\nRigid wall position: x={wall_pos[0]:.4f}, y={wall_pos[1]:.4f}, z={wall_pos[2]:.4f}")
    
    # Find SDF file
    sdf_path = Path("/workspaces/omx_ros2/ws/src/open_manipulator/open_manipulator_x_description/models/spring_assembly_wall/model.sdf")
    
    if not sdf_path.exists():
        print(f"\nError: SDF file not found: {sdf_path}")
        sys.exit(1)
    
    print(f"\n=== Updating SDF: {sdf_path} ===")
    
    # Backup original
    backup_path = sdf_path.with_suffix('.sdf.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy(sdf_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    # Update SDF
    if update_sdf(str(sdf_path), relative, wall_pos):
        print("\n=== SDF update complete ===")
        print("\nNext steps:")
        print("1. Rebuild the package: cd /workspaces/omx_ros2/ws && colcon build --packages-select open_manipulator_x_description")
        print("2. Source the workspace: source /workspaces/omx_ros2/ws/install/setup.bash")
        print("3. Test with: bash run_single_arm_test_gazebo.sh")
    else:
        print("\nSDF update failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
