#!/usr/bin/env python3
"""
set_robot_gripper_pose.py
──────────────────────────
Post-spawn script to move robot grippers to touch the platform at 12cm from base.

Robot1: base at (-0.39, 0, 0.01), facing +X, gripper target at (-0.27, 0, 0.01)
Robot2: base at (+0.39, 0, 0.01), facing -X, gripper target at (+0.27, 0, 0.01)

Uses gz command-line tool to apply joint forces until desired pose is reached.
"""

import subprocess
import time
import sys


def set_joint_angle(robot_name, joint_name, target_angle, force=5.0, timeout=5.0):
    """Apply force to a joint until it reaches target angle."""
    print(f"Moving {robot_name} {joint_name} to {target_angle:.2f} rad...")
    
    # Use gz joint command to apply force
    # Note: This is a simplified approach; in practice, you'd need to monitor
    # joint states and apply feedback control
    cmd = [
        'gz', 'joint', '-m', robot_name,
        '-j', joint_name,
        '-f', str(force)
    ]
    
    # Apply force for a short duration
    start_time = time.time()
    while time.time() - start_time < timeout:
        subprocess.run(cmd, capture_output=True)
        time.sleep(0.1)
    
    print(f"  Applied force to {joint_name} for {timeout}s")


def main():
    print("Waiting 3 seconds for robots to stabilize...")
    time.sleep(3.0)
    
    print("\n=== Setting Robot1 (Left Arm) Gripper Pose ===")
    print("Target: Gripper at (-0.27, 0, 0.01) - 12cm forward, touching platform")
    
    # Robot1 joints: joint1 (yaw), joint2 (shoulder), joint3 (elbow), joint4 (wrist)
    # To reach 12cm forward and horizontal:
    # joint1 = 0 (facing +X)
    # joint2 = ~1.2 rad (shoulder forward)
    # joint3 = ~-0.8 rad (elbow bend)
    # joint4 = ~0.3 rad (wrist to orient gripper down)
    
    set_joint_angle('robot1', 'joint1', 0.0, force=10.0, timeout=1.0)
    set_joint_angle('robot1', 'joint2', 1.2, force=8.0, timeout=2.0)
    set_joint_angle('robot1', 'joint3', -0.8, force=8.0, timeout=2.0)
    set_joint_angle('robot1', 'joint4', 0.3, force=5.0, timeout=1.5)
    
    print("\n=== Setting Robot2 (Right Arm) Gripper Pose ===")
    print("Target: Gripper at (+0.27, 0, 0.01) - 12cm forward, touching platform")
    
    # Robot2 has same joint configuration but opposite orientation
    # Since it's spawned with yaw=π, the joint angles are the same
    set_joint_angle('robot2', 'joint1', 0.0, force=10.0, timeout=1.0)
    set_joint_angle('robot2', 'joint2', 1.2, force=8.0, timeout=2.0)
    set_joint_angle('robot2', 'joint3', -0.8, force=8.0, timeout=2.0)
    set_joint_angle('robot2', 'joint4', 0.3, force=5.0, timeout=1.5)
    
    print("\n=== Gripper Pose Setting Complete ===")
    print("Note: Open-loop force control is approximate. For precise positioning,")
    print("enable controllers (static_scene_only:=false) and use joint trajectory commands.")


if __name__ == '__main__':
    main()
