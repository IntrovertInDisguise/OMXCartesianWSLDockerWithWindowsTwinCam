#!/usr/bin/env python3
"""
Numerical IK solver for OMX arm to reach 12cm forward at platform height.
Finds joint angles (joint2, joint3, joint4) to place end_effector at target.
"""

import math
import numpy as np
from scipy.optimize import minimize

# OMX arm kinematics constants
J1_X = 0.012; J1_Z = 0.017  # joint1 origin from link1
J2_Z = 0.0595                # joint2 origin from link2 (along Z)
J3_X = 0.024; J3_Z = 0.128  # joint3 origin from link3
J4_X = 0.124                 # joint4 origin from link4
EE_X = 0.126                 # end_effector origin from link5

# Joint limits
J2_LIMITS = (-1.5, 1.5)
J3_LIMITS = (-1.5, 1.4)
J4_LIMITS = (-1.7, 1.97)

def forward_kinematics(theta2, theta3, theta4):
    """Compute end effector position from joint angles."""
    # Joint2 position (world frame, link1 origin)
    j2_x = J1_X
    j2_z = J1_Z + J2_Z
    
    # After joint2 rotation by theta2 about Y
    c2, s2 = math.cos(theta2), math.sin(theta2)
    # Joint3 = joint2 + R_y(theta2) * (J3_X, 0, J3_Z)
    j3_x = j2_x + J3_X * c2 + J3_Z * s2
    j3_z = j2_z - J3_X * s2 + J3_Z * c2
    
    # After joint2+joint3 rotation
    c23, s23 = math.cos(theta2+theta3), math.sin(theta2+theta3)
    # Joint4 = joint3 + R_y(theta2+theta3) * (J4_X, 0, 0)
    j4_x = j3_x + J4_X * c23
    j4_z = j3_z - J4_X * s23
    
    # After joint2+joint3+joint4 rotation
    c234, s234 = math.cos(theta2+theta3+theta4), math.sin(theta2+theta3+theta4)
    # EE = joint4 + R_y(theta2+theta3+theta4) * (EE_X, 0, 0)
    ee_x = j4_x + EE_X * c234
    ee_z = j4_z - EE_X * s234
    
    return ee_x, ee_z

def gripper_direction(theta2, theta3, theta4):
    """Compute gripper forward direction (angle from horizontal)."""
    total = theta2 + theta3 + theta4
    # Local X axis after rotation: (cos(total), -sin(total)) in XZ plane
    return total

def ik_solve(target_x, target_z, gripper_angle_target=0.0):
    """
    Solve IK for target end effector position.
    gripper_angle_target: desired angle of gripper from horizontal (0 = horizontal forward)
    """
    def cost(q):
        theta2, theta3, theta4 = q
        ee_x, ee_z = forward_kinematics(theta2, theta3, theta4)
        pos_err = (ee_x - target_x)**2 + (ee_z - target_z)**2
        
        # Gripper orientation cost
        grip_angle = gripper_direction(theta2, theta3, theta4)
        orient_err = (grip_angle - gripper_angle_target)**2
        
        return pos_err * 1000 + orient_err * 10
    
    # Try multiple initial guesses
    best_result = None
    best_cost = float('inf')
    
    for j2_init in np.linspace(J2_LIMITS[0], J2_LIMITS[1], 7):
        for j3_init in np.linspace(J3_LIMITS[0], J3_LIMITS[1], 7):
            for j4_init in np.linspace(J4_LIMITS[0], J4_LIMITS[1], 5):
                x0 = [j2_init, j3_init, j4_init]
                bounds = [J2_LIMITS, J3_LIMITS, J4_LIMITS]
                
                result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds)
                
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result
    
    return best_result

def main():
    # Target: gripper TIPS at (0.12, 0.0) relative to link1
    # Gripper tips = end_effector_link at 0.126 from link5
    # We want gripper pointing horizontal or slightly downward so TIPS touch platform
    target_x = 0.12
    target_z = 0.0
    
    print("=" * 60)
    print("OMX Arm IK Solver - Target: (0.12, 0.0) from link1")
    print("Goal: gripper TIPS touching platform (horizontal/downward angle)")
    print("=" * 60)
    
    # Try orientations from horizontal (0°) to slightly downward (-30°)
    # This ensures tips are the lowest point, not the base
    for grip_angle_deg in [0, -5, -10, -15, -20, -25, -30]:
        grip_angle = math.radians(grip_angle_deg)
        result = ik_solve(target_x, target_z, grip_angle)
        
        if result is not None:
            theta2, theta3, theta4 = result.x
            ee_x, ee_z = forward_kinematics(theta2, theta3, theta4)
            
            print(f"\n--- Gripper angle: {grip_angle_deg}° from horizontal ---")
            print(f"  joint2 = {theta2:.4f} rad ({math.degrees(theta2):.1f}°)")
            print(f"  joint3 = {theta3:.4f} rad ({math.degrees(theta3):.1f}°)")
            print(f"  joint4 = {theta4:.4f} rad ({math.degrees(theta4):.1f}°)")
            print(f"  EE pos = ({ee_x:.4f}, {ee_z:.4f})")
            print(f"  Error  = ({abs(ee_x-target_x)*1000:.1f}mm, {abs(ee_z-target_z)*1000:.1f}mm)")
            print(f"  Cost   = {result.fun:.6f}")
    
    # Also check: what is the closest reach to (0.12, 0) with horizontal gripper?
    print("\n" + "=" * 60)
    print("Closest reachable point to (0.12, 0.0) with horizontal gripper")
    print("=" * 60)
    
    def closest_cost(q):
        theta2, theta3, theta4 = q
        ee_x, ee_z = forward_kinematics(theta2, theta3, theta4)
        pos_err = (ee_x - target_x)**2 + (ee_z - target_z)**2
        grip_angle = gripper_direction(theta2, theta3, theta4)
        orient_err = (grip_angle - 0.0)**2
        return pos_err * 1000 + orient_err * 100
    
    best_result = None
    best_cost = float('inf')
    for j2_init in np.linspace(J2_LIMITS[0], J2_LIMITS[1], 11):
        for j3_init in np.linspace(J3_LIMITS[0], J3_LIMITS[1], 11):
            j4_init = -(j2_init + j3_init)  # Force horizontal gripper
            j4_init = max(J4_LIMITS[0], min(J4_LIMITS[1], j4_init))
            x0 = [j2_init, j3_init, j4_init]
            bounds = [J2_LIMITS, J3_LIMITS, J4_LIMITS]
            result = minimize(closest_cost, x0, method='L-BFGS-B', bounds=bounds)
            if result.fun < best_cost:
                best_cost = result.fun
                best_result = result
    
    if best_result is not None:
        theta2, theta3, theta4 = best_result.x
        ee_x, ee_z = forward_kinematics(theta2, theta3, theta4)
        print(f"  joint2 = {theta2:.4f} rad ({math.degrees(theta2):.1f}°)")
        print(f"  joint3 = {theta3:.4f} rad ({math.degrees(theta3):.1f}°)")
        print(f"  joint4 = {theta4:.4f} rad ({math.degrees(theta4):.1f}°)")
        print(f"  EE pos = ({ee_x:.4f}, {ee_z:.4f})")
        print(f"  Distance from target = {math.sqrt((ee_x-target_x)**2 + (ee_z-target_z)**2)*100:.1f}cm")

if __name__ == '__main__':
    main()
