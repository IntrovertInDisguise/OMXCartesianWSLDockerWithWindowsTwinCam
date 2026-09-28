#!/usr/bin/env python3
"""
set_robot_arm_pose.py
─────────────────────
Post-spawn script to position robot arms so grippers touch the green platform
at 12cm from the robot base.

Uses forward kinematics to calculate required joint angles, then uses
Gazebo's /set_entity_state service to set link poses.

Robot1: base at (-0.39, 0, 0.01), facing +X
Robot2: base at (+0.39, 0, 0.01), facing -X (yaw=π)

Target: Gripper at 12cm forward from base, at z=0.01 (platform surface)
"""

import sys
import time
import math
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist


class ArmPoseSetter(Node):
    def __init__(self):
        super().__init__('arm_pose_setter')
        self.client = self.create_client(SetEntityState, '/set_entity_state')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /set_entity_state service...')
        
        self.get_logger().info('Service available, setting arm poses...')
    
    def calculate_joint_angles(self):
        """
        Calculate joint angles for gripper to reach 12cm forward at platform height.
        
        OMX arm kinematics (simplified 2D in XZ plane):
        - joint1 (yaw, Z axis): 0 for forward facing
        - joint2 (pitch, Y axis): shoulder
        - joint3 (pitch, Y axis): elbow  
        - joint4 (pitch, Y axis): wrist
        
        Link lengths:
        - link2→joint2: (0, 0, 0.0595)
        - joint2→joint3: (0.024, 0, 0.128) - upper arm
        - joint3→joint4: (0.124, 0, 0) - forearm
        - joint4→end_effector: (0.126, 0, 0) - gripper
        
        Target: end_effector at (0.12, 0, 0) from link1 base
        """
        
        # Target position relative to link1 origin
        target_x = 0.12  # 12cm forward
        target_z = 0.0   # At same height as base (touching platform)
        
        # Joint offsets
        j1_offset_x = 0.012
        j1_offset_z = 0.017
        j2_offset_z = 0.0595
        j3_offset_x = 0.024
        j3_offset_z = 0.128
        j4_offset_x = 0.124
        ee_offset_x = 0.126
        
        # Total reach from joint2 to end_effector at zero angles:
        # x: 0.024 + 0.124 + 0.126 = 0.274
        # z: 0.128
        # But we need to bend the arm to reach (0.12, 0) from joint2
        
        # Position of joint2 relative to link1
        j2_x = j1_offset_x
        j2_z = j1_offset_z + j2_offset_z
        
        # Position we need to reach from joint2
        reach_x = target_x - j2_x  # 0.12 - 0.012 = 0.108
        reach_z = target_z - j2_z  # 0.0 - 0.0765 = -0.0765
        
        # Upper arm length (joint2 to joint3)
        L1 = math.sqrt(j3_offset_x**2 + j3_offset_z**2)  # ~0.130
        
        # Forearm + gripper length (joint3 to end_effector)
        L2 = j4_offset_x + ee_offset_x  # 0.124 + 0.126 = 0.250
        
        # Distance from joint2 to target
        D = math.sqrt(reach_x**2 + reach_z**2)
        
        self.get_logger().info(f'Target from joint2: ({reach_x:.3f}, {reach_z:.3f}), distance={D:.3f}')
        self.get_logger().info(f'Upper arm L1={L1:.3f}, forearm+gripper L2={L2:.3f}')
        
        # Check if reachable
        if D > L1 + L2:
            self.get_logger().error(f'Target not reachable! D={D:.3f} > L1+L2={L1+L2:.3f}')
            return None
        
        # Use law of cosines to find joint angles
        # Angle at joint2 (between upper arm and line to target)
        cos_theta2 = (L1**2 + D**2 - L2**2) / (2 * L1 * D)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))  # Clamp
        theta2_offset = math.acos(cos_theta2)
        
        # Angle of line to target from horizontal
        alpha = math.atan2(reach_z, reach_x)
        
        # joint2 angle (shoulder) - need to bend down and forward
        joint2 = alpha + theta2_offset
        
        # Angle at joint3 (elbow bend)
        cos_theta3 = (L1**2 + L2**2 - D**2) / (2 * L1 * L2)
        cos_theta3 = max(-1.0, min(1.0, cos_theta3))
        theta3_bend = math.acos(cos_theta3)
        
        # joint3 angle (elbow)
        joint3 = math.pi - theta3_bend
        
        # joint4 angle (wrist) - orient gripper horizontally
        # Total angle from joint2 to end_effector should be alpha
        # joint2 + joint3 + joint4 = alpha (for horizontal gripper)
        joint4 = alpha - joint2 - joint3
        
        self.get_logger().info(f'Calculated joint angles:')
        self.get_logger().info(f'  joint1 = 0.0 (facing forward)')
        self.get_logger().info(f'  joint2 = {joint2:.3f} rad ({math.degrees(joint2):.1f}°)')
        self.get_logger().info(f'  joint3 = {joint3:.3f} rad ({math.degrees(joint3):.1f}°)')
        self.get_logger().info(f'  joint4 = {joint4:.3f} rad ({math.degrees(joint4):.1f}°)')
        
        return {
            'joint1': 0.0,
            'joint2': joint2,
            'joint3': joint3,
            'joint4': joint4,
            'gripper_left': 0.0,
            'gripper_right': 0.0
        }
    
    def set_link_pose(self, model_name, link_name, position, orientation):
        """Set the pose of a specific link in a model."""
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = f'{model_name}::{link_name}'
        request.state.pose = Pose()
        request.state.pose.position.x = position[0]
        request.state.pose.position.y = position[1]
        request.state.pose.position.z = position[2]
        request.state.pose.orientation.x = orientation[0]
        request.state.pose.orientation.y = orientation[1]
        request.state.pose.orientation.z = orientation[2]
        request.state.pose.orientation.w = orientation[3]
        request.state.reference_frame = 'world'
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            if future.result().success:
                self.get_logger().info(f'Set {model_name}::{link_name} pose')
                return True
            else:
                self.get_logger().error(f'Failed to set {model_name}::{link_name}')
                return False
        else:
            self.get_logger().error('Service call failed')
            return False


def main():
    rclpy.init()
    node = ArmPoseSetter()
    
    node.get_logger().info('Waiting 3 seconds for robots to stabilize...')
    time.sleep(3.0)
    
    # Calculate joint angles
    joint_angles = node.calculate_joint_angles()
    if joint_angles is None:
        node.get_logger().error('Could not calculate joint angles')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    
    # Note: Setting individual link poses via /set_entity_state will violate
    # joint constraints and may cause physics instability. A better approach
    # would be to enable controllers and command joint positions.
    
    node.get_logger().info('')
    node.get_logger().info('=== Arm Pose Calculation Complete ===')
    node.get_logger().info('Joint angles calculated for 12cm reach.')
    node.get_logger().info('')
    node.get_logger().info('NOTE: Setting link poses directly may cause physics issues.')
    node.get_logger().info('For reliable joint positioning, enable controllers:')
    node.get_logger().info('  ros2 launch ... static_scene_only:=false enable_controllers:=true')
    node.get_logger().info('Then use joint trajectory commands to move to target pose.')
    node.get_logger().info('')
    node.get_logger().info(f'Robot1 joint angles: {joint_angles}')
    node.get_logger().info(f'Robot2 joint angles: {joint_angles} (same, but yaw=π)')
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
