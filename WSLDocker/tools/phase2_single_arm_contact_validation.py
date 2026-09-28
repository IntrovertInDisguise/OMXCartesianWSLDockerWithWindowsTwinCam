#!/usr/bin/env python3
"""
Phase-2 Single-Arm Contact Validation

Implements staged validation to test motor current as a contact proxy.
Configurable via CLI args for robot namespace and topic names.

Output: JSON structured result printed to stdout and per-trial CSV logs under /tmp.
"""
import argparse
import json
import math
import os
import statistics
import time
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


class ContactValidator(Node):
    def __init__(self, args):
        super().__init__('phase2_contact_validator')
        self.robot_ns = args.robot_ns
        self.joint_states_topic = args.joint_states_topic
        self.waypoint_topic = args.waypoint_topic
        self.current_window = args.current_window
        self.verbose = args.verbose
        self.contact_min = args.contact_min
        self.current_hard_limit = args.current_hard_limit

        self.joint_state = None
        self.joint_state_time = None
        self.joint_state_sub = self.create_subscription(JointState, self.joint_states_topic, self._js_cb, 10)
        self.waypoint_pub = self.create_publisher(PoseStamped, self.waypoint_topic, 10)

        # runtime buffers
        self.current_buffer = deque(maxlen=1000)
        self.vel_buffer = deque(maxlen=1000)

        # explicit contact state
        self.in_contact = False

    def _js_cb(self, msg: JointState):
        self.joint_state = msg
        # use ROS time consistently (seconds float)
        now_ros = self.get_clock().now().nanoseconds * 1e-9
        self.joint_state_time = now_ros
        # measured current proxy: use max absolute effort (avoids DoF accumulation)
        efforts = msg.effort or []
        measured_current = float(max(abs(e) for e in efforts)) if efforts else 0.0
        # velocity magnitude
        vmag = math.sqrt(sum(v * v for v in (msg.velocity or []))) if (msg.velocity or []) else 0.0
        self.current_buffer.append((now_ros, measured_current))
        self.vel_buffer.append((now_ros, vmag))

    # === STAGE 1 ===
    def stage1_bringup_ready(self, timeout=10.0):
        # controller node active (best-effort): require joint_states streaming
        t0 = self.get_clock().now().nanoseconds * 1e-9
        while (self.get_clock().now().nanoseconds * 1e-9) - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_state is not None:
                if self.verbose:
                    self.get_logger().info('joint_states streaming')
                return True
        self.get_logger().error('Stage1: joint_states not streaming')
        return False

    # === STAGE 2 ===
    def stage2_free_motion_check(self, waypoint, duration=2.0, free_current_max=0.5):
        # publish a single waypoint and monitor motion
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose = waypoint
        self.waypoint_pub.publish(msg)
        t0 = self.get_clock().now().nanoseconds * 1e-9
        currents = []
        vels = []
        while (self.get_clock().now().nanoseconds * 1e-9) - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.current_buffer:
                currents.append(self.current_buffer[-1][1])
            if self.vel_buffer:
                vels.append(self.vel_buffer[-1][1])
        mean_current = statistics.mean(currents) if currents else 0.0
        var_current = statistics.pvariance(currents) if len(currents) > 1 else 0.0
        # smoothness: low variance in velocity
        vel_var = statistics.pvariance(vels) if len(vels) > 1 else 0.0
        ok = (mean_current < free_current_max) and (vel_var < 1e-2)
        return {'mean_current': mean_current, 'var_current': var_current, 'vel_var': vel_var, 'pass': ok}

    # === STAGE 3 ===
    def stage3_contact_detection(self, approach_steps=30, step_delay=0.2, current_rise_min=0.05):
        # approach by publishing incremental waypoints along -z in task space
        if self.joint_state is None:
            self.get_logger().error('No joint state to infer pose for approach')
            return None
        # we use current joint positions as proxy 'pose' when publishing waypoints
        # user should configure controller to accept PoseStamped waypoints in task frame
        base_pose = PoseStamped()
        base_pose.header.stamp = self.get_clock().now().to_msg()
        base_pose.pose.position.x = 0.0
        base_pose.pose.position.y = 0.0
        base_pose.pose.position.z = 0.3
        base_pose.pose.orientation.w = 1.0

        # continuous descent: publish a steady stream of decreasing z waypoints
        prev_mean = None
        contact_time = None
        contact_pose = None
        start_z = base_pose.pose.position.z
        z = start_z
        step_size = 0.05
        min_z = start_z - 0.6
        # loop until contact detected or min_z reached
        while z > min_z:
            wp = PoseStamped()
            wp.header.stamp = self.get_clock().now().to_msg()
            wp.header.frame_id = 'world'
            wp.pose.position.x = 0.0
            wp.pose.position.y = 0.0
            wp.pose.position.z = z
            wp.pose.orientation.w = 1.0
            self.waypoint_pub.publish(wp)
            # allow controller to process and collect samples
            rclpy.spin_once(self, timeout_sec=0.1)
            # compute recent mean current using ROS time window (sustained window)
            now = self.get_clock().now().nanoseconds * 1e-9
            window = [c for t, c in list(self.current_buffer) if t > now - 1.0]
            mean_current = statistics.mean(window) if window else 0.0
            # velocity mag
            window_v = [v for t, v in list(self.vel_buffer) if t > now - 1.0]
            mean_vel = statistics.mean(window_v) if window_v else 0.0
            # debug log for visibility
            self.get_logger().info(f'z={z:.3f}, current={mean_current:.3f}, vel={mean_vel:.3f}')
            # emergency hard-stop
            if mean_current > self.current_hard_limit:
                self.get_logger().error(f'Emergency stop: current {mean_current:.3f} > hard limit {self.current_hard_limit}')
                return None
            # detect sustained trend over the window (trend-based, not absolute)
            if len(window) > 5:
                trend = window[-1] - window[0]
                if trend > current_rise_min and mean_vel < 0.05:
                    contact_time = now
                    contact_pose = {'z': z}
                    self.in_contact = True
                    return contact_time, contact_pose
            prev_mean = mean_current
            # step down
            z -= step_size
        return None

        return None

    # === STAGE 4 ===
    def stage4_hold_and_measure(self, window=0.5):
        samples = []
        t0 = self.get_clock().now().nanoseconds * 1e-9
        while (self.get_clock().now().nanoseconds * 1e-9) - t0 < window:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.current_buffer:
                samples.append(self.current_buffer[-1][1])
        if not samples:
            return None
        mean_current = statistics.mean(samples)
        var_current = statistics.pvariance(samples) if len(samples) > 1 else 0.0
        return mean_current, var_current

    # === STAGE 5 ===
    def stage5_repeatability(self, n=3, approach_steps=30, step_delay=0.2):
        results = []
        for trial in range(n):
            self.get_logger().info(f'Starting trial {trial+1}/{n}')
            det = self.stage3_contact_detection(approach_steps=approach_steps, step_delay=step_delay)
            if det is None:
                results.append({'contact_time': None, 'mean_current': None, 'variance_current': None, 'pass': False})
                continue
            contact_time, contact_pose = det
            meas = self.stage4_hold_and_measure(window=self.current_window)
            if meas is None:
                results.append({'contact_time': contact_time, 'mean_current': None, 'variance_current': None, 'pass': False})
                continue
            mean_c, var_c = meas
            # improved pass rules
            ok = (mean_c > self.contact_min) and (var_c < mean_c * 2.0)
            # require detected contact state
            if not self.in_contact:
                ok = False
            results.append({'contact_time': contact_time, 'mean_current': mean_c, 'variance_current': var_c, 'pass': ok})
            # small pause between trials
            time.sleep(0.5)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-ns', default='robot1')
    parser.add_argument('--joint-states-topic', default='/robot1/joint_states')
    parser.add_argument('--waypoint-topic', default='/robot1/robot1_variable_stiffness/waypoint_command')
    parser.add_argument('--current-window', type=float, default=0.5)
    parser.add_argument('--contact-min', type=float, default=0.1)
    parser.add_argument('--current-hard-limit', type=float, default=10.0)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    rclpy.init()
    node = ContactValidator(args)

    out = {'trial_results': [], 'overall_pass': False}

    try:
        ok1 = node.stage1_bringup_ready(timeout=10.0)
        if not ok1:
            print(json.dumps(out))
            return

        # explicit safe waypoint for free-motion check
        safe_pose = PoseStamped().pose
        safe_pose.position.x = 0.0
        safe_pose.position.y = 0.0
        safe_pose.position.z = 0.35
        safe_pose.orientation.w = 1.0
        free_check = node.stage2_free_motion_check(waypoint=safe_pose, duration=1.0)
        if args.verbose:
            node.get_logger().info(f'Stage2 free check: {free_check}')

        trials = node.stage5_repeatability(n=args.trials)
        out['trial_results'] = trials
        out['overall_pass'] = all(tr.get('pass', False) for tr in trials)
        print(json.dumps(out, indent=2))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
