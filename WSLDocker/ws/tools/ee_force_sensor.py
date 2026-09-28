#!/usr/bin/env python3
"""
EE Contact Force Sensor — live force vector publisher
======================================================
Subscribes to the variable stiffness controller's deflection-based contact
wrench estimate and republishes a clean xyz force vector suitable for
closed-loop force feedback.

Published topics
  ~/ee_force            geometry_msgs/Vector3Stamped   (filtered xyz, root frame)
  ~/ee_force_magnitude  std_msgs/Float64               (scalar |F| for thresholding)

Subscribed topics  (auto-resolved from --namespace / --controller)
  <ctrl>/contact_wrench   geometry_msgs/WrenchStamped
  <ctrl>/contact_valid    std_msgs/Bool

Usage examples
  # Single hardware (namespace /omx)
  python3 ee_force_sensor.py

  # With custom namespace / controller
  python3 ee_force_sensor.py -n /robot1 -c robot1_variable_stiffness

  # Lower rate, higher deadzone
  python3 ee_force_sensor.py --rate 20 --deadzone 0.5

  # Log to CSV
  python3 ee_force_sensor.py --csv /tmp/ee_force_log.csv
"""
import argparse
import csv
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped, WrenchStamped
from std_msgs.msg import Bool, Float64


class EEForceSensor(Node):
    def __init__(self, args):
        super().__init__('ee_force_sensor')

        # ── resolve topic names ──────────────────────────────────────────
        ns = args.namespace.rstrip('/')
        ctrl = args.controller
        base = f'{ns}/{ctrl}' if ns else ctrl
        wrench_topic = f'{base}/contact_wrench'
        valid_topic  = f'{base}/contact_valid'

        self.get_logger().info(f'Subscribing: {wrench_topic}, {valid_topic}')

        # ── tunables ─────────────────────────────────────────────────────
        self.rate_hz    = max(1.0, args.rate)
        self.deadzone   = max(0.0, args.deadzone)
        self.csv_path   = args.csv

        # ── state ────────────────────────────────────────────────────────
        self._valid      = False
        self._force_xyz  = [0.0, 0.0, 0.0]
        self._frame_id   = ''
        self._stamp      = self.get_clock().now().to_msg()
        self._got_wrench = False

        # ── subscribers ──────────────────────────────────────────────────
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_cb, 10)
        self.create_subscription(Bool, valid_topic, self._valid_cb, 10)

        # ── publishers ───────────────────────────────────────────────────
        self.force_pub = self.create_publisher(Vector3Stamped, '~/ee_force', 10)
        self.mag_pub   = self.create_publisher(Float64, '~/ee_force_magnitude', 10)

        # ── timer at requested rate ──────────────────────────────────────
        period = 1.0 / self.rate_hz
        self.create_timer(period, self._publish_tick)

        # ── optional CSV logging ─────────────────────────────────────────
        self._csv_file   = None
        self._csv_writer = None
        if self.csv_path:
            write_header = not os.path.exists(self.csv_path)
            self._csv_file = open(self.csv_path, 'a', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            if write_header:
                self._csv_writer.writerow([
                    'stamp_sec', 'stamp_nsec', 'valid',
                    'fx', 'fy', 'fz', 'magnitude',
                ])
            self.get_logger().info(f'Logging to {self.csv_path}')

    # ── callbacks ────────────────────────────────────────────────────────
    def _wrench_cb(self, msg: WrenchStamped):
        self._force_xyz = [
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        ]
        self._frame_id = msg.header.frame_id
        self._stamp    = msg.header.stamp
        self._got_wrench = True

    def _valid_cb(self, msg: Bool):
        self._valid = msg.data

    # ── publish at fixed rate ────────────────────────────────────────────
    def _publish_tick(self):
        if not self._got_wrench:
            return  # nothing received yet

        fx, fy, fz = self._force_xyz
        mag = math.sqrt(fx * fx + fy * fy + fz * fz)

        # Apply deadzone — report zero if below threshold
        if mag < self.deadzone:
            fx, fy, fz, mag = 0.0, 0.0, 0.0, 0.0

        # If controller says invalid (singularity escape), zero out
        if not self._valid:
            fx, fy, fz, mag = 0.0, 0.0, 0.0, 0.0

        # Vector3Stamped
        v = Vector3Stamped()
        v.header.stamp    = self._stamp
        v.header.frame_id = self._frame_id
        v.vector.x = fx
        v.vector.y = fy
        v.vector.z = fz
        self.force_pub.publish(v)

        # Scalar magnitude
        m = Float64()
        m.data = mag
        self.mag_pub.publish(m)

        # CSV row
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                self._stamp.sec, self._stamp.nanosec,
                int(self._valid), fx, fy, fz, mag,
            ])

    # ── cleanup ──────────────────────────────────────────────────────────
    def destroy_node(self):
        if self._csv_file is not None:
            self._csv_file.close()
        super().destroy_node()


def build_parser():
    p = argparse.ArgumentParser(description='EE contact force publisher')
    p.add_argument('-n', '--namespace', default='/omx',
                   help='Robot namespace  (default: /omx)')
    p.add_argument('-c', '--controller', default='variable_stiffness_controller',
                   help='Controller name  (default: variable_stiffness_controller)')
    p.add_argument('-r', '--rate', type=float, default=50.0,
                   help='Publish rate Hz  (default: 50)')
    p.add_argument('-d', '--deadzone', type=float, default=0.1,
                   help='Min force magnitude [N] below which output is zero (default: 0.1)')
    p.add_argument('--csv', type=str, default=None,
                   help='Path to append CSV log  (default: none)')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    rclpy.init()
    node = EEForceSensor(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
