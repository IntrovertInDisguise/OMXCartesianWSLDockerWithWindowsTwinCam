#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
import os
import time

OUT_CSV = '/tmp/omx_deviated_waypoint.csv'

class Listener(Node):
    def __init__(self):
        super().__init__('deviated_listener')
        self.sub = self.create_subscription(JointState,
                                           '/omx/variable_stiffness_controller/deviated_waypoint',
                                           self.cb, 10)
        self.received = False

    def cb(self, msg: JointState):
        if self.received:
            return
        self.received = True
        ts = self.get_clock().now().to_msg().sec
        # Ensure header row
        write_header = not os.path.exists(OUT_CSV)
        with open(OUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['timestamp', 'joint_names', 'positions'])
            writer.writerow([ts, ' '.join(msg.name), ' '.join([str(p) for p in msg.position])])
        self.get_logger().info(f'Wrote deviated waypoint to {OUT_CSV}')
        # small delay to ensure file flushed
        time.sleep(0.1)
        rclpy.shutdown()


def main():
    rclpy.init()
    node = Listener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.received:
            node.get_logger().info('No deviated waypoint received before shutdown')
        node.destroy_node()

if __name__ == '__main__':
    main()
