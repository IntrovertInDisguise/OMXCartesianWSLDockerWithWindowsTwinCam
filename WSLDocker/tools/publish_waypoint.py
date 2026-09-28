#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import PoseStamped
import sys, time

def main():
    delay = float(sys.argv[1]) if len(sys.argv)>1 else 8.0
    frame = sys.argv[2] if len(sys.argv)>2 else 'offset'
    x = float(sys.argv[3]) if len(sys.argv)>3 else 0.03
    y = float(sys.argv[4]) if len(sys.argv)>4 else 0.0
    z = float(sys.argv[5]) if len(sys.argv)>5 else 0.0
    topic = '/omx/variable_stiffness_controller/waypoint_command'
    time.sleep(delay)
    rclpy.init()
    node = rclpy.create_node('wp_pub')
    pub = node.create_publisher(PoseStamped, topic, 10)
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0
    # publish a few times to ensure delivery
    for _ in range(3):
        pub.publish(msg)
        node.get_logger().info(f'Published waypoint to {topic} frame={frame} pos=({x},{y},{z})')
        time.sleep(0.1)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
