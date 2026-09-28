#!/usr/bin/env python3
"""
Live ArUco marker viewer for single-arm setup.

Runs the visualizer and opens RViz to display the annotated camera feed.
"""

import subprocess
import sys
import time
import os
import signal

def main():
    # Source ROS2
    ros_setup = "/opt/ros/humble/setup.bash"
    ws_setup = "/workspaces/omx_ros2/ws/install/setup.bash"
    
    env = os.environ.copy()
    env["BASH_ENV"] = ros_setup
    
    # Start visualizer in background
    print("Starting ArUco visualizer...")
    visualizer_cmd = [
        "python3", "/workspaces/omx_ros2/tools/single_arm_aruco_visual.py",
        "--color-topic", "/camera/camera/color/image_raw",
        "--camera-info-topic", "/camera/camera/color/camera_info",
    ]
    
    visualizer_proc = subprocess.Popen(
        visualizer_cmd,
        env=env,
        preexec_fn=os.setsid
    )
    
    # Wait for visualizer to start
    time.sleep(2)
    
    # Start RViz with overlay topic
    print("Starting RViz with live overlay...")
    
    # Create RViz config
    rviz_config = """
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Image
      Name: ArUco Overlay
      Topic:
        Value: /single_arm_aruco/overlay
    - Class: rviz_default_plugins/Image
      Name: Raw Camera
      Topic:
        Value: /camera/camera/color/image_raw
  Global Options:
    Fixed Frame: camera_color_optical_frame
"""
    
    config_path = "/tmp/aruco_live.rviz"
    with open(config_path, "w") as f:
        f.write(rviz_config)
    
    rviz_cmd = ["rviz2", "-d", config_path]
    rviz_proc = subprocess.Popen(rviz_cmd, env=env)
    
    print("\n" + "="*60)
    print("Live ArUco viewer running!")
    print("="*60)
    print(f"Visualizer PID: {visualizer_proc.pid}")
    print(f"RViz PID: {rviz_proc.pid}")
    print("\nRViz should show:")
    print("  - Top: Annotated camera feed with detected markers")
    print("  - Bottom: Raw camera feed")
    print("\nPress Ctrl+C to stop both processes.")
    print("="*60 + "\n")
    
    # Wait for Ctrl+C
    try:
        while True:
            # Check if processes are still running
            if visualizer_proc.poll() is not None:
                print("Visualizer stopped")
                break
            if rviz_proc.poll() is not None:
                print("RViz stopped, restarting...")
                rviz_proc = subprocess.Popen(rviz_cmd, env=env)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    # Cleanup
    try:
        os.killpg(os.getpgid(visualizer_proc.pid), signal.SIGTERM)
    except:
        pass
    try:
        os.killpg(os.getpgid(rviz_proc.pid), signal.SIGTERM)
    except:
        pass
    
    visualizer_proc.wait(timeout=5)
    rviz_proc.wait(timeout=5)
    print("Stopped.")

if __name__ == "__main__":
    main()
