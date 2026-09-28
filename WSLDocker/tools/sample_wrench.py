#!/usr/bin/env python3
import subprocess, re, math, time, os

def run_ros(cmd, timeout=10):
    full = "bash -lc 'source /opt/ros/humble/setup.bash && source /workspaces/omx_ros2/ws/install/setup.bash && %s'" % cmd
    return subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)


def sample(topic, attempts=8, delay=0.25):
    peak_f = 0.0
    peak_t = 0.0
    for i in range(attempts):
        p = run_ros(f"ros2 topic echo {topic} --once")
        s = p.stdout
        if not s:
            time.sleep(delay)
            continue
        fm = re.search(r"force:\s*\n\s*x:\s*([-+0-9.eE]+)\s*\n\s*y:\s*([-+0-9.eE]+)\s*\n\s*z:\s*([-+0-9.eE]+)", s)
        tm = re.search(r"torque:\s*\n\s*x:\s*([-+0-9.eE]+)\s*\n\s*y:\s*([-+0-9.eE]+)\s*\n\s*z:\s*([-+0-9.eE]+)", s)
        if not fm:
            fm = re.search(r"force:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]", s)
        if not tm:
            tm = re.search(r"torque:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]", s)
        if fm:
            try:
                fx, fy, fz = map(float, fm.groups())
                val = math.sqrt(fx*fx + fy*fy + fz*fz)
                peak_f = max(peak_f, val)
            except Exception:
                pass
        if tm:
            try:
                tx, ty, tz = map(float, tm.groups())
                valt = math.sqrt(tx*tx + ty*ty + tz*tz)
                peak_t = max(peak_t, valt)
            except Exception:
                pass
        time.sleep(delay)
    return peak_f, peak_t


if __name__ == '__main__':
    # publish waypoint presses
    for _ in range(3):
        run_ros("ros2 topic pub -1 /robot1/robot1_variable_stiffness/waypoint_command geometry_msgs/msg/PoseStamped \"{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, pose: {position: {x: 0.5, y: 0.0, z: 0.15}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}\"")
        run_ros("ros2 topic pub -1 /robot2/robot2_variable_stiffness/waypoint_command geometry_msgs/msg/PoseStamped \"{header: {stamp: {sec: 0, nanosec: 0}, frame_id: ''}, pose: {position: {x: 0.5, y: 0.0, z: 0.15}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}\"")
        time.sleep(0.25)

    p1f, p1t = sample('/robot1/robot1_variable_stiffness/contact_wrench')
    p2f, p2t = sample('/robot2/robot2_variable_stiffness/contact_wrench')
    print('robot1 peak force:', p1f, 'peak torque:', p1t)
    print('robot2 peak force:', p2f, 'peak torque:', p2t)
