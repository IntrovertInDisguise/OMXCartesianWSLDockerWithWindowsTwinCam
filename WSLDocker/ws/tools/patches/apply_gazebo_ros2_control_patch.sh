#!/bin/bash
# ============================================================
#  apply_gazebo_ros2_control_patch.sh
#  Patches libgazebo_ros2_control.so to fix multi-robot namespace
#  poisoning in gazebo_ros2_control v0.4.10 (ROS 2 Humble).
#
#  BUG: The plugin's Load() adds __ns:=/robotN to global rcl_context,
#       causing ALL subsequent plugin instances in the same gzserver
#       process to inherit the first plugin's namespace.
#  FIX: Remove __ns from global arguments. The namespace is already
#       handled by gazebo_ros::Node::Get(sdf) and passed explicitly
#       to the ControllerManager constructor.
#
#  Usage: bash tools/patches/apply_gazebo_ros2_control_patch.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/fix_gazebo_ros2_control_multi_robot.patch"

echo "[1/5] Downloading gazebo_ros2_control source..."
cd /tmp
apt-get source ros-humble-gazebo-ros2-control --download-only -q 2>/dev/null
tar xf ros-humble-gazebo-ros2-control_*.orig.tar.gz

SRC_DIR=$(ls -d /tmp/ros-humble-gazebo-ros2-control-*/  | head -1)
echo "[2/5] Applying patch to $SRC_DIR"
cd "$SRC_DIR"

# Apply the patch (simple sed since patch context may not match exactly)
python3 -c "
fpath = 'src/gazebo_ros2_control_plugin.cpp'
with open(fpath) as f:
    c = f.read()
old = '''    // Set namespace if tag is present
    if (sdf->HasElement(\"namespace\")) {
      std::string ns = sdf->GetElement(\"namespace\")->Get<std::string>();
      // prevent exception: namespace must be absolute, it must lead with a '/'
      if (ns.empty() || ns[0] != '/') {
        ns = '/' + ns;
      }
      std::string ns_arg = std::string(\"__ns:=\") + ns;
      arguments.push_back(RCL_REMAP_FLAG);
      arguments.push_back(ns_arg);
    }'''
new = '''    // MULTI-ROBOT FIX: Do NOT add __ns to global rcl arguments.
    // The namespace is already correctly handled by gazebo_ros::Node::Get(sdf)
    // and passed explicitly to ControllerManager. Adding __ns here poisons the
    // global rcl_context, causing subsequent plugin instances in the same
    // gzserver process to inherit the first plugin's namespace.'''
assert old in c, 'Patch target not found — source version may differ'
c = c.replace(old, new, 1)
with open(fpath, 'w') as f:
    f.write(c)
print('  Source patched.')
"

echo "[3/5] Configuring build..."
source /opt/ros/humble/setup.bash
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/ros/humble -DCMAKE_BUILD_TYPE=Release \
    2>&1 | tail -2

echo "[4/5] Building..."
make -j$(nproc) 2>&1 | tail -3

echo "[5/5] Installing patched library..."
cp libgazebo_ros2_control.so /opt/ros/humble/lib/libgazebo_ros2_control.so

echo ""
echo "✓ Patched libgazebo_ros2_control.so installed to /opt/ros/humble/lib/"
echo "  Multi-robot Gazebo simulation should now work correctly."

# Cleanup
cd /tmp && rm -rf "$SRC_DIR" ros-humble-gazebo-ros2-control_*
