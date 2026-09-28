#!/bin/bash
# Run launch file unit tests in isolation
# These tests verify launch file structure without launching any processes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Launch File Unit Tests ==="
echo "Testing launch file structure and validity..."
echo ""

cd "$WORKSPACE_DIR"

# Source ROS2 environment
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "ERROR: ROS2 Humble not found at /opt/ros/humble"
    exit 1
fi

# Source workspace
if [ -f install/setup.bash ]; then
    source install/setup.bash
else
    echo "ERROR: Workspace not built. Run 'colcon build' first."
    exit 1
fi

# Run the unit tests
echo "Running launch file unit tests..."
python3 -m pytest \
    ws/src/omx_variable_stiffness_controller/test/test_launch_files_unit.py \
    -v \
    --tb=short

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✓ All launch file unit tests passed!"
else
    echo "✗ Some tests failed. See output above for details."
fi

exit $exit_code
