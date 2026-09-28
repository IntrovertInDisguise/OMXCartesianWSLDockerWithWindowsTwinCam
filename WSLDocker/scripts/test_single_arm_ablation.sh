#!/usr/bin/env bash
# scripts/test_single_arm_ablation.sh
# ──────────────────────────────────────────────────────────────────────────────
# Isolated test runner for the single-arm ablation harness unit tests.
# Runs without ROS2 or hardware — plain pytest, < 1s.
#
# Usage:
#   bash scripts/test_single_arm_ablation.sh
# ──────────────────────────────────────────────────────────────────────────────
set -e
cd /workspaces/omx_ros2

echo "========================================================================"
echo "Running single-arm ablation unit tests"
echo "========================================================================"

python3 -m pytest tools/test_single_arm_ablation_unit.py -v "$@"

echo ""
echo "All single-arm ablation unit tests passed."
