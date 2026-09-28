#!/usr/bin/env bash
set -euo pipefail

# Create a reproducible virtual environment at .venv and install dev deps
# Usage: ./scripts/setup_python_env.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  # Create venv that can see system site-packages (ROS python packages)
  python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Activate and upgrade pip
source "$VENV_DIR/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel

if [ -f "$ROOT_DIR/requirements-dev.txt" ]; then
  python3 -m pip install -r "$ROOT_DIR/requirements-dev.txt"
fi

echo "Virtualenv prepared at $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate" 

# Ensure ROS python packages (e.g. rclpy) are visible inside the venv
source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
ROS_PKG_DIR="$(/usr/bin/python3 -c 'import importlib.util, os; spec = importlib.util.find_spec("rclpy"); print(os.path.dirname(spec.origin) if spec else "")' 2>/dev/null || true)"
if [ -n "$ROS_PKG_DIR" ]; then
  ROS_ROOT_DIR="$(dirname "$ROS_PKG_DIR")"
  VENV_SITEPKG="$VENV_DIR/bin/python -c 'import site,sys; print(site.getsitepackages()[0])'"
  # Resolve venv site-packages path robustly
  VENV_SITEPKG="$($VENV_DIR/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  if [ -d "$VENV_SITEPKG" ]; then
    # Add discovered ROS package root (e.g. .../dist-packages/rclpy)
    echo "$ROS_ROOT_DIR" > "$VENV_SITEPKG/ros.pth" || true
    # Also add common ROS site-package locations (local and global)
    PYDIR="$(/usr/bin/python3 -c 'import sys; print("python%d.%d"%sys.version_info[:2])')"
    echo "/opt/ros/humble/local/lib/${PYDIR}/dist-packages" >> "$VENV_SITEPKG/ros.pth" || true
    echo "/opt/ros/humble/lib/${PYDIR}/site-packages" >> "$VENV_SITEPKG/ros.pth" || true
    echo "Added ROS python package roots to $VENV_SITEPKG/ros.pth"
  fi
fi

# Ensure activating the venv also sources the ROS environment for runtime
ACTIVATE_FILE="$VENV_DIR/bin/activate"
if [ -f "$ACTIVATE_FILE" ] && ! grep -q "source /opt/ros/humble/setup.bash" "$ACTIVATE_FILE"; then
  cat >> "$ACTIVATE_FILE" <<'ACT'
# Source ROS environment when activating this virtualenv
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
fi
ACT
  echo "Patched $ACTIVATE_FILE to source ROS on activation"
fi
