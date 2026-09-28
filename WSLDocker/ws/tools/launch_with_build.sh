#!/usr/bin/env bash
set -euo pipefail

# launch_with_build.sh
# Wrapper to optionally build and always source the workspace before
# executing `ros2 launch` so launches run with the workspace environment.

WS="/workspaces/omx_ros2/ws"
INSTALL_SETUP="$WS/install/setup.bash"

BUILD=false
ARGS=()

# Workaround: some generated setup.bash reference COLCON_TRACE without
# a default; ensure it's defined to avoid 'unbound variable' when
# `set -u` is in effect.
export COLCON_TRACE=${COLCON_TRACE:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD=true
      shift
      ;;
    --no-build)
      BUILD=false
      shift
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$BUILD" == "true" || ! -f "$INSTALL_SETUP" ]]; then
  echo "[launch_with_build] Building workspace at $WS"
  (cd "$WS" && colcon build)
fi

if [[ -f "$INSTALL_SETUP" ]]; then
  # Some setup scripts reference variables without defaults. Temporarily
  # disable 'set -u' (nounset) while sourcing to avoid aborting on
  # unbound variables, then restore nounset behavior.
  set +u
  # shellcheck disable=SC1090
  source "$INSTALL_SETUP"
  set -u
  echo "[launch_with_build] Sourced $INSTALL_SETUP"
else
  echo "[launch_with_build] Warning: $INSTALL_SETUP not found; attempting local_setup.bash"
  if [[ -f "$WS/local_setup.bash" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$WS/local_setup.bash"
    set -u
    echo "[launch_with_build] Sourced $WS/local_setup.bash"
  else
    echo "[launch_with_build] No setup file found; proceeding without sourced workspace environment"
  fi
fi

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 [--build|--no-build] -- <ros2 launch args...>"
  exit 2
fi

echo "[launch_with_build] Exec: ros2 launch ${ARGS[*]}"
exec ros2 launch "${ARGS[@]}"
