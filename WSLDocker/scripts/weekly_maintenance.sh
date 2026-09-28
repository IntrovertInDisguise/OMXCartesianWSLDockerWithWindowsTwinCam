#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETENTION_DAYS="${OMX_MAINTENANCE_RETENTION_DAYS:-7}"
WIPE_BUILD=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/weekly_maintenance.sh [--wipe-build] [--retention-days N] [--dry-run]

  --wipe-build        Remove ws/build, ws/install, and ws/log contents after log pruning.
  --retention-days N  Delete log/cache files older than N days. Default: 7.
  --dry-run           Print what would be removed without deleting it.

This script is intended for scheduled maintenance. The blunt cleanup path in
scripts/cleanup_workspace.sh remains available for manual deep wipes.
EOF
}

run_find_delete() {
  local target_dir="$1"
  shift
  if [[ ! -d "$target_dir" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    find "$target_dir" "$@" -print
  else
    find "$target_dir" "$@" -delete
  fi
}

remove_dir_contents() {
  local target_dir="$1"
  if [[ ! -d "$target_dir" ]]; then
    return 0
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    find "$target_dir" -mindepth 1 -maxdepth 1 -print
  else
    find "$target_dir" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wipe-build)
      WIPE_BUILD=1
      shift
      ;;
    --retention-days)
      RETENTION_DAYS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[weekly-maintenance] repo=$REPO_ROOT retention_days=$RETENTION_DAYS wipe_build=$WIPE_BUILD dry_run=$DRY_RUN"

run_find_delete "$REPO_ROOT/logs" -type f -mtime +"$RETENTION_DAYS"
run_find_delete "$REPO_ROOT/ws/log" -type f -mtime +"$RETENTION_DAYS"
run_find_delete "$HOME/.ros/log" -type f -mtime +"$RETENTION_DAYS"
run_find_delete "$HOME/.gazebo" -type f -mtime +"$RETENTION_DAYS"
run_find_delete "$HOME/.cache/gazebo" -type f -mtime +"$RETENTION_DAYS"

if [[ -d /tmp ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    find /tmp -maxdepth 1 -type f \( -name '*gazebo*' -o -name '*dual_gazebo*' -o -name '*ros2*' \) -mtime +"$RETENTION_DAYS" -print
  else
    find /tmp -maxdepth 1 -type f \( -name '*gazebo*' -o -name '*dual_gazebo*' -o -name '*ros2*' \) -mtime +"$RETENTION_DAYS" -delete
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  find "$REPO_ROOT" -type d -name '__pycache__' -print
  find "$REPO_ROOT" -type f -name '*.pyc' -print
else
  find "$REPO_ROOT" -type d -name '__pycache__' -exec rm -rf {} +
  find "$REPO_ROOT" -type f -name '*.pyc' -delete
fi

if [[ "$WIPE_BUILD" -eq 1 ]]; then
  if pgrep -f 'colcon|ros2 launch|gzserver|gzclient|rviz2' >/dev/null 2>&1; then
    echo "[weekly-maintenance] active build or ROS/Gazebo processes detected; skipping ws/build, ws/install, and ws/log wipe"
  else
    remove_dir_contents "$REPO_ROOT/ws/build"
    remove_dir_contents "$REPO_ROOT/ws/install"
    remove_dir_contents "$REPO_ROOT/ws/log"
  fi
fi

echo "[weekly-maintenance] complete"