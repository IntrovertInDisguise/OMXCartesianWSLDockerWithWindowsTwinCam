#!/bin/bash
# setup_vnc_display.sh — Start Xvfb + x11vnc + noVNC for Gazebo GUI in WSLg/containers
#
# Usage:
#   source tools/setup_vnc_display.sh
#   # Then export DISPLAY=:99 before launching Gazebo
#
# After running, view the Gazebo GUI in your browser at:
#   http://localhost:6080/vnc.html
#
# If Xvfb/VNC setup fails, falls back to native DISPLAY=:0 (WSLg X11).

DISPLAY_NUM="${VNC_DISPLAY_NUM:-99}"
VNC_PORT="${VNC_PORT:-5900}"
WEB_PORT="${WEB_PORT:-6080}"

_vnc_fallback() {
    echo "VNC fallback: using native X11 display :0 (WSLg)"
    export DISPLAY=:0
    return 0
}

require_cmd() {
    local cmd="$1"
    local pkg_hint="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "WARNING: Required command '$cmd' is not installed."
        echo "  Install with: apt-get update && apt-get install -y $pkg_hint"
        return 1
    fi
}

# Check all required commands; fall back to native display if any are missing.
_missing=0
require_cmd Xvfb xvfb || _missing=1
require_cmd x11vnc x11vnc || _missing=1
require_cmd websockify websockify || _missing=1

if [ "$_missing" -eq 1 ]; then
    echo "Falling back to native DISPLAY=:0 (missing VNC dependencies)"
    _vnc_fallback
    return 0 2>/dev/null || exit 0
fi

# Cleanup existing sessions
pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "x11vnc.*:${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "websockify.*${WEB_PORT}" 2>/dev/null || true
sleep 1
rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}" 2>/dev/null || true

# Fix X11 socket permissions
chmod 1777 /tmp/.X11-unix/ 2>/dev/null || true

# 1. Start Xvfb (virtual framebuffer with GLX)
echo "Starting Xvfb on display :${DISPLAY_NUM} (1280x720x24)..."
Xvfb ":${DISPLAY_NUM}" -screen 0 1280x720x24 -ac +extension GLX +render -noreset > /tmp/xvfb.log 2>&1 &
XVFB_PID=$!
sleep 2

if ! kill -0 "$XVFB_PID" 2>/dev/null; then
    echo "WARNING: Xvfb failed to start (see /tmp/xvfb.log)"
    echo "Falling back to native DISPLAY=:0"
    _vnc_fallback
    return 0 2>/dev/null || exit 0
fi
echo "  Xvfb started (PID: $XVFB_PID)"

# 2. Start x11vnc (VNC server sharing the virtual display)
echo "Starting x11vnc on port ${VNC_PORT}..."
unset WAYLAND_DISPLAY
DISPLAY=":${DISPLAY_NUM}" x11vnc -display ":${DISPLAY_NUM}" -nopw -forever -xkb -shared -rfbport "${VNC_PORT}" -noxdamage > /tmp/x11vnc.log 2>&1 &
VNC_PID=$!
sleep 2

if ! kill -0 "$VNC_PID" 2>/dev/null; then
    echo "WARNING: x11vnc failed to start (see /tmp/x11vnc.log)"
    echo "Falling back to native DISPLAY=:0"
    _vnc_fallback
    return 0 2>/dev/null || exit 0
fi
echo "  x11vnc started (PID: $VNC_PID)"

# 3. Start noVNC/websockify (browser-based VNC viewer)
echo "Starting noVNC on port ${WEB_PORT}..."
websockify --web /usr/share/novnc "${WEB_PORT}" "localhost:${VNC_PORT}" > /tmp/novnc.log 2>&1 &
NOVNC_PID=$!
sleep 1

if ! kill -0 "$NOVNC_PID" 2>/dev/null; then
    echo "WARNING: noVNC failed to start (see /tmp/novnc.log)"
    echo "Falling back to native DISPLAY=:0"
    _vnc_fallback
    return 0 2>/dev/null || exit 0
fi
echo "  noVNC started (PID: $NOVNC_PID)"

# Export for Gazebo
export DISPLAY=":${DISPLAY_NUM}"
export LIBGL_ALWAYS_SOFTWARE=1

echo ""
echo "=========================================="
echo "  VNC pipeline is UP!"
echo "  Gazebo GUI: http://localhost:${WEB_PORT}/vnc.html"
echo "  DISPLAY=:${DISPLAY_NUM}"
echo "=========================================="
