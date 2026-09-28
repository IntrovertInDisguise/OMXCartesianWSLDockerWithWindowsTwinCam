#!/bin/bash
# run_udp_aruco_receiver.sh
# ──────────────────────────────────────────────────────────────────────────────
# Launch the WSL/Docker-side UDP ArUco receiver (udp_aruco_ros2_receiver.py) and
# print the exact Windows-side command to point the sender at THIS machine.
#
# WHY THE IP ALWAYS SEEMS TO MISMATCH
# ────────────────────────────────────
# The Windows sender and the WSL receiver do NOT live on the same network
# interface, even though they share one physical PC:
#
#   • Windows itself          → 127.0.0.1 (loopback) and a real LAN IP, e.g. 192.168.1.42
#   • WSL2 / Docker desktop    → its OWN virtual NIC with a SEPARATE IP, e.g. 172.28.123.45
#                                (run `ip addr show eth0` inside WSL to see it)
#
# So if the Windows sender targets 127.0.0.1, the packet is delivered to
# Windows' own loopback and the WSL receiver never sees it → "no packets".
# The sender MUST target the WSL IP, and the receiver must bind to that IP
# (or 0.0.0.0, which means "all interfaces" and is what the receiver does by
# default).
#
# IMPORTANT: a WSL2 IP is handed out by the WSL virtual switch and CHANGES on
# reboot / network change. Always re-run this script (or `hostname -I`) before
# starting the Windows sender, and never hard-code the IP in the sender.
#
# DOCKER / VS CODE DEV CONTAINER USERS
# ─────────────────────────────────────
# If the receiver runs inside a Docker container (this repo's
# .devcontainer/devcontainer.json), the container's IP (e.g. 172.17.0.2) is on
# Docker's bridge network and Windows CANNOT route to it. Two fixes:
#
#   1. (Recommended) Add to .devcontainer/devcontainer.json runArgs:
#        "--network=host"
#      then rebuild the container (Ctrl+Shift+P → "Dev Containers: Rebuild
#      Container"). The container then shares the host network, so Windows can
#      send straight to the host IP printed by this script — no port mapping.
#
#   2. (Alternative) Publish the port instead of host networking:
#        "forwardPorts": [5005],
#        "portsAttributes": { "5005": { "label": "UDP ArUco bridge", "protocol": "udp" } }
#      then rebuild and point the Windows sender at 127.0.0.1.
#
# This script auto-detects a Docker environment (/.dockerenv) and prints the
# correct connection steps for whichever fix you applied.
#
# FLOW:
#   Windows:  windows_aruco_udp_sender.py --udp-host <HOST_IP> --udp-port 5005
#   WSL/Docker: this script  (binds 0.0.0.0:5005, republishes ROS2 topics)
#
# Usage:
#   bash tools/run_udp_aruco_receiver.sh
#   bash tools/run_udp_aruco_receiver.sh --udp-port 5005
#   bash tools/run_udp_aruco_receiver.sh --print-windows-cmd   # only print the sender command
#   bash tools/run_udp_aruco_receiver.sh --bind-ip 172.28.123.45
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd /workspaces/omx_ros2

# ── Parse args ──────────────────────────────────────────────────────────────
UDP_PORT=5005
BIND_IP="0.0.0.0"
PRINT_WINDOWS_CMD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --udp-port) UDP_PORT="$2"; shift 2 ;;
        --bind-ip)  BIND_IP="$2"; shift 2 ;;
        --print-windows-cmd) PRINT_WINDOWS_CMD_ONLY=true; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Detect the IP the Windows sender must target ────────────────────────────
# Priority:
#   1. First non-loopback, non-docker-bridge IPv4 (the WSL2 eth0 / host IP that
#      Windows can actually route to).
#   2. Otherwise any non-loopback IPv4 (e.g. a Docker bridge) — usable only if
#      Windows can reach it (Docker Desktop port mapping, or the sender runs in
#      the same Docker network). We still print it but warn.
#   3. Otherwise 127.0.0.1 with a strong warning (Windows loopback != WSL/Docker
#      loopback, so this will NOT work cross-OS).
detect_target_ip() {
    local preferred fallback any
    preferred=$(hostname -I 2>/dev/null | tr ' ' '\n' \
        | grep -vE '^127\.|^172\.1[0-9]\.|^172\.2[0-9]\.|^172\.3[0-1]\.' \
        | grep -E '^[0-9]+\.' | head -n1)
    fallback=$(hostname -I 2>/dev/null | tr ' ' '\n' \
        | grep -vE '^127\.' | grep -E '^[0-9]+\.' | head -n1)
    any=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | head -n1)
    if [[ -n "$preferred" ]]; then
        echo "$preferred"
    elif [[ -n "$fallback" ]]; then
        echo "$fallback"
    elif [[ -n "$any" ]]; then
        echo "$any"
    else
        echo "127.0.0.1"
    fi
}

WSL_IP="$(detect_target_ip)"

# Detect whether we are inside a Docker container, and whether it uses the
# default bridge network (IP not routable from Windows) or host networking
# (IP IS routable from Windows — no port mapping needed).
IN_DOCKER=false
if [[ -f /.dockerenv ]] || grep -qa docker /proc/1/cgroup 2>/dev/null; then
    IN_DOCKER=true
fi

# A Docker bridge IP is in 172.17.0.0/16 (default bridge) or 172.18-31.0.0/16
# (user bridges). With --network=host the container shows the host's own
# RFC1918 IPs (10./172.16-31. except the bridge range /192.168.), which Windows
# CAN route to.
is_bridge_ip() {
    [[ "$1" == 172.17.* || "$1" == 172.1[89].* || "$1" == 172.2[0-9].* || "$1" == 172.3[01].* ]]
}

DOCKER_HOST_NET=false
if [[ "$IN_DOCKER" == true && -n "$WSL_IP" ]]; then
    if is_bridge_ip "$WSL_IP"; then
        DOCKER_HOST_NET=false   # bridge: not routable, needs port mapping
    else
        DOCKER_HOST_NET=true    # host networking: routable, use this IP directly
    fi
fi

echo "==================================================================="
echo " UDP ArUco bridge — connection info"
echo "==================================================================="
if [[ -n "$WSL_IP" ]]; then
    echo " WSL/Docker IP (use this on the Windows sender): ${WSL_IP}"
else
    echo " WARNING: could not auto-detect WSL IP. Run 'hostname -I' inside WSL"
    echo "          and pass it with --bind-ip / --udp-host on the sender."
fi
echo " Receiver bind IP : ${BIND_IP}  (0.0.0.0 = all interfaces)"
echo " Receiver UDP port: ${UDP_PORT}"
echo "-------------------------------------------------------------------"
echo "==================================================================="

if [[ "$DOCKER_HOST_NET" == true ]]; then
    # Docker with --network=host: the container shares the host network, so the
    # printed IP IS reachable from Windows. Point the sender straight at it.
    echo " ENVIRONMENT: Docker container with --network=host (host IP ${WSL_IP}"
    echo "              is routable from Windows — no port mapping needed)."
    echo "-------------------------------------------------------------------"
    echo " WINDOWS SENDER COMMAND (run natively on Windows):"
    echo "   python windows_aruco_udp_sender.py --realsense \\"
    echo "       --udp-host ${WSL_IP} --udp-port ${UDP_PORT} --show"
    echo "-------------------------------------------------------------------"
    echo " NOTE: this IP is handed out by the host network and may change on"
    echo "       reboot/network change. Re-run this script before each session."
elif [[ "$IN_DOCKER" == true ]]; then
    # Inside Docker on the default bridge (or Docker Desktop where --network=host
    # is NOT truly shared with Windows): the container IP (e.g. 172.17.0.2) is
    # NOT reachable from Windows. The sender must target the HOST LOOPBACK and
    # the container must have the UDP port PUBLISHED/MAPPED to the host.
    echo " ENVIRONMENT: Docker container (IP ${WSL_IP} is a bridge addr Windows"
    echo "              CANNOT route to — do NOT point the sender at it)."
    echo "-------------------------------------------------------------------"
    echo " FIX — publish the UDP port, then target the Windows host:"
    echo "   # VS Code dev container (.devcontainer/devcontainer.json) — use --publish,"
    echo "   # NOT forwardPorts (forwardPorts does NOT forward UDP):"
    echo "   \"runArgs\": [ \"--publish=${UDP_PORT}:${UDP_PORT}/udp\", ... ]"
    echo "   # then rebuild: Ctrl+Shift+P -> 'Dev Containers: Rebuild Container'"
    echo "   # Verify from the host: docker port <id> ${UDP_PORT}/udp"
    echo "   #   (should show '${UDP_PORT}/udp -> 0.0.0.0:${UDP_PORT}')"
    echo "   # Manual docker run:"
    echo "   docker run -p ${UDP_PORT}:${UDP_PORT}/udp ... <image>"
    echo ""
    echo " STEP — point the WINDOWS sender at the HOST LOOPBACK (Docker Desktop"
    echo "         forwards localhost into the container):"
    echo "   python windows_aruco_udp_sender.py --realsense \\"
    echo "       --udp-host 127.0.0.1 --udp-port ${UDP_PORT} --show"
    echo "   # or use host.docker.internal (Docker Desktop):"
    echo "   python windows_aruco_udp_sender.py --realsense \\"
    echo "       --udp-host host.docker.internal --udp-port ${UDP_PORT} --show"
    echo "-------------------------------------------------------------------"
    echo " NOTE: this receiver binds 0.0.0.0:${UDP_PORT} and will receive once the"
    echo "       host port is mapped/forwarded. No code change needed on the receiver side."
else
    # Bare WSL2 (or native Linux): use the WSL eth0 IP that Windows can route to.
    echo " WINDOWS SENDER COMMAND (run natively on Windows):"
    echo "   python windows_aruco_udp_sender.py --realsense \\"
    echo "       --udp-host ${WSL_IP:-<WSL_IP>} --udp-port ${UDP_PORT} --show"
    echo "==================================================================="
fi

if [[ "$PRINT_WINDOWS_CMD_ONLY" == true ]]; then
    exit 0
fi

# ── Sanity checks before launching ──────────────────────────────────────────
if [[ -z "${ROS_DISTRO:-}" ]]; then
    echo "[info] ROS_DISTRO not set — sourcing /opt/ros/humble/setup.bash"
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

RECEIVER_SCRIPT="tools/udp_aruco_ros2_receiver.py"
if [[ ! -f "$RECEIVER_SCRIPT" ]]; then
    echo "ERROR: receiver script not found: $RECEIVER_SCRIPT" >&2
    exit 1
fi

# Warn if the detected IP looks like a Docker bridge that Windows can't reach.
# (Skip when host networking is active — that case is already handled above.)
if [[ "$DOCKER_HOST_NET" != true && "$IN_DOCKER" != true && ("$WSL_IP" == 172.1[0-9].* || "$WSL_IP" == 172.2[0-9].* || "$WSL_IP" == 172.3[0-1].*) ]]; then
    echo "WARNING: detected IP ${WSL_IP} looks like a Docker bridge. Windows"
    echo "         cannot route to it directly. Either:"
    echo "           - run the sender INSIDE the same Docker network, or"
    echo "           - use Docker Desktop port mapping (publish UDP ${UDP_PORT}), or"
    echo "           - on real WSL2, use the WSL eth0 IP from 'hostname -I'"
    echo "             (the one that is NOT 127.x and NOT 172.1x/172.2x/172.3x.x)."
elif [[ "$DOCKER_HOST_NET" != true && "$IN_DOCKER" != true && "$WSL_IP" == 127.* ]]; then
    echo "WARNING: only loopback (${WSL_IP}) is available. Windows loopback is NOT"
    echo "         the same as WSL/Docker loopback, so the sender will not reach this"
    echo "         receiver. Use the WSL eth0 / Docker host IP instead."
fi

echo "[launch] Starting UDP ArUco receiver on ${BIND_IP}:${UDP_PORT} ..."
echo "[launch] Ctrl-C to stop."
echo ""

exec python3 "$RECEIVER_SCRIPT" \
    --ros-args \
    -p bind_ip:="${BIND_IP}" \
    -p udp_port:="${UDP_PORT}"
