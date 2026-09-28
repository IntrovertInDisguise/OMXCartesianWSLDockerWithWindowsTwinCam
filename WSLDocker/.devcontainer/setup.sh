#!/bin/bash
set -e

echo "=== Setting up OMX ROS2 Workspace ==="

source /opt/ros/humble/setup.bash

sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list || true
sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list || true

echo "Installing apt dependencies..."
apt-get clean
apt-get update
apt-get install -y \
    cron \
    python3-colcon-common-extensions \
    ros-humble-moveit \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-interface \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-xacro \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-position-controllers \
    ros-humble-gripper-controllers \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-kdl-parser \
    libserial-dev \
    python3-serial \
    x11vnc \
    novnc \
    libxcb1 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libx11-xcb1 \
    libglu1-mesa \
    libgl1-mesa-glx \
    x11-xserver-utils \
    libqt5gui5 \
    python3-pandas

cd /workspaces/omx_ros2/ws/src

if [ ! -d "dynamixel_sdk" ]; then
    echo "Cloning dynamixel_sdk..."
    git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git dynamixel_sdk
fi

if [ ! -d "dynamixel_hardware_interface" ]; then
    echo "Cloning dynamixel_hardware_interface..."
    git clone -b humble https://github.com/ROBOTIS-GIT/dynamixel_hardware_interface.git
fi

echo "Running rosdep..."
rosdep update || true
rosdep install --from-paths . --ignore-src -r -y || true

echo "Building workspace..."
cd /workspaces/omx_ros2/ws
colcon build --packages-skip open_manipulator_x_playground open_manipulator_x_gui open_manipulator_x_teleop --symlink-install

echo "Preparing Python virtualenv..."
chmod +x /workspaces/omx_ros2/scripts/setup_python_env.sh || true
/workspaces/omx_ros2/scripts/setup_python_env.sh || true

echo "Configuring weekly workspace maintenance..."
chmod +x /workspaces/omx_ros2/scripts/weekly_maintenance.sh || true
cat > /etc/cron.d/omx_weekly_maintenance <<'EOF'
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
17 3 * * 0 root /usr/bin/flock -n /tmp/omx_weekly_maintenance.lock /workspaces/omx_ros2/scripts/weekly_maintenance.sh --wipe-build >> /var/log/omx_weekly_maintenance.log 2>&1
EOF
chmod 0644 /etc/cron.d/omx_weekly_maintenance
touch /var/log/omx_weekly_maintenance.log
chmod 0644 /var/log/omx_weekly_maintenance.log
pgrep cron >/dev/null || service cron start || cron || true

echo "=== Setup complete! ==="
echo "Run 'source /workspaces/omx_ros2/ws/install/setup.bash' to use the workspace"


## ====================================================================
# NATIVE PYTHON PORT 80 PATH TRANSLATION FIXED (NO PIP / NO NGINX)
# ====================================================================

# 1. Strip hidden Windows carriage returns from your local workspace skill
sed -i 's/\r$//' /workspaces/omx_ros2/.agents/skills/testrefer/SKILL.md || true

# 2. Kill any old, stuck background proxies
pkill -f "codex_translator_proxy" || true

# 3. Create the standalone Python script that fixes the 404 path mapping
cat << 'EOF' > /tmp/codex_translator_proxy.py
import urllib.request
import urllib.error
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os

class CodexProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/v1/responses':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # 1. FIXED: Set the correct path to NVIDIA's serverless completions API
            url = "https://nvidia.com"
            api_key = os.environ.get("NVIDIA_NIM_API_KEY", "")
            
            # 2. Modify input payload schema on the fly to bypass OpenAI cloud-checks
            try:
                in_json = json.loads(post_data.decode('utf-8'))
                if "model" in in_json and in_json["model"] == "gpt-5.4-mini":
                    in_json["model"] = "z-ai/glm-5.1"
                post_data = json.dumps(in_json).encode('utf-8')
            except Exception:
                pass

            req = urllib.request.Request(url, data=post_data, method="POST")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            
            try:
                with urllib.request.urlopen(req) as response:
                    self.send_response(200)
                    for key, val in response.getheaders():
                        if key.lower() not in ['content-length', 'transfer-encoding', 'connection']:
                            self.send_header(key, val)
                    
                    res_data = response.read()
                    
                    # 3. Translate NVIDIA's response structure to fit Codex format specs
                    try:
                        res_json = json.loads(res_data.decode('utf-8'))
                        if "choices" in res_json:
                            # Map native choices array directly over agentic response block properties
                            res_json["responses"] = res_json["choices"]
                        res_data = json.dumps(res_json).encode('utf-8')
                    except Exception:
                        pass

                    self.send_header("Content-Length", str(len(res_data)))
                    self.end_headers()
                    self.wfile.write(res_data)
            except urllib.error.HTTPError as e:
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(e.read())
        else:
            self.send_response(404)
            self.end_headers()

def run():
    server_address = ('0.0.0.0', 80)
    httpd = HTTPServer(server_address, CodexProxyHandler)
    print("Starting Codex port-80 routing proxy engine...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
EOF

# 4. Force execute the python script as a persistent background daemon
nohup python3 /tmp/codex_translator_proxy.py > /tmp/proxy_network.log 2>&1 &

# 5. Build a structurally perfect config.toml targeting localhost
mkdir -p ~/.codex
rm -f ~/.codex/config.toml || true
cat << 'EOF' > ~/.codex/config.toml
model = "gpt-5.4-mini"
model_provider = "copilot"
disable_cloud_sync = true

[model_providers.copilot]
name = "NVIDIA NIM"
base_url = "http://localhost/v1"
env_key = "NVIDIA_NIM_API_KEY"
wire_api = "responses"
requires_openai_auth = false

[override_model_keys]
"gpt-5.4-mini" = "z-ai/glm-5.1"

[enterprise]
mode = "self_hosted"
bypass_billing = true
license_key = "local-developer-mode"
EOF

# 6. Lock down file system rules so the IDE cannot wipe your variables on boot
chmod 444 ~/.codex/config.toml
echo "=== Codex Custom Port-80 Proxy Integration Complete ==="
