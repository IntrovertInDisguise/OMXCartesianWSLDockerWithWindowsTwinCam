#!/usr/bin/env python3
"""Send recorder events and own the press-scoped rosbag process."""
import argparse, json, os, signal, socket, subprocess, sys, time
from datetime import datetime, timezone

def send(host, port, payload):
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(data, (host, port))

def stop_bag(pid_file, timeout=15.0):
    try:
        pid = int(open(pid_file, encoding="ascii").read().strip())
    except (FileNotFoundError, ValueError):
        return
    try:
        os.killpg(pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try: os.kill(pid, 0)
        except ProcessLookupError: break
        time.sleep(.1)
    else:
        try: os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: os.unlink(pid_file)
    except FileNotFoundError: pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("event", choices=("START_RUN", "PRESS_BEGIN", "STOP_RUN", "ABORT_RUN"))
    p.add_argument("--host", default=os.getenv("OMX_RECORDER_HOST", "host.docker.internal"))
    p.add_argument("--port", type=int, default=int(os.getenv("OMX_RECORDER_PORT", "5010")))
    p.add_argument("--folder")
    p.add_argument("--metadata-file")
    p.add_argument("--pid-file")
    p.add_argument("--rosbag-dir")
    p.add_argument("--topic", action="append", default=[])
    a = p.parse_args()
    payload = {"event": a.event, "sent_utc": datetime.now(timezone.utc).isoformat()}
    if a.event == "START_RUN":
        if not a.folder or not a.metadata_file: p.error("START_RUN requires --folder and --metadata-file")
        payload.update(folder=a.folder, metadata=json.load(open(a.metadata_file, encoding="utf-8")))
    if a.event == "PRESS_BEGIN" and a.rosbag_dir:
        if not a.pid_file: p.error("--rosbag-dir requires --pid-file")
        cmd = ["ros2", "bag", "record", "-o", a.rosbag_dir] + (a.topic or ["-a"])
        log = open(os.path.join(os.path.dirname(a.pid_file), "rosbag.log"), "ab", buffering=0)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with open(a.pid_file, "w", encoding="ascii") as f: f.write(str(proc.pid))
        payload["rosbag_started_utc"] = datetime.now(timezone.utc).isoformat()
        payload["rosbag_topics"] = a.topic or ["-a"]
    send(a.host, a.port, payload)
    if a.event in ("STOP_RUN", "ABORT_RUN") and a.pid_file:
        stop_bag(a.pid_file)

if __name__ == "__main__":
    main()