#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


DURATION_RE = re.compile(
    r"Dynamixel\s+(Read|Write)\s+Fail\s+"
    r"\(Duration:\s*"
    r"([-+0-9.eE]+)ms/"
    r"([-+0-9.eE]+)ms\)"
)

REBOOT_RE = re.compile(
    r"Dynamixel\s+Read\s+Fail\s*:\s*REBOOTING"
)


def parse_fault(line: str):
    if REBOOT_RE.search(line):
        return {
            "kind": "rebooting",
            "duration_ms": None,
            "timeout_ms": None,
            "line": line.rstrip(),
        }

    m = DURATION_RE.search(line)
    if not m:
        return None

    kind = m.group(1).lower()
    duration_ms = float(m.group(2))
    timeout_ms = float(m.group(3))

    if duration_ms < timeout_ms:
        return None

    return {
        "kind": f"{kind}_timeout",
        "duration_ms": duration_ms,
        "timeout_ms": timeout_ms,
        "line": line.rstrip(),
    }


def write_fault(path: Path, fault):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        f"wall_time_s={time.time():.9f}",
        f"kind={fault['kind']}",
    ]

    if fault["duration_ms"] is not None:
        fields.append(f"duration_ms={fault['duration_ms']:.6f}")

    if fault["timeout_ms"] is not None:
        fields.append(f"timeout_ms={fault['timeout_ms']:.6f}")

    fields.append(f"log_line={fault['line']}")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(fields) + "\n")
    tmp.replace(path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_hardware_now(bringup_pid: int):
    # Immediate safety action. The parent runner will perform the full
    # cleanup/finalization sequence afterward.
    try:
        os.kill(bringup_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    for pattern in (
        "ros2_control_node",
        "single_arm_hardware",
        "spawner",
    ):
        subprocess.run(
            ["pkill", "-TERM", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def notify_parent(parent_pid: int):
    try:
        os.kill(parent_pid, signal.SIGUSR1)
    except ProcessLookupError:
        pass


def scan_existing(log_path: Path):
    if not log_path.exists():
        return None

    with log_path.open("r", errors="replace") as f:
        for line in f:
            fault = parse_fault(line)
            if fault:
                return fault

    return None


def run_scan_only(log_path: Path) -> int:
    fault = scan_existing(log_path)

    if fault:
        print(
            "[WATCHDOG TEST] FAULT "
            f"kind={fault['kind']} "
            f"duration_ms={fault['duration_ms']} "
            f"timeout_ms={fault['timeout_ms']}"
        )
        return 2

    print("[WATCHDOG TEST] NO_FAULT")
    return 0


def monitor(
    log_path: Path,
    bringup_pid: int,
    parent_pid: int,
    fault_file: Path,
    poll_s: float,
) -> int:
    # The shell redirect creates the log before this process starts, but
    # tolerate a short creation delay anyway.
    while not log_path.exists():
        if not pid_alive(bringup_pid):
            return 0
        time.sleep(poll_s)

    # Scan anything written between ros2 launch and watchdog startup.
    with log_path.open("r", errors="replace") as f:
        while True:
            line = f.readline()

            if line:
                fault = parse_fault(line)

                if fault:
                    write_fault(fault_file, fault)

                    print(
                        "[HARDWARE-COMM-WATCHDOG] FAULT "
                        f"kind={fault['kind']} "
                        f"duration_ms={fault['duration_ms']} "
                        f"timeout_ms={fault['timeout_ms']}",
                        flush=True,
                    )

                    # Safety first: stop the hardware process immediately.
                    stop_hardware_now(bringup_pid)

                    # Then wake the parent runner so it can record diagnostics,
                    # shut down acquisition, and return a failed run status.
                    notify_parent(parent_pid)

                    return 2

                continue

            if not pid_alive(bringup_pid):
                return 0

            time.sleep(poll_s)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--log-file", required=True)
    p.add_argument("--bringup-pid", type=int)
    p.add_argument("--parent-pid", type=int)
    p.add_argument("--fault-file")
    p.add_argument("--poll-s", type=float, default=0.02)
    p.add_argument("--scan-only", action="store_true")

    a = p.parse_args()

    log_path = Path(a.log_file)

    if a.scan_only:
        return run_scan_only(log_path)

    if a.bringup_pid is None:
        p.error("--bringup-pid is required unless --scan-only is used")

    if a.parent_pid is None:
        p.error("--parent-pid is required unless --scan-only is used")

    if not a.fault_file:
        p.error("--fault-file is required unless --scan-only is used")

    return monitor(
        log_path=log_path,
        bringup_pid=a.bringup_pid,
        parent_pid=a.parent_pid,
        fault_file=Path(a.fault_file),
        poll_s=max(0.005, float(a.poll_s)),
    )


if __name__ == "__main__":
    sys.exit(main())
