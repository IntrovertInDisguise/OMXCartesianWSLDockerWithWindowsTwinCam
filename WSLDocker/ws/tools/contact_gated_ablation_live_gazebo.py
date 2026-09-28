#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


WORKSPACE_ROOT = Path("/workspaces/omx_ros2")
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
WS_SETUP_CANDIDATES = (
    WORKSPACE_ROOT / "ws/install/setup.bash",
    WORKSPACE_ROOT / "install/setup.bash",
)
HARNESS_PATH = WORKSPACE_ROOT / "tools/hardware_harness_contact_gated_load.py"
DEFAULT_OUTPUT_ROOT = Path("/tmp/contact_gated_ablation_live_gazebo")
CONTROLLER_START_MARKERS = {
    "robot1": "robot1.robot1_variable_stiffness]: [STATE] WAIT_AT_START done → MOVE_FORWARD",
    "robot2": "robot2.robot2_variable_stiffness]: [STATE] WAIT_AT_START done → MOVE_FORWARD",
}


def parse_csv_arg(value: str) -> List[float]:
    tokens = [token.strip() for token in value.replace(";", ",").split(",")]
    numbers = [float(token) for token in tokens if token]
    if not numbers:
        raise argparse.ArgumentTypeError("at least one value is required")
    return numbers


def resolve_workspace_setup() -> Path:
    for candidate in WS_SETUP_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No workspace setup.bash found; tried: "
        + ", ".join(str(candidate) for candidate in WS_SETUP_CANDIDATES)
    )


def load_ros_env() -> Dict[str, str]:
    ws_setup = resolve_workspace_setup()
    command = (
        f"source {ROS_SETUP} && "
        f"source {ws_setup} && "
        "env -0"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
    )
    env: Dict[str, str] = {}
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        key, _, raw_value = entry.partition(b"=")
        env[key.decode("utf-8")] = raw_value.decode("utf-8")
    return env


def cleanup_processes() -> None:
    patterns = [
        "gzserver",
        "gzclient",
        "spawn_entity",
        "hardware_harness_contact_gated_load.py",
        "dual_gazebo_variable_stiffness.launch.py",
    ]
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    deadline = time.time() + 20.0
    while time.time() < deadline:
        alive = False
        for pattern in ("gzserver", "gzclient", "spawn_entity"):
            probe = subprocess.run(
                ["pgrep", "-f", pattern],
                check=False,
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                alive = True
                break
        if not alive:
            return
        time.sleep(0.5)


def launch_gazebo(env: Dict[str, str], log_path: Path, *, gui: bool, display: str) -> subprocess.Popen[str]:
    launch_env = dict(env)
    launch_env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    if gui:
        launch_env["DISPLAY"] = display
        launch_env.pop("SDL_AUDIODRIVER", None)
        launch_env.pop("GAZEBO_HEADLESS_RENDERING", None)
    else:
        launch_env.setdefault("SDL_AUDIODRIVER", "dummy")
        launch_env.setdefault("GAZEBO_HEADLESS_RENDERING", "1")

    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "ros2",
            "launch",
            "omx_variable_stiffness_controller",
            "dual_gazebo_variable_stiffness.launch.py",
            f"gui:={'true' if gui else 'false'}",
            "enable_controllers:=true",
            "enable_live_plot:=false",
        ],
        cwd=str(WORKSPACE_ROOT),
        env=launch_env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process._log_handle = handle  # type: ignore[attr-defined]
    return process


def stop_process(process: Optional[subprocess.Popen[str]]) -> None:
    if process is None:
        return
    handle = getattr(process, "_log_handle", None)
    try:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10.0)
    except ProcessLookupError:
        pass
    finally:
        if handle is not None:
            handle.close()


def topic_ready(env: Dict[str, str], topic: str, expected_line: str) -> bool:
    probe = subprocess.run(
        ["ros2", "topic", "info", topic],
        cwd=str(WORKSPACE_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0 and expected_line in probe.stdout


def wait_for_topics(env: Dict[str, str], launch_process: subprocess.Popen[str], timeout_s: float) -> bool:
    checks = [
        ("/robot1/robot1_variable_stiffness/waypoint_command", "Subscription count: 1"),
        ("/robot2/robot2_variable_stiffness/waypoint_command", "Subscription count: 1"),
        ("/robot1/robot1_variable_stiffness/waypoint_active", "Publisher count: 1"),
        ("/robot2/robot2_variable_stiffness/waypoint_active", "Publisher count: 1"),
    ]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if launch_process.poll() is not None:
            return False
        if all(topic_ready(env, topic, expected) for topic, expected in checks):
            return True
        time.sleep(1.0)
    return False


def launch_log_contains_markers(log_path: Path, markers: Dict[str, str]) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in markers.values())


def wait_for_controller_start(
    log_path: Path,
    launch_process: subprocess.Popen[str],
    timeout_s: float,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if launch_process.poll() is not None:
            return False
        if launch_log_contains_markers(log_path, CONTROLLER_START_MARKERS):
            return True
        time.sleep(1.0)
    return False


def extract_json_payload(text: str) -> Any:
    positions = [index for index, char in enumerate(text) if char == "["]
    for index in reversed(positions):
        try:
            return json.loads(text[index:])
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON payload found in harness output")


def run_harness(
    env: Dict[str, str],
    log_path: Path,
    *,
    k_lateral_cases_npm: List[float],
    peak_load_cases_n: List[float],
    hold_duration_s: float,
    repeats_n: int,
    extra_args: List[str],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    harness_env = dict(env)
    harness_env["OMX_LOG_DIR"] = str(output_dir / "harness_logs")
    Path(harness_env["OMX_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    command = [
        "python3",
        str(HARNESS_PATH),
        "--ignore-local-frame-ee-safety",
        "--ablate",
        "--hold-duration-s",
        str(hold_duration_s),
        "--repeats-n",
        str(repeats_n),
        "--k-lateral-cases-npm",
        ",".join(str(value) for value in k_lateral_cases_npm),
        "--peak-load-cases-n",
        ",".join(str(value) for value in peak_load_cases_n),
    ]
    command.extend(extra_args)

    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=str(WORKSPACE_ROOT),
            env=harness_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    text = log_path.read_text(encoding="utf-8")
    payload = extract_json_payload(text)
    if not isinstance(payload, list):
        raise ValueError("unexpected harness JSON payload")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, output=text)
    return payload


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    stage0 = next((item for item in results if item.get("stage") == 0), None)
    combos = [item for item in results if "case" in item and item.get("stage") != 0]
    passed_combos = [item for item in combos if item.get("pass")]
    total_trials = sum(item.get("trial_count", 1) for item in combos)
    passed_trials = sum(item.get("pass_count", 1 if item.get("pass") else 0) for item in combos)
    return {
        "stage0_pass": bool(stage0 and stage0.get("pass")),
        "case_count": len(combos),
        "passed_case_count": len(passed_combos),
        "total_trials": total_trials,
        "passed_trials": passed_trials,
        "failed_cases": [item.get("case") for item in combos if not item.get("pass")],
        "cases": combos,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live Gazebo contact-gated ablation slice")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for launch logs, harness logs, and summary output",
    )
    parser.add_argument(
        "--k-lateral-cases-npm",
        type=parse_csv_arg,
        default=[10.0, 18.0],
        help="Comma-separated K_lat cases to run",
    )
    parser.add_argument(
        "--peak-load-cases-n",
        type=parse_csv_arg,
        default=[0.05, 0.07],
        help="Comma-separated peak-load cases to run",
    )
    parser.add_argument(
        "--hold-duration-s",
        type=float,
        default=6.0,
        help="Per-target dwell duration passed through to the harness",
    )
    parser.add_argument(
        "--repeats-n",
        type=int,
        default=1,
        help="Number of trials per (K_lat, peak-load) combo (e.g. 5 for Gazebo, 20 for hardware)",
    )
    parser.add_argument(
        "--topic-timeout-s",
        type=float,
        default=180.0,
        help="Seconds to wait for controller-facing topics after Gazebo launch",
    )
    parser.add_argument(
        "--controller-start-timeout-s",
        type=float,
        default=180.0,
        help="Seconds to wait for both controllers to leave WAIT_AT_START after topics are ready",
    )
    parser.add_argument("--gui", action="store_true", help="Run Gazebo with gzclient over X11")
    parser.add_argument(
        "--display",
        default=os.environ.get("DISPLAY", ":0"),
        help="X11 display to use when --gui is set",
    )
    parser.add_argument(
        "harness_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to hardware_harness_contact_gated_load.py after --",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_log = output_dir / "gazebo_launch.log"
    harness_log = output_dir / "contact_gated_ablation.log"
    summary_path = output_dir / "summary.json"

    extra_args = list(args.harness_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    env = load_ros_env()
    launch_process: Optional[subprocess.Popen[str]] = None
    cleanup_processes()
    try:
        launch_process = launch_gazebo(env, launch_log, gui=args.gui, display=args.display)
        if not wait_for_topics(env, launch_process, args.topic_timeout_s):
            raise RuntimeError(
                f"controller topics did not become ready within {args.topic_timeout_s:.0f}s; see {launch_log}"
            )
        if not wait_for_controller_start(launch_log, launch_process, args.controller_start_timeout_s):
            raise RuntimeError(
                "controllers did not leave WAIT_AT_START within "
                f"{args.controller_start_timeout_s:.0f}s after topics were ready; see {launch_log}"
            )

        results = run_harness(
            env,
            harness_log,
            k_lateral_cases_npm=args.k_lateral_cases_npm,
            peak_load_cases_n=args.peak_load_cases_n,
            hold_duration_s=args.hold_duration_s,
            repeats_n=args.repeats_n,
            extra_args=extra_args,
            output_dir=output_dir,
        )
        summary = summarize_results(results)
        summary.update(
            {
                "k_lateral_cases_npm": args.k_lateral_cases_npm,
                "peak_load_cases_n": args.peak_load_cases_n,
                "hold_duration_s": args.hold_duration_s,
                "repeats_n": args.repeats_n,
                "launch_log": str(launch_log),
                "harness_log": str(harness_log),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(
            f"Ablation run complete: {summary['passed_case_count']}/{summary['case_count']} combos passed "
            f"({summary['passed_trials']}/{summary['total_trials']} trials). "
            f"Summary: {summary_path}"
        )
        for case in summary["cases"]:
            trial_count = case.get("trial_count", 1)
            pass_count = case.get("pass_count", 1 if case.get("pass") else 0)
            status = "PASS" if case.get("pass") else "FAIL"
            if trial_count > 1:
                print(f"{case.get('case')}: {status} ({pass_count}/{trial_count} trials) -> {case.get('message')}")
            else:
                print(f"{case.get('case')}: {status} -> {case.get('message')}")
        return 0 if summary["stage0_pass"] and summary["passed_case_count"] == summary["case_count"] else 1
    finally:
        stop_process(launch_process)
        cleanup_processes()


if __name__ == "__main__":
    sys.exit(main())