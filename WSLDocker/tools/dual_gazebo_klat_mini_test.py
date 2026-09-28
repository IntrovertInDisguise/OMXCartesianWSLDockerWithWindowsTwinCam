#!/usr/bin/env python3
"""Run a small 6-case dual-Gazebo lateral-stiffness ablation sweep.

Each case reuses the validated dual staged-compression launch, but overrides the
press-mode YAMLs with a case-specific y-axis stiffness profile. The harness also
pushes slightly deeper than the baseline press config and requires a sustained
final hold so each case actually reaches and dwells at the deepest compression
level before it is marked as passed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Dict, List

import yaml


WORKSPACE_ROOT = Path("/workspaces/omx_ros2")
ROBOT1_BASE_YAML = WORKSPACE_ROOT / "ws/src/omx_variable_stiffness_controller/config/robot1_gazebo_variable_stiffness.yaml"
ROBOT2_BASE_YAML = WORKSPACE_ROOT / "ws/src/omx_variable_stiffness_controller/config/robot2_gazebo_variable_stiffness.yaml"
OUTPUT_ROOT = Path("/tmp/dual_gazebo_klat_mini_test")
PROFILE_PEAKS_NPM = [10.0, 14.0, 18.0, 22.0, 25.0, 28.0]
FINAL_STAGE_COUNT = 4
DEFAULT_COMPRESSION_OFFSET_MM = 12.0
DEFAULT_FINAL_HOLD_S = 12.0

FAILURE_MARKERS = [
    "Failed to configure controller",
    "process has died",
    "Parameter 'joints' is empty",
    "[ERROR]",
    "Traceback (most recent call last)",
]


def cleanup_processes() -> None:
    for pattern in ["gzserver", "gzclient", "spawn_entity.py", "ros2 launch omx_variable_stiffness_controller dual_gazebo_variable_stiffness.launch.py"]:
        subprocess.run(["pkill", "-f", pattern], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.time() + 20.0
    while time.time() < deadline:
        alive = False
        for pattern in ["gzserver", "gzclient", "spawn_entity.py"]:
            probe = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
            if probe.returncode == 0 and probe.stdout.strip():
                alive = True
                break
        if not alive:
            return
        time.sleep(0.5)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def get_controller_key(robot_name: str) -> str:
    return f"/{robot_name}/{robot_name}_variable_stiffness"


def deepen_compression_profile(controller_block: Dict, compression_offset_m: float) -> None:
    if compression_offset_m <= 0.0:
        return

    stage_targets = [float(value) for value in controller_block.get("compression_stage_x", [])]
    if not stage_targets:
        return

    stage_count = len(stage_targets)
    controller_block["compression_stage_x"] = [
        round(target + compression_offset_m * ((index + 1) / stage_count), 6)
        for index, target in enumerate(stage_targets)
    ]

    end_position = [float(value) for value in controller_block["end_position"]]
    end_position[0] = round(end_position[0] + compression_offset_m, 6)
    controller_block["end_position"] = end_position


def build_case_yaml(
    base_data: Dict,
    robot_name: str,
    target_peak: float,
    compression_offset_m: float,
) -> Dict:
    case_data = json.loads(json.dumps(base_data))
    controller_key = get_controller_key(robot_name)
    controller_block = case_data[controller_key]["ros__parameters"]
    y_profile = controller_block["stiffness_profile_y"]
    baseline_peak = max(float(value) for value in y_profile)
    scale = target_peak / baseline_peak
    controller_block["stiffness_profile_y"] = [round(float(value) * scale, 6) for value in y_profile]
    deepen_compression_profile(controller_block, compression_offset_m)
    return case_data


def parse_case_result(text: str) -> Dict[str, object]:
    markers = {
        "robot1_start": "robot1.robot1_variable_stiffness]: [STATE] WAIT_AT_START done → MOVE_FORWARD",
        "robot2_start": "robot2.robot2_variable_stiffness]: [STATE] WAIT_AT_START done → MOVE_FORWARD",
        "robot1_stage2": "robot1.robot1_variable_stiffness]: [STATE] WAIT_AT_END done → MOVE_FORWARD | advancing compression stage 2/4",
        "robot2_stage2": "robot2.robot2_variable_stiffness]: [STATE] WAIT_AT_END done → MOVE_FORWARD | advancing compression stage 2/4",
        "robot1_stage3": "robot1.robot1_variable_stiffness]: [STATE] WAIT_AT_END done → MOVE_FORWARD | advancing compression stage 3/4",
        "robot2_stage3": "robot2.robot2_variable_stiffness]: [STATE] WAIT_AT_END done → MOVE_FORWARD | advancing compression stage 3/4",
        "robot1_stage4": "robot1.robot1_variable_stiffness]: [STATE] WAIT_AT_END done → MOVE_FORWARD | advancing compression stage 4/4",
        "robot2_stage4": "robot2.robot2_variable_stiffness]: [STATE] WAIT_AT_END done → MOVE_FORWARD | advancing compression stage 4/4",
        "robot1_gripper": "robot1.gripper_controller]: Received & accepted new action goal",
        "robot2_gripper": "robot2.gripper_controller]: Received & accepted new action goal",
    }
    result = {name: marker in text for name, marker in markers.items()}
    result["robot1_forward_done_count"] = text.count(
        "robot1.robot1_variable_stiffness]: [STATE] MOVE_FORWARD done → WAIT_AT_END"
    )
    result["robot2_forward_done_count"] = text.count(
        "robot2.robot2_variable_stiffness]: [STATE] MOVE_FORWARD done → WAIT_AT_END"
    )
    result["robot1_final_target_reached"] = result["robot1_forward_done_count"] >= FINAL_STAGE_COUNT
    result["robot2_final_target_reached"] = result["robot2_forward_done_count"] >= FINAL_STAGE_COUNT
    result["move_return_seen"] = "MOVE_RETURN" in text
    result["failures"] = [marker for marker in FAILURE_MARKERS if marker in text]
    result["ready_for_final_hold"] = (
        all(result[name] for name in markers)
        and result["robot1_final_target_reached"]
        and result["robot2_final_target_reached"]
        and not result["move_return_seen"]
        and not result["failures"]
    )
    result["passed"] = False
    return result


def launch_case(
    env: Dict[str, str],
    robot1_yaml: Path,
    robot2_yaml: Path,
    log_path: Path,
    timeout_s: float,
    gui_enabled: bool,
    final_hold_s: float,
) -> Dict[str, object]:
    cmd = [
        "ros2",
        "launch",
        "omx_variable_stiffness_controller",
        "dual_gazebo_variable_stiffness.launch.py",
        f"gui:={'true' if gui_enabled else 'false'}",
        "enable_live_plot:=false",
        "enable_controllers:=true",
        f"robot1_press_yaml:={robot1_yaml}",
        f"robot2_press_yaml:={robot2_yaml}",
    ]

    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)

    started = time.time()
    final_hold_started_at = None
    try:
        while time.time() - started < timeout_s:
            text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            parsed = parse_case_result(text)
            if parsed["failures"]:
                parsed["elapsed_s"] = round(time.time() - started, 2)
                return parsed
            if parsed["ready_for_final_hold"]:
                if final_hold_started_at is None:
                    final_hold_started_at = time.time()
                final_hold_elapsed = time.time() - final_hold_started_at
                if final_hold_elapsed >= final_hold_s:
                    parsed["passed"] = True
                    parsed["elapsed_s"] = round(time.time() - started, 2)
                    parsed["final_hold_s"] = round(final_hold_elapsed, 2)
                    return parsed
            elif final_hold_started_at is not None:
                final_hold_started_at = None

            if proc.poll() is not None:
                parsed["elapsed_s"] = round(time.time() - started, 2)
                if not parsed["failures"]:
                    parsed["failures"] = [f"launch_exited_{proc.returncode}"]
                return parsed
            time.sleep(1.0)

        parsed = parse_case_result(log_path.read_text(encoding="utf-8"))
        parsed["elapsed_s"] = round(time.time() - started, 2)
        parsed["failures"] = list(parsed["failures"]) + [f"timeout>{timeout_s:.0f}s"]
        return parsed
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        cleanup_processes()


def build_env(gui_enabled: bool, display: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    if gui_enabled:
        env["DISPLAY"] = display
        env.pop("SDL_AUDIODRIVER", None)
        env.pop("GAZEBO_HEADLESS_RENDERING", None)
    else:
        env.setdefault("SDL_AUDIODRIVER", "dummy")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 6-case dual-Gazebo K_lat mini test")
    parser.add_argument("--timeout", type=float, default=150.0, help="Per-case timeout in seconds")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT), help="Directory for logs and generated YAMLs")
    parser.add_argument("--peaks", nargs="*", type=float, default=PROFILE_PEAKS_NPM, help="Target y-stiffness peak values to sweep")
    parser.add_argument(
        "--compression-offset-mm",
        type=float,
        default=DEFAULT_COMPRESSION_OFFSET_MM,
        help="Extra final x compression to distribute across the staged targets",
    )
    parser.add_argument(
        "--final-hold-s",
        type=float,
        default=DEFAULT_FINAL_HOLD_S,
        help="Required dwell at the deepest compression target before a case can pass",
    )
    parser.add_argument("--gui", action="store_true", help="Run each case visually through gzclient over X11")
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":0"), help="X11 display to use when --gui is set")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    log_dir = output_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_r1 = load_yaml(ROBOT1_BASE_YAML)
    base_r2 = load_yaml(ROBOT2_BASE_YAML)
    env = build_env(args.gui, args.display)
    compression_offset_m = args.compression_offset_mm / 1000.0

    results: List[Dict[str, object]] = []
    cleanup_processes()

    for index, peak in enumerate(args.peaks, start=1):
        case_name = f"case_{index:02d}_ky_peak_{str(peak).replace('.', 'p')}"
        robot1_case_yaml = config_dir / f"{case_name}_robot1.yaml"
        robot2_case_yaml = config_dir / f"{case_name}_robot2.yaml"
        write_yaml(robot1_case_yaml, build_case_yaml(base_r1, "robot1", peak, compression_offset_m))
        write_yaml(robot2_case_yaml, build_case_yaml(base_r2, "robot2", peak, compression_offset_m))

        mode = f"gui display {args.display}" if args.gui else "headless"
        print(
            f"[{index}/{len(args.peaks)}] Running {case_name} "
            f"(target lateral peak {peak:.1f} N/m, compression +{args.compression_offset_mm:.1f} mm, {mode})"
        )
        result = launch_case(
            env,
            robot1_case_yaml,
            robot2_case_yaml,
            log_dir / f"{case_name}.log",
            args.timeout,
            args.gui,
            args.final_hold_s,
        )
        result["case"] = case_name
        result["target_y_peak_npm"] = peak
        result["compression_offset_mm"] = args.compression_offset_mm
        result["required_final_hold_s"] = args.final_hold_s
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  -> {status} in {result['elapsed_s']} s")
        if result["failures"]:
            print(f"     failures: {result['failures']}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    passed = sum(1 for item in results if item["passed"])
    print(f"\nMini test complete: {passed}/{len(results)} passed")
    print(f"Summary: {summary_path}")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{item['case']}: {status} peak={item['target_y_peak_npm']:.1f} elapsed={item['elapsed_s']}s")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())