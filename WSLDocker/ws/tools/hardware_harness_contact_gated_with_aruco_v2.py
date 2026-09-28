#!/usr/bin/env python3
"""Contact-gated load-step harness with an ArUco z-trim precheck.

This v2 variant keeps the validated contact-gated 5-stage protocol from
hardware_harness_contact_gated_load.py and adds the missing pre-attempt ArUco
alignment gate so one harness can cover:

- stage-0 calibration / specimen ingestion,
- K_lat x peak-load ablation grids,
- deferred K_lat application after bilateral precontact,
- mutual-contact rosbag windows,
- external host-side START/STOP camera triggering, and
- ArUco-driven z-trim auto-alignment before each compression attempt.

The ArUco loop here is intentionally narrow: it consumes only the published
z-trim recommendations and does not attempt to servo roll or y-centering.
Those signals are still surfaced in status messages for the operator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from std_msgs.msg import Bool, Float64

from hardware_harness_contact_gated_load import (
    ContactGatedLoadStepHarness,
    build_arg_parser as build_contact_gated_arg_parser,
)

try:
    from tools.ablation_config import (
        DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
        DEFAULT_ALIGNMENT_POLICY,
        DEFAULT_ALIGNMENT_SOURCE,
    )
    from tools.aruco_alignment_utils import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROLL_DEG_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
    )
except ImportError:
    from ablation_config import (
        DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
        DEFAULT_ALIGNMENT_POLICY,
        DEFAULT_ALIGNMENT_SOURCE,
    )
    from aruco_alignment_utils import (
        ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC,
        ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC,
        ARUCO_ALIGNMENT_ROLL_DEG_TOPIC,
        ARUCO_ALIGNMENT_VALID_TOPIC,
    )


ALIGNMENT_SOURCE_CHOICES = ("off", "auto", "aruco")
ALIGNMENT_POLICY_CHOICES = ("off", "warn", "manual", "auto_then_manual")


class ContactGatedWithArucoV2Harness(ContactGatedLoadStepHarness):
    """Contact-gated load-step harness with an ArUco pre-attempt alignment gate.

    Depending on ``alignment_policy``, the gate may auto-apply small z trims,
    warn only, prompt for manual realignment, or be disabled entirely.
    """

    def __init__(
        self,
        *args: Any,
        alignment_source: str = DEFAULT_ALIGNMENT_SOURCE,
        alignment_policy: str = DEFAULT_ALIGNMENT_POLICY,
        alignment_max_auto_z_trim: float = DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
        non_interactive: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.alignment_source = str(alignment_source or DEFAULT_ALIGNMENT_SOURCE).strip().lower()
        if self.alignment_source not in ALIGNMENT_SOURCE_CHOICES:
            raise ValueError(
                f"alignment_source must be one of {', '.join(ALIGNMENT_SOURCE_CHOICES)}"
            )

        self.alignment_policy = str(alignment_policy or DEFAULT_ALIGNMENT_POLICY).strip().lower()
        if self.alignment_policy not in ALIGNMENT_POLICY_CHOICES:
            raise ValueError(
                f"alignment_policy must be one of {', '.join(ALIGNMENT_POLICY_CHOICES)}"
            )

        self.alignment_max_auto_z_trim = max(0.0, float(alignment_max_auto_z_trim))

        # When True, avoid blocking operator prompts so the harness may run in CI.
        self.non_interactive = bool(non_interactive)

        # In non-interactive (CI) mode, tolerate slower controller settle by
        # increasing the precontact waypoint timeout when the user did not
        # explicitly request a larger value.
        try:
            current_timeout = getattr(self, "precontact_waypoint_timeout_s", None)
            if self.non_interactive and current_timeout is not None and current_timeout < 12.0:
                self.get_logger().info(
                    f"Non-interactive: increasing precontact_waypoint_timeout_s from {current_timeout} to 12.0s"
                )
                self.precontact_waypoint_timeout_s = 12.0
        except Exception:
            pass

        self.aruco_alignment_valid = False
        self.aruco_robot1_z_trim = float("nan")
        self.aruco_robot2_z_trim = float("nan")
        self.aruco_alignment_roll_deg = float("nan")
        self.aruco_alignment_center_y_error = float("nan")
        self.alignment_robot1_z_trim = 0.0
        self.alignment_robot2_z_trim = 0.0
        self.last_alignment_source = "unchecked"
        self.last_alignment_status = "unchecked"
        self.last_alignment_message = "alignment not checked yet"
        self._last_alignment_stage_record: Optional[Dict[str, Any]] = None

        self.create_subscription(Bool, ARUCO_ALIGNMENT_VALID_TOPIC, self._cb_aruco_alignment_valid, 10)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_ROBOT1_Z_TRIM_TOPIC, self._cb_aruco_robot1_z_trim, 10)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_ROBOT2_Z_TRIM_TOPIC, self._cb_aruco_robot2_z_trim, 10)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_ROLL_DEG_TOPIC, self._cb_aruco_alignment_roll_deg, 10)
        self.create_subscription(Float64, ARUCO_ALIGNMENT_CENTER_Y_ERROR_TOPIC, self._cb_aruco_alignment_center_y_error, 10)

    def _cb_aruco_alignment_valid(self, msg: Bool) -> None:
        self.aruco_alignment_valid = bool(msg.data)

    def _cb_aruco_robot1_z_trim(self, msg: Float64) -> None:
        self.aruco_robot1_z_trim = float(msg.data)

    def _cb_aruco_robot2_z_trim(self, msg: Float64) -> None:
        self.aruco_robot2_z_trim = float(msg.data)

    def _cb_aruco_alignment_roll_deg(self, msg: Float64) -> None:
        self.aruco_alignment_roll_deg = float(msg.data)

    def _cb_aruco_alignment_center_y_error(self, msg: Float64) -> None:
        self.aruco_alignment_center_y_error = float(msg.data)

    def publish_offsets(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        repeats: Optional[int] = None,
        dt: Optional[float] = None,
    ) -> None:
        # Rely on the base harness to apply the configured alignment trims.
        # The base `HardwareHarnessAdaptive.publish_offsets` reads
        # `alignment_robot*_z_trim`/`alignment_robot*_y_trim` and applies them
        # automatically so avoid double-applying here.
        super().publish_offsets(x1, y1, z1, x2, y2, z2, repeats=repeats, dt=dt)

    def _set_last_alignment(self, source: str, status: str, message: str) -> None:
        self.last_alignment_source = source
        self.last_alignment_status = status
        self.last_alignment_message = message

    def _prompt_manual_alignment(self, reason: str) -> None:
        self.get_logger().warn(
            f"\n{'=' * 60}\n"
            "  ACTION REQUIRED: ArUco alignment indicates a mismatch.\n"
            f"  {reason}\n"
            "  Realign manually, then press ENTER to re-check.\n"
            f"{'=' * 60}"
        )
        if self.non_interactive:
            self.get_logger().warn("Non-interactive mode: skipping manual alignment prompt")
            self.spin_for(1.0)
            return

        try:
            input("  >> Realign manually, then press ENTER to re-check: ")
        except Exception:
            return
        self.spin_for(1.0)

    def _evaluate_alignment(self) -> Tuple[bool, str, str, Optional[Tuple[float, float]]]:
        if self.alignment_source not in {"auto", "aruco"}:
            return True, "off", "ArUco alignment disabled", None

        if not self.aruco_alignment_valid:
            return False, "aruco", "ArUco alignment topic is not valid yet", None

        trim1 = self.aruco_robot1_z_trim
        trim2 = self.aruco_robot2_z_trim
        if not (self.fin(trim1) and self.fin(trim2)):
            return False, "aruco", "ArUco recommended z trims are unavailable", None

        max_trim = max(abs(trim1), abs(trim2))
        detail_parts: List[str] = []
        if self.fin(self.aruco_alignment_roll_deg):
            detail_parts.append(f"roll={self.aruco_alignment_roll_deg:+.2f} deg")
        if self.fin(self.aruco_alignment_center_y_error):
            detail_parts.append(f"center_y_error={self.aruco_alignment_center_y_error:+.4f} m")
        detail_suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""

        if self.alignment_policy == "manual":
            if max_trim <= 5e-4:
                return True, "aruco", f"ArUco alignment already within tolerance{detail_suffix}", None
            return False, "aruco", (
                "manual realignment requested; recommended z trims "
                f"r1={trim1:+.4f} m r2={trim2:+.4f} m{detail_suffix}"
            ), None

        if max_trim <= self.alignment_max_auto_z_trim:
            return True, "aruco", (
                f"ArUco alignment trim accepted: r1={trim1:+.4f} m "
                f"r2={trim2:+.4f} m{detail_suffix}"
            ), (trim1, trim2)

        return False, "aruco", (
            "ArUco recommends larger z trims than allowed: "
            f"r1={trim1:+.4f} m r2={trim2:+.4f} m > limit "
            f"{self.alignment_max_auto_z_trim:.4f} m{detail_suffix}"
        ), None

    def ensure_camera_alignment(self, label: str) -> Tuple[bool, str]:
        if self.alignment_source == "off" or self.alignment_policy == "off":
            self.alignment_robot1_z_trim = 0.0
            self.alignment_robot2_z_trim = 0.0
            self._set_last_alignment("off", "skipped", "camera alignment check disabled")
            return True, "alignment skipped"

        self.spin_for(0.5)

        attempts = 0
        while True:
            ok, source, message, trims = self._evaluate_alignment()
            if ok:
                status = "ok"
                if trims is not None and self.alignment_policy == "auto_then_manual":
                    self.alignment_robot1_z_trim = trims[0]
                    self.alignment_robot2_z_trim = trims[1]
                    status = "auto_applied"
                else:
                    self.alignment_robot1_z_trim = 0.0
                    self.alignment_robot2_z_trim = 0.0
                self._set_last_alignment(source, status, message)
                self.get_logger().info(f"[{label}] Alignment [{source}] {message}")
                return True, f"alignment: {message}"

            self._set_last_alignment(source, "needs_manual", message)
            if self.alignment_policy == "warn":
                self.alignment_robot1_z_trim = 0.0
                self.alignment_robot2_z_trim = 0.0
                self.get_logger().warn(f"[{label}] Alignment warning [{source}] {message}")
                return True, f"alignment warning: {message}"

            if self.alignment_policy in {"auto_then_manual", "manual"} and attempts < 2:
                self.get_logger().warn(f"[{label}] Alignment [{source}] {message}")
                self._prompt_manual_alignment(message)
                attempts += 1
                continue

            self.alignment_robot1_z_trim = 0.0
            self.alignment_robot2_z_trim = 0.0
            self.get_logger().warn(f"[{label}] Alignment failed [{source}] {message}")
            return False, f"alignment: {message}"

    def _build_alignment_stage_record(self, aligned: bool, align_msg: str) -> Dict[str, Any]:
        return {
            "stage": "alignment",
            "pass": aligned,
            "message": align_msg,
            "info": {
                "alignment_source": self.last_alignment_source,
                "alignment_status": self.last_alignment_status,
                "alignment_message": self.last_alignment_message,
                "alignment_robot1_z_trim_m": self.alignment_robot1_z_trim,
                "alignment_robot2_z_trim_m": self.alignment_robot2_z_trim,
                "aruco_alignment_valid": self.aruco_alignment_valid,
                "aruco_robot1_z_trim_m": self.aruco_robot1_z_trim,
                "aruco_robot2_z_trim_m": self.aruco_robot2_z_trim,
                "aruco_roll_deg": self.aruco_alignment_roll_deg,
                "aruco_center_y_error_m": self.aruco_alignment_center_y_error,
            },
        }

    def stage1_liveness(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Override the base-stage liveness check to retry missing waypoint subscribers.

        The base implementation returns immediately after a single `wait_for_subscribers`
        attempt which can cause spurious aborts if controllers are still starting up.
        Retry a few times with short pauses before giving up.
        """
        max_attempts = 3
        attempt = 0
        last_info: Dict[str, Any] = {}
        while attempt < max_attempts:
            pubs_ok = self.wait_for_subscribers(timeout=10.0)
            js_ok = self.wait_joint_states(timeout=5.0)
            pose_ok = self.wait_for_poses(timeout=3.0)
            ok = pubs_ok and js_ok and pose_ok

            info = {
                "pub1_subs": self.pub1.get_subscription_count(),
                "pub2_subs": self.pub2.get_subscription_count(),
                "js1_seen": self.js1 is not None,
                "js2_seen": self.js2 is not None,
                "ee1_seen": self.ee1 is not None,
                "ee2_seen": self.ee2 is not None,
                "des1_seen": self.des1 is not None,
                "des2_seen": self.des2 is not None,
                "wp1_status_seen": self.wp1_active is not None,
                "wp2_status_seen": self.wp2_active is not None,
                "contact_valid_topics_seen": bool(self.contact_valid_1 or self.contact_valid_2),
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
            }
            last_info = info

            if ok:
                message = "publishers, joint states, and Cartesian pose topics present"
                return True, message, info

            # If waypoint subscribers are missing, retry a few times before failing.
            if not pubs_ok:
                attempt += 1
                if attempt < max_attempts:
                    self.get_logger().warn(
                        f"Stage 1: waypoint subscribers missing; retrying ({attempt}/{max_attempts})"
                    )
                    # small backoff to allow controllers to register
                    self.spin_for(1.0)
                    continue
                message = "waypoint subscribers missing after retries"
                return False, message, info

            # Otherwise surface joint_states/pose failures immediately
            if not js_ok:
                message = "joint_states missing"
                return False, message, info
            message = "Cartesian pose topics missing"
            return False, message, info

    def _prepare_case(
        self,
        *,
        case_name: str,
        peak_load_n: float,
        load_targets_n: Sequence[float],
        k_lateral_npm: Optional[float],
    ) -> None:
        super()._prepare_case(
            case_name=case_name,
            peak_load_n=peak_load_n,
            load_targets_n=load_targets_n,
            k_lateral_npm=k_lateral_npm,
        )

        aligned, align_msg = self.ensure_camera_alignment(case_name)
        self._last_alignment_stage_record = self._build_alignment_stage_record(aligned, align_msg)
        if not aligned:
            self.safe_abort()
            raise ValueError(align_msg)

        # Re-issue the neutral command after any accepted auto-trim so each
        # compression attempt starts from the aligned zero-offset pose.
        self._publish_zero_offsets()
        self._wait_for_arms_idle(timeout_s=12.0)

    def _run_case(
        self,
        *,
        case_name: str,
        peak_load_n: float,
        load_targets_n: Sequence[float],
        k_lateral_npm: Optional[float],
    ) -> Dict[str, Any]:
        self._last_alignment_stage_record = None
        result = super()._run_case(
            case_name=case_name,
            peak_load_n=peak_load_n,
            load_targets_n=load_targets_n,
            k_lateral_npm=k_lateral_npm,
        )
        if self._last_alignment_stage_record is not None:
            result_stages = result.setdefault("stages", [])
            result_stages.insert(0, self._last_alignment_stage_record)
        return result


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the contact-gated bilateral load-step harness with ArUco pre-alignment",
        parents=[build_contact_gated_arg_parser(add_help=False)],
        add_help=add_help,
    )
    parser.add_argument(
        "--alignment-source",
        choices=ALIGNMENT_SOURCE_CHOICES,
        default=DEFAULT_ALIGNMENT_SOURCE,
        help=(
            "Alignment signal source. 'off' skips the ArUco precheck, while "
            "'auto' and 'aruco' both require valid /spring_monitor/aruco_alignment/* topics."
        ),
    )
    parser.add_argument(
        "--alignment-policy",
        choices=ALIGNMENT_POLICY_CHOICES,
        default=DEFAULT_ALIGNMENT_POLICY,
        help=(
            "How to react to the recommended ArUco z trims. 'auto_then_manual' "
            "auto-applies trims within the limit and otherwise prompts for manual realignment."
        ),
    )
    parser.add_argument(
        "--alignment-max-auto-z-trim",
        type=float,
        default=DEFAULT_ALIGNMENT_MAX_AUTO_Z_TRIM_M,
        help="Largest ArUco-recommended absolute z trim (metres) that may be auto-applied.",
    )
    parser.add_argument(
        "--pretrial-calibrate",
        choices=("prompt", "always", "skip"),
        default="prompt",
        help=(
            "Run a short pre-trial ArUco recalibration step before starting the harness.\n"
            "'prompt' asks the operator, 'always' runs it automatically, 'skip' leaves existing files alone.\n"
            "The step cleans previous capture artifacts and regenerates marker_map and extrinsics under logs/aruco_hw_run_host."
        ),
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Run harness without blocking prompts (CI-friendly).",
    )
    parser.add_argument(
        "--preserve-alignment-during-trial",
        action="store_true",
        default=False,
        help=(
            "When set, do not temporarily disable ArUco alignment for the trial. "
            "This preserves auto-applied z trims during the trial run (use with care)."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help=(
            "Number of calibration+trial cycles to perform. Set to 1 (default) for a single"
            " run, or >1 to repeat. Set 0 for an infinite loop (CTRL-C to stop)."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help=(
            "Run a safe smoke-test: invoke the pretrial helper and dependency checks, "
            "then exit without starting the hardware harness. Useful for CI and local verification."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_arg_parser(add_help=True)
    args, _unknown = parser.parse_known_args(argv)
    return args


def main() -> None:
    args = parse_args()

    # Smoke-test path: run pretrial helper and basic dependency checks, then exit
    if getattr(args, "smoke_test", False):
        pretrial = getattr(args, "pretrial_calibrate", "prompt")
        proceed = True
        if pretrial and pretrial != "skip":
            if pretrial == "prompt":
                if getattr(args, "non_interactive", False):
                    print("Non-interactive: skipping pretrial prompt")
                    proceed = False
                else:
                    try:
                        print("Pre-trial marker update: prompt operator")
                        resp = input(
                            "Run pretrial marker update (will wipe logs/aruco_hw_run_host)? [Y/n]: "
                        ).strip().lower()
                        if resp not in ("", "y", "yes"):
                            proceed = False
                    except Exception:
                        proceed = False

            if proceed:
                print("Running pre-trial marker update (cleanup, resample, fit)")
                try:
                    script = os.path.join(os.path.dirname(__file__), "pretrial_marker_update.py")
                    cmd = [sys.executable, script, "--out-dir", "logs/aruco_hw_run_host", "--clean", "--max-frames", "40", "--rate", "4.0"]
                    subprocess.check_call(cmd)
                    print("Pre-trial marker update completed successfully")
                except subprocess.CalledProcessError as exc:
                    print(f"Pre-trial marker update failed: {exc}; continuing")
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"Pre-trial marker step error: {exc}; continuing")

        # Run sampler dependency check
        try:
            print("Running sampler dependency check (--check-only)")
            sampler = os.path.join(os.path.dirname(__file__), "sample_aruco_sampler.py")
            subprocess.check_call([sys.executable, sampler, "--check-only"])
            print("Sampler dependency check passed")
        except subprocess.CalledProcessError as exc:
            print(f"Sampler check failed: {exc}")
        except Exception as exc:
            print(f"Sampler check error: {exc}")

        print("Smoke test complete")
        return

    rclpy.init()
    node = ContactGatedWithArucoV2Harness(
        push_axis_height_z=args.push_axis_height_z,
        load_targets_n=args.load_targets_n,
        hold_duration_s=args.hold_duration_s,
        upper_force_limit_n=args.upper_force_limit_n,
        load_target_tolerance_n=args.load_target_tolerance_n,
        load_seek_step_m=args.load_seek_step_m,
        load_balance_step_m=args.load_balance_step_m,
        max_additional_load_travel_m=args.max_additional_load_travel_m,
        min_stage_advance_m=args.min_stage_advance_m,
        stage4_handoff_target_n=args.stage4_handoff_target_n,
        defer_k_lateral_until_after_precontact=args.defer_k_lateral_until_after_precontact,
        external_camera_trigger_file=args.external_camera_trigger_file,
        ignore_local_frame_ee_safety=args.ignore_local_frame_ee_safety,
        precontact_waypoint_timeout_s=args.precontact_waypoint_timeout_s,
        precontact_contact_threshold_n=args.precontact_contact_threshold_n,
        precontact_velocity_limit=args.precontact_velocity_limit,
        load_step_waypoint_timeout_s=args.load_step_waypoint_timeout_s,
        calibration_json_path=args.calibration_json,
        enable_ablation=args.ablate,
        n_repeats=args.repeats_n,
        k_lateral_npm=args.k_lateral_npm,
        k_lateral_case_values_npm=args.k_lateral_cases_npm,
        peak_load_case_values_n=args.peak_load_cases_n,
        spring_specimen=args.spring_specimen,
        spring_stiffness_npm=args.spring_stiffness_npm,
        alignment_source=args.alignment_source,
        alignment_policy=args.alignment_policy,
        alignment_max_auto_z_trim=args.alignment_max_auto_z_trim,
        non_interactive=args.non_interactive,
    )
    # Run repeated calibration+trial cycles (interactive detach/reattach)
    cycles = int(getattr(args, "cycles", 1) or 1)
    pretrial = getattr(args, "pretrial_calibrate", "prompt")

    results: List[Dict[str, Any]] = []
    cycle_idx = 0
    try:
        while True:
            cycle_idx += 1
            if cycles > 0 and cycle_idx > cycles:
                break

            # Pre-trial marker update (camera calibration) if requested
            if pretrial and pretrial != "skip":
                proceed = True
                if pretrial == "prompt":
                    try:
                        node.get_logger().info(f"Cycle {cycle_idx}: Pre-trial marker update: prompt operator")
                        resp = input("Run pretrial marker update (will wipe logs/aruco_hw_run_host)? [Y/n]: ").strip().lower()
                        if resp not in ("", "y", "yes"):
                            proceed = False
                    except Exception:
                        proceed = False

                if proceed:
                    node.get_logger().info(f"Cycle {cycle_idx}: Running pre-trial marker update (cleanup, resample, fit)")
                    try:
                        script = os.path.join(os.path.dirname(__file__), "pretrial_marker_update.py")
                        cmd = [sys.executable, script, "--out-dir", "logs/aruco_hw_run_host", "--clean", "--max-frames", "40", "--rate", "4.0"]
                        subprocess.check_call(cmd)
                        node.get_logger().info(f"Cycle {cycle_idx}: Pre-trial marker update completed successfully")
                    except subprocess.CalledProcessError as exc:
                        node.get_logger().warn(f"Cycle {cycle_idx}: Pre-trial marker update failed: {exc}; continuing without updated markers")
                    except Exception as exc:  # pragma: no cover - defensive
                        node.get_logger().warn(f"Cycle {cycle_idx}: Pre-trial marker step error: {exc}; continuing")

            # Run stage-0 robot calibration to prepare load targets
            ok, msg, info = node.stage0_calibration()
            stage0_record = {"stage": 0, "pass": ok, "message": msg, "info": info}
            results.append(stage0_record)
            node.get_logger().info(f"Stage 0: {msg} -> {ok}")
            if not ok:
                node.write_results(results)
                node.get_logger().warn(f"Cycle {cycle_idx}: Stage 0 failed; aborting cycles")
                break

            # Prompt operator to detach camera and stop host recorder
            max_cycles_text = "∞" if cycles == 0 else str(cycles)
            print(f"\n=== Cycle {cycle_idx} / {max_cycles_text} ===")
            print("Please detach the RealSense/ArUco camera and stop the host recorder now.")
            print("When ready, press ENTER to continue with the trial (or type 'auto' to publish alignment-accept topics).")
            if getattr(args, "non_interactive", False):
                node.get_logger().info("Non-interactive: auto-continuing without operator input")
                resp = ""
            else:
                try:
                    resp = input("  >> Continue? [ENTER] ").strip().lower()
                except Exception:
                    resp = ""

            if resp == "auto":
                try:
                    pub_cmd = (
                        "source /opt/ros/humble/setup.bash && source ws/install/setup.bash && source /workspaces/omx_ros2/.venv/bin/activate && "
                        "ros2 topic pub --once /spring_monitor/aruco_alignment/valid std_msgs/msg/Bool \"data: true\" && "
                        "ros2 topic pub --once /spring_monitor/aruco_alignment/robot1_z_trim std_msgs/msg/Float64 \"data: 0.0\" && "
                        "ros2 topic pub --once /spring_monitor/aruco_alignment/robot2_z_trim std_msgs/msg/Float64 \"data: 0.0\""
                    )
                    subprocess.check_call(["bash", "-lc", pub_cmd])
                except Exception as exc:
                    node.get_logger().warn(f"Cycle {cycle_idx}: auto publish failed: {exc}")

            # Temporarily skip ArUco blocking alignment for the trial unless
            # the operator requested to preserve alignment. When preserved,
            # any auto-applied z trims will remain active during the trial.
            orig_policy = node.alignment_policy
            orig_source = node.alignment_source
            if not getattr(args, "preserve_alignment_during_trial", False):
                node.alignment_policy = "warn"
                node.alignment_source = "off"

            try:
                # Run a single case/trial using the prepared calibration values
                trial_record = node._run_case(
                    case_name=node.current_case_name,
                    peak_load_n=node.current_case_peak_load_n,
                    load_targets_n=node.load_targets_n,
                    k_lateral_npm=(node.current_case_k_lateral_npm if hasattr(node, "current_case_k_lateral_npm") else None),
                )
                trial_record["cycle_index"] = cycle_idx
                results.append(trial_record)
                node.write_results(results)
                print(json.dumps(trial_record, indent=2))
            except Exception as exc:
                node.get_logger().warn(f"Cycle {cycle_idx}: trial failed: {exc}")
            finally:
                node.alignment_policy = orig_policy
                node.alignment_source = orig_source

            # Prompt operator to reattach and perform manual calibration
            print("Reattach the RealSense/ArUco camera now and then press ENTER to run manual calibration (pretrial).")
            if not getattr(args, "non_interactive", False):
                try:
                    input("  >> Reattached and ready? Press ENTER to run manual calibrate: ")
                except Exception:
                    pass

            # Run manual pretrial calibration (prompt) if requested
            if pretrial and pretrial != "skip":
                try:
                    script = os.path.join(os.path.dirname(__file__), "pretrial_marker_update.py")
                    cmd = [sys.executable, script, "--out-dir", "logs/aruco_hw_run_host", "--clean", "--max-frames", "40", "--rate", "4.0"]
                    subprocess.check_call(cmd)
                    node.get_logger().info(f"Cycle {cycle_idx}: Manual pretrial marker update completed successfully")
                except Exception as exc:
                    node.get_logger().warn(f"Cycle {cycle_idx}: Manual pretrial marker update failed: {exc}; continuing")

        print("All cycles complete")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()