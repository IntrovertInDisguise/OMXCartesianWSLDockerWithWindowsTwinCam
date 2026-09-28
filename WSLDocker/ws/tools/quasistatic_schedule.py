#!/usr/bin/env python3
"""Build and install deterministic monotonic quasistatic compression schedules.

The hardware runner uses this module instead of shell floating-point loops so
stage targets are reproducible and the final target is clipped exactly to the
requested end position.  It has no ROS dependency and is intentionally easy to
unit test outside the devcontainer.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence


STAGE_KEYS = (
    "compression_stage_x",
    "compression_stage_wait_durations",
)


def _finite_decimal(value: float | str, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if "." in text else text + ".0"


def _yaml_vector(values: Iterable[Decimal]) -> str:
    return "[" + ", ".join(_format_decimal(value) for value in values) + "]"


@dataclass(frozen=True)
class QuasistaticSchedule:
    start_x: Decimal
    end_x: Decimal
    step_m: Decimal
    move_duration_s: Decimal
    hold_duration_s: Decimal
    targets: tuple[Decimal, ...]

    @property
    def stage_count(self) -> int:
        return len(self.targets)

    @property
    def waits(self) -> tuple[Decimal, ...]:
        return (self.hold_duration_s,) * self.stage_count

    @property
    def nominal_duration_s(self) -> Decimal:
        return Decimal(self.stage_count) * (
            self.move_duration_s + self.hold_duration_s
        )

    @property
    def targets_yaml(self) -> str:
        return _yaml_vector(self.targets)

    @property
    def waits_yaml(self) -> str:
        return _yaml_vector(self.waits)


def build_schedule(
    start_x: float | str,
    end_x: float | str,
    step_m: float | str,
    move_duration_s: float | str,
    hold_duration_s: float | str,
) -> QuasistaticSchedule:
    start = _finite_decimal(start_x, "start_x")
    end = _finite_decimal(end_x, "end_x")
    step = _finite_decimal(step_m, "step_m")
    move = _finite_decimal(move_duration_s, "move_duration_s")
    hold = _finite_decimal(hold_duration_s, "hold_duration_s")

    if end <= start:
        raise ValueError(f"end_x must be greater than start_x, got {start} -> {end}")
    if step <= 0:
        raise ValueError(f"step_m must be > 0, got {step}")
    if move <= 0:
        raise ValueError(f"move_duration_s must be > 0, got {move}")
    if hold <= 0:
        raise ValueError(f"hold_duration_s must be > 0, got {hold}")

    stroke = end - start
    stage_count = int(math.ceil(stroke / step))
    targets = tuple(min(start + step * index, end) for index in range(1, stage_count + 1))

    if not targets or targets[-1] != end:
        raise RuntimeError("internal schedule error: final target does not equal end_x")
    if any(a >= b for a, b in zip((start,) + targets[:-1], targets)):
        raise RuntimeError("internal schedule error: targets are not strictly increasing")

    return QuasistaticSchedule(
        start_x=start,
        end_x=end,
        step_m=step,
        move_duration_s=move,
        hold_duration_s=hold,
        targets=targets,
    )


def _replace_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"^(?P<indent>\s*){re.escape(key)}:\s*[^#\n]*(?P<comment>\s*#.*)?$",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"missing required YAML scalar: {key}")
    comment = match.group("comment") or ""
    replacement = f"{match.group('indent')}{key}: {value}{comment}"
    return text[: match.start()] + replacement + text[match.end() :]


def _remove_stage_lines(text: str) -> str:
    for key in STAGE_KEYS:
        text = re.sub(
            rf"^[ \t]*{re.escape(key)}:[ \t]*\[[^\n]*\][ \t]*(?:#.*)?\n?",
            "",
            text,
            flags=re.MULTILINE,
        )
    return text


def render_controller_yaml(
    text: str,
    schedule: QuasistaticSchedule | None,
) -> str:
    """Return controller YAML with staged compression enabled or cleared.

    Stage arrays are kept on single lines because ROS 2's parameter-file parser
    must infer them as non-empty ``double_array`` values.
    """
    text = _remove_stage_lines(text)
    if schedule is None:
        return text

    text = _replace_scalar(text, "move_duration", _format_decimal(schedule.move_duration_s))
    text = _replace_scalar(text, "wait_duration", _format_decimal(schedule.hold_duration_s))
    text = _replace_scalar(text, "return_to_start_between_stages", "false")
    text = _replace_scalar(text, "hold_final_compression", "true")

    anchor = re.search(
        r"^(?P<indent>\s*)initial_wait_duration:\s*[^\n]*$",
        text,
        re.MULTILINE,
    )
    if not anchor:
        raise ValueError("missing YAML insertion anchor: initial_wait_duration")
    indent = anchor.group("indent")
    stage_block = (
        f"\n{indent}compression_stage_x: {schedule.targets_yaml}"
        f"\n{indent}compression_stage_wait_durations: {schedule.waits_yaml}"
    )
    return text[: anchor.end()] + stage_block + text[anchor.end() :]


def update_controller_yaml(path: Path, schedule: QuasistaticSchedule | None) -> None:
    original = path.read_text(encoding="utf-8")
    updated = render_controller_yaml(original, schedule)
    path.write_text(updated, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-x")
    parser.add_argument("--end-x")
    parser.add_argument("--step-m")
    parser.add_argument("--move-duration-s")
    parser.add_argument("--hold-duration-s")
    parser.add_argument("--update-yaml", type=Path)
    parser.add_argument(
        "--clear-yaml-stages",
        action="store_true",
        help="Remove previously injected stage arrays from --update-yaml.",
    )
    parser.add_argument(
        "--print-lines",
        action="store_true",
        help="Print stage count, nominal duration, targets YAML, and waits YAML.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.clear_yaml_stages:
        if args.update_yaml is None:
            parser.error("--clear-yaml-stages requires --update-yaml")
        update_controller_yaml(args.update_yaml, None)
        return 0

    required = {
        "--start-x": args.start_x,
        "--end-x": args.end_x,
        "--step-m": args.step_m,
        "--move-duration-s": args.move_duration_s,
        "--hold-duration-s": args.hold_duration_s,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    schedule = build_schedule(
        args.start_x,
        args.end_x,
        args.step_m,
        args.move_duration_s,
        args.hold_duration_s,
    )
    if args.update_yaml is not None:
        update_controller_yaml(args.update_yaml, schedule)
    if args.print_lines:
        print(schedule.stage_count)
        print(_format_decimal(schedule.nominal_duration_s))
        print(schedule.targets_yaml)
        print(schedule.waits_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
