#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence


DEFAULT_TARGET_DEVICES = ("0403:6014", "8086:0b3a")
USBIPD_LINE_RE = re.compile(r"^\s*(\S+)\s+([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\s+(.*\S)?\s*$")
LSUSB_LINE_RE = re.compile(
    r"^Bus\s+(\d{3})\s+Device\s+(\d{3}):\s+ID\s+([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\s*(.*)$"
)


@dataclass(frozen=True)
class UsbDeviceMatch:
    location: str
    vidpid: str
    description: str


def normalize_vidpid(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{4}", normalized):
        raise ValueError(f"Invalid VID:PID '{value}'. Expected format like 0403:6014.")
    return normalized


def parse_usbipd_list(output: str, targets: Iterable[str]) -> List[UsbDeviceMatch]:
    target_set = {normalize_vidpid(target) for target in targets}
    matches: List[UsbDeviceMatch] = []

    for line in output.splitlines():
        found = USBIPD_LINE_RE.match(line)
        if not found:
            continue

        busid, vidpid, description = found.groups()
        normalized_vidpid = normalize_vidpid(vidpid)
        if normalized_vidpid in target_set:
            matches.append(UsbDeviceMatch(busid, normalized_vidpid, (description or "").strip()))

    return matches


def parse_lsusb_list(output: str, targets: Iterable[str]) -> List[UsbDeviceMatch]:
    target_set = {normalize_vidpid(target) for target in targets}
    matches: List[UsbDeviceMatch] = []

    for line in output.splitlines():
        found = LSUSB_LINE_RE.match(line.strip())
        if not found:
            continue

        bus, device, vidpid, description = found.groups()
        normalized_vidpid = normalize_vidpid(vidpid)
        if normalized_vidpid in target_set:
            matches.append(UsbDeviceMatch(f"{bus}:{device}", normalized_vidpid, description.strip()))

    return matches


def is_wsl() -> bool:
    release = platform.release().lower()
    return bool(
        os.environ.get("WSL_DISTRO_NAME")
        or os.environ.get("WSL_INTEROP")
        or release.endswith("microsoft-standard-wsl2")
        or "microsoft" in release
    )


def find_command(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def run_command(args: Sequence[str], suppress_stderr: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if suppress_stderr else subprocess.PIPE,
    )


def print_matches(prefix: str, matches: Sequence[UsbDeviceMatch]) -> None:
    for match in matches:
        if match.description:
            print(f"{prefix}{match.vidpid} @ {match.location} :: {match.description}")
        else:
            print(f"{prefix}{match.vidpid} @ {match.location}")


def attach_with_usbipd(targets: Sequence[str], wsl_distro: str, dry_run: bool) -> int:
    usbipd = find_command(("usbipd", "usbipd.exe"))
    if not usbipd:
        print("usbipd was not found. On Windows/WSL, install usbipd-win first.", file=sys.stderr)
        return 1

    listed = run_command((usbipd, "list"))
    matches = parse_usbipd_list(listed.stdout, targets)
    if not matches:
        print("No matching usbipd devices found for the requested VID:PID list.", file=sys.stderr)
        return 1

    print_matches("Found ", matches)
    for match in matches:
        bind_cmd = (usbipd, "bind", "--busid", match.location)
        attach_cmd = (usbipd, "attach", "--busid", match.location, "--wsl", wsl_distro)

        if dry_run:
            print("DRY RUN:", " ".join(bind_cmd))
            print("DRY RUN:", " ".join(attach_cmd))
            continue

        run_command(bind_cmd, suppress_stderr=True, check=False)
        run_command(attach_cmd)

    print("All specified USB devices attached via usbipd.")
    return 0


def explain_linux_equivalent(targets: Sequence[str]) -> int:
    print("Linux detected. No usbipd attach step is required.")
    print("Expose devices directly to the container and prefer /dev/serial/by-id paths.")

    lsusb = find_command(("lsusb",))
    if lsusb:
        listed = run_command((lsusb,))
        matches = parse_lsusb_list(listed.stdout, targets)
        if matches:
            print_matches("Detected ", matches)
        else:
            print("No matching USB devices were found in lsusb output.")
    else:
        print("lsusb is not available; skipping USB inventory.")

    serial_by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    tty_devices = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    if serial_by_id:
        print("Serial by-id paths:")
        for path in serial_by_id:
            print(f"  {path}")
    if tty_devices:
        print("TTY device paths:")
        for path in tty_devices:
            print(f"  {path}")

    print("Recommended Docker flags:")
    print("  --privileged")
    print("  -v /dev/serial:/dev/serial")
    for path in tty_devices[:4]:
        print(f"  --device={path}")

    return 0


def explain_macos_equivalent() -> int:
    print("macOS detected.")
    print("Docker Desktop does not offer a reliable generic USB passthrough path like usbipd.")
    print("Recommended options:")
    print("  1. Use the Linux/devcontainer path on a host with direct USB device access.")
    print("  2. Use a Linux VM with USB passthrough.")
    print("  3. Run the hardware-connected workflow on Windows+WSL2 with usbipd or on native Linux.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-platform USB preparation helper for FTDI serial converters and RealSense devices."
    )
    parser.add_argument(
        "--device",
        dest="devices",
        action="append",
        default=None,
        help="VID:PID to target. Repeat for multiple devices. Defaults to 0403:6014 and 8086:0b3a.",
    )
    parser.add_argument(
        "--wsl",
        default="docker-desktop",
        help="WSL target for usbipd attach. Default: docker-desktop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without changing device attachment state.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        targets = [normalize_vidpid(value) for value in (args.devices or list(DEFAULT_TARGET_DEVICES))]
    except ValueError as error:
        parser.error(str(error))

    system_name = platform.system().lower()
    usbipd = find_command(("usbipd", "usbipd.exe"))

    if system_name == "windows" or (is_wsl() and usbipd):
        return attach_with_usbipd(targets, args.wsl, args.dry_run)
    if system_name == "linux":
        return explain_linux_equivalent(targets)
    if system_name == "darwin":
        return explain_macos_equivalent()

    print(f"Unsupported platform '{platform.system()}'.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())