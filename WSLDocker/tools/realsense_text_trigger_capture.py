"""
Trigger RealSense color video capture from a text command file.

This recorder watches a single text file for `START` / `STOP` / `QUIT`
commands and records color-only AVI clips from an attached Intel RealSense
camera. It supports a small readiness handshake via `--ready-file` and
writes a per-clip sidecar JSON into `--sidecar-dir` when provided.

The script is tolerant of missing hardware dependencies when run with
`--emulate` (useful for smoke testing on hosts without pyrealsense2).

Command file examples:
    START
    START name=trial_001
    START name=trial_001 duration=5
    STOP
    QUIT

Aliases:
    1 -> START
    0 -> STOP
    EXIT -> QUIT

The watcher ignores whatever is already in the command file when it
starts (it reads only new changes). The recorder will optionally write a
`READY` file to signal the orchestrator that it is prepared to accept
commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Lazy/hard imports kept out of module import-time so test runs that
# merely import repository modules don't fail when hardware libs are
# missing. Real hardware capture requires `pyrealsense2`, `numpy`, and
# `cv2` at runtime unless `--emulate` is used.
_HAVE_PYRS = False
_HAVE_CV2 = False
_HAVE_NUMPY = False

try:
    import numpy as _np  # type: ignore
    _HAVE_NUMPY = True
except Exception:
    _np = None  # type: ignore

try:
    import cv2 as _cv2  # type: ignore
    _HAVE_CV2 = True
except Exception:
    _cv2 = None  # type: ignore

try:
    import pyrealsense2 as _rs  # type: ignore
    _HAVE_PYRS = True
except Exception:
    _rs = None  # type: ignore


@dataclass
class CaptureCommand:
    action: str
    name: Optional[str] = None
    duration_s: Optional[float] = None


class CommandWatcher:
    def __init__(self, command_path: Path):
        self.command_path = command_path
        self._last_stamp = self._get_stamp()

    def _get_stamp(self) -> Optional[Tuple[int, int]]:
        if not self.command_path.exists():
            return None
        stat = self.command_path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def poll(self) -> Optional[str]:
        stamp = self._get_stamp()
        if stamp is None or stamp == self._last_stamp:
            return None

        self._last_stamp = stamp
        try:
            text = self.command_path.read_text(encoding="utf-8")
        except Exception:
            return None
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not lines:
            return None
        return lines[-1]


class RealSenseTextTriggerRecorder:
    def __init__(
        self,
        output_dir: Path,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
        fourcc: str,
        extension: str,
        poll_period_s: float,
        start_timeout_s: float,
        ready_file: Optional[Path] = None,
        sidecar_dir: Optional[Path] = None,
        emulate: bool = False,
    ):
        self.output_dir = output_dir
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.extension = extension.lstrip(".")
        self.poll_period_s = poll_period_s
        self.start_timeout_s = start_timeout_s
        self.ready_file = ready_file
        self.sidecar_dir = sidecar_dir
        self.emulate = bool(emulate)

        self.pipeline = None
        self.writer = None
        self.output_path = None
        self.record_started_at = None
        self.record_duration_s = None
        self.frame_size = None

    def log(self, message: str) -> None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def _atomic_write(self, path: Path, content: str) -> None:
        # Write to a temporary file in the same directory and replace.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use mkstemp in the path's directory to ensure cross-device replace.
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except Exception:
                    pass
            try:
                os.replace(tmp, str(path))
            except Exception:
                # Best-effort cleanup if replace fails.
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                raise
        except Exception:
            # Ensure the temp file is removed on any error and re-raise.
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            raise

    def _write_ready(self) -> None:
        if not self.ready_file:
            return
        try:
            self._atomic_write(self.ready_file, "READY\n")
        except Exception as exc:
            self.log(f"Could not write READY file {self.ready_file}: {exc}")

    def _sanitize_name(self, requested_name: Optional[str]) -> str:
        if requested_name:
            cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", requested_name.strip())
            cleaned = cleaned.strip("._")
            if cleaned:
                return cleaned
        return dt.datetime.now().strftime("capture_%Y%m%d_%H%M%S")

    def _open_pipeline(self):
        if self.emulate:
            return "EMULATED"
        if not _HAVE_PYRS:
            raise RuntimeError("pyrealsense2 is unavailable; run with --emulate to use software-only mode")

        rs = _rs  # type: ignore
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = pipeline.start(config)
        device = profile.get_device()
        has_rgb = False
        for sensor in device.query_sensors():
            try:
                if sensor.supports(rs.camera_info.name) and sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    has_rgb = True
                    break
            except Exception:
                continue

        if not has_rgb:
            pipeline.stop()
            raise RuntimeError("Connected RealSense device does not expose an RGB Camera sensor")

        return pipeline

    def _wait_for_color_image(self, pipeline) -> 'Any':
        # Return a numpy-array compatible image for the first available color frame.
        if pipeline == "EMULATED":
            if not _HAVE_NUMPY:
                raise RuntimeError("Emulation requires numpy to synthesize frames")
            # Synthesized mid-gray frame
            return _np.full((self.height, self.width, 3), 128, dtype=_np.uint8)

        rs = _rs  # type: ignore
        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if color_frame:
                import numpy as _np_local  # type: ignore

                return _np_local.asanyarray(color_frame.get_data())
        raise RuntimeError(f"Timed out waiting for color frames after {self.start_timeout_s:.1f} seconds")

    def _open_writer(self, output_path: Path, image) -> 'Any':
        if not _HAVE_CV2:
            raise RuntimeError("OpenCV (cv2) is required to encode output files")
        height, width = image.shape[:2]
        fourcc_value = _cv2.VideoWriter_fourcc(*self.fourcc)
        writer = _cv2.VideoWriter(str(output_path), fourcc_value, self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open cv2.VideoWriter for '{output_path}'. Try a different fourcc or extension.")
        self.frame_size = (width, height)
        return writer

    def _write_sidecar(self, output_path: Path, command: CaptureCommand) -> None:
        if not self.sidecar_dir:
            return
        payload = {
            "name": command.name,
            "requested_duration_s": command.duration_s,
            "started_at": dt.datetime.utcnow().isoformat() + "Z",
            "output_file": str(output_path.name),
            "frame_size": self.frame_size,
            "fps": self.fps,
        }
        sidecar_path = Path(self.sidecar_dir) / (output_path.name + ".json")
        try:
            Path(self.sidecar_dir).mkdir(parents=True, exist_ok=True)
            self._atomic_write(sidecar_path, json.dumps(payload) + "\n")
        except Exception as exc:
            self.log(f"Could not write sidecar JSON {sidecar_path}: {exc}")

    def start_recording(self, command: CaptureCommand) -> None:
        if self.writer is not None:
            self.stop_recording("Restarting recording on new START command")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_name = self._sanitize_name(command.name)
        output_path = self.output_dir / f"{output_name}.{self.extension}"

        pipeline = None
        writer = None
        try:
            pipeline = self._open_pipeline()
            image = self._wait_for_color_image(pipeline)
            writer = self._open_writer(output_path, image)
            writer.write(image)
        except Exception:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if pipeline is not None and pipeline != "EMULATED":
                try:
                    pipeline.stop()
                except Exception:
                    pass
            raise

        self.pipeline = pipeline
        self.writer = writer
        self.output_path = output_path
        self.record_started_at = time.monotonic()
        self.record_duration_s = command.duration_s

        duration_text = "until STOP" if command.duration_s is None else f"for {command.duration_s:.2f} seconds"
        self.log(f"Recording started: {output_path} ({duration_text})")
        # Write sidecar metadata as soon as the recording starts so the
        # orchestrator/analysis pipeline can see clip metadata even if the
        # process crashes later.
        try:
            self._write_sidecar(output_path, command)
        except Exception:
            pass

    def stop_recording(self, reason: str) -> None:
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
        if self.pipeline is not None and self.pipeline != "EMULATED":
            try:
                self.pipeline.stop()
            except Exception:
                pass

        if self.output_path is not None:
            self.log(f"Recording stopped: {self.output_path} ({reason})")
        else:
            self.log(f"Recording stopped ({reason})")

        self.pipeline = None
        self.writer = None
        self.output_path = None
        self.record_started_at = None
        self.record_duration_s = None
        self.frame_size = None

    def capture_one_frame(self) -> None:
        if self.pipeline is None or self.writer is None:
            return

        if self.pipeline == "EMULATED":
            img = _np.full((self.height, self.width, 3), int(time.time() * 10) % 255, dtype=_np.uint8)
            if self.frame_size is not None and (img.shape[1], img.shape[0]) != self.frame_size:
                import cv2 as _cv2_local  # type: ignore

                img = _cv2_local.resize(img, self.frame_size, interpolation=_cv2_local.INTER_AREA)
            self.writer.write(img)
            return

        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        import numpy as _np_local  # type: ignore

        image = _np_local.asanyarray(color_frame.get_data())
        if self.frame_size is not None and (image.shape[1], image.shape[0]) != self.frame_size:
            import cv2 as _cv2_local  # type: ignore

            image = _cv2_local.resize(image, self.frame_size, interpolation=_cv2_local.INTER_AREA)

        self.writer.write(image)

    def duration_elapsed(self) -> bool:
        if self.record_started_at is None or self.record_duration_s is None:
            return False
        return (time.monotonic() - self.record_started_at) >= self.record_duration_s


def parse_command(line: str) -> CaptureCommand:
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        raise ValueError(f"Could not parse command line '{line}': {exc}")

    if not tokens:
        raise ValueError("Command line is empty")

    verb = tokens[0].strip().upper()
    if verb in {"START", "RECORD", "1"}:
        action = "start"
    elif verb in {"STOP", "0"}:
        action = "stop"
    elif verb in {"QUIT", "EXIT"}:
        action = "quit"
    else:
        raise ValueError(f"Unknown command '{tokens[0]}'")

    if action != "start":
        return CaptureCommand(action=action)

    name = None
    duration_s = None

    for token in tokens[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"name", "file", "filename"}:
                name = value
            elif key in {"duration", "seconds", "sec"}:
                duration_s = float(value)
            else:
                raise ValueError(f"Unknown START option '{key}'")
        elif name is None:
            name = token
        elif duration_s is None:
            duration_s = float(token)
        else:
            raise ValueError(f"Unexpected extra token '{token}'")

    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration must be greater than zero")

    return CaptureCommand(action=action, name=name, duration_s=duration_s)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch a text file and trigger RealSense color video capture")
    parser.add_argument("command_file", help="Path to the text file that receives START/STOP/QUIT commands")
    parser.add_argument("--output-dir", default="captures", help="Directory for saved video files")
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number")
    parser.add_argument("--width", type=int, default=1280, help="Color stream width")
    parser.add_argument("--height", type=int, default=720, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Color stream frame rate")
    parser.add_argument("--fourcc", default="MJPG", help="OpenCV fourcc code, e.g. MJPG, XVID, mp4v")
    parser.add_argument("--extension", default="avi", help="Output file extension without dot")
    parser.add_argument("--poll-period", type=float, default=0.2, help="Idle polling interval in seconds")
    parser.add_argument(
        "--start-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for the first valid color frame when a recording starts",
    )
    parser.add_argument("--ready-file", default=None, help="Optional path to write a READY file when the recorder is ready")
    parser.add_argument("--sidecar-dir", default=None, help="Optional directory to write per-clip JSON sidecars")
    parser.add_argument("--emulate", action="store_true", help="Do not require pyrealsense2; synthesize frames in software")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if len(args.fourcc) != 4:
        raise SystemExit("--fourcc must be exactly 4 characters")

    # If not emulating, ensure the heavy deps exist before proceeding.
    if not args.emulate:
        if not (_HAVE_PYRS and _HAVE_CV2 and _HAVE_NUMPY):
            missing = []
            if not _HAVE_PYRS:
                missing.append("pyrealsense2")
            if not _HAVE_CV2:
                missing.append("opencv-python (cv2)")
            if not _HAVE_NUMPY:
                missing.append("numpy")
            raise SystemExit(f"Missing dependencies: {', '.join(missing)}. Or run with --emulate for software-only mode.")

    command_path = Path(args.command_file)
    recorder = RealSenseTextTriggerRecorder(
        output_dir=Path(args.output_dir),
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        extension=args.extension,
        poll_period_s=args.poll_period,
        start_timeout_s=args.start_timeout,
        ready_file=Path(args.ready_file) if args.ready_file else None,
        sidecar_dir=Path(args.sidecar_dir) if args.sidecar_dir else None,
        emulate=args.emulate,
    )
    watcher = CommandWatcher(command_path)

    recorder.log(f"Watching command file: {command_path}")
    recorder.log("Write START, STOP, or QUIT into the file and save it to trigger an action")

    # Announce readiness if requested.
    if recorder.ready_file is not None:
        try:
            recorder._write_ready()
            recorder.log(f"Wrote READY file: {recorder.ready_file}")
        except Exception as exc:
            recorder.log(f"Could not write READY file: {exc}")

    try:
        while True:
            raw_command = watcher.poll()
            if raw_command is not None:
                try:
                    command = parse_command(raw_command)
                    recorder.log(f"Command received: {raw_command}")
                    if command.action == "start":
                        recorder.start_recording(command)
                    elif command.action == "stop":
                        if recorder.writer is not None:
                            recorder.stop_recording("STOP command")
                        else:
                            recorder.log("STOP ignored because no recording is active")
                    elif command.action == "quit":
                        if recorder.writer is not None:
                            recorder.stop_recording("QUIT command")
                        recorder.log("Exiting")
                        return 0
                except Exception as exc:
                    recorder.log(f"Command error: {exc}")

            if recorder.writer is None:
                time.sleep(recorder.poll_period_s)
                continue

            try:
                recorder.capture_one_frame()
            except Exception as exc:
                recorder.stop_recording(f"capture error: {exc}")
                continue

            if recorder.duration_elapsed():
                recorder.stop_recording("duration elapsed")
    except KeyboardInterrupt:
        if recorder.writer is not None:
            recorder.stop_recording("KeyboardInterrupt")
        recorder.log("Interrupted")
        return 0


if __name__ == "__main__":
    sys.exit(main())
