#!/usr/bin/env python3
"""Low-latency dual-RealSense RGB recorder for Open Manipulator X spring experiments.

Both cameras start recording immediately when this program starts.  An
optional UDP listener accepts a later ``CONTACT`` event from WSL/Docker and
records the nearest paired video frame; it never gates recording.  ``STOP``
is also accepted, but Ctrl+C or ``q`` remains the normal stop mechanism.

Only colour streams are captured.  Frames from the two cameras are acquired
in one process and written as a paired item so a queue overflow drops both
frames together rather than silently desynchronising the views.  A small
``camera_sync.csv`` sidecar stores frame timestamps and event anchors; it is
negligible compared with the two RGB videos and enables post-hoc drift/offset
correction without adding depth-stream storage or capture-time computer vision.

Example (Windows PowerShell)::

    py realsense_dual_rgb_recorder.py --top-serial 1234 --side-serial 5678 \
        --output-dir D:\\Paper1PushExpt\\recorded_videos --udp-port 5006

The WSL/Docker sender may send ``CONTACT <run_id> <robot_timestamp>`` to
``host.docker.internal:5006``.  UDP is optional; recording continues if no
packet arrives.  The existing ArUco pose stream can continue using its own
port (for example 5005).

Requirements::

    py -m pip install numpy opencv-python pyrealsense2
"""

from __future__ import annotations

import argparse
import csv
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_OUTPUT_DIR = Path("recorded_videos")
DEFAULT_TOP_SERIAL = "348522071053"
DEFAULT_SIDE_SERIAL = "347622073030"

stop_event = threading.Event()


def request_stop(signum=None, frame=None):
    stop_event.set()


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


@dataclass
class PairFrame:
    index: int
    top: np.ndarray
    side: np.ndarray
    host_monotonic_ns: int
    top_timestamp_ms: float
    side_timestamp_ms: float


@dataclass
class UdpEvent:
    event: str
    host_monotonic_ns: int
    host_wall_time_s: float
    payload: str


class PairWriterThread(threading.Thread):
    """Write paired RGB frames asynchronously with one bounded queue."""

    def __init__(
        self,
        output_top: Path,
        output_side: Path,
        top_size: Tuple[int, int],
        side_size: Tuple[int, int],
        fps: float,
        frame_queue: queue.Queue,
        codec: str,
    ):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.error: Optional[BaseException] = None
        self.frames_written = 0
        # Authoritative MP4 index -> captured PairFrame map. Capture indices are
        # not equivalent to video indices when queue overflow drops a pair.
        # Store timing/index metadata only.  Never retain PairFrame here:
        # PairFrame contains two full RGB numpy arrays and would make memory
        # usage grow by ~5.3 MiB for every written 1280x720 pair.
        self.written_records: List[Tuple[int, int, int, float, float]] = []
        if len(codec) != 4:
            raise ValueError("--codec must be a four-character OpenCV codec")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.top_writer = cv2.VideoWriter(str(output_top), fourcc, fps, top_size)
        self.side_writer = cv2.VideoWriter(str(output_side), fourcc, fps, side_size)
        if not self.top_writer.isOpened() or not self.side_writer.isOpened():
            self.top_writer.release()
            self.side_writer.release()
            raise RuntimeError("Could not open one or both RGB video writers")

    def run(self):
        try:
            while not stop_event.is_set() or not self.frame_queue.empty():
                try:
                    item: PairFrame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self.top_writer.write(item.top)
                    self.side_writer.write(item.side)
                    self.frames_written += 1
                    self.written_records.append((
                        self.frames_written,
                        item.index,
                        item.host_monotonic_ns,
                        item.top_timestamp_ms,
                        item.side_timestamp_ms,
                    ))
                finally:
                    self.frame_queue.task_done()
        except BaseException as exc:
            self.error = exc
            stop_event.set()
        finally:
            self.top_writer.release()
            self.side_writer.release()


class UdpEventThread(threading.Thread):
    """Listen for optional CONTACT/STOP packets without blocking capture."""

    def __init__(self, bind_host: str, port: int, events: List[UdpEvent], lock: threading.Lock):
        super().__init__(daemon=True)
        self.bind_host = bind_host
        self.port = int(port)
        self.events = events
        self.lock = lock
        self.sock: Optional[socket.socket] = None
        self.error: Optional[BaseException] = None

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.bind_host, self.port))
            self.sock.settimeout(0.25)
            print(f"[INFO] Optional UDP event listener: {self.bind_host}:{self.port}")
            while not stop_event.is_set():
                try:
                    raw, _address = self.sock.recvfrom(4096)
                except socket.timeout:
                    continue
                payload = raw.decode("utf-8", errors="replace").strip()
                if not payload:
                    continue
                event = payload.split()[0].upper()
                now_ns = time.monotonic_ns()
                record = UdpEvent(
                    event=event,
                    host_monotonic_ns=now_ns,
                    host_wall_time_s=time.time(),
                    payload=payload,
                )
                with self.lock:
                    self.events.append(record)
                print(f"[UDP] {payload}")
                if event == "PING":
                    reply = f"PONG windows_monotonic_ns={now_ns} windows_wall_time_s={record.host_wall_time_s:.9f}"
                    try:
                        self.sock.sendto(reply.encode("utf-8"), _address)
                    except OSError as exc:
                        print(f"[WARN] PONG reply failed: {exc}", file=sys.stderr)
                # CONTACT and ABORT are timestamp markers only. Only STOP closes
                # the files, allowing a configurable post-abort video tail.
                if event == "STOP":
                    stop_event.set()
        except BaseException as exc:
            self.error = exc
            # UDP is optional: a bind failure must not stop RGB capture.
            print(f"[WARN] UDP listener unavailable: {exc}", file=sys.stderr)
        finally:
            if self.sock is not None:
                self.sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record paired top/side RealSense RGB videos.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List connected RealSense devices and exit.",
    )
    parser.add_argument("--top-serial", default=DEFAULT_TOP_SERIAL, help="RealSense serial number for the top camera.")
    parser.add_argument("--side-serial", default=DEFAULT_SIDE_SERIAL, help="RealSense serial number for the side camera.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--side-width", type=int, default=None)
    parser.add_argument("--side-height", type=int, default=None)
    parser.add_argument("--side-fps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--udp-bind-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=5006)
    parser.add_argument("--no-udp", action="store_true", help="Disable the optional UDP event listener.")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def list_devices() -> List[str]:
    context = rs.context()
    devices = context.query_devices()
    serials = []
    for device in devices:
        try:
            serials.append(device.get_info(rs.camera_info.serial_number))
        except RuntimeError:
            continue
    return serials


def print_devices() -> int:
    """Print connected RealSense serial numbers for camera assignment."""
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        print("No RealSense devices detected.")
        return 1

    print("Connected RealSense devices:")
    for index, device in enumerate(devices, start=1):
        def _info(kind) -> str:
            try:
                return device.get_info(kind)
            except RuntimeError:
                return "unknown"

        serial = _info(rs.camera_info.serial_number)
        name = _info(rs.camera_info.name)
        firmware = _info(rs.camera_info.firmware_version)
        usb = _info(rs.camera_info.usb_type_descriptor)
        print(f"  [{index}] serial={serial}  name={name}  firmware={firmware}  usb={usb}")
    return 0


def choose_serials(top_serial: Optional[str], side_serial: Optional[str]) -> Tuple[str, str]:
    available = list_devices()
    if top_serial and side_serial:
        if top_serial == side_serial:
            raise ValueError("--top-serial and --side-serial must identify different cameras")
        missing = [serial for serial in (top_serial, side_serial) if serial not in available]
        if missing:
            raise RuntimeError(f"Requested serials not detected: {missing}. Available: {available}")
        return top_serial, side_serial
    if len(available) < 2:
        raise RuntimeError(f"Two RealSense cameras are required; detected serials: {available}")
    top = top_serial or available[0]
    remaining = [serial for serial in available if serial != top]
    side = side_serial or remaining[0]
    if top not in available or side not in available:
        raise RuntimeError(f"Requested serials not detected. Available: {available}")
    return top, side


def start_pipeline(serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def nearest_frame(records: Sequence[Tuple[int, int, float, float]], target_ns: int) -> Optional[int]:
    """Return the paired frame index nearest an event timestamp."""
    if not records:
        return None
    return min(records, key=lambda pair: abs(pair[1] - target_ns))[0]


def main() -> int:
    args = parse_args()
    if args.list_devices:
        return print_devices()

    if min(args.width, args.height, args.fps, args.queue_size) <= 0:
        print("Width, height, FPS, and queue size must be positive.", file=sys.stderr)
        return 2
    side_width = args.side_width or args.width
    side_height = args.side_height or args.height
    side_fps = args.side_fps or args.fps
    if min(side_width, side_height, side_fps) <= 0:
        print("Side-camera dimensions and FPS must be positive.", file=sys.stderr)
        return 2

    top_serial, side_serial = choose_serials(args.top_serial, args.side_serial)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_top = args.output_dir / f"realsense_top_{args.width}x{args.height}_{timestamp}.mp4"
    output_side = args.output_dir / f"realsense_side_{side_width}x{side_height}_{timestamp}.mp4"
    sync_path = args.output_dir / f"realsense_dual_sync_{timestamp}.csv"

    events: List[UdpEvent] = []
    event_lock = threading.Lock()
    udp_thread = None
    if not args.no_udp:
        udp_thread = UdpEventThread(args.udp_bind_host, args.udp_port, events, event_lock)
        udp_thread.start()

    top_pipeline = side_pipeline = None
    writer_thread = None
    frame_queue: queue.Queue = queue.Queue(maxsize=args.queue_size)
    paired_records: List[Tuple[int, int, float, float]] = []
    dropped_pairs = 0
    captured_pairs = 0
    started_wall = time.time()
    started_mono = time.monotonic_ns()

    try:
        print(f"[INFO] Starting top camera {top_serial} immediately")
        print(f"[INFO] Starting side camera {side_serial} immediately")
        top_pipeline = start_pipeline(top_serial, args.width, args.height, args.fps)
        side_pipeline = start_pipeline(side_serial, side_width, side_height, side_fps)
        writer_thread = PairWriterThread(
            output_top,
            output_side,
            (args.width, args.height),
            (side_width, side_height),
            float(min(args.fps, side_fps)),
            frame_queue,
            args.codec,
        )
        writer_thread.start()
        print(f"[INFO] Top RGB:  {output_top.resolve()}")
        print(f"[INFO] Side RGB: {output_side.resolve()}")
        print("[INFO] Recording now; UDP CONTACT is optional. Press q or Ctrl+C to stop.")

        while not stop_event.is_set():
            top_frames = top_pipeline.wait_for_frames(timeout_ms=5000)
            side_frames = side_pipeline.wait_for_frames(timeout_ms=5000)
            top_color = top_frames.get_color_frame()
            side_color = side_frames.get_color_frame()
            if not top_color or not side_color:
                continue
            host_ns = time.monotonic_ns()
            item = PairFrame(
                index=captured_pairs + 1,
                top=np.asanyarray(top_color.get_data()).copy(),
                side=np.asanyarray(side_color.get_data()).copy(),
                host_monotonic_ns=host_ns,
                top_timestamp_ms=float(top_color.get_timestamp()),
                side_timestamp_ms=float(side_color.get_timestamp()),
            )
            captured_pairs += 1
            paired_records.append((item.index, host_ns, item.top_timestamp_ms, item.side_timestamp_ms))
            try:
                frame_queue.put_nowait(item)
            except queue.Full:
                # Drop the complete pair to keep both videos aligned.
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    pass
                dropped_pairs += 1
                try:
                    frame_queue.put_nowait(item)
                except queue.Full:
                    dropped_pairs += 1

            if writer_thread.error is not None:
                raise RuntimeError(f"Writer failed: {writer_thread.error}")

            if not args.no_preview:
                top_preview = cv2.resize(item.top, (640, 360))
                side_preview = cv2.resize(item.side, (640, 360))
                preview = np.hstack((top_preview, side_preview))
                cv2.imshow("Open Manipulator X dual RGB recording: top | side", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        stop_event.set()
    finally:
        stop_event.set()
        if top_pipeline is not None:
            try:
                top_pipeline.stop()
            except RuntimeError:
                pass
        if side_pipeline is not None:
            try:
                side_pipeline.stop()
            except RuntimeError:
                pass
        if writer_thread is not None:
            writer_thread.join(timeout=15)
        if udp_thread is not None:
            udp_thread.join(timeout=1)
        cv2.destroyAllWindows()

    written_records: List[Tuple[int, int, int, float, float]] = (
        list(writer_thread.written_records) if writer_thread is not None else []
    )
    written_timing = [
        (video_index, host_ns, top_ts, side_ts)
        for video_index, capture_index, host_ns, top_ts, side_ts in written_records
    ]

    with sync_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "record_type",
            "video_frame_index",
            "capture_index",
            "host_monotonic_ns",
            "host_wall_time_s",
            "top_timestamp_ms",
            "side_timestamp_ms",
            "event",
            "payload",
        ])
        for video_index, capture_index, host_ns, top_ts, side_ts in written_records:
            writer.writerow([
                "frame",
                video_index,
                capture_index,
                host_ns,
                started_wall + (host_ns - started_mono) / 1e9,
                top_ts,
                side_ts,
                "",
                "",
            ])
        with event_lock:
            for event in events:
                frame_index = nearest_frame(written_timing, event.host_monotonic_ns)
                writer.writerow([
                    "event",
                    frame_index if frame_index is not None else "",
                    "",
                    event.host_monotonic_ns,
                    event.host_wall_time_s,
                    "",
                    "",
                    event.event,
                    event.payload,
                ])

  
    written = writer_thread.frames_written if writer_thread else 0
    # Compute total wall‑clock duration of the capture session.
    duration_cap = time.time() - started_wall
    print("\n[INFO] Recording stopped")
    print(f"[INFO] Captured duration: {duration_cap}")
    print(f"[INFO] Captured paired frames: {captured_pairs}")
    print(f"[INFO] Written paired frames:  {written}")
    print(f"[INFO] Dropped paired frames:  {dropped_pairs}")
    print(f"[INFO] Sync metadata: {sync_path.resolve()}")
    return 0 if writer_thread is None or writer_thread.error is None else 1



if __name__ == "__main__":
    raise SystemExit(main())
