#!/usr/bin/env python3
"""Finalize OMX run data into an enriched harness snapshot CSV.

The raw ``single_arm_harness_v3_snapshot.csv`` is never modified.  For every
raw snapshot found beneath ``--run-folder`` this script writes a sibling
``single_arm_harness_v3_snapshot_enriched.csv`` that retains all original
columns and appends:

* high-rate motor Present Current feedback and current in A;
* pre-safety controller current request;
* Dynamixel Goal Current register readback (no controller C++ patch);
* nearest-sample timing/staleness diagnostics;
* MOVE_FORWARD / CONTACT / ABORT / STOP state and relative times;
* nearest *written* RealSense video frame when the Windows camera sync CSV is
  visible (normally /mnt/d/Paper1PushExpt inside WSL, if mounted into the
  devcontainer).

The harness and motor logger share the ROS clock on hardware, so motor samples
are joined directly on ``ros_time_s`` <-> ``rx_ros_time_s``.  Camera frames use
periodic UDP SYNC anchors to fit Linux wall/ROS-system time against the Windows
monotonic clock.  Camera columns remain empty with ``camera_sync_available=0``
when the Windows directory is not visible; the robot/motor/event enrichment is
still complete.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
MOTOR_JOINTS = ("joint1", "joint2", "joint3", "joint4")


def finite_float(value, default=float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def parse_kv(payload: str) -> Dict[str, str]:
    return {k: v for k, v in KV_RE.findall(payload or "")}


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def nearest_index(sorted_times: Sequence[float], target: float) -> Optional[int]:
    if not sorted_times or not math.isfinite(target):
        return None
    i = bisect.bisect_left(sorted_times, target)
    if i <= 0:
        return 0
    if i >= len(sorted_times):
        return len(sorted_times) - 1
    return i - 1 if abs(sorted_times[i - 1] - target) <= abs(sorted_times[i] - target) else i


@dataclass
class Event:
    name: str
    wall_s: float
    mono_ns: int
    payload: str
    kv: Dict[str, str]


def load_events(path: Optional[Path], run_id: str) -> List[Event]:
    if path is None or not path.exists():
        return []
    _, rows = read_csv(path)
    out: List[Event] = []
    seen = set()
    for row in rows:
        name = (row.get("event") or "").strip().upper()
        if not name:
            continue
        row_run = (row.get("run_id") or "").strip()
        payload = row.get("payload") or ""
        if run_id and row_run and row_run != run_id:
            continue
        if run_id and not row_run and run_id not in payload:
            continue
        key = (name, payload)
        if key in seen:
            continue
        seen.add(key)
        out.append(Event(
            name=name,
            wall_s=finite_float(row.get("linux_wall_time_s")),
            mono_ns=int(finite_float(row.get("linux_monotonic_ns"), 0.0)),
            payload=payload,
            kv=parse_kv(payload),
        ))
    return out


def first_event(events: Sequence[Event], name: str) -> Optional[Event]:
    name = name.upper()
    return next((e for e in events if e.name == name), None)


def event_ros_time(event: Optional[Event]) -> Tuple[float, str]:
    if event is None:
        return float("nan"), "missing"
    for key in ("contact_ros_time_s", "ros_log_time_s"):
        x = finite_float(event.kv.get(key))
        if math.isfinite(x):
            return x, key
    # Hardware runs use ROS system time; bridge wall time is the fallback for
    # ABORT/STOP events that originate outside a ROS message callback.
    if math.isfinite(event.wall_s):
        return event.wall_s, "linux_wall_time_s"
    return float("nan"), "missing"


def load_motor_feedback(path: Optional[Path]) -> Tuple[List[float], List[Dict[str, str]]]:
    if path is None or not path.exists():
        return [], []
    _, rows = read_csv(path)
    pairs = []
    for row in rows:
        if (row.get("record_type") or "") != "feedback":
            continue
        t = finite_float(row.get("rx_ros_time_s"))
        if math.isfinite(t):
            pairs.append((t, row))
    pairs.sort(key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def affine_fit(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, List[float]]:
    n = len(xs)
    if n < 2:
        raise ValueError("at least two clock anchors are required")
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var <= 0:
        raise ValueError("clock anchors have zero time span")
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    b = my - a * mx
    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    return a, b, residuals


def robust_affine_fit(xs: List[float], ys: List[float]) -> Tuple[float, float, List[float], int]:
    a0, b0, r0 = affine_fit(xs, ys)
    med = statistics.median(r0)
    mad = statistics.median([abs(r - med) for r in r0])
    if mad <= 1e-12:
        keep = [True] * len(xs)
    else:
        sigma = 1.4826 * mad
        keep = [abs(r - med) <= 4.0 * sigma for r in r0]
        if sum(keep) < 2:
            keep = [True] * len(xs)
    kx = [x for x, k in zip(xs, keep) if k]
    ky = [y for y, k in zip(ys, keep) if k]
    a, b, _ = affine_fit(kx, ky)
    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    return a, b, residuals, sum(keep)


@dataclass
class CameraMap:
    path: Path
    frame_win_s: List[float]
    frame_rows: List[Dict[str, str]]
    linux_per_win_a: float
    linux_per_win_b: float
    fit_p95_ms: float
    fit_max_ms: float
    anchors_total: int
    anchors_used: int

    def windows_time_from_linux(self, linux_s: float) -> float:
        return (linux_s - self.linux_per_win_b) / self.linux_per_win_a


def camera_csv_has_run(path: Path, run_id: str) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("record_type") or "") != "event":
                    continue
                payload = row.get("payload") or ""
                if run_id in payload:
                    return True
    except OSError:
        return False
    return False


def discover_camera_sync(search_dir: Optional[Path], run_id: str) -> Optional[Path]:
    if search_dir is None or not search_dir.exists():
        return None
    candidates = sorted(search_dir.glob("realsense_dual_sync_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if camera_csv_has_run(p, run_id):
            return p
    return None


def load_camera_map(path: Optional[Path], run_id: str) -> Optional[CameraMap]:
    if path is None or not path.exists():
        return None
    _, rows = read_csv(path)
    frames: List[Tuple[float, Dict[str, str]]] = []
    xs: List[float] = []
    ys: List[float] = []
    seen_sync = set()
    for row in rows:
        rt = row.get("record_type") or ""
        if rt == "frame":
            ns = finite_float(row.get("host_monotonic_ns"))
            if math.isfinite(ns):
                frames.append((ns * 1e-9, row))
        elif rt == "event" and (row.get("event") or "").upper() == "SYNC":
            payload = row.get("payload") or ""
            if run_id and run_id not in payload:
                continue
            if payload in seen_sync:
                continue
            seen_sync.add(payload)
            kv = parse_kv(payload)
            linux = finite_float(kv.get("linux_wall_time_s"))
            win_ns = finite_float(row.get("host_monotonic_ns"))
            if math.isfinite(linux) and math.isfinite(win_ns):
                xs.append(win_ns * 1e-9)
                ys.append(linux)
    if len(frames) == 0 or len(xs) < 2:
        return None
    frames.sort(key=lambda p: p[0])
    a, b, residuals, used = robust_affine_fit(xs, ys)
    abs_ms = sorted(abs(r) * 1000.0 for r in residuals)
    p95_i = min(len(abs_ms) - 1, max(0, math.ceil(0.95 * len(abs_ms)) - 1))
    return CameraMap(
        path=path,
        frame_win_s=[x for x, _ in frames],
        frame_rows=[r for _, r in frames],
        linux_per_win_a=a,
        linux_per_win_b=b,
        fit_p95_ms=abs_ms[p95_i],
        fit_max_ms=max(abs_ms),
        anchors_total=len(xs),
        anchors_used=used,
    )


def append_columns() -> List[str]:
    cols = [
        "telemetry_match_ros_time_s", "motor_telem_dt_ms",
        "pre_safety_request_age_s", "goal_register_age_s",
    ]
    for j in MOTOR_JOINTS:
        cols += [
            f"{j}_present_current_raw_highrate",
            f"{j}_present_current_A_highrate",
            f"{j}_torque_est_Nm_highrate",
            f"{j}_pre_safety_request_raw",
            f"{j}_pre_safety_request_A",
            f"{j}_request_model_equivalent_Nm",
            f"{j}_goal_current_register_raw",
            f"{j}_goal_current_register_A",
        ]
    cols += [
        "run_status",
        "move_forward_started", "contact_established", "abort_detected",
        "u2d2_loss_after_contact", "normal_move_end",
        "t_from_move_forward_start_s", "t_from_contact_s", "t_from_abort_s",
        "move_forward_start_time_source", "contact_time_source", "abort_time_source",
        "abort_reason", "stop_reason", "ignored_precontact_dxl_lines",
        "camera_sync_available", "camera_video_frame_index", "camera_capture_index",
        "camera_frame_dt_ms", "camera_top_timestamp_ms", "camera_side_timestamp_ms",
        "camera_clock_fit_abs_residual_p95_ms", "camera_clock_fit_abs_residual_max_ms",
    ]
    return cols


def enrich_one(snapshot: Path, motor_csv: Optional[Path], events_csv: Optional[Path],
               camera_map: Optional[CameraMap], run_id: str) -> Tuple[Path, Dict[str, object]]:
    original_cols, rows = read_csv(snapshot)
    motor_times, motor_rows = load_motor_feedback(motor_csv)
    events = load_events(events_csv, run_id)

    start_ev = first_event(events, "MOVE_FORWARD_START")
    contact_ev = first_event(events, "CONTACT")
    abort_ev = first_event(events, "ABORT")
    end_ev = first_event(events, "MOVE_FORWARD_END")
    stop_ev = first_event(events, "STOP")
    start_t, start_src = event_ros_time(start_ev)
    contact_t, contact_src = event_ros_time(contact_ev)
    abort_t, abort_src = event_ros_time(abort_ev)
    end_t, _ = event_ros_time(end_ev)

    abort_reason = abort_ev.kv.get("reason", "") if abort_ev else ""
    stop_reason = stop_ev.kv.get("reason", "") if stop_ev else ""
    ignored_pre = contact_ev.kv.get("ignored_precontact_dxl_lines", "") if contact_ev else ""
    u2d2_reasons = {
        "persistent_dynamixel_read_failure", "dynamixel_rebooting",
        "stale_joint_feedback", "ros2_control_process_exited",
    }
    u2d2_abort = abort_reason in u2d2_reasons
    if abort_ev is not None:
        run_status = "aborted_post_contact"
    elif end_ev is not None:
        run_status = "normal_completed"
    elif stop_reason == "precontact_run_terminated":
        run_status = "precontact_terminated"
    else:
        run_status = "incomplete_or_unknown"

    extras = append_columns()
    fieldnames = original_cols + [c for c in extras if c not in original_cols]
    out_path = snapshot.with_name("single_arm_harness_v3_snapshot_enriched.csv")

    valid_ros_rows = 0
    motor_matched = 0
    camera_matched = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            t = finite_float(row.get("ros_time_s"))
            if not math.isfinite(t):
                t = finite_float(row.get("timestamp"))
            if math.isfinite(t):
                valid_ros_rows += 1

            mi = nearest_index(motor_times, t)
            if mi is not None:
                mr = motor_rows[mi]
                mt = motor_times[mi]
                out["telemetry_match_ros_time_s"] = f"{mt:.9f}"
                out["motor_telem_dt_ms"] = f"{(mt - t) * 1000.0:.6f}" if math.isfinite(t) else ""
                out["pre_safety_request_age_s"] = mr.get("pre_safety_request_age_s", "")
                out["goal_register_age_s"] = mr.get("goal_register_age_s", "")
                for j in MOTOR_JOINTS:
                    mapping = {
                        f"{j}_present_current_raw_highrate": f"{j}_present_current_raw",
                        f"{j}_present_current_A_highrate": f"{j}_present_current_A",
                        f"{j}_torque_est_Nm_highrate": f"{j}_torque_est_Nm",
                        f"{j}_pre_safety_request_raw": f"{j}_pre_safety_request_raw_latest",
                        f"{j}_pre_safety_request_A": f"{j}_pre_safety_request_A_latest",
                        f"{j}_request_model_equivalent_Nm": f"{j}_request_model_equivalent_Nm_latest",
                        f"{j}_goal_current_register_raw": f"{j}_goal_current_register_raw_latest",
                        f"{j}_goal_current_register_A": f"{j}_goal_current_register_A_latest",
                    }
                    for dst, src in mapping.items():
                        out[dst] = mr.get(src, "")
                motor_matched += 1
            else:
                for c in extras[:4 + 8 * len(MOTOR_JOINTS)]:
                    out.setdefault(c, "")

            out["run_status"] = run_status
            out["move_forward_started"] = int(math.isfinite(start_t) and math.isfinite(t) and t >= start_t)
            out["contact_established"] = int(math.isfinite(contact_t) and math.isfinite(t) and t >= contact_t)
            out["abort_detected"] = int(math.isfinite(abort_t) and math.isfinite(t) and t >= abort_t)
            out["u2d2_loss_after_contact"] = int(u2d2_abort and math.isfinite(abort_t) and math.isfinite(t) and t >= abort_t)
            out["normal_move_end"] = int(end_ev is not None and math.isfinite(end_t) and math.isfinite(t) and t >= end_t)
            out["t_from_move_forward_start_s"] = f"{t - start_t:.9f}" if math.isfinite(t) and math.isfinite(start_t) else ""
            out["t_from_contact_s"] = f"{t - contact_t:.9f}" if math.isfinite(t) and math.isfinite(contact_t) else ""
            out["t_from_abort_s"] = f"{t - abort_t:.9f}" if math.isfinite(t) and math.isfinite(abort_t) else ""
            out["move_forward_start_time_source"] = start_src
            out["contact_time_source"] = contact_src
            out["abort_time_source"] = abort_src
            out["abort_reason"] = abort_reason
            out["stop_reason"] = stop_reason
            out["ignored_precontact_dxl_lines"] = ignored_pre

            out["camera_sync_available"] = 0
            for c in (
                "camera_video_frame_index", "camera_capture_index", "camera_frame_dt_ms",
                "camera_top_timestamp_ms", "camera_side_timestamp_ms",
                "camera_clock_fit_abs_residual_p95_ms", "camera_clock_fit_abs_residual_max_ms",
            ):
                out[c] = ""
            if camera_map is not None and math.isfinite(t):
                win_t = camera_map.windows_time_from_linux(t)
                ci = nearest_index(camera_map.frame_win_s, win_t)
                if ci is not None:
                    fr = camera_map.frame_rows[ci]
                    out["camera_sync_available"] = 1
                    out["camera_video_frame_index"] = fr.get("video_frame_index", fr.get("frame_index", ""))
                    out["camera_capture_index"] = fr.get("capture_index", "")
                    out["camera_frame_dt_ms"] = f"{(camera_map.frame_win_s[ci] - win_t) * 1000.0:.6f}"
                    out["camera_top_timestamp_ms"] = fr.get("top_timestamp_ms", "")
                    out["camera_side_timestamp_ms"] = fr.get("side_timestamp_ms", "")
                    out["camera_clock_fit_abs_residual_p95_ms"] = f"{camera_map.fit_p95_ms:.6f}"
                    out["camera_clock_fit_abs_residual_max_ms"] = f"{camera_map.fit_max_ms:.6f}"
                    camera_matched += 1
            w.writerow(out)

    summary = {
        "raw_snapshot": str(snapshot),
        "enriched_snapshot": str(out_path),
        "rows": len(rows),
        "rows_with_finite_time": valid_ros_rows,
        "motor_feedback_rows_available": len(motor_rows),
        "rows_motor_matched": motor_matched,
        "event_log": str(events_csv) if events_csv else None,
        "events_found": [e.name for e in events],
        "run_status": run_status,
        "abort_reason": abort_reason or None,
        "stop_reason": stop_reason or None,
        "camera_sync_csv": str(camera_map.path) if camera_map else None,
        "rows_camera_matched": camera_matched,
        "camera_clock_fit_p95_ms": camera_map.fit_p95_ms if camera_map else None,
        "camera_clock_fit_max_ms": camera_map.fit_max_ms if camera_map else None,
    }
    out_path.with_suffix(".metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path, summary


def main() -> int:
    p = argparse.ArgumentParser(description="Create canonical enriched OMX harness snapshot(s).")
    p.add_argument("--run-folder", type=Path, required=True)
    p.add_argument("--run-id", default="")
    p.add_argument("--motor-csv", type=Path, default=None)
    p.add_argument("--event-log", type=Path, default=None)
    p.add_argument("--camera-sync-csv", type=Path, default=None)
    p.add_argument("--camera-search-dir", type=Path, default=Path("/mnt/d/Paper1PushExpt"))
    p.add_argument("--camera-wait-s", type=float, default=15.0,
                   help="When the Windows log mount exists, wait this long for its final sync CSV to appear.")
    p.add_argument("--strict-motor", action="store_true", help="Fail if high-rate motor telemetry is missing.")
    a = p.parse_args()

    run_folder = a.run_folder.resolve()
    run_id = a.run_id or run_folder.name
    snapshots = sorted(run_folder.rglob("single_arm_harness_v3_snapshot.csv"))
    if not snapshots:
        print(f"[ENRICH] No harness snapshot found beneath {run_folder}")
        return 3

    motor = a.motor_csv or (run_folder / "motor_current_torque_telemetry.csv")
    if not motor.exists():
        if a.strict_motor:
            print(f"[ENRICH] ERROR: motor telemetry missing: {motor}")
            return 4
        print(f"[ENRICH] WARNING: motor telemetry missing: {motor}")
        motor = None

    event_log = a.event_log or (run_folder / "rgb_sync_events.csv")
    if not event_log.exists():
        print(f"[ENRICH] WARNING: local event log missing: {event_log}")
        event_log = None

    camera_csv = a.camera_sync_csv
    if camera_csv is None and a.camera_search_dir.exists():
        deadline = time.monotonic() + max(0.0, a.camera_wait_s)
        while True:
            camera_csv = discover_camera_sync(a.camera_search_dir, run_id)
            if camera_csv is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.25)
    camera_map = None
    if camera_csv is not None:
        try:
            camera_map = load_camera_map(camera_csv, run_id)
        except Exception as exc:
            print(f"[ENRICH] WARNING: camera sync unusable ({camera_csv}): {exc}")
    if camera_map is None:
        print("[ENRICH] Camera sync not visible/usable; camera columns will be marked unavailable.")
    else:
        print(f"[ENRICH] Camera sync: {camera_map.path}")
        print(f"[ENRICH] Clock fit p95={camera_map.fit_p95_ms:.3f} ms max={camera_map.fit_max_ms:.3f} ms")

    outputs = []
    summaries = []
    for snapshot in snapshots:
        out, summary = enrich_one(snapshot, motor, event_log, camera_map, run_id)
        outputs.append(out)
        summaries.append(summary)
        print(f"[ENRICH] Wrote {out}")

    # One-run/one-snapshot convenience copy at RUN_FOLDER root.  The original
    # raw snapshot remains untouched in the harness subdirectory.
    if len(outputs) == 1:
        canonical = run_folder / "single_arm_harness_v3_snapshot_enriched.csv"
        if outputs[0] != canonical:
            shutil.copy2(outputs[0], canonical)
            meta_src = outputs[0].with_suffix(".metadata.json")
            if meta_src.exists():
                shutil.copy2(meta_src, canonical.with_suffix(".metadata.json"))
        print(f"[ENRICH] Canonical run snapshot: {canonical}")

    index = {
        "run_id": run_id,
        "run_folder": str(run_folder),
        "enriched_snapshots": [str(p) for p in outputs],
        "summaries": summaries,
    }
    (run_folder / "enriched_snapshot_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
