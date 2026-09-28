#!/usr/bin/env python3
"""Align AIC7 loop timestamps to WRITTEN video frames using UDP clock probes.

No ROS, RealSense, OpenCV, or third-party dependency. Run after both programs
have closed. Output retains every original robot column. Camera times are
host paired-frame receipt times, not synchronized exposure timestamps.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def align(run_dir):
    run_dir = Path(run_dir).resolve()
    complete = json.loads((run_dir / 'aic7_complete.json').read_text(encoding='utf-8-sig'))
    if not complete.get('capture_relative'):
        raise ValueError('Robot run has no Capture data (see exit_code in aic7_complete.json)')
    capture = (run_dir / complete['capture_relative']).resolve()
    if run_dir not in capture.parents:
        raise ValueError('Capture path escapes run directory')
    probes = read_rows(run_dir / 'robot_clock_sync.csv')
    if len(probes) < 3:
        raise ValueError('At least three valid clock probes are required')
    # One minimum-RTT observation per five-second Linux-time bin.
    base = int(probes[0]['robot_midpoint_ns'])
    bins = {}
    for p in probes:
        p = {k: int(v) for k, v in p.items()}
        if not (0 < p['rtt_ns'] <= 100_000_000):
            continue
        bucket = (p['robot_midpoint_ns'] - base) // 5_000_000_000
        if bucket not in bins or p['rtt_ns'] < bins[bucket]['rtt_ns']:
            bins[bucket] = p
    anchors = sorted(bins.values(), key=lambda p: p['robot_midpoint_ns'])
    if not anchors:
        raise ValueError('No acceptable clock probes')
    times = [p['robot_midpoint_ns'] for p in anchors]
    sync_files = sorted(run_dir.glob('realsense_dual_sync_*.csv'))
    if len(sync_files) != 1:
        raise ValueError(f'Expected exactly one camera sync CSV, found {len(sync_files)}')
    camera_rows = read_rows(sync_files[0])
    session_id = complete['session_id']
    if not any(r['record_type'] == 'event' and f'session={session_id}' in r['payload'] for r in camera_rows):
        raise ValueError('Camera sidecar has no event for this AIC7 session')
    frames = [r for r in camera_rows if r['record_type'] == 'frame']
    frames.sort(key=lambda r: int(r['host_monotonic_ns']))
    if not frames:
        raise ValueError('Camera sidecar contains no written frames')
    frame_times = [int(r['host_monotonic_ns']) for r in frames]
    if any(b <= a for a, b in zip(frame_times, frame_times[1:])):
        raise ValueError('Camera host timestamps are not strictly increasing')
    extra = ['camera_windows_monotonic_ns_est', 'camera_clock_extrapolated',
             'camera_clock_anchor_gap_s', 'camera_clock_probe_half_rtt_ms',
             'camera_clock_quality', 'camera_frame_covered',
             'camera_video_frame_index', 'camera_video_frame_zero_based',
             'camera_capture_index', 'camera_pair_host_monotonic_ns',
             'camera_frame_minus_robot_ms', 'camera_top_timestamp_ms',
             'camera_side_timestamp_ms']
    output = capture / 'data_camera_aligned.csv'
    temporary = output.with_suffix('.csv.tmp')
    rows = covered = extrapolated = weak = 0
    max_distance_ms = 0.0
    with (capture / 'data.csv').open(newline='', encoding='utf-8') as source, temporary.open('w', newline='', encoding='utf-8') as destination:
        reader = csv.DictReader(source)
        if not reader.fieldnames or 'robot_monotonic_ns' not in reader.fieldnames:
            raise ValueError('data.csv lacks AIC7 robot_monotonic_ns; old AIC6 captures cannot be absolutely aligned from t alone')
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames + extra)
        writer.writeheader()
        for row in reader:
            robot_ns = int(row['robot_monotonic_ns'])
            right = bisect.bisect_left(times, robot_ns)
            is_extrapolated = robot_ns < times[0] or robot_ns > times[-1]
            if right == 0:
                a = b = anchors[0]
            elif right == len(anchors):
                a = b = anchors[-1]
            else:
                a, b = anchors[right - 1], anchors[right]
            span = b['robot_midpoint_ns'] - a['robot_midpoint_ns']
            weight = (robot_ns - a['robot_midpoint_ns']) / span if span else 0.0
            offset = a['offset_windows_minus_robot_ns'] + weight * (b['offset_windows_minus_robot_ns'] - a['offset_windows_minus_robot_ns'])
            windows_ns = robot_ns + round(offset)
            half_rtt_ms = max(a['rtt_ns'], b['rtt_ns']) / 2e6
            age_s = min(abs(robot_ns - a['robot_midpoint_ns']), abs(robot_ns - b['robot_midpoint_ns'])) / 1e9
            quality = 'review' if span > 15e9 or age_s > 10 or half_rtt_ms > 10 else 'nominal'
            row.update(dict(camera_windows_monotonic_ns_est=windows_ns,
                            camera_clock_extrapolated=int(is_extrapolated),
                            camera_clock_anchor_gap_s=f'{span / 1e9:.6f}',
                            camera_clock_probe_half_rtt_ms=f'{half_rtt_ms:.6f}',
                            camera_clock_quality=quality))
            in_range = frame_times[0] <= windows_ns <= frame_times[-1]
            row['camera_frame_covered'] = int(in_range)
            if in_range:
                idx = bisect.bisect_left(frame_times, windows_ns)
                candidates = [k for k in (idx - 1, idx) if 0 <= k < len(frames)]
                nearest = min(candidates, key=lambda k: abs(frame_times[k] - windows_ns))
                frame = frames[nearest]
                delta_ms = (frame_times[nearest] - windows_ns) / 1e6
                row.update(dict(camera_video_frame_index=frame['video_frame_index'],
                                camera_video_frame_zero_based=int(frame['video_frame_index']) - 1,
                                camera_capture_index=frame['capture_index'],
                                camera_pair_host_monotonic_ns=frame['host_monotonic_ns'],
                                camera_frame_minus_robot_ms=f'{delta_ms:.6f}',
                                camera_top_timestamp_ms=frame['top_timestamp_ms'],
                                camera_side_timestamp_ms=frame['side_timestamp_ms']))
                max_distance_ms = max(max_distance_ms, abs(delta_ms))
                covered += 1
            writer.writerow(row)
            rows += 1
            extrapolated += int(is_extrapolated)
            weak += int(quality == 'review')
    temporary.replace(output)
    summary = dict(session_id=session_id, robot_exit_code=complete['exit_code'],
                   robot_rows=rows, rows_with_camera_coverage=covered,
                   clock_extrapolated_rows=extrapolated, clock_review_rows=weak,
                   valid_clock_probes=len(probes), selected_clock_anchors=len(anchors),
                   written_paired_frames=len(frames),
                   max_nearest_pair_distance_ms=max_distance_ms,
                   clock_method='Piecewise linear interpolation of minimum-RTT offsets from five-second bins; constant offset outside anchors',
                   limits=['Host software timestamp alignment, not camera hardware synchronization.',
                           'Half RTT describes a single probe midpoint uncertainty under a constant offset; it excludes camera/USB buffering, interpolation drift and sensor read latency.',
                           'Top and side frames have independent device timestamps; paired frames need not have simultaneous exposures.',
                           'Video indices are based on written_records; dropped pairs do not shift the mapping.',
                           'Rows outside camera coverage have blank video indices. Review large frame distance, extrapolation and clock-quality flags.'])
    (run_dir / 'camera_alignment_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Aligned data: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    align(args.run_dir)


if __name__ == '__main__':
    main()
