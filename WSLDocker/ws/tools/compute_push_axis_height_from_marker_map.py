#!/usr/bin/env python3
"""Compute push-axis height (z) from a marker map JSON.

Prints the mean z (metres) of the spring-cap markers found in the marker
map. If marker IDs are known, pass them with --spring-ids (comma list).

Usage:
  python3 tools/compute_push_axis_height_from_marker_map.py
  python3 tools/compute_push_axis_height_from_marker_map.py --marker-map logs/aruco_hw_run_host/marker_map_computed.json --spring-ids 4,5
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple


def _extract_tz_from_value(v) -> Optional[float]:
    # v may be a dict with 'matrix' (4x4), or with 'tvec' list, or with 'rvec'/'tvec'
    if not isinstance(v, dict):
        return None
    if "matrix" in v:
        try:
            M = v["matrix"]
            tz = float(M[2][3])
            return tz
        except Exception:
            return None
    if "tvec" in v:
        try:
            t = v["tvec"]
            return float(t[2])
        except Exception:
            return None
    # older formats may put tvec at top-level list-like
    for key in ("pose", "translation", "t"):
        if key in v:
            try:
                t = v[key]
                return float(t[2])
            except Exception:
                pass
    return None


def find_spring_cap_zs_from_markermap(data: Dict) -> Tuple[List[float], List[str]]:
    tzs: List[float] = []
    found_labels: List[str] = []

    # Case A: top-level 'markers' list with labelled entries
    if isinstance(data.get("markers"), list):
        for entry in data.get("markers", []):
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label") or entry.get("name") or "").lower()
            if "spring" in label or "cap" in label:
                tz = _extract_tz_from_value(entry)
                if tz is not None and math.isfinite(tz):
                    tzs.append(tz)
                    found_labels.append(label or str(entry.get("marker_id") or entry.get("id") or "?"))
        if tzs:
            return tzs, found_labels

    # Case B: numeric-key dict mapping id -> entry
    # Prefer a small set of common spring-cap ids if present
    candidate_ids = [4, 5, 1, 3, 2, 0, 6]
    # normalize mapping keys
    numeric_map = {}
    for k, v in data.items():
        try:
            ik = int(k)
            numeric_map[ik] = v
        except Exception:
            # skip non-numeric keys
            continue

    if numeric_map:
        # if user-specified candidate ids exist, use them in order
        for cid in candidate_ids:
            if cid in numeric_map:
                tz = _extract_tz_from_value(numeric_map[cid])
                if tz is not None and math.isfinite(tz):
                    tzs.append(tz)
                    found_labels.append(str(cid))
        # fallback: if no candidates found, take any entries that look like spring caps
        if not tzs:
            for ik, v in numeric_map.items():
                tz = _extract_tz_from_value(v)
                if tz is not None and math.isfinite(tz):
                    tzs.append(tz)
                    found_labels.append(str(ik))

    return tzs, found_labels


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--marker-map", default="logs/aruco_hw_run_host/marker_map_computed.json")
    p.add_argument(
        "--spring-ids",
        default=None,
        help="Comma-separated marker ids for spring caps (e.g. 4,5). If provided, these ids are used when present.",
    )
    p.add_argument(
        "--metal-platform-id",
        default=None,
        help="Marker id (int) for the metal platform center to compute midpoint with spring-cap mean.",
    )
    p.add_argument(
        "--world-marker-id",
        default=None,
        help="Marker id (int) for the ground/world marker whose z should be considered world-zero.",
    )
    args = p.parse_args(argv)

    path = os.path.expanduser(args.marker_map)
    if not os.path.exists(path):
        print(f"Marker map not found: {path}")
        return 2

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - IO
        print(f"Failed to load marker map {path}: {exc}")
        return 2

    # If explicit spring ids requested, try them first
    tzs: List[float] = []
    labels: List[str] = []
    if args.spring_ids:
        try:
            ids = [int(x.strip()) for x in args.spring_ids.split(",") if x.strip()]
        except Exception:
            ids = []
        # if data is mapping from id->val
        if isinstance(data, dict) and ids:
            for sid in ids:
                v = data.get(str(sid)) or data.get(sid)
                if v is None:
                    continue
                tz = _extract_tz_from_value(v)
                if tz is not None and math.isfinite(tz):
                    tzs.append(tz)
                    labels.append(str(sid))

    # if none found yet, try label-based or candidate-based inference
    if not tzs:
        tzs, labels = find_spring_cap_zs_from_markermap(data)

    if not tzs:
        print("No spring-cap z values found in marker map. Provide --spring-ids or run pretrial capture.")
        return 3

    # If a metal platform id was provided, compute midpoint between metal platform z
    # and the mean spring-cap z values. Optionally rebase both to the world marker
    # (subtract world marker z) when --world-marker-id is provided.
    metal_z: Optional[float] = None
    if args.metal_platform_id:
        try:
            mid = int(args.metal_platform_id)
        except Exception:
            mid = None
        if mid is not None and isinstance(data, dict):
            v = data.get(str(mid)) or data.get(mid)
            if v is not None:
                metal_z_cand = _extract_tz_from_value(v)
                if metal_z_cand is not None and math.isfinite(metal_z_cand):
                    metal_z = metal_z_cand

    # fallback: if markers list contains a labelled 'metal' or 'platform', use it
    if metal_z is None and isinstance(data.get("markers"), list):
        for entry in data.get("markers", []):
            label = str(entry.get("label") or entry.get("name") or "").lower()
            if "metal" in label or "platform" in label:
                tzm = _extract_tz_from_value(entry)
                if tzm is not None and math.isfinite(tzm):
                    metal_z = tzm
                    break

    mean_z = sum(tzs) / len(tzs)
    if metal_z is not None:
        # if world marker specified, attempt to rebase
        world_z: Optional[float] = None
        if args.world_marker_id:
            try:
                wid = int(args.world_marker_id)
            except Exception:
                wid = None
            if wid is not None and isinstance(data, dict):
                v = data.get(str(wid)) or data.get(wid)
                if v is not None:
                    wz = _extract_tz_from_value(v)
                    if wz is not None and math.isfinite(wz):
                        world_z = wz

        if world_z is not None:
            metal_z_world = metal_z - world_z
            spring_mean_world = mean_z - world_z
            midpoint = (metal_z_world + spring_mean_world) / 2.0
            print(f"metal_platform_z (raw): {metal_z:.6f}")
            print(f"spring_mean_z (raw): {mean_z:.6f}")
            print(f"world_marker_z (raw): {world_z:.6f}")
            print(f"push_axis_height_z: {midpoint:.6f}")
            srcs = [f"world:{args.world_marker_id}", f"metal:{args.metal_platform_id or 'found_by_label'}"] + [f"spring:{l}" for l in labels]
            print(f"derived_from_markers: {','.join(srcs)}")
            print()
            print("Suggested harness command (shell):")
            print(f"OMX_HARNESS_V3_PUSH_AXIS_HEIGHT_Z={midpoint:.6f} python3 tools/hardware_harness_contact_gated_load.py --push-axis-height-z {midpoint:.6f} --repeats-n 1")
            print()
            print("Or pass the value directly to the harness via --push-axis-height-z")
            return 0

        # no world marker found, behave as before
        midpoint = (metal_z + mean_z) / 2.0
        print(f"metal_platform_z: {metal_z:.6f}")
        print(f"spring_mean_z: {mean_z:.6f}")
        print(f"push_axis_height_z: {midpoint:.6f}")
        srcs = [f"metal:{args.metal_platform_id or 'found_by_label'}"] + [f"spring:{l}" for l in labels]
        print(f"derived_from_markers: {','.join(srcs)}")
        print()
        print("Suggested harness command (shell):")
        print(f"OMX_HARNESS_V3_PUSH_AXIS_HEIGHT_Z={midpoint:.6f} python3 tools/hardware_harness_contact_gated_load.py --push-axis-height-z {midpoint:.6f} --repeats-n 1")
        print()
        print("Or pass the value directly to the harness via --push-axis-height-z")
        return 0

    # fallback behaviour: no metal marker found, return spring mean
    print(f"push_axis_height_z: {mean_z:.6f}")
    print(f"derived_from_markers: {','.join(labels)}")
    print()
    print("Suggested harness command (shell):")
    print(f"OMX_HARNESS_V3_PUSH_AXIS_HEIGHT_Z={mean_z:.6f} python3 tools/hardware_harness_contact_gated_load.py --push-axis-height-z {mean_z:.6f} --repeats-n 1")
    print()
    print("Or pass the value directly to the harness via --push-axis-height-z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
