#!/usr/bin/env python3
"""Estimate marker-size scale from ArUco detections and regenerate overlays.

Computes the ratio between detected corner-to-center pixel distances (from
each detection's `rvec`/`tvec`) and the predicted corner distances (from the
computed `marker_map` + fitted extrinsics). Uses the median ratio to scale the
marker length and rewrites overlay images to `--out-dir`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from statistics import median
from typing import Dict, List

import cv2
import numpy as np


def load_marker_map(path: str) -> Dict[int, np.ndarray]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[int, np.ndarray] = {}
    for k, v in data.items():
        mid = int(k)
        if isinstance(v, dict) and "matrix" in v:
            M = np.asarray(v["matrix"], dtype=float)
        elif isinstance(v, dict) and ("rvec" in v or "tvec" in v):
            rvec = v.get("rvec", [0.0, 0.0, 0.0])
            tvec = v.get("tvec", [0.0, 0.0, 0.0])
            R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
            t = np.asarray(tvec, dtype=float).reshape(3)
            M = np.eye(4, dtype=float)
            M[:3, :3] = R
            M[:3, 3] = t
        else:
            raise RuntimeError(f"Unrecognized marker-map format for id {k}")
        out[mid] = M
    return out


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records-jsonl", default="logs/aruco_hw_run_host/aruco_records.jsonl")
    p.add_argument("--camera-extrinsics", default="logs/aruco_hw_run_host/camera_extrinsics.json")
    p.add_argument("--marker-map", default="logs/aruco_hw_run_host/marker_map_computed.json")
    p.add_argument("--marker-length", type=float, default=0.05,
                   help="Current marker length used for projections (m)")
    p.add_argument("--out-dir", default="logs/aruco_hw_run_host/validation_scaled")
    p.add_argument("--max-overlays", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.camera_extrinsics, "r", encoding="utf-8") as fh:
        ce = json.load(fh)
    if "world_to_camera" in ce:
        T_world_to_cam = np.asarray(ce["world_to_camera"], dtype=float)
    else:
        T_cam_to_world = np.asarray(ce.get("camera_to_world"), dtype=float)
        T_world_to_cam = invert_homogeneous(T_cam_to_world)

    marker_map = load_marker_map(args.marker_map)

    half_pred = args.marker_length / 2.0
    corners_marker_pred = np.array([[-half_pred, half_pred, 0.0], [half_pred, half_pred, 0.0], [half_pred, -half_pred, 0.0], [-half_pred, -half_pred, 0.0]], dtype=float)

    ratios: List[float] = []
    per_marker = defaultdict(list)
    overlays_saved = 0

    with open(args.records_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cam_info = rec.get("camera_info", {})
            k = cam_info.get("k") or cam_info.get("K")
            d = cam_info.get("d", None)
            if k is None:
                continue
            K = np.array([[k[0], k[1], k[2]],[k[3], k[4], k[5]],[k[6], k[7], k[8]]], dtype=float)
            dist = np.asarray(d, dtype=float) if d is not None else None

            img_path = rec.get("color_path")
            img = None
            if overlays_saved < args.max_overlays and img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)

            for det in rec.get("detections", []):
                mid = int(det.get("marker_id"))
                if mid not in marker_map:
                    continue
                center_px = det.get("center_px")
                if center_px is None:
                    continue

                # project detected corners using detection's rvec/tvec and the assumed marker length
                rvec_det = np.asarray(det.get("rvec"), dtype=float)
                tvec_det = np.asarray(det.get("tvec"), dtype=float)
                half_det = args.marker_length / 2.0
                corners_det_obj = np.array([[-half_det, half_det, 0.0], [half_det, half_det, 0.0], [half_det, -half_det, 0.0], [-half_det, -half_det, 0.0]], dtype=float)
                pts_det, _ = cv2.projectPoints(corners_det_obj, rvec_det, tvec_det, K, dist)
                pts_det = pts_det.reshape(-1, 2)
                # mean distance from center to corners (detected)
                d_det = float(np.mean(np.linalg.norm(pts_det - np.asarray(center_px, dtype=float), axis=1)))

                # predicted corners from marker_map + extrinsics
                T_world_marker = np.asarray(marker_map[mid])
                T_cam_marker_pred = T_world_to_cam @ T_world_marker
                R_cam_marker = T_cam_marker_pred[:3, :3]
                t_cam_marker = T_cam_marker_pred[:3, 3]

                pts_pred = []
                for c in corners_marker_pred:
                    p = R_cam_marker @ c + t_cam_marker
                    if p[2] <= 0:
                        break
                    u = K[0, 0] * (p[0] / p[2]) + K[0, 2]
                    v = K[1, 1] * (p[1] / p[2]) + K[1, 2]
                    pts_pred.append((u, v))
                if len(pts_pred) != 4:
                    continue
                pts_pred = np.asarray(pts_pred, dtype=float)
                # predicted center (average)
                center_pred = pts_pred.mean(axis=0)
                d_pred = float(np.mean(np.linalg.norm(pts_pred - center_pred, axis=1)))

                if d_pred <= 1e-6:
                    continue

                ratio = d_det / d_pred
                ratios.append(ratio)
                per_marker[mid].append(ratio)

                # draw overlay with predicted (current) corners for debugging (we will regen later)
                if img is not None and overlays_saved < args.max_overlays:
                    pts = pts_pred.astype(int)
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.circle(img, (int(round(center_px[0])), int(round(center_px[1]))), 3, (0, 255, 0), -1)

            if img is not None and overlays_saved < args.max_overlays:
                out_path = os.path.join(args.out_dir, f"pre_overlay_{rec.get('ts_ns')}.png")
                cv2.imwrite(out_path, img)
                overlays_saved += 1

    if not ratios:
        print("No valid ratios computed; nothing to do.")
        return

    med = float(median(ratios))
    per_marker_med = {str(k): float(median(v)) for k, v in per_marker.items() if v}
    new_marker_length = args.marker_length * med

    summary = {
        "num_pairs": len(ratios),
        "median_ratio": med,
        "new_marker_length_m": new_marker_length,
        "per_marker_median_ratio": per_marker_med,
    }

    summary_path = os.path.join(args.out_dir, "marker_scale_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))

    # regenerate overlays using the scaled marker length (predicted corners)
    half_new = new_marker_length / 2.0
    corners_marker_new = np.array([[-half_new, half_new, 0.0], [half_new, half_new, 0.0], [half_new, -half_new, 0.0], [-half_new, -half_new, 0.0]], dtype=float)
    regen_dir = os.path.join(args.out_dir, "overlays_scaled")
    os.makedirs(regen_dir, exist_ok=True)
    saved = 0

    with open(args.records_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            if saved >= args.max_overlays:
                break
            rec = json.loads(line)
            cam_info = rec.get("camera_info", {})
            k = cam_info.get("k") or cam_info.get("K")
            if k is None:
                continue
            K = np.array([[k[0], k[1], k[2]],[k[3], k[4], k[5]],[k[6], k[7], k[8]]], dtype=float)

            img_path = rec.get("color_path")
            if not img_path or not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)

            for det in rec.get("detections", []):
                mid = int(det.get("marker_id"))
                if mid not in marker_map:
                    continue
                center_px = det.get("center_px")
                if center_px is None:
                    continue

                T_world_marker = np.asarray(marker_map[mid])
                T_cam_marker_pred = T_world_to_cam @ T_world_marker
                R_cam_marker = T_cam_marker_pred[:3, :3]
                t_cam_marker = T_cam_marker_pred[:3, 3]

                pts = []
                for c in corners_marker_new:
                    p = R_cam_marker @ c + t_cam_marker
                    if p[2] <= 0:
                        break
                    u = K[0, 0] * (p[0] / p[2]) + K[0, 2]
                    v = K[1, 1] * (p[1] / p[2]) + K[1, 2]
                    pts.append((int(round(u)), int(round(v))))
                if len(pts) != 4:
                    continue

                cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(img, (int(round(center_px[0])), int(round(center_px[1]))), 3, (0, 0, 255), -1)

            out_path = os.path.join(regen_dir, f"overlay_scaled_{rec.get('ts_ns')}.png")
            cv2.imwrite(out_path, img)
            saved += 1

    print(f"Saved {saved} scaled overlays to {regen_dir}")


if __name__ == "__main__":
    main()
