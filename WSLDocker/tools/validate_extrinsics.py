#!/usr/bin/env python3
"""Validate fitted camera extrinsics against ArUco captures.

Outputs reprojection statistics and overlay images showing predicted marker
corners (from the computed marker map + extrinsics) vs detected centers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

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


def project_point(p_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> Tuple[float, float]:
    x = p_cam[0] / p_cam[2]
    y = p_cam[1] / p_cam[2]
    # no distortion expected in our captures, but handle zero dist
    if dist is None or np.allclose(dist, 0.0):
        u = K[0, 0] * x + K[0, 2]
        v = K[1, 1] * y + K[1, 2]
        return float(u), float(v)
    # use cv2.projectPoints for distortion handling
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    p = np.asarray([[p_cam[0], p_cam[1], p_cam[2]]], dtype=float)
    pts, _ = cv2.projectPoints(p, rvec, tvec, K, dist)
    return float(pts[0, 0, 0]), float(pts[0, 0, 1])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--records-jsonl", default="logs/aruco_hw_run_host/aruco_records.jsonl")
    p.add_argument("--camera-extrinsics", default="logs/aruco_hw_run_host/camera_extrinsics.json")
    p.add_argument("--marker-map", default="logs/aruco_hw_run_host/marker_map_computed.json")
    p.add_argument("--marker-length", type=float, default=0.05)
    p.add_argument("--out-dir", default="logs/aruco_hw_run_host/validation")
    p.add_argument("--max-overlays", type=int, default=12, help="How many overlay images to save")
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

    half = args.marker_length / 2.0
    corners_marker = np.array([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]], dtype=float)

    errors_px: List[float] = []
    errors_z: List[float] = []
    errors_corner: List[float] = []
    per_marker = defaultdict(list)
    per_marker_corner = defaultdict(list)
    overlay_count = 0

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
            if overlay_count < args.max_overlays and img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)

            for det in rec.get("detections", []):
                mid = int(det.get("marker_id"))
                if mid not in marker_map:
                    continue
                center_px = det.get("center_px")
                depth_center = det.get("depth_center_m")

                T_world_marker = np.asarray(marker_map[mid])
                T_cam_marker_pred = T_world_to_cam @ T_world_marker
                p_cam = T_cam_marker_pred[:3, 3]

                # project predicted center
                u_pred, v_pred = project_point(p_cam, K, dist)

                if center_px is not None:
                    err_px = math.hypot(u_pred - float(center_px[0]), v_pred - float(center_px[1]))
                    errors_px.append(err_px)
                    per_marker[mid].append(err_px)

                if depth_center is not None:
                    err_z = float(p_cam[2]) - float(depth_center)
                    errors_z.append(err_z)

                # project detection corners (from detector rvec/tvec) and predicted corners
                det_r = det.get("rvec")
                det_t = det.get("tvec")
                det_pts_img = None
                if det_r is not None and det_t is not None:
                    rvec_det = np.asarray(det_r, dtype=float).reshape(3)
                    tvec_det = np.asarray(det_t, dtype=float).reshape(3)
                    try:
                        det_proj, _ = cv2.projectPoints(corners_marker, rvec_det, tvec_det, K, dist)
                        det_pts_img = [(int(round(p[0])), int(round(p[1]))) for p in det_proj.reshape(-1, 2)]
                    except Exception:
                        det_pts_img = None

                # predicted corners from marker_map + extrinsics
                R_cam_marker = T_cam_marker_pred[:3, :3]
                t_cam_marker = T_cam_marker_pred[:3, 3]
                pred_pts = []
                pred_pts_f = []
                for c in corners_marker:
                    p = R_cam_marker @ c + t_cam_marker
                    u, v = project_point(p, K, dist)
                    pred_pts.append((int(round(u)), int(round(v))))
                    pred_pts_f.append((u, v))

                # compute per-corner pixel errors (prediction vs detection)
                if det_pts_img is not None:
                    for (up, vp), (ud, vd) in zip(pred_pts_f, det_proj.reshape(-1, 2)):
                        e = math.hypot(up - float(ud), vp - float(vd))
                        errors_corner.append(e)
                        per_marker_corner[mid].append(e)

                # draw overlay: predicted corners (green), detection corners (red), centers (blue)
                if img is not None and overlay_count < args.max_overlays:
                    if pred_pts:
                        cv2.polylines(img, [np.array(pred_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                    if det_pts_img is not None:
                        cv2.polylines(img, [np.array(det_pts_img, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
                    if center_px is not None:
                        cv2.circle(img, (int(round(center_px[0])), int(round(center_px[1]))), 4, (255, 0, 0), -1)
                    cv2.circle(img, (int(round(u_pred)), int(round(v_pred))), 4, (0, 255, 255), -1)

            if img is not None and overlay_count < args.max_overlays:
                out_path = os.path.join(args.out_dir, f"overlay_{rec.get('ts_ns')}.png")
                cv2.imwrite(out_path, img)
                overlay_count += 1

    def stats(xs: List[float]):
        if not xs:
            return {"count": 0}
        a = np.asarray(xs, dtype=float)
        return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std())}

    summary = {
        "pixel_error": stats(errors_px),
        "depth_z_error_m": stats(errors_z),
        "per_marker_pixel_error": {str(k): stats(v) for k, v in per_marker.items()},
        "corner_error_px": stats(errors_corner),
        "per_marker_corner_error_px": {str(k): stats(v) for k, v in per_marker_corner.items()},
        "overlay_images_saved": overlay_count,
    }

    out_summary = os.path.join(args.out_dir, "validation_summary.json")
    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Overlays (up to {args.max_overlays}) saved to {args.out_dir}")


if __name__ == "__main__":
    main()
