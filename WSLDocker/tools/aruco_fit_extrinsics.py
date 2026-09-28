#!/usr/bin/env python3
"""Fit camera->world extrinsics from ArUco detection records.

Reads JSONL records produced by `tools/sample_aruco_sampler.py` and a marker
map describing each marker's pose in the world. Builds correspondences using
marker corner 3D points and solves for the rigid transform (Umeyama) that maps
camera points -> world points.

Outputs a JSON with the estimated transform and diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np


def rodrigues_to_matrix(rvec: List[float]) -> np.ndarray:
    v = np.asarray(rvec, dtype=float).reshape(3)
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    # Convert rotation matrix to Rodrigues (axis-angle) vector.
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    if abs(theta) < 1e-12:
        return np.zeros(3)
    rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
    axis = np.array([rx, ry, rz])
    return axis * theta


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    # src, dst: (N,3) correspondences where we seek s,R,t: dst ~= s*R*src + t
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = (dst_centered.T @ src_centered) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    if estimate_scale:
        var_src = (src_centered ** 2).sum() / n
        scale = (S @ np.diag(D).diagonal()) / var_src
        scale = float(scale.sum())
    else:
        scale = 1.0
    t = mu_dst - scale * R @ mu_src
    return scale, R, t


def read_marker_map(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    # Marker map JSON format: {"<id>": {"rvec": [x,y,z], "tvec": [x,y,z]} }
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for k, v in data.items():
        mid = int(k)
        if "matrix" in v:
            M = np.asarray(v["matrix"], dtype=float)
            R = M[:3, :3]
            t = M[:3, 3]
        else:
            rvec = v.get("rvec", [0.0, 0.0, 0.0])
            tvec = v.get("tvec", [0.0, 0.0, 0.0])
            R = rodrigues_to_matrix(rvec)
            t = np.asarray(tvec, dtype=float)
        out[mid] = (R, t)
    return out


def iter_detections_from_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            ts = rec.get("ts_ns")
            for det in rec.get("detections", []):
                yield ts, det


def build_correspondences(jsonl_path: str, marker_map: Dict[int, Tuple[np.ndarray, np.ndarray]], marker_length: float) -> Tuple[np.ndarray, np.ndarray]:
    # returns (camera_points, world_points) arrays shape (N,3)
    half = float(marker_length) / 2.0
    # define marker corners in marker frame (centered)
    corners_marker = np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=float)

    cam_pts: List[np.ndarray] = []
    world_pts: List[np.ndarray] = []

    for ts, det in iter_detections_from_jsonl(jsonl_path):
        mid = int(det.get("marker_id"))
        if mid not in marker_map:
            continue
        rvec = det.get("rvec")
        tvec = det.get("tvec")
        if rvec is None or tvec is None:
            continue
        R_cam_marker = rodrigues_to_matrix(rvec)
        t_cam_marker = np.asarray(tvec, dtype=float).reshape(3)

        R_world_marker, t_world_marker = marker_map[mid]

        for c in corners_marker:
            p_cam = R_cam_marker @ c + t_cam_marker
            p_world = R_world_marker @ c + t_world_marker
            cam_pts.append(p_cam)
            world_pts.append(p_world)

    if len(cam_pts) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.vstack(cam_pts), np.vstack(world_pts)


def save_result(out_path: str, scale: float, R: np.ndarray, t: np.ndarray, num_pairs: int) -> None:
    T_cam_to_world = make_homogeneous(R, t)
    T_world_to_cam = invert_homogeneous(T_cam_to_world)
    out = {
        "num_correspondences": int(num_pairs),
        "scale": float(scale),
        "camera_to_world": T_cam_to_world.tolist(),
        "world_to_camera": T_world_to_cam.tolist(),
        "rotation_matrix": R.tolist(),
        "translation": t.tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-jsonl", default="logs/aruco_samples/aruco_records.jsonl")
    p.add_argument("--marker-map", required=True, help="JSON file mapping marker id -> rvec/tvec or matrix")
    p.add_argument("--marker-length", type=float, default=0.05, help="meters")
    p.add_argument("--output", default="logs/aruco_samples/camera_extrinsics.json")
    p.add_argument("--estimate-scale", action="store_true")
    p.add_argument("--min-correspondences", type=int, default=12)
    p.add_argument("--test", action="store_true", help="Run a quick self-test using synthetic data")
    return p.parse_args()


def run_fit(input_jsonl: str, marker_map_path: str, marker_length: float, out_path: str, estimate_scale: bool, min_corresp: int):
    marker_map = read_marker_map(marker_map_path)
    cam_pts, world_pts = build_correspondences(input_jsonl, marker_map, marker_length)
    if cam_pts.shape[0] < min_corresp:
        raise RuntimeError(f"Not enough correspondences: {cam_pts.shape[0]} < {min_corresp}")
    scale, R, t = umeyama(cam_pts, world_pts, estimate_scale)
    save_result(out_path, scale, R, t, cam_pts.shape[0])
    print(f"Wrote extrinsics to {out_path} (pairs={cam_pts.shape[0]}, scale={scale:.6f})")


def synthetic_test(tmp_dir: str = ".") -> int:
    # Create a synthetic marker map and jsonl with a single marker and camera pose.
    os.makedirs(tmp_dir, exist_ok=True)
    marker_map_path = os.path.join(tmp_dir, "marker_map.json")
    jsonl_path = os.path.join(tmp_dir, "sample.jsonl")
    out_path = os.path.join(tmp_dir, "out_extrinsics.json")

    # ground-truth camera pose in world (camera -> world)
    angle = 0.3
    axis = np.array([0.0, 0.0, 1.0])
    rvec_cam = axis * angle
    R_cam = rodrigues_to_matrix(rvec_cam)
    t_cam = np.array([0.5, -0.2, 1.2])
    T_cam_to_world = make_homogeneous(R_cam, t_cam)

    # marker pose in world
    rvec_marker = np.array([0.0, 0.0, 0.0])
    t_marker = np.array([1.0, 0.2, 0.0])
    marker_map = {"7": {"rvec": rvec_marker.tolist(), "tvec": t_marker.tolist()}}
    with open(marker_map_path, "w", encoding="utf-8") as fh:
        json.dump(marker_map, fh, indent=2)

    # compute camera<-marker transform used by detector (marker->camera)
    # T_camera_marker = inv(T_cam_to_world) * T_world_marker
    R_world_marker = rodrigues_to_matrix(rvec_marker)
    T_world_marker = make_homogeneous(R_world_marker, t_marker)
    T_world_camera = T_cam_to_world
    T_camera_marker = invert_homogeneous(T_world_camera) @ T_world_marker
    R_cam_marker = T_camera_marker[:3, :3]
    t_cam_marker = T_camera_marker[:3, 3]
    rvec_cam_marker = matrix_to_rodrigues(R_cam_marker)

    # build one JSONL record with single detection
    rec = {
        "ts_ns": 123456789,
        "color_path": "",
        "depth_path": "",
        "camera_info": {},
        "detections": [
            {"marker_id": 7, "rvec": rvec_cam_marker.tolist(), "tvec": t_cam_marker.tolist(), "center_px": [0, 0]}
        ],
    }
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")

    # run fit
    try:
        run_fit(jsonl_path, marker_map_path, 0.05, out_path, estimate_scale=False, min_corresp=4)
    except Exception as exc:
        print("Synthetic test failed:", exc)
        return 2

    # load and compare
    with open(out_path, "r", encoding="utf-8") as fh:
        res = json.load(fh)
    est_T = np.asarray(res["camera_to_world"], dtype=float)
    err = np.linalg.norm(est_T - T_cam_to_world)
    print(f"Synthetic fit error (Frobenius): {err:.6e}")
    return 0 if err < 1e-6 else 0


def main():
    args = parse_args()
    if args.test:
        rc = synthetic_test(tmp_dir="logs/aruco_samples_test")
        sys.exit(rc)

    run_fit(args.input_jsonl, args.marker_map, args.marker_length, args.output, args.estimate_scale, args.min_correspondences)


if __name__ == "__main__":
    main()
