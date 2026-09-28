#!/usr/bin/env python3
"""Sample ArUco poses from color+aligned-depth streams at a fixed rate.

Saves per-frame JSONL records and a CSV of detected marker poses (camera->marker
rvec/tvec) suitable for downstream least-squares extrinsic fitting.

Usage (ROS2 env sourced):
  python3 tools/sample_aruco_sampler.py --output-dir logs/aruco_samples --max-frames 50
  python3 tools/sample_aruco_sampler.py --check-only    # verify dependencies
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
except Exception:  # pragma: no cover - allow --check-only without ROS
    rclpy = None
    Node = None
    Image = None
    CameraInfo = None

try:
    from tools.depth_frame_utils import decode_sensor_image, save_sensor_image, camera_info_to_dict
except Exception:
    from depth_frame_utils import decode_sensor_image, save_sensor_image, camera_info_to_dict

try:
    from tools.aruco_alignment_utils import _get_aruco_dictionary, _camera_matrix_from_info
except Exception:
    from aruco_alignment_utils import _get_aruco_dictionary, _camera_matrix_from_info


DEFAULT_COLOR_TOPIC = "/camera/color/image_raw"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
DEFAULT_INFO_TOPIC = "/camera/color/camera_info"


def _stamp_to_seconds(msg_hdr) -> float:
    if msg_hdr is None:
        return 0.0
    s = getattr(msg_hdr, "stamp", None)
    if s is None:
        return 0.0
    sec = int(getattr(s, "sec", 0) or 0)
    nsec = int(getattr(s, "nanosec", 0) or 0)
    return float(sec) + float(nsec) * 1e-9


if Node is not None:
    BaseNode = Node if Node is not None else object


    class ArucoSamplerNode(BaseNode):
        def __init__(
            self,
            output_dir: str,
            rate_hz: float = 4.0,
            marker_length_m: float = 0.02,
            dictionary_name: str = "DICT_4X4_50",
            depth_scale: float = 0.001,
            color_topic: str = DEFAULT_COLOR_TOPIC,
            depth_topic: str = DEFAULT_DEPTH_TOPIC,
            info_topic: str = DEFAULT_INFO_TOPIC,
            max_frames: Optional[int] = None,
            duration_s: Optional[float] = None,
        ) -> None:
            # Only initialize ROS node/subscriptions if rclpy is available.
            if rclpy is not None:
                super().__init__("aruco_sampler")
                qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
                self.sub_color = self.create_subscription(Image, color_topic, self._cb_color, qos)
                self.sub_depth = self.create_subscription(Image, depth_topic, self._cb_depth, qos)
                self.sub_info = self.create_subscription(CameraInfo, info_topic, self._cb_info, qos)
            else:
                # placeholders for type checker and offline checks
                self.sub_color = None
                self.sub_depth = None
                self.sub_info = None

            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            self.records_path = os.path.join(self.output_dir, "aruco_records.jsonl")
            self.csv_path = os.path.join(self.output_dir, "aruco_detections.csv")
            self.camera_info_path = os.path.join(self.output_dir, "camera_info.json")

            # live buffers
            self.last_color: Optional[Image] = None
            self.last_depth: Optional[Image] = None
            self.last_info: Optional[CameraInfo] = None

            self.rate_hz = float(rate_hz)
            self.marker_length_m = float(marker_length_m)
            self.dictionary_name = str(dictionary_name)
            self.depth_scale = float(depth_scale)
            self.max_frames = int(max_frames) if max_frames is not None else None
            self.duration_s = float(duration_s) if duration_s is not None else None

            # output files
            self._jsonl_handle = open(self.records_path, "a", encoding="utf-8")
            self._csv_handle = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_handle)
            # write header if empty
            if os.path.getsize(self.csv_path) == 0:
                header = [
                    "ts_ns",
                    "frame_id",
                    "marker_id",
                    "center_x_px",
                    "center_y_px",
                    "rvec_x",
                    "rvec_y",
                    "rvec_z",
                    "tvec_x",
                    "tvec_y",
                    "tvec_z",
                    "depth_center_m",
                    "color_path",
                    "depth_path",
                ]
                self._csv_writer.writerow(header)
                self._csv_handle.flush()

            self.start_time = time.time()
            self.frame_count = 0

            # detection resources
            if cv2 is None:
                self.get_logger().error("OpenCV (cv2) is unavailable; aruco detection requires OpenCV with aruco module")
                raise RuntimeError("cv2 unavailable")
            try:
                self.dictionary = _get_aruco_dictionary(self.dictionary_name)
            except Exception as exc:
                self.get_logger().error(f"Failed to get ArUco dictionary {self.dictionary_name}: {exc}")
                raise

            # DetectorParameters_create() exists in some OpenCV builds; others expose
            # DetectorParameters() as the constructor. Support both for compatibility.
            try:
                self.detector_params = cv2.aruco.DetectorParameters_create()
            except AttributeError:
                self.detector_params = cv2.aruco.DetectorParameters()

            # sampling timer
            self.timer = self.create_timer(1.0 / max(0.1, self.rate_hz), self._timer_cb)

        # ---- ROS callbacks
        def _cb_color(self, msg: Image) -> None:
            self.last_color = msg

        def _cb_depth(self, msg: Image) -> None:
            self.last_depth = msg

        def _cb_info(self, msg: CameraInfo) -> None:
            self.last_info = msg

        # ---- timer
        def _timer_cb(self) -> None:
            # check termination by duration
            if self.duration_s is not None and (time.time() - self.start_time) > float(self.duration_s):
                self.get_logger().info("Duration reached; shutting down sampler")
                self.destroy_node()
                rclpy.shutdown()
                return

            if self.max_frames is not None and self.frame_count >= int(self.max_frames):
                self.get_logger().info("Max frames reached; shutting down sampler")
                self.destroy_node()
                rclpy.shutdown()
                return

            if self.last_color is None or self.last_depth is None or self.last_info is None:
                # not yet ready
                return

            # ensure approximate synchronicity
            t_color = _stamp_to_seconds(self.last_color.header)
            t_depth = _stamp_to_seconds(self.last_depth.header)
            t_info = _stamp_to_seconds(self.last_info.header)
            if abs(t_color - t_depth) > 0.10 or abs(t_color - t_info) > 0.10:
                # skip until near-synchronous frames available
                return

            try:
                self._process_frame(self.last_color, self.last_depth, self.last_info)
            except Exception as exc:
                self.get_logger().error(f"Frame processing failed: {exc}")

        def _process_frame(self, color_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
            ts_ns = int(getattr(getattr(color_msg, "header", None), "stamp", None).sec) * 1_000_000_000 + int(getattr(getattr(color_msg, "header", None), "stamp", None).nanosec)

            # save images and camera_info sidecar
            color_path = os.path.join(self.output_dir, f"color_{ts_ns}.ppm")
            depth_path = os.path.join(self.output_dir, f"depth_{ts_ns}.pgm")
            try:
                save_sensor_image(color_msg, color_path)
            except Exception:
                # not fatal
                color_path = ""
            try:
                save_sensor_image(depth_msg, depth_path)
            except Exception:
                depth_path = ""

            try:
                camera_info = camera_info_to_dict(info_msg)
                with open(os.path.join(self.output_dir, f"camera_info_{ts_ns}.json"), "w", encoding="utf-8") as fh:
                    json.dump(camera_info, fh)
            except Exception:
                camera_info = {}

            # convert color to gray for detection
            color_arr = decode_sensor_image(color_msg)
            if color_arr.ndim == 3:
                gray = cv2.cvtColor(color_arr, cv2.COLOR_RGB2GRAY)
            else:
                gray = color_arr

            cam_mat, dist_coeffs = _camera_matrix_from_info(info_msg)
            if cam_mat is None:
                self.get_logger().warning("Camera intrinsics missing; skipping detection")
                return

            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.detector_params)
            detections: List[Dict[str, Any]] = []
            if ids is None or len(ids) == 0:
                # write a record with zero detections
                rec = {
                    "ts_ns": ts_ns,
                    "color_path": color_path,
                    "depth_path": depth_path,
                    "camera_info": camera_info,
                    "detections": [],
                }
                print(json.dumps(rec, indent=2))
                self._jsonl_handle.write(json.dumps(rec) + "\n")
                self._jsonl_handle.flush()
                return

            # estimate pose for detected markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, float(self.marker_length_m), cam_mat, dist_coeffs)

            depth_arr = decode_sensor_image(depth_msg)

            for idx, marker_id_arr in enumerate(np.asarray(ids).reshape(-1)):
                marker_id = int(marker_id_arr)
                c = corners[idx].reshape(-1, 2)
                center_x = float(np.mean(c[:, 0]))
                center_y = float(np.mean(c[:, 1]))
                rvec = [float(x) for x in np.asarray(rvecs[idx]).reshape(3).tolist()]
                tvec = [float(x) for x in np.asarray(tvecs[idx]).reshape(3).tolist()]

                # read median depth in 3x3 neighborhood
                h, w = depth_arr.shape[:2]
                px = int(round(center_x))
                py = int(round(center_y))
                depth_center_m: Optional[float] = None
                if 0 <= px < w and 0 <= py < h:
                    x0 = max(0, px - 1); x1 = min(w, px + 2)
                    y0 = max(0, py - 1); y1 = min(h, py + 2)
                    region = depth_arr[y0:y1, x0:x1]
                    if region.size:
                        # handle integer vs float depth encodings
                        if np.issubdtype(region.dtype, np.floating):
                            vals = region.astype(float).flatten()
                            vals = vals[np.isfinite(vals) & (vals > 0.0)]
                            if vals.size:
                                depth_center_m = float(np.median(vals))
                        else:
                            vals = region.flatten()
                            vals = vals[vals > 0]
                            if vals.size:
                                depth_center_m = float(np.median(vals)) * float(self.depth_scale)

                det = {
                    "marker_id": marker_id,
                    "center_px": [center_x, center_y],
                    "rvec": rvec,
                    "tvec": tvec,
                    "depth_center_m": None if depth_center_m is None else float(depth_center_m),
                }
                detections.append(det)

                # CSV row
                self._csv_writer.writerow([
                    ts_ns,
                    getattr(info_msg.header, "frame_id", ""),
                    marker_id,
                    center_x,
                    center_y,
                    rvec[0], rvec[1], rvec[2],
                    tvec[0], tvec[1], tvec[2],
                    None if depth_center_m is None else float(depth_center_m),
                    color_path,
                    depth_path,
                ])

            rec = {
                "ts_ns": ts_ns,
                "color_path": color_path,
                "depth_path": depth_path,
                "camera_info": camera_info,
                "detections": detections,
            }
            # write JSONL
            self._jsonl_handle.write(json.dumps(rec) + "\n")
            self._jsonl_handle.flush()
            self._csv_handle.flush()

            # increment frame count
            self.frame_count += 1

        def destroy_node(self) -> None:
            try:
                self._jsonl_handle.close()
            except Exception:
                pass
            try:
                self._csv_handle.close()
            except Exception:
                pass
            super().destroy_node()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ArUco sampler for color+aligned-depth streams")
    p.add_argument("--output-dir", default="logs/aruco_samples", help="Directory to write JSON/CSV and images")
    p.add_argument("--rate", type=float, default=4.0, help="Sampling rate in Hz (default 4)")
    p.add_argument("--marker-length-m", type=float, default=0.02, help="ArUco marker side length in meters")
    p.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary name")
    p.add_argument("--depth-scale", type=float, default=0.001, help="Scale to convert integer depth units to meters (default 0.001)")
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many processed frames")
    p.add_argument("--duration-s", type=float, default=None, help="Stop after this many seconds")
    p.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    p.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    p.add_argument("--camera-info-topic", default=DEFAULT_INFO_TOPIC)
    p.add_argument("--check-only", action="store_true", help="Check dependencies only and exit")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.check_only:
        ok = True
        if cv2 is None:
            print("ERROR: cv2 is not available. Install OpenCV with contrib (aruco).")
            ok = False
        else:
            if not hasattr(cv2, "aruco"):
                print("ERROR: cv2 available but aruco module missing. Install opencv-contrib-python.")
                ok = False
        try:
            _get_aruco_dictionary(args.dictionary)
        except Exception as exc:
            print(f"ERROR: failed to get ArUco dictionary {args.dictionary}: {exc}")
            ok = False
        try:
            # quick depth_frame_utils import check
            _ = decode_sensor_image
        except Exception as exc:
            print(f"ERROR: depth_frame_utils unavailable: {exc}")
            ok = False
        print("OK" if ok else "FAILED")
        return 0 if ok else 2

    if rclpy is None:
        print("ERROR: rclpy is not importable. Run this inside a sourced ROS2 environment.")
        return 3

    rclpy.init()
    try:
        node = ArucoSamplerNode(
            output_dir=args.output_dir,
            rate_hz=args.rate,
            marker_length_m=args.marker_length_m,
            dictionary_name=args.dictionary,
            depth_scale=args.depth_scale,
            color_topic=args.color_topic,
            depth_topic=args.depth_topic,
            info_topic=args.camera_info_topic,
            max_frames=args.max_frames,
            duration_s=args.duration_s,
        )
    except Exception as exc:
        print(f"Failed to start sampler: {exc}")
        if rclpy is not None:
            rclpy.shutdown()
        return 4

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
