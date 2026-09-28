#!/usr/bin/env python3
"""
windows_aruco_udp_sender.py

Run this natively on Windows.

Pipeline:
    Camera / RealSense / OpenCV
        -> Detect ArUco markers
        -> Estimate marker pose using camera intrinsics
        -> Send marker poses as UDP JSON packets to WSL / another machine

Install on Windows:
    pip install opencv-contrib-python numpy

If using Intel RealSense directly:
    pip install pyrealsense2

Examples:
    # RealSense mode, try localhost first
    python windows_aruco_udp_sender.py --realsense --udp-host 127.0.0.1 --udp-port 5005 --show

    # If localhost does not work, use WSL IP from `hostname -I` inside WSL
    python windows_aruco_udp_sender.py --realsense --udp-host <WSL_IP> --udp-port 5005 --show

    # Webcam mode with manual intrinsics
    python windows_aruco_udp_sender.py --camera-index 0 --fx 615 --fy 615 --cx 320 --cy 240 --udp-host 127.0.0.1 --show
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# -------------------------------------------------------------------------
# Defaults carried over from the original single-arm ArUco ROS2 visualizer
# -------------------------------------------------------------------------

DEFAULT_MARKER_LENGTH_M = 0.038  # 38 mm markers

DEFAULT_MARKERS = [
    ("ground_world", 6),
    ("metal_platform", 0),
    ("spring_cap1", 1),
    ("spring_cap2", 3),
    ("green_platform_r2", 4),
]

COLORS = {
    "ground_world": (0, 200, 0),
    "metal_platform": (200, 100, 0),
    "spring_cap1": (0, 0, 255),
    "spring_cap2": (0, 165, 255),
    "green_platform_r2": (0, 255, 255),
    "unknown": (128, 128, 128),
}


@dataclass
class MarkerDetection:
    marker_id: int
    label: str
    corners: np.ndarray
    center_px: Tuple[float, float]
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        return COLORS.get(self.label, COLORS["unknown"])


def parse_marker_arg(value: str) -> Tuple[str, int]:
    """
    Parse marker argument in LABEL:ID form.
    Example:
        --marker robot_base:4
    """
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"Marker must be LABEL:ID, got {value}")
    label, mid = value.rsplit(":", 1)
    return label.strip(), int(mid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows ArUco UDP pose sender")

    # UDP
    parser.add_argument(
        "--udp-host",
        default="127.0.0.1",
        help="Receiver IP. Try 127.0.0.1 first. If not working, use WSL IP from hostname -I.",
    )
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument(
        "--send-every-n",
        type=int,
        default=1,
        help="Send every N frames. 1 means send every frame.",
    )

    # Camera
    parser.add_argument(
        "--realsense",
        action="store_true",
        help="Use Intel RealSense via pyrealsense2 and get intrinsics automatically.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index if not using RealSense.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)

    # Calibration for non-RealSense mode
    parser.add_argument(
        "--calib-npz",
        type=str,
        default=None,
        help="Optional npz with camera_matrix and dist_coeffs arrays.",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    # ArUco
    parser.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH_M)
    parser.add_argument("--dictionary", default="DICT_4X4_50")
    parser.add_argument(
        "--marker",
        action="append",
        dest="markers",
        type=parse_marker_arg,
        help="Marker as LABEL:ID. Can repeat. Default uses the single-arm marker layout.",
    )

    # Display/debug
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")

    return parser.parse_args()


def get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is missing. Install opencv-contrib-python:\n"
            "    pip install opencv-contrib-python"
        )

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def get_detector_parameters():
    # Compatible with newer OpenCV
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()

    # Compatible with older OpenCV
    return cv2.aruco.DetectorParameters_create()


def make_camera_matrix_from_args(
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    For webcam mode, pose estimation needs camera intrinsics.

    Priority:
        1. --calib-npz
        2. --fx --fy --cx --cy
        3. None, meaning only pixel detections are sent, no metric pose
    """
    if args.calib_npz:
        data = np.load(args.calib_npz)
        camera_matrix = data["camera_matrix"].astype(np.float64)
        dist_coeffs = data["dist_coeffs"].astype(np.float64).reshape(-1)
        return camera_matrix, dist_coeffs

    if None not in (args.fx, args.fy, args.cx, args.cy):
        camera_matrix = np.array(
            [
                [args.fx, 0.0, args.cx],
                [0.0, args.fy, args.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros(5, dtype=np.float64)
        return camera_matrix, dist_coeffs

    return None, None


def rvec_to_quaternion_xyzw(rvec: np.ndarray) -> List[float]:
    """
    Convert OpenCV rotation vector to quaternion [x, y, z, w].

    This orientation is marker orientation with respect to the camera frame.
    """
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def detect_markers(
    frame: np.ndarray,
    dictionary,
    detector_params,
    marker_map: Dict[int, str],
    marker_length_m: float,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
) -> List[MarkerDetection]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        corners_list, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        corners_list, ids, _ = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=detector_params,
        )

    detections: List[MarkerDetection] = []

    if ids is None or len(ids) == 0:
        return detections

    rvecs = None
    tvecs = None

    if camera_matrix is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners_list,
            marker_length_m,
            camera_matrix,
            dist_coeffs,
        )

    for i, raw_id in enumerate(ids.flatten()):
        marker_id = int(raw_id)
        label = marker_map.get(marker_id, f"unknown_{marker_id}")
        corners = corners_list[i].reshape(-1, 2)
        center = corners.mean(axis=0)

        det = MarkerDetection(
            marker_id=marker_id,
            label=label,
            corners=corners,
            center_px=(float(center[0]), float(center[1])),
            rvec=rvecs[i].flatten() if rvecs is not None else None,
            tvec=tvecs[i].flatten() if tvecs is not None else None,
        )
        detections.append(det)

    return detections


def detections_to_packet(
    detections: List[MarkerDetection],
    frame_count: int,
    camera_frame: str,
) -> Dict[str, Any]:
    packet: Dict[str, Any] = {
        "timestamp": time.time(),
        "frame": frame_count,
        "camera_frame": camera_frame,
        "markers": [],
    }

    for det in detections:
        entry: Dict[str, Any] = {
            "id": det.marker_id,
            "label": det.label,
            "center_px": list(det.center_px),
            "corners_px": det.corners.tolist(),
        }

        if det.tvec is not None:
            t = det.tvec.flatten()
            entry["position_camera_m"] = [float(t[0]), float(t[1]), float(t[2])]

        if det.rvec is not None:
            r = det.rvec.flatten()
            entry["rotation_vec"] = [float(r[0]), float(r[1]), float(r[2])]
            entry["orientation_xyzw_camera"] = rvec_to_quaternion_xyzw(r)

        packet["markers"].append(entry)

    return packet


def draw_overlay(
    frame: np.ndarray,
    detections: List[MarkerDetection],
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
    marker_length_m: float,
    frame_count: int,
) -> np.ndarray:
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        color = det.color_bgr
        corners = det.corners.astype(int)

        cv2.polylines(overlay, [corners], True, color, 3)

        cx, cy = int(det.center_px[0]), int(det.center_px[1])
        cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

        label_text = f"{det.label} [ID={det.marker_id}]"
        cv2.putText(
            overlay,
            label_text,
            (corners[0][0], max(corners[0][1] - 10, 20)),
            font,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        if det.tvec is not None:
            t = det.tvec.flatten()
            pos_text = f"x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f} m"
            cv2.putText(
                overlay,
                pos_text,
                (corners[0][0], corners[0][1] + 20),
                font,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if det.rvec is not None and det.tvec is not None and camera_matrix is not None:
            cv2.drawFrameAxes(
                overlay,
                camera_matrix,
                dist_coeffs,
                det.rvec,
                det.tvec,
                marker_length_m * 1.5,
                2,
            )

    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    status = f"Frame {frame_count} | {len(detections)} markers | UDP sender"
    cv2.putText(
        overlay,
        status,
        (8, 23),
        font,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return overlay


class RealSenseCapture:
    def __init__(self, width: int, height: int, fps: int):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install using:\n"
                "    pip install pyrealsense2"
            ) from exc

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        profile = self.pipeline.start(self.config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()

        self.camera_matrix = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        coeffs = list(intr.coeffs)
        while len(coeffs) < 5:
            coeffs.append(0.0)
        self.dist_coeffs = np.array(coeffs[:5], dtype=np.float64)

        print("[INFO] RealSense intrinsics:")
        print(self.camera_matrix)
        print("[INFO] Distortion:", self.dist_coeffs)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        frame = np.asanyarray(color_frame.get_data())
        return True, frame

    def release(self) -> None:
        self.pipeline.stop()


class OpenCVCapture:
    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        return ok, frame

    def release(self) -> None:
        self.cap.release()


def main() -> None:
    args = parse_args()

    marker_list = args.markers if args.markers else DEFAULT_MARKERS
    marker_map = {marker_id: label for label, marker_id in marker_list}

    dictionary = get_aruco_dictionary(args.dictionary)
    detector_params = get_detector_parameters()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_addr = (args.udp_host, args.udp_port)

    print("[INFO] UDP target:", receiver_addr)
    print("[INFO] Marker map:", marker_map)
    print(f"[INFO] Marker length: {args.marker_length * 1000:.1f} mm")

    if args.realsense:
        camera = RealSenseCapture(args.width, args.height, args.fps)
        camera_matrix = camera.camera_matrix
        dist_coeffs = camera.dist_coeffs
    else:
        camera = OpenCVCapture(args.camera_index, args.width, args.height, args.fps)
        camera_matrix, dist_coeffs = make_camera_matrix_from_args(args)

        if camera_matrix is None:
            print()
            print("[WARNING] No camera intrinsics provided.")
            print("[WARNING] Marker IDs and pixel corners will be sent, but metric pose will be unavailable.")
            print("[WARNING] For real x/y/z + orientation, use one of:")
            print("          1. --realsense")
            print("          2. --calib-npz calibration.npz")
            print("          3. --fx --fy --cx --cy")
            print()

    frame_count = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[WARNING] Failed to read camera frame")
                time.sleep(0.01)
                continue

            frame_count += 1

            detections = detect_markers(
                frame=frame,
                dictionary=dictionary,
                detector_params=detector_params,
                marker_map=marker_map,
                marker_length_m=args.marker_length,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )

            packet = detections_to_packet(
                detections=detections,
                frame_count=frame_count,
                camera_frame=args.camera_frame,
            )

            if frame_count % max(1, args.send_every_n) == 0:
                payload = json.dumps(packet).encode("utf-8")
                sock.sendto(payload, receiver_addr)

            if args.print_json and detections:
                print(json.dumps(packet, indent=2))

            if args.show:
                overlay = draw_overlay(
                    frame=frame,
                    detections=detections,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    marker_length_m=args.marker_length,
                    frame_count=frame_count,
                )
                cv2.imshow("Windows ArUco UDP Sender", overlay)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if frame_count % 30 == 0:
                ids = [f"{d.label}[{d.marker_id}]" for d in detections]
                print(f"[INFO] Frame {frame_count}: sent {len(detections)} markers {ids}")

    except KeyboardInterrupt:
        pass

    finally:
        camera.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
