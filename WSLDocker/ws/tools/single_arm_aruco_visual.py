#!/usr/bin/env python3
"""
single_arm_aruco_visual.py
───────────────────────────
Single-arm ArUco visualizer for real hardware.

Shows a live annotated camera feed with ArUco marker identification.
Designed for the single-robot (robot2-only) setup where robot2 pushes
cap2 against a spring backed by a rigid wall blocking cap1.

Tracked markers (default single-arm layout)
────────────────────────────────────────────
- `metal_platform`      (ID=0) — central metal platform
- `spring_cap1`         (ID=1) — cap blocked by rigid wall
- `spring_cap2`         (ID=3) — cap pushed by robot2
- `green_platform_r2`   (ID=4) — robot2's green base platform
- `ground_world`        (ID=6) — ground reference marker

Usage
─────
  # Live annotated overlay (view with image_tools or RViz)
  python3 tools/single_arm_aruco_visual.py

  # Save annotated snapshot to disk
  python3 tools/single_arm_aruco_visual.py --save-snapshot /tmp/aruco_snapshot.png

  # Print detections once and exit
  python3 tools/single_arm_aruco_visual.py --print-once --capture-timeout-s 10

  # View the overlay in a separate terminal:
  ros2 run image_tools showimage --ros-args -r image:=/single_arm_aruco/overlay

Topics
──────
  Subscribes:
    /camera/color/image_raw (sensor_msgs/Image)
    /camera/color/camera_info (sensor_msgs/CameraInfo)

  Publishes:
    /single_arm_aruco/overlay (sensor_msgs/Image) — annotated camera feed
    /single_arm_aruco/detections_json (std_msgs/String) — JSON summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String as StringMsg

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tools.aruco_alignment_utils_single import (
        _get_aruco_dictionary,
        _get_detector_parameters,
        _gray_image,
    )
except ImportError:
    try:
        from aruco_alignment_utils_single import (
            _get_aruco_dictionary,
            _get_detector_parameters,
            _gray_image,
        )
    except ImportError:
        _get_aruco_dictionary = None
        _get_detector_parameters = None
        _gray_image = None

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_COLOR_TOPIC = "/camera/color/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/color/camera_info"
DEFAULT_OVERLAY_TOPIC = "/single_arm_aruco/overlay"
DEFAULT_DETECTIONS_TOPIC = "/single_arm_aruco/detections_json"
DEFAULT_MARKER_LENGTH_M = 0.038  # 38mm markers

# Default marker layout for single-arm (robot2-only) setup
DEFAULT_MARKERS = [
    ("ground_world", 6),
    ("metal_platform", 0),
    ("spring_cap1", 1),        # Blocked by rigid wall
    ("spring_cap2", 3),        # Pushed by robot2
    ("green_platform_r2", 4),
]

# Colors for drawing (BGR format for OpenCV)
COLORS = {
    "ground_world": (0, 200, 0),        # Green
    "metal_platform": (200, 100, 0),    # Blue
    "spring_cap1": (0, 0, 255),         # Red — blocked by wall
    "spring_cap2": (0, 165, 255),       # Orange — pushed by robot2
    "green_platform_r2": (0, 255, 255), # Yellow
    "unknown": (128, 128, 128),         # Gray
}


@dataclass
class MarkerDetection:
    """A single detected ArUco marker with pose."""
    marker_id: int
    label: str
    corners: np.ndarray                      # 4x2 pixel corners
    center_px: Tuple[float, float]
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None

    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        return COLORS.get(self.label, COLORS["unknown"])

    @property
    def position_str(self) -> str:
        if self.tvec is not None:
            t = self.tvec.flatten()
            return f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
        return "N/A"


def parse_marker_arg(value: str) -> Tuple[str, int]:
    """Parse LABEL:ID format marker argument."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"Marker must be LABEL:ID, got: {value}")
    label, mid = value.rsplit(":", 1)
    return (label.strip(), int(mid))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-arm ArUco visualizer")
    parser.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    parser.add_argument("--camera-info-topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    parser.add_argument("--overlay-topic", default=DEFAULT_OVERLAY_TOPIC)
    parser.add_argument("--detections-topic", default=DEFAULT_DETECTIONS_TOPIC)
    parser.add_argument("--marker-length", type=float, default=DEFAULT_MARKER_LENGTH_M,
                        help="ArUco marker side length in metres (default: 0.038)")
    parser.add_argument("--marker", action="append", dest="markers", type=parse_marker_arg,
                        help="Marker as LABEL:ID (repeatable). Default: see DEFAULT_MARKERS")
    parser.add_argument("--save-snapshot", type=str, default=None,
                        help="Save first annotated frame to this path")
    parser.add_argument("--save-frames-dir", type=str, default=None,
                        help="Continuously save annotated frames to this directory")
    parser.add_argument("--print-once", action="store_true",
                        help="Print detections after a few frames and exit")
    parser.add_argument("--capture-timeout-s", type=float, default=30.0,
                        help="Timeout for first frame capture (seconds)")
    return parser.parse_args()


class SingleArmArucoVisual(Node):
    """Single-arm ArUco marker visualizer with annotated camera overlay."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("single_arm_aruco_visual")

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required but not available")

        self.args = args
        self.bridge = CvBridge() if CvBridge else None

        # Build marker ID -> label map
        self.marker_map: Dict[int, str] = {}
        marker_list = args.markers if args.markers else list(DEFAULT_MARKERS)
        for label, marker_id in marker_list:
            self.marker_map[marker_id] = label

        self.marker_length = args.marker_length
        self.frame_count = 0
        self.last_detections: List[MarkerDetection] = []
        self.best_detections: List[MarkerDetection] = []  # Track best detection across frames
        # Accumulate detections by marker ID across ALL frames — a marker detected
        # in frame 3 plus a different marker in frame 7 both appear in the final output.
        self.accumulated_detections: Dict[int, MarkerDetection] = {}
        self.last_frame: Optional[np.ndarray] = None  # Keep last frame for combined overlay
        self.snapshot_saved = False
        self.printed_once = False

        # Camera intrinsics (updated from CameraInfo)
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

        # QoS — use RELIABLE + TRANSIENT_LOCAL to match RealSense image publisher
        qos_image = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # QoS for camera_info — use VOLATILE (sensor data QoS)
        qos_camera_info = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.create_subscription(Image, args.color_topic, self._cb_color, qos_image)
        self.create_subscription(CameraInfo, args.camera_info_topic, self._cb_camera_info, qos_camera_info)

        # Publishers
        self.overlay_pub = self.create_publisher(Image, args.overlay_topic, 1)
        self.detections_pub = self.create_publisher(StringMsg, args.detections_topic, 1)

        self.get_logger().info(
            f"Single-arm ArUco visual started. Tracking {len(self.marker_map)} markers: "
            f"{self.marker_map}"
        )
        self.get_logger().info(f"  Color topic: {args.color_topic}")
        self.get_logger().info(f"  Overlay topic: {args.overlay_topic}")
        self.get_logger().info(f"  Marker length: {args.marker_length * 1000:.1f} mm")

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _cb_camera_info(self, msg: CameraInfo) -> None:
        """Update camera intrinsics from CameraInfo."""
        if self.camera_matrix is not None:
            return
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.array(msg.d, dtype=np.float64)
        if k[0, 0] > 0:
            self.camera_matrix = k
            self.dist_coeffs = d if len(d) >= 4 else np.zeros(5)
            self.get_logger().info(
                f"  Camera intrinsics received: fx={k[0,0]:.1f} fy={k[1,1]:.1f} "
                f"cx={k[0,2]:.1f} cy={k[1,2]:.1f}"
            )

    def _cb_color(self, msg: Image) -> None:
        """Process incoming color image: detect markers, draw overlay, publish."""
        # Convert ROS Image -> OpenCV
        if self.bridge:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        else:
            enc = msg.encoding if msg.encoding else "bgr8"
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        self.frame_count += 1

        # Detect markers
        detections = self._detect_markers(frame)
        self.last_detections = detections
        self.last_frame = frame.copy()

        # Track best detection (most markers) across all frames
        if len(detections) > len(self.best_detections):
            self.best_detections = list(detections)

        # Accumulate detections by marker ID across ALL frames
        for det in detections:
            self.accumulated_detections[det.marker_id] = det

        # Draw annotated overlay
        overlay = self._draw_overlay(frame.copy(), detections)

        # Publish overlay image
        if self.overlay_pub.get_subscription_count() > 0 or True:  # Always publish
            if self.bridge:
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            else:
                overlay_msg = Image()
                overlay_msg.height = overlay.shape[0]
                overlay_msg.width = overlay.shape[1]
                overlay_msg.encoding = "bgr8"
                overlay_msg.step = overlay.shape[1] * 3
                overlay_msg.data = overlay.tobytes()
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

        # Publish detections JSON
        if self.detections_pub.get_subscription_count() > 0:
            det_msg = StringMsg()
            det_msg.data = self._detections_to_json(detections)
            self.detections_pub.publish(det_msg)

        # Log first few frames
        if self.frame_count <= 2:
            ids = [f"ID={d.marker_id}({d.label})" for d in detections]
            self.get_logger().info(f"  Frame {self.frame_count}: {len(detections)} markers {ids}")
        elif self.frame_count % 10 == 0:
            acc_ids = sorted(self.accumulated_detections.keys())
            self.get_logger().info(
                f"  Frame {self.frame_count}: {len(detections)} markers, "
                f"best frame={len(self.best_detections)}, "
                f"accumulated={len(self.accumulated_detections)} IDs={acc_ids}"
            )

        # Save snapshot after collecting frames.
        # Use accumulated detections (union of all frames) drawn on the last frame.
        if self.args.save_snapshot and not self.snapshot_saved and self.frame_count >= 5:
            combined = list(self.accumulated_detections.values())
            if combined and self.last_frame is not None:
                snap_overlay = self._draw_overlay(self.last_frame.copy(), combined)
                cv2.imwrite(self.args.save_snapshot, snap_overlay)
                self.get_logger().info(
                    f"Snapshot saved: {self.args.save_snapshot} "
                    f"({len(combined)} markers accumulated across {self.frame_count} frames)"
                )
            elif self.best_detections:
                snap_overlay = self._draw_overlay(frame.copy(), self.best_detections)
                cv2.imwrite(self.args.save_snapshot, snap_overlay)
                self.get_logger().info(
                    f"Snapshot saved: {self.args.save_snapshot} "
                    f"({len(self.best_detections)} markers in best frame)"
                )
            self.snapshot_saved = True

        # Save frames continuously if requested
        if self.args.save_frames_dir:
            os.makedirs(self.args.save_frames_dir, exist_ok=True)
            path = os.path.join(self.args.save_frames_dir, f"frame_{self.frame_count:06d}.png")
            cv2.imwrite(path, overlay)

        # Print once and exit (use accumulated detections — all markers seen across all frames)
        if self.args.print_once and not self.printed_once and self.frame_count >= 5:
            combined = list(self.accumulated_detections.values())
            to_print = combined if combined else (self.best_detections if self.best_detections else detections)
            self._print_detections(to_print)
            self.printed_once = True
            raise SystemExit(0)

    # ── Detection ───────────────────────────────────────────────────────────

    def _detect_markers(self, frame: np.ndarray) -> List[MarkerDetection]:
        """Detect ArUco markers using the same pipeline as camera_aruco.py."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the same detection approach as camera_aruco.py
        dictionary = _get_aruco_dictionary("DICT_4X4_50") if _get_aruco_dictionary else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = _get_detector_parameters("default") if _get_detector_parameters else None

        try:
            # Try new API first (OpenCV 4.7+)
            detector = cv2.aruco.ArucoDetector(dictionary, params or cv2.aruco.DetectorParameters())
            corners_list, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Legacy API (OpenCV 4.5-4.6)
            corners_list, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        detections: List[MarkerDetection] = []
        if ids is None or len(ids) == 0:
            return detections

        # Estimate poses
        rvecs = tvecs = None
        if self.camera_matrix is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners_list, self.marker_length, self.camera_matrix, self.dist_coeffs
            )

        for i, marker_id_raw in enumerate(ids.flatten()):
            marker_id = int(marker_id_raw)
            label = self.marker_map.get(marker_id, f"unknown_{marker_id}")
            corners = corners_list[i].reshape(-1, 2)
            center = tuple(corners.mean(axis=0))

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

    # ── Drawing ─────────────────────────────────────────────────────────────

    def _draw_overlay(
        self, frame: np.ndarray, detections: List[MarkerDetection]
    ) -> np.ndarray:
        """Draw annotated overlay on the frame."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw each detected marker
        for det in detections:
            color = det.color_bgr
            corners = det.corners.astype(int)

            # Marker outline (thick colored border)
            cv2.polylines(frame, [corners], True, color, 3)

            # Center crosshair
            cx, cy = int(det.center_px[0]), int(det.center_px[1])
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

            # Label text with background
            label_text = f"{det.label} [ID={det.marker_id}]"
            (tw, th), baseline = cv2.getTextSize(label_text, font, 0.6, 2)

            # Position label above the top-left corner
            label_x = int(corners[0][0])
            label_y = max(int(corners[0][1]) - 8, th + 8)

            # Background rectangle
            cv2.rectangle(frame,
                          (label_x - 3, label_y - th - baseline - 3),
                          (label_x + tw + 5, label_y + 5),
                          color, -1)
            # White text
            cv2.putText(frame, label_text, (label_x, label_y - 2),
                        font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Position text below label
            if det.tvec is not None:
                pos_text = det.position_str
                (tw2, th2), _ = cv2.getTextSize(pos_text, font, 0.4, 1)
                pos_y = label_y + 4
                cv2.rectangle(frame,
                              (label_x - 3, pos_y - 2),
                              (label_x + tw2 + 5, pos_y + th2 + 4),
                              (40, 40, 40), -1)
                cv2.putText(frame, pos_text, (label_x, pos_y + th2),
                            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw 3D axis if pose is known
            if det.rvec is not None and det.tvec is not None and self.camera_matrix is not None:
                cv2.drawFrameAxes(
                    frame, self.camera_matrix, self.dist_coeffs,
                    det.rvec, det.tvec, self.marker_length * 1.5, 2
                )

        # ── Status bar at top ───────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 0), -1)
        status = f"Frame {self.frame_count} | {len(detections)} markers"
        if detections:
            status += " | " + ", ".join(d.label for d in detections)
        cv2.putText(frame, status, (8, 23), font, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        # ── Legend at bottom ────────────────────────────────────────────────
        legend_y = h - 8
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        for i, (label, mid) in enumerate(DEFAULT_MARKERS if not self.args.markers else self.args.markers):
            color = COLORS.get(label, COLORS["unknown"])
            detected = any(d.label == label for d in detections)
            x = 8 + i * 170
            text = f"ID={mid}: {label}"
            thickness = 2 if detected else 1
            text_color = color if detected else (80, 80, 80)
            cv2.putText(frame, text, (x, legend_y), font, 0.4, text_color, thickness, cv2.LINE_AA)

        return frame

    # ── Output ──────────────────────────────────────────────────────────────

    def _detections_to_json(self, detections: List[MarkerDetection]) -> str:
        result = {
            "timestamp": time.time(),
            "frame": self.frame_count,
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
                entry["position_camera_m"] = det.tvec.tolist()
            if det.rvec is not None:
                entry["rotation_vec"] = det.rvec.tolist()
            result["markers"].append(entry)
        return json.dumps(result)

    def _print_detections(self, detections: List[MarkerDetection]) -> None:
        print(f"\n{'='*60}")
        print(f"  Single-Arm ArUco Snapshot (Frame {self.frame_count})")
        print(f"{'='*60}")
        if not detections:
            print("  No markers detected")
        else:
            for det in detections:
                print(f"  [{det.label}] ID={det.marker_id}")
                print(f"    Center: ({det.center_px[0]:.1f}, {det.center_px[1]:.1f}) px")
                if det.tvec is not None:
                    t = det.tvec.flatten()
                    print(f"    Position (camera frame): x={t[0]:.4f} y={t[1]:.4f} z={t[2]:.4f} m")
                if det.corners is not None:
                    print(f"    Corners (px): {det.corners.tolist()}")
                print()
        print(f"{'='*60}\n")


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = SingleArmArucoVisual(args)
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    except Exception as e:
        # Catch ExternalShutdownException and other ROS errors
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            # Ignore double-shutdown errors
            pass


if __name__ == "__main__":
    main()
