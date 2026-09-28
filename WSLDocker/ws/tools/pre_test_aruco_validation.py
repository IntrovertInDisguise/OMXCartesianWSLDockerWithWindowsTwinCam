#!/usr/bin/env python3
"""
Pre-test ArUco validation: detect marker drift before running the harness test.

This script runs BEFORE the hardware harness test to verify that the physical
setup hasn't drifted from the expected configuration. It:
1. Captures ArUco marker positions
2. Compares the rigid spring-cap spacing against the SDF model
3. Reports drift and passes/fails validation

Usage:
    python3 tools/pre_test_aruco_validation.py [--max-drift-mm 10]

Exit codes:
    0 - Validation passed (drift within tolerance)
    1 - Validation failed (drift exceeds tolerance)
    2 - Could not detect required markers
    3 - Internal validator error (camera/image processing failure)
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:
    cv2 = None
    CvBridge = None

# Expected SDF positions (relative to metal_platform)
EXPECTED_SDF_POSITIONS = {
    'spring_cap_robot1': np.array([-0.0865, 0.0, 0.08]),
    'spring_cap_robot2': np.array([0.0865, 0.0, 0.08]),
}

# Marker ID to SDF link name mapping
MARKER_TO_LINK = {
    1: 'spring_cap_robot1',
    3: 'spring_cap_robot2',
}

REQUIRED_MARKERS = [1, 3]  # spring_cap1, spring_cap2
CAP_SPACING_EXPECTED = float(np.linalg.norm(
    EXPECTED_SDF_POSITIONS['spring_cap_robot2'] - EXPECTED_SDF_POSITIONS['spring_cap_robot1']
))

# The start_position used by the harness is derived from the METAL PLATFORM
# marker (ID 0), NOT the spring-cap markers (1, 3) validated above. A misread
# of marker 0 would feed a wrong start_position yet still pass the cap-spacing
# gate. So we additionally sanity-check marker 0 against its KNOWN WORLD pose —
# the ground/world frame is the only true reference. Other markers can drift, so
# we do NOT validate marker 0 relative to them; we anchor to world directly.
PLATFORM_MARKER_ID = 0
# Marker 0 is glued to the metal platform at a fixed world location.
EXPECTED_PLATFORM_MARKER_WORLD = np.array([-0.12, -0.12, 0.0505])
PLATFORM_MARKER_MAX_DRIFT_MM = 30.0  # generous: only catches gross misreads


def _marker_camera_to_world(tvec_cam, T_cam_to_world):
    """Transform a camera-frame tvec into the world frame."""
    p = np.array([tvec_cam[0], tvec_cam[1], tvec_cam[2], 1.0], dtype=float)
    return (T_cam_to_world @ p)[:3]

VALIDATION_PASSED = 0
VALIDATION_FAILED_DRIFT = 1
VALIDATION_FAILED_NO_MARKERS = 2
VALIDATION_FAILED_INTERNAL = 3


def pretest_frame_path():
    """Path where the diagnostic frame is saved.

    Honors ``OMX_PRETEST_FRAME_DIR`` (exported by the runner so the frame lands
    in the per-run log folder) and falls back to ``/tmp`` for standalone use.
    """
    base_dir = os.environ.get("OMX_PRETEST_FRAME_DIR", "/tmp")
    try:
        os.makedirs(base_dir, exist_ok=True)
    except OSError:
        base_dir = "/tmp"
    return os.path.join(base_dir, "pretest_frame.png")


class PreTestValidator(Node):
    """Pre-test ArUco marker validation node.

    Two detection sources are supported:
      * Image-based (default): subscribes to a camera ``Image`` topic and runs
        OpenCV ArUco detection in-container. Requires cv2 + cv_bridge.
      * UDP (image-free): subscribes to ``/single_arm_aruco/detections_json``
        (published by ``udp_aruco_ros2_receiver.py``) and reads each marker's
        ``position_camera_m`` tvec directly. No camera frames enter the
        container. Pass ``use_udp=True`` to select this mode.
    """

    def __init__(
        self,
        max_drift_mm: float,
        color_topic: str,
        camera_info_topic: str,
        use_udp: bool = False,
        detections_topic: str = "/single_arm_aruco/detections_json",
        camera_extrinsics: str = "",
    ):
        super().__init__('pre_test_aruco_validator')

        self.max_drift_mm = max_drift_mm
        self.use_udp = use_udp
        self.bridge = CvBridge() if CvBridge is not None else None

        # Camera→world extrinsics (needed to world-anchor marker 0, which drives
        # start_position). Loaded from a JSON file produced by aruco_fit_extrinsics.
        self.T_cam_to_world = None
        if camera_extrinsics and os.path.isfile(camera_extrinsics):
            try:
                with open(camera_extrinsics, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                mat = data.get("camera_to_world")
                if mat is not None:
                    self.T_cam_to_world = np.asarray(mat, dtype=float).reshape(4, 4)
                    self.get_logger().info(
                        f"Loaded camera→world extrinsics from {camera_extrinsics}"
                    )
                else:
                    self.get_logger().warning(
                        f"{camera_extrinsics} has no 'camera_to_world' key"
                    )
            except Exception as exc:
                self.get_logger().warning(f"Failed to load extrinsics: {exc}")
        elif camera_extrinsics:
            self.get_logger().warning(f"Extrinsics file not found: {camera_extrinsics}")

        # Camera intrinsics (image mode only)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Detected markers
        self.detected_markers = {}
        self.frame_count = 0
        self.validation_status = VALIDATION_PASSED
        self.validation_error = None
        self.last_detected_ids = []
        self.last_rejected_candidates = 0
        self.last_frame_shape = None

        # ArUco setup (image mode only)
        self.dictionary = None
        self.params = None
        self.marker_length = 0.038  # 38mm markers

        # QoS
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        if self.use_udp:
            self.create_subscription(String, detections_topic, self.cb_detections_json, qos)
            self.get_logger().info(
                f"Pre-test validator started in UDP mode (max drift: {max_drift_mm} mm) "
                f"subscribing to {detections_topic}"
            )
        else:
            if cv2 is None or CvBridge is None:
                self.get_logger().error(
                    "Image-based validation requested but cv2/cv_bridge unavailable"
                )
                self.validation_status = VALIDATION_FAILED_INTERNAL
                self.validation_error = "cv2/cv_bridge unavailable"
                return
            self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.params = cv2.aruco.DetectorParameters_create()
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.create_subscription(CameraInfo, camera_info_topic, self.cb_camera_info, qos)
            self.create_subscription(Image, color_topic, self.cb_image, qos)
            self.get_logger().info(
                f"Pre-test validator started in image mode (max drift: {max_drift_mm} mm)"
            )

    def cb_camera_info(self, msg: CameraInfo):
        """Receive camera intrinsics."""
        if self.camera_matrix is None:
            k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            d = np.array(msg.d, dtype=np.float64)
            if k[0, 0] > 0:
                self.camera_matrix = k
                self.dist_coeffs = d if len(d) >= 4 else np.zeros(5)
                self.get_logger().info(f"Camera intrinsics received: fx={k[0,0]:.1f}")

    def cb_detections_json(self, msg: String):
        """Consume a UDP ArUco detection packet (image-free path)."""
        self.frame_count += 1
        try:
            packet = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"Bad detections_json packet: {exc}")
            return

        markers = packet.get("markers", [])
        ids = [m.get("id") for m in markers]
        self.last_detected_ids = ids
        self.get_logger().info(
            f"UDP frame {self.frame_count}: markers={len(markers)} ids={ids}"
        )

        for m in markers:
            marker_id = m.get("id")
            if marker_id not in REQUIRED_MARKERS and marker_id != PLATFORM_MARKER_ID:
                continue
            pos = m.get("position_camera_m")
            if not pos or len(pos) != 3:
                continue
            self.detected_markers[marker_id] = np.array(
                [float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64
            )

    def cb_image(self, msg: Image):
        """Process image and detect markers."""
        if self.camera_matrix is None:
            self.get_logger().warning("Camera intrinsics not available yet; skipping frame")
            return

        self.frame_count += 1

        try:
            # Convert to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.last_frame_shape = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
            self.last_detected_ids = ids.flatten().tolist() if ids is not None else []
            self.last_rejected_candidates = len(rejected) if rejected is not None else 0

            self.get_logger().info(
                f"Frame {self.frame_count}: intrinsics=OK shape={frame.shape[:2]} "
                f"detected_ids={self.last_detected_ids} rejected_candidates={self.last_rejected_candidates}"
            )

            if self.frame_count == 30:
                try:
                    frame_path = pretest_frame_path()
                    cv2.imwrite(frame_path, frame)
                    self.get_logger().info(f"Saved {frame_path}")
                except Exception as exc:
                    self.get_logger().warning(f"Failed to save {frame_path}: {exc}")

            if ids is not None:
                # Estimate poses
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs
                )

                # Store detected markers
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in REQUIRED_MARKERS or marker_id == PLATFORM_MARKER_ID:
                        tvec = tvecs[i].flatten()
                        self.detected_markers[marker_id] = tvec.copy()
        except Exception as exc:
            self.validation_status = VALIDATION_FAILED_INTERNAL
            self.validation_error = str(exc)
            self.get_logger().error(f"Image processing failed: {exc}")
    
    def validate(self, timeout_s: float = 10.0) -> int:
        """
        Run validation: wait for markers, compute drift, report results.

        Returns one of the validation status constants.
        """
        start_time = time.time()
        self.validation_status = VALIDATION_PASSED
        self.validation_error = None

        # Wait for required markers
        self.get_logger().info("Waiting for required markers...")
        while time.time() - start_time < timeout_s:
            if self.validation_status == VALIDATION_FAILED_INTERNAL:
                return self.validation_status

            rclpy.spin_once(self, timeout_sec=0.1)

            # In UDP mode there are no camera intrinsics to wait for; just look
            # for the spring-cap markers needed for the frame-invariant spacing
            # validation.
            if self.use_udp:
                if all(mid in self.detected_markers for mid in REQUIRED_MARKERS):
                    break
            elif self.camera_matrix is not None and all(
                mid in self.detected_markers for mid in REQUIRED_MARKERS
            ):
                break

        # Check if we got all markers
        missing = [mid for mid in REQUIRED_MARKERS if mid not in self.detected_markers]
        if missing:
            self.get_logger().error(
                f"Required markers not detected: required_ids={REQUIRED_MARKERS} "
                f"detected_ids={self.last_detected_ids} frames_received={self.frame_count}"
            )
            return VALIDATION_FAILED_NO_MARKERS

        # The camera frame can rotate arbitrarily, so compare only the rigid
        # cap-to-cap separation magnitude.
        actual_spacing_vec = self.detected_markers[3] - self.detected_markers[1]
        actual_spacing = float(np.linalg.norm(actual_spacing_vec))
        
        # Compute drift
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("PRE-TEST VALIDATION RESULTS")
        self.get_logger().info("="*70)

        drift_mm = abs(actual_spacing - CAP_SPACING_EXPECTED) * 1000.0
        status = "✅ PASS" if drift_mm <= self.max_drift_mm else "❌ FAIL"
        drift_report = [
            "spring_cap_spacing:",
            f"  Expected spacing: {CAP_SPACING_EXPECTED:.4f} m",
            f"  Actual spacing:   {actual_spacing:.4f} m",
            f"  Drift: {drift_mm:.1f} mm  |  {status}",
        ]
        max_drift = drift_mm / 1000.0

        self.get_logger().info("\n" + "\n".join(drift_report))

        # ── Platform marker (ID 0) world-anchored sanity check ──────────────
        # start_position is derived from marker 0, so a gross misread of it
        # would feed a wrong start even though the cap-spacing gate passed.
        # Anchor to the WORLD frame (the only true reference); other markers can
        # drift, so we do NOT compare marker 0 to them. Requires camera
        # extrinsics to transform the camera-frame tvec into world.
        if PLATFORM_MARKER_ID in self.detected_markers:
            pm_cam = self.detected_markers[PLATFORM_MARKER_ID]
            if self.T_cam_to_world is not None:
                pm_world = _marker_camera_to_world(pm_cam, self.T_cam_to_world)
                platform_drift_mm = float(np.linalg.norm(pm_world - EXPECTED_PLATFORM_MARKER_WORLD)) * 1000.0
                pm_status = "✅ OK" if platform_drift_mm <= PLATFORM_MARKER_MAX_DRIFT_MM else "⚠️  WARN"
                self.get_logger().info(
                    f"platform_marker(0) world=({pm_world[0]:.3f},{pm_world[1]:.3f},{pm_world[2]:.3f}) "
                    f"expected=({EXPECTED_PLATFORM_MARKER_WORLD[0]:.3f},{EXPECTED_PLATFORM_MARKER_WORLD[1]:.3f},"
                    f"{EXPECTED_PLATFORM_MARKER_WORLD[2]:.3f}) drift={platform_drift_mm:.1f} mm  | {pm_status}"
                )
                if platform_drift_mm > PLATFORM_MARKER_MAX_DRIFT_MM:
                    self.get_logger().error(
                        "Platform marker 0 world drift exceeds %.1f mm — possible misread "
                        "or rig displacement. start_position derived from this marker may be "
                        "wrong. Verify camera extrinsics and ArUco before running." % PLATFORM_MARKER_MAX_DRIFT_MM
                    )
            else:
                self.get_logger().warning(
                    "Platform marker 0 detected but camera extrinsics not loaded "
                    "(--camera-extrinsics / OMX_CAMERA_EXTRINSICS) — cannot world-validate "
                    "the marker that drives start_position."
                )
        else:
            self.get_logger().warning(
                "Platform marker 0 NOT detected — cannot sanity-check the marker "
                "that drives start_position. start_position will use known geometry."
            )

        self.get_logger().info("\n" + "="*70)

        # Final verdict
        if max_drift * 1000 <= self.max_drift_mm:
            self.get_logger().info(f"✅ VALIDATION PASSED (max drift: {max_drift*1000:.1f} mm <= {self.max_drift_mm} mm)")
            self.get_logger().info("="*70 + "\n")
            return VALIDATION_PASSED
        else:
            self.get_logger().error(f"❌ VALIDATION FAILED (max drift: {max_drift*1000:.1f} mm > {self.max_drift_mm} mm)")
            self.get_logger().error("Physical setup has drifted. Please check marker placement.")
            self.get_logger().info("="*70 + "\n")
            return VALIDATION_FAILED_DRIFT


def main():
    parser = argparse.ArgumentParser(description="Pre-test ArUco marker validation")
    parser.add_argument("--max-drift-mm", type=float, default=10.0,
                        help="Maximum allowed drift in mm (default: 10.0)")
    parser.add_argument("--color-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--detections-topic", default="/single_arm_aruco/detections_json",
                        help="Topic for UDP image-free detections (used with --use-udp)")
    parser.add_argument("--use-udp", action="store_true",
                        help="Image-free mode: read marker poses from UDP detections_json "
                             "instead of running in-container camera detection")
    parser.add_argument("--timeout-s", type=float, default=10.0,
                        help="Timeout for marker detection (default: 10.0)")
    parser.add_argument(
        "--camera-extrinsics", default=os.environ.get("OMX_CAMERA_EXTRINSICS", ""),
        help="Path to camera_extrinsics.json (camera→world 4×4). Required to "
             "world-anchor the platform marker (ID 0) that drives start_position.",
    )
    args = parser.parse_args()

    rclpy.init()
    validator = PreTestValidator(
        args.max_drift_mm,
        args.color_topic,
        args.camera_info_topic,
        use_udp=args.use_udp,
        detections_topic=args.detections_topic,
        camera_extrinsics=args.camera_extrinsics,
    )
    
    try:
        status = validator.validate(args.timeout_s)
        sys.exit(status)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
