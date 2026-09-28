#!/usr/bin/env python3
"""
aruco_alignment_viewer.py
─────────────────────────
Live camera overlay for spring cap alignment guidance.

Subscribes to the RealSense color stream and annotates every frame with:
  • Detected ArUco marker outlines  (green = within tolerance, orange = needs adjustment)
  • Dashed yellow target-zone boxes showing WHERE each spring cap must end up
  • Cyan guide lines at the effective z-reference row (midpoint between cap and platform)
  • Blue dashed guide line at the metal platform row
  • Directional arrows from the current cap centre to the target row
  • Semi-transparent HUD: z-trim recommendations, spring roll, alignment status

Annotated images are published to:
  /spring_monitor/aruco_alignment/annotated_image

View with:
  ros2 run rqt_image_view rqt_image_view  →  select the topic above
  # or, if DISPLAY is available:
  python3 tools/aruco_alignment_viewer.py --show-window

The viewer is also launched automatically by dual_hardware_variable_stiffness.launch.py
when  enable_depth_camera:=true  and  show_camera_feed:=true  (window mode), or
always in headless-publish mode when  enable_depth_camera:=true.

Usage:
  source /opt/ros/humble/setup.bash
  source /workspaces/omx_ros2/ws/install/setup.bash
  python3 tools/aruco_alignment_viewer.py
  python3 tools/aruco_alignment_viewer.py --show-window
  python3 tools/aruco_alignment_viewer.py --show-window --display-scale 0.75
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV (cv2) is not available.", file=sys.stderr)
    sys.exit(1)

try:
    from tools.ablation_config import (
        DEFAULT_ARUCO_DICTIONARY_NAME,
        DEFAULT_ARUCO_MARKER_LENGTH_M,
        DEFAULT_ARUCO_METAL_PLATFORM_MARKER_ID,
        DEFAULT_ARUCO_ROBOT1_MARKER_ID,
        DEFAULT_ARUCO_ROBOT2_MARKER_ID,
        DEFAULT_ARUCO_TARGET_ROW_FRACTION,
    )
except ImportError:
    from ablation_config import (
        DEFAULT_ARUCO_DICTIONARY_NAME,
        DEFAULT_ARUCO_MARKER_LENGTH_M,
        DEFAULT_ARUCO_METAL_PLATFORM_MARKER_ID,
        DEFAULT_ARUCO_ROBOT1_MARKER_ID,
        DEFAULT_ARUCO_ROBOT2_MARKER_ID,
        DEFAULT_ARUCO_TARGET_ROW_FRACTION,
    )

try:
    from tools.aruco_alignment_utils import (
        _camera_matrix_from_info,
        _get_aruco_dictionary,
        _get_detector_parameters,
    )
except ImportError:
    from aruco_alignment_utils import (
        _camera_matrix_from_info,
        _get_aruco_dictionary,
        _get_detector_parameters,
    )

try:
    from tools.depth_frame_utils import decode_sensor_image
except ImportError:
    from depth_frame_utils import decode_sensor_image

# ── Topic ──────────────────────────────────────────────────────────────────────
ANNOTATED_IMAGE_TOPIC = "/spring_monitor/aruco_alignment/annotated_image"
DEFAULT_COLOR_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_COLOR_CAMERA_INFO_TOPIC = "/camera/color/camera_info"

# ── Thresholds (mirrors harness limits) ────────────────────────────────────────
Z_TRIM_OK_M = 0.006   # 6 mm
ROLL_OK_DEG = 2.0

# ── BGR palette ────────────────────────────────────────────────────────────────
_GREEN  = ( 50, 215,  50)   # cap detected, within tolerance
_ORANGE = ( 30, 140, 255)   # cap detected, out of tolerance
_GREY   = (120, 120, 120)   # cap not detected
_BLUE   = (210, 140,  20)   # metal platform marker
_YELLOW = (  0, 230, 230)   # target zone box
_CYAN   = ( 30, 230, 255)   # target guide line per cap
_WHITE  = (230, 230, 230)   # spring roll line
_DARK   = ( 18,  18,  18)   # HUD background
_G_GOOD = ( 40, 210, 110)   # HUD good value
_R_BAD  = ( 50,  70, 240)   # HUD bad value


# ── Drawing helpers ─────────────────────────────────────────────────────────────

def _finite(v: float) -> bool:
    return not math.isnan(v) and not math.isinf(v)


def _draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color,
    thickness: int = 2,
    dash: int = 12,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    for sx, sy, ex, ey in [
        (x1, y1, x2, y1), (x2, y1, x2, y2),
        (x2, y2, x1, y2), (x1, y2, x1, y1),
    ]:
        dx, dy = ex - sx, ey - sy
        dist = math.hypot(dx, dy)
        if dist < 1:
            continue
        n = max(1, int(dist / (dash * 2)))
        for k in range(n):
            t0 = k * 2 * dash / dist
            t1 = min(1.0, (k * 2 + 1) * dash / dist)
            cv2.line(
                img,
                (int(sx + dx * t0), int(sy + dy * t0)),
                (int(sx + dx * t1), int(sy + dy * t1)),
                color, thickness,
            )


def _draw_dashed_hline(
    img: np.ndarray, row: int, color, thickness: int = 1, dash: int = 18
) -> None:
    w = img.shape[1]
    for x in range(0, w, dash * 2):
        cv2.line(img, (x, row), (min(x + dash, w - 1), row), color, thickness)


def _draw_marker_outline(
    img: np.ndarray, corners: np.ndarray, color, thickness: int = 2
) -> Tuple[int, int]:
    """Outline a detected marker and return its (cx, cy) centre."""
    pts = corners.reshape(-1, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.circle(img, (cx, cy), 4, color, -1)
    return cx, cy


def _label(img: np.ndarray, text: str, origin: Tuple[int, int], color,
           scale: float = 0.44) -> None:
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _image_to_ros(bgr: np.ndarray, stamp, frame_id: str) -> Image:
    h, w = bgr.shape[:2]
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = h
    msg.width = w
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = bgr.tobytes()
    return msg


# ── Node ────────────────────────────────────────────────────────────────────────

class ArucoAlignmentViewer(Node):
    def __init__(
        self,
        image_topic: str,
        camera_info_topic: str,
        robot1_marker_id: int,
        robot2_marker_id: int,
        metal_platform_marker_id: Optional[int],
        marker_length_m: float,
        dictionary_name: str,
        target_row_fraction: float,
        z_trim_limit_m: float,
        show_window: bool,
        display_scale: float,
    ) -> None:
        super().__init__("aruco_alignment_viewer")
        self._r1_id = int(robot1_marker_id)
        self._r2_id = int(robot2_marker_id)
        self._plat_id = (
            int(metal_platform_marker_id)
            if metal_platform_marker_id is not None
            else None
        )
        self._marker_len = float(marker_length_m)
        self._dict_name = dictionary_name
        self._target_row_frac = float(target_row_fraction)
        self._z_limit = float(z_trim_limit_m)
        self._show_window = show_window
        self._scale = float(display_scale)
        self._win = "Spring Alignment — live"

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._img: Optional[Image] = None
        self._info: Optional[CameraInfo] = None
        self.create_subscription(
            Image, image_topic, lambda m: setattr(self, "_img", m), qos
        )
        self.create_subscription(
            CameraInfo, camera_info_topic, lambda m: setattr(self, "_info", m), qos
        )
        self._pub = self.create_publisher(Image, ANNOTATED_IMAGE_TOPIC, 1)
        self.create_timer(0.25, self._tick)  # ~4 FPS — matches in-container RealSense throughput

        if show_window:
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            base_w = int(640 * display_scale)
            base_h = int(480 * display_scale)
            cv2.resizeWindow(self._win, base_w, base_h)

        self.get_logger().info(
            f"Alignment viewer ready — annotated feed on {ANNOTATED_IMAGE_TOPIC}"
        )

    def _tick(self) -> None:
        if self._img is None or self._info is None:
            return
        try:
            bgr = self._render(self._img, self._info)
        except Exception as exc:
            self.get_logger().warn(f"Render error: {exc}", throttle_duration_sec=5.0)
            return
        self._pub.publish(
            _image_to_ros(bgr, self._img.header.stamp, self._img.header.frame_id)
        )
        if self._show_window:
            disp = (
                bgr
                if self._scale == 1.0
                else cv2.resize(
                    bgr,
                    (int(bgr.shape[1] * self._scale), int(bgr.shape[0] * self._scale)),
                )
            )
            cv2.imshow(self._win, disp)
            cv2.waitKey(1)

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self, image_msg: Image, info_msg: CameraInfo) -> np.ndarray:
        # Decode to BGR
        raw = decode_sensor_image(image_msg)
        if raw.ndim == 2:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif raw.shape[2] == 3:
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            bgr = raw.copy()
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ArUco detection
        dictionary = _get_aruco_dictionary(self._dict_name)
        params = _get_detector_parameters()
        corners_list, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=params
        )
        ids_flat = (
            []
            if ids is None
            else [int(v) for v in np.asarray(ids).reshape(-1)]
        )

        # Gather corners, centres, depths
        camera_matrix, dist_coeffs = _camera_matrix_from_info(info_msg)
        fy = float(camera_matrix[1, 1]) if camera_matrix is not None else 0.0

        det_corners: Dict[int, np.ndarray] = {}
        det_centers: Dict[int, Tuple[int, int]] = {}
        det_depths: Dict[int, float] = {}

        for i, mid in enumerate(ids_flat):
            c = np.asarray(corners_list[i]).reshape(-1, 2)
            det_corners[mid] = c
            det_centers[mid] = (int(np.mean(c[:, 0])), int(np.mean(c[:, 1])))

        if ids_flat and camera_matrix is not None:
            try:
                _, tvecs_raw, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners_list, self._marker_len, camera_matrix, dist_coeffs
                )
                for i, mid in enumerate(ids_flat):
                    d = float(np.asarray(tvecs_raw[i]).reshape(-1)[2])
                    if d > 0.0:
                        det_depths[mid] = d
            except Exception:
                pass

        # Platform row
        platform_row: Optional[float] = None
        if self._plat_id is not None and self._plat_id in det_centers:
            _, py = det_centers[self._plat_id]
            if self._plat_id in det_depths:
                platform_row = float(py)

        fallback_row = float(h) * self._target_row_frac

        # Per-cap data
        caps: Dict[str, dict] = {}
        for slot, mid in [("robot1", self._r1_id), ("robot2", self._r2_id)]:
            info: dict = {"mid": mid, "seen": mid in det_centers}
            if info["seen"]:
                cx, cy = det_centers[mid]
                info["cx"], info["cy"] = cx, cy
                info["corners"] = det_corners[mid]
                depth = det_depths.get(mid, 0.0)
                eff_row = (
                    (float(cy) + platform_row) / 2.0
                    if platform_row is not None
                    else fallback_row
                )
                info["target_row"] = eff_row
                c = info["corners"]
                info["box_r"] = max(24, int(
                    max(
                        np.linalg.norm(c[0] - c[1]),
                        np.linalg.norm(c[1] - c[2]),
                    ) * 1.6
                ))
                if fy > 0 and depth > 0:
                    info["trim_m"] = -(float(cy) - eff_row) * depth / fy
                else:
                    info["trim_m"] = float("nan")
            caps[slot] = info

        # ── Draw platform marker + guide line ─────────────────────────────
        if self._plat_id is not None and self._plat_id in det_corners:
            _draw_marker_outline(bgr, det_corners[self._plat_id], _BLUE, thickness=2)
            cx_p, cy_p = det_centers[self._plat_id]
            _label(bgr, f"ID{self._plat_id} platform", (cx_p + 8, cy_p - 6), _BLUE)
        if platform_row is not None:
            _draw_dashed_hline(bgr, int(platform_row), _BLUE, thickness=1, dash=14)
            _label(bgr, "platform row", (6, int(platform_row) - 6), _BLUE, scale=0.40)

        # ── Draw per-cap overlays ─────────────────────────────────────────
        for slot, info in caps.items():
            mid = info["mid"]
            trim = info.get("trim_m", float("nan"))
            trim_ok = _finite(trim) and abs(trim) <= self._z_limit
            cap_col = (
                _GREEN if (info["seen"] and trim_ok)
                else (_ORANGE if info["seen"] else _GREY)
            )

            if not info["seen"]:
                # Ghost target box at fallback row in the expected screen half
                gx = w // 4 if slot == "robot1" else 3 * w // 4
                gy = int(fallback_row)
                r = 36
                _draw_dashed_rect(bgr, (gx - r, gy - r), (gx + r, gy + r), _GREY, thickness=1)
                _label(bgr, f"{slot} — not detected", (gx - r, gy - r - 7), _GREY)
                continue

            cx, cy = info["cx"], info["cy"]
            target_row = int(round(info["target_row"]))
            box_r = info["box_r"]

            # 1. Detected marker outline (green/orange)
            _draw_marker_outline(bgr, info["corners"], cap_col, thickness=2)
            _label(bgr, f"ID{mid}", (cx + 6, cy - 8), cap_col)

            # 2. Dashed target-zone box at target row, same column
            _draw_dashed_rect(
                bgr,
                (cx - box_r, target_row - box_r),
                (cx + box_r, target_row + box_r),
                _YELLOW, thickness=2,
            )
            _label(bgr, "target", (cx - box_r, target_row - box_r - 7), _YELLOW)

            # 3. Cyan guide line at effective target row
            _draw_dashed_hline(bgr, target_row, _CYAN, thickness=1, dash=20)

            # 4. Arrow from current centre to target (skip if already there)
            if abs(cy - target_row) > 5:
                arr_col = _GREEN if trim_ok else _ORANGE
                cv2.arrowedLine(
                    bgr, (cx, cy), (cx, target_row), arr_col, 2, tipLength=0.25
                )

            # 5. Trim label between arrow endpoints
            trim_str = f"{trim * 1000:+.1f} mm" if _finite(trim) else "?"
            mid_y = (cy + target_row) // 2
            _label(bgr, f"{slot}: {trim_str}", (cx + 10, mid_y), cap_col, scale=0.42)

        # Spring roll line between the two caps
        r1, r2 = caps.get("robot1", {}), caps.get("robot2", {})
        roll_deg = float("nan")
        if r1.get("seen") and r2.get("seen"):
            dx = float(r2["cx"] - r1["cx"])
            dy = float(r2["cy"] - r1["cy"])
            if abs(dx) > 1.0:
                roll_deg = math.degrees(math.atan2(dy, dx))
            cv2.line(bgr, (r1["cx"], r1["cy"]), (r2["cx"], r2["cy"]), _WHITE, 1)

        # HUD
        self._draw_hud(bgr, r1, r2, roll_deg, platform_row is not None)
        return bgr

    def _draw_hud(
        self,
        bgr: np.ndarray,
        r1: dict,
        r2: dict,
        roll_deg: float,
        platform_visible: bool,
    ) -> None:
        trim1 = r1.get("trim_m", float("nan"))
        trim2 = r2.get("trim_m", float("nan"))
        r1_ok = r1.get("seen") and _finite(trim1) and abs(trim1) <= self._z_limit
        r2_ok = r2.get("seen") and _finite(trim2) and abs(trim2) <= self._z_limit
        roll_ok = _finite(roll_deg) and abs(roll_deg) <= ROLL_OK_DEG
        all_ok = r1.get("seen") and r2.get("seen") and r1_ok and r2_ok and roll_ok

        def _ts(v: float, seen: bool) -> str:
            if not seen:
                return "---"
            return f"{v * 1000:+.1f} mm" if _finite(v) else "?"

        rows = [
            ("STATUS",    "ALIGNED" if all_ok else "ADJUST",
             _G_GOOD if all_ok else _R_BAD),
            ("platform",  "visible" if platform_visible else "not seen",
             _G_GOOD if platform_visible else _GREY),
            ("r1 z-trim", _ts(trim1, r1.get("seen", False)),
             _G_GOOD if r1_ok else _R_BAD),
            ("r2 z-trim", _ts(trim2, r2.get("seen", False)),
             _G_GOOD if r2_ok else _R_BAD),
            ("roll",      f"{roll_deg:+.1f}°" if _finite(roll_deg) else "---",
             _G_GOOD if roll_ok else _R_BAD),
        ]

        line_h, pad, box_w = 22, 8, 215
        box_h = len(rows) * line_h + pad * 2
        roi = bgr[pad: pad + box_h, pad: pad + box_w]
        roi[:] = (
            roi.astype(np.float32) * 0.35
            + np.array(_DARK, dtype=np.float32) * 0.65
        ).astype(np.uint8)
        for i, (label, val, color) in enumerate(rows):
            y = pad * 2 + i * line_h
            cv2.putText(
                bgr, f"{label}: {val}",
                (pad + 6, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live ArUco alignment overlay — publishes annotated image and optionally shows a window"
    )
    ap.add_argument("--image-topic",        default=DEFAULT_COLOR_IMAGE_TOPIC)
    ap.add_argument("--camera-info-topic",  default=DEFAULT_COLOR_CAMERA_INFO_TOPIC)
    ap.add_argument("--robot1-marker-id",   type=int, default=DEFAULT_ARUCO_ROBOT1_MARKER_ID)
    ap.add_argument("--robot2-marker-id",   type=int, default=DEFAULT_ARUCO_ROBOT2_MARKER_ID)
    ap.add_argument("--metal-platform-marker-id", type=int,
                    default=DEFAULT_ARUCO_METAL_PLATFORM_MARKER_ID,
                    help="Set to -1 to disable platform-midpoint z reference")
    ap.add_argument("--marker-length-m",    type=float, default=DEFAULT_ARUCO_MARKER_LENGTH_M)
    ap.add_argument("--dictionary",         default=DEFAULT_ARUCO_DICTIONARY_NAME)
    ap.add_argument("--target-row-fraction", type=float, default=DEFAULT_ARUCO_TARGET_ROW_FRACTION,
                    help="Fallback target row when platform marker is not visible (0=top, 1=bottom)")
    ap.add_argument("--z-trim-limit-m",     type=float, default=Z_TRIM_OK_M,
                    help="Threshold for marking a trim as OK (green vs orange)")
    ap.add_argument("--show-window",        action="store_true",
                    help="Open a local OpenCV display window (requires DISPLAY to be set)")
    ap.add_argument("--display-scale",      type=float, default=1.0,
                    help="Window scale factor, e.g. 0.5 for half size")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    plat_id = args.metal_platform_marker_id if args.metal_platform_marker_id >= 0 else None
    node = ArucoAlignmentViewer(
        image_topic=args.image_topic,
        camera_info_topic=args.camera_info_topic,
        robot1_marker_id=args.robot1_marker_id,
        robot2_marker_id=args.robot2_marker_id,
        metal_platform_marker_id=plat_id,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary,
        target_row_fraction=args.target_row_fraction,
        z_trim_limit_m=args.z_trim_limit_m,
        show_window=args.show_window,
        display_scale=args.display_scale,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if args.show_window:
            cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
