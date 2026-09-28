#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_COLOR_ENCODINGS = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
}
_MONO8_ENCODINGS = {"mono8", "8uc1"}
_MONO16_ENCODINGS = {"mono16", "16uc1"}


def image_stamp_ns(image_msg: Any) -> int:
    header = getattr(image_msg, "header", None)
    stamp = getattr(header, "stamp", None)
    sec = int(getattr(stamp, "sec", 0) or 0)
    nanosec = int(getattr(stamp, "nanosec", 0) or 0)
    return sec * 1_000_000_000 + nanosec


def suggested_image_extension(encoding: str) -> str:
    normalized = str(encoding).lower()
    if normalized in _COLOR_ENCODINGS:
        return ".ppm"
    if normalized in _MONO8_ENCODINGS or normalized in _MONO16_ENCODINGS:
        return ".pgm"
    raise ValueError(f"Unsupported image encoding: {encoding!r}")


def decode_sensor_image(image_msg: Any) -> np.ndarray:
    encoding = str(getattr(image_msg, "encoding", "")).lower()
    width = int(getattr(image_msg, "width"))
    height = int(getattr(image_msg, "height"))
    step = int(getattr(image_msg, "step"))
    data = bytes(getattr(image_msg, "data"))

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    if encoding in _COLOR_ENCODINGS:
        channels = _COLOR_ENCODINGS[encoding]
        row_width = width * channels
        if step < row_width:
            raise ValueError(
                f"Step {step} is smaller than expected color row width {row_width}"
            )
        array = np.frombuffer(data, dtype=np.uint8)
        expected = height * step
        if array.size < expected:
            raise ValueError(
                f"Image data too short for {height} rows with step {step}: "
                f"got {array.size} bytes"
            )
        array = array[:expected].reshape(height, step)[:, :row_width]
        array = array.reshape(height, width, channels)
        if encoding.startswith("bgr"):
            array = array[..., [2, 1, 0]]
        if channels == 4:
            array = array[..., :3]
        return np.ascontiguousarray(array)

    if encoding in _MONO8_ENCODINGS:
        if step < width:
            raise ValueError(
                f"Step {step} is smaller than expected mono8 row width {width}"
            )
        array = np.frombuffer(data, dtype=np.uint8)
        expected = height * step
        if array.size < expected:
            raise ValueError(
                f"Image data too short for {height} rows with step {step}: "
                f"got {array.size} bytes"
            )
        array = array[:expected].reshape(height, step)[:, :width]
        return np.ascontiguousarray(array)

    if encoding in _MONO16_ENCODINGS:
        if step < width * 2:
            raise ValueError(
                f"Step {step} is smaller than expected mono16 row width {width * 2}"
            )
        array = np.frombuffer(data, dtype=np.uint16)
        if bool(getattr(image_msg, "is_bigendian", False)) != (sys.byteorder == "big"):
            array = array.byteswap()
        row_width = step // 2
        expected = height * row_width
        if array.size < expected:
            raise ValueError(
                f"Image data too short for {height} rows with step {step}: "
                f"got {array.size * 2} bytes"
            )
        array = array[:expected].reshape(height, row_width)[:, :width]
        return np.ascontiguousarray(array)

    raise ValueError(f"Unsupported image encoding: {getattr(image_msg, 'encoding', None)!r}")


def save_sensor_image(image_msg: Any, file_path: str) -> Dict[str, Any]:
    array = decode_sensor_image(image_msg)
    if array.ndim == 3:
        _write_ppm_rgb(array, file_path)
    elif array.dtype == np.uint8:
        _write_pgm_u8(array, file_path)
    elif array.dtype == np.uint16:
        _write_pgm_u16(array, file_path)
    else:
        raise ValueError(f"Unsupported decoded dtype: {array.dtype}")

    return {
        "path": file_path,
        "encoding": str(getattr(image_msg, "encoding", "")),
        "width": int(getattr(image_msg, "width")),
        "height": int(getattr(image_msg, "height")),
        "stamp_ns": image_stamp_ns(image_msg),
    }


def load_saved_sensor_image(file_path: str) -> np.ndarray:
    with open(file_path, "rb") as handle:
        magic = _read_netpbm_token(handle)
        width = int(_read_netpbm_token(handle))
        height = int(_read_netpbm_token(handle))
        max_value = int(_read_netpbm_token(handle))
        payload = handle.read()

    if magic == b"P6":
        if max_value != 255:
            raise ValueError(f"Unsupported PPM max value: {max_value}")
        expected = width * height * 3
        if len(payload) < expected:
            raise ValueError(
                f"PPM payload too short for {width}x{height} RGB image: {len(payload)} bytes"
            )
        return np.frombuffer(payload[:expected], dtype=np.uint8).reshape(height, width, 3)

    if magic == b"P5":
        if max_value <= 255:
            expected = width * height
            if len(payload) < expected:
                raise ValueError(
                    f"PGM payload too short for {width}x{height} mono8 image: {len(payload)} bytes"
                )
            return np.frombuffer(payload[:expected], dtype=np.uint8).reshape(height, width)
        if max_value == 65535:
            expected = width * height * 2
            if len(payload) < expected:
                raise ValueError(
                    f"PGM payload too short for {width}x{height} mono16 image: {len(payload)} bytes"
                )
            array = np.frombuffer(payload[:expected], dtype=">u2").reshape(height, width)
            return np.array(array, dtype=np.uint16)
        raise ValueError(f"Unsupported PGM max value: {max_value}")

    raise ValueError(f"Unsupported saved image magic: {magic!r}")


def export_saved_sensor_stream_video(
    frame_paths: Sequence[str],
    output_path: str,
    fps: float,
) -> Dict[str, Any]:
    if fps <= 0.0:
        raise ValueError("Video export FPS must be > 0")
    if not frame_paths:
        raise ValueError("At least one saved frame is required for video export")

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ValueError(
            "OpenCV video export requires cv2 in the active Python environment"
        ) from exc

    frames = [load_saved_sensor_image(path) for path in frame_paths]
    prepared_frames, source_kind, preview_meta = _prepare_video_frames(frames)
    height, width = prepared_frames[0].shape[:2]

    for candidate_path, codec_name in _video_writer_candidates(output_path):
        candidate_dir = os.path.dirname(candidate_path)
        if candidate_dir:
            os.makedirs(candidate_dir, exist_ok=True)
        writer = cv2.VideoWriter(
            candidate_path,
            cv2.VideoWriter_fourcc(*codec_name),
            float(fps),
            (width, height),
        )
        if writer is None or not writer.isOpened():
            if writer is not None:
                writer.release()
            continue
        try:
            for frame in prepared_frames:
                writer.write(frame)
        finally:
            writer.release()
        return {
            "path": candidate_path,
            "fps": float(fps),
            "frame_count": len(prepared_frames),
            "width": width,
            "height": height,
            "codec": codec_name,
            "source_kind": source_kind,
            **preview_meta,
        }

    raise ValueError(
        "Unable to open OpenCV VideoWriter for either MP4 or AVI export"
    )


def camera_info_to_dict(camera_info_msg: Any) -> Dict[str, Any]:
    header = getattr(camera_info_msg, "header", None)
    return {
        "frame_id": getattr(header, "frame_id", ""),
        "stamp_ns": image_stamp_ns(camera_info_msg),
        "width": int(getattr(camera_info_msg, "width", 0) or 0),
        "height": int(getattr(camera_info_msg, "height", 0) or 0),
        "distortion_model": str(getattr(camera_info_msg, "distortion_model", "")),
        "d": list(getattr(camera_info_msg, "d", [])),
        "k": list(getattr(camera_info_msg, "k", [])),
        "r": list(getattr(camera_info_msg, "r", [])),
        "p": list(getattr(camera_info_msg, "p", [])),
        "binning_x": int(getattr(camera_info_msg, "binning_x", 0) or 0),
        "binning_y": int(getattr(camera_info_msg, "binning_y", 0) or 0),
    }


def _write_ppm_rgb(array: np.ndarray, file_path: str) -> None:
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError(f"PPM writer expects 3 channels, got {channels}")
    payload = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    with open(file_path, "wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(payload)


def _write_pgm_u8(array: np.ndarray, file_path: str) -> None:
    height, width = array.shape
    payload = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    with open(file_path, "wb") as handle:
        handle.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        handle.write(payload)


def _write_pgm_u16(array: np.ndarray, file_path: str) -> None:
    height, width = array.shape
    payload = np.ascontiguousarray(array, dtype=">u2").tobytes()
    with open(file_path, "wb") as handle:
        handle.write(f"P5\n{width} {height}\n65535\n".encode("ascii"))
        handle.write(payload)


def _read_netpbm_token(handle: Any) -> bytes:
    token = bytearray()
    while True:
        ch = handle.read(1)
        if not ch:
            if token:
                return bytes(token)
            raise ValueError("Unexpected EOF while reading saved image header")
        if ch == b"#":
            handle.readline()
            if token:
                return bytes(token)
            continue
        if ch in b" \t\r\n":
            if token:
                return bytes(token)
            continue
        token.extend(ch)


def _prepare_video_frames(
    frames: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], str, Dict[str, Any]]:
    prepared: List[np.ndarray] = []
    first_shape = tuple(frames[0].shape)
    preview_meta: Dict[str, Any] = {}

    if any(tuple(frame.shape) != first_shape for frame in frames):
        raise ValueError("Saved stream frames must all share the same dimensions")

    if frames[0].ndim == 3:
        if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
            raise ValueError("Saved color stream frames must all be RGB images")
        for frame in frames:
            prepared.append(np.ascontiguousarray(frame[..., ::-1], dtype=np.uint8))
        return prepared, "rgb8", preview_meta

    if any(frame.ndim != 2 for frame in frames):
        raise ValueError("Saved stream frames must be consistently mono or RGB")

    first_dtype = frames[0].dtype
    if any(frame.dtype != first_dtype for frame in frames):
        raise ValueError("Saved mono stream frames must all share the same dtype")

    if np.issubdtype(first_dtype, np.unsignedinteger) and first_dtype.itemsize == 2:
        lo, hi = _mono16_preview_range(frames)
        preview_meta["preview_min"] = int(lo)
        preview_meta["preview_max"] = int(hi)
        for frame in frames:
            normalized = _normalize_mono16_frame(frame, lo, hi)
            prepared.append(np.repeat(normalized[:, :, None], 3, axis=2))
        return prepared, "mono16", preview_meta

    if first_dtype == np.uint8:
        for frame in frames:
            prepared.append(np.repeat(np.ascontiguousarray(frame, dtype=np.uint8)[:, :, None], 3, axis=2))
        return prepared, "mono8", preview_meta

    raise ValueError(f"Unsupported saved frame dtype for video export: {first_dtype}")


def _mono16_preview_range(frames: Sequence[np.ndarray]) -> Tuple[int, int]:
    mins: List[int] = []
    maxs: List[int] = []
    for frame in frames:
        native = np.asarray(frame, dtype=np.uint16)
        nonzero = native[native > 0]
        if nonzero.size:
            mins.append(int(nonzero.min()))
            maxs.append(int(nonzero.max()))
    if not mins:
        return 0, 1
    lo = min(mins)
    hi = max(maxs)
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def _normalize_mono16_frame(frame: np.ndarray, lo: int, hi: int) -> np.ndarray:
    native = np.asarray(frame, dtype=np.uint16)
    if hi <= lo:
        return np.zeros(native.shape, dtype=np.uint8)
    scaled = (native.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _video_writer_candidates(output_path: str) -> List[Tuple[str, str]]:
    base, ext = os.path.splitext(output_path)
    if ext.lower() == ".avi":
        return [(output_path, "MJPG"), (base + ".mp4", "mp4v")]
    return [(output_path, "mp4v"), (base + ".avi", "MJPG")]
