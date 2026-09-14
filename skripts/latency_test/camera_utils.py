#!/usr/bin/env python3
"""Unified camera capture and latency benchmarking utilities for Raspberry Pi.

Provides:
  1. Low-latency Picamera2 acquisition (queue=True, zero queue backlog).
  2. Mock/OpenCV fallback stream for cross-platform simulation and testing.
  3. Standardized latency metric tracking and summary table formatting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")


class FrameRequest:
    """Wrapper around captured frame and metadata."""

    def __init__(self, frame_rgb: np.ndarray, metadata: dict[str, Any]):
        self.frame_rgb = frame_rgb
        self.metadata = metadata

    def make_array(self, name: str = "main") -> np.ndarray:
        return self.frame_rgb

    def get_metadata(self) -> dict[str, Any]:
        return self.metadata

    def release(self) -> None:
        pass


class CameraStream:
    """Unified camera interface supporting Picamera2 and OpenCV/Mock sources."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 400,
        fps: float = 72.0,
        shutter_us: int = 1000,
        gain: float = 0.0,
        source: str | None = None,
        use_mock: bool = False,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.shutter_us = shutter_us
        self.gain = gain
        self.source = source
        self.use_mock = use_mock
        self.is_picamera = False
        self._picam2 = None
        self._cv_cap = None
        self._frame_count = 0
        self._start_monotonic_ns = 0

    def start(self) -> None:
        self._start_monotonic_ns = time.monotonic_ns()

        if self.use_mock:
            print("\n" + "!" * 60)
            print(" [CAMERA MODE] SIMULATION / MOCK STREAM ACTIVE (--mock requested)")
            print(" Generating synthetic dark IR frame with 4 simulated test dots.")
            print("!" * 60 + "\n")
            return

        if self.source is not None:
            src = int(self.source) if self.source.isdigit() else self.source
            self._cv_cap = cv2.VideoCapture(src)
            if not self._cv_cap.isOpened():
                raise RuntimeError(f"Cannot open video/camera source: {self.source}")
            self._cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cv_cap.set(cv2.CAP_PROP_FPS, self.fps)
            print(f"\n[CAMERA MODE] OpenCV Video Source Active: {self.source}\n")
            return

        # Attempt to load Picamera2 (including searching system dist-packages on Pi)
        system_dist = Path("/usr/lib/python3/dist-packages")
        if system_dist.exists() and str(system_dist) not in sys.path:
            sys.path.append(str(system_dist))

        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "\n" + "=" * 70 + "\n"
                " ERROR: Picamera2 is not installed or not accessible inside this Python environment.\n"
                " To fix on Raspberry Pi:\n"
                "   1. Install system package: sudo apt install -y python3-picamera2\n"
                "   2. Recreate your venv with system packages enabled:\n"
                "        python3 -m venv .venv --system-site-packages\n"
                "   Or run with '--mock' if you intentionally want to test offline simulation.\n"
                "=" * 70
            ) from exc

        try:
            self._picam2 = Picamera2(self.camera_id)
            controls = {"FrameRate": self.fps, "Brightness": 0.0}
            if self.shutter_us > 0:
                controls["ExposureTime"] = self.shutter_us
            if self.gain > 0:
                controls["AnalogueGain"] = self.gain
            if (
                self.shutter_us > 0
                and self.gain > 0
                and "AeEnable" in self._picam2.camera_controls
            ):
                controls["AeEnable"] = False

            config = self._picam2.create_video_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"},
                raw=None,
                buffer_count=4,
                controls=controls,
                display=None,
                encode=None,
                queue=True,
            )
            self._picam2.configure(config)
            self._picam2.start()
            time.sleep(1.0)
            self.is_picamera = True
            print("\n" + "=" * 60)
            print(f" [CAMERA MODE] REAL HARDWARE PICAMERA2 ACTIVE (Camera {self.camera_id})")
            print(f" Resolution: {self.width}x{self.height} @ {self.fps} FPS | Shutter: {self.shutter_us}us")
            print("=" * 60 + "\n")
        except Exception as exc:
            raise RuntimeError(
                f"\n[Picamera2 Init Failed] Could not open camera {self.camera_id}: {exc}\n"
                "Check camera ribbon connection and verify with 'rpicam-hello'."
            ) from exc


    def capture_request(self) -> FrameRequest:
        if self.is_picamera and self._picam2 is not None:
            req = self._picam2.capture_request()
            try:
                meta = req.get_metadata()
                frame = req.make_array("main")
                return FrameRequest(frame, meta)
            finally:
                req.release()

        # Mock / OpenCV frame generation
        now_ns = time.monotonic_ns()
        frame_interval_us = int(1_000_000 / max(1.0, self.fps))
        sensor_ts = now_ns - int(self.shutter_us * 1000)

        metadata = {
            "SensorTimestamp": sensor_ts,
            "ExposureTime": float(self.shutter_us),
            "AnalogueGain": float(self.gain if self.gain > 0 else 1.0),
            "FrameDuration": float(frame_interval_us),
        }

        if self._cv_cap is not None and self._cv_cap.isOpened():
            ret, frame = self._cv_cap.read()
            if ret:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return FrameRequest(frame, metadata)

        # Generate synthetic dark IR frame with 4 realistic bright LED spots moving on the training bar
        t_sec = (now_ns - self._start_monotonic_ns) / 1e9
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 4 simulated bar markers moving in a smooth 3D trajectory
        cx = self.width / 2.0 + np.sin(t_sec * 1.5) * 60.0
        cy = self.height / 2.0 + np.cos(t_sec * 1.2) * 40.0
        theta = t_sec * 0.8

        marker_offsets = [
            (-30.0, -15.0),
            (30.0, 15.0),
            (-70.0, 5.0),
            (-15.0, 25.0),
        ]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for ox, oy in marker_offsets:
            rx = cx + (ox * cos_t - oy * sin_t)
            ry = cy + (ox * sin_t + oy * cos_t)
            if 10 <= rx < self.width - 10 and 10 <= ry < self.height - 10:
                cv2.circle(frame, (int(round(rx)), int(round(ry))), 5, (255, 255, 255), -1)
                cv2.circle(frame, (int(round(rx)), int(round(ry))), 3, (255, 255, 255), -1)

        self._frame_count += 1
        return FrameRequest(frame, metadata)

    def stop(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
        if self._cv_cap is not None:
            try:
                self._cv_cap.release()
            except Exception:
                pass


class LatencyTracker:
    """Tracks latency metrics per stage and computes statistical summary."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.times = {
            "acquire": [],
            "preprocess": [],
            "inference": [],
            "postprocess": [],
            "total": [],
            "frame_age": [],
            "sensor_to_prediction": [],
            "exposure_to_prediction": [],
        }
        self.frame_durations_us: list[float] = []
        self.previous_sensor_timestamp: int | None = None
        self.skipped_frames = 0
        self.frame_number = 0

    def record_frame(
        self,
        t0_ns: int,
        t1_ns: int,
        t2_ns: int,
        t3_ns: int,
        t4_ns: int,
        metadata: dict[str, Any],
    ) -> tuple[float, float, float, float, float, float | None, float | None]:
        acquire_ms = (t1_ns - t0_ns) / 1e6
        preprocess_ms = (t2_ns - t1_ns) / 1e6
        inference_ms = (t3_ns - t2_ns) / 1e6
        postprocess_ms = (t4_ns - t3_ns) / 1e6
        total_ms = (t4_ns - t0_ns) / 1e6

        self.times["acquire"].append(acquire_ms)
        self.times["preprocess"].append(preprocess_ms)
        self.times["inference"].append(inference_ms)
        self.times["postprocess"].append(postprocess_ms)
        self.times["total"].append(total_ms)

        sensor_timestamp = metadata.get("SensorTimestamp")
        exposure_us = metadata.get("ExposureTime")
        frame_duration_us = metadata.get("FrameDuration")

        frame_age_ms = None
        sensor_ms = None
        if sensor_timestamp is not None:
            sensor_timestamp = int(sensor_timestamp)
            frame_age_ms = (t1_ns - sensor_timestamp) / 1e6
            sensor_ms = (t4_ns - sensor_timestamp) / 1e6
            self.times["frame_age"].append(frame_age_ms)
            self.times["sensor_to_prediction"].append(sensor_ms)

            if exposure_us is not None:
                exposure_start = sensor_timestamp - int(float(exposure_us) * 1000)
                self.times["exposure_to_prediction"].append((t4_ns - exposure_start) / 1e6)

            if self.previous_sensor_timestamp is not None and frame_duration_us:
                frame_steps = max(
                    1,
                    round(
                        (sensor_timestamp - self.previous_sensor_timestamp)
                        / (float(frame_duration_us) * 1000)
                    ),
                )
                self.skipped_frames += max(0, frame_steps - 1)
            self.previous_sensor_timestamp = sensor_timestamp

        if frame_duration_us:
            self.frame_durations_us.append(float(frame_duration_us))

        self.frame_number += 1
        return (
            acquire_ms,
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            total_ms,
            sensor_ms,
            frame_age_ms,
        )

    def print_summary(self, elapsed_s: float) -> dict[str, Any]:
        processed_fps = self.frame_number / elapsed_s if elapsed_s > 0 else 0.0
        camera_fps = (
            1_000_000 / float(np.median(self.frame_durations_us))
            if self.frame_durations_us
            else processed_fps
        )

        print(f"\n=== Summary: {self.model_name} ===")
        print(f"Processed frames : {self.frame_number}")
        print(f"Elapsed          : {elapsed_s:.2f}s")
        print(f"Processed rate   : {processed_fps:.2f} FPS")
        print(f"Camera rate      : {camera_fps:.2f} FPS (metadata)")
        print(f"Skipped frames   : ~{self.skipped_frames} (estimated)")

        print("\nmetric                     mean   median      p95      min      max")
        print("                            ms       ms       ms       ms       ms")

        summary_dict = {
            "model": self.model_name,
            "frames": self.frame_number,
            "elapsed_s": elapsed_s,
            "processed_fps": processed_fps,
            "camera_fps": camera_fps,
            "skipped_frames": self.skipped_frames,
            "metrics": {},
        }

        for name, values in self.times.items():
            if not values:
                continue
            data = np.asarray(values, dtype=np.float64)
            mean_v = float(data.mean())
            median_v = float(np.median(data))
            p95_v = float(np.percentile(data, 95))
            min_v = float(data.min())
            max_v = float(data.max())
            summary_dict["metrics"][name] = {
                "mean": mean_v,
                "median": median_v,
                "p95": p95_v,
                "min": min_v,
                "max": max_v,
            }
            print(
                f"{name:25s}"
                f" {mean_v:8.2f} {median_v:8.2f} {p95_v:8.2f} {min_v:8.2f} {max_v:8.2f}"
            )

        return summary_dict


def optional_ms(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:7.2f}"


def pose_text(pose: np.ndarray) -> str:
    return " ".join(
        f"{name}={float(value):+.5f}" for name, value in zip(POSE_NAMES, pose)
    )


def add_standard_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera", type=int, default=0, help="Camera index for Picamera2")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=400, help="Frame height (default: 400)")
    parser.add_argument("--fps", type=float, default=72.0, help="Target FPS (default: 72.0)")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads (default: 4)")
    parser.add_argument(
        "--shutter-us",
        type=int,
        default=1000,
        help="Exposure time in microseconds (0 for auto, default: 1000us)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.0,
        help="Analogue gain (0 for auto, default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Duration in seconds to run (0 for infinite until Ctrl+C)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print log every Nth processed frame (0 disables per-frame logging, default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all per-frame terminal output to maximize performance (prints only final summary)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional video file path or USB device index (fallback if Picamera2 unavailable)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force synthetic dark IR camera stream for offline testing",
    )

