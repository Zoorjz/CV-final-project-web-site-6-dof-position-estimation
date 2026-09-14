#!/usr/bin/env python3
"""Real-Time Low-Latency 6-DoF Pose Inference & Latency Test: Classical Feature + ML Pipeline.

Captures frames with Picamera2 (queue=True, zero backlog) on Raspberry Pi,
executes sub-pixel optical blob detection, Hungarian marker tracking,
V4 24-dimensional feature extraction, dual TaskMLP (translation & rotation)
inference, and causal AllFrameFilter postprocessing.

Usage:
  python realtime_ml_latency.py
  python realtime_ml_latency.py --mock --duration 10
  python realtime_ml_latency.py --guided-detection --threads 2
"""

from __future__ import annotations

import argparse
from collections import deque
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ML_MODEL_DIR = HERE / "models" / "ml"
if ML_MODEL_DIR.exists():
    sys.path.insert(0, str(ML_MODEL_DIR))
else:
    sys.path.insert(0, str(HERE.parent.parent / "ML" / "pi_inference"))

from baseline import detect, features  # noqa: E402
from evaluate_full_frames import initial_slots, partial_slot_valid, temporal_input  # noqa: E402
from guided_detection import guided_detect  # noqa: E402
from postprocess_predictions import AllFrameFilter  # noqa: E402
from separate_tasks import TaskMLP, task_predict  # noqa: E402
from tracker import MarkerTracker  # noqa: E402
from camera_utils import (  # noqa: E402
    CameraStream,
    LatencyTracker,
    add_standard_camera_args,
    optional_ms,
    pose_text,
)


def quaternion_xyzw_from_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    vector = cv2.Rodrigues(rotation_matrix)[0].reshape(3)
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.r_[vector / angle * np.sin(angle / 2.0), np.cos(angle / 2.0)]


class MLStreamingPoseEngine:
    """End-to-end streaming ML pose estimator."""

    def __init__(
        self,
        translation_path: Path,
        rotation_path: Path,
        guided: bool = False,
        threads: int = 2,
        velocity_decay: float = 0.15,
        partial_max_age: float = 0.1,
        threshold: int | None = None,
        min_area: int | None = None,
        max_area: int | None = None,
        top_blobs: bool = True,
    ) -> None:
        torch.set_num_threads(max(1, threads))
        self.checkpoints = {}
        self.models = {}

        for task, path in (("translation", translation_path), ("rotation", rotation_path)):
            if not path.is_file():
                raise FileNotFoundError(f"Model checkpoint not found: {path}")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ck.get("task") != task or ck.get("config", {}).get("feature_version") != 4:
                raise ValueError(f"Checkpoint {path} must be a V4 separate-task model")
            model = TaskMLP(ck["input_width"], task, ck["config"])
            model.load_state_dict(ck["model"])
            model.eval()
            self.checkpoints[task] = ck
            self.models[task] = model

        self.cfg = self.checkpoints["translation"]["config"].copy()
        self.detector_cfg = self.cfg.get("detector", {}).copy()
        if threshold is not None:
            self.detector_cfg["threshold"] = threshold
        if min_area is not None:
            self.detector_cfg["min_area"] = min_area
        if max_area is not None:
            self.detector_cfg["max_area"] = max_area

        self.top_blobs = top_blobs
        self.frames = {
            task: ck["config"].get(f"{task}_frames", 1)
            for task, ck in self.checkpoints.items()
        }
        max_f = max(self.frames.values())
        self.history = deque(maxlen=max_f)
        self.quality = deque(maxlen=max_f)
        self.tracker = MarkerTracker(width=6, **self.cfg["tracker"])
        self.filter = AllFrameFilter(velocity_decay=velocity_decay)
        self.guided = guided
        self.partial_max_age = partial_max_age
        self.previous_time = None
        self.previous_quaternion = None
        self.established = False

    def preprocess_and_extract(
        self, frame_rgb: np.ndarray, timestamp_s: float
    ) -> tuple[dict[str, torch.Tensor], bool, int, int, np.ndarray, str]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if frame_rgb.ndim == 3 else frame_rgb

        gap = (
            self.previous_time is not None
            and (timestamp_s - self.previous_time) > self.cfg.get("max_context_gap_s", 0.05)
        )

        if self.guided:
            raw_blobs, _ = guided_detect(gray, self.detector_cfg, self.tracker, timestamp_s)
        else:
            raw_blobs = detect(gray, self.detector_cfg, shape=True)

        blobs = raw_blobs
        # If noise/ambient light creates >4 blobs, filter to the top 4 brightest candidates
        if len(blobs) > 4 and self.top_blobs:
            rank = np.lexsort((blobs[:, 1], blobs[:, 0], -blobs[:, 2] * blobs[:, 3]))
            blobs = blobs[rank[:4]]

        state = self.tracker.update(blobs, timestamp_s)
        if state["reset"] or gap:
            self.history.clear()
            self.quality.clear()
            self.established = False
            if state["reset"]:
                self.filter.reset_measurements()

        self.established = self.established or state["ready"]
        valid = partial_slot_valid(
            state, self.tracker.last_seen, timestamp_s, self.established, self.partial_max_age
        )
        self.quality.append(valid)
        slots = initial_slots(blobs) if self.tracker.blobs is None else state["slots"]
        self.history.append((features(slots, gray.shape, version=4), timestamp_s, state["ready"]))

        normal = True
        model_inputs = {}
        for task, ck in self.checkpoints.items():
            val, eligible, _ = temporal_input(self.history, self.frames[task])
            normal &= eligible
            model_inputs[task] = val

        matched_dots = int(state["observed"].sum())
        return model_inputs, normal, len(raw_blobs), matched_dots, raw_blobs, state.get("reason", "unknown")

    @torch.inference_mode()
    def infer_models(self, model_inputs: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
        predictions = {}
        for task, ck in self.checkpoints.items():
            inp = model_inputs[task][None]
            predictions[task] = task_predict(
                self.models[task], inp, ck["stats"], task, "cpu"
            )[0].numpy()
        return predictions["translation"], predictions["rotation"]

    def postprocess_state(
        self,
        raw_p: np.ndarray,
        raw_r: np.ndarray,
        timestamp_s: float,
        normal: bool,
    ) -> tuple[np.ndarray, str, str]:
        max_f = max(self.frames.values())
        partial = (
            not normal and len(self.quality) == max_f and all(self.quality)
        )
        p, r, status, reason = self.filter.step(
            timestamp_s, raw_p.astype(float), raw_r.astype(float), normal, partial
        )
        q = quaternion_xyzw_from_matrix(r)
        if self.previous_quaternion is not None and np.dot(q, self.previous_quaternion) < 0:
            q = -q
        self.previous_quaternion = q
        self.previous_time = timestamp_s

        pose = np.concatenate([p, q]).astype(np.float32)
        return pose, status, reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_camera_args(parser)
    parser.add_argument(
        "--translation",
        type=Path,
        default=ML_MODEL_DIR / "translation.pt",
        help="Path to translation MLP checkpoint",
    )
    parser.add_argument(
        "--rotation",
        type=Path,
        default=ML_MODEL_DIR / "rotation.pt",
        help="Path to rotation MLP checkpoint",
    )
    parser.add_argument("--guided-detection", action="store_true", help="Enable guided adaptive detection")
    parser.add_argument("--threshold", type=int, default=None, help="Blob intensity threshold (0-255, default: model cfg 100)")
    parser.add_argument("--min-area", type=int, default=None, help="Minimum blob area in pixels (default: model cfg 3)")
    parser.add_argument("--max-area", type=int, default=None, help="Maximum blob area in pixels (default: model cfg 2000)")
    parser.add_argument("--no-top-blobs", action="store_true", help="Disable auto-selection of top 4 brightest blobs when >4 blobs detected")
    parser.add_argument("--velocity-decay", type=float, default=0.15, help="Filter velocity decay rate")
    parser.add_argument("--save-snapshot", type=Path, default=None, help="Save an annotated debug JPEG snapshot showing detected blobs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print(" 6-DoF Real-Time Latency Test: Classical ML Pipeline")
    print("=" * 60)
    print(f"Translation Model : {args.translation}")
    print(f"Rotation Model    : {args.rotation}")
    print(f"Guided Detection  : {args.guided_detection}")
    print(f"Camera            : {args.width}x{args.height} @ {args.fps:g} FPS (shutter={args.shutter_us}us)")
    print(f"Top 4 Filtering   : {'Disabled' if args.no_top_blobs else 'Enabled (selects 4 brightest LEDs)'}")
    print("Stop              : Ctrl+C\n")

    engine = MLStreamingPoseEngine(
        args.translation,
        args.rotation,
        guided=args.guided_detection,
        threads=args.threads,
        velocity_decay=args.velocity_decay,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        top_blobs=not args.no_top_blobs,
    )
    tracker = LatencyTracker("ML (Classical Blob + Dual MLP)")

    # Model warm-up
    print("Warming up ML models...")
    for task, model in engine.models.items():
        dummy_in = torch.zeros((1, engine.checkpoints[task]["input_width"]), dtype=torch.float32)
        task_predict(model, dummy_in, engine.checkpoints[task]["stats"], task, "cpu")

    stream = CameraStream(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        shutter_us=args.shutter_us,
        gain=args.gain,
        source=args.source,
        use_mock=args.mock,
    )
    stream.start()

    started_ns = time.monotonic_ns()
    frame_idx = 0
    saved_snapshot = False

    try:
        while args.duration == 0 or (time.monotonic_ns() - started_ns) / 1e9 < args.duration:
            t0 = time.monotonic_ns()
            req = stream.capture_request()
            metadata = req.get_metadata()
            frame_rgb = req.make_array("main")
            t1 = time.monotonic_ns()

            time_s = (t1 - started_ns) / 1e9
            model_inputs, normal, n_blobs, n_dots, raw_blobs, tracker_reason = engine.preprocess_and_extract(frame_rgb, time_s)
            t2 = time.monotonic_ns()

            raw_p, raw_r = engine.infer_models(model_inputs)
            t3 = time.monotonic_ns()

            pose, status, reason = engine.postprocess_state(raw_p, raw_r, time_s, normal)
            t4 = time.monotonic_ns()

            # Optional snapshot saving for visual debugging
            if args.save_snapshot and not saved_snapshot and frame_idx > 10:
                vis = frame_rgb.copy()
                for b in raw_blobs:
                    cx, cy = int(b[0]), int(b[1])
                    cv2.circle(vis, (cx, cy), 6, (0, 255, 255), 2)
                cv2.imwrite(str(args.save_snapshot), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                print(f"\n[Snapshot] Saved annotated debug frame to {args.save_snapshot} ({len(raw_blobs)} blobs)")
                saved_snapshot = True

            (
                acq_ms,
                pre_ms,
                infer_ms,
                post_ms,
                total_ms,
                sensor_ms,
                age_ms,
            ) = tracker.record_frame(t0, t1, t2, t3, t4, metadata)

            if not args.quiet and args.print_every > 0 and frame_idx % args.print_every == 0:
                exp_text = f"{metadata.get('ExposureTime', args.shutter_us):.0f}us"
                status_detail = f"{status}:{tracker_reason}" if status == "held" else status
                print(
                    f"#{frame_idx:06d} "
                    f"total={total_ms:6.2f}ms "
                    f"sensor={optional_ms(sensor_ms)}ms "
                    f"age={optional_ms(age_ms)}ms | "
                    f"acq={acq_ms:5.2f} pre={pre_ms:5.2f} infer={infer_ms:5.2f} post={post_ms:5.2f} | "
                    f"exp={exp_text} blobs={n_blobs} dots={n_dots} [{status_detail}] | "
                    f"{pose_text(pose)}",
                    flush=True,
                )

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        finished_ns = time.monotonic_ns()
        stream.stop()

    elapsed_s = (finished_ns - started_ns) / 1e9
    tracker.print_summary(elapsed_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
