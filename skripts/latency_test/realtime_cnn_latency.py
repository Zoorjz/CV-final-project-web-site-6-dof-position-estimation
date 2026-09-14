#!/usr/bin/env python3
"""Real-Time Low-Latency 6-DoF Pose Inference & Latency Test: Temporal-3 CNN.

Captures frames with Picamera2 (queue=True, zero backlog) on Raspberry Pi,
executes the YOLO classification backbone once per frame with temporal feature
caching (fusing t-10, t-5, t), and applies causal 6-DoF Kalman filtering.

Usage:
  python realtime_cnn_latency.py
  python realtime_cnn_latency.py --mock --duration 10
  python realtime_cnn_latency.py --shutter-us 1000 --threads 4 --no-kalman
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

# Patch ultralytics torchvision check if torchvision is not installed
_orig_meta_version = importlib.metadata.version


def _safe_meta_version(distribution_name: str) -> str:
    try:
        return _orig_meta_version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        if distribution_name == "torchvision":
            return "0.0.0"
        raise


importlib.metadata.version = _safe_meta_version

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
CNN_MODEL_DIR = HERE / "models" / "cnn"
if CNN_MODEL_DIR.exists():
    sys.path.insert(0, str(CNN_MODEL_DIR))
else:
    sys.path.insert(0, str(HERE.parent.parent / "CNN" / "bar_pose_temporal3"))


from bar_pose_regression.checkpointing import load_checkpoint  # noqa: E402
from bar_pose_regression.config import ModelConfig  # noqa: E402
from bar_pose_regression.geometry import (  # noqa: E402
    matrix_to_quaternion,
    rotation_6d_to_matrix,
)
from bar_pose_regression.model import BarPoseRegressor  # noqa: E402
from bar_pose_regression.preprocess import letterbox_rgb  # noqa: E402
from camera_utils import (  # noqa: E402
    CameraStream,
    LatencyTracker,
    add_standard_camera_args,
    optional_ms,
    pose_text,
)


class CNNPosePredictor:
    """Temporal-3 CNN pose estimator with feature map sliding window cache."""

    def __init__(self, checkpoint_path: Path, backbone_path: Path, threads: int = 4) -> None:
        torch.set_num_threads(max(1, threads))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CNN Checkpoint not found: {checkpoint_path}")
        if not backbone_path.is_file():
            raise FileNotFoundError(f"Backbone weights not found: {backbone_path}")

        ck = load_checkpoint(checkpoint_path)
        data_cfg = ck["config"]["data"]
        model_cfg = dict(ck["config"]["model"])
        model_cfg["weights"] = str(backbone_path)

        self.image_size = int(data_cfg["image_size"])
        self.num_frames = int(data_cfg.get("temporal_frames", 3))
        self.stride = int(data_cfg.get("temporal_stride", 5))

        fields = getattr(ModelConfig, "__dataclass_fields__", {})
        mc = ModelConfig(**{k: model_cfg[k] for k in fields if k in model_cfg})
        model = BarPoseRegressor.from_config(mc, self.image_size, num_frames=self.num_frames)
        model.load_state_dict(ck["model_state"])
        model.eval()

        self.backbone = model.backbone
        self.pose_head = model.pose_head

        stats = ck["translation_stats"]
        self.t_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.t_std = torch.tensor(stats["std"], dtype=torch.float32)
        self.log_depth = bool(stats.get("log_depth", False))

        self._feature_cache: dict[int, torch.Tensor] = {}

    def preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """Letterbox resize to 256x256 normalized float32 tensor."""
        img = Image.fromarray(frame_rgb)
        return letterbox_rgb(img, self.image_size).unsqueeze(0)

    @torch.inference_mode()
    def infer(self, tensor_batch: torch.Tensor, frame_idx: int) -> np.ndarray:
        """Runs backbone on new frame, caches map, fuses temporal window, regresses pose."""
        feat = self.backbone(tensor_batch)
        self._feature_cache[frame_idx] = feat

        # Retain only required temporal history
        keep_oldest = frame_idx - (self.num_frames - 1) * self.stride
        for old in [k for k in self._feature_cache if k < keep_oldest]:
            del self._feature_cache[old]

        # Assemble causal window [t-(N-1)s, ..., t-s, t]
        window = []
        for offset in range(self.num_frames - 1, -1, -1):
            idx = max(0, frame_idx - offset * self.stride)
            window.append(self._feature_cache.get(idx, self._feature_cache[frame_idx]))
        stacked = torch.cat(window, dim=0)

        t_norm, r_6d = self.pose_head(stacked)
        t_pred = t_norm * self.t_std + self.t_mean
        if self.log_depth:
            t_pred = t_pred.clone()
            t_pred[..., 2] = torch.exp(t_pred[..., 2])

        q_pred = matrix_to_quaternion(rotation_6d_to_matrix(r_6d))
        pose = torch.cat((t_pred, q_pred), dim=-1)[0]
        return pose.cpu().numpy().astype(np.float32)


class CNN6DoFKalmanFilter:
    """Causal state filter: Kinematic constant-velocity for position + SO(3) SLERP for rotation."""

    def __init__(
        self,
        pos_noise_m: float = 0.015,
        accel_noise_m: float = 0.5,
        rot_tau_s: float = 0.06,
    ) -> None:
        self.pos_noise = pos_noise_m
        self.accel_noise = accel_noise_m
        self.rot_tau = rot_tau_s
        self.x = None  # [px, py, pz, vx, vy, vz]
        self.P = None
        self.rotation = None
        self.last_t = None

    def step(self, t: float, pos_m: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        R_meas = quaternion_to_matrix(quat_xyzw)
        if self.x is None or self.last_t is None:
            self.x = np.concatenate([pos_m, [0.0, 0.0, 0.0]])
            self.P = np.diag([self.pos_noise**2] * 3 + [0.5**2] * 3)
            self.rotation = R_meas.copy()
            self.last_t = t
            return np.concatenate([pos_m, quat_xyzw])

        dt = max(1e-4, t - self.last_t)
        self.last_t = t

        # Kinematic constant-velocity prediction
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        G = np.vstack([np.eye(3) * (0.5 * dt * dt), np.eye(3) * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (self.accel_noise**2) * (G @ G.T)

        # Measurement update for position
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        y = pos_m - self.x[:3]
        S = H @ self.P @ H.T + np.eye(3) * (self.pos_noise**2)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        # SO(3) Geodesic SLERP filter
        alpha = 1.0 - np.exp(-dt / self.rot_tau)
        R_rel = self.rotation.T @ R_meas
        rotvec = cv2.Rodrigues(R_rel)[0].ravel()
        self.rotation = self.rotation @ cv2.Rodrigues(alpha * rotvec)[0]
        q_filtered = matrix_to_quaternion_np(self.rotation)
        if np.dot(q_filtered, quat_xyzw) < 0:
            q_filtered = -q_filtered

        return np.concatenate([self.x[:3], q_filtered])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion_np(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_camera_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CNN_MODEL_DIR / "best_combined.pt",
        help="Path to trained CNN checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        default=CNN_MODEL_DIR / "yolo26n-cls.pt",
        help="Path to YOLO backbone weights",
    )
    parser.add_argument("--no-kalman", action="store_true", help="Disable 6-DoF Kalman filter")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print(" 6-DoF Real-Time Latency Test: Temporal-3 CNN")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Backbone   : {args.backbone}")
    print(f"Camera     : {args.width}x{args.height} @ {args.fps:g} FPS (shutter={args.shutter_us}us)")
    print(f"Kalman     : {'Disabled' if args.no_kalman else 'Enabled (Constant-Velocity + SO3 SLERP)'}")
    print("Stop       : Ctrl+C\n")

    predictor = CNNPosePredictor(args.checkpoint, args.backbone, threads=args.threads)
    kf = None if args.no_kalman else CNN6DoFKalmanFilter()
    tracker = LatencyTracker("CNN (Temporal-3 YOLO)")

    # Model warm-up
    print("Warming up CNN model...")
    dummy_input = torch.zeros((1, 3, predictor.image_size, predictor.image_size), dtype=torch.float32)
    for _ in range(5):
        predictor.infer(dummy_input, 0)

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

    try:
        while args.duration == 0 or (time.monotonic_ns() - started_ns) / 1e9 < args.duration:
            t0 = time.monotonic_ns()
            req = stream.capture_request()
            metadata = req.get_metadata()
            frame_rgb = req.make_array("main")
            t1 = time.monotonic_ns()

            tensor_batch = predictor.preprocess(frame_rgb)
            t2 = time.monotonic_ns()

            raw_pose = predictor.infer(tensor_batch, frame_idx)
            t3 = time.monotonic_ns()

            time_s = (t3 - started_ns) / 1e9
            if kf is not None:
                pose = kf.step(time_s, raw_pose[:3], raw_pose[3:])
            else:
                pose = raw_pose
            t4 = time.monotonic_ns()

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
                print(
                    f"#{frame_idx:06d} "
                    f"total={total_ms:6.2f}ms "
                    f"sensor={optional_ms(sensor_ms)}ms "
                    f"age={optional_ms(age_ms)}ms | "
                    f"acq={acq_ms:5.2f} pre={pre_ms:5.2f} infer={infer_ms:5.2f} post={post_ms:5.2f} | "
                    f"exp={exp_text} | "
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
