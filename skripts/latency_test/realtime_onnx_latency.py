#!/usr/bin/env python3
"""Real-Time Low-Latency 6-DoF Pose Inference & Latency Test: Single-Frame Static ONNX Runtime.

Captures frames with Picamera2 (queue=True, zero backlog) on Raspberry Pi,
executes the static exported ONNX graph (bar_pose_256.onnx), and canonicalizes
the 7D pose quaternion.

Usage:
  python realtime_onnx_latency.py
  python realtime_onnx_latency.py --mock --duration 10
  python realtime_onnx_latency.py --threads 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ONNX_MODEL_DIR = HERE / "models" / "onnx"

from camera_utils import (  # noqa: E402
    CameraStream,
    LatencyTracker,
    add_standard_camera_args,
    optional_ms,
    pose_text,
)

INPUT_SIZE = 256


def load_contract(model_path: Path) -> dict:
    contract_path = model_path.with_name("model_contract.json")
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX Model not found: {model_path}")
    if not contract_path.is_file():
        raise FileNotFoundError(f"Contract not found: {contract_path}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {
        "input_name": "image",
        "input_shape": [1, 3, 256, 256],
        "output_name": "pose",
        "output_shape": [1, 7],
        "output_contract": "pose7_xyzw",
        "dtype": "float32",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(
                f"unsupported model_contract.json: {key}={contract.get(key)!r}, "
                f"expected {value!r}"
            )
    return contract


def make_session(model_path: Path, contract: dict, threads: int):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("install ONNX Runtime: python -m pip install onnxruntime") from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.name != contract["input_name"] or list(model_input.shape) != contract["input_shape"]:
        raise ValueError("ONNX input metadata does not match model_contract.json")
    if model_output.name != contract["output_name"] or list(model_output.shape) != contract["output_shape"]:
        raise ValueError("ONNX output metadata does not match model_contract.json")
    return session


def preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    """Letterbox resize to 256x256, RGB float32 / 255."""
    image = Image.fromarray(frame_rgb)
    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
    canvas.paste(
        resized,
        (
            (INPUT_SIZE - resized_width) // 2,
            (INPUT_SIZE - resized_height) // 2,
        ),
    )

    array = np.asarray(canvas, dtype=np.float32)
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(array / 255.0, dtype=np.float32)


def canonicalize_pose(output: np.ndarray) -> np.ndarray:
    pose = np.asarray(output, dtype=np.float32).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"invalid pose output: shape={pose.shape}")

    pose = pose.copy()
    quaternion_norm = float(np.linalg.norm(pose[3:].astype(np.float64)))
    if quaternion_norm <= 1e-8:
        raise ValueError("predicted quaternion has zero norm")
    pose[3:] /= quaternion_norm
    if pose[6] < 0:
        pose[3:] *= -1
    return pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_camera_args(parser)
    default_model = (
        ONNX_MODEL_DIR / "bar_pose_256.onnx"
        if (ONNX_MODEL_DIR / "bar_pose_256.onnx").exists()
        else HERE / "bar_pose_256.onnx"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help="Path to ONNX model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    contract = load_contract(model_path)
    session = make_session(model_path, contract, args.threads)

    print("=" * 60)
    print(" 6-DoF Real-Time Latency Test: Single-Frame ONNX Runtime")
    print("=" * 60)
    print(f"Model  : {model_path}")
    print(f"Camera : {args.width}x{args.height} @ {args.fps:g} FPS (shutter={args.shutter_us}us)")
    print("Stop   : Ctrl+C\n")

    # Warm up ONNX Runtime
    print("Warming up ONNX Runtime session...")
    zero_input = np.zeros(contract["input_shape"], dtype=np.float32)
    for _ in range(5):
        session.run([contract["output_name"]], {contract["input_name"]: zero_input})

    tracker = LatencyTracker("ONNX (Single-Frame Static 256)")

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

            batch = preprocess(frame_rgb)
            t2 = time.monotonic_ns()

            output = session.run([contract["output_name"]], {contract["input_name"]: batch})[0]
            t3 = time.monotonic_ns()

            pose = canonicalize_pose(output)
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

            if frame_idx % args.print_every == 0:
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
