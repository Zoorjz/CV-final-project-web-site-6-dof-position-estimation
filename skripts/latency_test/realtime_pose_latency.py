#!/usr/bin/env python3
"""Run the ONNX pose model on the latest Raspberry Pi camera frame.

There is no growing application frame queue. The camera keeps running, and
Picamera2 retains only the latest completed frame. If inference is slower than
the camera, intermediate frames are skipped.

Press Ctrl+C to stop and print latency statistics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SIZE = 256
POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("bar_pose_256.onnx"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--fps", type=float, default=72.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--shutter-us",
        type=int,
        default=100,
        help="0 leaves exposure automatic (default: 100)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.0,
        help="0 leaves analogue gain automatic (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to run; 0 means until Ctrl+C",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="print every Nth processed frame (default: 1)",
    )
    args = parser.parse_args()

    if args.camera < 0:
        parser.error("--camera must be >= 0")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be > 0")
    if args.fps <= 0 or args.threads <= 0:
        parser.error("--fps and --threads must be > 0")
    if args.shutter_us < 0 or args.gain < 0 or args.duration < 0:
        parser.error("--shutter-us, --gain and --duration must be >= 0")
    if args.print_every <= 0:
        parser.error("--print-every must be > 0")
    return args


def load_contract(model_path: Path) -> dict:
    contract_path = model_path.with_name("model_contract.json")
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not contract_path.is_file():
        raise FileNotFoundError(f"contract not found: {contract_path}")

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
    """Match letterbox_rgb(): RGB, black padding, 256x256, CHW float32 / 255."""

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


def summary(values: list[float]) -> tuple[float, float, float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    return (
        float(data.mean()),
        float(np.median(data)),
        float(np.percentile(data, 95)),
        float(data.min()),
        float(data.max()),
    )


def optional_ms(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:7.2f}"


def pose_text(pose: np.ndarray) -> str:
    return " ".join(
        f"{name}={float(value):+.5f}" for name, value in zip(POSE_NAMES, pose)
    )


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    contract = load_contract(model_path)
    session = make_session(model_path, contract, args.threads)

    # Warm up ONNX Runtime before measuring camera frames.
    zero_input = np.zeros(contract["input_shape"], dtype=np.float32)
    for _ in range(5):
        session.run(
            [contract["output_name"]],
            {contract["input_name"]: zero_input},
        )

    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError(
            "Picamera2 is missing. Install python3-picamera2 from apt."
        ) from exc

    camera = Picamera2(args.camera)
    controls = {"FrameRate": args.fps, "Brightness": 0.0}
    if args.shutter_us > 0:
        controls["ExposureTime"] = args.shutter_us
    if args.gain > 0:
        controls["AnalogueGain"] = args.gain
    if args.shutter_us > 0 and args.gain > 0 and "AeEnable" in camera.camera_controls:
        controls["AeEnable"] = False

    # Picamera2 calls this format BGR888, but the NumPy array is ordered [R, G, B].
    # queue=True retains the latest completed frame instead of building a backlog.
    config = camera.create_video_configuration(
        main={"size": (args.width, args.height), "format": "BGR888"},
        raw=None,
        buffer_count=4,
        controls=controls,
        display=None,
        encode=None,
        queue=True,
    )
    camera.configure(config)

    times = {
        "acquire": [],
        "preprocess": [],
        "inference": [],
        "postprocess": [],
        "total": [],
        "frame_age": [],
        "sensor_to_prediction": [],
        "exposure_to_prediction": [],
    }
    frame_durations_us: list[float] = []
    previous_sensor_timestamp: int | None = None
    skipped_frames = 0
    frame_number = 0

    print(f"Model       : {model_path}")
    print(f"Camera      : {args.width}x{args.height} @ {args.fps:g} FPS")
    print("Frame policy: latest frame only; intermediate camera frames may be skipped")
    print("Stop        : Ctrl+C\n")

    camera.start()
    time.sleep(1.0)
    started = time.monotonic_ns()

    try:
        while args.duration == 0 or (time.monotonic_ns() - started) / 1e9 < args.duration:
            t0 = time.monotonic_ns()
            request = camera.capture_request()
            try:
                metadata = request.get_metadata()
                frame_rgb = request.make_array("main")
            finally:
                request.release()
            t1 = time.monotonic_ns()

            batch = preprocess(frame_rgb)
            t2 = time.monotonic_ns()

            output = session.run(
                [contract["output_name"]],
                {contract["input_name"]: batch},
            )[0]
            t3 = time.monotonic_ns()

            pose = canonicalize_pose(output)
            t4 = time.monotonic_ns()

            acquire_ms = (t1 - t0) / 1e6
            preprocess_ms = (t2 - t1) / 1e6
            inference_ms = (t3 - t2) / 1e6
            postprocess_ms = (t4 - t3) / 1e6
            total_ms = (t4 - t0) / 1e6

            times["acquire"].append(acquire_ms)
            times["preprocess"].append(preprocess_ms)
            times["inference"].append(inference_ms)
            times["postprocess"].append(postprocess_ms)
            times["total"].append(total_ms)

            sensor_timestamp = metadata.get("SensorTimestamp")
            exposure_us = metadata.get("ExposureTime")
            gain = metadata.get("AnalogueGain")
            frame_duration_us = metadata.get("FrameDuration")

            frame_age_ms = None
            sensor_ms = None
            if sensor_timestamp is not None:
                sensor_timestamp = int(sensor_timestamp)
                frame_age_ms = (t1 - sensor_timestamp) / 1e6
                sensor_ms = (t4 - sensor_timestamp) / 1e6
                times["frame_age"].append(frame_age_ms)
                times["sensor_to_prediction"].append(sensor_ms)

                if exposure_us is not None:
                    exposure_start = sensor_timestamp - int(float(exposure_us) * 1000)
                    times["exposure_to_prediction"].append((t4 - exposure_start) / 1e6)

                if previous_sensor_timestamp is not None and frame_duration_us:
                    frame_steps = max(
                        1,
                        round(
                            (sensor_timestamp - previous_sensor_timestamp)
                            / (float(frame_duration_us) * 1000)
                        ),
                    )
                    skipped_frames += max(0, frame_steps - 1)
                previous_sensor_timestamp = sensor_timestamp

            if frame_duration_us:
                frame_durations_us.append(float(frame_duration_us))

            frame_number += 1
            if frame_number % args.print_every == 0:
                exposure_text = "n/a" if exposure_us is None else f"{float(exposure_us):.0f}us"
                gain_text = "n/a" if gain is None else f"{float(gain):.2f}x"
                print(
                    f"#{frame_number:06d} "
                    f"total={total_ms:7.2f}ms "
                    f"sensor={optional_ms(sensor_ms)}ms "
                    f"age={optional_ms(frame_age_ms)}ms | "
                    f"acq={acquire_ms:6.2f} "
                    f"pre={preprocess_ms:6.2f} "
                    f"infer={inference_ms:6.2f} "
                    f"post={postprocess_ms:6.2f} | "
                    f"exp={exposure_text} gain={gain_text} | "
                    f"{pose_text(pose)}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        finished = time.monotonic_ns()
        camera.stop()
        camera.close()

    elapsed_s = (finished - started) / 1e9
    processed_fps = frame_number / elapsed_s if elapsed_s > 0 else 0.0

    print("\n=== Summary ===")
    print(f"Processed frames : {frame_number}")
    print(f"Elapsed          : {elapsed_s:.2f}s")
    print(f"Processed rate   : {processed_fps:.2f} FPS")
    if frame_durations_us:
        camera_fps = 1_000_000 / float(np.median(frame_durations_us))
        print(f"Camera rate      : {camera_fps:.2f} FPS (metadata)")
    print(f"Skipped frames   : ~{skipped_frames} (estimated)")

    print("\nmetric                     mean   median      p95      min      max")
    print("                            ms       ms       ms       ms       ms")
    for name, values in times.items():
        if not values:
            continue
        mean, median, p95, minimum, maximum = summary(values)
        print(
            f"{name:25s}"
            f" {mean:8.2f} {median:8.2f} {p95:8.2f} {minimum:8.2f} {maximum:8.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
