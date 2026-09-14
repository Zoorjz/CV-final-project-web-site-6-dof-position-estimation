#!/usr/bin/env python3
"""Unified Multi-Model Latency Benchmark & Comparison Orchestrator.

Allows running individual 6-DoF models or automated back-to-back comparative
benchmarks across all 3 pipelines (CNN, ML, PnP) + ONNX runtime.

Usage:
  # Run a single model
  python run_latency_benchmark.py --model cnn --duration 10
  python run_latency_benchmark.py --model ml --duration 10
  python run_latency_benchmark.py --model pnp --duration 10

  # Run comparative benchmark across all pipelines
  python run_latency_benchmark.py --benchmark-all --duration 5 --mock
  python run_latency_benchmark.py --benchmark-all --duration 10 --output-report latency_report.md
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = {
    "cnn": HERE / "realtime_cnn_latency.py",
    "ml": HERE / "realtime_ml_latency.py",
    "pnp": HERE / "realtime_pnp_latency.py",
    "onnx": HERE / "realtime_onnx_latency.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn", "ml", "pnp", "onnx"],
        default="pnp",
        help="Model to benchmark when running a single test (default: pnp)",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Run all available models sequentially and produce a comparative matrix",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Test duration per model in seconds (default: 5.0s)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=400, help="Capture height (default: 400)")
    parser.add_argument("--fps", type=float, default=72.0, help="Target capture FPS (default: 72.0)")
    parser.add_argument("--shutter-us", type=int, default=1000, help="Exposure in microseconds (default: 1000us)")
    parser.add_argument("--threads", type=int, default=4, help="CPU worker threads (default: 4)")
    parser.add_argument("--mock", action="store_true", help="Force synthetic dark IR stream for offline testing")
    parser.add_argument("--source", type=str, default=None, help="Video file or OpenCV camera source")
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional path to save formatted Markdown comparison report",
    )
    return parser.parse_args()


def run_single_model(script_path: Path, args: argparse.Namespace) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--camera", str(args.camera),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        "--shutter-us", str(args.shutter_us),
        "--threads", str(args.threads),
        "--duration", str(args.duration),
        "--print-every", "10",
    ]
    if args.mock:
        cmd.append("--mock")
    if args.source:
        cmd.extend(["--source", args.source])

    print(f"\n>>> Running: {script_path.name} (duration={args.duration}s)...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout + ("\n" + res.stderr if res.stderr else "")


def parse_summary_table(output: str) -> dict[str, dict[str, float]]:
    """Extracts summary metrics from standard output table."""
    metrics = {}
    lines = output.splitlines()
    in_summary = False
    for line in lines:
        if "=== Summary:" in line:
            in_summary = True
            continue
        if in_summary:
            parts = line.split()
            if len(parts) == 6 and parts[1].replace(".", "").isdigit():
                metric_name = parts[0]
                try:
                    metrics[metric_name] = {
                        "mean": float(parts[1]),
                        "median": float(parts[2]),
                        "p95": float(parts[3]),
                        "min": float(parts[4]),
                        "max": float(parts[5]),
                    }
                except ValueError:
                    pass
    return metrics


def main() -> int:
    args = parse_args()

    if not args.benchmark_all:
        script = SCRIPTS.get(args.model)
        if not script or not script.exists():
            print(f"Error: Script {script} not found.")
            return 1
        code, out = run_single_model(script, args)
        print(out)
        return code

    print("=" * 70)
    print(" Running Multi-Pipeline Real-Time 6-DoF Benchmark Comparison")
    print(f" Test Duration per Model : {args.duration}s")
    print(f" Target Camera Spec      : {args.width}x{args.height} @ {args.fps:g} FPS (shutter={args.shutter_us}us)")
    print(f" CPU Worker Threads      : {args.threads}")
    print("=" * 70)

    results = {}
    for name, script in SCRIPTS.items():
        if not script.exists():
            continue
        code, out = run_single_model(script, args)
        if code == 0:
            metrics = parse_summary_table(out)
            results[name] = {"success": True, "metrics": metrics, "raw": out}
            print(f"  [PASS] {name.upper()}: Processed successfully.")
        else:
            results[name] = {"success": False, "raw": out}
            print(f"  [SKIP/FAIL] {name.upper()}: Exited with code {code}.")

    # Generate comparative table
    print("\n" + "=" * 80)
    print(" COMPARATIVE LATENCY BENCHMARK MATRIX (Values in Milliseconds)")
    print("=" * 80)
    header = f"{'Pipeline / Model':<22} | {'Total Mean':<11} | {'Inference':<10} | {'Preprocess':<11} | {'Postprocess':<12} | {'Sensor-to-Pred':<14}"
    print(header)
    print("-" * 88)

    md_report = f"""# 6-DoF Real-Time Pose Estimation Latency Benchmark

**Benchmark Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`  
**Test Configuration**: `{args.width}x{args.height} @ {args.fps:g} FPS` | Shutter: `{args.shutter_us} µs` | Threads: `{args.threads}` | Test Duration: `{args.duration}s per model`

---

## 1. Summary Comparison Matrix

| Pipeline / Model | Total Latency (Mean) | Core Inference | Preprocessing | Postprocessing / Filtering | Sensor to Prediction |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for name, data in results.items():
        if not data.get("success") or not data.get("metrics"):
            row = f"{name.upper():<22} | {'FAILED / SKIPPED':<63}"
            print(row)
            md_report += f"| **{name.upper()}** | *Failed / Not Available* | - | - | - | - |\n"
            continue

        m = data["metrics"]
        total_mean = f"{m.get('total', {}).get('mean', 0.0):.2f} ms"
        infer_mean = f"{m.get('inference', {}).get('mean', 0.0):.2f} ms"
        pre_mean = f"{m.get('preprocess', {}).get('mean', 0.0):.2f} ms"
        post_mean = f"{m.get('postprocess', {}).get('mean', 0.0):.2f} ms"
        sensor_mean = f"{m.get('sensor_to_prediction', {}).get('mean', 0.0):.2f} ms"

        row = f"{name.upper():<22} | {total_mean:<11} | {infer_mean:<10} | {pre_mean:<11} | {post_mean:<12} | {sensor_mean:<14}"
        print(row)
        md_report += f"| **{name.upper()}** | `{total_mean}` | `{infer_mean}` | `{pre_mean}` | `{post_mean}` | `{sensor_mean}` |\n"

    print("=" * 88)

    if args.output_report:
        args.output_report.write_text(md_report, encoding="utf-8")
        print(f"\nSaved Markdown comparison report to: {args.output_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
