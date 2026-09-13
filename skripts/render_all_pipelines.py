#!/usr/bin/env python3
"""render_all_pipelines.py

Master Rendering Orchestrator:
Executes synchronized benchmark video generation for BOTH:
  1. Geometry-Based Pipeline (Classical SQPnP & 13-State EKF) -> data/renders/geometry-based/renders_YYYYMMDD_HHMMSS/
  2. Data-Driven Pipeline (CNN Temporal-3 & ML Feature Regressor) -> data/renders/data-driven/renders_YYYYMMDD_HHMMSS/

Guarantees identical duration, start frame, frame rate, and web-compatible H.264 video encoding across all pipelines.

Usage:
  # Render all pipelines with synchronized 200 frames:
  python skripts/render_all_pipelines.py --start 2131 --count 200

  # Render only geometry-based pipeline:
  python skripts/render_all_pipelines.py --pipeline geometry --start 2131 --count 200

  # Render only data-driven pipeline:
  python skripts/render_all_pipelines.py --pipeline data-driven --start 2131 --count 200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_command_streamed(cmd: list[str], title: str) -> int:
    """Executes a subprocess command with live streamed output."""
    print(f"\n=======================================================")
    print(f"[{title}] Starting Pipeline Execution...")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================\n")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode == 0:
        print(f"\n[{title}] Successfully completed in {elapsed:.1f}s (Exit code: {proc.returncode})")
    else:
        print(f"\n[{title}] Failed with return code {proc.returncode} after {elapsed:.1f}s")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Master Synchronized Video Renderer for All 6-DoF Tracking Pipelines.")
    parser.add_argument("--start", type=int, default=2131, help="Start frame index (default: 2131)")
    parser.add_argument("--count", type=int, default=200, help="Number of frames to render (default: 200)")
    parser.add_argument("--pipeline", type=str, default="all", choices=["all", "geometry", "data-driven", "cnn", "ml"], help="Which pipeline(s) to render (default: all)")
    parser.add_argument("--no-trails", action="store_true", help="Disable fading 3D trajectory trails")
    parser.add_argument("--trail-length", type=int, default=35, help="Length of fading trajectory trail in frames")
    parser.add_argument("--trail-thickness", type=int, default=3, help="Line thickness for 3D trajectory trails")

    # Filter Tuning
    parser.add_argument("--pos-noise", type=float, default=15.0, help="Kalman filter position measurement noise in mm")
    parser.add_argument("--accel-noise", type=float, default=500.0, help="Kalman filter process acceleration noise in mm/s^2")
    parser.add_argument("--rot-tau", type=float, default=0.06, help="Kalman filter rotation time constant in seconds")
    parser.add_argument("--ekf-r-scale", type=float, default=1.0, help="EKF measurement noise multiplier")
    parser.add_argument("--ekf-q-scale", type=float, default=1.0, help="EKF process dynamics noise multiplier")

    parser.add_argument("--publish-to-web", action="store_true", help="Automatically copy rendered videos to public/data/renders and update public/api/dataset-info.json for GitHub Pages deployment")

    args = parser.parse_args()

    t_start = time.time()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    python_exe = sys.executable

    print("\n" + "=" * 65)
    print("  6-DoF TRACKING PIPELINE SYNCHRONIZED MASTER RENDERER")
    print("=" * 65)
    print(f"  Target Frames : {args.start} .. {args.start + args.count - 1} ({args.count} frames)")
    print(f"  Active Mode   : {args.pipeline.upper()}")
    print(f"  Timestamp     : {timestamp_str}")
    if args.publish_to_web:
        print(f"  Web Publish   : ENABLED (Syncing to public/data/renders/)")
    print("=" * 65 + "\n")

    rendered_geom = False
    rendered_data = False
    geom_dir = None
    dd_dir = None

    # 1. Run Geometry-Based Pipeline (render_aligned_video.py)
    if args.pipeline in ["all", "geometry"]:
        geom_dir = ROOT / "data" / "renders" / "geometry-based" / f"renders_{timestamp_str}"
        cmd_geom = [
            python_exe,
            str(ROOT / "skripts" / "render_aligned_video.py"),
            "--start", str(args.start),
            "--count", str(args.count),
            "--out-dir", str(geom_dir),
            "--trail-length", str(args.trail_length),
            "--trail-thickness", str(args.trail_thickness),
            "--ekf-r-scale", str(args.ekf_r_scale),
            "--ekf-q-scale", str(args.ekf_q_scale),
        ]
        if args.no_trails:
            cmd_geom.append("--no-trails")

        code = run_command_streamed(cmd_geom, "Geometry-Based Pipeline (PnP + EKF)")
        if code != 0:
            print(f"[Error] Geometry-based render failed with exit code {code}.")
            sys.exit(code)
        rendered_geom = True

    # 2. Run Data-Driven Pipeline (visualize_predictions.py)
    if args.pipeline in ["all", "data-driven", "cnn", "ml"]:
        dd_mode = "all" if args.pipeline in ["all", "data-driven"] else args.pipeline
        dd_dir = ROOT / "data" / "renders" / "data-driven" / f"renders_{timestamp_str}"
        cmd_dd = [
            python_exe,
            str(ROOT / "skripts" / "visualize_predictions.py"),
            "--pipeline", dd_mode,
            "--start", str(args.start),
            "--count", str(args.count),
            "--out-dir", str(dd_dir),
            "--trail-length", str(args.trail_length),
            "--trail-thickness", str(args.trail_thickness),
            "--pos-noise", str(args.pos_noise),
            "--accel-noise", str(args.accel_noise),
            "--rot-tau", str(args.rot_tau),
        ]
        if args.no_trails:
            cmd_dd.append("--no-trails")

        code = run_command_streamed(cmd_dd, f"Data-Driven Pipeline ({dd_mode.upper()})")
        if code != 0:
            print(f"[Error] Data-driven render failed with exit code {code}.")
            sys.exit(code)
        rendered_data = True

    # 3. Publish to web (public/data/renders) if requested
    if args.publish_to_web:
        import shutil
        import json
        
        pub_renders = ROOT / "public" / "data" / "renders"
        pub_geom = pub_renders / "geometry-based"
        pub_dd = pub_renders / "data-driven"
        pub_geom.mkdir(parents=True, exist_ok=True)
        pub_dd.mkdir(parents=True, exist_ok=True)
        
        if rendered_geom and geom_dir and geom_dir.exists():
            for mp4 in geom_dir.glob("*.mp4"):
                shutil.copy2(mp4, pub_geom / mp4.name)
            readme = geom_dir / "README.md"
            if readme.exists():
                shutil.copy2(readme, pub_geom / "README.md")
            print(f"[Web Publish] Copied geometry renders to {pub_geom}")

        if rendered_data and dd_dir and dd_dir.exists():
            for mp4 in dd_dir.glob("*.mp4"):
                shutil.copy2(mp4, pub_dd / mp4.name)
            readme = dd_dir / "README.md"
            if readme.exists():
                shutil.copy2(readme, pub_dd / "README.md")
            print(f"[Web Publish] Copied data-driven renders to {pub_dd}")

        # Update dataset-info.json
        info_json_path = ROOT / "public" / "api" / "dataset-info.json"
        info_json_path.parent.mkdir(parents=True, exist_ok=True)
        fps = 35.997
        duration = args.count / fps
        metadata = {
            "activeDirectory": f"renders_{timestamp_str}",
            "timestamp": timestamp_str,
            "frameCount": args.count,
            "duration": round(duration, 2),
            "startFrame": args.start,
            "fps": round(fps, 3),
            "timeSyncOffset": 18.97
        }
        with open(info_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Web Publish] Updated {info_json_path}")

    total_time = time.time() - t_start
    print("\n" + "=" * 65)
    print("  ALL REQUESTED PIPELINES SUCCESSFULLY RENDERED & ENCODED")
    print("=" * 65)
    print(f"  Total Wall Time: {total_time:.1f}s")
    if rendered_geom:
        print(f"  Geometry-Based Output : {ROOT / 'data' / 'renders' / 'geometry-based' / f'renders_{timestamp_str}'}")
    if rendered_data:
        print(f"  Data-Driven Output    : {ROOT / 'data' / 'renders' / 'data-driven' / f'renders_{timestamp_str}'}")
    if args.publish_to_web:
        print(f"  Web Public Directory  : {ROOT / 'public' / 'data' / 'renders'}")
        print("  -> Ready to commit and deploy via: git add public/ && git commit -m 'Update web renders'")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
