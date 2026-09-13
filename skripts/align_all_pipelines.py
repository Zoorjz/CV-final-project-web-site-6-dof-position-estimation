#!/usr/bin/env python3
"""align_all_pipelines.py

Consolidated benchmark and alignment comparison across all three tracking paradigms:
  1. Classical Optical PnP (+ 13-State EKF)
  2. CNN Temporal-3 Direct Pose Regressor (bar_pose_temporal3)
  3. ML Streaming MLP Tracker (pi_inference)

Generates:
  - Multi-pipeline trajectory overlay plot (PnP vs CNN vs ML vs Aligned Ground Truth)
  - Comprehensive statistical error comparison table (Median/Mean/RMSE Position & Rotation Errors)
  - JSON benchmark summary

Usage:
    python skripts/align_all_pipelines.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R_scipy

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Multi-pipeline Tracking Benchmark & GT Alignment Comparison.")
    parser.add_argument("--pnp-csv", type=str, default=str(ROOT / "data" / "aligned_tracking_comparison.csv"))
    parser.add_argument("--cnn-csv", type=str, default=str(ROOT / "data" / "cnn_aligned_tracking_comparison.csv"))
    parser.add_argument("--ml-csv", type=str, default=str(ROOT / "data" / "ml_aligned_tracking_comparison.csv"))
    parser.add_argument("--out-plot", type=str, default=str(ROOT / "skripts" / "multi_pipeline_comparison.png"))
    parser.add_argument("--out-json", type=str, default=str(ROOT / "data" / "multi_pipeline_benchmark.json"))
    args = parser.parse_args()

    results = {}

    # Load datasets
    df_pnp = pd.read_csv(args.pnp_csv) if Path(args.pnp_csv).exists() else None
    df_cnn = pd.read_csv(args.cnn_csv) if Path(args.cnn_csv).exists() else None
    df_ml = pd.read_csv(args.ml_csv) if Path(args.ml_csv).exists() else None

    # Compute stats
    def get_stats(df: pd.DataFrame, name: str):
        if df is None:
            return None
        pos_err = df["pos_error_mm"].values
        rot_err = df["rot_error_deg"].values
        return {
            "name": name,
            "num_frames": int(len(df)),
            "median_pos_error_mm": float(np.median(pos_err)),
            "mean_pos_error_mm": float(np.mean(pos_err)),
            "rmse_pos_error_mm": float(np.sqrt(np.mean(pos_err**2))),
            "p95_pos_error_mm": float(np.percentile(pos_err, 95)),
            "median_rot_error_deg": float(np.median(rot_err)),
            "mean_rot_error_deg": float(np.mean(rot_err)),
            "rmse_rot_error_deg": float(np.sqrt(np.mean(rot_err**2))),
            "p95_rot_error_deg": float(np.percentile(rot_err, 95))
        }

    stats_pnp = get_stats(df_pnp, "Classical SQPnP")
    stats_cnn = get_stats(df_cnn, "CNN Temporal-3")
    stats_ml = get_stats(df_ml, "ML Pi-Streaming")

    benchmark_summary = {
        "Classical_PnP": stats_pnp,
        "CNN_Temporal3": stats_cnn,
        "ML_Pi_Inference": stats_ml
    }

    # Print Table
    print("\n" + "=" * 80)
    print("           6-DoF TRACKING PIPELINE ACCURACY BENCHMARK (vs VR GROUND TRUTH)     ")
    print("=" * 80)
    print(f"{'Pipeline':<20} | {'Pos Med (mm)':<12} | {'Pos RMSE (mm)':<13} | {'Rot Med (deg)':<12} | {'Rot RMSE (deg)':<13} | {'Frames':<8}")
    print("-" * 80)

    for st in [stats_pnp, stats_cnn, stats_ml]:
        if st is not None:
            print(f"{st['name']:<20} | {st['median_pos_error_mm']:<12.2f} | {st['rmse_pos_error_mm']:<13.2f} | "
                  f"{st['median_rot_error_deg']:<12.2f} | {st['rmse_rot_error_deg']:<13.2f} | {st['num_frames']:<8}")
    print("=" * 80 + "\n")

    # Save JSON summary
    with open(args.out_json, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"[Summary] Saved benchmark stats to {args.out_json}")

    # Generate Consolidated Figure
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1. Position Z (Depth) Comparison
    if df_pnp is not None:
        axs[0].plot(df_pnp["video_time_s"], df_pnp["gt_aligned_z_mm"], "k-", lw=1.8, label="GT Depth Z (mm)", alpha=0.9)
        axs[0].plot(df_pnp["video_time_s"], df_pnp["pnp_z_mm"], color="#00bcd4", lw=1.2, label="PnP Depth Z (mm)", alpha=0.8)
    if df_cnn is not None:
        axs[0].plot(df_cnn["video_time_s"], df_cnn["cnn_z_mm"], color="#9c27b0", lw=1.2, label="CNN Depth Z (mm)", alpha=0.8)
    if df_ml is not None:
        axs[0].plot(df_ml["video_time_s"], df_ml["ml_z_mm"], color="#ff9800", lw=1.2, label="ML Depth Z (mm)", alpha=0.8)

    axs[0].set_ylabel("Depth Z in Cam (mm)")
    axs[0].set_title("Multi-Pipeline 6-DoF Tracking Comparison vs VR Ground Truth", fontsize=14, fontweight="bold")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right", ncol=4)

    # 2. 3D Position Error
    if df_pnp is not None:
        axs[1].plot(df_pnp["video_time_s"], df_pnp["pos_error_mm"], color="#00bcd4", lw=1.0, label=f"PnP Pos Err (Med: {stats_pnp['median_pos_error_mm']:.1f} mm)", alpha=0.7)
    if df_cnn is not None:
        axs[1].plot(df_cnn["video_time_s"], df_cnn["pos_error_mm"], color="#9c27b0", lw=1.0, label=f"CNN Pos Err (Med: {stats_cnn['median_pos_error_mm']:.1f} mm)", alpha=0.7)
    if df_ml is not None:
        axs[1].plot(df_ml["video_time_s"], df_ml["pos_error_mm"], color="#ff9800", lw=1.0, label=f"ML Pos Err (Med: {stats_ml['median_pos_error_mm']:.1f} mm)", alpha=0.7)

    axs[1].set_ylabel("3D Position Error (mm)")
    axs[1].set_ylim(0, 250)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right", ncol=3)

    # 3. 3D Rotation Error
    if df_pnp is not None:
        axs[2].plot(df_pnp["video_time_s"], df_pnp["rot_error_deg"], color="#00bcd4", lw=1.0, label=f"PnP Rot Err (Med: {stats_pnp['median_rot_error_deg']:.1f}°)", alpha=0.7)
    if df_cnn is not None:
        axs[2].plot(df_cnn["video_time_s"], df_cnn["rot_error_deg"], color="#9c27b0", lw=1.0, label=f"CNN Rot Err (Med: {stats_cnn['median_rot_error_deg']:.1f}°)", alpha=0.7)
    if df_ml is not None:
        axs[2].plot(df_ml["video_time_s"], df_ml["rot_error_deg"], color="#ff9800", lw=1.0, label=f"ML Rot Err (Med: {stats_ml['median_rot_error_deg']:.1f}°)", alpha=0.7)

    axs[2].set_ylabel("Rotation Error (deg)")
    axs[2].set_xlabel("Video Time (seconds)")
    axs[2].set_ylim(0, 45)
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right", ncol=3)

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    plt.close()
    print(f"[Plot] Saved multi-pipeline comparison plot to {args.out_plot}")


if __name__ == "__main__":
    main()
