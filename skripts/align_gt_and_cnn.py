#!/usr/bin/env python3
"""align_gt_and_cnn.py

Solves temporal synchronization and spatial calibration (Model-to-Camera transformation)
between the CNN 6-DoF predictions (CNN/bar_pose_temporal3) and the Ground Truth
VR Controller tracking log (from Unity/Meta Quest OpenXR).

Formulation:
    1. Temporal Synchronization:
       t_gt = t_video + delta_t
    2. Ground Truth in OpenCV Camera Frame:
       T_gt_in_cam(t) = T_cam_to_vr * T_vr_to_ctrl(t) * T_ctrl_to_bar
    3. Model to Camera Extrinsic Calibration:
       T_cnn_in_cam(t) = T_model_to_cam * T_cnn_raw(t)
       p_cam = R_M * p_cnn_raw + t_M
       R_cam = R_M * R_cnn_raw

Usage:
    python skripts/align_gt_and_cnn.py --plot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

ROOT = Path(__file__).resolve().parent.parent
CNN_DIR = ROOT / "CNN" / "bar_pose_temporal3"
if str(CNN_DIR) not in sys.path:
    sys.path.insert(0, str(CNN_DIR))

from infer_video import VideoPosePredictor  # noqa: E402

# Coordinate parity matrix: Unity Left-Handed to Standard Right-Handed
S_LHS_TO_RHS = np.diag([1.0, -1.0, 1.0])


def unity_to_rhs_pose(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts Unity Left-Handed coordinates to Right-Handed standard frame."""
    pos_rhs = pos @ S_LHS_TO_RHS
    R_lhs = R_scipy.from_quat(quat).as_matrix()
    R_rhs = S_LHS_TO_RHS @ R_lhs @ S_LHS_TO_RHS
    if np.linalg.det(R_rhs) < 0:
        R_rhs = -R_rhs
    return pos_rhs, R_rhs


def extract_cnn_trajectory(
    video_path: str,
    timestamps_path: str,
    checkpoint_path: str,
    backbone_path: str,
    threads: int = 4
) -> pd.DataFrame:
    """Extracts 6-DoF bar pose predictions across all frames using the Temporal-3 CNN model."""
    df_ts = pd.read_csv(timestamps_path) if Path(timestamps_path).exists() else None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[CNN] Loading VideoPosePredictor from {checkpoint_path}...", flush=True)
    predictor = VideoPosePredictor(Path(checkpoint_path), Path(backbone_path), threads=threads)

    print(f"[CNN] Running inference on {video_path} ({total_frames} frames)...", flush=True)
    times, valids, tvecs, quats = [], [], [], []
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if df_ts is not None and frame_idx < len(df_ts):
            t_sec = df_ts["pts_ms"].iloc[frame_idx] / 1000.0
        else:
            t_sec = frame_idx / video_fps

        times.append(t_sec)
        try:
            pose = predictor.predict(frame_idx, frame)
            t_m = pose[:3]
            q = pose[3:]
            q_norm = q / np.linalg.norm(q) if np.linalg.norm(q) > 1e-6 else np.array([0., 0., 0., 1.])

            tvecs.append(t_m * 1000.0)  # Convert to mm
            quats.append(q_norm)
            valids.append(bool(np.isfinite(t_m).all() and np.isfinite(q).all()))
        except Exception:
            tvecs.append(np.array([np.nan, np.nan, np.nan]))
            quats.append(np.array([np.nan, np.nan, np.nan, np.nan]))
            valids.append(False)

        frame_idx += 1
        if frame_idx % 500 == 0 or frame_idx == total_frames:
            elapsed = time.time() - t0
            fps_inf = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  [CNN] Frame {frame_idx}/{total_frames} ({elapsed:.1f}s, {fps_inf:.1f} fps)", flush=True)

    cap.release()
    tvecs = np.array(tvecs)
    quats = np.array(quats)

    df_cnn = pd.DataFrame({
        "frame_idx": np.arange(len(times)),
        "time_s": times,
        "valid": valids,
        "tx_mm": tvecs[:, 0],
        "ty_mm": tvecs[:, 1],
        "tz_mm": tvecs[:, 2],
        "qx": quats[:, 0],
        "qy": quats[:, 1],
        "qz": quats[:, 2],
        "qw": quats[:, 3]
    })

    print(f"[CNN] Done in {time.time()-t0:.2f}s: {df_cnn['valid'].sum()}/{len(df_cnn)} valid frames "
          f"({df_cnn['valid'].mean()*100:.1f}%).", flush=True)
    return df_cnn


def solve_model_to_camera_alignment(
    df_cnn: pd.DataFrame,
    df_gt: pd.DataFrame,
    calib_pnp: dict
) -> tuple[dict, pd.DataFrame]:
    """Solves the Model-to-Camera coordinate transformation mapping CNN output into OpenCV camera space."""
    print("[Alignment] Solving Model-to-Camera transformation for CNN...", flush=True)

    dt = calib_pnp["time_offset_seconds"]
    R_X = np.array(calib_pnp["camera_to_vr_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_X = np.array(calib_pnp["camera_to_vr_extrinsics"]["translation_m"], dtype=np.float64)
    R_Y = np.array(calib_pnp["controller_to_bar_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_Y = np.array(calib_pnp["controller_to_bar_extrinsics"]["translation_m"], dtype=np.float64)

    t_v = df_cnn["time_s"].values
    t_g = t_v + dt
    t_gt_all = df_gt["timestamp_seconds"].values

    mask = (t_g >= t_gt_all[0] + 0.5) & (t_g <= t_gt_all[-1] - 0.5) & df_cnn["valid"].values
    indices = np.where(mask)[0]

    cols_p = ["right_px", "right_py", "right_pz"]
    cols_q = ["right_qx", "right_qy", "right_qz", "right_qw"]

    p_gt_vr = np.zeros((len(indices), 3))
    q_gt_vr = np.zeros((len(indices), 4))
    for d in range(3):
        p_gt_vr[:, d] = np.interp(t_g[indices], t_gt_all, df_gt[cols_p[d]].values)
    for d in range(4):
        q_gt_vr[:, d] = np.interp(t_g[indices], t_gt_all, df_gt[cols_q[d]].values)

    p_gt_vr[:, 1] = -p_gt_vr[:, 1]  # LHS to RHS
    q_gt_vr /= np.linalg.norm(q_gt_vr, axis=1, keepdims=True)
    R_gt_vr = R_scipy.from_quat(q_gt_vr).as_matrix()

    # Ground Truth Bar in OpenCV Camera Frame: T_cam_to_bar = X * T_vr_to_ctrl * Y
    p_gt_cam = np.einsum("ij,nj->ni", R_X, np.einsum("nij,j->ni", R_gt_vr, t_Y) + p_gt_vr) + t_X
    R_gt_cam = np.einsum("ij,njk,kl->nil", R_X, R_gt_vr, R_Y)

    # Raw CNN model outputs
    p_cnn_raw = df_cnn.loc[indices, ["tx_mm", "ty_mm", "tz_mm"]].values / 1000.0
    q_cnn_raw = df_cnn.loc[indices, ["qx", "qy", "qz", "qw"]].values
    R_cnn_raw = R_scipy.from_quat(q_cnn_raw).as_matrix()

    # Solve optimal rigid transformation T_M = (R_M, t_M) such that:
    # p_cam = R_M * p_cnn + t_M
    # R_cam = R_M * R_cnn
    def loss(p):
        rx, ry, rz, tx, ty, tz = p
        R_M = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
        t_M = np.array([tx, ty, tz])
        p_p = (R_M @ p_cnn_raw.T).T + t_M
        pos_err = (p_gt_cam - p_p) * 2.0
        R_p = np.einsum("ij,njk->nik", R_M, R_cnn_raw)
        R_diff = np.einsum("nji,njk->nik", R_gt_cam, R_p)
        rot_err = R_scipy.from_matrix(R_diff).as_rotvec()
        return np.concatenate([pos_err.ravel(), rot_err.ravel()])

    # Grid search for initial rotation seed
    best_cost = 1e9
    best_p0 = [0, 0, 0, 0, 0, 0]
    for rx in [-np.pi/2, 0, np.pi/2, np.pi]:
        for ry in [-np.pi/2, 0, np.pi/2, np.pi]:
            for rz in [-np.pi/2, 0, np.pi/2, np.pi]:
                p0_cand = [rx, ry, rz, 0, 0, 0]
                R_cand = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
                t_cand = p_gt_cam.mean(axis=0) - (R_cand @ p_cnn_raw.mean(axis=0))
                p0_cand[3:6] = t_cand.tolist()
                c = np.sum(loss(p0_cand)**2)
                if c < best_cost:
                    best_cost = c
                    best_p0 = p0_cand

    res = least_squares(loss, best_p0, loss="soft_l1", f_scale=0.1)
    rx, ry, rz, tx, ty, tz = res.x
    R_M = R_scipy.from_rotvec([rx, ry, rz])
    t_M = np.array([tx, ty, tz])

    # Transform all CNN predictions to OpenCV Camera space
    p_cnn_cam = (R_M.as_matrix() @ p_cnn_raw.T).T + t_M
    R_cnn_cam = np.einsum("ij,njk->nik", R_M.as_matrix(), R_cnn_raw)
    q_cnn_cam = R_scipy.from_matrix(R_cnn_cam).as_quat()

    # Compute errors vs Ground Truth in camera space
    pos_errors_mm = np.linalg.norm(p_gt_cam - p_cnn_cam, axis=1) * 1000.0
    R_diff = np.einsum("nji,njk->nik", R_gt_cam, R_cnn_cam)
    rot_errors_deg = np.degrees(np.linalg.norm(R_scipy.from_matrix(R_diff).as_rotvec(), axis=1))

    calib_dict = {
        "model": "CNN_Temporal3",
        "time_offset_seconds": float(dt),
        "model_to_camera_transform": {
            "translation_m": t_M.tolist(),
            "rotation_rotvec": [float(rx), float(ry), float(rz)],
            "rotation_euler_deg": R_M.as_euler("xyz", degrees=True).tolist(),
            "rotation_matrix": R_M.as_matrix().tolist()
        },
        "camera_to_vr_extrinsics": calib_pnp["camera_to_vr_extrinsics"],
        "controller_to_bar_extrinsics": calib_pnp["controller_to_bar_extrinsics"],
        "metrics": {
            "num_synchronized_frames": int(len(indices)),
            "median_pos_error_mm": float(np.median(pos_errors_mm)),
            "mean_pos_error_mm": float(np.mean(pos_errors_mm)),
            "rmse_pos_error_mm": float(np.sqrt(np.mean(pos_errors_mm**2))),
            "median_rot_error_deg": float(np.median(rot_errors_deg)),
            "mean_rot_error_deg": float(np.mean(rot_errors_deg)),
            "rmse_rot_error_deg": float(np.sqrt(np.mean(rot_errors_deg**2)))
        }
    }

    q_gt_cam = R_scipy.from_matrix(R_gt_cam).as_quat()

    df_aligned = pd.DataFrame({
        "frame_idx": indices,
        "video_time_s": t_v[indices],
        "gt_time_s": t_g[indices],
        "cnn_cam_x_mm": p_cnn_cam[:, 0] * 1000.0,
        "cnn_cam_y_mm": p_cnn_cam[:, 1] * 1000.0,
        "cnn_cam_z_mm": p_cnn_cam[:, 2] * 1000.0,
        "cnn_cam_qx": q_cnn_cam[:, 0],
        "cnn_cam_qy": q_cnn_cam[:, 1],
        "cnn_cam_qz": q_cnn_cam[:, 2],
        "cnn_cam_qw": q_cnn_cam[:, 3],
        "gt_aligned_x_mm": p_gt_cam[:, 0] * 1000.0,
        "gt_aligned_y_mm": p_gt_cam[:, 1] * 1000.0,
        "gt_aligned_z_mm": p_gt_cam[:, 2] * 1000.0,
        "gt_aligned_qx": q_gt_cam[:, 0],
        "gt_aligned_qy": q_gt_cam[:, 1],
        "gt_aligned_qz": q_gt_cam[:, 2],
        "gt_aligned_qw": q_gt_cam[:, 3],
        "pos_error_mm": pos_errors_mm,
        "rot_error_deg": rot_errors_deg
    })

    print("[CNN Calibration] Success! Transformation Model -> OpenCV Camera Frame:", flush=True)
    print(f"  Rotation Euler (XYZ): {R_M.as_euler('xyz', degrees=True)} deg")
    print(f"  Translation: {t_M} m")
    print(f"  Accuracy in Camera: Median Pos Error = {np.median(pos_errors_mm):.2f} mm (RMSE: {np.sqrt(np.mean(pos_errors_mm**2)):.2f} mm)")
    print(f"                      Median Rot Error = {np.median(rot_errors_deg):.2f} deg (RMSE: {np.sqrt(np.mean(rot_errors_deg**2)):.2f} deg)")

    return calib_dict, df_aligned


def plot_alignment_results(df_aligned: pd.DataFrame, out_path: str):
    """Generates a publication-grade diagnostic comparison plot."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    t = df_aligned["video_time_s"]

    # 1. Position Trajectories in Camera Frame
    axs[0].plot(t, df_aligned["cnn_cam_x_mm"], "r-", label="CNN Cam X (mm)", alpha=0.8)
    axs[0].plot(t, df_aligned["gt_aligned_x_mm"], "r--", label="GT X (mm)", alpha=0.8)
    axs[0].plot(t, df_aligned["cnn_cam_y_mm"], "g-", label="CNN Cam Y (mm)", alpha=0.8)
    axs[0].plot(t, df_aligned["gt_aligned_y_mm"], "g--", label="GT Y (mm)", alpha=0.8)
    axs[0].plot(t, df_aligned["cnn_cam_z_mm"], "b-", label="CNN Cam Z (mm)", alpha=0.8)
    axs[0].plot(t, df_aligned["gt_aligned_z_mm"], "b--", label="GT Z (mm)", alpha=0.8)
    axs[0].set_ylabel("Position in Cam (mm)")
    axs[0].set_title("Calibrated 6-DoF Pose Comparison: CNN (Temporal-3) vs VR Ground Truth (in Camera Frame)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right", ncol=3)

    # 2. Position Error
    axs[1].plot(t, df_aligned["pos_error_mm"], "m-", lw=1.2, label=f"3D Pos Error (Median: {df_aligned['pos_error_mm'].median():.1f} mm)")
    axs[1].axhline(df_aligned["pos_error_mm"].median(), color="black", linestyle=":", label="Median Error")
    axs[1].set_ylabel("Pos Error (mm)")
    axs[1].set_ylim(0, min(500.0, df_aligned["pos_error_mm"].quantile(0.95) * 1.5))
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    # 3. Rotation Angles
    cnn_q = np.array(df_aligned[["cnn_cam_qx", "cnn_cam_qy", "cnn_cam_qz", "cnn_cam_qw"]].values, dtype=np.float64, copy=True)
    gt_q = np.array(df_aligned[["gt_aligned_qx", "gt_aligned_qy", "gt_aligned_qz", "gt_aligned_qw"]].values, dtype=np.float64, copy=True)
    cnn_eulers = R_scipy.from_quat(cnn_q).as_euler("xyz", degrees=True)
    gt_eulers = R_scipy.from_quat(gt_q).as_euler("xyz", degrees=True)
    axs[2].plot(t, cnn_eulers[:, 0], "r-", label="CNN Cam Roll", alpha=0.8)
    axs[2].plot(t, gt_eulers[:, 0], "r--", label="GT Roll", alpha=0.8)
    axs[2].plot(t, cnn_eulers[:, 1], "g-", label="CNN Cam Pitch", alpha=0.8)
    axs[2].plot(t, gt_eulers[:, 1], "g--", label="GT Pitch", alpha=0.8)
    axs[2].plot(t, cnn_eulers[:, 2], "b-", label="CNN Cam Yaw", alpha=0.8)
    axs[2].plot(t, gt_eulers[:, 2], "b--", label="GT Yaw", alpha=0.8)
    axs[2].set_ylabel("Euler Angles (deg)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right", ncol=3)

    # 4. Rotation Error
    axs[3].plot(t, df_aligned["rot_error_deg"], "c-", lw=1.2, label=f"Angular Error (Median: {df_aligned['rot_error_deg'].median():.2f}°)")
    axs[3].axhline(df_aligned["rot_error_deg"].median(), color="black", linestyle=":", label="Median Error")
    axs[3].set_ylabel("Rot Error (deg)")
    axs[3].set_xlabel("Video Time (seconds)")
    axs[3].set_ylim(0, min(80.0, df_aligned["rot_error_deg"].quantile(0.95) * 1.5))
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved CNN alignment diagnostic figure to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Solve Model-to-Camera Alignment between CNN Predictions and VR Controller GT.")
    parser.add_argument("--video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--video-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--gt-csv", type=str, default=str(ROOT / "data" / "GT_pos" / "20260911_130547_284_P01" / "pose_before_render.csv"))
    parser.add_argument("--alignment-pnp", type=str, default=str(ROOT / "data" / "alignment_calibration.json"))
    parser.add_argument("--checkpoint", type=str, default=str(CNN_DIR / "best_combined.pt"))
    parser.add_argument("--backbone", type=str, default=str(CNN_DIR / "yolo26n-cls.pt"))
    parser.add_argument("--cache", type=str, default=str(ROOT / "data" / "cnn_trajectory_extracted.csv"))
    parser.add_argument("--out-calib", type=str, default=str(ROOT / "data" / "cnn_alignment_calibration.json"))
    parser.add_argument("--out-csv", type=str, default=str(ROOT / "data" / "cnn_aligned_tracking_comparison.csv"))
    parser.add_argument("--plot", action="store_true", default=True, help="Save diagnostic comparison plot")
    parser.add_argument("--plot-out", type=str, default=str(ROOT / "skripts" / "cnn_alignment_plots.png"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    # 1. Extract or load CNN trajectory
    if Path(args.cache).exists():
        print(f"[CNN] Loading cached trajectory from {args.cache}", flush=True)
        df_cnn = pd.read_csv(args.cache)
    else:
        df_cnn = extract_cnn_trajectory(args.video, args.video_ts, args.checkpoint, args.backbone, threads=args.threads)
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        df_cnn.to_csv(args.cache, index=False)
        print(f"[CNN] Saved extracted trajectory to {args.cache}", flush=True)

    # 2. Load Ground Truth & PnP Alignment
    print(f"[GT] Loading Ground Truth poses from {args.gt_csv}", flush=True)
    df_gt = pd.read_csv(args.gt_csv)
    with open(args.alignment_pnp) as f:
        calib_pnp = json.load(f)

    # 3. Solve Model to Camera Frame Alignment
    calib_dict, df_aligned = solve_model_to_camera_alignment(df_cnn, df_gt, calib_pnp)

    # 4. Save Outputs
    Path(args.out_calib).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_calib, "w") as f:
        json.dump(calib_dict, f, indent=2)
    print(f"[Output] Saved CNN alignment calibration to {args.out_calib}", flush=True)

    df_aligned.to_csv(args.out_csv, index=False)
    print(f"[Output] Saved CNN aligned comparison dataset to {args.out_csv}", flush=True)

    if args.plot:
        plot_alignment_results(df_aligned, args.plot_out)


if __name__ == "__main__":
    main()
