#!/usr/bin/env python3
"""align_gt_and_pnp.py

Solves the temporal and spatial alignment (Hand-Eye Calibration) between
the 6-DoF optical PnP bar tracking (from video) and the Ground Truth Right
VR Controller tracking log (from Unity/Meta Quest OpenXR).

Theoretical formulation:
-------------------------
1. Temporal Synchronization:
   t_gt = t_video + delta_t
   Determined via cross-correlation of angular velocities and non-linear
   least squares alignment.

2. Spatial Hand-Eye Calibration:
   The physical bar has 4 optical markers (frame O_bar).
   The VR controller is rigidly mounted to the bar (frame O_ctrl).
   The camera is fixed in the room/VR world (frame O_cam).
   
   At any time t:
       T_cam_to_bar(t) = T_cam_to_vr * T_vr_to_ctrl(t) * T_ctrl_to_bar
   
   Where:
       - T_cam_to_bar(t): Measured by camera PnP [R_pnp(t) | t_pnp(t)]
       - T_vr_to_ctrl(t): Measured by VR tracking [R_gt(t) | p_gt(t)]
       - T_cam_to_vr: Static 6-DoF camera extrinsics in VR world (X)
       - T_ctrl_to_bar: Static 6-DoF controller-to-bar mounting offset (Y)

Usage:
    python skripts/align_gt_and_pnp.py --plot
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R_scipy

ROOT = Path(__file__).resolve().parent.parent

# Coordinate parity matrix: Unity Left-Handed (X right, Y up, Z forward)
# to Standard Right-Handed (X right, Y down, Z forward)
S_LHS_TO_RHS = np.diag([1.0, -1.0, 1.0])


def unity_to_rhs_pose(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts Unity Left-Handed coordinates to Right-Handed standard frame."""
    pos_rhs = pos @ S_LHS_TO_RHS
    R_lhs = R_scipy.from_quat(quat).as_matrix()
    R_rhs = S_LHS_TO_RHS @ R_lhs @ S_LHS_TO_RHS
    if np.linalg.det(R_rhs) < 0:
        R_rhs = -R_rhs
    return pos_rhs, R_rhs


def detect_blobs(gray: np.ndarray, thresh: int = 240, min_area: float = 2.0, max_area: float = 1500.0) -> tuple[np.ndarray, np.ndarray]:
    """Detects sub-pixel centroids of bright optical LED markers."""
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []
    for c in contours:
        M = cv2.moments(c)
        if min_area < M["m00"] < max_area:
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
            areas.append(M["m00"])
    return np.array(pts), np.array(areas)


def solve_pnp_frame(
    img_pts: np.ndarray,
    marker_pos: dict[str, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    perms: list[tuple[str, ...]],
    preferred_perm: tuple[str, ...] | None = None,
    reproj_limit: float = 50.0
) -> tuple[np.ndarray, np.ndarray, float, tuple[str, ...]] | None:
    """Evaluates marker permutations to find the best valid PnP pose."""
    perm_list = [preferred_perm] + [p for p in perms if p != preferred_perm] if preferred_perm is not None else perms
    best = None
    for perm in perm_list:
        obj = np.array([marker_pos[m] for m in perm])
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R_mat, _ = cv2.Rodrigues(rvec)
        cam_pts = (R_mat @ obj.T).T + tvec.ravel()
        if (cam_pts[:, 2] <= 0).any():
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean()
        if not np.isfinite(err) or err > reproj_limit:
            continue
        if best is None or err < best[2]:
            best = (rvec, tvec, err, perm)
        if preferred_perm is not None and perm == preferred_perm and err < 2.0:
            break
    return best


def extract_pnp_trajectory(
    video_path: str,
    timestamps_path: str,
    K: np.ndarray,
    dist: np.ndarray,
    marker_pos: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Extracts 6-DoF PnP poses across all frames of the input video."""
    df_ts = pd.read_csv(timestamps_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[PnP] Extracting poses from {video_path} ({total_frames} frames)...")
    
    marker_ids = sorted(marker_pos.keys())
    perms = list(itertools.permutations(marker_ids))

    times, valids, tvecs, quats, errs, perms_used = [], [], [], [], [], []
    frame_idx = 0
    prev_perm = None
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t_sec = df_ts['pts_ms'].iloc[frame_idx] / 1000.0 if frame_idx < len(df_ts) else frame_idx / 30.0
        times.append(t_sec)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts, areas = detect_blobs(gray)
        
        pose_found = False
        if len(pts) >= 4:
            top_pts = pts[np.argsort(areas)[::-1][:4]]
            res = solve_pnp_frame(top_pts, marker_pos, K, dist, perms, preferred_perm=prev_perm)
            if res is not None:
                rvec, tvec, err, perm = res
                prev_perm = perm
                tvecs.append(tvec.ravel())
                rot = R_scipy.from_rotvec(rvec.ravel())
                quats.append(rot.as_quat())
                errs.append(err)
                valids.append(True)
                perms_used.append(str(perm))
                pose_found = True

        if not pose_found:
            prev_perm = None
            tvecs.append(np.array([np.nan, np.nan, np.nan]))
            quats.append(np.array([np.nan, np.nan, np.nan, np.nan]))
            errs.append(np.nan)
            valids.append(False)
            perms_used.append("")

        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"  [PnP] Frame {frame_idx}/{total_frames} ({time.time()-t0:.1f}s), valid: {sum(valids)}")

    cap.release()
    tvecs = np.array(tvecs)
    quats = np.array(quats)
    
    df_pnp = pd.DataFrame({
        'frame_idx': np.arange(len(times)),
        'time_s': times,
        'valid': valids,
        'tx_mm': tvecs[:, 0],
        'ty_mm': tvecs[:, 1],
        'tz_mm': tvecs[:, 2],
        'qx': quats[:, 0],
        'qy': quats[:, 1],
        'qz': quats[:, 2],
        'qw': quats[:, 3],
        'reproj_err_px': errs,
        'perm': perms_used
    })
    print(f"[PnP] Done in {time.time()-t0:.2f}s: {df_pnp['valid'].sum()}/{len(df_pnp)} valid poses "
          f"({df_pnp['valid'].mean()*100:.1f}%), median reproj err: {df_pnp['reproj_err_px'].median():.3f} px.")
    return df_pnp


def solve_spatial_temporal_alignment(
    df_pnp: pd.DataFrame,
    df_gt: pd.DataFrame,
    initial_dt_guess: float = 18.97,
    dt_search_radius: float = 0.5
) -> tuple[dict, pd.DataFrame]:
    """Solves time synchronization offset and 12-DoF Hand-Eye calibration."""
    print("[Alignment] Starting joint temporal and spatial Hand-Eye calibration...")
    
    pnp_valid = df_pnp['valid'].values
    t_pnp = df_pnp['time_s'].values
    pnp_quats = df_pnp[['qx', 'qy', 'qz', 'qw']].values
    pnp_tvecs_m = df_pnp[['tx_mm', 'ty_mm', 'tz_mm']].values / 1000.0

    t_gt = df_gt['timestamp_seconds'].values
    gt_pos_raw = df_gt[['right_px', 'right_py', 'right_pz']].values
    gt_quat_raw = df_gt[['right_qx', 'right_qy', 'right_qz', 'right_qw']].values

    # Convert GT to RHS
    gt_pos_rhs = np.zeros_like(gt_pos_raw)
    gt_R_rhs = []
    for i in range(len(df_gt)):
        p_r, R_r = unity_to_rhs_pose(gt_pos_raw[i], gt_quat_raw[i])
        gt_pos_rhs[i] = p_r
        gt_R_rhs.append(R_r)
    gt_R_rhs = np.array(gt_R_rhs)

    gt_quats_rhs = R_scipy.from_matrix(gt_R_rhs).as_quat()
    for i in range(1, len(gt_quats_rhs)):
        if np.dot(gt_quats_rhs[i], gt_quats_rhs[i-1]) < 0:
            gt_quats_rhs[i] = -gt_quats_rhs[i]

    def eval_at_dt(dt: float, params: np.ndarray):
        t_targets = t_pnp + dt
        mask = pnp_valid & (t_targets >= t_gt[0] + 0.5) & (t_targets <= t_gt[-1] - 0.5)
        indices = np.where(mask)[0]
        t_sync = t_targets[indices]

        # Interpolate GT positions and quaternions
        gt_p_sync = np.zeros((len(indices), 3))
        for d in range(3):
            gt_p_sync[:, d] = np.interp(t_sync, t_gt, gt_pos_rhs[:, d])
            
        gt_q_sync = np.zeros((len(indices), 4))
        for d in range(4):
            gt_q_sync[:, d] = np.interp(t_sync, t_gt, gt_quats_rhs[:, d])
        gt_q_sync /= np.linalg.norm(gt_q_sync, axis=1, keepdims=True)
        gt_R_sync = R_scipy.from_quat(gt_q_sync).as_matrix()

        pnp_p_sync = pnp_tvecs_m[indices]
        pnp_R_sync = R_scipy.from_quat(pnp_quats[indices]).as_matrix()

        rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = params
        R_X = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
        t_X = np.array([tx, ty, tz])
        R_Y = R_scipy.from_rotvec([ox, oy, oz]).as_matrix()
        t_Y = np.array([bx, by, bz])

        # Predicted bar pose from GT:
        # T_cam_to_bar = X * T_vr_to_ctrl * Y
        R_pred = np.einsum('ij,njk,kl->nil', R_X, gt_R_sync, R_Y)
        p_pred = np.einsum('ij,nj->ni', R_X, np.einsum('nij,j->ni', gt_R_sync, t_Y) + gt_p_sync) + t_X

        pos_err = pnp_p_sync - p_pred
        R_diff = np.einsum('nji,njk->nik', pnp_R_sync, R_pred)
        rot_err = R_scipy.from_matrix(R_diff).as_rotvec()

        return indices, pos_err, rot_err, pnp_p_sync, p_pred, pnp_R_sync, R_pred, gt_p_sync, gt_R_sync

    # Search fine time offsets
    best_cost = 1e9
    best_opt = None
    best_dt = initial_dt_guess
    
    dt_candidates = np.arange(initial_dt_guess - dt_search_radius, initial_dt_guess + dt_search_radius, 0.01)
    p0 = [-0.35, -0.02, -0.06, 1.80, -0.07, 1.47, -1.44, 1.05, -1.45, 0.023, -0.007, -0.0015]

    for dt_cand in dt_candidates:
        def loss(p):
            indices, pos_err, rot_err, _, _, _, _, _, _ = eval_at_dt(dt_cand, p)
            sub = slice(0, len(indices), 4)
            return np.concatenate([rot_err[sub].ravel(), (pos_err[sub] * 3.0).ravel()])
        
        res = least_squares(loss, p0, loss='soft_l1', f_scale=0.1, method='trf')
        if res.cost < best_cost:
            best_cost = res.cost
            best_opt = res.x
            best_dt = dt_cand

    # Final evaluation on all frames
    indices, pos_err, rot_err, p_pnp, p_pred, R_pnp, R_pred, gt_p_sync, gt_R_sync = eval_at_dt(best_dt, best_opt)
    
    pos_errors_mm = np.linalg.norm(pos_err, axis=1) * 1000.0
    rot_errors_deg = np.degrees(np.linalg.norm(rot_err, axis=1))

    rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = best_opt
    R_X = R_scipy.from_rotvec([rx, ry, rz])
    t_X = np.array([tx, ty, tz])
    R_Y = R_scipy.from_rotvec([ox, oy, oz])
    t_Y = np.array([bx, by, bz])

    calib_dict = {
        "time_offset_seconds": float(best_dt),
        "camera_to_vr_extrinsics": {
            "translation_m": t_X.tolist(),
            "rotation_rotvec": [float(rx), float(ry), float(rz)],
            "rotation_euler_deg": R_X.as_euler('xyz', degrees=True).tolist(),
            "rotation_matrix": R_X.as_matrix().tolist()
        },
        "controller_to_bar_extrinsics": {
            "translation_m": t_Y.tolist(),
            "translation_mm": (t_Y * 1000.0).tolist(),
            "rotation_rotvec": [float(ox), float(oy), float(oz)],
            "rotation_euler_deg": R_Y.as_euler('xyz', degrees=True).tolist(),
            "rotation_matrix": R_Y.as_matrix().tolist()
        },
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

    # Build synchronized dataset table
    q_pred = R_scipy.from_matrix(R_pred).as_quat()
    q_pnp_sync = R_scipy.from_matrix(R_pnp).as_quat()
    
    df_aligned = pd.DataFrame({
        'frame_idx': indices,
        'video_time_s': t_pnp[indices],
        'gt_time_s': t_pnp[indices] + best_dt,
        'pnp_x_mm': p_pnp[:, 0] * 1000.0,
        'pnp_y_mm': p_pnp[:, 1] * 1000.0,
        'pnp_z_mm': p_pnp[:, 2] * 1000.0,
        'pnp_qx': q_pnp_sync[:, 0],
        'pnp_qy': q_pnp_sync[:, 1],
        'pnp_qz': q_pnp_sync[:, 2],
        'pnp_qw': q_pnp_sync[:, 3],
        'gt_aligned_x_mm': p_pred[:, 0] * 1000.0,
        'gt_aligned_y_mm': p_pred[:, 1] * 1000.0,
        'gt_aligned_z_mm': p_pred[:, 2] * 1000.0,
        'gt_aligned_qx': q_pred[:, 0],
        'gt_aligned_qy': q_pred[:, 1],
        'gt_aligned_qz': q_pred[:, 2],
        'gt_aligned_qw': q_pred[:, 3],
        'pos_error_mm': pos_errors_mm,
        'rot_error_deg': rot_errors_deg
    })

    print(f"[Alignment] Success!")
    print(f"  Time Offset (dt): {best_dt:.4f} s")
    print(f"  Camera in VR: Pos = {t_X} m, Euler = {R_X.as_euler('xyz', degrees=True)} deg")
    print(f"  Controller Offset: Pos = {t_Y*1000.0} mm, Euler = {R_Y.as_euler('xyz', degrees=True)} deg")
    print(f"  Median Pos Error: {np.median(pos_errors_mm):.2f} mm (Mean: {np.mean(pos_errors_mm):.2f} mm)")
    print(f"  Median Rot Error: {np.median(rot_errors_deg):.2f} deg (Mean: {np.mean(rot_errors_deg):.2f} deg)")

    return calib_dict, df_aligned


def plot_alignment_results(df_aligned: pd.DataFrame, out_path: str):
    """Generates a publication-grade diagnostic plot of aligned trajectories."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    t = df_aligned['video_time_s']
    
    # 1. Position Trajectory X, Y, Z
    axs[0].plot(t, df_aligned['pnp_x_mm'], 'r-', label='PnP X (mm)', alpha=0.8)
    axs[0].plot(t, df_aligned['gt_aligned_x_mm'], 'r--', label='GT X (mm)', alpha=0.8)
    axs[0].plot(t, df_aligned['pnp_y_mm'], 'g-', label='PnP Y (mm)', alpha=0.8)
    axs[0].plot(t, df_aligned['gt_aligned_y_mm'], 'g--', label='GT Y (mm)', alpha=0.8)
    axs[0].plot(t, df_aligned['pnp_z_mm'], 'b-', label='PnP Z (mm)', alpha=0.8)
    axs[0].plot(t, df_aligned['gt_aligned_z_mm'], 'b--', label='GT Z (mm)', alpha=0.8)
    axs[0].set_ylabel("Position in Cam (mm)")
    axs[0].set_title("Synchronized 6-DoF Pose Comparison: Optical PnP vs Aligned VR Ground Truth")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper right', ncol=3)

    # 2. Position Error
    axs[1].plot(t, df_aligned['pos_error_mm'], 'm-', lw=1.2, label=f"3D Pos Error (Median: {df_aligned['pos_error_mm'].median():.1f} mm)")
    axs[1].axhline(df_aligned['pos_error_mm'].median(), color='black', linestyle=':', label='Median Error')
    axs[1].set_ylabel("Pos Error (mm)")
    axs[1].set_ylim(0, min(150.0, df_aligned['pos_error_mm'].quantile(0.95)*1.5))
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper right')

    # 3. Rotation Angles
    pnp_eulers = R_scipy.from_quat(df_aligned[['pnp_qx', 'pnp_qy', 'pnp_qz', 'pnp_qw']].values).as_euler('xyz', degrees=True)
    gt_eulers = R_scipy.from_quat(df_aligned[['gt_aligned_qx', 'gt_aligned_qy', 'gt_aligned_qz', 'gt_aligned_qw']].values).as_euler('xyz', degrees=True)
    axs[2].plot(t, pnp_eulers[:, 0], 'r-', label='PnP Roll', alpha=0.8)
    axs[2].plot(t, gt_eulers[:, 0], 'r--', label='GT Roll', alpha=0.8)
    axs[2].plot(t, pnp_eulers[:, 1], 'g-', label='PnP Pitch', alpha=0.8)
    axs[2].plot(t, gt_eulers[:, 1], 'g--', label='GT Pitch', alpha=0.8)
    axs[2].plot(t, pnp_eulers[:, 2], 'b-', label='PnP Yaw', alpha=0.8)
    axs[2].plot(t, gt_eulers[:, 2], 'b--', label='GT Yaw', alpha=0.8)
    axs[2].set_ylabel("Euler Angles (deg)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper right', ncol=3)

    # 4. Rotation Error
    axs[3].plot(t, df_aligned['rot_error_deg'], 'c-', lw=1.2, label=f"Angular Error (Median: {df_aligned['rot_error_deg'].median():.2f}°)")
    axs[3].axhline(df_aligned['rot_error_deg'].median(), color='black', linestyle=':', label='Median Error')
    axs[3].set_ylabel("Rot Error (deg)")
    axs[3].set_xlabel("Video Time (seconds)")
    axs[3].set_ylim(0, min(30.0, df_aligned['rot_error_deg'].quantile(0.95)*1.5))
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved alignment diagnostic figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Solve Temporal & Spatial Alignment between Video PnP and VR Controller GT.")
    parser.add_argument("--video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--video-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--gt-csv", type=str, default=str(ROOT / "data" / "GT_pos" / "20260911_130547_284_P01" / "pose_before_render.csv"))
    parser.add_argument("--calib", type=str, default=str(ROOT / "data" / "camera_calibration.yaml"))
    parser.add_argument("--geometry", type=str, default=str(ROOT / "data" / "geometry.yaml"))
    parser.add_argument("--pnp-cache", type=str, default=str(ROOT / "skripts" / "pnp_trajectory_extracted.csv"))
    parser.add_argument("--out-calib", type=str, default=str(ROOT / "data" / "alignment_calibration.json"))
    parser.add_argument("--out-csv", type=str, default=str(ROOT / "data" / "aligned_tracking_comparison.csv"))
    parser.add_argument("--plot", action="store_true", default=True, help="Save diagnostic comparison plot")
    parser.add_argument("--plot-out", type=str, default=str(ROOT / "skripts" / "alignment_plots.png"))
    args = parser.parse_args()

    # Load Calibration & Geometry
    cal = yaml.safe_load(open(args.calib))
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    K[:2, :] *= 0.5  # 1280x800 -> 640x400
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

    geom = yaml.safe_load(open(args.geometry))
    marker_pos = {m["id"]: np.array(m["position"], dtype=np.float64) for m in geom["markers"]}

    # 1. Get PnP Trajectory
    if Path(args.pnp_cache).exists():
        print(f"[PnP] Loading cached PnP trajectory from {args.pnp_cache}")
        df_pnp = pd.read_csv(args.pnp_cache)
    else:
        df_pnp = extract_pnp_trajectory(args.video, args.video_ts, K, dist, marker_pos)
        df_pnp.to_csv(args.pnp_cache, index=False)

    # 2. Load GT Pose
    print(f"[GT] Loading Ground Truth poses from {args.gt_csv}")
    df_gt = pd.read_csv(args.gt_csv)

    # 3. Solve Alignment
    calib_dict, df_aligned = solve_spatial_temporal_alignment(df_pnp, df_gt)

    # 4. Save Outputs
    Path(args.out_calib).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_calib, "w") as f:
        json.dump(calib_dict, f, indent=2)
    print(f"[Output] Saved alignment calibration to {args.out_calib}")

    df_aligned.to_csv(args.out_csv, index=False)
    print(f"[Output] Saved aligned comparison dataset to {args.out_csv}")

    if args.plot:
        plot_alignment_results(df_aligned, args.plot_out)


if __name__ == "__main__":
    main()
