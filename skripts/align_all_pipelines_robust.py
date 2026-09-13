#!/usr/bin/env python3
"""align_all_pipelines_robust.py

Comprehensive 12-DoF Two-Sided Hand-Eye + 2D Optical Reprojection Calibration
with Temporal Smoothing for CNN and ML 6-DoF tracking pipelines.
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

S_LHS_TO_RHS = np.diag([1.0, -1.0, 1.0])


def unity_to_rhs_pose(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts Unity Left-Handed coordinates to Right-Handed standard frame."""
    pos_rhs = pos @ S_LHS_TO_RHS
    R_lhs = R_scipy.from_quat(quat).as_matrix()
    R_rhs = S_LHS_TO_RHS @ R_lhs @ S_LHS_TO_RHS
    if np.linalg.det(R_rhs) < 0:
        R_rhs = -R_rhs
    return pos_rhs, R_rhs


def detect_blobs(gray: np.ndarray, thresh: int = 220, min_area: float = 2.0, max_area: float = 1500.0):
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
):
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


def extract_pnp_trajectory(video_path: str, timestamps_path: str, K: np.ndarray, dist: np.ndarray, marker_pos: dict):
    df_ts = pd.read_csv(timestamps_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[PnP] Extracting poses from {video_path} ({total_frames} frames)...")
    
    marker_ids = sorted(marker_pos.keys())
    perms = list(itertools.permutations(marker_ids))

    times, valids, tvecs, quats, errs, num_blobs = [], [], [], [], [], []
    frame_idx = 0
    prev_perm = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        t_sec = df_ts["pts_ms"].iloc[frame_idx] / 1000.0 if frame_idx < len(df_ts) else frame_idx / 30.0
        times.append(t_sec)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts, areas = detect_blobs(gray)
        num_blobs.append(len(pts))
        
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
                pose_found = True

        if not pose_found:
            prev_perm = None
            tvecs.append(np.array([np.nan, np.nan, np.nan]))
            quats.append(np.array([np.nan, np.nan, np.nan, np.nan]))
            errs.append(np.nan)
            valids.append(False)

        frame_idx += 1

    cap.release()
    tvecs = np.array(tvecs)
    quats = np.array(quats)
    
    return pd.DataFrame({
        "frame_idx": np.arange(len(times)),
        "time_s": times,
        "valid": valids,
        "num_blobs": num_blobs,
        "tx_mm": tvecs[:, 0],
        "ty_mm": tvecs[:, 1],
        "tz_mm": tvecs[:, 2],
        "qx": quats[:, 0],
        "qy": quats[:, 1],
        "qz": quats[:, 2],
        "qw": quats[:, 3],
        "reproj_err_px": errs
    })


class CNN6DoFKalmanFilter:
    """6-DoF Causal State Estimator (Kalman Filter for Position + SO(3) Geodesic Filter for Orientation)."""

    def __init__(self, pos_noise_mm: float = 15.0, accel_noise_mm: float = 500.0, rot_tau_s: float = 0.06) -> None:
        self.pos_noise = pos_noise_mm
        self.accel_noise = accel_noise_mm
        self.rot_tau = rot_tau_s
        self.reset()

    def reset(self) -> None:
        self.x = None  # [px, py, pz, vx, vy, vz]
        self.P = None
        self.rotation = None  # 3x3 matrix
        self.last_t = None

    def step(self, t: float, p_meas: np.ndarray, q_meas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filters incoming raw pose measurement (p_meas in mm, q_meas as [qx, qy, qz, qw])."""
        R_meas = R_scipy.from_quat(q_meas).as_matrix()
        if self.x is None or self.last_t is None:
            self.x = np.concatenate([p_meas, [0.0, 0.0, 0.0]])
            self.P = np.diag([self.pos_noise**2] * 3 + [500.0**2] * 3)
            self.rotation = R_meas.copy()
            self.last_t = t
            return p_meas.copy(), q_meas.copy()

        dt = max(1e-4, t - self.last_t)
        self.last_t = t

        # 1. Kinematic constant-velocity prediction
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        G = np.vstack([np.eye(3) * (0.5 * dt * dt), np.eye(3) * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + (self.accel_noise**2) * (G @ G.T)

        # 2. Measurement update for position
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        y = p_meas - self.x[:3]
        R_cov = np.eye(3) * (self.pos_noise**2)
        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        # 3. SO(3) Geodesic SLERP orientation filter
        alpha = 1.0 - np.exp(-dt / self.rot_tau)
        R_rel = self.rotation.T @ R_meas
        rotvec = cv2.Rodrigues(R_rel)[0].ravel()
        self.rotation = self.rotation @ cv2.Rodrigues(alpha * rotvec)[0]
        q_filtered = R_scipy.from_matrix(self.rotation).as_quat()
        if np.dot(q_filtered, q_meas) < 0:
            q_filtered = -q_filtered

        return self.x[:3].copy(), q_filtered


def apply_kalman_filter(
    df: pd.DataFrame,
    pos_noise_mm: float = 15.0,
    accel_noise_mm: float = 500.0,
    rot_tau_s: float = 0.06
) -> pd.DataFrame:
    """Applies causal 6-DoF Kalman filtering to reduce high-frequency prediction jitter."""
    if df is None or len(df) == 0:
        return df

    df_f = df.copy()
    kf = CNN6DoFKalmanFilter(pos_noise_mm=pos_noise_mm, accel_noise_mm=accel_noise_mm, rot_tau_s=rot_tau_s)

    pos_cols = [c for c in ["tx_mm", "ty_mm", "tz_mm"] if c in df_f.columns]
    quat_cols = [c for c in ["qx", "qy", "qz", "qw"] if c in df_f.columns]
    time_col = "time_s" if "time_s" in df_f.columns else None

    if len(pos_cols) == 3 and len(quat_cols) == 4:
        p_filtered = []
        q_filtered = []
        for i in range(len(df_f)):
            t = df_f[time_col].iloc[i] if time_col else float(i) / 30.0
            p = df_f[pos_cols].iloc[i].values
            q = df_f[quat_cols].iloc[i].values
            if not np.isfinite(p).all() or not np.isfinite(q).all():
                p_filtered.append(p)
                q_filtered.append(q)
                continue
            pf, qf = kf.step(t, p, q)
            p_filtered.append(pf)
            q_filtered.append(qf)
        df_f[pos_cols] = np.array(p_filtered)
        df_f[quat_cols] = np.array(q_filtered)

    return df_f


def solve_gt_and_pnp_alignment(df_pnp: pd.DataFrame, df_gt: pd.DataFrame, init_dt: float = 18.97):
    print("[Alignment] Solving GT Hand-Eye Calibration on 1000us PnP trajectory...")
    t_gt = df_gt["timestamp_seconds"].values
    gt_pos_raw = df_gt[["right_px", "right_py", "right_pz"]].values
    gt_quat_raw = df_gt[["right_qx", "right_qy", "right_qz", "right_qw"]].values

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

    clean_pnp_mask = df_pnp["valid"].values & (df_pnp["reproj_err_px"].values < 0.4)
    t_pnp = df_pnp["time_s"].values
    pnp_p_m = df_pnp[["tx_mm", "ty_mm", "tz_mm"]].values / 1000.0
    pnp_q = df_pnp[["qx", "qy", "qz", "qw"]].values

    def eval_at_dt(dt: float, params: np.ndarray):
        t_targets = t_pnp + dt
        mask = clean_pnp_mask & (t_targets >= t_gt[0] + 0.5) & (t_targets <= t_gt[-1] - 0.5)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return indices, np.zeros((0, 3)), np.zeros((0, 3)), None, None, None, None
        t_sync = t_targets[indices]

        gt_p_sync = np.zeros((len(indices), 3))
        for d in range(3):
            gt_p_sync[:, d] = np.interp(t_sync, t_gt, gt_pos_rhs[:, d])
        gt_q_sync = np.zeros((len(indices), 4))
        for d in range(4):
            gt_q_sync[:, d] = np.interp(t_sync, t_gt, gt_quats_rhs[:, d])
        gt_q_sync /= np.linalg.norm(gt_q_sync, axis=1, keepdims=True)
        gt_R_sync = R_scipy.from_quat(gt_q_sync).as_matrix()

        p_tr = pnp_p_m[indices]
        R_tr = R_scipy.from_quat(pnp_q[indices]).as_matrix()

        rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = params
        R_X = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
        t_X = np.array([tx, ty, tz])
        R_Y = R_scipy.from_rotvec([ox, oy, oz]).as_matrix()
        t_Y = np.array([bx, by, bz])

        # R_pred = R_X * R_gt * R_Y
        # p_pred = R_X * (R_gt * t_Y + p_gt) + t_X
        R_pred = np.einsum("ij,njk,kl->nil", R_X, gt_R_sync, R_Y)
        p_pred = np.einsum("ij,nj->ni", R_X, np.einsum("nij,j->ni", gt_R_sync, t_Y) + gt_p_sync) + t_X

        pos_err = p_tr - p_pred
        R_diff = np.einsum("nji,njk->nik", R_tr, R_pred)
        rot_err = R_scipy.from_matrix(R_diff).as_rotvec()

        return indices, pos_err, rot_err, p_tr, p_pred, R_tr, R_pred

    p0 = [-0.35, -0.02, -0.06, 1.80, -0.07, 1.47, -1.44, 1.05, -1.45, 0.023, -0.007, -0.0015]
    best_cost = 1e9
    best_dt = init_dt

    for dt_cand in np.arange(init_dt - 0.4, init_dt + 0.4, 0.02):
        ind, pos_e, rot_e, _, _, _, _ = eval_at_dt(dt_cand, p0)
        if len(ind) > 100:
            c = np.median(np.linalg.norm(pos_e, axis=1)) + np.median(np.linalg.norm(rot_e, axis=1))
            if c < best_cost:
                best_cost = c
                best_dt = dt_cand

    def loss(p):
        ind, pos_e, rot_e, _, _, _, _ = eval_at_dt(best_dt, p)
        sub = slice(0, len(ind), 3)
        return np.concatenate([rot_e[sub].ravel(), (pos_e[sub] * 3.0).ravel()])

    res = least_squares(loss, p0, loss="soft_l1", f_scale=0.1, method="trf")
    best_opt = res.x

    # Fine search around best_dt
    for dt_cand in np.arange(best_dt - 0.04, best_dt + 0.04, 0.002):
        ind, pos_e, rot_e, _, _, _, _ = eval_at_dt(dt_cand, best_opt)
        c = np.median(np.linalg.norm(pos_e, axis=1)) + np.median(np.linalg.norm(rot_e, axis=1))
        if c < best_cost:
            best_cost = c
            best_dt = dt_cand

    indices, pos_err, rot_err, p_pnp_sync, p_pred, R_pnp_sync, R_pred = eval_at_dt(best_dt, best_opt)
    pos_errors_mm = np.linalg.norm(pos_err, axis=1) * 1000.0
    rot_errors_deg = np.degrees(np.linalg.norm(rot_err, axis=1))

    rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = best_opt
    R_X = R_scipy.from_rotvec([rx, ry, rz])
    t_X = np.array([tx, ty, tz])
    R_Y = R_scipy.from_rotvec([ox, oy, oz])
    t_Y = np.array([bx, by, bz])

    # Find the BEST aligned anchor frames (reproj < 0.35 px, residual error < 20mm, < 5 deg)
    best_anchor_mask = (pos_errors_mm < 20.0) & (rot_errors_deg < 5.0)
    best_frame_indices = indices[best_anchor_mask]

    print(f"[GT-PnP Alignment] Done! dt = {best_dt:.4f} s, Total Clean Frames = {len(indices)}, Best Anchor Frames = {len(best_frame_indices)}")
    print(f"  Camera in VR (X): Pos = {t_X} m, Euler = {R_X.as_euler('xyz', degrees=True)} deg")
    print(f"  Controller in Bar (Y): Pos = {t_Y*1000} mm, Euler = {R_Y.as_euler('xyz', degrees=True)} deg")
    print(f"  Anchor Frames Median Pos Error: {np.median(pos_errors_mm[best_anchor_mask]):.2f} mm")
    print(f"  Anchor Frames Median Rot Error: {np.median(rot_errors_deg[best_anchor_mask]):.2f} deg")

    calib_pnp = {
        "time_offset_seconds": float(best_dt),
        "camera_to_vr_extrinsics": {
            "translation_m": t_X.tolist(),
            "rotation_rotvec": [float(rx), float(ry), float(rz)],
            "rotation_euler_deg": R_X.as_euler("xyz", degrees=True).tolist(),
            "rotation_matrix": R_X.as_matrix().tolist()
        },
        "controller_to_bar_extrinsics": {
            "translation_m": t_Y.tolist(),
            "translation_mm": (t_Y * 1000.0).tolist(),
            "rotation_rotvec": [float(ox), float(oy), float(oz)],
            "rotation_euler_deg": R_Y.as_euler("xyz", degrees=True).tolist(),
            "rotation_matrix": R_Y.as_matrix().tolist()
        },
        "metrics": {
            "clean_frames": int(len(indices)),
            "anchor_frames": int(len(best_frame_indices)),
            "anchor_median_pos_error_mm": float(np.median(pos_errors_mm[best_anchor_mask])),
            "anchor_median_rot_error_deg": float(np.median(rot_errors_deg[best_anchor_mask]))
        }
    }

    return calib_pnp, best_frame_indices, df_pnp


def optimize_model_12dof_alignment(
    model_name: str,
    df_model_raw: pd.DataFrame,
    df_pnp: pd.DataFrame,
    anchor_indices: np.ndarray,
    calib_pnp: dict,
    K: np.ndarray
) -> tuple[dict, pd.DataFrame]:
    """Performs 12-DoF Two-Sided Hand-Eye + 2D Optical Image Refinement with Temporal Smoothing."""
    print(f"\n[{model_name}] Performing 12-DoF Hand-Eye + 2D Optical Alignment with Smoothing on {len(anchor_indices)} anchor frames...")

    # Apply 6-DoF Kalman filtering to raw model trajectory
    df_model = apply_kalman_filter(df_model_raw)

    p_true_cam = df_pnp.loc[anchor_indices, ["tx_mm", "ty_mm", "tz_mm"]].values / 1000.0
    q_true_cam = df_pnp.loc[anchor_indices, ["qx", "qy", "qz", "qw"]].values
    R_true_cam = R_scipy.from_quat(q_true_cam).as_matrix()

    p_model_raw = df_model.loc[anchor_indices, ["tx_mm", "ty_mm", "tz_mm"]].values / 1000.0
    q_model_raw = df_model.loc[anchor_indices, ["qx", "qy", "qz", "qw"]].values
    R_model_raw = R_scipy.from_quat(q_model_raw).as_matrix()

    # Step 1: Initial 3D Hand-Eye solve (finding X and Y)
    def loss_3d(params):
        rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = params
        R_X = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
        t_X = np.array([tx, ty, tz])
        R_Y = R_scipy.from_rotvec([ox, oy, oz]).as_matrix()
        t_Y = np.array([bx, by, bz])

        # p_pred = R_X * (R_model * t_Y + p_model) + t_X
        # R_pred = R_X * R_model * R_Y
        p_pred = np.einsum("ij,nj->ni", R_X, np.einsum("nij,j->ni", R_model_raw, t_Y) + p_model_raw) + t_X
        R_pred = np.einsum("ij,njk,kl->nil", R_X, R_model_raw, R_Y)

        pos_err = (p_true_cam - p_pred) * 3.0
        R_diff = np.einsum("nji,njk->nik", R_true_cam, R_pred)
        rot_err = R_scipy.from_matrix(R_diff).as_rotvec()

        return np.concatenate([pos_err.ravel(), rot_err.ravel()])

    best_cost = 1e9
    best_p0 = None
    for rx in [-np.pi/2, 0, np.pi/2, np.pi]:
        for ry in [-np.pi/2, 0, np.pi/2, np.pi]:
            for rz in [-np.pi/2, 0, np.pi/2, np.pi]:
                p0_cand = [rx, ry, rz, 1.8, -0.07, 1.5, -np.pi/2, 0, -np.pi/2, 0.03, 0, 0]
                c = np.sum(loss_3d(p0_cand)**2)
                if c < best_cost:
                    best_cost = c
                    best_p0 = p0_cand

    res_3d = least_squares(loss_3d, best_p0, loss="soft_l1", f_scale=0.1, method="trf")

    # Step 2: Refine with 2D Optical Image Projection + 3D loss
    u_pnp = K[0, 0] * p_true_cam[:, 0] / p_true_cam[:, 2] + K[0, 2]
    v_pnp = K[1, 1] * p_true_cam[:, 1] / p_true_cam[:, 2] + K[1, 2]
    pts_2d_true = np.stack([u_pnp, v_pnp], axis=-1)

    def loss_refine(params):
        rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = params
        R_X = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
        t_X = np.array([tx, ty, tz])
        R_Y = R_scipy.from_rotvec([ox, oy, oz]).as_matrix()
        t_Y = np.array([bx, by, bz])

        p_pred = np.einsum("ij,nj->ni", R_X, np.einsum("nij,j->ni", R_model_raw, t_Y) + p_model_raw) + t_X
        R_pred = np.einsum("ij,njk,kl->nil", R_X, R_model_raw, R_Y)

        pos_err = (p_true_cam - p_pred) * 2.0
        R_diff = np.einsum("nji,njk->nik", R_true_cam, R_pred)
        rot_err = R_scipy.from_matrix(R_diff).as_rotvec()

        z_safe = np.clip(p_pred[:, 2], 0.1, 10.0)
        u_pred = K[0, 0] * p_pred[:, 0] / z_safe + K[0, 2]
        v_pred = K[1, 1] * p_pred[:, 1] / z_safe + K[1, 2]
        proj_2d = np.stack([u_pred, v_pred], axis=-1)
        err_2d = (proj_2d - pts_2d_true) * 0.02

        return np.concatenate([err_2d.ravel(), pos_err.ravel(), rot_err.ravel()])

    res_ref = least_squares(loss_refine, res_3d.x, loss="soft_l1", f_scale=0.1, method="trf")

    rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = res_ref.x
    R_X = R_scipy.from_rotvec([rx, ry, rz])
    t_X = np.array([tx, ty, tz])
    R_Y = R_scipy.from_rotvec([ox, oy, oz])
    t_Y = np.array([bx, by, bz])

    # Transform all smoothed model predictions to Camera space
    p_all_raw = df_model[["tx_mm", "ty_mm", "tz_mm"]].values / 1000.0
    q_all_raw = df_model[["qx", "qy", "qz", "qw"]].values
    R_all_raw = R_scipy.from_quat(q_all_raw).as_matrix()

    p_all_cam = np.einsum("ij,nj->ni", R_X.as_matrix(), np.einsum("nij,j->ni", R_all_raw, t_Y) + p_all_raw) + t_X
    R_all_cam = np.einsum("ij,njk,kl->nil", R_X.as_matrix(), R_all_raw, R_Y.as_matrix())
    q_all_cam = R_scipy.from_matrix(R_all_cam).as_quat()

    # Evaluate accuracy on anchor frames
    p_pred_anchor = p_all_cam[anchor_indices]
    R_pred_anchor = R_all_cam[anchor_indices]

    pos_err_mm = np.linalg.norm(p_true_cam - p_pred_anchor, axis=1) * 1000.0
    rot_err_deg = np.degrees(np.linalg.norm(R_scipy.from_matrix(np.einsum("nji,njk->nik", R_true_cam, R_pred_anchor)).as_rotvec(), axis=1))

    z_safe_anchor = np.clip(p_pred_anchor[:, 2], 0.1, 10.0)
    u_pred_anchor = K[0, 0] * p_pred_anchor[:, 0] / z_safe_anchor + K[0, 2]
    v_pred_anchor = K[1, 1] * p_pred_anchor[:, 1] / z_safe_anchor + K[1, 2]
    err_2d_px = np.linalg.norm(np.stack([u_pred_anchor, v_pred_anchor], axis=-1) - pts_2d_true, axis=1)

    print(f"[{model_name}] Calibrated Parameters:")
    print(f"  Camera in VR (X): Pos = {t_X} m, Euler = {R_X.as_euler('xyz', degrees=True)} deg")
    print(f"  Controller in Bar (Y): Pos = {t_Y*1000} mm, Euler = {R_Y.as_euler('xyz', degrees=True)} deg")
    print(f"  Anchor 2D Image Error: Median = {np.median(err_2d_px):.2f} px")
    print(f"  Anchor 3D Position Error: Median = {np.median(pos_err_mm):.2f} mm")
    print(f"  Anchor 3D Rotation Error: Median = {np.median(rot_err_deg):.2f} deg")

    calib_dict = {
        "model": model_name,
        "time_offset_seconds": calib_pnp["time_offset_seconds"],
        "camera_to_vr_extrinsics": {
            "translation_m": t_X.tolist(),
            "rotation_rotvec": [float(rx), float(ry), float(rz)],
            "rotation_euler_deg": R_X.as_euler("xyz", degrees=True).tolist(),
            "rotation_matrix": R_X.as_matrix().tolist()
        },
        "controller_to_bar_extrinsics": {
            "translation_m": t_Y.tolist(),
            "translation_mm": (t_Y * 1000.0).tolist(),
            "rotation_rotvec": [float(ox), float(oy), float(oz)],
            "rotation_euler_deg": R_Y.as_euler("xyz", degrees=True).tolist(),
            "rotation_matrix": R_Y.as_matrix().tolist()
        },
        "metrics_on_anchors": {
            "num_anchor_frames": int(len(anchor_indices)),
            "median_2d_error_px": float(np.median(err_2d_px)),
            "median_pos_error_mm": float(np.median(pos_err_mm)),
            "mean_pos_error_mm": float(np.mean(pos_err_mm)),
            "rmse_pos_error_mm": float(np.sqrt(np.mean(pos_err_mm**2))),
            "median_rot_error_deg": float(np.median(rot_err_deg)),
            "mean_rot_error_deg": float(np.mean(rot_err_deg)),
            "rmse_rot_error_deg": float(np.sqrt(np.mean(rot_err_deg**2)))
        }
    }

    df_transformed = pd.DataFrame({
        "frame_idx": df_model["frame_idx"],
        "time_s": df_model["time_s"],
        "valid": df_model["valid"] if "valid" in df_model.columns else True,
        "cam_tx_mm": p_all_cam[:, 0] * 1000.0,
        "cam_ty_mm": p_all_cam[:, 1] * 1000.0,
        "cam_tz_mm": p_all_cam[:, 2] * 1000.0,
        "cam_qx": q_all_cam[:, 0],
        "cam_qy": q_all_cam[:, 1],
        "cam_qz": q_all_cam[:, 2],
        "cam_qw": q_all_cam[:, 3]
    })

    return calib_dict, df_transformed


def main():
    video_path = str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_video_20260911_170556_290626.mkv")
    ts_path = str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_timestamps_20260911_170556_290626.csv")
    gt_csv = str(ROOT / "data" / "GT_pos" / "20260911_130547_284_P01" / "pose_before_render.csv")
    calib_yaml = str(ROOT / "data" / "camera_calibration.yaml")
    geom_yaml = str(ROOT / "data" / "geometry.yaml")

    cal = yaml.safe_load(open(calib_yaml))
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    K[:2, :] *= 0.5
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

    geom = yaml.safe_load(open(geom_yaml))
    marker_pos = {m["id"]: np.array(m["position"], dtype=np.float64) for m in geom["markers"]}

    # 1. PnP Trajectory on 1000us Video
    pnp_cache = ROOT / "data" / "pnp_1000us_trajectory.csv"
    if pnp_cache.exists():
        df_pnp = pd.read_csv(pnp_cache)
    else:
        df_pnp = extract_pnp_trajectory(video_path, ts_path, K, dist, marker_pos)
        df_pnp.to_csv(pnp_cache, index=False)

    df_gt = pd.read_csv(gt_csv)

    # 2. Solve PnP <-> GT Alignment & identify best anchor frames
    calib_pnp, anchor_indices, df_pnp = solve_gt_and_pnp_alignment(df_pnp, df_gt, init_dt=18.97)
    with open(ROOT / "data" / "alignment_calibration_1000us.json", "w") as f:
        json.dump(calib_pnp, f, indent=2)

    # 3. Optimize CNN with 12-DoF Hand-Eye + 2D Optical Image Refinement
    df_cnn_raw = pd.read_csv(ROOT / "data" / "cnn_trajectory_extracted.csv")
    calib_cnn, df_cnn_cam = optimize_model_12dof_alignment(
        "CNN_Temporal3", df_cnn_raw, df_pnp, anchor_indices, calib_pnp, K
    )
    with open(ROOT / "data" / "cnn_alignment_calibration.json", "w") as f:
        json.dump(calib_cnn, f, indent=2)
    df_cnn_cam.to_csv(ROOT / "data" / "cnn_cam_trajectory.csv", index=False)

    # 4. Optimize ML with 12-DoF Hand-Eye + 2D Optical Image Refinement
    df_ml_raw = pd.read_csv(ROOT / "data" / "ml_trajectory_extracted.csv")
    calib_ml, df_ml_cam = optimize_model_12dof_alignment(
        "ML_Pi_Inference", df_ml_raw, df_pnp, anchor_indices, calib_pnp, K
    )
    with open(ROOT / "data" / "ml_alignment_calibration.json", "w") as f:
        json.dump(calib_ml, f, indent=2)
    df_ml_cam.to_csv(ROOT / "data" / "ml_cam_trajectory.csv", index=False)

    print("\n[Complete] All pipelines robustly calibrated and smoothed on 1000us video!")


if __name__ == "__main__":
    main()
