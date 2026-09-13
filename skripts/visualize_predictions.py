#!/usr/bin/env python3
"""visualize_predictions.py

Generates synchronized benchmark video sequences and multi-stage pipeline comparisons
for data-driven models (CNN Temporal-3, ML Pi-Streaming Feature Regressor, and Ground Truth).

Sequential Pipeline Stages:
  A. CNN Pipeline (Direct Deep Neural Pose Estimation):
     1. 01_bright_raw.mp4 / 02_bright_gt.mp4     : Ambient visual stream (10,000 µs).
     2. 03_dark_raw.mp4 / 04_dark_gt.mp4         : High-contrast dark IR stream (1,000 µs).
     3. 05_cnn_raw.mp4 / 06_cnn_gt.mp4           : Raw direct CNN 6-DoF pose predictions (with neural jitter) & violet trail.
     4. 07_cnn_kf_raw.mp4 / 08_cnn_kf_gt.mp4     : Causal 6-DoF Kalman filtered CNN pose & trail.

  B. ML Pipeline (Learned 2D-to-3D Feature Regressor):
     1. 01_bright_raw.mp4 / 02_bright_gt.mp4     : Ambient visual stream (10,000 µs).
     2. 03_dark_raw.mp4 / 04_dark_gt.mp4         : High-contrast dark IR stream (1,000 µs).
     3. 05_dark_filtration.mp4 / 06_dark_filtration_gt.mp4 : Filtered optical stream (isolated LEDs, zero background noise).
     4. 07_dark_blobs.mp4 / 08_dark_blobs_gt.mp4 : Filtered stream + 2D candidate & tracking blob rings.
     5. 09_ml_raw.mp4 / 10_ml_gt.mp4             : Raw ML feature regressor 6-DoF pose & amber gold trail.
     6. 11_ml_kf_raw.mp4 / 12_ml_kf_gt.mp4       : Causal 6-DoF Kalman filtered ML pose & gold trail.

Color Conventions (Visibly Distinct):
  - Ground Truth (VR Controller) : Bright Green / Yellow / Orange Axes | Bright Green Trail (0, 255, 0)
  - CNN Raw                      : Orchid / Mint / Violet Axes   | Lavender Violet Trail (230, 80, 180)
  - CNN + Kalman Filter          : Electric Violet Axes          | Bright Indigo Trail (255, 100, 140)
  - ML Raw                       : Coral / Chartreuse / Gold     | Amber Gold Trail (50, 180, 255)
  - ML + Kalman Filter           : Tangerine / SpringGreen / Gold| Canary Yellow Trail (0, 215, 255)
  - Classical PnP (Reference)    : Red / Green / Blue Axes       | Electric Cyan Trail (255, 220, 50)
  - Classical EKF (Reference)    : Red / Green / SkyBlue Axes    | Aqua Emerald Trail (220, 255, 0)

Outputs are placed into a newly generated timestamped directory:
  data/renders/data-driven/renders_YYYYMMDD_HHMMSS/
along with comprehensive Markdown documentation (README.md).

Usage:
  # Render all data-driven pipeline stages (CNN & ML):
  python skripts/visualize_predictions.py --pipeline all --start 2131 --count 200

  # Render only CNN pipeline:
  python skripts/visualize_predictions.py --pipeline cnn --start 2131 --count 200

  # Render only ML pipeline:
  python skripts/visualize_predictions.py --pipeline ml --start 2131 --count 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R_scipy

ROOT = Path(__file__).resolve().parent.parent

# Coordinate parity matrix: Unity Left-Handed to Standard Right-Handed standard frame
S_LHS_TO_RHS = np.diag([1.0, -1.0, 1.0])


def unity_to_rhs_pose(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts Unity Left-Handed coordinates to Right-Handed standard frame."""
    pos_rhs = pos @ S_LHS_TO_RHS
    R_lhs = R_scipy.from_quat(quat).as_matrix()
    R_rhs = S_LHS_TO_RHS @ R_lhs @ S_LHS_TO_RHS
    if np.linalg.det(R_rhs) < 0:
        R_rhs = -R_rhs
    return pos_rhs, R_rhs


def draw_3d_axes_clean(
    img: np.ndarray,
    R_mat: np.ndarray,
    tvec_mm: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    axis_length_mm: float = 80.0,
    thickness: int = 2,
    colors: tuple[tuple[int, int, int], ...] = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
):
    """Draws clean 3D coordinate frame axes (X, Y, Z) without text labels."""
    rvec, _ = cv2.Rodrigues(R_mat)
    origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    axes_3d = np.array([
        [axis_length_mm, 0.0, 0.0],
        [0.0, axis_length_mm, 0.0],
        [0.0, 0.0, axis_length_mm]
    ], dtype=np.float64)

    proj_origin, _ = cv2.projectPoints(origin_3d, rvec, tvec_mm, K, dist)
    proj_axes, _ = cv2.projectPoints(axes_3d, rvec, tvec_mm, K, dist)

    o_pt = tuple(proj_origin.reshape(-1, 2)[0].astype(int))
    h, w = img.shape[:2]

    for axis_pt, color in zip(proj_axes.reshape(-1, 2), colors):
        a_pt = tuple(axis_pt.astype(int))
        if -200 <= o_pt[0] < w + 200 and -200 <= o_pt[1] < h + 200:
            cv2.line(img, o_pt, a_pt, color, thickness, cv2.LINE_AA)


def draw_trail_legend(
    vis: np.ndarray,
    items: list[tuple[str, tuple[int, int, int]]],
    margin_right: int = 12,
    margin_bottom: int = 12
) -> None:
    """Draws a clean, semi-transparent legend overlay on the bottom-right corner."""
    if not items:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    font_thickness = 1
    line_h = 16
    pad_x = 8
    pad_y = 6
    indicator_len = 14
    spacing = 6

    max_text_w = 0
    for label, _ in items:
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        if tw > max_text_w:
            max_text_w = tw

    box_w = pad_x * 2 + indicator_len + spacing + max_text_w
    box_h = pad_y * 2 + len(items) * line_h

    h, w = vis.shape[:2]
    x2 = w - margin_right
    y2 = h - margin_bottom
    x1 = x2 - box_w
    y1 = y2 - box_h

    if x1 < 0 or y1 < 0:
        return

    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 23, 42), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 80, 95), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

    for idx, (label, col) in enumerate(items):
        item_y = y1 + pad_y + idx * line_h + line_h // 2
        line_start_x = x1 + pad_x
        line_end_x = line_start_x + indicator_len
        cv2.line(vis, (line_start_x, item_y), (line_end_x, item_y), col, 2, cv2.LINE_AA)
        cv2.circle(vis, ((line_start_x + line_end_x) // 2, item_y), 3, col, -1, cv2.LINE_AA)
        text_x = line_end_x + spacing
        text_y = item_y + 4
        cv2.putText(vis, label, (text_x, text_y), font, font_scale, (240, 240, 240), font_thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Pre-filtration & Optical Blob Detection Pipeline
# ---------------------------------------------------------------------------

def filter_and_detect_blobs(
    gray: np.ndarray,
    is_dark: bool = True,
    min_area: float = 2.0,
    max_area: float = 1500.0,
    min_circularity: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applies multi-stage filtration BEFORE blob detection to leave ONLY bright LED markers."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)

    max_val = float(blurred.max())
    if max_val < 40:
        return np.empty((0, 2)), np.empty((0,)), np.zeros_like(gray), np.zeros_like(gray)

    thresh_val = max(180, int(0.75 * max_val)) if is_dark else 240
    _, bw = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bw_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []
    bw_blobs_only = np.zeros_like(bw_clean)

    for c in contours:
        M = cv2.moments(c)
        area = M["m00"]
        if not (min_area < area < max_area):
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
            if circularity < min_circularity:
                continue

        cv2.drawContours(bw_blobs_only, [c], -1, 255, -1)

        cx = M["m10"] / area
        cy = M["m01"] / area
        pts.append((cx, cy))
        areas.append(area)

    return np.array(pts), np.array(areas), bw_clean, bw_blobs_only


# ---------------------------------------------------------------------------
# 6-DoF Causal State Estimator (Kalman Filter + SO(3) Geodesic Filter)
# ---------------------------------------------------------------------------

class Pose6DoFKalmanFilter:
    """6-DoF Causal State Estimator: Kinematic CV Position Filter + SO(3) Geodesic Filter for Orientation."""

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


def apply_kalman_filter_to_trajectory(
    df: pd.DataFrame,
    pos_noise_mm: float = 15.0,
    accel_noise_mm: float = 500.0,
    rot_tau_s: float = 0.06
) -> pd.DataFrame:
    """Applies causal 6-DoF Kalman filtering to reduce high-frequency prediction jitter."""
    if df is None or len(df) == 0:
        return df

    df_f = df.copy()
    kf = Pose6DoFKalmanFilter(pos_noise_mm=pos_noise_mm, accel_noise_mm=accel_noise_mm, rot_tau_s=rot_tau_s)

    pos_cols = [c for c in ["tx_mm", "ty_mm", "tz_mm"] if c in df_f.columns]
    quat_cols = [c for c in ["qx", "qy", "qz", "qw"] if c in df_f.columns]
    time_col = "time_s" if "time_s" in df_f.columns else None

    if len(pos_cols) == 3 and len(quat_cols) == 4:
        p_filtered = []
        q_filtered = []
        for i in range(len(df_f)):
            t = df_f[time_col].iloc[i] if time_col else float(i) / 30.0
            p = df_f[pos_cols].iloc[i].values.astype(np.float64)
            q = df_f[quat_cols].iloc[i].values.astype(np.float64)
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


def transform_model_to_cam(
    df_model: pd.DataFrame,
    R_X: np.ndarray,
    t_X: np.ndarray,
    R_Y: np.ndarray,
    t_Y: np.ndarray
) -> pd.DataFrame:
    """Transforms model space trajectory to Camera frame coordinates [cam_tx_mm, cam_ty_mm, cam_tz_mm, cam_qx, cam_qy, cam_qz, cam_qw]."""
    df_out = df_model.copy()
    p_raw = df_model[["tx_mm", "ty_mm", "tz_mm"]].values.astype(np.float64) / 1000.0  # meters
    q_raw = df_model[["qx", "qy", "qz", "qw"]].values.astype(np.float64)
    R_raw = R_scipy.from_quat(q_raw).as_matrix()

    # p_cam = R_X * (R_raw * t_Y + p_raw) + t_X
    # R_cam = R_X * R_raw * R_Y
    p_cam = np.einsum("ij,nj->ni", R_X, np.einsum("nij,j->ni", R_raw, t_Y) + p_raw) + t_X
    R_cam = np.einsum("ij,njk,kl->nil", R_X, R_raw, R_Y)
    q_cam = R_scipy.from_matrix(R_cam).as_quat()

    df_out["cam_tx_mm"] = p_cam[:, 0] * 1000.0
    df_out["cam_ty_mm"] = p_cam[:, 1] * 1000.0
    df_out["cam_tz_mm"] = p_cam[:, 2] * 1000.0
    df_out["cam_qx"] = q_cam[:, 0]
    df_out["cam_qy"] = q_cam[:, 1]
    df_out["cam_qz"] = q_cam[:, 2]
    df_out["cam_qw"] = q_cam[:, 3]
    return df_out


# ---------------------------------------------------------------------------
# Video Variant Rendering Core
# ---------------------------------------------------------------------------

# Color Palette Config (Visibly distinct in BGR)
COLOR_PALETTE = {
    "gt": {
        "axes": ((0, 255, 0), (50, 255, 255), (0, 165, 255)),  # Bright Green X, Yellow Y, Orange Z
        "trail": (0, 255, 0),                                    # Bright Green
        "dot": (0, 255, 0)
    },
    "cnn_raw": {
        "axes": ((180, 50, 240), (100, 240, 50), (240, 50, 180)),  # Orchid X, Mint Y, Violet Z
        "trail": (230, 80, 180),                                    # Soft Lavender Violet
        "dot": (230, 80, 180)
    },
    "cnn_kf": {
        "axes": ((220, 30, 180), (130, 255, 80), (255, 120, 200)), # Deep Violet X, Bright Mint Y, Lavender Z
        "trail": (255, 100, 140),                                   # Electric Indigo
        "dot": (255, 100, 140)
    },
    "ml_raw": {
        "axes": ((50, 160, 255), (150, 255, 50), (255, 180, 50)),   # Coral X, Chartreuse Y, Amber Gold Z
        "trail": (50, 180, 255),                                    # Amber Gold
        "dot": (50, 180, 255)
    },
    "ml_kf": {
        "axes": ((0, 130, 255), (80, 255, 160), (0, 215, 255)),    # Tangerine X, SpringGreen Y, Pure Gold Z
        "trail": (0, 215, 255),                                     # Canary Yellow Gold
        "dot": (0, 215, 255)
    }
}


def render_video_variant(
    stage_id: str,
    output_filename: str,
    stream_type: str,              # 'dark' or 'bright'
    use_filtered_stream: bool,      # True: optical background subtraction mask
    show_blobs: bool,               # True: 2D detected blob markers
    model_type: str | None,         # 'cnn_raw', 'cnn_kf', 'ml_raw', 'ml_kf', or None
    df_model_cam: pd.DataFrame | None, # Pre-transformed Camera Space Trajectory
    show_gt: bool,                  # True: VR Ground Truth overlay
    show_trails: bool,
    trail_length: int,
    trail_thickness: int,
    video_path: str,
    timestamps_path: str,
    gt_pos_rhs: np.ndarray | None,
    gt_quats_rhs: np.ndarray | None,
    t_gt_all: np.ndarray | None,
    dt_sync_gt: float,
    R_X_gt: np.ndarray,
    t_X_gt: np.ndarray,
    R_Y_gt: np.ndarray,
    t_Y_gt: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    start_frame: int = 0,
    count_frames: int | None = None
) -> dict:
    """Renders a single video variant with the specified visual layers and web-ready encoding."""
    df_ts = pd.read_csv(timestamps_path) if Path(timestamps_path).exists() else None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    total_in_frames = len(df_ts) if df_ts is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Real-time frame rate calculation
    if df_ts is not None and len(df_ts) > 1:
        total_time_s = (df_ts['pts_ms'].iloc[-1] - df_ts['pts_ms'].iloc[0]) / 1000.0
        calculated_fps = (len(df_ts) - 1) / total_time_s if total_time_s > 0 else 30.0
    else:
        calculated_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    end_frame = total_in_frames if count_frames is None else min(start_frame + count_frames, total_in_frames)
    frames_to_process = end_frame - start_frame

    print(f"\n[Render {stage_id}] -> {Path(output_filename).name}")
    print(f"  Stream: {stream_type.upper()} ({width}x{height} @ {calculated_fps:.2f} fps) | Frames: {start_frame}..{end_frame-1} ({frames_to_process} frames)")
    print(f"  Layers: FilteredBase={use_filtered_stream}, Blobs={show_blobs}, Model={model_type}, GT={show_gt}, Trails={show_trails}")

    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, calculated_fps, (width, height))

    if start_frame > 0:
        for _ in range(start_frame):
            cap.grab()

    model_trail = []
    gt_trail = []
    is_dark = (stream_type == 'dark')
    t0 = time.time()

    legend_items = []
    if show_trails:
        if model_type == "cnn_raw":
            legend_items.append(("CNN (Raw)", COLOR_PALETTE["cnn_raw"]["trail"]))
        elif model_type == "cnn_kf":
            legend_items.append(("CNN + KF", COLOR_PALETTE["cnn_kf"]["trail"]))
        elif model_type == "ml_raw":
            legend_items.append(("ML (Raw)", COLOR_PALETTE["ml_raw"]["trail"]))
        elif model_type == "ml_kf":
            legend_items.append(("ML + KF", COLOR_PALETTE["ml_kf"]["trail"]))
        if show_gt:
            legend_items.append(("Ground Truth", COLOR_PALETTE["gt"]["trail"]))

    for idx, f_idx in enumerate(range(start_frame, end_frame)):
        ret, frame = cap.read()
        if not ret:
            break

        t_video = df_ts['pts_ms'].iloc[f_idx] / 1000.0 if (df_ts is not None and f_idx < len(df_ts)) else f_idx / calculated_fps
        t_gt = t_video + dt_sync_gt

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_pts, areas, bw_clean, bw_blobs = filter_and_detect_blobs(gray, is_dark=is_dark)

        # 1. Base visual layer: Filtered optical stream vs raw footage
        if use_filtered_stream:
            filtered_spots = cv2.bitwise_and(frame, frame, mask=bw_blobs)
            mask_glow = cv2.cvtColor(bw_blobs, cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(filtered_spots, 0.75, mask_glow, 0.25, 0)
        else:
            vis = frame.copy()

        # Identify top-4 blobs for tracking
        blob_idx = set()
        if len(detected_pts) >= 4:
            top4 = np.argsort(areas)[::-1][:4]
            blob_idx = set(top4)

        # 2. 2D Blob Detection Layer (Candidate Cyan rings + Top-4 Red tracking rings)
        if show_blobs:
            for i, (cx, cy) in enumerate(detected_pts):
                if i not in blob_idx:
                    px, py = int(round(cx)), int(round(cy))
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(vis, (px, py), 7, (255, 200, 0), 2, cv2.LINE_AA)
                        cv2.circle(vis, (px, py), 2, (0, 255, 150), -1, cv2.LINE_AA)

            if blob_idx:
                for i in blob_idx:
                    cx, cy = detected_pts[i]
                    px, py = int(round(cx)), int(round(cy))
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(vis, (px, py), 8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.circle(vis, (px, py), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # 3. Model 6-DoF Pose Layer
        if model_type is not None and df_model_cam is not None and f_idx < len(df_model_cam):
            is_valid = bool(df_model_cam['valid'].iloc[f_idx]) if 'valid' in df_model_cam.columns else True
            if is_valid:
                row = df_model_cam.iloc[f_idx]
                tvec_mm = np.array([[row['cam_tx_mm']], [row['cam_ty_mm']], [row['cam_tz_mm']]], dtype=np.float64)
                q_mod = np.array([row['cam_qx'], row['cam_qy'], row['cam_qz'], row['cam_qw']], dtype=np.float64)
                R_mod_cam = R_scipy.from_quat(q_mod).as_matrix()

                p_cfg = COLOR_PALETTE.get(model_type, COLOR_PALETTE["cnn_raw"])
                draw_3d_axes_clean(
                    vis, R_mod_cam, tvec_mm, K, dist,
                    axis_length_mm=80.0, thickness=2,
                    colors=p_cfg["axes"]
                )

                if show_trails:
                    rvec_mod, _ = cv2.Rodrigues(R_mod_cam)
                    proj_mod_origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec_mod, tvec_mm, K, dist)
                    pt = tuple(proj_mod_origin.reshape(-1, 2)[0].astype(int))
                    model_trail.append(pt)
            else:
                if show_trails:
                    model_trail.append(None)

        # 4. Ground Truth 3D Pose Layer (Using accurate camera-to-VR calibration)
        if show_gt and gt_pos_rhs is not None and t_gt_all is not None:
            gt_in_bounds = (t_gt >= t_gt_all[0]) and (t_gt <= t_gt_all[-1])
            if gt_in_bounds:
                p_gt_vr = np.array([np.interp(t_gt, t_gt_all, gt_pos_rhs[:, d]) for d in range(3)])
                q_gt_vr = np.array([np.interp(t_gt, t_gt_all, gt_quats_rhs[:, d]) for d in range(4)])
                q_gt_vr /= np.linalg.norm(q_gt_vr)
                R_gt_vr = R_scipy.from_quat(q_gt_vr).as_matrix()

                # T_cam_to_bar = X * T_vr_to_ctrl * Y
                R_gt_cam = R_X_gt @ R_gt_vr @ R_Y_gt
                p_gt_cam = R_X_gt @ (R_gt_vr @ t_Y_gt + p_gt_vr) + t_X_gt  # meters
                tvec_gt_cam_mm = (p_gt_cam * 1000.0).reshape(3, 1)

                draw_3d_axes_clean(
                    vis, R_gt_cam, tvec_gt_cam_mm, K, dist,
                    axis_length_mm=75.0, thickness=2,
                    colors=COLOR_PALETTE["gt"]["axes"]
                )

                if show_trails:
                    rvec_gt_cam, _ = cv2.Rodrigues(R_gt_cam)
                    proj_gt_origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec_gt_cam, tvec_gt_cam_mm, K, dist)
                    gt_pt = tuple(proj_gt_origin.reshape(-1, 2)[0].astype(int))
                    gt_trail.append(gt_pt)
            else:
                if show_trails:
                    gt_trail.append(None)

        # 5. Fading 3D Trajectory Trails
        if show_trails:
            if len(model_trail) > trail_length:
                model_trail.pop(0)
            if len(gt_trail) > trail_length:
                gt_trail.pop(0)

            # Model Trail (Violet for CNN, Amber Gold for ML)
            if model_type is not None:
                p_cfg = COLOR_PALETTE.get(model_type, COLOR_PALETTE["cnn_raw"])
                col_base = p_cfg["trail"]
                for tr_i in range(1, len(model_trail)):
                    alpha = 0.30 + 0.70 * (tr_i / float(trail_length))
                    p1, p2 = model_trail[tr_i - 1], model_trail[tr_i]
                    if p1 is not None and p2 is not None:
                        if 0 <= p1[0] < width and 0 <= p1[1] < height and 0 <= p2[0] < width and 0 <= p2[1] < height:
                            c = (int(col_base[0] * alpha), int(col_base[1] * alpha), int(col_base[2] * alpha))
                            cv2.line(vis, p1, p2, c, trail_thickness, cv2.LINE_AA)
                            if tr_i == len(model_trail) - 1:
                                cv2.circle(vis, p2, 4, col_base, -1, cv2.LINE_AA)

            # GT Trail (Magenta)
            col_gt = COLOR_PALETTE["gt"]["trail"]
            for tr_i in range(1, len(gt_trail)):
                alpha = 0.30 + 0.70 * (tr_i / float(trail_length))
                g1, g2 = gt_trail[tr_i - 1], gt_trail[tr_i]
                if g1 is not None and g2 is not None:
                    if 0 <= g1[0] < width and 0 <= g1[1] < height and 0 <= g2[0] < width and 0 <= g2[1] < height:
                        c = (int(col_gt[0] * alpha), int(col_gt[1] * alpha), int(col_gt[2] * alpha))
                        cv2.line(vis, g1, g2, c, trail_thickness, cv2.LINE_AA)
                        if tr_i == len(gt_trail) - 1:
                            cv2.circle(vis, g2, 4, col_gt, -1, cv2.LINE_AA)

            if legend_items:
                draw_trail_legend(vis, legend_items)

        writer.write(vis)

        if (idx + 1) % 500 == 0 or idx == frames_to_process - 1:
            elapsed = time.time() - t0
            print(f"    Processed {idx + 1}/{frames_to_process} frames ({elapsed:.1f}s, {(idx + 1)/elapsed:.1f} fps)...")

    cap.release()
    writer.release()

    # Convert to web-compatible H.264 (yuv420p + faststart)
    temp_h264 = output_filename.replace(".mp4", "_temp_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", output_filename,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        temp_h264
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(temp_h264):
            time.sleep(0.05)
            os.remove(output_filename)
            os.rename(temp_h264, output_filename)
            print(f"    [Web Codec] Converted {Path(output_filename).name} to H.264 (yuv420p)")
    except Exception as e:
        print(f"    [Warning] Failed to re-encode {Path(output_filename).name} to H.264: {e}")

    return {
        "filename": Path(output_filename).name,
        "stage": stage_id,
        "stream": stream_type,
        "layers": {
            "filtered_base": use_filtered_stream,
            "blobs": show_blobs,
            "model": model_type,
            "gt": show_gt,
            "trails": show_trails
        },
        "frames": frames_to_process,
        "fps": calculated_fps,
        "duration_s": frames_to_process / calculated_fps
    }


# ---------------------------------------------------------------------------
# Documentation & Metadata Generator
# ---------------------------------------------------------------------------

def generate_metadata_markdown(
    out_dir: Path,
    timestamp_str: str,
    pipeline_mode: str,
    calib_gt: dict,
    video_records: list[dict],
    start_frame: int,
    count_frames: int | None,
    show_trails: bool,
    trail_length: int,
    kf_settings: dict
):
    """Generates the comprehensive README.md documentation file inside the timestamped directory."""
    dt_sync = calib_gt["time_offset_seconds"]
    cam_trans = calib_gt["camera_to_vr_extrinsics"]["translation_m"]
    cam_euler = calib_gt["camera_to_vr_extrinsics"]["rotation_euler_deg"]
    ctrl_trans_mm = calib_gt["controller_to_bar_extrinsics"]["translation_mm"]
    ctrl_euler = calib_gt["controller_to_bar_extrinsics"]["rotation_euler_deg"]

    md_content = f"""# Data-Driven 6-DoF Tracking Render Dataset

**Generation Timestamp**: `{timestamp_str}`  
**Pipeline Mode**: `{pipeline_mode.upper()}`  
**Output Directory**: `{out_dir.resolve()}`  

---

## 1. Video Specifications & Data-Driven Pipeline Breakdown

* **Start Frame**: `{start_frame}`
* **Frame Count**: `{count_frames if count_frames is not None else 'All Frames'}`
* **Trails Enabled**: `{show_trails}` (Length: `{trail_length}` frames)
* **Text / HUD Overlay**: `Disabled (Clean Academic Visuals)`

### Calibration & Alignment Parameters
* **Time Synchronization Offset ($\\Delta t$)**: `+{dt_sync:.4f} s` ($t_{{\\text{{GT}}}} = t_{{\\text{{video}}}} + {dt_sync:.4f}\\text{{ s}}$)
* **Camera Extrinsics in VR World ($T_{{\\text{{cam}}\\to\\text{{vr}}}}$)**:
  * Translation: `[{cam_trans[0]:.4f}, {cam_trans[1]:.4f}, {cam_trans[2]:.4f}] m`
  * Euler Angles (XYZ): `[{cam_euler[0]:.2f}°, {cam_euler[1]:.2f}°, {cam_euler[2]:.2f}°]`
* **Controller-to-Bar Offset ($T_{{\\text{{ctrl}}\\to\\text{{bar}}}}$)**:
  * Translation: `[{ctrl_trans_mm[0]:.2f}, {ctrl_trans_mm[1]:.2f}, {ctrl_trans_mm[2]:.2f}] mm`
  * Euler Angles (XYZ): `[{ctrl_euler[0]:.2f}°, {ctrl_euler[1]:.2f}°, {ctrl_euler[2]:.2f}°]`

### Pipeline Architectures

#### A. CNN Pipeline (Direct Deep Neural Pose Estimation)
1. **Ambient Visual Context (10,000 µs)**: 10,000 µs full-frame ambient illumination capturing the tracking environment.
2. **Short-Shutter Tensor Input (1,000 µs)**: 1,000 µs high-contrast dark IR frame passed directly to convolutional layers.
3. **CNN Direct 6-DoF Pose**: Temporal-3 ResNet regressing rigid body pose $[\\mathbf{{t}}, \\mathbf{{q}}]$ directly without explicit 2D blob extraction (demonstrates raw neural inference jitter).
4. **CNN + Kalman Filter**: Causal 6-DoF state estimator (kinematic CV position + $\\mathrm{{SO}}(3)$ geodesic orientation filter) suppressing neural jitter.

#### B. ML Pipeline (Learned 2D-to-3D Feature Regressor)
1. **Ambient Visual Context (10,000 µs)**: 10,000 µs ambient visual stream.
2. **Short-Shutter Tensor Input (1,000 µs)**: 1,000 µs short-shutter frame.
3. **Optical Pre-Filtration**: Multi-stage $3\\times 3$ Gaussian smoothing, dynamic peak threshold ($T \\ge 180$), and morphological opening.
4. **2D Blob Extraction**: Sub-pixel moments extracting 2D centroid coordinates and spatial features.
5. **ML Regressor Output**: Trained Random Forest / MLP feature regression predicting 6-DoF pose (demonstrates raw regressor predictions).
6. **ML + Kalman Filter**: 6-DoF causal state estimator smoothing regression output.

### 6-DoF Kalman Filter Hyperparameters
* **Position Measurement Noise ($\\sigma_{{\\text{{pos}}}}$)**: `{kf_settings['pos_noise']} mm`
* **Acceleration Process Noise ($\\sigma_{{\\text{{accel}}}}$)**: `{kf_settings['accel_noise']} mm/s²`
* **Orientation Time Constant ($\\tau_{{\\text{{rot}}}}$)**: `{kf_settings['rot_tau']} s`

---

## 2. Generated Video Files Matrix

| Filename | Shutter / Stream | Pipeline Stage & Visual Layers Included | Duration | Frame Count |
| :--- | :--- | :--- | :--- | :--- |
"""
    for v in video_records:
        layers = v["layers"]
        desc_parts = []
        if layers["filtered_base"] and not layers["blobs"] and not layers["model"] and not layers["gt"]:
            desc_parts.append("Filtered Stream (Isolated LEDs, No Markers)")
        elif layers["filtered_base"] and not layers["blobs"] and not layers["model"] and layers["gt"]:
            desc_parts.append("Filtered Stream (No Markers) + VR Ground Truth 3D Pose")
        elif layers["blobs"]:
            desc_parts.append("Filtered Stream + 2D Detected Blobs (Cyan Candidates & Red Tracking)")
            if layers["gt"]:
                desc_parts.append("VR Ground Truth 3D Pose")
        elif layers["model"] == "cnn_raw":
            desc_parts.append("Dark Video + Raw CNN Direct 6-DoF Pose (Orchid Axes & Violet Trail)")
            if layers["gt"]:
                desc_parts.append("VR Ground Truth 3D Pose")
        elif layers["model"] == "cnn_kf":
            desc_parts.append("Dark Video + Kalman Filtered CNN 6-DoF Pose (Deep Violet Axes & Indigo Trail)")
            if layers["gt"]:
                desc_parts.append("VR Ground Truth 3D Pose")
        elif layers["model"] == "ml_raw":
            desc_parts.append("Filtered Backdrop + Raw ML Feature Regressor 6-DoF Pose (Coral Axes & Amber Trail)")
            if layers["gt"]:
                desc_parts.append("VR Ground Truth 3D Pose")
        elif layers["model"] == "ml_kf":
            desc_parts.append("Filtered Backdrop + Kalman Filtered ML 6-DoF Pose (Tangerine Axes & Canary Gold Trail)")
            if layers["gt"]:
                desc_parts.append("VR Ground Truth 3D Pose")
        elif layers["gt"]:
            desc_parts.append("VR Ground Truth 3D Pose (Bright Green/Yellow/Orange Axes & Trail)")
        else:
            desc_parts.append("Raw Footage (No Overlay)")

        desc_str = " + ".join(desc_parts)
        shutter_str = "1000 µs (Dark IR)" if v["stream"] == "dark" else "10000 µs (Bright Visual)"
        md_content += f"| `{v['filename']}` | {shutter_str} | {desc_str} | {v['duration_s']:.2f} s | {v['frames']} frames |\n"

    md_content += """
---

## 3. Visual Layer & Distinct Color Conventions

* **Ground Truth (VR Controller)**: Bright Green X, Yellow Y, Orange Z Axes (`RGB: 0, 255, 0`) | Vibrant Bright Green Trail (`RGB: 0, 255, 0`).
* **CNN Raw (Direct Neural)**: Orchid X, Mint Y, Violet Z Axes | Soft Lavender Violet Trail (`RGB: 180, 80, 230`).
* **CNN + Kalman Filter**: Electric Violet X, Mint Y, Lavender Z Axes | Bright Electric Indigo Trail (`RGB: 140, 100, 255`).
* **ML Raw (Feature Regressor)**: Coral X, Chartreuse Y, Amber Gold Z Axes | Warm Amber Gold Trail (`RGB: 255, 180, 50`).
* **ML + Kalman Filter**: Tangerine X, SpringGreen Y, Pure Gold Z Axes | Vibrant Canary Yellow Trail (`RGB: 255, 215, 0`).
* **Filtered Optical Stream**: Ambient noise suppressed, leaving only true optical LED emissions with soft clean glow.
* **Cyan Circles**: Sub-pixel detected 2D candidate LED centroids.
* **Red Circles**: The 4 optical LED blobs selected for tracking.
"""

    readme_path = out_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[Settings] Saved settings and video descriptions to {readme_path}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render Data-Driven 6-DoF Tracking Video Sequences (CNN & ML).")
    parser.add_argument("--pipeline", type=str, default="all", choices=["cnn", "ml", "all"], help="Pipeline(s) to render: 'cnn', 'ml', or 'all' (default: all)")
    parser.add_argument("--dark-video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--dark-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--bright-video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--bright-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--gt-csv", type=str, default=str(ROOT / "data" / "GT_pos" / "20260911_130547_284_P01" / "pose_before_render.csv"))
    parser.add_argument("--gt-calib", type=str, default=str(ROOT / "data" / "alignment_calibration.json"), help="Exact camera-to-VR ground truth calibration JSON")
    parser.add_argument("--calib", type=str, default=str(ROOT / "data" / "camera_calibration.yaml"))
    parser.add_argument("--cnn-calib", type=str, default=str(ROOT / "data" / "cnn_alignment_calibration.json"))
    parser.add_argument("--ml-calib", type=str, default=str(ROOT / "data" / "ml_alignment_calibration.json"))
    parser.add_argument("--cnn-raw-csv", type=str, default=str(ROOT / "data" / "cnn_trajectory_extracted.csv"), help="Raw un-smoothed CNN inference CSV")
    parser.add_argument("--ml-raw-csv", type=str, default=str(ROOT / "data" / "ml_trajectory_extracted.csv"), help="Raw un-smoothed ML inference CSV")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output folder path (defaults to data/renders/data-driven/renders_YYYYMMDD_HHMMSS/)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--count", type=int, default=None, help="Number of frames to render")
    parser.add_argument("--no-trails", action="store_true", help="Disable fading 3D trajectory trails")
    parser.add_argument("--trail-length", type=int, default=35, help="Length of fading trajectory trail in frames")
    parser.add_argument("--trail-thickness", type=int, default=3, help="Line thickness for 3D trajectory trails")
    parser.add_argument("--pos-noise", type=float, default=15.0, help="Kalman filter position measurement noise in mm")
    parser.add_argument("--accel-noise", type=float, default=500.0, help="Kalman filter process acceleration noise in mm/s^2")
    parser.add_argument("--rot-tau", type=float, default=0.06, help="Kalman filter rotation time constant in seconds")

    args = parser.parse_args()

    # 1. Initialize output directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ROOT / "data" / "renders" / "data-driven" / f"renders_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Init] Output directory initialized at: {out_dir.resolve()}")

    # 2. Load Camera Intrinsics
    cal = yaml.safe_load(open(args.calib))
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    K[:2, :] *= 0.5  # 1280x800 -> 640x400
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

    # 3. Load Ground Truth Calibration (exact same as render_aligned_video.py)
    with open(args.gt_calib) as f:
        calib_gt = json.load(f)
    dt_sync_gt = calib_gt["time_offset_seconds"]
    R_X_gt = np.array(calib_gt["camera_to_vr_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_X_gt = np.array(calib_gt["camera_to_vr_extrinsics"]["translation_m"], dtype=np.float64)
    R_Y_gt = np.array(calib_gt["controller_to_bar_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_Y_gt = np.array(calib_gt["controller_to_bar_extrinsics"]["translation_m"], dtype=np.float64)

    # 4. Load CNN Model Extrinsics (from cnn_alignment_calibration.json)
    with open(args.cnn_calib) as f:
        calib_cnn = json.load(f)
    R_X_cnn = np.array(calib_cnn["camera_to_vr_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_X_cnn = np.array(calib_cnn["camera_to_vr_extrinsics"]["translation_m"], dtype=np.float64)
    R_Y_cnn = np.array(calib_cnn["controller_to_bar_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_Y_cnn = np.array(calib_cnn["controller_to_bar_extrinsics"]["translation_m"], dtype=np.float64)

    # 5. Load ML Model Extrinsics (from ml_alignment_calibration.json)
    with open(args.ml_calib) as f:
        calib_ml = json.load(f)
    R_X_ml = np.array(calib_ml["camera_to_vr_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_X_ml = np.array(calib_ml["camera_to_vr_extrinsics"]["translation_m"], dtype=np.float64)
    R_Y_ml = np.array(calib_ml["controller_to_bar_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_Y_ml = np.array(calib_ml["controller_to_bar_extrinsics"]["translation_m"], dtype=np.float64)

    # 6. Load Ground Truth Data
    df_gt = pd.read_csv(args.gt_csv)
    t_gt_all = df_gt['timestamp_seconds'].values
    gt_pos_raw = df_gt[['right_px', 'right_py', 'right_pz']].values
    gt_quat_raw = df_gt[['right_qx', 'right_qy', 'right_qz', 'right_qw']].values

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

    # 7. Process CNN Trajectories (Raw vs Kalman Filtered)
    df_cnn_raw_cam = None
    df_cnn_kf_cam = None
    if Path(args.cnn_raw_csv).exists():
        df_cnn_raw_source = pd.read_csv(args.cnn_raw_csv)
        # Transform raw un-smoothed predictions
        df_cnn_raw_cam = transform_model_to_cam(df_cnn_raw_source, R_X_cnn, t_X_cnn, R_Y_cnn, t_Y_cnn)
        # Apply 6-DoF Kalman filter and then transform
        df_cnn_kf_source = apply_kalman_filter_to_trajectory(
            df_cnn_raw_source, pos_noise_mm=args.pos_noise, accel_noise_mm=args.accel_noise, rot_tau_s=args.rot_tau
        )
        df_cnn_kf_cam = transform_model_to_cam(df_cnn_kf_source, R_X_cnn, t_X_cnn, R_Y_cnn, t_Y_cnn)

    # 8. Process ML Trajectories (Raw vs Kalman Filtered)
    df_ml_raw_cam = None
    df_ml_kf_cam = None
    if Path(args.ml_raw_csv).exists():
        df_ml_raw_source = pd.read_csv(args.ml_raw_csv)
        # Transform raw un-smoothed predictions
        df_ml_raw_cam = transform_model_to_cam(df_ml_raw_source, R_X_ml, t_X_ml, R_Y_ml, t_Y_ml)
        # Apply 6-DoF Kalman filter and then transform
        df_ml_kf_source = apply_kalman_filter_to_trajectory(
            df_ml_raw_source, pos_noise_mm=args.pos_noise, accel_noise_mm=args.accel_noise, rot_tau_s=args.rot_tau
        )
        df_ml_kf_cam = transform_model_to_cam(df_ml_kf_source, R_X_ml, t_X_ml, R_Y_ml, t_Y_ml)

    show_trails = not args.no_trails
    video_records = []
    render_cnn = args.pipeline in ["cnn", "all"]
    render_ml = args.pipeline in ["ml", "all"]

    # -----------------------------------------------------------------------
    # Base Videos (Shared): Bright Raw/GT and Dark Raw/GT
    # -----------------------------------------------------------------------
    # 1. Bright Video Raw
    rec = render_video_variant(
        stage_id="Bright-Raw",
        output_filename=str(out_dir / "01_bright_raw.mp4"),
        stream_type="bright",
        use_filtered_stream=False, show_blobs=False, model_type=None, df_model_cam=None, show_gt=False,
        show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.bright_video, timestamps_path=args.bright_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
        K=K, dist=dist, start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 2. Bright Video + GT
    rec = render_video_variant(
        stage_id="Bright-GT",
        output_filename=str(out_dir / "02_bright_gt.mp4"),
        stream_type="bright",
        use_filtered_stream=False, show_blobs=False, model_type=None, df_model_cam=None, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.bright_video, timestamps_path=args.bright_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
        K=K, dist=dist, start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 3. Dark Video Raw
    rec = render_video_variant(
        stage_id="Dark-Raw",
        output_filename=str(out_dir / "03_dark_raw.mp4"),
        stream_type="dark",
        use_filtered_stream=False, show_blobs=False, model_type=None, df_model_cam=None, show_gt=False,
        show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
        K=K, dist=dist, start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 4. Dark Video + GT
    rec = render_video_variant(
        stage_id="Dark-GT",
        output_filename=str(out_dir / "04_dark_gt.mp4"),
        stream_type="dark",
        use_filtered_stream=False, show_blobs=False, model_type=None, df_model_cam=None, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
        K=K, dist=dist, start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # -----------------------------------------------------------------------
    # CNN Pipeline Specific Stages
    # -----------------------------------------------------------------------
    if render_cnn:
        # 5. CNN Raw Predictions (Raw neural inference with natural jitter)
        rec = render_video_variant(
            stage_id="CNN-Raw",
            output_filename=str(out_dir / "05_cnn_raw.mp4"),
            stream_type="dark",
            use_filtered_stream=False, show_blobs=False, model_type="cnn_raw", df_model_cam=df_cnn_raw_cam, show_gt=False,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 6. CNN Raw + GT
        rec = render_video_variant(
            stage_id="CNN-Raw-GT",
            output_filename=str(out_dir / "06_cnn_gt.mp4"),
            stream_type="dark",
            use_filtered_stream=False, show_blobs=False, model_type="cnn_raw", df_model_cam=df_cnn_raw_cam, show_gt=True,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 7. CNN + Kalman Filter (Smooth causal state estimator)
        rec = render_video_variant(
            stage_id="CNN-KF-Raw",
            output_filename=str(out_dir / "07_cnn_kf_raw.mp4"),
            stream_type="dark",
            use_filtered_stream=False, show_blobs=False, model_type="cnn_kf", df_model_cam=df_cnn_kf_cam, show_gt=False,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 8. CNN + Kalman Filter + GT
        rec = render_video_variant(
            stage_id="CNN-KF-GT",
            output_filename=str(out_dir / "08_cnn_kf_gt.mp4"),
            stream_type="dark",
            use_filtered_stream=False, show_blobs=False, model_type="cnn_kf", df_model_cam=df_cnn_kf_cam, show_gt=True,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

    # -----------------------------------------------------------------------
    # ML Pipeline Specific Stages
    # -----------------------------------------------------------------------
    if render_ml:
        # 5b. Optical Filtration Raw
        rec = render_video_variant(
            stage_id="ML-Filtration-Raw",
            output_filename=str(out_dir / "05_dark_filtration.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=False, model_type=None, df_model_cam=None, show_gt=False,
            show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 6b. Optical Filtration + GT
        rec = render_video_variant(
            stage_id="ML-Filtration-GT",
            output_filename=str(out_dir / "06_dark_filtration_gt.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=False, model_type=None, df_model_cam=None, show_gt=True,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 7b. Blob Detection Raw
        rec = render_video_variant(
            stage_id="ML-Blobs-Raw",
            output_filename=str(out_dir / "07_dark_blobs.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=True, model_type=None, df_model_cam=None, show_gt=False,
            show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 8b. Blob Detection + GT
        rec = render_video_variant(
            stage_id="ML-Blobs-GT",
            output_filename=str(out_dir / "08_dark_blobs_gt.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=True, model_type=None, df_model_cam=None, show_gt=True,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 9b. ML Predictions Raw (Raw feature regression inferences)
        rec = render_video_variant(
            stage_id="ML-Raw",
            output_filename=str(out_dir / "09_ml_raw.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=False, model_type="ml_raw", df_model_cam=df_ml_raw_cam, show_gt=False,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 10b. ML Predictions + GT
        rec = render_video_variant(
            stage_id="ML-GT",
            output_filename=str(out_dir / "10_ml_gt.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=False, model_type="ml_raw", df_model_cam=df_ml_raw_cam, show_gt=True,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 11b. ML + Kalman Filter Raw (Smooth causal state estimator)
        rec = render_video_variant(
            stage_id="ML-KF-Raw",
            output_filename=str(out_dir / "11_ml_kf_raw.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=False, model_type="ml_kf", df_model_cam=df_ml_kf_cam, show_gt=False,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

        # 12b. ML + Kalman Filter + GT
        rec = render_video_variant(
            stage_id="ML-KF-GT",
            output_filename=str(out_dir / "12_ml_kf_gt.mp4"),
            stream_type="dark",
            use_filtered_stream=True, show_blobs=False, model_type="ml_kf", df_model_cam=df_ml_kf_cam, show_gt=True,
            show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
            video_path=args.dark_video, timestamps_path=args.dark_ts,
            gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
            dt_sync_gt=dt_sync_gt, R_X_gt=R_X_gt, t_X_gt=t_X_gt, R_Y_gt=R_Y_gt, t_Y_gt=t_Y_gt,
            K=K, dist=dist, start_frame=args.start, count_frames=args.count
        )
        video_records.append(rec)

    # Generate Markdown Documentation File
    generate_metadata_markdown(
        out_dir=out_dir,
        timestamp_str=timestamp_str,
        pipeline_mode=args.pipeline,
        calib_gt=calib_gt,
        video_records=video_records,
        start_frame=args.start,
        count_frames=args.count,
        show_trails=show_trails,
        trail_length=args.trail_length,
        kf_settings={
            "pos_noise": args.pos_noise,
            "accel_noise": args.accel_noise,
            "rot_tau": args.rot_tau
        }
    )

    print(f"\n=======================================================")
    print(f"Data-Driven videos ({len(video_records)} files) successfully saved to:")
    print(f"  {out_dir.resolve()}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
