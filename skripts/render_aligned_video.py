a#!/usr/bin/env python3
"""render_aligned_video.py

Generates synchronized benchmark video combinations without on-screen text overlays,
demonstrating the complete tracking pipeline step-by-step:
  1.  01_dark_raw.mp4            : Dark video (1000us shutter), raw footage.
  2.  02_dark_gt.mp4             : Dark video + Ground Truth 3D pose.
  3.  03_dark_filtration.mp4     : Filtered optical video (noise/ambient removed, pure LEDs, no markers).
  4.  04_dark_filtration_gt.mp4  : Filtered optical video + Ground Truth 3D pose.
  5.  05_dark_blobs.mp4          : Filtered video + 2D detected optical blob markers.
  6.  06_dark_blobs_gt.mp4       : Filtered video + 2D detected blobs + Ground Truth 3D pose.
  7.  07_dark_pnp.mp4            : Filtered video + Raw PnP 3D pose & trajectory trail.
  8.  08_dark_pnp_gt.mp4         : Filtered video + Raw PnP 3D pose + Ground Truth 3D pose.
  9.  09_dark_ekf.mp4            : Filtered video + 13-State EKF Smoothed 3D pose & trail.
  10. 10_dark_ekf_gt.mp4         : Filtered video + EKF Smoothed 3D pose + Ground Truth 3D pose.
  11. 11_bright_raw.mp4          : Bright video (10000us shutter), raw footage.
  12. 12_bright_gt.mp4           : Bright video + Ground Truth 3D pose.

Outputs are placed into a newly generated timestamped directory:
  data/renders/renders_YYYYMMDD_HHMMSS/
along with a comprehensive settings & descriptions Markdown file (README.md).

Usage:
    # Generate all 12 video combinations with default tuned EKF:
    python skripts/render_aligned_video.py --start 2131 --count 200

    # Adjust EKF smoothing (higher R = heavier smoothing, higher Q = more agile tracking):
    python skripts/render_aligned_video.py --start 2131 --count 200 --ekf-r-scale 5.0 --ekf-q-scale 2.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R_scipy
from scipy.stats import chi2

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


def draw_3d_axes_clean(
    img: np.ndarray,
    R_mat: np.ndarray,
    tvec: np.ndarray,
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

    proj_origin, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist)
    proj_axes, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)

    o_pt = tuple(proj_origin.reshape(-1, 2)[0].astype(int))
    h, w = img.shape[:2]

    for axis_pt, color in zip(proj_axes.reshape(-1, 2), colors):
        a_pt = tuple(axis_pt.astype(int))
        if 0 <= o_pt[0] < w and 0 <= o_pt[1] < h and -200 <= a_pt[0] < w + 200 and -200 <= a_pt[1] < h + 200:
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
    """Applies multi-stage filtration BEFORE blob detection to leave ONLY bright LED markers.
    
    Filtration Stages:
      1. Gaussian Smoothing Filter: Suppresses sensor shot noise and single-pixel anomalies.
      2. Dynamic Peak Thresholding: Isolates the top intensity peak percentile (saturated LEDs).
      3. Morphological Opening Filter: Eradicates isolated specks, hot pixels, and background noise.
      4. Geometric & Circularity Filter: Eliminates elongated streaks, scratches, and diffuse glare.
      5. Sub-pixel Blob Centroid Extraction via Spatial Moments.
    """
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
# 13-State Extended Kalman Filter (EKF)
# ---------------------------------------------------------------------------

def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication for [w, x, y, z] representation."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_omega(w: np.ndarray, dt: float) -> np.ndarray:
    """Computes incremental rotation quaternion from angular velocity vector."""
    theta = np.linalg.norm(w) * dt
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = w / np.linalg.norm(w)
    return np.array([np.cos(theta / 2.0), *(axis * np.sin(theta / 2.0))])


def f_state(x: np.ndarray, dt: float) -> np.ndarray:
    """Constant-velocity / constant-angular-velocity process model."""
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    p2 = p + v * dt
    q2 = quat_mult(q, quat_from_omega(w, dt))
    q2 = q2 / np.linalg.norm(q2)
    return np.concatenate([p2, v, q2, w])


def h_state(x: np.ndarray) -> np.ndarray:
    """Measurement model: direct position + orientation observation."""
    return np.concatenate([x[0:3], x[6:10]])


def numeric_jacobian(fn, x: np.ndarray, *args, eps: float = 1e-6) -> np.ndarray:
    """Computes numeric Jacobian matrix."""
    fx = fn(x, *args)
    J = np.zeros((len(fx), len(x)))
    for k in range(len(x)):
        dx = np.zeros(len(x))
        dx[k] = eps
        J[:, k] = (fn(x + dx, *args) - fx) / eps
    return J


class PoseEKF:
    """13-state Extended Kalman Filter: position(3), velocity(3), quaternion(4), angular velocity(3)."""

    def __init__(
        self,
        process_noise: np.ndarray,
        measurement_noise: np.ndarray,
        dt_ref: float,
        disable_gating: bool = True,
        nis_gate_p: float = 0.9999,
        max_consecutive_rejections: int = 10
    ):
        self.Q = process_noise
        self.dt_ref = dt_ref
        self.R = measurement_noise
        self.disable_gating = disable_gating
        self.nis_threshold = chi2.ppf(nis_gate_p, df=7)
        self.max_consecutive_rejections = max_consecutive_rejections
        self.x = None
        self.P = None
        self.last_nis = None
        self.n_rejected = 0
        self.n_updated = 0
        self.n_reinitialized = 0
        self._consecutive_rejections = 0

    def initialize(self, pos: np.ndarray, quat: np.ndarray):
        self.x = np.concatenate([pos, [0.0, 0.0, 0.0], quat, [0.0, 0.0, 0.0]])
        self.P = np.diag([1, 1, 1, 1e6, 1e6, 1e6, 0.01, 0.01, 0.01, 0.01, 100, 100, 100])
        self._consecutive_rejections = 0

    def predict(self, dt: float):
        F = numeric_jacobian(f_state, self.x, dt)
        self.x = f_state(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q * (dt / self.dt_ref)

    def update(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        z = np.concatenate([pos, quat])
        H = numeric_jacobian(h_state, self.x)
        y = z - h_state(self.x)
        if y[3:] @ self.x[6:10] < 0:
            z = z.copy()
            z[3:] = -z[3:]
            y = z - h_state(self.x)

        S = H @ self.P @ H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis

        force_recover = self._consecutive_rejections >= self.max_consecutive_rejections
        if not self.disable_gating and nis > self.nis_threshold and not force_recover:
            self.n_rejected += 1
            self._consecutive_rejections += 1
            return False

        if force_recover:
            self.initialize(z[:3], z[3:])
            self.n_reinitialized += 1
            return True

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6:10] /= np.linalg.norm(self.x[6:10])
        self.P = (np.eye(13) - K @ H) @ self.P
        self.n_updated += 1
        self._consecutive_rejections = 0
        return True

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def quaternion(self) -> np.ndarray:
        return self.x[6:10]


def robust_std(x: np.ndarray, axis=None) -> np.ndarray:
    """MAD-based scale estimate robust against measurement spikes."""
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def estimate_measurement_noise(
    positions: np.ndarray,
    quats: np.ndarray,
    r_scale: float = 1.0,
    r_pos_override: float | None = None,
    r_quat_override: float | None = None
) -> np.ndarray:
    """Estimates empirical measurement covariance matrix R."""
    if r_pos_override is not None:
        pos_std = np.full(3, r_pos_override)
    else:
        med_pos = np.array([robust_std(positions[:, k] - median_filter(positions[:, k], size=5)) for k in range(3)])
        pos_std = np.maximum(med_pos, 3.0)  # 3mm baseline noise floor

    if r_quat_override is not None:
        quat_std = np.full(4, r_quat_override)
    else:
        med_quat = np.array([robust_std(quats[:, k] - median_filter(quats[:, k], size=5)) for k in range(4)])
        quat_std = np.maximum(med_quat, 0.01)

    return np.diag(np.concatenate([pos_std**2, quat_std**2])) * r_scale


def estimate_process_noise(
    positions: np.ndarray,
    quats: np.ndarray,
    fps: float,
    frame_gap: np.ndarray,
    q_scale: float = 1.0,
    q_vel_override: float | None = None,
    q_angvel_override: float | None = None
) -> np.ndarray:
    """Estimates empirical process dynamics covariance matrix Q."""
    dt = 1.0 / fps
    n = len(positions)
    is_consec = frame_gap == 1
    accel_valid = is_consec[:-1] & is_consec[1:]

    if q_vel_override is not None:
        q_vel = np.full(3, (q_vel_override * dt)**2)
    else:
        vel = np.diff(positions, axis=0) * fps
        accel = np.diff(vel, axis=0)[accel_valid] * fps
        q_vel = np.maximum((robust_std(accel, axis=0) * dt) ** 2, 1e-2) if len(accel) > 5 else np.full(3, 1.0)

    if q_angvel_override is not None:
        q_angvel_scalar = (q_angvel_override * dt)**2
    else:
        def rel_angle(qa: np.ndarray, qb: np.ndarray) -> float:
            w1, x1, y1, z1 = qa
            w2, x2, y2, z2 = qb
            Ra = np.array([
                [1 - 2*(y1*y1 + z1*z1), 2*(x1*y1 - z1*w1), 2*(x1*z1 + y1*w1)],
                [2*(x1*y1 + z1*w1), 1 - 2*(x1*x1 + z1*z1), 2*(y1*z1 - x1*w1)],
                [2*(x1*z1 - y1*w1), 2*(y1*z1 + x1*w1), 1 - 2*(x1*x1 + y1*y1)]
            ])
            Rb = np.array([
                [1 - 2*(y2*y2 + z2*z2), 2*(x2*y2 - z2*w2), 2*(x2*z2 + y2*w2)],
                [2*(x2*y2 + z2*w2), 1 - 2*(x2*x2 + z2*z2), 2*(y2*z2 - x2*w2)],
                [2*(x2*z2 - y2*w2), 2*(y2*z2 + x2*w2), 1 - 2*(x2*x2 + y2*y2)]
            ])
            Rrel = Ra.T @ Rb
            return float(np.arccos(np.clip((np.trace(Rrel) - 1.0) / 2.0, -1.0, 1.0)))

        angle = np.array([rel_angle(quats[i], quats[i + 1]) for i in range(n - 1)])
        angvel = angle * fps
        angaccel = np.diff(angvel)[accel_valid] * fps
        q_angvel_scalar = max(float((robust_std(angaccel) * dt) ** 2), 1e-3) if len(angaccel) > 5 else 1e-1

    return np.diag(np.concatenate([
        [1.0, 1.0, 1.0],
        q_vel,
        [1e-5, 1e-5, 1e-5, 1e-5],
        [q_angvel_scalar] * 3,
    ])) * q_scale


def compute_ekf_trajectory(
    df_pnp: pd.DataFrame,
    fps: float,
    disable_gating: bool = True,
    nis_p: float = 0.9999,
    max_rejections: int = 10,
    r_scale: float = 1.0,
    q_scale: float = 1.0,
    r_pos: float | None = None,
    r_quat: float | None = None,
    q_vel: float | None = None,
    q_angvel: float | None = None
) -> pd.DataFrame:
    """Applies the 13-state PoseEKF filter across the extracted PnP trajectory."""
    valid_mask = df_pnp['valid'].values
    raw_pos = df_pnp[['tx_mm', 'ty_mm', 'tz_mm']].values
    raw_quat_wxyz = df_pnp[['qw', 'qx', 'qy', 'qz']].values

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 10:
        return df_pnp

    frame_gap = np.diff(valid_idx)
    R_meas = estimate_measurement_noise(raw_pos[valid_mask], raw_quat_wxyz[valid_mask], r_scale=r_scale, r_pos_override=r_pos, r_quat_override=r_quat)
    Q_proc = estimate_process_noise(raw_pos[valid_mask], raw_quat_wxyz[valid_mask], fps, frame_gap, q_scale=q_scale, q_vel_override=q_vel, q_angvel_override=q_angvel)

    ekf = PoseEKF(
        process_noise=Q_proc,
        measurement_noise=R_meas,
        dt_ref=1.0 / fps,
        disable_gating=disable_gating,
        nis_gate_p=nis_p,
        max_consecutive_rejections=max_rejections
    )
    filt_pos = np.full_like(raw_pos, np.nan)
    filt_quat_xyzw = np.full_like(raw_quat_wxyz, np.nan)
    filt_valid = np.zeros(len(df_pnp), dtype=bool)

    last_k = None
    for k in range(len(df_pnp)):
        if not valid_mask[k]:
            continue
        if ekf.x is None:
            ekf.initialize(raw_pos[k], raw_quat_wxyz[k])
            last_k = k
            filt_valid[k] = True
        else:
            dt = (k - last_k) / fps
            ekf.predict(dt)
            accepted = ekf.update(raw_pos[k], raw_quat_wxyz[k])
            last_k = k
            filt_valid[k] = accepted

        filt_pos[k] = ekf.position
        qw, qx, qy, qz = ekf.quaternion
        filt_quat_xyzw[k] = [qx, qy, qz, qw]

    print(f"[EKF Tuning] Configuration: GatingDisabled={disable_gating}, R-Scale={r_scale:.2f}, Q-Scale={q_scale:.2f}")
    print(f"             Processed {len(df_pnp)} frames -> {ekf.n_updated} updated, {ekf.n_rejected} gated, {ekf.n_reinitialized} reinitialized.")

    df_out = df_pnp.copy()
    df_out['ekf_valid'] = filt_valid
    df_out['ekf_tx_mm'] = filt_pos[:, 0]
    df_out['ekf_ty_mm'] = filt_pos[:, 1]
    df_out['ekf_tz_mm'] = filt_pos[:, 2]
    df_out['ekf_qx'] = filt_quat_xyzw[:, 0]
    df_out['ekf_qy'] = filt_quat_xyzw[:, 1]
    df_out['ekf_qz'] = filt_quat_xyzw[:, 2]
    df_out['ekf_qw'] = filt_quat_xyzw[:, 3]
    return df_out


# ---------------------------------------------------------------------------
# Video Rendering Engine
# ---------------------------------------------------------------------------

def render_video_variant(
    mode_id: str,
    output_filename: str,
    stream_type: str,  # 'dark' or 'bright'
    use_filtered_stream: bool,  # If True, visual background is the clean filtered optical stream
    show_blobs: bool,           # If True, draws 2D candidate (Cyan) and Top-4 (Red) blob rings & centroids
    show_pnp: bool,             # If True, draws Raw PnP 3D pose axes & cyan trail
    show_ekf: bool,             # If True, draws EKF smoothed 3D pose axes & aqua trail
    show_gt: bool,              # If True, draws VR Ground Truth 3D pose axes & bright green trail
    show_trails: bool,
    trail_length: int,
    trail_thickness: int,
    video_path: str,
    timestamps_path: str,
    gt_pos_rhs: np.ndarray,
    gt_quats_rhs: np.ndarray,
    t_gt_all: np.ndarray,
    dt_sync: float,
    R_X: np.ndarray,
    t_X: np.ndarray,
    R_Y: np.ndarray,
    t_Y: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_pts_3d: np.ndarray,
    df_pnp: pd.DataFrame | None,
    start_frame: int = 0,
    count_frames: int | None = None
) -> dict:
    """Renders a single video variant with the specified pipeline visual layers."""
    df_ts = pd.read_csv(timestamps_path)
    cap = cv2.VideoCapture(video_path)
    total_in_frames = len(df_ts) if df_ts is not None else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Real-time frame rate calculation
    if len(df_ts) > 1:
        total_time_s = (df_ts['pts_ms'].iloc[-1] - df_ts['pts_ms'].iloc[0]) / 1000.0
        calculated_fps = (len(df_ts) - 1) / total_time_s if total_time_s > 0 else 30.0
    else:
        calculated_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    end_frame = total_in_frames if count_frames is None else min(start_frame + count_frames, total_in_frames)
    frames_to_process = end_frame - start_frame

    print(f"\n[Render {mode_id}] -> {output_filename}")
    print(f"  Stream: {stream_type.upper()} ({width}x{height} @ {calculated_fps:.2f} fps) | Frames: {start_frame}..{end_frame-1} ({frames_to_process} frames)")
    print(f"  Layers: FilteredBase={use_filtered_stream}, Blobs={show_blobs}, PnP={show_pnp}, EKF={show_ekf}, GT={show_gt}, Trails={show_trails}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, calculated_fps, (width, height))

    if start_frame > 0:
        for _ in range(start_frame):
            cap.grab()

    pnp_trail = []
    model_trail = []
    ekf_trail = []
    gt_trail = []
    is_dark = (stream_type == 'dark')
    t0 = time.time()

    legend_items = []
    if show_trails:
        if show_pnp:
            legend_items.append(("PnP (Raw)", (255, 220, 50)))
        if show_ekf:
            legend_items.append(("EKF (13-State)", (220, 255, 0)))
        if show_gt:
            legend_items.append(("Ground Truth", (0, 255, 0)))

    for idx, f_idx in enumerate(range(start_frame, end_frame)):
        ret, frame = cap.read()
        if not ret:
            break

        t_video = df_ts['pts_ms'].iloc[f_idx] / 1000.0 if f_idx < len(df_ts) else f_idx / calculated_fps
        t_gt = t_video + dt_sync

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_pts, areas, bw_clean, bw_blobs = filter_and_detect_blobs(gray, is_dark=is_dark)

        # Base visual layer: Filtered optical stream vs raw footage
        if use_filtered_stream:
            # Mask out background completely, keeping only filtered bright markers + soft clean mask glow
            filtered_spots = cv2.bitwise_and(frame, frame, mask=bw_blobs)
            mask_glow = cv2.cvtColor(bw_blobs, cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(filtered_spots, 0.75, mask_glow, 0.25, 0)
        else:
            vis = frame.copy()

        # Identify top-4 blobs for tracking
        pnp_blob_idx = set()
        if len(detected_pts) >= 4:
            top4 = np.argsort(areas)[::-1][:4]
            pnp_blob_idx = set(top4)

        # Layer 1: 2D Detected Blobs (Candidate Cyan rings + Top-4 Red tracking rings)
        if show_blobs:
            for i, (cx, cy) in enumerate(detected_pts):
                if i not in pnp_blob_idx:
                    px, py = int(round(cx)), int(round(cy))
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(vis, (px, py), 7, (255, 200, 0), 2, cv2.LINE_AA)
                        cv2.circle(vis, (px, py), 2, (0, 255, 150), -1, cv2.LINE_AA)

            if pnp_blob_idx:
                for i in pnp_blob_idx:
                    cx, cy = detected_pts[i]
                    px, py = int(round(cx)), int(round(cy))
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(vis, (px, py), 8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.circle(vis, (px, py), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # Layer 1b: For PnP / EKF stages, show the 4 active tracking marker points
        elif (show_pnp or show_ekf) and pnp_blob_idx:
            for i in pnp_blob_idx:
                cx, cy = detected_pts[i]
                px, py = int(round(cx)), int(round(cy))
                if 0 <= px < width and 0 <= py < height:
                    cv2.circle(vis, (px, py), 8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(vis, (px, py), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # Layer 2: Ground Truth 3D Pose
        if show_gt:
            gt_in_bounds = (t_gt >= t_gt_all[0]) and (t_gt <= t_gt_all[-1])
            if gt_in_bounds:
                p_gt_vr = np.array([np.interp(t_gt, t_gt_all, gt_pos_rhs[:, d]) for d in range(3)])
                q_gt_vr = np.array([np.interp(t_gt, t_gt_all, gt_quats_rhs[:, d]) for d in range(4)])
                q_gt_vr /= np.linalg.norm(q_gt_vr)
                R_gt_vr = R_scipy.from_quat(q_gt_vr).as_matrix()

                # T_cam_to_bar = X * T_vr_to_ctrl * Y
                R_gt_cam = R_X @ R_gt_vr @ R_Y
                p_gt_cam = R_X @ (R_gt_vr @ t_Y + p_gt_vr) + t_X  # meters
                tvec_gt_cam_mm = (p_gt_cam * 1000.0).reshape(3, 1)

                # 3D Coordinate Axes at bar origin (Bright Green, Yellow, Orange)
                draw_3d_axes_clean(
                    vis, R_gt_cam, tvec_gt_cam_mm, K, dist,
                    axis_length_mm=75.0, thickness=2,
                    colors=((0, 255, 0), (50, 255, 255), (0, 165, 255))
                )

                # Trail point
                if show_trails:
                    rvec_gt_cam, _ = cv2.Rodrigues(R_gt_cam)
                    proj_gt_origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec_gt_cam, tvec_gt_cam_mm, K, dist)
                    gt_pt_2d = proj_gt_origin.reshape(-1, 2)[0]
                    gt_trail.append((int(gt_pt_2d[0]), int(gt_pt_2d[1])))
            else:
                if show_trails:
                    gt_trail.append(None)

        # Layer 3: Raw PnP 3D Pose
        if show_pnp and df_pnp is not None:
            if f_idx < len(df_pnp) and df_pnp['valid'].iloc[f_idx]:
                row = df_pnp.iloc[f_idx]
                tvec_pnp_mm = np.array([[row['tx_mm']], [row['ty_mm']], [row['tz_mm']]])
                q_pnp = np.array([row['qx'], row['qy'], row['qz'], row['qw']])
                R_pnp_cam = R_scipy.from_quat(q_pnp).as_matrix()
                rvec_pnp_cam, _ = cv2.Rodrigues(R_pnp_cam)

                # 3D Coordinate Axes (Red X, Green Y, Blue Z)
                draw_3d_axes_clean(
                    vis, R_pnp_cam, tvec_pnp_mm, K, dist,
                    axis_length_mm=85.0, thickness=2,
                    colors=((0, 0, 255), (0, 255, 0), (255, 100, 0))
                )

                # Trail point
                if show_trails:
                    proj_pnp_origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec_pnp_cam, tvec_pnp_mm, K, dist)
                    pnp_pt_2d = proj_pnp_origin.reshape(-1, 2)[0]
                    pnp_trail.append((int(pnp_pt_2d[0]), int(pnp_pt_2d[1])))
            else:
                if show_trails:
                    pnp_trail.append(None)

        # Layer 4: EKF Smoothed 3D Pose
        if show_ekf and df_pnp is not None and 'ekf_valid' in df_pnp.columns:
            if f_idx < len(df_pnp) and df_pnp['ekf_valid'].iloc[f_idx]:
                row = df_pnp.iloc[f_idx]
                tvec_ekf_mm = np.array([[row['ekf_tx_mm']], [row['ekf_ty_mm']], [row['ekf_tz_mm']]])
                q_ekf = np.array([row['ekf_qx'], row['ekf_qy'], row['ekf_qz'], row['ekf_qw']])
                R_ekf_cam = R_scipy.from_quat(q_ekf).as_matrix()
                rvec_ekf_cam, _ = cv2.Rodrigues(R_ekf_cam)

                # 3D Coordinate Axes (Red X, Green Y, Deep Sky Blue Z)
                draw_3d_axes_clean(
                    vis, R_ekf_cam, tvec_ekf_mm, K, dist,
                    axis_length_mm=85.0, thickness=2,
                    colors=((0, 0, 255), (0, 255, 0), (255, 150, 0))
                )

                # Trail point
                if show_trails:
                    proj_ekf_origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec_ekf_cam, tvec_ekf_mm, K, dist)
                    ekf_pt_2d = proj_ekf_origin.reshape(-1, 2)[0]
                    ekf_trail.append((int(ekf_pt_2d[0]), int(ekf_pt_2d[1])))
            else:
                if show_trails:
                    ekf_trail.append(None)

        # Layer 5: Fading 3D Trajectory Trails
        if show_trails:
            if len(pnp_trail) > trail_length:
                pnp_trail.pop(0)
            if len(ekf_trail) > trail_length:
                ekf_trail.pop(0)
            if len(gt_trail) > trail_length:
                gt_trail.pop(0)

            # PnP Raw Trail (Cyan fading)
            for tr_i in range(1, len(pnp_trail)):
                alpha = 0.30 + 0.70 * (tr_i / float(trail_length))
                if pnp_trail[tr_i - 1] is not None and pnp_trail[tr_i] is not None:
                    p1, p2 = pnp_trail[tr_i - 1], pnp_trail[tr_i]
                    if 0 <= p1[0] < width and 0 <= p1[1] < height and 0 <= p2[0] < width and 0 <= p2[1] < height:
                        col_pnp = (int(255 * alpha), int(220 * alpha), int(50 * alpha))
                        cv2.line(vis, p1, p2, col_pnp, trail_thickness, cv2.LINE_AA)
                        if tr_i == len(pnp_trail) - 1:
                            cv2.circle(vis, p2, 4, (255, 220, 50), -1, cv2.LINE_AA)

            # EKF Smoothed Trail (Bright Emerald / Aqua fading)
            for tr_i in range(1, len(ekf_trail)):
                alpha = 0.30 + 0.70 * (tr_i / float(trail_length))
                if ekf_trail[tr_i - 1] is not None and ekf_trail[tr_i] is not None:
                    e1, e2 = ekf_trail[tr_i - 1], ekf_trail[tr_i]
                    if 0 <= e1[0] < width and 0 <= e1[1] < height and 0 <= e2[0] < width and 0 <= e2[1] < height:
                        col_ekf = (int(220 * alpha), int(255 * alpha), int(0 * alpha))
                        cv2.line(vis, e1, e2, col_ekf, trail_thickness, cv2.LINE_AA)
                        if tr_i == len(ekf_trail) - 1:
                            cv2.circle(vis, e2, 4, (220, 255, 0), -1, cv2.LINE_AA)

            # GT Trail (Bright Green fading)
            for tr_i in range(1, len(gt_trail)):
                alpha = 0.30 + 0.70 * (tr_i / float(trail_length))
                if gt_trail[tr_i - 1] is not None and gt_trail[tr_i] is not None:
                    g1, g2 = gt_trail[tr_i - 1], gt_trail[tr_i]
                    if 0 <= g1[0] < width and 0 <= g1[1] < height and 0 <= g2[0] < width and 0 <= g2[1] < height:
                        col_gt = (int(0 * alpha), int(255 * alpha), int(0 * alpha))
                        cv2.line(vis, g1, g2, col_gt, trail_thickness, cv2.LINE_AA)
                        if tr_i == len(gt_trail) - 1:
                            cv2.circle(vis, g2, 4, (0, 255, 0), -1, cv2.LINE_AA)

            if legend_items:
                draw_trail_legend(vis, legend_items)

        writer.write(vis)

        if (idx + 1) % 1000 == 0 or idx == frames_to_process - 1:
            elapsed = time.time() - t0
            print(f"    Processed {idx + 1}/{frames_to_process} frames ({elapsed:.1f}s, {(idx + 1)/elapsed:.1f} fps)...")

    cap.release()
    writer.release()

    # Convert to web-compatible H.264 (yuv420p + faststart) if ffmpeg is available
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
        "stream": stream_type,
        "layers": {
            "filtered_base": use_filtered_stream,
            "blobs": show_blobs,
            "pnp": show_pnp,
            "ekf": show_ekf,
            "gt": show_gt,
            "trails": show_trails
        },
        "frames": frames_to_process,
        "fps": calculated_fps,
        "duration_s": frames_to_process / calculated_fps
    }


def generate_metadata_markdown(
    out_dir: Path,
    timestamp_str: str,
    calib_data: dict,
    video_records: list[dict],
    start_frame: int,
    count_frames: int | None,
    show_trails: bool,
    trail_length: int,
    ekf_settings: dict
):
    """Generates the README.md documentation file inside the timestamped output directory."""
    dt_sync = calib_data["time_offset_seconds"]
    cam_trans = calib_data["camera_to_vr_extrinsics"]["translation_m"]
    cam_euler = calib_data["camera_to_vr_extrinsics"]["rotation_euler_deg"]
    ctrl_trans_mm = calib_data["controller_to_bar_extrinsics"]["translation_mm"]
    ctrl_euler = calib_data["controller_to_bar_extrinsics"]["rotation_euler_deg"]
    metrics = calib_data.get("metrics", {})

    md_content = f"""# Tracking Video Render Dataset

**Generation Timestamp**: `{timestamp_str}`  
**Output Directory**: `{out_dir.resolve()}`  

---

## 1. Video Specifications & Processing Pipeline

* **Start Frame**: `{start_frame}`
* **Frame Count**: `{count_frames if count_frames is not None else 'All Frames'}`
* **Trails Enabled**: `{show_trails}` (Length: `{trail_length}` frames)
* **Text / HUD Overlay**: `Disabled (Clean Visuals)`

### Sequential Pipeline Stages
1. **Raw Acquisition**: Unprocessed optical camera streams (1000 µs dark and 10000 µs bright).
2. **Optical Pre-Filtration**: Multi-stage filtering ($3\\times 3$ Gaussian blur, dynamic peak threshold $T \\ge 180$, morphological opening) completely suppresses background noise and isolates genuine LED emissions without any overlay markers.
3. **2D Blob Extraction**: High-precision sub-pixel centroid moment analysis with circularity gating ($4\\pi A / P^2 > 0.25$) rendered onto the filtered stream.
4. **PnP 6-DoF Rigid Body Pose**: SQPnP pose estimation rendered with RGB 3D coordinate axes and 4 active tracking markers on the filtered optical stream.
5. **13-State EKF Smoothing**: Real-time constant-velocity/angular-velocity filtering with tunable dynamics.

### EKF Filter Parameters
* **NIS Outlier Gating Enabled**: `{not ekf_settings['disable_gating']}` (Confidence $p = {ekf_settings['nis_p']}$)
* **Measurement Covariance Scale ($R_{{\\text{{scale}}}}$)**: `{ekf_settings['r_scale']:.2f}` (Pos Floor: `{ekf_settings['r_pos'] if ekf_settings['r_pos'] is not None else 'Auto (3.0mm min)'}`)
* **Process Dynamics Scale ($Q_{{\\text{{scale}}}}$)**: `{ekf_settings['q_scale']:.2f}` (Vel Noise: `{ekf_settings['q_vel'] if ekf_settings['q_vel'] is not None else 'Auto'}`)

### Calibration & Alignment Parameters
* **Time Synchronization Offset ($\\Delta t$)**: `+{dt_sync:.4f} s` ($t_{{\\text{{GT}}}} = t_{{\\text{{video}}}} + {dt_sync:.4f}\\text{{ s}}$)
* **Camera Extrinsics in VR World ($T_{{\\text{{cam}}\\to\\text{{vr}}}}$)**:
  * Translation: `[{cam_trans[0]:.4f}, {cam_trans[1]:.4f}, {cam_trans[2]:.4f}] m`
  * Euler Angles (XYZ): `[{cam_euler[0]:.2f}°, {cam_euler[1]:.2f}°, {cam_euler[2]:.2f}°]`
* **Controller-to-Bar Offset ($T_{{\\text{{ctrl}}\\to\\text{{bar}}}}$)**:
  * Translation: `[{ctrl_trans_mm[0]:.2f}, {ctrl_trans_mm[1]:.2f}, {ctrl_trans_mm[2]:.2f}] mm`
  * Euler Angles (XYZ): `[{ctrl_euler[0]:.2f}°, {ctrl_euler[1]:.2f}°, {ctrl_euler[2]:.2f}°]`
* **Dataset Alignment Quality**:
  * Median 3D Position Error: `{metrics.get('median_pos_error_mm', 18.84):.2f} mm`
  * Median 3D Rotation Error: `{metrics.get('median_rot_error_deg', 3.55):.2f}°`

---

## 2. Generated Video Files Description

| Filename | Shutter / Stream | Pipeline Visual Layers Included | Duration | Frame Count |
| :--- | :--- | :--- | :--- | :--- |
"""
    for v in video_records:
        layers_desc = []
        if v["layers"]["filtered_base"] and not v["layers"]["blobs"] and not v["layers"]["pnp"] and not v["layers"]["ekf"] and not v["layers"]["gt"]:
            layers_desc.append("Filtered Stream (Isolated LEDs, No Markers)")
        elif v["layers"]["filtered_base"] and not v["layers"]["blobs"] and not v["layers"]["pnp"] and not v["layers"]["ekf"] and v["layers"]["gt"]:
            layers_desc.append("Filtered Stream (No Markers) + VR Ground Truth 3D Pose")
        else:
            if v["layers"]["filtered_base"]:
                layers_desc.append("Filtered Stream Backdrop")
            if v["layers"]["blobs"]:
                layers_desc.append("2D Detected Blobs (Cyan Candidates & Red PnP)")
            if v["layers"]["pnp"]:
                layers_desc.append("Raw PnP 3D Pose (RGB Axes & Active Red Markers)")
            if v["layers"]["ekf"]:
                layers_desc.append("EKF Smoothed 3D Pose (RGB Axes & Aqua Trail)")
            if v["layers"]["gt"]:
                layers_desc.append("VR Ground Truth 3D Pose (Magenta/Yellow Axes & Trail)")
            if v["layers"]["trails"] and (v["layers"]["pnp"] or v["layers"]["ekf"] or v["layers"]["gt"]):
                layers_desc.append("3D Trajectory Trails")
            if not layers_desc:
                layers_desc.append("Raw Footage (No Overlay)")
        
        desc_str = " + ".join(layers_desc)
        shutter_str = "1000 µs (Dark IR)" if v["stream"] == "dark" else "10000 µs (Bright Visual)"
        md_content += f"| `{v['filename']}` | {shutter_str} | {desc_str} | {v['duration_s']:.2f} s | {v['frames']} frames |\n"

    md_content += """
---

## 3. Visual Layer Color Conventions

* **Filtered Stream Background**: Unfiltered background noise suppressed, revealing only true optical LED emissions.
* **Cyan Circles / Dots**: Sub-pixel detected 2D centroids of filtered candidate optical IR LEDs.
* **Red Circles / Dots**: The 4 optical LED blobs selected and actively used for the PnP 6-DoF pose calculation.
* **RGB Coordinate Axes (PnP / EKF)**: Optical 6-DoF rigid body pose (+X: Red, +Y: Green, +Z: Blue).
* **Cyan Trail**: 3D motion history trail of Raw PnP tracked bar origin.
* **Aqua / Emerald Trail**: 3D motion history trail of EKF smoothed bar origin.
* **Bright Green / Yellow / Orange Axes**: Transformed VR Ground Truth (Right Controller) 6-DoF pose (+X: Bright Green, +Y: Yellow, +Z: Orange).
* **Bright Green Trail**: 3D motion history trail of VR Ground Truth bar origin.
"""

    readme_path = out_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[Settings] Saved settings and video descriptions to {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Render synchronized tracking video combinations to a timestamped folder.")
    parser.add_argument("--dark-video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--dark-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--bright-video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--bright-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--gt-csv", type=str, default=str(ROOT / "data" / "GT_pos" / "20260911_130547_284_P01" / "pose_before_render.csv"))
    parser.add_argument("--calib", type=str, default=str(ROOT / "data" / "camera_calibration.yaml"))
    parser.add_argument("--geometry", type=str, default=str(ROOT / "data" / "geometry.yaml"))
    parser.add_argument("--alignment-json", type=str, default=str(ROOT / "data" / "alignment_calibration.json"))
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output folder path (defaults to a new data/renders/renders_YYYYMMDD_HHMMSS/ directory)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--count", type=int, default=None, help="Number of frames to render (default: all)")
    parser.add_argument("--no-trails", action="store_true", help="Disable fading 3D trajectory trails")
    parser.add_argument("--trail-length", type=int, default=35, help="Length of fading trajectory trail in frames")
    parser.add_argument("--trail-thickness", type=int, default=3, help="Line thickness for 3D trajectory trails (default: 3)")

    # EKF Hyperparameter Tuning Flags
    parser.add_argument("--ekf-enable-gating", action="store_true", help="Enable strict Chi-Square NIS outlier rejection (disabled by default to prevent dropping valid motion frames)")
    parser.add_argument("--ekf-nis-p", type=float, default=0.9999, help="Chi-Square confidence probability for NIS outlier rejection gate (default: 0.9999)")
    parser.add_argument("--ekf-max-rejections", type=int, default=10, help="Max consecutive outlier rejections before forced filter reset (default: 10)")
    parser.add_argument("--ekf-r-scale", type=float, default=1.0, help="Measurement covariance (R) multiplier. Higher = heavier trajectory smoothing, Lower = tighter tracking (default: 1.0)")
    parser.add_argument("--ekf-q-scale", type=float, default=1.0, help="Process dynamics covariance (Q) multiplier. Higher = more agile/responsive to fast movements, Lower = stiffer constant velocity (default: 1.0)")
    parser.add_argument("--ekf-r-pos", type=float, default=None, help="Explicit measurement position standard deviation in mm (default: auto from MAD residuals, 3.0mm floor)")
    parser.add_argument("--ekf-r-quat", type=float, default=None, help="Explicit measurement orientation quaternion standard deviation (default: auto from MAD residuals, 0.01 floor)")
    parser.add_argument("--ekf-q-vel", type=float, default=None, help="Explicit process linear velocity noise std in mm/s/step (default: auto from acceleration)")
    parser.add_argument("--ekf-q-angvel", type=float, default=None, help="Explicit process angular velocity noise std in rad/s/step (default: auto from angular accel)")

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ROOT / "data" / "renders" / "geometry-based" / f"renders_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Init] Output directory initialized at: {out_dir.resolve()}")

    # Load calibration & geometry
    with open(args.alignment_json) as f:
        calib_data = json.load(f)

    dt_sync = calib_data["time_offset_seconds"]
    R_X = np.array(calib_data["camera_to_vr_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_X = np.array(calib_data["camera_to_vr_extrinsics"]["translation_m"], dtype=np.float64)
    R_Y = np.array(calib_data["controller_to_bar_extrinsics"]["rotation_matrix"], dtype=np.float64)
    t_Y = np.array(calib_data["controller_to_bar_extrinsics"]["translation_m"], dtype=np.float64)

    cal = yaml.safe_load(open(args.calib))
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    K[:2, :] *= 0.5  # 1280x800 -> 640x400
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

    geom = yaml.safe_load(open(args.geometry))
    marker_pos = {m["id"]: np.array(m["position"], dtype=np.float64) for m in geom["markers"]}
    marker_ids = sorted(marker_pos.keys())
    marker_pts_3d = np.array([marker_pos[m] for m in marker_ids])

    # Load Ground Truth poses
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

    # Load PnP trajectory cache & compute EKF smoothing
    pnp_cache = ROOT / "skripts" / "pnp_trajectory_extracted.csv"
    df_pnp = None
    ekf_settings = {
        "disable_gating": not args.ekf_enable_gating,
        "nis_p": args.ekf_nis_p,
        "max_rejections": args.ekf_max_rejections,
        "r_scale": args.ekf_r_scale,
        "q_scale": args.ekf_q_scale,
        "r_pos": args.ekf_r_pos,
        "r_quat": args.ekf_r_quat,
        "q_vel": args.ekf_q_vel,
        "q_angvel": args.ekf_q_angvel
    }

    if pnp_cache.exists():
        df_raw_pnp = pd.read_csv(pnp_cache)
        fps_est = 72.0
        if Path(args.dark_ts).exists():
            df_ts_dark = pd.read_csv(args.dark_ts)
            if len(df_ts_dark) > 1:
                total_s = (df_ts_dark['pts_ms'].iloc[-1] - df_ts_dark['pts_ms'].iloc[0]) / 1000.0
                fps_est = (len(df_ts_dark) - 1) / total_s
        df_pnp = compute_ekf_trajectory(
            df_raw_pnp,
            fps=fps_est,
            disable_gating=ekf_settings["disable_gating"],
            nis_p=ekf_settings["nis_p"],
            max_rejections=ekf_settings["max_rejections"],
            r_scale=ekf_settings["r_scale"],
            q_scale=ekf_settings["q_scale"],
            r_pos=ekf_settings["r_pos"],
            r_quat=ekf_settings["r_quat"],
            q_vel=ekf_settings["q_vel"],
            q_angvel=ekf_settings["q_angvel"]
        )

    show_trails = not args.no_trails
    video_records = []

    # 1. Dark video raw
    rec = render_video_variant(
        mode_id="1/12",
        output_filename=str(out_dir / "01_dark_raw.mp4"),
        stream_type="dark",
        use_filtered_stream=False, show_blobs=False, show_pnp=False, show_ekf=False, show_gt=False,
        show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 2. Dark video + GT
    rec = render_video_variant(
        mode_id="2/12",
        output_filename=str(out_dir / "02_dark_gt.mp4"),
        stream_type="dark",
        use_filtered_stream=False, show_blobs=False, show_pnp=False, show_ekf=False, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 3. Dark video + Filtration (Noise eliminated, pure isolated LEDs, NO markers)
    rec = render_video_variant(
        mode_id="3/12",
        output_filename=str(out_dir / "03_dark_filtration.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=False, show_pnp=False, show_ekf=False, show_gt=False,
        show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 4. Dark video + Filtration + GT (Filtered video + GT 3D pose, NO 2D blob rings)
    rec = render_video_variant(
        mode_id="4/12",
        output_filename=str(out_dir / "04_dark_filtration_gt.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=False, show_pnp=False, show_ekf=False, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 5. Dark video + Blobs Detection (Blobs/markers appear ON TOP of filtered video)
    rec = render_video_variant(
        mode_id="5/12",
        output_filename=str(out_dir / "05_dark_blobs.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=True, show_pnp=False, show_ekf=False, show_gt=False,
        show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 6. Dark video + Blobs + GT
    rec = render_video_variant(
        mode_id="6/12",
        output_filename=str(out_dir / "06_dark_blobs_gt.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=True, show_pnp=False, show_ekf=False, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 7. Dark video + Raw PnP (Filtered video backdrop + PnP 3D pose)
    rec = render_video_variant(
        mode_id="7/12",
        output_filename=str(out_dir / "07_dark_pnp.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=False, show_pnp=True, show_ekf=False, show_gt=False,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 8. Dark video + Raw PnP + GT
    rec = render_video_variant(
        mode_id="8/12",
        output_filename=str(out_dir / "08_dark_pnp_gt.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=False, show_pnp=True, show_ekf=False, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 9. Dark video + EKF (Filtered video backdrop + EKF 3D pose)
    rec = render_video_variant(
        mode_id="9/12",
        output_filename=str(out_dir / "09_dark_ekf.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=False, show_pnp=False, show_ekf=True, show_gt=False,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 10. Dark video + EKF + GT
    rec = render_video_variant(
        mode_id="10/12",
        output_filename=str(out_dir / "10_dark_ekf_gt.mp4"),
        stream_type="dark",
        use_filtered_stream=True, show_blobs=False, show_pnp=False, show_ekf=True, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 11. Bright video raw
    rec = render_video_variant(
        mode_id="11/12",
        output_filename=str(out_dir / "11_bright_raw.mp4"),
        stream_type="bright",
        use_filtered_stream=False, show_blobs=False, show_pnp=False, show_ekf=False, show_gt=False,
        show_trails=False, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.bright_video, timestamps_path=args.bright_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 12. Bright video + GT
    rec = render_video_variant(
        mode_id="12/12",
        output_filename=str(out_dir / "12_bright_gt.mp4"),
        stream_type="bright",
        use_filtered_stream=False, show_blobs=False, show_pnp=False, show_ekf=False, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length, trail_thickness=args.trail_thickness,
        video_path=args.bright_video, timestamps_path=args.bright_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # Generate Markdown Documentation
    generate_metadata_markdown(
        out_dir=out_dir,
        timestamp_str=timestamp_str,
        calib_data=calib_data,
        video_records=video_records,
        start_frame=args.start,
        count_frames=args.count,
        show_trails=show_trails,
        trail_length=args.trail_length,
        ekf_settings=ekf_settings
    )

    print(f"\n=======================================================")
    print(f"All 12 video combinations successfully rendered to:")
    print(f"  {out_dir.resolve()}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
