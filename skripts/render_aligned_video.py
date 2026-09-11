#!/usr/bin/env python3
"""render_aligned_video.py

Generates synchronized benchmark video combinations without on-screen text overlays:
  1. 01_dark_raw.mp4        : Dark video (1000us shutter), raw footage.
  2. 02_dark_gt.mp4         : Dark video + Ground Truth 3D pose & reprojected markers.
  3. 03_dark_blobs.mp4      : Dark video + 2D detected optical blob markers.
  4. 04_dark_pnp.mp4        : Dark video + PnP 3D pose & reprojected markers.
  5. 05_dark_pnp_gt.mp4     : Dark video + PnP 3D pose + Ground Truth 3D pose.
  6. 06_bright_raw.mp4      : Bright video (10000us shutter), raw footage.
  7. 07_bright_gt.mp4       : Bright video + Ground Truth 3D pose & reprojected markers.

Outputs are placed into a timestamped directory (e.g. data/renders_YYYYMMDD_HHMMSS/)
along with a generated settings & descriptions Markdown file (README.md).

Usage:
    # Generate all 7 video combinations:
    python skripts/render_aligned_video.py

    # Generate a specific frame range (e.g. 500 frames starting from middle):
    python skripts/render_aligned_video.py --start 2131 --count 500

    # Disable trajectory trails:
    python skripts/render_aligned_video.py --no-trails
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
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


def detect_blobs(gray: np.ndarray, thresh: int, min_area: float = 2.0, max_area: float = 1500.0) -> list[tuple[float, float]]:
    """Detects sub-pixel centroids of bright optical LED markers."""
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        M = cv2.moments(c)
        if min_area < M["m00"] < max_area:
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return pts


def render_video_variant(
    mode_id: str,
    output_filename: str,
    stream_type: str,  # 'dark' or 'bright'
    show_blobs: bool,
    show_pnp: bool,
    show_gt: bool,
    show_trails: bool,
    trail_length: int,
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
    """Renders a single video variant with the specified visual layers."""
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
    print(f"  Layers: Blobs={show_blobs}, PnP={show_pnp}, GT={show_gt}, Trails={show_trails}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, calculated_fps, (width, height))

    # Accurate seek using cap.grab()
    if start_frame > 0:
        for _ in range(start_frame):
            cap.grab()

    pnp_trail = []
    gt_trail = []
    blob_thresh = 50 if stream_type == 'dark' else 240
    t0 = time.time()

    for idx, f_idx in enumerate(range(start_frame, end_frame)):
        ret, frame = cap.read()
        if not ret:
            break

        t_video = df_ts['pts_ms'].iloc[f_idx] / 1000.0 if f_idx < len(df_ts) else f_idx / calculated_fps
        t_gt = t_video + dt_sync

        vis = frame.copy()

        # Layer 1: Detected 2D Blobs (Green rings)
        if show_blobs:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_pts = detect_blobs(gray, thresh=blob_thresh)
            for cx, cy in detected_pts:
                cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 255, 100), 1, cv2.LINE_AA)
                cv2.circle(vis, (int(round(cx)), int(round(cy))), 1, (0, 255, 100), -1, cv2.LINE_AA)

        # Layer 2: Ground Truth 3D Pose & Reprojections
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

                rvec_gt_cam, _ = cv2.Rodrigues(R_gt_cam)

                # Reproject GT markers (Magenta circles & crosshairs)
                proj_gt_markers, _ = cv2.projectPoints(marker_pts_3d, rvec_gt_cam, tvec_gt_cam_mm, K, dist)
                for pt in proj_gt_markers.reshape(-1, 2):
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(vis, (px, py), 5, (255, 0, 220), 1, cv2.LINE_AA)
                        cv2.drawMarker(vis, (px, py), (255, 0, 220), cv2.MARKER_CROSS, 6, 1)

                # 3D Coordinate Axes (Magenta, Yellow, Cyan)
                draw_3d_axes_clean(
                    vis, R_gt_cam, tvec_gt_cam_mm, K, dist,
                    axis_length_mm=75.0, thickness=2,
                    colors=((255, 50, 220), (50, 255, 255), (0, 165, 255))
                )

                # Trail point
                if show_trails:
                    proj_gt_origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec_gt_cam, tvec_gt_cam_mm, K, dist)
                    gt_pt_2d = proj_gt_origin.reshape(-1, 2)[0]
                    gt_trail.append((int(gt_pt_2d[0]), int(gt_pt_2d[1])))
            else:
                if show_trails:
                    gt_trail.append(None)

        # Layer 3: PnP 3D Pose & Reprojections
        if show_pnp and df_pnp is not None:
            if f_idx < len(df_pnp) and df_pnp['valid'].iloc[f_idx]:
                row = df_pnp.iloc[f_idx]
                tvec_pnp_mm = np.array([[row['tx_mm']], [row['ty_mm']], [row['tz_mm']]])
                q_pnp = np.array([row['qx'], row['qy'], row['qz'], row['qw']])
                R_pnp_cam = R_scipy.from_quat(q_pnp).as_matrix()
                rvec_pnp_cam, _ = cv2.Rodrigues(R_pnp_cam)

                # Reproject PnP markers (Cyan rings)
                proj_pnp_markers, _ = cv2.projectPoints(marker_pts_3d, rvec_pnp_cam, tvec_pnp_mm, K, dist)
                for pt in proj_pnp_markers.reshape(-1, 2):
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= px < width and 0 <= py < height:
                        cv2.circle(vis, (px, py), 4, (255, 255, 0), 1, cv2.LINE_AA)

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

        # Layer 4: Fading 3D Trajectory Trails
        if show_trails:
            if len(pnp_trail) > trail_length:
                pnp_trail.pop(0)
            if len(gt_trail) > trail_length:
                gt_trail.pop(0)

            # PnP Trail (Cyan fading)
            for tr_i in range(1, len(pnp_trail)):
                alpha = (tr_i / float(trail_length)) ** 1.5
                if pnp_trail[tr_i - 1] is not None and pnp_trail[tr_i] is not None:
                    p1, p2 = pnp_trail[tr_i - 1], pnp_trail[tr_i]
                    if 0 <= p1[0] < width and 0 <= p1[1] < height and 0 <= p2[0] < width and 0 <= p2[1] < height:
                        col_pnp = (int(255 * alpha), int(220 * alpha), int(50 * alpha))
                        cv2.line(vis, p1, p2, col_pnp, 1, cv2.LINE_AA)

            # GT Trail (Magenta fading)
            for tr_i in range(1, len(gt_trail)):
                alpha = (tr_i / float(trail_length)) ** 1.5
                if gt_trail[tr_i - 1] is not None and gt_trail[tr_i] is not None:
                    g1, g2 = gt_trail[tr_i - 1], gt_trail[tr_i]
                    if 0 <= g1[0] < width and 0 <= g1[1] < height and 0 <= g2[0] < width and 0 <= g2[1] < height:
                        col_gt = (int(220 * alpha), int(50 * alpha), int(255 * alpha))
                        cv2.line(vis, g1, g2, col_gt, 1, cv2.LINE_AA)

        writer.write(vis)

        if (idx + 1) % 1000 == 0 or idx == frames_to_process - 1:
            elapsed = time.time() - t0
            print(f"    Processed {idx + 1}/{frames_to_process} frames ({elapsed:.1f}s, {(idx + 1)/elapsed:.1f} fps)...")

    cap.release()
    writer.release()

    return {
        "filename": Path(output_filename).name,
        "stream": stream_type,
        "layers": {
            "blobs": show_blobs,
            "pnp": show_pnp,
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
    trail_length: int
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

## 1. Video Specifications & Settings

* **Start Frame**: `{start_frame}`
* **Frame Count**: `{count_frames if count_frames is not None else 'All Frames'}`
* **Trails Enabled**: `{show_trails}` (Length: `{trail_length}` frames)
* **Text / HUD Overlay**: `Disabled (Clean Visuals)`

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

| Filename | Shutter / Stream | Visual Layers Included | Duration | Frame Count |
| :--- | :--- | :--- | :--- | :--- |
"""
    for v in video_records:
        layers_desc = []
        if v["layers"]["blobs"]:
            layers_desc.append("2D Blob Centroids (Green)")
        if v["layers"]["pnp"]:
            layers_desc.append("PnP 3D Pose (RGB) & Reprojections (Cyan)")
        if v["layers"]["gt"]:
            layers_desc.append("VR Ground Truth 3D Pose (Magenta/Yellow)")
        if v["layers"]["trails"] and (v["layers"]["pnp"] or v["layers"]["gt"]):
            layers_desc.append("3D Trajectory Trails")
        if not layers_desc:
            layers_desc.append("Raw Footage (No Overlay)")
        
        desc_str = " + ".join(layers_desc)
        shutter_str = "1000 µs (Dark IR)" if v["stream"] == "dark" else "10000 µs (Bright Visual)"
        md_content += f"| `{v['filename']}` | {shutter_str} | {desc_str} | {v['duration_s']:.2f} s | {v['frames']} frames |\n"

    md_content += """
---

## 3. Visual Layer Color Conventions

* **Green Rings / Dots**: Sub-pixel detected 2D centroids of optical IR LEDs.
* **RGB Coordinate Axes**: Optical PnP 6-DoF rigid body pose (+X: Red, +Y: Green, +Z: Blue).
* **Cyan Rings**: 3D Optical marker positions reprojected into camera view via PnP pose.
* **Cyan Trail**: 3D motion history trail of PnP tracked bar origin.
* **Magenta / Yellow / Orange Axes**: Transformed VR Ground Truth (Right Controller) 6-DoF pose (+X: Magenta, +Y: Yellow, +Z: Orange).
* **Magenta Rings & Crosshairs**: 3D Optical marker positions reprojected from VR Ground Truth pose into camera view.
* **Magenta Trail**: 3D motion history trail of VR Ground Truth bar origin.
"""

    readme_path = out_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[Settings] Saved settings and video descriptions to {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Render 7 clean tracking video combinations to a timestamped folder.")
    parser.add_argument("--dark-video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--dark-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_1000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--bright-video", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_video_20260911_170556_290626.mkv"))
    parser.add_argument("--bright-ts", type=str, default=str(ROOT / "data" / "GT_videos" / "dataset_20260911_170556_290626" / "dataset_10000us_timestamps_20260911_170556_290626.csv"))
    parser.add_argument("--gt-csv", type=str, default=str(ROOT / "data" / "GT_pos" / "20260911_130547_284_P01" / "pose_before_render.csv"))
    parser.add_argument("--calib", type=str, default=str(ROOT / "data" / "camera_calibration.yaml"))
    parser.add_argument("--geometry", type=str, default=str(ROOT / "data" / "geometry.yaml"))
    parser.add_argument("--alignment-json", type=str, default=str(ROOT / "data" / "alignment_calibration.json"))
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output folder path (defaults to data/renders_YYYYMMDD_HHMMSS/)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--count", type=int, default=None, help="Number of frames to render (default: all)")
    parser.add_argument("--no-trails", action="store_true", help="Disable fading 3D trajectory trails")
    parser.add_argument("--trail-length", type=int, default=35, help="Length of fading trajectory trail in frames")
    args = parser.parse_args()

    # Create timestamped output directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / f"renders_{timestamp_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Init] Output directory initialized at: {out_dir}")

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

    # Load PnP trajectory cache
    pnp_cache = ROOT / "skripts" / "pnp_trajectory_extracted.csv"
    df_pnp = pd.read_csv(pnp_cache) if pnp_cache.exists() else None

    show_trails = not args.no_trails
    video_records = []

    # 1. Dark video (1000 shutter) raw
    rec = render_video_variant(
        mode_id="1/7",
        output_filename=str(out_dir / "01_dark_raw.mp4"),
        stream_type="dark",
        show_blobs=False, show_pnp=False, show_gt=False,
        show_trails=False, trail_length=args.trail_length,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 2. Dark video + GT
    rec = render_video_variant(
        mode_id="2/7",
        output_filename=str(out_dir / "02_dark_gt.mp4"),
        stream_type="dark",
        show_blobs=False, show_pnp=False, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 3. Dark video + Blobs detection
    rec = render_video_variant(
        mode_id="3/7",
        output_filename=str(out_dir / "03_dark_blobs.mp4"),
        stream_type="dark",
        show_blobs=True, show_pnp=False, show_gt=False,
        show_trails=False, trail_length=args.trail_length,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 4. Dark video + PnP full
    rec = render_video_variant(
        mode_id="4/7",
        output_filename=str(out_dir / "04_dark_pnp.mp4"),
        stream_type="dark",
        show_blobs=False, show_pnp=True, show_gt=False,
        show_trails=show_trails, trail_length=args.trail_length,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 5. Dark video + PnP + GT
    rec = render_video_variant(
        mode_id="5/7",
        output_filename=str(out_dir / "05_dark_pnp_gt.mp4"),
        stream_type="dark",
        show_blobs=False, show_pnp=True, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length,
        video_path=args.dark_video, timestamps_path=args.dark_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 6. Bright video (10000 shutter) raw
    rec = render_video_variant(
        mode_id="6/7",
        output_filename=str(out_dir / "06_bright_raw.mp4"),
        stream_type="bright",
        show_blobs=False, show_pnp=False, show_gt=False,
        show_trails=False, trail_length=args.trail_length,
        video_path=args.bright_video, timestamps_path=args.bright_ts,
        gt_pos_rhs=gt_pos_rhs, gt_quats_rhs=gt_quats_rhs, t_gt_all=t_gt_all,
        dt_sync=dt_sync, R_X=R_X, t_X=t_X, R_Y=R_Y, t_Y=t_Y,
        K=K, dist=dist, marker_pts_3d=marker_pts_3d, df_pnp=df_pnp,
        start_frame=args.start, count_frames=args.count
    )
    video_records.append(rec)

    # 7. Bright video + GT
    rec = render_video_variant(
        mode_id="7/7",
        output_filename=str(out_dir / "07_bright_gt.mp4"),
        stream_type="bright",
        show_blobs=False, show_pnp=False, show_gt=True,
        show_trails=show_trails, trail_length=args.trail_length,
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
        trail_length=args.trail_length
    )

    print(f"\n=======================================================")
    print(f"All 7 video combinations successfully rendered to:")
    print(f"  {out_dir.resolve()}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
