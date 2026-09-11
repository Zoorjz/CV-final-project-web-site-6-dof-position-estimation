import cv2
import pandas as pd
import numpy as np
import yaml
import itertools
from scipy.spatial.transform import Rotation as R_scipy
from scipy.signal import correlate, correlation_lags

# 1. Load calibration and geometry
calib_path = 'data/camera_calibration.yaml'
geom_path = 'data/geometry.yaml'
cal = yaml.safe_load(open(calib_path))
K = np.array(cal["camera_matrix"], dtype=np.float64)
K[:2, :] *= 0.5  # 1280x800 -> 640x400
dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

geom = yaml.safe_load(open(geom_path))
marker_pos = {m["id"]: np.array(m["position"], dtype=np.float64) for m in geom["markers"]}
marker_ids = sorted(marker_pos.keys())
perms = list(itertools.permutations(marker_ids))

# 2. Load Video and Timestamps
ts_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_timestamps_20260911_170556_290626.csv'
df_ts = pd.read_csv(ts_path)
video_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_video_20260911_170556_290626.mkv'

# 3. Load GT Poses
gt_path = 'data/GT_pos/20260911_130547_284_P01/pose_before_render.csv'
df_gt = pd.read_csv(gt_path)

print("Video timestamps range (pts_ms/1000):", df_ts['pts_ms'].iloc[0]/1000.0, "..", df_ts['pts_ms'].iloc[-1]/1000.0)
print("GT timestamps range (timestamp_seconds):", df_gt['timestamp_seconds'].iloc[0], "..", df_gt['timestamp_seconds'].iloc[-1])

# Extract PnP trajectory
cap = cv2.VideoCapture(video_path)
pnp_times = []
pnp_tvecs = []
pnp_rvecs = []
pnp_quats = [] # [x, y, z, w]
pnp_errs = []
pnp_valid = []

frame_idx = 0
prev_perm = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []
    for c in contours:
        M = cv2.moments(c)
        if 2.0 < M["m00"] < 1500.0:
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
            areas.append(M["m00"])
    pts = np.array(pts)
    areas = np.array(areas)
    
    t_sec = df_ts['pts_ms'].iloc[frame_idx] / 1000.0
    pnp_times.append(t_sec)
    
    pose_found = False
    if len(pts) >= 4:
        top_pts = pts[np.argsort(areas)[::-1][:4]]
        # try preferred first
        perm_list = [prev_perm] + [p for p in perms if p != prev_perm] if prev_perm is not None else perms
        best = None
        for perm in perm_list:
            obj = np.array([marker_pos[m] for m in perm])
            ok, rvec, tvec = cv2.solvePnP(obj, top_pts.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            R_mat, _ = cv2.Rodrigues(rvec)
            cam_pts = (R_mat @ obj.T).T + tvec.ravel()
            if (cam_pts[:, 2] <= 0).any():
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
            err = np.linalg.norm(proj.reshape(-1, 2) - top_pts, axis=1).mean()
            if not np.isfinite(err) or err > 50.0:
                continue
            if best is None or err < best[2]:
                best = (rvec, tvec, err, perm)
            if prev_perm is not None and perm == prev_perm and err < 2.0:
                break
        if best is not None:
            rvec, tvec, err, perm = best
            prev_perm = perm
            pnp_tvecs.append(tvec.ravel())
            pnp_rvecs.append(rvec.ravel())
            # Convert rvec to scipy quaternion (x, y, z, w)
            rot = R_scipy.from_rotvec(rvec.ravel())
            pnp_quats.append(rot.as_quat())
            pnp_errs.append(err)
            pnp_valid.append(True)
            pose_found = True
            
    if not pose_found:
        prev_perm = None
        pnp_tvecs.append(np.array([np.nan, np.nan, np.nan]))
        pnp_rvecs.append(np.array([np.nan, np.nan, np.nan]))
        pnp_quats.append(np.array([np.nan, np.nan, np.nan, np.nan]))
        pnp_errs.append(np.nan)
        pnp_valid.append(False)
        
    frame_idx += 1

cap.release()

pnp_times = np.array(pnp_times)
pnp_tvecs = np.array(pnp_tvecs) # mm
pnp_quats = np.array(pnp_quats)
pnp_valid = np.array(pnp_valid)

print(f"Extracted {len(pnp_times)} frames, {np.sum(pnp_valid)} valid PnP poses.")

# Save extracted PnP to file for fast reuse
df_pnp_out = pd.DataFrame({
    'time_s': pnp_times,
    'valid': pnp_valid,
    'tx_mm': pnp_tvecs[:, 0],
    'ty_mm': pnp_tvecs[:, 1],
    'tz_mm': pnp_tvecs[:, 2],
    'qx': pnp_quats[:, 0],
    'qy': pnp_quats[:, 1],
    'qz': pnp_quats[:, 2],
    'qw': pnp_quats[:, 3],
    'err': pnp_errs
})
df_pnp_out.to_csv('skripts/pnp_trajectory_extracted.csv', index=False)
print("Saved extracted PnP trajectory to skripts/pnp_trajectory_extracted.csv")
