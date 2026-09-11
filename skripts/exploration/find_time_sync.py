import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 1. Load PnP trajectory
df_pnp = pd.read_csv('skripts/pnp_trajectory_extracted.csv')
pnp_valid = df_pnp['valid'].values
t_pnp = df_pnp['time_s'].values
pnp_quats = df_pnp[['qx', 'qy', 'qz', 'qw']].values  # scipy format (x, y, z, w)
pnp_tvec = df_pnp[['tx_mm', 'ty_mm', 'tz_mm']].values / 1000.0  # in meters

# 2. Load GT trajectory
df_gt = pd.read_csv('data/GT_pos/20260911_130547_284_P01/pose_before_render.csv')
t_gt = df_gt['timestamp_seconds'].values
gt_pos = df_gt[['right_px', 'right_py', 'right_pz']].values  # meters
gt_quats = df_gt[['right_qx', 'right_qy', 'right_qz', 'right_qw']].values # (x, y, z, w)

# Compute angular speed for PnP
pnp_ang_speed = np.zeros(len(t_pnp))
for i in range(1, len(t_pnp)-1):
    if pnp_valid[i-1] and pnp_valid[i+1]:
        dt = t_pnp[i+1] - t_pnp[i-1]
        if 0.001 < dt < 0.2:
            r1 = R_scipy.from_quat(pnp_quats[i-1])
            r2 = R_scipy.from_quat(pnp_quats[i+1])
            diff_rot = r1.inv() * r2
            angle = diff_rot.magnitude()
            pnp_ang_speed[i] = angle / dt

# Compute linear speed for PnP
pnp_lin_speed = np.zeros(len(t_pnp))
for i in range(1, len(t_pnp)-1):
    if pnp_valid[i-1] and pnp_valid[i+1]:
        dt = t_pnp[i+1] - t_pnp[i-1]
        if 0.001 < dt < 0.2:
            dist = np.linalg.norm(pnp_tvec[i+1] - pnp_tvec[i-1])
            pnp_lin_speed[i] = dist / dt

# Compute angular speed for GT
gt_ang_speed = np.zeros(len(t_gt))
for i in range(1, len(t_gt)-1):
    dt = t_gt[i+1] - t_gt[i-1]
    if 0.001 < dt < 0.2:
        r1 = R_scipy.from_quat(gt_quats[i-1])
        r2 = R_scipy.from_quat(gt_quats[i+1])
        diff_rot = r1.inv() * r2
        angle = diff_rot.magnitude()
        gt_ang_speed[i] = angle / dt

# Compute linear speed for GT
gt_lin_speed = np.zeros(len(t_gt))
for i in range(1, len(t_gt)-1):
    dt = t_gt[i+1] - t_gt[i-1]
    if 0.001 < dt < 0.2:
        dist = np.linalg.norm(gt_pos[i+1] - gt_pos[i-1])
        gt_lin_speed[i] = dist / dt

# Let's resample both onto a regular 100 Hz grid
t_grid_pnp = np.arange(0, t_pnp[-1], 0.01)
pnp_ang_interp = np.interp(t_grid_pnp, t_pnp[pnp_valid], pnp_ang_speed[pnp_valid])
pnp_lin_interp = np.interp(t_grid_pnp, t_pnp[pnp_valid], pnp_lin_speed[pnp_valid])

t_grid_gt = np.arange(t_gt[0], t_gt[-1], 0.01)
gt_ang_interp = np.interp(t_grid_gt, t_gt, gt_ang_speed)
gt_lin_interp = np.interp(t_grid_gt, t_gt, gt_lin_speed)

# Search for time offset: t_gt = t_video + t_offset
# Candidate offsets from 0s to 50s
candidate_offsets = np.arange(10.0, 50.0, 0.01)
corrs_ang = []
corrs_lin = []

for offset in candidate_offsets:
    # Overlapping window:
    t_min = max(t_grid_pnp[0] + offset, t_grid_gt[0])
    t_max = min(t_grid_pnp[-1] + offset, t_grid_gt[-1])
    if t_max - t_min < 20.0:
        corrs_ang.append(0)
        corrs_lin.append(0)
        continue
    
    t_eval_gt = np.arange(t_min, t_max, 0.01)
    t_eval_pnp = t_eval_gt - offset
    
    sig_pnp_ang = np.interp(t_eval_pnp, t_grid_pnp, pnp_ang_interp)
    sig_gt_ang = np.interp(t_eval_gt, t_grid_gt, gt_ang_interp)
    
    sig_pnp_lin = np.interp(t_eval_pnp, t_grid_pnp, pnp_lin_interp)
    sig_gt_lin = np.interp(t_eval_gt, t_grid_gt, gt_lin_interp)
    
    # Pearson correlation
    c_ang = np.corrcoef(sig_pnp_ang, sig_gt_ang)[0, 1]
    c_lin = np.corrcoef(sig_pnp_lin, sig_gt_lin)[0, 1]
    corrs_ang.append(c_ang)
    corrs_lin.append(c_lin)

corrs_ang = np.array(corrs_ang)
corrs_lin = np.array(corrs_lin)

best_idx_ang = np.nanargmax(corrs_ang)
best_offset_ang = candidate_offsets[best_idx_ang]
best_corr_ang = corrs_ang[best_idx_ang]

best_idx_lin = np.nanargmax(corrs_lin)
best_offset_lin = candidate_offsets[best_idx_lin]
best_corr_lin = corrs_lin[best_idx_lin]

print(f"Best Angular Speed Correlation: {best_corr_ang:.4f} at time offset: {best_offset_ang:.3f} s")
print(f"Best Linear Speed Correlation: {best_corr_lin:.4f} at time offset: {best_offset_lin:.3f} s")

# Top 5 peaks in angular correlation
sorted_indices = np.argsort(corrs_ang)[::-1][:5]
print("\nTop 5 Angular Offset Candidates:")
for idx in sorted_indices:
    print(f"  Offset: {candidate_offsets[idx]:.3f} s -> Corr: {corrs_ang[idx]:.4f}")
