import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt

# 1. Load PnP trajectory and GT
df_pnp = pd.read_csv('skripts/pnp_trajectory_extracted.csv')
df_gt = pd.read_csv('data/GT_pos/20260911_130547_284_P01/pose_before_render.csv')
df_ts = pd.read_csv('data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_timestamps_20260911_170556_290626.csv')

print("PnP total frames:", len(df_pnp))
print("GT total rows:", len(df_gt))

# Let's inspect the second half (frames 2131 to 4262, video time ~59s to 118s)
# Video time at frame 2131:
t_mid = df_ts['pts_ms'].iloc[2131] / 1000.0
print(f"Frame 2131 video pts time: {t_mid:.2f} s")

# Compute angular speed profiles
t_pnp = df_pnp['time_s'].values
pnp_valid = df_pnp['valid'].values
pnp_quats = df_pnp[['qx', 'qy', 'qz', 'qw']].values

t_gt = df_gt['timestamp_seconds'].values
gt_quats = df_gt[['right_qx', 'right_qy', 'right_qz', 'right_qw']].values

pnp_ang_speed = np.zeros(len(t_pnp))
for i in range(1, len(t_pnp)-1):
    if pnp_valid[i-1] and pnp_valid[i+1]:
        dt = t_pnp[i+1] - t_pnp[i-1]
        if 0.001 < dt < 0.2:
            r1 = R_scipy.from_quat(pnp_quats[i-1])
            r2 = R_scipy.from_quat(pnp_quats[i+1])
            pnp_ang_speed[i] = (r1.inv() * r2).magnitude() / dt

gt_ang_speed = np.zeros(len(t_gt))
for i in range(1, len(t_gt)-1):
    dt = t_gt[i+1] - t_gt[i-1]
    if 0.001 < dt < 0.2:
        r1 = R_scipy.from_quat(gt_quats[i-1])
        r2 = R_scipy.from_quat(gt_quats[i+1])
        gt_ang_speed[i] = (r1.inv() * r2).magnitude() / dt

# Let's check cross-correlation specifically in windows:
# Window 1: First half (0 to 60s)
# Window 2: Second half (60 to 118s)
# Window 3: End section (90 to 118s)

def find_best_dt_window(t_v_start, t_v_end, name):
    mask_v = (t_pnp >= t_v_start) & (t_pnp <= t_v_end) & pnp_valid
    t_sub_pnp = t_pnp[mask_v]
    speed_sub_pnp = pnp_ang_speed[mask_v]
    
    t_grid = np.arange(t_v_start, t_v_end, 0.01)
    sig_pnp = np.interp(t_grid, t_sub_pnp, speed_sub_pnp)
    
    candidate_dts = np.arange(10.0, 30.0, 0.01)
    best_c = -1
    best_dt = None
    
    for dt in candidate_dts:
        t_gt_grid = t_grid + dt
        if t_gt_grid[0] < t_gt[0] or t_gt_grid[-1] > t_gt[-1]:
            continue
        sig_gt = np.interp(t_gt_grid, t_gt, gt_ang_speed)
        c = np.corrcoef(sig_pnp, sig_gt)[0, 1]
        if c > best_c:
            best_c = c
            best_dt = dt
            
    print(f"[{name}] Best dt in [{t_v_start:.1f}s..{t_v_end:.1f}s]: dt = {best_dt:.3f} s (corr: {best_c:.4f})")
    return best_dt, best_c

find_best_dt_window(10.0, 50.0, "First Half (10-50s)")
find_best_dt_window(50.0, 90.0, "Middle (50-90s)")
find_best_dt_window(90.0, 115.0, "Second Half / End (90-115s)")
find_best_dt_window(15.0, 115.0, "Full Duration (15-115s)")
