import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import minimize

df_pnp = pd.read_csv('skripts/pnp_trajectory_extracted.csv')
df_gt = pd.read_csv('data/GT_pos/20260911_130547_284_P01/pose_before_render.csv')

t_pnp = df_pnp['time_s'].values
pnp_valid = df_pnp['valid'].values
pnp_quats = df_pnp[['qx', 'qy', 'qz', 'qw']].values

t_gt = df_gt['timestamp_seconds'].values
gt_quats = df_gt[['right_qx', 'right_qy', 'right_qz', 'right_qw']].values

# Compute angular speed profiles
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

# Let's search for an affine mapping: t_gt = alpha * t_video + beta
# alpha around 1.0 (or 0.8 to 1.3), beta around 0 to 40
best_corr = -1
best_alpha = 1.0
best_beta = 0.0

# Grid search
for alpha in np.arange(0.80, 1.25, 0.01):
    for beta in np.arange(10.0, 35.0, 0.2):
        t_eval_v = np.arange(15.0, 105.0, 0.05)
        t_eval_gt = alpha * t_eval_v + beta
        
        if t_eval_gt[0] < t_gt[0] or t_eval_gt[-1] > t_gt[-1]:
            continue
            
        sig_pnp = np.interp(t_eval_v, t_pnp[pnp_valid], pnp_ang_speed[pnp_valid])
        sig_gt = np.interp(t_eval_gt, t_gt, gt_ang_speed)
        
        c = np.corrcoef(sig_pnp, sig_gt)[0, 1]
        if c > best_corr:
            best_corr = c
            best_alpha = alpha
            best_beta = beta

print(f"Global Affine Time Mapping: t_gt = {best_alpha:.4f} * t_video + {best_beta:.4f}")
print(f"Peak Correlation: {best_corr:.4f}")

# Refine with local search around best_alpha, best_beta
for alpha in np.arange(best_alpha - 0.02, best_alpha + 0.02, 0.001):
    for beta in np.arange(best_beta - 0.5, best_beta + 0.5, 0.01):
        t_eval_v = np.arange(15.0, 105.0, 0.05)
        t_eval_gt = alpha * t_eval_v + beta
        
        if t_eval_gt[0] < t_gt[0] or t_eval_gt[-1] > t_gt[-1]:
            continue
            
        sig_pnp = np.interp(t_eval_v, t_pnp[pnp_valid], pnp_ang_speed[pnp_valid])
        sig_gt = np.interp(t_eval_gt, t_gt, gt_ang_speed)
        
        c = np.corrcoef(sig_pnp, sig_gt)[0, 1]
        if c > best_corr:
            best_corr = c
            best_alpha = alpha
            best_beta = beta

print(f"\nRefined Affine Time Mapping: t_gt = {best_alpha:.5f} * t_video + {best_beta:.5f}")
print(f"Refined Peak Correlation: {best_corr:.4f}")
