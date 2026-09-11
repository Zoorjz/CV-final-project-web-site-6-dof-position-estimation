import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import minimize, least_squares
import cv2

# 1. Load PnP trajectory
df_pnp = pd.read_csv('skripts/pnp_trajectory_extracted.csv')
pnp_valid = df_pnp['valid'].values
t_pnp = df_pnp['time_s'].values
pnp_quats = df_pnp[['qx', 'qy', 'qz', 'qw']].values  # (x, y, z, w)
pnp_tvecs = df_pnp[['tx_mm', 'ty_mm', 'tz_mm']].values / 1000.0  # in meters

# 2. Load GT trajectory
df_gt = pd.read_csv('data/GT_pos/20260911_130547_284_P01/pose_before_render.csv')
t_gt = df_gt['timestamp_seconds'].values
gt_pos = df_gt[['right_px', 'right_py', 'right_pz']].values  # meters (Unity LHS)
gt_quats = df_gt[['right_qx', 'right_qy', 'right_qz', 'right_qw']].values # (x, y, z, w)

# Note: Unity is Left-Handed (X right, Y up, Z forward).
# In OpenCV/Camera RHS: X right, Y down, Z forward.
# Let's convert Unity coordinate frame to OpenCV-like RHS:
# pos_rhs = [x, -y, z]
# quat_rhs = [qx, -qy, qz, -qw] (or conjugated)
# To be completely general, we can also let the 3D rotation matrix R_cam_to_vr handle any arbitrary orientation/reflection.

# Let's search time offsets from 17.5s to 20.5s in 5ms steps
candidate_offsets = np.arange(17.5, 20.5, 0.005)

# For a given candidate offset, interpolate GT poses to PnP timestamps
# and compute the rotation alignment quality
results = []

for offset in candidate_offsets:
    # Target GT times for each PnP frame
    t_targets = t_pnp + offset
    
    # Valid mask where both PnP is valid and target GT time is within GT range
    mask = pnp_valid & (t_targets >= t_gt[0]) & (t_targets <= t_gt[-1])
    if np.sum(mask) < 200:
        continue
    
    t_eval = t_targets[mask]
    
    # Interpolate GT positions and quaternions
    gt_pos_interp = np.zeros((len(t_eval), 3))
    for dim in range(3):
        gt_pos_interp[:, dim] = np.interp(t_eval, t_gt, gt_pos[:, dim])
        
    # SLERP or normalized linear interp for quaternions
    gt_quat_interp = np.zeros((len(t_eval), 4))
    for dim in range(4):
        gt_quat_interp[:, dim] = np.interp(t_eval, t_gt, gt_quats[:, dim])
    gt_quat_interp /= np.linalg.norm(gt_quat_interp, axis=1, keepdims=True)
    
    # Compute relative rotation angles between pairs of steps (step size dt ~ 0.1s = 3-7 frames)
    step = 5
    pnp_m_quats = pnp_quats[mask]
    
    pnp_rot_angles = []
    gt_rot_angles = []
    
    for i in range(0, len(pnp_m_quats) - step, step):
        r1_pnp = R_scipy.from_quat(pnp_m_quats[i])
        r2_pnp = R_scipy.from_quat(pnp_m_quats[i+step])
        ang_pnp = (r1_pnp.inv() * r2_pnp).magnitude()
        
        r1_gt = R_scipy.from_quat(gt_quat_interp[i])
        r2_gt = R_scipy.from_quat(gt_quat_interp[i+step])
        ang_gt = (r1_gt.inv() * r2_gt).magnitude()
        
        pnp_rot_angles.append(ang_pnp)
        gt_rot_angles.append(ang_gt)
        
    pnp_rot_angles = np.array(pnp_rot_angles)
    gt_rot_angles = np.array(gt_rot_angles)
    
    corr = np.corrcoef(pnp_rot_angles, gt_rot_angles)[0, 1]
    rmse_ang = np.sqrt(np.mean((pnp_rot_angles - gt_rot_angles)**2))
    
    results.append((offset, corr, rmse_ang, len(pnp_rot_angles)))

df_res = pd.DataFrame(results, columns=['offset', 'corr', 'rmse_ang', 'n_pairs'])
df_res = df_res.sort_values(by='corr', ascending=False)
print("Top 10 time sync candidates by rotation step correlation:")
print(df_res.head(10).to_string(index=False))

best_offset = df_res.iloc[0]['offset']
print(f"\nOptimal time offset: {best_offset:.4f} s (Correlation: {df_res.iloc[0]['corr']:.4f})")
