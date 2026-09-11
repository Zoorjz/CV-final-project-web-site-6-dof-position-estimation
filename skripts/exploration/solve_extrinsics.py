import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares
import cv2

# 1. Load PnP and GT
df_pnp = pd.read_csv('skripts/pnp_trajectory_extracted.csv')
pnp_valid = df_pnp['valid'].values
t_pnp = df_pnp['time_s'].values
pnp_quats = df_pnp[['qx', 'qy', 'qz', 'qw']].values  # (x, y, z, w)
pnp_tvecs = df_pnp[['tx_mm', 'ty_mm', 'tz_mm']].values / 1000.0  # in meters

df_gt = pd.read_csv('data/GT_pos/20260911_130547_284_P01/pose_before_render.csv')
t_gt = df_gt['timestamp_seconds'].values
gt_pos_raw = df_gt[['right_px', 'right_py', 'right_pz']].values
gt_quat_raw = df_gt[['right_qx', 'right_qy', 'right_qz', 'right_qw']].values

# Convert Unity LHS to RHS (y -> -y)
S = np.diag([1.0, -1.0, 1.0])

def unity_to_rhs_pose(pos, quat):
    pos_rhs = pos @ S
    R_lhs = R_scipy.from_quat(quat).as_matrix()
    R_rhs = S @ R_lhs @ S
    # Ensure det(R) = +1
    if np.linalg.det(R_rhs) < 0:
        R_rhs = -R_rhs
    return pos_rhs, R_rhs

gt_pos_rhs = np.zeros_like(gt_pos_raw)
gt_R_rhs = []
for i in range(len(df_gt)):
    p_r, R_r = unity_to_rhs_pose(gt_pos_raw[i], gt_quat_raw[i])
    gt_pos_rhs[i] = p_r
    gt_R_rhs.append(R_r)
gt_R_rhs = np.array(gt_R_rhs)

time_offset = 18.990
t_targets = t_pnp + time_offset

mask = pnp_valid & (t_targets >= t_gt[0] + 0.5) & (t_targets <= t_gt[-1] - 0.5)
indices = np.where(mask)[0]
print(f"Number of synchronized valid frames: {len(indices)}")

t_sync = t_targets[indices]

# Interpolate GT to synchronized timestamps
gt_p_sync = np.zeros((len(indices), 3))
gt_R_sync = []

for dim in range(3):
    gt_p_sync[:, dim] = np.interp(t_sync, t_gt, gt_pos_rhs[:, dim])

# Interpolate rotations using SLERP
gt_quats_rhs = R_scipy.from_matrix(gt_R_rhs).as_quat()
# unflip quaternions for smooth interpolation
for i in range(1, len(gt_quats_rhs)):
    if np.dot(gt_quats_rhs[i], gt_quats_rhs[i-1]) < 0:
        gt_quats_rhs[i] = -gt_quats_rhs[i]

gt_q_sync = np.zeros((len(indices), 4))
for dim in range(4):
    gt_q_sync[:, dim] = np.interp(t_sync, t_gt, gt_quats_rhs[:, dim])
gt_q_sync /= np.linalg.norm(gt_q_sync, axis=1, keepdims=True)
gt_R_sync = R_scipy.from_quat(gt_q_sync).as_matrix()

pnp_p_sync = pnp_tvecs[indices]
pnp_R_sync = R_scipy.from_quat(pnp_quats[indices]).as_matrix()

# Now solve for X = (R_X, t_X) and Y = (R_Y, t_Y) such that:
# T_cam_to_bar = X * T_vr_to_ctrl * Y
# where:
# T_cam_to_bar = (R_pnp, p_pnp)
# T_vr_to_ctrl = (R_gt, p_gt)
#
# So predicted PnP pose:
# R_pred = R_X @ R_gt @ R_Y
# p_pred = R_X @ (R_gt @ t_Y + p_gt) + t_X

def loss_func(params):
    rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = params
    R_X = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
    t_X = np.array([tx, ty, tz])
    
    R_Y = R_scipy.from_rotvec([ox, oy, oz]).as_matrix()
    t_Y = np.array([bx, by, bz])
    
    # Subsample for fast optimization (every 5th frame)
    sub = slice(0, len(indices), 5)
    
    # Predicted rotations
    # R_pred = R_X @ R_gt @ R_Y
    R_pred = np.einsum('ij,njk,kl->nil', R_X, gt_R_sync[sub], R_Y)
    
    # Rotation error: R_pnp.T @ R_pred -> rotvec
    R_diff = np.einsum('nji,njk->nik', pnp_R_sync[sub], R_pred)
    rot_err = R_scipy.from_matrix(R_diff).as_rotvec()  # (N, 3)
    
    # Predicted positions
    # p_pred = R_X @ (R_gt @ t_Y + p_gt) + t_X
    p_pred = np.einsum('ij,nj->ni', R_X, np.einsum('nij,j->ni', gt_R_sync[sub], t_Y) + gt_p_sync[sub]) + t_X
    pos_err = pnp_p_sync[sub] - p_pred # (N, 3) in meters
    
    # Combine residuals (position in meters * 2.0 to balance with radians)
    return np.concatenate([rot_err.ravel(), (pos_err * 2.0).ravel()])

# Initialize params
init_params = np.zeros(12)
# Initial guess for camera-to-world: roughly face forward/down
res = least_squares(loss_func, init_params, method='lm')

print("Optimization converged:", res.success)
opt_params = res.x
rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz = opt_params

R_X = R_scipy.from_rotvec([rx, ry, rz])
t_X = np.array([tx, ty, tz])
R_Y = R_scipy.from_rotvec([ox, oy, oz])
t_Y = np.array([bx, by, bz])

print(f"\nCamera in VR World (X = T_cam_to_vr):")
print(f"  Translation (m): {t_X}")
print(f"  Euler angles (deg): {R_X.as_euler('xyz', degrees=True)}")

print(f"\nController to Bar Offset (Y = T_ctrl_to_bar):")
print(f"  Translation (m): {t_Y} (i.e. {t_Y*1000} mm)")
print(f"  Euler angles (deg): {R_Y.as_euler('xyz', degrees=True)}")

# Evaluate residuals across ALL synchronized frames
R_pred_all = np.einsum('ij,njk,kl->nil', R_X.as_matrix(), gt_R_sync, R_Y.as_matrix())
p_pred_all = np.einsum('ij,nj->ni', R_X.as_matrix(), np.einsum('nij,j->ni', gt_R_sync, t_Y) + gt_p_sync) + t_X

pos_errors_mm = np.linalg.norm(pnp_p_sync - p_pred_all, axis=1) * 1000.0

R_diff_all = np.einsum('nji,njk->nik', pnp_R_sync, R_pred_all)
rot_errors_deg = np.degrees(R_scipy.from_matrix(R_diff_all).magnitude())

print(f"\nAlignment Quality over {len(indices)} frames:")
print(f"  Position Error (mm): Median = {np.median(pos_errors_mm):.2f} mm, Mean = {np.mean(pos_errors_mm):.2f} mm, RMSE = {np.sqrt(np.mean(pos_errors_mm**2)):.2f} mm")
print(f"  Rotation Error (deg): Median = {np.median(rot_errors_deg):.2f} deg, Mean = {np.mean(rot_errors_deg):.2f} deg, RMSE = {np.sqrt(np.mean(rot_errors_deg**2)):.2f} deg")
