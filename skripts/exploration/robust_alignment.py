import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import least_squares
import json

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

S = np.diag([1.0, -1.0, 1.0])

def unity_to_rhs_pose(pos, quat):
    pos_rhs = pos @ S
    R_lhs = R_scipy.from_quat(quat).as_matrix()
    R_rhs = S @ R_lhs @ S
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

gt_quats_rhs = R_scipy.from_matrix(gt_R_rhs).as_quat()
for i in range(1, len(gt_quats_rhs)):
    if np.dot(gt_quats_rhs[i], gt_quats_rhs[i-1]) < 0:
        gt_quats_rhs[i] = -gt_quats_rhs[i]

def evaluate_alignment(time_offset, rx, ry, rz, tx, ty, tz, ox, oy, oz, bx, by, bz):
    t_targets = t_pnp + time_offset
    mask = pnp_valid & (t_targets >= t_gt[0] + 0.5) & (t_targets <= t_gt[-1] - 0.5)
    indices = np.where(mask)[0]
    t_sync = t_targets[indices]
    
    gt_p_sync = np.zeros((len(indices), 3))
    for dim in range(3):
        gt_p_sync[:, dim] = np.interp(t_sync, t_gt, gt_pos_rhs[:, dim])
        
    gt_q_sync = np.zeros((len(indices), 4))
    for dim in range(4):
        gt_q_sync[:, dim] = np.interp(t_sync, t_gt, gt_quats_rhs[:, dim])
    gt_q_sync /= np.linalg.norm(gt_q_sync, axis=1, keepdims=True)
    gt_R_sync = R_scipy.from_quat(gt_q_sync).as_matrix()
    
    pnp_p_sync = pnp_tvecs[indices]
    pnp_R_sync = R_scipy.from_quat(pnp_quats[indices]).as_matrix()
    
    R_X = R_scipy.from_rotvec([rx, ry, rz]).as_matrix()
    t_X = np.array([tx, ty, tz])
    R_Y = R_scipy.from_rotvec([ox, oy, oz]).as_matrix()
    t_Y = np.array([bx, by, bz])
    
    R_pred = np.einsum('ij,njk,kl->nil', R_X, gt_R_sync, R_Y)
    p_pred = np.einsum('ij,nj->ni', R_X, np.einsum('nij,j->ni', gt_R_sync, t_Y) + gt_p_sync) + t_X
    
    pos_err = pnp_p_sync - p_pred # meters
    R_diff = np.einsum('nji,njk->nik', pnp_R_sync, R_pred)
    rot_err = R_scipy.from_matrix(R_diff).as_rotvec()
    
    return indices, pos_err, rot_err, pnp_p_sync, p_pred, pnp_R_sync, R_pred

# Fine-tune time offset around 18.990 with step 1ms
best_cost = 1e9
best_res = None
best_dt = 18.990

for dt_cand in np.arange(18.90, 19.10, 0.01):
    def loss(p):
        indices, pos_err, rot_err, _, _, _, _ = evaluate_alignment(dt_cand, *p)
        sub = slice(0, len(indices), 4)
        return np.concatenate([rot_err[sub].ravel(), (pos_err[sub] * 3.0).ravel()])
    
    p0 = [-0.134, -0.057, -0.172, 1.76, -0.35, 1.45, -1.77, -0.38, -1.59, 0.02, -0.02, -0.007]
    res = least_squares(loss, p0, loss='soft_l1', f_scale=0.1, method='trf')
    cost = res.cost
    if cost < best_cost:
        best_cost = cost
        best_res = res
        best_dt = dt_cand

print(f"Optimal dt: {best_dt:.3f} s")
p_opt = best_res.x
indices, pos_err, rot_err, p_pnp, p_pred, R_pnp, R_pred = evaluate_alignment(best_dt, *p_opt)

pos_errors_mm = np.linalg.norm(pos_err, axis=1) * 1000.0
rot_errors_deg = np.degrees(np.linalg.norm(rot_err, axis=1))

print(f"Robust Solved Parameters (12 DoF + Time Sync {best_dt:.3f}s):")
print(f"Camera Extrinsics T_cam_to_vr:")
print(f"  Translation (m): {p_opt[3:6]}")
print(f"  Rotation rotvec: {p_opt[0:3]}")
print(f"  Rotation Euler (deg): {R_scipy.from_rotvec(p_opt[0:3]).as_euler('xyz', degrees=True)}")

print(f"\nController to Bar Extrinsics T_ctrl_to_bar:")
print(f"  Translation (m): {p_opt[9:12]} (in mm: {p_opt[9:12]*1000.0})")
print(f"  Rotation rotvec: {p_opt[6:9]}")
print(f"  Rotation Euler (deg): {R_scipy.from_rotvec(p_opt[6:9]).as_euler('xyz', degrees=True)}")

print(f"\nFinal Statistics over {len(indices)} frames:")
print(f"  Position Error (mm): Median = {np.median(pos_errors_mm):.1f} mm, Mean = {np.mean(pos_errors_mm):.1f} mm")
print(f"  Rotation Error (deg): Median = {np.median(rot_errors_deg):.2f} deg, Mean = {np.mean(rot_errors_deg):.2f} deg")

# Save calibration results to YAML / JSON for downstream video rendering script
calib_results = {
    "time_offset_seconds": float(best_dt),
    "camera_to_vr_extrinsics": {
        "translation_m": p_opt[3:6].tolist(),
        "rotation_rotvec": p_opt[0:3].tolist(),
        "rotation_euler_deg": R_scipy.from_rotvec(p_opt[0:3]).as_euler('xyz', degrees=True).tolist(),
        "rotation_matrix": R_scipy.from_rotvec(p_opt[0:3]).as_matrix().tolist()
    },
    "controller_to_bar_extrinsics": {
        "translation_m": p_opt[9:12].tolist(),
        "translation_mm": (p_opt[9:12]*1000.0).tolist(),
        "rotation_rotvec": p_opt[6:9].tolist(),
        "rotation_euler_deg": R_scipy.from_rotvec(p_opt[6:9]).as_euler('xyz', degrees=True).tolist(),
        "rotation_matrix": R_scipy.from_rotvec(p_opt[6:9]).as_matrix().tolist()
    },
    "metrics": {
        "num_synchronized_frames": int(len(indices)),
        "median_pos_error_mm": float(np.median(pos_errors_mm)),
        "mean_pos_error_mm": float(np.mean(pos_errors_mm)),
        "median_rot_error_deg": float(np.median(rot_errors_deg)),
        "mean_rot_error_deg": float(np.mean(rot_errors_deg))
    }
}

with open('data/alignment_calibration.json', 'w') as f:
    json.dump(calib_results, f, indent=2)
print("\nSaved calibration results to data/alignment_calibration.json")
