import cv2
import pandas as pd
import numpy as np
import yaml
import itertools
import os

video_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_video_20260911_170556_290626.mkv'
calib_path = 'data/camera_calibration.yaml'
geom_path = 'data/geometry.yaml'

cal = yaml.safe_load(open(calib_path))
K = np.array(cal["camera_matrix"], dtype=np.float64)
K[:2, :] *= 0.5
dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

geom = yaml.safe_load(open(geom_path))
marker_pos = {m["id"]: np.array(m["position"], dtype=np.float64) for m in geom["markers"]}
marker_ids = sorted(marker_pos.keys())
perms = list(itertools.permutations(marker_ids))

THRESH = 60
MIN_BLOB_AREA = 2.0

def detect_blobs(gray, thresh=THRESH, min_area=MIN_BLOB_AREA):
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > min_area:
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
            areas.append(M["m00"])
    return np.array(pts), np.array(areas)

os.makedirs('skripts/debug_output', exist_ok=True)

cap = cv2.VideoCapture(video_path)

# Let's check frames 100, 500, 1000, 2000
target_frames = [100, 500, 1000, 1500, 2000, 2500, 3000]

for f_idx in target_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts, areas = detect_blobs(gray)
    print(f"Frame {f_idx}: detected {len(pts)} blobs with areas {areas}")
    
    # Try all methods (SQPNP, IPPE, ITERATIVE)
    if len(pts) >= 4:
        top_pts = pts[np.argsort(areas)[::-1][:4]]
        # Let's test all permutations
        best_solves = {}
        for flag_name, flag_val in [('SQPNP', cv2.SOLVEPNP_SQPNP), ('IPPE', cv2.SOLVEPNP_IPPE), ('ITERATIVE', cv2.SOLVEPNP_ITERATIVE)]:
            best_err = 1e9
            best_p = None
            for perm in perms:
                obj = np.array([marker_pos[m] for m in perm])
                try:
                    ok, rvec, tvec = cv2.solvePnP(obj, top_pts.astype(np.float64), K, dist, flags=flag_val)
                    if ok:
                        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                        err = np.linalg.norm(proj.reshape(-1, 2) - top_pts, axis=1).mean()
                        if err < best_err:
                            best_err = err
                            best_p = (rvec, tvec, perm, proj.reshape(-1, 2))
                except Exception as e:
                    pass
            best_solves[flag_name] = (best_err, best_p)
            print(f"  Frame {f_idx} [{flag_name}]: best error = {best_err:.3f} px")
            
        # Draw on frame
        vis = frame.copy()
        for i, pt in enumerate(top_pts):
            cv2.circle(vis, (int(round(pt[0])), int(round(pt[1]))), 5, (0, 255, 0), -1)
            cv2.putText(vis, f"B{i}", (int(pt[0])+8, int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        err, best_p = best_solves['SQPNP']
        if best_p is not None:
            rvec, tvec, perm, projs = best_p
            for i, (p_pt, m_id) in enumerate(zip(projs, perm)):
                cv2.circle(vis, (int(round(p_pt[0])), int(round(p_pt[1]))), 3, (0, 0, 255), -1)
                cv2.putText(vis, f"{m_id}", (int(p_pt[0])-15, int(p_pt[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.imwrite(f'skripts/debug_output/frame_{f_idx}.png', vis)

cap.release()
print("Saved debug frames to skripts/debug_output/")
