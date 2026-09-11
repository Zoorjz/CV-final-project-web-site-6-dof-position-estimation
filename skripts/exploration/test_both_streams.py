import cv2
import pandas as pd
import numpy as np
import yaml
import itertools
import time

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

def detect_blobs_1k(gray, thresh=50, min_area=2.0, max_area=1000.0):
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []
    for c in contours:
        M = cv2.moments(c)
        if min_area < M["m00"] < max_area:
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
            areas.append(M["m00"])
    return np.array(pts), np.array(areas)

def detect_blobs_10k(gray, thresh=220, min_area=2.0, max_area=1000.0):
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []
    for c in contours:
        M = cv2.moments(c)
        if min_area < M["m00"] < max_area:
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
            areas.append(M["m00"])
    return np.array(pts), np.array(areas)

def solve_pose(pts, marker_pos, K, dist):
    best = None
    for perm in perms:
        obj = np.array([marker_pos[m] for m in perm])
        ok, rvec, tvec = cv2.solvePnP(obj, pts.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.T).T + tvec.ravel()
        if (cam_pts[:, 2] <= 0).any():
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - pts, axis=1).mean()
        if not np.isfinite(err) or err > 50.0:
            continue
        if best is None or err < best[2]:
            best = (rvec, tvec, err, perm)
    return best

print("Testing 1000us video (Stream B)...")
cap_1k = cv2.VideoCapture('data/GT_videos/dataset_20260911_170556_290626/dataset_1000us_video_20260911_170556_290626.mkv')
errors_1k = []
valid_1k = 0
total_1k = 0
while True:
    ret, frame = cap_1k.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts, areas = detect_blobs_1k(gray)
    if len(pts) == 4:
        res = solve_pose(pts, marker_pos, K, dist)
        if res is not None:
            valid_1k += 1
            errors_1k.append(res[2])
    elif len(pts) > 4:
        top_pts = pts[np.argsort(areas)[::-1][:4]]
        res = solve_pose(top_pts, marker_pos, K, dist)
        if res is not None:
            valid_1k += 1
            errors_1k.append(res[2])
    total_1k += 1
cap_1k.release()
print(f"1000us: {valid_1k}/{total_1k} valid PnP solves ({valid_1k/total_1k*100:.1f}%), median reproj err: {np.median(errors_1k):.3f} px, mean: {np.mean(errors_1k):.3f} px")

print("\nTesting 10000us video (Stream A) with high threshold...")
cap_10k = cv2.VideoCapture('data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_video_20260911_170556_290626.mkv')
errors_10k = []
valid_10k = 0
total_10k = 0
while True:
    ret, frame = cap_10k.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts, areas = detect_blobs_10k(gray, thresh=240, max_area=1500)
    if len(pts) == 4:
        res = solve_pose(pts, marker_pos, K, dist)
        if res is not None:
            valid_10k += 1
            errors_10k.append(res[2])
    elif len(pts) > 4:
        top_pts = pts[np.argsort(areas)[::-1][:4]]
        res = solve_pose(top_pts, marker_pos, K, dist)
        if res is not None:
            valid_10k += 1
            errors_10k.append(res[2])
    total_10k += 1
cap_10k.release()
print(f"10000us: {valid_10k}/{total_10k} valid PnP solves ({valid_10k/total_10k*100:.1f}%), median reproj err: {np.median(errors_10k):.3f} px, mean: {np.mean(errors_10k):.3f} px")
