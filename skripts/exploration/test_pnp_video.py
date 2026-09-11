import cv2
import pandas as pd
import numpy as np
import yaml
import itertools
import time

video_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_video_20260911_170556_290626.mkv'
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

def solve_frame_pose(img_pts, marker_pos, K, dist, preferred_perm=None):
    if preferred_perm is not None:
        perm_list = [preferred_perm] + [p for p in perms if p != preferred_perm]
    else:
        perm_list = perms

    best = None
    for perm in perm_list:
        obj = np.array([marker_pos[m] for m in perm])
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.T).T + tvec.ravel()
        if (cam_pts[:, 2] <= 0).any():
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean()
        if not np.isfinite(err) or err > 50.0:
            continue
        if best is None or err < best[2]:
            best = (rvec, tvec, err, perm)
        if preferred_perm is not None and perm == preferred_perm and err < 2.0:
            break
    return best

cap = cv2.VideoCapture(video_path)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Testing PnP on video: {n_frames} frames")

blob_counts = []
pnp_successes = []
reproj_errors = []
poses = []

t0 = time.time()
frame_idx = 0
prev_perm = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts, areas = detect_blobs(gray)
    n_blobs = len(pts)
    blob_counts.append(n_blobs)

    pose = None
    if n_blobs >= 4:
        top_pts = pts[np.argsort(areas)[::-1][:4]]
        res = solve_frame_pose(top_pts.astype(np.float64), marker_pos, K, dist, preferred_perm=prev_perm)
        if res is not None:
            rvec, tvec, err, perm = res
            prev_perm = perm
            pose = (rvec.ravel(), tvec.ravel(), err)
            reproj_errors.append(err)
            pnp_successes.append(frame_idx)
        else:
            prev_perm = None
    else:
        prev_perm = None

    poses.append(pose)
    frame_idx += 1
    if frame_idx % 1000 == 0:
        print(f"Processed {frame_idx}/{n_frames} frames ({time.time()-t0:.1f}s), valid PnP so far: {len(pnp_successes)}")

cap.release()
print(f"Finished {frame_idx} frames in {time.time()-t0:.2f}s")
print(f"Total valid PnP poses: {len(pnp_successes)} / {frame_idx} ({len(pnp_successes)/frame_idx*100:.1f}%)")
if reproj_errors:
    print(f"Median reprojection error: {np.median(reproj_errors):.3f} px, Mean: {np.mean(reproj_errors):.3f} px")
