import cv2
import pandas as pd
import numpy as np
import yaml

# Check video
video_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_video_20260911_170556_290626.mkv'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Video: {width}x{height}, fps={fps}, count={count}')
ret, frame = cap.read()
if ret:
    print(f'Frame shape: {frame.shape}, mean_brightness={frame.mean():.2f}')
cap.release()

# Check video timestamps
ts_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_timestamps_20260911_170556_290626.csv'
df_ts = pd.read_csv(ts_path)
print(f"Video timestamps count: {len(df_ts)}, pts_ms range: {df_ts['pts_ms'].min()} .. {df_ts['pts_ms'].max()} ms")
print(f"Video duration: {(df_ts['pts_ms'].max() - df_ts['pts_ms'].min())/1000.0:.2f} s")
print(f"Video ISO range: {df_ts['frame_time_iso8601'].iloc[0]} .. {df_ts['frame_time_iso8601'].iloc[-1]}")

# Check GT pos
gt_path = 'data/GT_pos/20260911_130547_284_P01/pose_before_render.csv'
df_gt = pd.read_csv(gt_path)
print(f"GT rows: {len(df_gt)}, timestamps range: {df_gt['timestamp_seconds'].min():.3f} .. {df_gt['timestamp_seconds'].max():.3f} s ({(df_gt['timestamp_seconds'].max() - df_gt['timestamp_seconds'].min()):.2f} s duration)")
print(f"Right tracked count: {df_gt['right_is_tracked'].sum()}, Right pose valid: {df_gt['right_pose_valid'].sum()}")
