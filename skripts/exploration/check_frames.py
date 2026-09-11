import cv2
import pandas as pd
import numpy as np
import yaml

video_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_video_20260911_170556_290626.mkv'
ts_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_timestamps_20260911_170556_290626.csv'
all_frames_path = 'data/GT_videos/dataset_20260911_170556_290626/dataset_all_frames_20260911_170556_290626.csv'

df_ts = pd.read_csv(ts_path)
df_all = pd.read_csv(all_frames_path)

cap = cv2.VideoCapture(video_path)
actual_frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    actual_frame_count += 1
cap.release()

print(f"Video actual readable frames: {actual_frame_count}")
print(f"Timestamps CSV row count: {len(df_ts)}")
print(f"Timestamps output_frame_index max: {df_ts['output_frame_index'].max()}, min: {df_ts['output_frame_index'].min()}")
print(f"All frames CSV row count: {len(df_all)}")
print("Stream types in all_frames:", df_all['stream'].value_counts().to_dict() if 'stream' in df_all.columns else "N/A")
