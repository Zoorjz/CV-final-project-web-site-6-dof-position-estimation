import json
import pandas as pd
from datetime import datetime, timezone

# 1. Video ISO timestamp of first frame (PTS=0)
df_ts = pd.read_csv('data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_timestamps_20260911_170556_290626.csv')
video_iso_0 = df_ts['frame_time_iso8601'].iloc[0]
dt_video_0 = datetime.fromisoformat(video_iso_0).astimezone(timezone.utc)
print("Video frame 0 UTC:", dt_video_0.isoformat())

# 2. GT session.json
with open('data/GT_pos/20260911_130547_284_P01/session.json') as f:
    sess = json.load(f)

utc_start_str = sess['utcStart']
dt_gt_start = datetime.fromisoformat(utc_start_str.replace('Z', '+00:00'))
unity_anchor_sec = sess['unityRealtimeAnchorSeconds']
print("GT session utcStart:", dt_gt_start.isoformat())
print("GT unityRealtimeAnchorSeconds:", unity_anchor_sec)

delta_utc_sec = (dt_video_0 - dt_gt_start).total_seconds()
expected_gt_timestamp_for_video_0 = unity_anchor_sec + delta_utc_sec
print(f"Delta UTC (Video_0 - GT_start): {delta_utc_sec:.4f} s")
print(f"Expected GT timestamp_seconds for video frame 0: {expected_gt_timestamp_for_video_0:.4f} s")
print(f"Expected time offset (gt_timestamp - video_pts_sec): {expected_gt_timestamp_for_video_0:.4f} s")
