import pandas as pd
import numpy as np

# Load video timestamps
df_ts = pd.read_csv('data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_timestamps_20260911_170556_290626.csv')
df_gt = pd.read_csv('data/GT_pos/20260911_130547_284_P01/pose_before_render.csv')

print("Video timestamps columns:", df_ts.columns.tolist())
print(df_ts[['output_frame_index', 'sensor_frame_index', 'sensor_timestamp', 'pts_ms', 'elapsed_ms', 'frame_unix_ms']].head(10))

# Check sensor_timestamp diffs:
# sensor_timestamp in nanoseconds or ticks?
sensor_ts = df_ts['sensor_timestamp'].values
diff_sensor_ns = np.diff(sensor_ts)
print(f"\nSensor timestamp: min={sensor_ts[0]}, max={sensor_ts[-1]}, total elapsed (ns)={sensor_ts[-1] - sensor_ts[0]}")
print(f"Total sensor time: {(sensor_ts[-1] - sensor_ts[0])/1e9:.3f} s")
print(f"Total pts_ms time: {(df_ts['pts_ms'].iloc[-1] - df_ts['pts_ms'].iloc[0])/1000.0:.3f} s")
print(f"Total elapsed_ms time: {(df_ts['elapsed_ms'].iloc[-1] - df_ts['elapsed_ms'].iloc[0])/1000.0:.3f} s")
print(f"Total frame_unix_ms time: {(df_ts['frame_unix_ms'].iloc[-1] - df_ts['frame_unix_ms'].iloc[0])/1000.0:.3f} s")

# Check GT timestamps
gt_ts = df_gt['timestamp_seconds'].values
print(f"\nGT timestamp_seconds: min={gt_ts[0]}, max={gt_ts[-1]}, total elapsed={gt_ts[-1] - gt_ts[0]:.3f} s")
print(f"GT rows: {len(gt_ts)}, median dt: {np.median(np.diff(gt_ts))*1000.0:.3f} ms (~{1.0/np.median(np.diff(gt_ts)):.1f} Hz)")
print(f"GT unity_frame: min={df_gt['unity_frame'].iloc[0]}, max={df_gt['unity_frame'].iloc[-1]}, total frames={df_gt['unity_frame'].iloc[-1] - df_gt['unity_frame'].iloc[0]}")

# Check if there are sequences or pauses in GT
if 'sequence_id' in df_gt.columns:
    print("GT sequences:", df_gt['sequence_id'].value_counts().to_dict())
