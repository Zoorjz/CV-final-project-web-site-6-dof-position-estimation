import pandas as pd
import numpy as np

df_10k = pd.read_csv('data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_timestamps_20260911_170556_290626.csv')
df_1k = pd.read_csv('data/GT_videos/dataset_20260911_170556_290626/dataset_1000us_timestamps_20260911_170556_290626.csv')

print("10000us timestamps head:")
print(df_10k[['output_frame_index', 'sensor_frame_index', 'pts_ms', 'elapsed_ms', 'mean_luma']].head())

print("\n1000us timestamps head:")
print(df_1k[['output_frame_index', 'sensor_frame_index', 'pts_ms', 'elapsed_ms', 'mean_luma']].head())
