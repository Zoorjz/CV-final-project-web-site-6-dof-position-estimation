import cv2

cap_10k = cv2.VideoCapture('data/GT_videos/dataset_20260911_170556_290626/dataset_10000us_video_20260911_170556_290626.mkv')
cap_1k = cv2.VideoCapture('data/GT_videos/dataset_20260911_170556_290626/dataset_1000us_video_20260911_170556_290626.mkv')

for i in range(10):
    ret1, f1 = cap_10k.read()
    ret2, f2 = cap_1k.read()
    if i == 5:
        cv2.imwrite('skripts/debug_output/sample_10000us.png', f1)
        cv2.imwrite('skripts/debug_output/sample_1000us.png', f2)
        print(f"10000us frame 5: shape={f1.shape}, min={f1.min()}, max={f1.max()}, mean={f1.mean():.1f}")
        print(f"1000us frame 5: shape={f2.shape}, min={f2.min()}, max={f2.max()}, mean={f2.mean():.1f}")

cap_10k.release()
cap_1k.release()
