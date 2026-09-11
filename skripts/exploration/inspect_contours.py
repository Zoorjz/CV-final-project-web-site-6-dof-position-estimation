import cv2
import numpy as np

frame = cv2.imread('skripts/debug_output/frame_100.png')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

_, bw = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Total contours: {len(contours)}")
for i, c in enumerate(contours):
    M = cv2.moments(c)
    area = M["m00"]
    if area > 0:
        cx = M["m10"] / area
        cy = M["m01"] / area
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
        print(f"Contour {i}: area={area:8.1f}, center=({cx:6.1f}, {cy:6.1f}), circ={circ:.2f}")
