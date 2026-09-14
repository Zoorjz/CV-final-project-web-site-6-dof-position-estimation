# Tracking Video Render Dataset (Geometry-Based PnP & EKF)

**Generation Timestamp**: `20260914_165912`  
**Output Directory**: `C:\OneDrive_1trb\OneDrive\UTN Studies\CV\Final Proj\web-site\data\renders\geometry-based\renders_20260914_165911\simulated`  

---

## 1. Video Specifications & Processing Pipeline

* **Start Frame**: `2131`
* **Frame Count**: `200`
* **Trails Enabled**: `True` (Length: `75` frames)
* **HUD Overlay**: `Bottom-Left FPS Badge & Bottom-Right Legend`

### Sequential Pipeline Stages
1. **Raw Acquisition**: Unprocessed optical camera streams (1000 µs dark and 10000 µs bright).
2. **Optical Pre-Filtration**: Multi-stage filtering ($3\times 3$ Gaussian blur, dynamic peak threshold $T \ge 180$, morphological opening) completely suppresses background noise and isolates genuine LED emissions without any overlay markers.
3. **2D Blob Extraction**: High-precision sub-pixel centroid moment analysis with circularity gating ($4\pi A / P^2 > 0.25$) rendered onto the filtered stream.
4. **PnP 6-DoF Rigid Body Pose**: SQPnP pose estimation rendered with RGB 3D coordinate axes and 4 active tracking markers on the filtered optical stream (75 FPS capacity on Raspberry Pi 4B).
5. **13-State EKF Smoothing**: Real-time constant-velocity/angular-velocity filtering with tunable dynamics.

### EKF Filter Parameters
* **NIS Outlier Gating Enabled**: `False` (Confidence $p = 0.9999$)
* **Measurement Covariance Scale ($R_{\text{scale}}$)**: `1.00` (Pos Floor: `Auto (3.0mm min)`)
* **Process Dynamics Scale ($Q_{\text{scale}}$)**: `1.00` (Vel Noise: `Auto`)

### Calibration & Alignment Parameters
* **Time Synchronization Offset ($\Delta t$)**: `+18.9700 s` ($t_{\text{GT}} = t_{\text{video}} + 18.9700\text{ s}$)
* **Camera Extrinsics in VR World ($T_{\text{cam}\to\text{vr}}$)**:
  * Translation: `[1.8089, -0.0725, 1.4789] m`
  * Euler Angles (XYZ): `[-20.09°, -2.15°, -3.42°]`
* **Controller-to-Bar Offset ($T_{\text{ctrl}\to\text{bar}}$)**:
  * Translation: `[23.24, -7.10, -1.51] mm`
  * Euler Angles (XYZ): `[-90.46°, -18.59°, -90.78°]`
* **Dataset Alignment Quality**:
  * Median 3D Position Error: `18.84 mm`
  * Median 3D Rotation Error: `3.55°`

---

## 2. Generated Video Files Description

| Filename | Shutter / Stream | Pipeline Visual Layers Included | Duration | Frame Count | FPS HUD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `01_dark_raw.mp4` | 1000 µs (Dark IR) | Raw Footage (No Overlay) | 5.56 s | 200 frames | `75.0 FPS` |
| `02_dark_gt.mp4` | 1000 µs (Dark IR) | VR Ground Truth 3D Pose (Bright Green Axes & Trail) + 3D Trajectory Trails | 5.56 s | 200 frames | `75.0 FPS` |
| `03_dark_filtration.mp4` | 1000 µs (Dark IR) | Filtered Stream (Isolated LEDs, No Markers) | 5.56 s | 200 frames | `75.0 FPS` |
| `04_dark_filtration_gt.mp4` | 1000 µs (Dark IR) | Filtered Stream (No Markers) + VR Ground Truth 3D Pose | 5.56 s | 200 frames | `75.0 FPS` |
| `05_dark_blobs.mp4` | 1000 µs (Dark IR) | Filtered Stream Backdrop + 2D Detected Blobs (Cyan Candidates & Red PnP) | 5.56 s | 200 frames | `75.0 FPS` |
| `06_dark_blobs_gt.mp4` | 1000 µs (Dark IR) | Filtered Stream Backdrop + 2D Detected Blobs (Cyan Candidates & Red PnP) + VR Ground Truth 3D Pose (Bright Green Axes & Trail) + 3D Trajectory Trails | 5.56 s | 200 frames | `75.0 FPS` |
| `07_dark_pnp.mp4` | 1000 µs (Dark IR) | Filtered Stream Backdrop + Raw PnP 3D Pose (RGB Axes & Active Red Markers) + 3D Trajectory Trails | 5.56 s | 200 frames | `75.0 FPS` |
| `08_dark_pnp_gt.mp4` | 1000 µs (Dark IR) | Filtered Stream Backdrop + Raw PnP 3D Pose (RGB Axes & Active Red Markers) + VR Ground Truth 3D Pose (Bright Green Axes & Trail) + 3D Trajectory Trails | 5.56 s | 200 frames | `75.0 FPS` |
| `09_dark_ekf.mp4` | 1000 µs (Dark IR) | Filtered Stream Backdrop + EKF Smoothed 3D Pose (RGB Axes & Aqua Trail) + 3D Trajectory Trails | 5.56 s | 200 frames | `75.0 FPS` |
| `10_dark_ekf_gt.mp4` | 1000 µs (Dark IR) | Filtered Stream Backdrop + EKF Smoothed 3D Pose (RGB Axes & Aqua Trail) + VR Ground Truth 3D Pose (Bright Green Axes & Trail) + 3D Trajectory Trails | 5.56 s | 200 frames | `75.0 FPS` |
| `11_bright_raw.mp4` | 10000 µs (Bright Visual) | Raw Footage (No Overlay) | 5.56 s | 200 frames | `75.0 FPS` |
| `12_bright_gt.mp4` | 10000 µs (Bright Visual) | VR Ground Truth 3D Pose (Bright Green Axes & Trail) + 3D Trajectory Trails | 5.56 s | 200 frames | `75.0 FPS` |

---

## 3. Visual Layer Color Conventions

* **Filtered Stream Background**: Unfiltered background noise suppressed, revealing only true optical LED emissions.
* **Cyan Circles / Dots**: Sub-pixel detected 2D centroids of filtered candidate optical IR LEDs.
* **Red Circles / Dots**: The 4 optical LED blobs selected and actively used for the PnP 6-DoF pose calculation.
* **RGB Coordinate Axes (PnP / EKF)**: Optical 6-DoF rigid body pose (+X: Red, +Y: Green, +Z: Blue).
* **Cyan Trail**: 3D motion history trail of Raw PnP tracked bar origin.
* **Aqua / Emerald Trail**: 3D motion history trail of EKF smoothed bar origin.
* **Bright Green / Yellow / Orange Axes**: Transformed VR Ground Truth (Right Controller) 6-DoF pose (+X: Bright Green, +Y: Yellow, +Z: Orange).
* **Bright Green Trail**: 3D motion history trail of VR Ground Truth bar origin.
