# Tracking Video Render Dataset

**Generation Timestamp**: `20260911_192034`  
**Output Directory**: `C:\OneDrive_1trb\OneDrive\UTN Studies\CV\Final Proj\web-site\data\renders_20260911_192034`  

---

## 1. Video Specifications & Settings

* **Start Frame**: `2131`
* **Frame Count**: `200`
* **Trails Enabled**: `True` (Length: `35` frames)
* **Text / HUD Overlay**: `Disabled (Clean Visuals)`

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

| Filename | Shutter / Stream | Visual Layers Included | Duration | Frame Count |
| :--- | :--- | :--- | :--- | :--- |
| `01_dark_raw.mp4` | 1000 µs (Dark IR) | Raw Footage (No Overlay) | 5.56 s | 200 frames |
| `02_dark_gt.mp4` | 1000 µs (Dark IR) | VR Ground Truth 3D Pose (Magenta/Yellow) + 3D Trajectory Trails | 5.56 s | 200 frames |
| `03_dark_blobs.mp4` | 1000 µs (Dark IR) | 2D Blob Centroids (Green) | 5.56 s | 200 frames |
| `04_dark_pnp.mp4` | 1000 µs (Dark IR) | PnP 3D Pose (RGB) & Reprojections (Cyan) + 3D Trajectory Trails | 5.56 s | 200 frames |
| `05_dark_pnp_gt.mp4` | 1000 µs (Dark IR) | PnP 3D Pose (RGB) & Reprojections (Cyan) + VR Ground Truth 3D Pose (Magenta/Yellow) + 3D Trajectory Trails | 5.56 s | 200 frames |
| `06_bright_raw.mp4` | 10000 µs (Bright Visual) | Raw Footage (No Overlay) | 5.56 s | 200 frames |
| `07_bright_gt.mp4` | 10000 µs (Bright Visual) | VR Ground Truth 3D Pose (Magenta/Yellow) + 3D Trajectory Trails | 5.56 s | 200 frames |

---

## 3. Visual Layer Color Conventions

* **Green Rings / Dots**: Sub-pixel detected 2D centroids of optical IR LEDs.
* **RGB Coordinate Axes**: Optical PnP 6-DoF rigid body pose (+X: Red, +Y: Green, +Z: Blue).
* **Cyan Rings**: 3D Optical marker positions reprojected into camera view via PnP pose.
* **Cyan Trail**: 3D motion history trail of PnP tracked bar origin.
* **Magenta / Yellow / Orange Axes**: Transformed VR Ground Truth (Right Controller) 6-DoF pose (+X: Magenta, +Y: Yellow, +Z: Orange).
* **Magenta Rings & Crosshairs**: 3D Optical marker positions reprojected from VR Ground Truth pose into camera view.
* **Magenta Trail**: 3D motion history trail of VR Ground Truth bar origin.
