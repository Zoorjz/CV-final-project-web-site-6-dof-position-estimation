# Data-Driven 6-DoF Tracking Render Dataset

**Generation Timestamp**: `20260914_174144`  
**Pipeline Mode**: `ALL`  
**Output Directory**: `C:\OneDrive_1trb\OneDrive\UTN Studies\CV\Final Proj\web-site\data\renders\data-driven\renders_20260914_173600`  

---

## 1. Video Specifications & Data-Driven Pipeline Breakdown

* **Start Frame**: `1250`
* **Frame Count**: `3000`
* **Trails Enabled**: `True` (Length: `75` frames)
* **HUD Overlay**: `Bottom-Left FPS Badge & Bottom-Right Legend`

### Calibration & Alignment Parameters
* **Time Synchronization Offset ($\Delta t$)**: `+18.9700 s` ($t_{\text{GT}} = t_{\text{video}} + 18.9700\text{ s}$)
* **Camera Extrinsics in VR World ($T_{\text{cam}\to\text{vr}}$)**:
  * Translation: `[1.8089, -0.0725, 1.4789] m`
  * Euler Angles (XYZ): `[-20.09°, -2.15°, -3.42°]`
* **Controller-to-Bar Offset ($T_{\text{ctrl}\to\text{bar}}$)**:
  * Translation: `[23.24, -7.10, -1.51] mm`
  * Euler Angles (XYZ): `[-90.46°, -18.59°, -90.78°]`

### Pipeline Architectures

#### A. CNN Pipeline (Direct Deep Neural Pose Estimation @ 5 FPS)
1. **Ambient Visual Context (10,000 µs)**: 10,000 µs full-frame ambient illumination capturing the tracking environment.
2. **Short-Shutter Tensor Input (1,000 µs)**: 1,000 µs high-contrast dark IR frame passed directly to convolutional layers.
3. **CNN Direct 6-DoF Pose**: Temporal-3 ResNet regressing rigid body pose $[\mathbf{t}, \mathbf{q}]$ directly (5 FPS inference on Raspberry Pi 4B).
4. **CNN + Kalman Filter**: Causal 6-DoF state estimator (kinematic CV position + $\mathrm{SO}(3)$ geodesic orientation filter) smoothing 5 Hz neural predictions.

#### B. ML Pipeline (Learned 2D-to-3D Feature Regressor @ 25 FPS)
1. **Ambient Visual Context (10,000 µs)**: 10,000 µs ambient visual stream.
2. **Short-Shutter Tensor Input (1,000 µs)**: 1,000 µs short-shutter frame.
3. **Optical Pre-Filtration**: Multi-stage $3\times 3$ Gaussian smoothing, dynamic peak threshold ($T \ge 180$), and morphological opening.
4. **2D Blob Extraction**: Sub-pixel moments extracting 2D centroid coordinates and spatial features (25 FPS on Pi 4B).
5. **ML Regressor Output**: Trained Random Forest / MLP feature regression predicting 6-DoF pose (25 FPS on Pi 4B).
6. **ML + Kalman Filter**: 6-DoF causal state estimator smoothing regression output.

### 6-DoF Kalman Filter Hyperparameters
* **Position Measurement Noise ($\sigma_{\text{pos}}$)**: `15.0 mm`
* **Acceleration Process Noise ($\sigma_{\text{accel}}$)**: `500.0 mm/s²`
* **Orientation Time Constant ($\tau_{\text{rot}}$)**: `0.06 s`

---

## 2. Generated Video Files Matrix

| Filename | Shutter / Stream | Pipeline Stage & Visual Layers Included | Duration | Frame Count | FPS HUD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `01_bright_raw.mp4` | 10000 µs (Bright Visual) | Raw Footage (No Overlay) | 83.34 s | 3000 frames | `75.0 FPS` |
| `02_bright_gt.mp4` | 10000 µs (Bright Visual) | VR Ground Truth 3D Pose (Bright Green/Yellow/Orange Axes & Trail) | 83.34 s | 3000 frames | `75.0 FPS` |
| `03_dark_raw.mp4` | 1000 µs (Dark IR) | Raw Footage (No Overlay) | 83.34 s | 3000 frames | `75.0 FPS` |
| `04_dark_gt.mp4` | 1000 µs (Dark IR) | VR Ground Truth 3D Pose (Bright Green/Yellow/Orange Axes & Trail) | 83.34 s | 3000 frames | `75.0 FPS` |
| `05_cnn_raw.mp4` | 1000 µs (Dark IR) | Dark Video + Raw CNN Direct 6-DoF Pose (Orchid Axes & Violet Trail) | 83.34 s | 3000 frames | `75.0 FPS` |
| `06_cnn_gt.mp4` | 1000 µs (Dark IR) | Dark Video + Raw CNN Direct 6-DoF Pose (Orchid Axes & Violet Trail) + VR Ground Truth 3D Pose | 83.34 s | 3000 frames | `75.0 FPS` |
| `07_cnn_kf_raw.mp4` | 1000 µs (Dark IR) | Dark Video + Kalman Filtered CNN 6-DoF Pose (Deep Violet Axes & Indigo Trail) | 83.34 s | 3000 frames | `75.0 FPS` |
| `08_cnn_kf_gt.mp4` | 1000 µs (Dark IR) | Dark Video + Kalman Filtered CNN 6-DoF Pose (Deep Violet Axes & Indigo Trail) + VR Ground Truth 3D Pose | 83.34 s | 3000 frames | `75.0 FPS` |
| `05_dark_filtration.mp4` | 1000 µs (Dark IR) | Filtered Stream (Isolated LEDs, No Markers) | 83.34 s | 3000 frames | `75.0 FPS` |
| `06_dark_filtration_gt.mp4` | 1000 µs (Dark IR) | Filtered Stream (No Markers) + VR Ground Truth 3D Pose | 83.34 s | 3000 frames | `75.0 FPS` |
| `07_dark_blobs.mp4` | 1000 µs (Dark IR) | Filtered Stream + 2D Detected Blobs (Cyan Candidates & Red Tracking) | 83.34 s | 3000 frames | `75.0 FPS` |
| `08_dark_blobs_gt.mp4` | 1000 µs (Dark IR) | Filtered Stream + 2D Detected Blobs (Cyan Candidates & Red Tracking) + VR Ground Truth 3D Pose | 83.34 s | 3000 frames | `75.0 FPS` |
| `09_ml_raw.mp4` | 1000 µs (Dark IR) | Filtered Backdrop + Raw ML Feature Regressor 6-DoF Pose (Coral Axes & Amber Trail) | 83.34 s | 3000 frames | `75.0 FPS` |
| `10_ml_gt.mp4` | 1000 µs (Dark IR) | Filtered Backdrop + Raw ML Feature Regressor 6-DoF Pose (Coral Axes & Amber Trail) + VR Ground Truth 3D Pose | 83.34 s | 3000 frames | `75.0 FPS` |
| `11_ml_kf_raw.mp4` | 1000 µs (Dark IR) | Filtered Backdrop + Kalman Filtered ML 6-DoF Pose (Tangerine Axes & Canary Gold Trail) | 83.34 s | 3000 frames | `75.0 FPS` |
| `12_ml_kf_gt.mp4` | 1000 µs (Dark IR) | Filtered Backdrop + Kalman Filtered ML 6-DoF Pose (Tangerine Axes & Canary Gold Trail) + VR Ground Truth 3D Pose | 83.34 s | 3000 frames | `75.0 FPS` |

---

## 3. Visual Layer & Distinct Color Conventions

* **Ground Truth (VR Controller)**: Bright Green X, Yellow Y, Orange Z Axes (`RGB: 0, 255, 0`) | Vibrant Bright Green Trail (`RGB: 0, 255, 0`).
* **CNN Raw (Direct Neural)**: Orchid X, Mint Y, Violet Z Axes | Soft Lavender Violet Trail (`RGB: 180, 80, 230`).
* **CNN + Kalman Filter**: Electric Violet X, Mint Y, Lavender Z Axes | Bright Electric Indigo Trail (`RGB: 140, 100, 255`).
* **ML Raw (Feature Regressor)**: Coral X, Chartreuse Y, Amber Gold Z Axes | Warm Amber Gold Trail (`RGB: 255, 180, 50`).
* **ML + Kalman Filter**: Tangerine X, SpringGreen Y, Pure Gold Z Axes | Vibrant Canary Yellow Trail (`RGB: 255, 215, 0`).
* **Filtered Optical Stream**: Ambient noise suppressed, leaving only true optical LED emissions with soft clean glow.
* **Cyan Circles**: Sub-pixel detected 2D candidate LED centroids.
* **Red Circles**: The 4 optical LED blobs selected for tracking.
