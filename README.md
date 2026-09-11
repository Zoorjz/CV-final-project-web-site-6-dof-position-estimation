# 6-DoF Rigid Body Tracking — Web Presentation & Comparison Platform

An interactive, data-driven research presentation and dual-pipeline synchronized video comparison tool for evaluating **6-DoF Rigid Body Optical Tracking (Classical PnP / EKF)** against **Data-Driven Deep Learning Pipelines (CNN / ML)**.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Node.js**: v18+ (tested on Node v22.17.0)
- **Python**: 3.10+ (with OpenCV, NumPy, SciPy, Pandas, PyYAML)
- **FFmpeg**: Required for automated browser-compatible H.264 video rendering

### 2. Install & Launch Web Application
```bash
# Install dependencies
npm install

# Start Vite development server
npm run dev
```
Open **`http://localhost:5173/`** in your browser.

### 3. (Optional) Generate New Dataset Renders
To re-render tracking videos across all 12 pipeline combinations:
```bash
# Render 500 frames starting at frame 2200 into data/renders/renders_YYYYMMDD_HHMMSS/
python skripts/render_aligned_video.py --start 2200 --count 500
```
> [!NOTE]
> The website server automatically detects and serves the newest timestamped `renders_YYYYMMDD_HHMMSS` directory inside `data/renders/`.

---

## 🌟 Key Features

### 1. Data-Driven Presentation & Report
- **Modular JS Configuration**: All report content, figures, and benchmark metrics are defined in [`src/config/reportContent.js`](./src/config/reportContent.js).
- **Responsive Alternating Layouts**:
  - `hero`: Title banner with real calibration error statistics (**18.84 mm** position, **3.55°** angular, $\Delta t = +18.97\text{ s}$).
  - `image-right` / `image-left`: Two-column split grids combining narrative text with research figures.
  - `image-bottom`: Full-width visual analysis card displaying multi-axis error and trajectory distributions.

### 2. Dual Synchronized Video Comparison
- **Side-by-Side Dual Players**:
  - **Left Player (3D Classical Pipeline)**: 6 discrete stages (*Regular Video*, *Short Shutter Video*, *Filtration*, *Blob Detection*, *PnP*, *EKF*).
  - **Right Player (Data Driven Pipeline)**: Dropdown switcher supporting **CNN** (3 stages) and **ML** (5 stages).
- **Interactive Discrete Timeline Sliders**: Clickable tick marks for every stage; dynamically reveals the stage title, exposure pill, description, and status notices.
- **Bi-Directional Video Sync**:
  - Master Play/Pause with Spacebar shortcut.
  - Sub-frame continuous drift correction loop (150ms interval).
  - Frame-accurate stepping (`-1 Fr` / `+1 Fr` at ~36 fps).
  - Multi-speed playback (`0.25x`, `0.5x`, `1.0x`, `1.5x`, `2.0x`).
  - Seamless state preservation across stage, dropdown, and Ground Truth switches.
- **Global Ground Truth Toggle**: Single switch simultaneously enables the Meta Quest VR Ground Truth 6-DoF pose overlay across both pipelines.

---

## 📂 Project Structure

```
web-site/
├── README.md                            # General project & user documentation
├── AGENTS.md                            # Deep architectural guide for developers & AI coding agents
├── index.html                           # Semantic HTML5 application entry point
├── package.json                         # Node dependencies & Vite build scripts
├── vite.config.js                       # Vite server with dynamic dataset discovery & range streaming
├── src/
│   ├── main.js                          # Application bootstrap
│   ├── config/
│   │   ├── reportContent.js             # Data-driven presentation blocks (text, figures, stats)
│   │   └── videoPipelineConfig.js       # Pipeline stages, video mapping, and fallback metadata
│   ├── components/
│   │   ├── PresentationRenderer.js      # Dynamic report presentation block generator
│   │   ├── VideoPlayer.js               # Discrete stage slider & video player component
│   │   └── VideoSyncController.js       # Dual-video synchronization, timeline & drift controller
│   ├── utils/
│   │   └── videoPreloader.js            # Video track preloader for instant transitions
│   └── styles/
│       ├── main.css                     # Academic design tokens, typography, and base layout
│       ├── presentation.css             # Presentation card grids and figure styling
│       ├── comparison.css               # Dual video player, slider ticks, and control panel
│       └── components.css               # Buttons, switches, and helper utilities
├── data/
│   ├── renders/                         # Active dataset renders (e.g. renders_20260911_233706)
│   ├── GT_pos/                          # VR Ground Truth trajectories (.csv)
│   ├── GT_videos/                       # Dual-exposure raw optical footage (.mkv / .csv)
│   ├── camera_calibration.yaml          # Camera intrinsic matrix & distortion coefficients
│   ├── geometry.yaml                    # 3D rigid body optical marker coordinates
│   └── alignment_calibration.json       # Spatial extrinsics & temporal synchronization offset
└── skripts/
    ├── render_aligned_video.py          # 12-layer video rendering & H.264 conversion engine
    ├── align_gt_and_pnp.py              # Multi-sensor spatial/temporal calibration solver
    └── pnp_ekf_pipeline.py              # PnP pose extraction & EKF smoothing algorithm
```

---

## 🎬 Video Dataset & Processing Pipeline

The dataset in `data/renders/` includes 12 synchronized render streams for each sequence:

| Filename | Shutter | Pipeline Stage Description |
| :--- | :--- | :--- |
| `01_dark_raw.mp4` | 1000 µs | Raw optical footage (no overlays) |
| `02_dark_gt.mp4` | 1000 µs | Raw dark footage + VR Ground Truth 6-DoF pose |
| `03_dark_filtration.mp4` | 1000 µs | Optical pre-filtration (noise eliminated, pure LEDs, no markers) |
| `04_dark_filtration_gt.mp4` | 1000 µs | Filtered optical footage + VR Ground Truth 6-DoF pose |
| `05_dark_blobs.mp4` | 1000 µs | Filtered backdrop + 2D sub-pixel centroids (Cyan candidates & Red PnP) |
| `06_dark_blobs_gt.mp4` | 1000 µs | Filtered backdrop + 2D blobs + VR Ground Truth 6-DoF pose |
| `07_dark_pnp.mp4` | 1000 µs | Filtered backdrop + SQPnP 3D pose (RGB axes) & cyan trajectory trail |
| `08_dark_pnp_gt.mp4` | 1000 µs | Filtered backdrop + Raw PnP 3D pose + VR Ground Truth 3D pose |
| `09_dark_ekf.mp4` | 1000 µs | Filtered backdrop + 13-State EKF Smoothed 3D pose & bright aqua trail |
| `10_dark_ekf_gt.mp4` | 1000 µs | Filtered backdrop + EKF Smoothed 3D pose + VR Ground Truth 3D pose |
| `11_bright_raw.mp4` | 10000 µs | Ambient visual raw footage (no overlays) |
| `12_bright_gt.mp4` | 10000 µs | Ambient visual raw footage + VR Ground Truth 6-DoF pose |

---

## 🛠️ Modifying & Extending the Platform

### Adding a Report Section
Edit [`src/config/reportContent.js`](./src/config/reportContent.js) and append an item to `reportBlocks`:
```javascript
{
  id: "error-breakdown",
  layout: "image-right", // Options: "image-right", "image-left", "image-bottom"
  tag: "Error Analysis",
  title: "Occlusion & Robustness Evaluation",
  content: ["Description text..."],
  highlights: ["Takeaway 1", "Takeaway 2"],
  image: {
    src: "/assets/my_figure.png",
    caption: "Figure X: Analysis caption."
  }
}
```

### Adding a New Model or Pipeline Stage
Edit [`src/config/videoPipelineConfig.js`](./src/config/videoPipelineConfig.js):
- Add your stage object to `PIPELINE_CONFIG.pnp3d.stages` or under `PIPELINE_CONFIG.dataDriven.modes`.
- Map the `raw` and `gt` filenames located in `data/renders/`.

---

## 📄 License & Credits
Developed as part of the **Computer Vision Final Project — UTN Studies (2026)**.
Built with Vite, Vanilla JavaScript, HTML5 Video, and CSS custom properties.
