# 6-DoF Rigid Body Tracking — Web Presentation & Comparison Platform

An interactive, data-driven research presentation and dual-pipeline synchronized video comparison tool for evaluating **6-DoF Rigid Body Optical Tracking (Classical SQPnP & 13-State EKF)** against **Data-Driven Pipelines (Deep CNN & Machine Learning Feature Regressors)**.

---

## 🚀 Quick Start & Prerequisites

### 1. Prerequisites
* **Web Frontend / Server:**
  * **Node.js**: `v18.x` or higher (tested on Node `v20` / `v22`)
  * **npm**: `v9.x` or higher (bundled with Node.js)
* **Video Rendering Scripts (Optional / for offline generation only):**
  * **Python**: `3.10+` with `opencv-python`, `numpy`, `scipy`, `pandas`, `pyyaml`, `torch`
  * **FFmpeg**: Configured with `libx264` support for automated web H.264 video encoding

---

### 2. Starting the Local Development Server

```bash
# 1. Install frontend dependencies
npm install

# 2. Start the local development server (Vite)
npm run dev
```

The terminal will launch the local development server:
```
  VITE v5.4.21  ready in 180 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```
Open **`http://localhost:5173/`** in your browser.

---

### 3. Building and Previewing the Production Bundle

To test the optimized static production build locally:

```bash
# Build the production bundle into dist/
npm run build

# Preview the built static site locally
npm run preview
```

---

## 🌐 Deploying to GitHub Pages (Option A: Automated GitHub Actions)

This repository is configured with an automated **GitHub Actions** workflow ([`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml)). Whenever you push commits to the `master` or `main` branch, GitHub Actions builds and publishes the website automatically.

### Enabling GitHub Pages in your Repository:
1. Open your GitHub repository in your browser.
2. Navigate to **Settings** $\rightarrow$ **Pages** (under "Code and automation").
3. Under **Build and deployment**:
   * Set **Source** to **`GitHub Actions`**.
4. Push your code:
   ```bash
   git add .
   git commit -m "Configure GitHub Pages deployment"
   git push origin master
   ```
5. Your website will be live in ~30 seconds at:
   `https://<your-github-username>.github.io/<repository-name>/`

---

## 🎬 How to Render New Videos & Update the Website

When you train new models, tune Kalman filters, or re-run video tracking benchmarks:

### One-Step Render & Web Sync:
Run the master orchestrator script with the `--publish-to-web` flag:

```bash
# Render all pipelines (Classical PnP/EKF + CNN + ML) for 3000 frames and sync to web assets
python skripts/render_all_pipelines.py --start 1250 --count 3000 --publish-to-web
```

### What `--publish-to-web` does:
1. Generates 28 web-optimized `.mp4` video files re-encoded in **H.264 (`yuv420p` / `+faststart`)**.
2. Automatically copies the videos into `public/data/renders/geometry-based/` and `public/data/renders/data-driven/`.
3. Updates `public/api/dataset-info.json` with new frame counts, duration, and timestamps.

### Publishing Updated Videos:
Simply commit and push:
```bash
git add public/data/renders/ public/api/dataset-info.json
git commit -m "Update benchmark videos with newly tuned EKF parameters"
git push origin master
```
GitHub Actions will automatically re-build and deploy the updated video set to GitHub Pages!

---

## 🌟 Key Features & Architecture

### 1. Data-Driven Research Presentation
* **Config-Driven Content**: All research narrative, figures, and benchmark statistics are centrally defined in [`src/config/reportContent.js`](./src/config/reportContent.js).
* **Responsive Visual Blocks**:
  * `hero`: Dynamic title header with real calibration error statistics (**18.84 mm** median 3D position error, **3.55°** angular, $\Delta t = +18.97\text{ s}$).
  * `image-right` / `image-left`: Multi-column cards pairing academic explanations with trajectory and error figures.
  * `image-bottom`: High-resolution figures displaying multi-axis residual distributions and tracking charts.

### 2. Bi-Directional Synchronized Comparison Engine
* **Side-by-Side Dual Players**:
  * **Left Player (Classical Geometry-based)**: Discrete slider stages (*Long Exposure*, *Short Exposure*, *Filtration*, *Blob Detection*, *PnP*, *EKF*).
  * **Right Player (Data-Driven)**: Interactive dropdown switcher for **CNN** (4 stages) and **ML Regressor** (6 stages).
* **Frame-Accurate Synchronization**:
  * Master Play / Pause coordination (`Spacebar` shortcut).
  * Continuous drift correction loop ($< 50\text{ ms}$ tolerance).
  * Discrete stage slider with click-to-snap navigation.
  * Frame stepper (`-1 Frame` / `+1 Frame` at ~36 fps) with `Left`/`Right` arrow keys.
  * Variable playback speed (`0.25x`, `0.5x`, `1.0x`, `1.5x`, `2.0x`).
  * State preservation across pipeline switches, slider moves, and ground truth toggles.
* **Global Ground Truth Toggle**: Single switch simultaneously displays the Meta Quest VR Ground Truth 6-DoF pose overlay on both players.

---

## 📂 Project Structure

```
web-site/
├── .github/
│   └── workflows/
│       └── deploy.yml                   # Automated GitHub Actions deployment to GitHub Pages
├── README.md                            # Setup guide and user documentation
├── AGENTS.md                            # Developer reference manual & architecture specification
├── index.html                           # Semantic HTML5 web application entry point
├── package.json                         # Node dependencies and build scripts
├── vite.config.js                       # Vite configuration with relative base './' for static hosting
├── public/                              # Static public assets (bundled directly into dist/)
│   ├── api/
│   │   └── dataset-info.json            # Static sequence metadata (frames, fps, timestamps)
│   ├── assets/                          # Alignment and calibration plot images
│   ├── images/                          # Presentation figures and animation GIFs
│   └── data/
│       └── renders/
│           ├── geometry-based/          # 12 Classical SQPnP + 13-State EKF video tracks (.mp4)
│           └── data-driven/             # 16 CNN & ML Regressor video tracks (.mp4)
├── src/
│   ├── main.js                          # Web application bootstrap orchestrator
│   ├── config/
│   │   ├── reportContent.js             # Presentation report cards configuration
│   │   └── videoPipelineConfig.js       # Discrete pipeline stages and video URL resolver
│   ├── components/
│   │   ├── PresentationRenderer.js      # Dynamic report card generator
│   │   ├── VideoPlayer.js               # Discrete tick slider & pipeline video card
│   │   └── VideoSyncController.js       # Bi-directional dual video sync & drift engine
│   ├── utils/
│   │   └── videoPreloader.js            # Video track background preloader
│   └── styles/
│       ├── main.css                     # Global reset, typography, and layout tokens
│       ├── presentation.css             # Presentation card grids and figure styling
│       ├── comparison.css               # Dual player cards, sliders, and sync control panel
│       └── components.css               # Badges, switches, and UI buttons
└── skripts/
    ├── render_all_pipelines.py          # Master renderer with --publish-to-web support
    ├── render_aligned_video.py          # Geometry-based PnP/EKF video generator
    └── visualize_predictions.py         # CNN & ML data-driven video generator
```

---

## 📄 License & Academic Reference
Developed as part of the **Computer Vision Final Project — UTN Studies (2026)**.  
Built with Vite, Vanilla JavaScript (ESM), HTML5 `<video>`, and CSS custom properties.

