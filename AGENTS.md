# AGENTS.md — Technical Architecture & Developer Reference

This document provides a comprehensive, low-level technical specification of the **6-DoF Tracking Web Presentation & Comparison Platform**. It is designed specifically for AI coding agents and core contributors working on features, refactoring, or pipeline integrations.

---

## 1. System Overview & Technology Constraints

| Layer | Technology | Architectural Rationale |
| :--- | :--- | :--- |
| **Build & Dev Server** | Vite 5.x | Zero-config ES module bundling, instant HMR, lightweight footprint. |
| **Frontend Core** | Vanilla JavaScript (ESM) | Zero framework bloat, direct DOM control, deterministic lifecycle. |
| **Styling** | Vanilla CSS + CSS Variables | Academic design tokens, scoped component sheets, zero CSS-in-JS overhead. |
| **Video Decoding** | Native HTML5 `<video>` | Hardware-accelerated decoding, low memory footprint. |
| **Video Codec** | H.264 (`libx264` / `yuv420p`) | Mandatory for web browser video decoding (MPEG-4/FMP4 is not supported in HTML5). |
| **Backend / Middleware** | Node.js / Vite Middleware | Custom byte-range streamer (HTTP 206) & dynamic dataset resolver. |

---

## 2. Directory Structure & Key Artifacts

```
web-site/
├── README.md                            # High-level overview & user guide
├── AGENTS.md                            # This developer/agent reference manual
├── index.html                           # Semantic DOM anchors for renderer injection
├── vite.config.js                       # HTTP 206 range streamer + /api/dataset-info + dynamic folder resolver
├── package.json                         # Scripts ("dev", "build", "preview")
├── src/
│   ├── main.js                          # Bootstrap orchestrator
│   ├── config/
│   │   ├── reportContent.js             # Data-driven presentation report blocks
│   │   └── videoPipelineConfig.js       # Pipeline stages, dropdown modes, file mapping
│   ├── components/
│   │   ├── PresentationRenderer.js      # Presentation card generator
│   │   ├── VideoPlayer.js               # Pipeline video card & discrete tick slider
│   │   └── VideoSyncController.js       # Bi-directional dual video sync & drift engine
│   ├── utils/
│   │   └── videoPreloader.js            # Video track preloading & cache utility
│   └── styles/
│       ├── main.css                     # Global reset, typography, header/footer
│       ├── presentation.css             # Presentation card grids & figure styles
│       ├── comparison.css               # Dual player cards, sliders, sync panel
│       └── components.css               # Badges, toggles, utility classes
├── data/
│   └── renders/                         # Canonical folder containing timestamped subdirectories
│       └── renders_YYYYMMDD_HHMMSS/     # 12 rendered .mp4 tracks + README.md
└── skripts/
    └── render_aligned_video.py          # Python video generator with auto H.264 FFmpeg re-encoder
```

---

## 3. Core Subsystems & Invariants

### 3.1 Dynamic Dataset Discovery (`vite.config.js`)

#### The Problem
Render scripts create new timestamped directories (e.g., `data/renders/renders_20260911_233706/`). The frontend should always serve the newest rendered sequence without requiring hardcoded path edits.

#### Resolution Algorithm
1. When a request for `/data/renders/{filename}` arrives:
   - Middleware invokes `resolveLatestRendersDirectory()`.
   - Scans `data/renders/` for subdirectories prefixed with `renders_`.
   - Sorts descending lexicographically (`b.name.localeCompare(a.name)`).
   - Resolves the requested file from the newest directory.
2. Endpoint `/api/dataset-info`:
   - Returns a JSON payload with `activeDirectory`, `timestamp`, `frameCount`, `duration`, `startFrame`, `fps`, and `files`.
   - Dynamically parsed from `README.md` in the active render directory.

#### Byte-Range Streaming (HTTP 206 Partial Content)
HTML5 `<video>` scrubbing requires HTTP 206 responses with `Content-Range` and `Accept-Ranges: bytes`. `vite.config.js` implements a custom chunk streamer using `fs.createReadStream(filePath, { start, end })`.

---

### 3.2 Bi-Directional Video Synchronization (`VideoSyncController.js`)

#### Master-Follower vs Peer-to-Peer Sync
Both players operate in a **peer-synchronized mesh** coordinated by `VideoSyncController`:
- Master control buttons (Play/Pause, Step Forward/Back, Timeline Scrubber, Speed) broadcast state to both `<video>` elements.
- Intercepts native `play` and `pause` events on either video and propagates to the peer while preventing infinite re-triggering loops via `this.isSyncingInternally = true`.

#### Drift Correction Loop
```javascript
this.syncInterval = setInterval(() => {
  if (!this.isPlaying || this.isScrubbing || !this.v1 || !this.v2) return;
  const drift = Math.abs(this.v1.currentTime - this.v2.currentTime);
  if (drift > 0.05) {
    // Snap lagging or leading video to match reference
    this.v2.currentTime = this.v1.currentTime;
  }
}, 150);
```

#### State Preservation Across Transitions
When the user moves a discrete slider tick, switches the architecture dropdown (CNN $\leftrightarrow$ ML), or toggles the Ground Truth checkbox:
1. Current `currentTime` and `paused` status are captured.
2. `videoElement.src` is updated.
3. A one-time `canplay` listener restores `videoElement.currentTime = savedTime` and resumes playback if it was playing.

---

### 3.3 Video Pipeline & Stage Mapping Contract (`videoPipelineConfig.js`)

#### Data Schema
```javascript
export const PIPELINE_CONFIG = {
  pnp3d: {
    id: "pnp3d",
    title: "3D Pipeline",
    subtitle: "PnP",
    badge: "Classical Vision",
    stages: [
      {
        id: "stage_id",
        name: "Stage Name",
        shutter: "1,000 µs",
        description: "Stage explanation...",
        files: {
          raw: "01_dark_raw.mp4",
          gt: "02_dark_gt.mp4"
        },
        isPlaceholder: false, // Set to true if render is not yet available
        placeholderNote: "Render in progress..." // Optional note if placeholder
      }
    ]
  },
  dataDriven: {
    id: "dataDriven",
    title: "Data Driven Pipeline",
    defaultMode: "CNN",
    modes: {
      CNN: { id: "CNN", name: "CNN", stages: [...] },
      ML:  { id: "ML",  name: "ML",  stages: [...] }
    }
  }
};
```

#### Ground Truth & Simulation Mapping Rule
`resolveVideoUrl(stage, showGT, isSimulated = false)`:
- When `showGT == true`: returns `stage.files.gt || stage.files.raw`.
- When `showGT == false`: returns `stage.files.raw`.
- When `isSimulated == true`: maps target filename into the nested `simulated/` subdirectory (e.g. `data-driven/simulated/05_cnn_raw.mp4` or `geometry-based/simulated/07_dark_pnp.mp4`).

#### Raspberry Pi 4B Hardware FPS Emulation Contract
- **PnP / Classical Geometry**: 75.0 FPS throughput capacity (updates every camera frame @ 36 FPS camera input).
- **ML (Feature Regressor)**: 25.0 FPS throughput capacity (updates pose/blobs/trail every ~40 ms).
- **CNN (Direct Deep Pose)**: 5.0 FPS throughput capacity (updates pose/trail every ~200 ms with sample-and-hold latency & discrete stepped trail).
- **All Renders**: Include a large, high-visibility bottom-left HUD badge displaying active frame rate (e.g. `5.0 FPS`, `25.0 FPS`, `75.0 FPS`, `36.0 FPS`) with model-matched glowing status dot.

---

### 3.4 Report Presentation Blocks (`reportContent.js`)

#### Layout Types
1. `hero`: Banner layout with project badges, titles, and a 4-metric statistics grid (`stats: [{ label, value, subtext }]`).
2. `image-right`: Left column has title, paragraphs, and `highlights: [...]`; right column contains `image: { src, caption }`.
3. `image-left`: Left column contains `image: { src, caption }`; right column has title, paragraphs, and highlights.
4. `image-bottom`: Full-width visual analysis card with large figure and captions below.

---

## 4. Video Rendering Pipeline (`skripts/render_aligned_video.py`)

### 12-File Render Matrix
The script renders 12 synchronized `.mp4` video files:

```
01_dark_raw.mp4            <- Raw 1,000 µs dark IR frame
02_dark_gt.mp4             <- Raw dark IR + VR GT 3D pose
03_dark_filtration.mp4     <- Optical pre-filtered stream (clean LEDs, no markers)
04_dark_filtration_gt.mp4  <- Filtered stream + VR GT 3D pose
05_dark_blobs.mp4          <- Filtered stream + 2D detected blob rings
06_dark_blobs_gt.mp4       <- Filtered stream + 2D blobs + VR GT 3D pose
07_dark_pnp.mp4            <- Filtered stream + SQPnP 3D pose & cyan trail
08_dark_pnp_gt.mp4         <- Filtered stream + SQPnP 3D pose + VR GT 3D pose
09_dark_ekf.mp4            <- Filtered stream + 13-State EKF smoothed pose & aqua trail
10_dark_ekf_gt.mp4         <- Filtered stream + EKF smoothed pose + VR GT 3D pose
11_bright_raw.mp4          <- Raw 10,000 µs ambient visual frame
12_bright_gt.mp4           <- Raw bright visual + VR GT 3D pose
```

### Automatic Web H.264 Re-encoding
OpenCV's `cv2.VideoWriter` generates ISO MPEG-4 Part 2 files (`mp4v` / `FMP4`), which fail to decode in web browsers. `render_aligned_video.py` includes an automatic post-write FFmpeg conversion:
```python
cmd = [
    "ffmpeg", "-y", "-i", output_filename,
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    temp_h264
]
```

---

## 5. Developer & Agent Recipes

### Recipe 1: Adding a New Data-Driven Architecture (e.g. Transformer)
1. Open `src/config/videoPipelineConfig.js`.
2. In `PIPELINE_CONFIG.dataDriven.modes`, add a new key:
```javascript
Transformer: {
  id: "Transformer",
  name: "Transformer",
  fullName: "Spatial-Temporal Vision Transformer",
  badge: "Attention-Based 6-DoF Pose",
  stages: [
    {
      id: "regular_video",
      name: "Regular Video",
      shutter: "10,000 µs",
      description: "Visual token inputs.",
      files: { raw: "11_bright_raw.mp4", gt: "12_bright_gt.mp4" },
      isPlaceholder: false
    },
    {
      id: "transformer_output",
      name: "Transformer Output",
      shutter: "1,000 µs",
      description: "Attention-weighted 6-DoF output.",
      files: { raw: "09_dark_ekf.mp4", gt: "10_dark_ekf_gt.mp4" },
      isPlaceholder: true,
      placeholderNote: "Transformer weights training in progress"
    }
  ]
}
```
3. In `src/components/VideoPlayer.js`, add the `<option value="Transformer">` to the mode dropdown markup.

---

### Recipe 2: Adding a Custom Presentation Block
1. Open `src/config/reportContent.js`.
2. Append to `reportBlocks`:
```javascript
{
  id: "noise-analysis",
  layout: "image-right",
  tag: "Robustness",
  title: "Ambient Light Rejection Analysis",
  content: [
    "Analysis of signal-to-noise ratio under direct halogen and sunlight interference."
  ],
  highlights: [
    "Peak thresholding maintains >99.2% marker detection.",
    "Circularity filtering drops 100% of elongated specular glare."
  ],
  image: {
    src: "/assets/sample_rendered_frame.png",
    caption: "Figure 6: Robust optical isolation under high ambient noise."
  }
}
```
The renderer (`PresentationRenderer.js`) will dynamically build the card and insert it into the DOM.

---

### Recipe 3: Running Tests and Validation
- **Build verification**: `npm run build` (ensures zero syntax errors and produces a clean bundle in `dist/`).
- **Dev mode**: `npm run dev` (starts server on `http://localhost:5173/`).
- **Dataset API check**: Navigate to `http://localhost:5173/api/dataset-info` to verify active dataset JSON payload.
