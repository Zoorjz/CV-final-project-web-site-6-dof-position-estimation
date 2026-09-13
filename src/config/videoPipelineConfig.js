/**
 * Video Pipeline Configuration
 * 
 * Defines the available pipelines, discrete stages, descriptions, and video mappings.
 * Renders are dynamically served from the latest timestamped folder in /data/renders (e.g. data/renders/renders_20260911_233706).
 */

const rawBase = import.meta.env.BASE_URL || "./";
const basePrefix = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;
export const VIDEO_DATASET_BASE = `${basePrefix}data/renders`;

export const PIPELINE_CONFIG = {
  // Left side: Classical Geometry-based Pipeline (PnP & EKF)
  pnp3d: {
    id: "pnp3d",
    title: "Geometry based",
    subtitle: "PnP",
    badge: "Classical Vision",
    stages: [
      {
        id: "regular_video",
        name: "Long Exposure",
        shutter: "10,000 µs",
        description: "10,000 µs ambient visual stream showing the full room environment and tracking rig.",
        files: {
          raw: "geometry-based/11_bright_raw.mp4",
          gt: "geometry-based/12_bright_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "short_shutter_video",
        name: "Short Exposure",
        shutter: "1,000 µs",
        description: "1,000 µs high-speed dark IR frame eliminating ambient illumination and motion blur.",
        files: {
          raw: "geometry-based/01_dark_raw.mp4",
          gt: "geometry-based/02_dark_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "filtration",
        name: "Filtration",
        shutter: "1,000 µs",
        description: "Multi-stage optical pre-filtration (3x3 Gaussian blur, dynamic peak threshold T>=180, morphological opening) isolating genuine LEDs with zero background noise.",
        files: {
          raw: "geometry-based/03_dark_filtration.mp4",
          gt: "geometry-based/04_dark_filtration_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "blob_detection",
        name: "Blob Detection",
        shutter: "1,000 µs",
        description: "Filtered stream with sub-pixel 2D centroid moments: Cyan candidate rings & Top-4 Red tracking markers.",
        files: {
          raw: "geometry-based/05_dark_blobs.mp4",
          gt: "geometry-based/06_dark_blobs_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "pnp",
        name: "PnP",
        shutter: "1,000 µs",
        description: "SQPnP 6-DoF rigid body pose estimation ([R | t]) with RGB 3D axes, active red markers, and cyan motion trail.",
        files: {
          raw: "geometry-based/07_dark_pnp.mp4",
          gt: "geometry-based/08_dark_pnp_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "ekf",
        name: "EKF",
        shutter: "1,000 µs",
        description: "13-State Extended Kalman Filter constant-velocity state smoothing with RGB 3D axes and bright aqua trajectory trail.",
        files: {
          raw: "geometry-based/09_dark_ekf.mp4",
          gt: "geometry-based/10_dark_ekf_gt.mp4"
        },
        isPlaceholder: false
      }
    ]
  },

  // Right side: Data-Driven Pipeline with dropdown variants
  dataDriven: {
    id: "dataDriven",
    title: "Data driven",
    defaultMode: "CNN",
    modes: {
      CNN: {
        id: "CNN",
        name: "CNN",
        fullName: "Convolutional Neural Network",
        badge: "Deep Learning (Direct Pose)",
        stages: [
          {
            id: "regular_video",
            name: "Long Exposure",
            shutter: "10,000 µs",
            description: "Ambient RGB visual input capturing full room context and tracking environment.",
            files: {
              raw: "data-driven/01_bright_raw.mp4",
              gt: "data-driven/02_bright_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "dark_video",
            name: "Short Exposure",
            shutter: "1,000 µs",
            description: "1,000 µs high-contrast input tensor passed directly to convolutional layers for marker localization.",
            files: {
              raw: "data-driven/03_dark_raw.mp4",
              gt: "data-driven/04_dark_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "cnn_output",
            name: "CNN Output",
            shutter: "1,000 µs",
            description: "Direct 6-DoF pose prediction [q, t] regressed by deep neural network (Orchid axes & violet trail).",
            files: {
              raw: "data-driven/05_cnn_raw.mp4",
              gt: "data-driven/06_cnn_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "cnn_kf",
            name: "CNN + Kalman Filter",
            shutter: "1,000 µs",
            description: "Causal 6-DoF Kalman-filtered trajectory (Deep Violet axes & Indigo trail) eliminating high-frequency jitter.",
            files: {
              raw: "data-driven/07_cnn_kf_raw.mp4",
              gt: "data-driven/08_cnn_kf_gt.mp4"
            },
            isPlaceholder: false
          }
        ]
      },
      ML: {
        id: "ML",
        name: "ML",
        fullName: "Machine Learning (Feature Regressor)",
        badge: "Learned 2D-to-3D Regressor",
        stages: [
          {
            id: "regular_video",
            name: "Long Exposure",
            shutter: "10,000 µs",
            description: "Ambient visual frame capturing scene lighting and background spatial context.",
            files: {
              raw: "data-driven/01_bright_raw.mp4",
              gt: "data-driven/02_bright_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "dark_video",
            name: "Short Exposure",
            shutter: "1,000 µs",
            description: "1,000 µs short-shutter frame isolating active optical IR LED markers.",
            files: {
              raw: "data-driven/03_dark_raw.mp4",
              gt: "data-driven/04_dark_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "filtration",
            name: "Filtration",
            shutter: "1,000 µs",
            description: "Optical pre-filtration isolating genuine LED emissions with background suppressed.",
            files: {
              raw: "data-driven/05_dark_filtration.mp4",
              gt: "data-driven/06_dark_filtration_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "blob_detection",
            name: "Blob Detection",
            shutter: "1,000 µs",
            description: "Extracted 2D centroid coordinates and spatial moments compiled into tabular feature vectors.",
            files: {
              raw: "data-driven/07_dark_blobs.mp4",
              gt: "data-driven/08_dark_blobs_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "ml_output",
            name: "ML Algorithm Output",
            shutter: "1,000 µs",
            description: "Predicted 6-DoF pose output from trained Random Forest / MLP regressor (Coral axes & amber trail).",
            files: {
              raw: "data-driven/09_ml_raw.mp4",
              gt: "data-driven/10_ml_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "ml_kf",
            name: "ML + Kalman Filter",
            shutter: "1,000 µs",
            description: "Causal 6-DoF Kalman-filtered trajectory (Tangerine axes & canary gold trail) smoothing ML regression.",
            files: {
              raw: "data-driven/11_ml_kf_raw.mp4",
              gt: "data-driven/12_ml_kf_gt.mp4"
            },
            isPlaceholder: false
          }
        ]
      }
    }
  }
};

/**
 * Resolves the absolute video URL given a stage object and showGT boolean flag.
 */
export function resolveVideoUrl(stage, showGT) {
  if (!stage || !stage.files) return "";
  const filename = showGT ? (stage.files.gt || stage.files.raw) : stage.files.raw;
  return `${VIDEO_DATASET_BASE}/${filename}`;
}

/**
 * Collects all unique video URLs used in all pipelines for preloading.
 */
export function getAllPipelineVideoUrls() {
  const urls = new Set();
  
  // 3D Pipeline
  PIPELINE_CONFIG.pnp3d.stages.forEach(stage => {
    if (stage.files.raw) urls.add(`${VIDEO_DATASET_BASE}/${stage.files.raw}`);
    if (stage.files.gt) urls.add(`${VIDEO_DATASET_BASE}/${stage.files.gt}`);
  });

  // Data Driven Modes
  Object.values(PIPELINE_CONFIG.dataDriven.modes).forEach(mode => {
    mode.stages.forEach(stage => {
      if (stage.files.raw) urls.add(`${VIDEO_DATASET_BASE}/${stage.files.raw}`);
      if (stage.files.gt) urls.add(`${VIDEO_DATASET_BASE}/${stage.files.gt}`);
    });
  });

  return Array.from(urls);
}
