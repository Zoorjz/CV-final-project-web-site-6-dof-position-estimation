/**
 * Video Pipeline Configuration
 * 
 * Defines the available pipelines, stage definitions, step metadata, and video file mappings.
 * Videos are loaded from the latest renders directory: /data/renders
 */

export const VIDEO_DATASET_BASE = "/data/renders";

export const PIPELINE_CONFIG = {
  // Left side: Classical 3D Pipeline
  pnp3d: {
    id: "pnp3d",
    title: "3D Pipeline",
    subtitle: "PnP",
    badge: "Classical Vision",
    stages: [
      {
        id: "regular_video",
        name: "Regular Video",
        shutter: "10,000 µs",
        description: "10,000 µs ambient visual stream showing full scene environment and tracking rig.",
        files: {
          raw: "06_bright_raw.mp4",
          gt: "07_bright_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "short_shutter_video",
        name: "Short Shutter Video",
        shutter: "1,000 µs",
        description: "1,000 µs high-speed dark IR frame eliminating ambient illumination and motion blur.",
        files: {
          raw: "01_dark_raw.mp4",
          gt: "02_dark_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "filtration",
        name: "Filtration",
        shutter: "1,000 µs",
        description: "Adaptive thresholding and morphological filtering to isolate active optical IR LED markers.",
        files: {
          raw: "01_dark_raw.mp4",
          gt: "02_dark_gt.mp4"
        },
        isPlaceholder: true,
        placeholderNote: "Filter mask render in progress"
      },
      {
        id: "blob_detection",
        name: "Blob Detection",
        shutter: "1,000 µs",
        description: "Sub-pixel centroid localization (green rings) and ellipse contour fitting for optical markers.",
        files: {
          raw: "03_dark_blobs.mp4",
          gt: "03_dark_blobs.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "pnp",
        name: "PnP",
        shutter: "1,000 µs",
        description: "Perspective-n-Point 6-DoF rigid body pose estimation ([R | t]) with 3D coordinate axes and motion trails.",
        files: {
          raw: "04_dark_pnp.mp4",
          gt: "05_dark_pnp_gt.mp4"
        },
        isPlaceholder: false
      },
      {
        id: "ekf",
        name: "EKF",
        shutter: "1,000 µs",
        description: "Extended Kalman Filter state smoothing incorporating constant-velocity dynamics and jitter suppression.",
        files: {
          raw: "04_dark_pnp.mp4",
          gt: "05_dark_pnp_gt.mp4"
        },
        isPlaceholder: true,
        placeholderNote: "EKF trajectory render in progress"
      }
    ]
  },

  // Right side: Data-Driven Pipeline with dropdown variants
  dataDriven: {
    id: "dataDriven",
    title: "Data Driven Pipeline",
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
            name: "Regular Video",
            shutter: "10,000 µs",
            description: "Ambient RGB visual input provided to the neural convolutional feature backbone.",
            files: {
              raw: "06_bright_raw.mp4",
              gt: "07_bright_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "dark_video",
            name: "Dark Video",
            shutter: "1,000 µs",
            description: "1,000 µs high-contrast input tensor passed to convolutional layers for marker localization.",
            files: {
              raw: "01_dark_raw.mp4",
              gt: "02_dark_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "cnn_output",
            name: "CNN Output",
            shutter: "1,000 µs",
            description: "Direct 6-DoF pose prediction [q, t] regressed by deep convolutional neural network.",
            files: {
              raw: "04_dark_pnp.mp4",
              gt: "05_dark_pnp_gt.mp4"
            },
            isPlaceholder: true,
            placeholderNote: "CNN checkpoint inference render in progress"
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
            name: "Regular Video",
            shutter: "10,000 µs",
            description: "Ambient visual frame capturing scene lighting and background spatial context.",
            files: {
              raw: "06_bright_raw.mp4",
              gt: "07_bright_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "dark_video",
            name: "Dark Video",
            shutter: "1,000 µs",
            description: "1,000 µs short-shutter frame isolating active IR LED markers.",
            files: {
              raw: "01_dark_raw.mp4",
              gt: "02_dark_gt.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "filtration",
            name: "Filtration",
            shutter: "1,000 µs",
            description: "Intensity thresholding and spatial bandpass filtering for robust feature extraction.",
            files: {
              raw: "01_dark_raw.mp4",
              gt: "02_dark_gt.mp4"
            },
            isPlaceholder: true,
            placeholderNote: "ML pre-processing render in progress"
          },
          {
            id: "blob_detection",
            name: "Blob Detection",
            shutter: "1,000 µs",
            description: "Extracted 2D centroid coordinates and spatial moments compiled into tabular feature vectors.",
            files: {
              raw: "03_dark_blobs.mp4",
              gt: "03_dark_blobs.mp4"
            },
            isPlaceholder: false
          },
          {
            id: "ml_output",
            name: "ML Algorithm Output",
            shutter: "1,000 µs",
            description: "Predicted 6-DoF pose output from trained Random Forest / MLP feature regression model.",
            files: {
              raw: "04_dark_pnp.mp4",
              gt: "05_dark_pnp_gt.mp4"
            },
            isPlaceholder: true,
            placeholderNote: "ML regressor output render in progress"
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
