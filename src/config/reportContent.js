/**
 * Report Content Configuration
 * 
 * Data-driven presentation blocks vertically arranged on the webpage.
 * Easily add, remove, or modify blocks to update the report.
 * 
 * Supported layout types:
 * - 'hero'         : Title banner with project metadata and badges
 * - 'image-right'  : Text content on the left, image/figure on the right
 * - 'image-left'   : Image/figure on the left, text content on the right
 * - 'image-bottom' : Full-width visual with detailed caption & technical notes below
 * - 'stats-grid'   : Grid of key performance and calibration metrics
 */

export const reportMetadata = {
  title: "6-DoF Rigid Body Tracking & Benchmark",
  subtitle: "Comparative Analysis: Classical PnP/EKF vs. Data-Driven Deep Learning Pipelines",
  authors: ["Computer Vision Final Project", "UTN Studies"],
  date: "September 2026",
  version: "1.0",
  tags: ["6-DoF Pose Estimation", "PnP", "EKF", "CNN", "Meta Quest GT", "Dual-Exposure"]
};

export const reportBlocks = [
  {
    id: "hero",
    layout: "hero",
    badge: "Final Research Report & Interactive Demo",
    title: "Optical 6-DoF Rigid Body Tracking",
    subtitle: "High-precision optical pose estimation benchmarked against millimeter-accurate Meta Quest VR Ground Truth across classical vision and deep learning pipelines.",
    stats: [
      { label: "Median Position Error", value: "18.84 mm", subtext: "Rigid body translation accuracy" },
      { label: "Median Angular Error", value: "3.55°", subtext: "3D Euler orientation accuracy" },
      { label: "Shutter Speeds", value: "1ms / 10ms", subtext: "Dark IR vs. Ambient Visual" },
      { label: "Temporal Alignment", value: "Δt = +18.97 s", subtext: "Affine cross-correlation sync" }
    ]
  },
  {
    id: "motivation-hardware",
    layout: "image-right",
    tag: "Hardware & Acquisition",
    title: "Dual-Exposure Capture & Marker Array",
    content: [
      "Precise 6-DoF tracking in uncontrolled indoor environments faces extreme challenges from ambient illumination, motion blur, and visual occlusions. To solve this, our acquisition system employs an interleaved dual-exposure Raspberry Pi HQ camera rig.",
      "The tracking target consists of a custom rigid bar equipped with calibrated Infrared (IR) LED optical markers. By capturing alternating high-speed short-exposure frames (**1,000 µs / Dark IR**) and regular exposure frames (**10,000 µs / Bright Visual**), we achieve crisp marker isolation without sacrificing scene context."
    ],
    highlights: [
      "1,000 µs IR dark stream isolates active markers at near-zero background noise.",
      "10,000 µs ambient stream preserves visual scene features for data-driven modeling.",
      "Rigid body geometry defined with sub-millimeter calibrated 3D marker coordinates."
    ],
    image: {
      src: "/assets/sample_1000us.png",
      alt: "Short shutter 1000 µs IR capture showing isolated LED markers",
      caption: "Figure 1: Isolated IR optical marker centroids under 1,000 µs short-shutter exposure."
    }
  },
  {
    id: "ground-truth-calibration",
    layout: "image-left",
    tag: "Sensor Calibration",
    title: "VR Ground Truth & Coordinate Alignment",
    content: [
      "To rigorously evaluate pose estimation accuracy, a Meta Quest 6-DoF VR controller was mechanically coupled to the optical marker bar. The VR tracking system provides high-frequency ground-truth trajectories with millimeter accuracy.",
      "A dual-stage optimization pipeline resolves the coordinate frame transformations: the camera-to-world transform ($T_{\\text{cam} \\to \\text{vr}}$) and controller-to-bar offset ($T_{\\text{ctrl} \\to \\text{bar}}$), alongside an affine temporal cross-correlation ($t_{\\text{GT}} = t_{\\text{video}} + 18.9700\\text{ s}$)."
    ],
    highlights: [
      "Spatial Extrinsics: $T_{\\text{cam}\\to\\text{vr}}$ estimated via non-linear least squares.",
      "Temporal offset: $\\Delta t = +18.9700\\text{ s}$ verified with motion velocity peaks.",
      "Controller offset: $T_{\\text{ctrl}\\to\\text{bar}} = [23.24, -7.10, -1.51]\\text{ mm}$."
    ],
    image: {
      src: "/assets/sample_10000us.png",
      alt: "Regular exposure 10000 µs visual frame showing ambient tracking environment",
      caption: "Figure 2: 10,000 µs ambient visual frame displaying controller rig & tracking bar."
    }
  },
  {
    id: "spatial-temporal-results",
    layout: "image-bottom",
    tag: "Benchmark & Trajectory",
    title: "Spatial-Temporal Alignment & Error Evaluation",
    content: [
      "The estimated optical trajectory is quantitatively compared against the transformed VR ground truth over a continuous 200-frame motion sequence. Trajectories exhibit consistent spatial fidelity across complex 3D helical sweeps and sharp rotational maneuvers.",
      "The alignment yields a **median 3D translation error of 18.84 mm** and a **median angular orientation error of 3.55°**, establishing a robust baseline for evaluating both classical geometry-based and learned neural estimators."
    ],
    image: {
      src: "/assets/alignment_plots.png",
      alt: "Alignment curves and 3D trajectory comparison between PnP and VR Ground Truth",
      caption: "Figure 3: Synchronized 3D trajectory curves, Euler angles, and residual error distributions over time."
    }
  },
  {
    id: "classical-pipeline",
    layout: "image-right",
    tag: "Classical Vision",
    title: "3D Geometric Pipeline: PnP & EKF",
    content: [
      "The classical 3D pipeline leverages rigorous projective geometry. High-speed dark frames are processed through adaptive thresholding and 2D sub-pixel blob centroiding.",
      "The resulting 2D-3D point correspondences are solved using the Perspective-n-Point (PnP) algorithm with RANSAC outlier rejection, followed by an Extended Kalman Filter (EKF) constant-velocity motion model that suppresses high-frequency jitter."
    ],
    highlights: [
      "Sub-pixel ellipse fitting achieves $<0.2\\text{ px}$ centroid precision.",
      "Robust PnP solver estimates 6-DoF transformation matrix $[R | t]$.",
      "EKF smoothing guarantees temporal continuity and velocity estimation."
    ],
    image: {
      src: "/assets/sample_rendered_frame.png",
      alt: "Overlaid 3D Coordinate axes and trajectory trail on optical frame",
      caption: "Figure 4: Rendered optical 3D pose (RGB axes) and trajectory trail aligned with VR Ground Truth."
    }
  },
  {
    id: "data-driven-pipeline",
    layout: "image-left",
    tag: "Data-Driven Approaches",
    title: "Deep Learning & Regressor Pipelines",
    content: [
      "In parallel with classical geometric methods, two data-driven paradigms are investigated: an end-to-end **Convolutional Neural Network (CNN)** predicting 6-DoF pose directly from intensity images, and a hybrid **Machine Learning (ML)** regressor mapping extracted 2D blob features to 3D poses.",
      "These models offer superior resilience under severe optical marker occlusions and ambient reflections where classical 2D-3D correspondence matching may fail."
    ],
    highlights: [
      "CNN Pipeline: Direct end-to-end regression from raw frame to $[q, t]$.",
      "ML Pipeline: Geometric feature embedding + Random Forest / MLP regressor.",
      "Comparative benchmark highlights tradeoffs between computational cost and accuracy."
    ],
    image: {
      src: "/assets/sample_rendered_frame.png",
      alt: "Visual representation of data-driven tracking pipeline",
      caption: "Figure 5: 6-DoF pose estimation under data-driven neural and machine learning estimators."
    }
  }
];
