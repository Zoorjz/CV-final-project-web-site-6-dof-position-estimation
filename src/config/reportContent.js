/**
 * Report Content Configuration
 * 
 * Data-driven presentation blocks configured from project presentation materials.
 * Separated into presentationBlocks (top) and resultsBlocks (under interactive comparison).
 */

export const reportMetadata = {
  title: "Real-Time 6D Pose Estimation of a Rigid Training Bar",
  subtitle: "Comparative Study of Geometry-Based and Data-Driven Approaches on Embedded Hardware",
  authors: ["Computer Vision Final Project", "UTN Studies"],
  date: "September 2026",
  version: "1.0",
  tags: ["6-DoF Pose Estimation", "PnP", "EKF", "CNN", "Machine Learning", "Raspberry Pi 4B", "Meta Quest GT"]
};

export const presentationBlocks = [
  {
    id: "hero",
    layout: "hero",
    badge: "Computer Vision Final Project Presentation",
    title: "Real-Time 6D Pose Estimation of a Rigid Training Bar",
    subtitle: "Comparative Study of Geometry-Based and Data-Driven Approaches on Embedded Hardware",
    stats: [
      { label: "Hardware Target", value: "Raspberry Pi 4B", subtext: "Real-time edge compute" },
      { label: "Geometry PnP", value: "~0.2 px Reproj", subtext: "Weak 3D depth observability" },
      { label: "Temporal-3 CNN", value: "10.2 cm / 20.4°", subtext: "Median error vs VR Ground Truth" },
      { label: "Ground Truth", value: "Meta Quest Pro", subtext: "90 Hz Inside-Out Tracking" }
    ]
  },
  {
    id: "research-objective",
    layout: "image-right",
    tag: "Research Objective",
    title: "6-DoF Pose Estimation for Robotic Strength Training",
    subsections: [
      {
        title: "Primary Research Goal",
        items: [
          "Development of a real-time 6-DoF tracking system (3× Cartesian positions, 3× Euler rotations) for the barbell of a motorized robotic strength training machine."
        ]
      },
      {
        title: "Key Enabled Capabilities",
        items: [
          "<strong>Enhanced Exercise Safety</strong>: Real-time velocity tracking, trajectory boundary enforcement, and instant emergency load shedding during user failure or instability.",
          "<strong>Kinematic Exercise Classification</strong>: Automated exercise identification, movement phase segmentation, and repetition analytics derived from 3D trajectory curves.",
          "<strong>Training Gamification & Biofeedback</strong>: Immersive interactive feedback in AR/VR environments, enabling live biomechanical form guidance and virtual coaching."
        ]
      },
      {
        title: "Hardware & Operational Constraints",
        items: [
          "<strong>Edge Computational Budget</strong>: Constrained processing capacity and thermal profile of the Raspberry Pi 4B single-board computer.",
          "<strong>Real-Time Tracking Mandate</strong>: Strict requirement for low-latency, deterministic 6-DoF state updates without frame dropping."
        ]
      }
    ],
    image: {
      src: "/images/image7.gif",
      alt: "Interactive VR Workout Simulation & Bar Tracking",
      caption: "Figure 1: Real-time 6-DoF bar tracking enables dynamic safety, exercise classification, and immersive VR workout gamification."
    }
  },
  {
    id: "comparative-methodology",
    layout: "dual-image-right",
    cardType: "toggle",
    tag: "Comparative Methodology",
    title: "Geometry-Based vs. Data-Driven 6-DoF Architectures",
    subsections: [
      {
        title: "Evaluated Algorithmic Paradigms",
        items: [
          "<strong>Geometry-Based Pipeline (SQPnP + EKF)</strong>: Monocular 5-stage solver evaluating all 24 marker correspondences per frame (~0.2 px reprojection error). However, with 3 of 4 markers clustered and near-collinear, monocular depth and rotation remain weakly constrained in 3D.",
          "<strong>Data-Driven Temporal CNN</strong>: Two-stage architecture (YOLO classification backbone + continuous 6D SO(3) rotation & log-depth metric head) fusing 3 temporal frames to overcome depth and rotational ambiguities.",
          "<strong>Classical Machine Learning (ML)</strong>: Lightweight regression models trained on handcrafted 2D geometric and blob features as a fast edge baseline."
        ]
      },
      {
        title: "Interactive Comparison Platform Guide",
        items: [
          "<strong>Synchronized Side-by-Side Players</strong>: Compare two independent pipelines in lockstep with synchronized timeline scrubbing and playback.",
          "<strong>Pipeline Stage Timeline</strong>: Step through intermediate algorithm representations (Raw Frames &rarr; Optical Filtration &rarr; Blob Extraction &rarr; 6-DoF Pose Solver &rarr; EKF Filter).",
          "<strong>Simulate Raspberry Pi 4B FPS Toggle</strong>: Experience on-device execution cadence (5.0 FPS for CNN, 25.0 FPS for ML, 75.0 FPS for PnP) with hardware latency emulation.",
          "<strong>Ground Truth Reference Toggle</strong>: Overlay the 90 Hz high-precision Meta Quest Touch Pro optical reference trajectory to evaluate drift and accuracy."
        ]
      }
    ],
    images: [
      {
        src: "/images/toggle_simulate_rpi.png",
        alt: "Simulate Raspberry Pi 4B Toggle",
        caption: "Simulate Raspberry Pi 4B hardware FPS (5 / 25 / 75 FPS)"
      },
      {
        src: "/images/toggle_ground_truth.png",
        alt: "Show Ground Truth Toggle",
        caption: "Meta Quest Touch Pro 90 Hz ground truth reference toggle"
      }
    ]
  }
];

export const resultsBlocks = [
  {
    id: "performance-tradeoffs",
    layout: "image-bottom",
    tag: "Results & Analysis",
    title: "Real-Time Throughput Benchmarks & Optical Robustness Trade-offs",
    subsections: [
      {
        title: "Hardware Throughput & 3D Ground Truth Accuracy",
        items: [
          "<strong>Geometry-Based PnP &mdash; High Speed, Weak 3D Observability</strong>: Runs at 75.0 FPS on Pi 4 (300&ndash;400 FPS on desktop) with near-perfect 2D reprojection (~0.2 px). However, because 4 markers (3 clustered and near-collinear) weakly constrain monocular depth, 3D deviations against independent VR Ground Truth remain large (~0.6&ndash;0.8 m translation error, ~80&ndash;90&deg; rotation error).",
          "<strong>Temporal-3 CNN Regressor &mdash; 10.2 cm / 20.4&deg; vs Ground Truth</strong>: Runs at 5.0 FPS on Pi 4 (~200 ms latency). By fusing 3 temporal frames and optimizing log-depth with metric-space loss, it achieves a median translation error of <strong>10.2 cm</strong> (z-axis MAE cut to <strong>6.9 cm</strong>) and median rotation error of <strong>20.4&deg;</strong> on held-out temporal validation windows.",
          "<strong>Handcrafted Feature ML Regressor &mdash; 25.0 FPS Edge Baseline</strong>: Delivers 25.0 FPS (~40 ms latency); provides a faster edge baseline than deep neural networks, but exhibits higher positional error and sensitivity to 2D blob detection noise."
        ]
      },
      {
        title: "Monocular Ambiguities & Environmental Fault Tolerance",
        items: [
          "<strong>Resolving Depth & Rotation Ambiguities (CNN)</strong>: Predicting depth in log space paired with metric-space loss reduces z-axis error from 8.8 cm to <strong>6.9 cm</strong>. 3-frame temporal context resolves collinear rotation ambiguity, cutting median rotation error from 28.2&deg; to <strong>20.4&deg;</strong>.",
          "<strong>Environmental Noise & Failure Modes</strong>: Classical marker-based pipelines (PnP / ML) rely on clean optical thresholding and fail under ambient window backlighting, specular reflections, or bright background glare. The CNN is the <strong>only architecture robust against unconstrained lighting disturbances</strong>."
        ]
      }
    ],
    table: {
      headers: ["Pipeline Architecture", "RPi 4B Frame Rate", "Inference Latency", "2D Reprojection", "3D Translation vs GT", "3D Rotation vs GT", "Optical Glare Robustness"],
      rows: [
        ["Geometry PnP (SQPnP + EKF)", "75.0 FPS (300-400 desktop)", "13.3 ms", "~0.2 px (Near-perfect)", "~60–80 cm (Weak depth constraint)", "~80–90° (Collinear ambiguity)", "Fails under ambient glare"],
        ["ML Feature Regressor", "25.0 FPS", "40.0 ms", "N/A (Handcrafted features)", "Degraded / Jitter on noisy blobs", "Degraded / High error", "Fails under ambient glare"],
        ["Temporal-3 CNN (YOLO + Log-Depth)", "5.0 FPS", "200.0 ms", "N/A (Direct 6D metric pose)", "10.2 cm Median (6.9 cm z MAE)", "20.4° Median (23.6° Mean)", "Fully robust against glare & flares"]
      ]
    },
    highlights: [
      "Monocular PnP achieves near-perfect 2D reprojection (~0.2 px), but near-collinear markers leave 3D depth and rotation weakly constrained (~60–80 cm, ~80–90° error vs VR Ground Truth).",
      "Temporal-3 CNN directly addresses these ambiguities: 3-frame temporal fusion lowers rotation error from 28.2° to 20.4°, while log-depth metric loss reduces z-axis MAE to 6.9 cm.",
      "The CNN architecture achieves 10.2 cm median translation and 20.4° median rotation error on temporal holdout validation, while maintaining total robustness against severe window glare and specular flares."
    ]
  }
];

export const reportBlocks = [...presentationBlocks, ...resultsBlocks];
