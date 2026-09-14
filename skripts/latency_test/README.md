# 6-DoF Real-Time Tracking Latency & Benchmark Suite

Self-contained low-latency real-time video capture and 6-DoF pose estimation benchmark suite for Raspberry Pi (and desktop simulation). Packed with all weights, configurations, and models in a single directory.

---

## 1. Overview of the 3 Tracking Pipelines

| Pipeline | Technique / Architecture | Preprocessing | Core Inference | Postprocessing | Typical Pi Profile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** (`realtime_cnn_latency.py`) | Temporal-3 YOLO Backbone + 6D Rotation Head | 256x256 Letterbox Float32 | YOLO Feature Map + Temporal Late Fusion | 6-DoF Kinematic Kalman Filter + SO(3) SLERP | Heavyweight DL (~10–25 ms) |
| **ML** (`realtime_ml_latency.py`) | Classical Blob Detection + MarkerTracker + Dual TaskMLP | 24D V4 Dot Feature Extraction | Separate Translation & Rotation MLPs | `AllFrameFilter` with sign continuity & velocity decay | Ultra-Fast ML (~2–4 ms) |
| **PnP** (`realtime_pnp_latency.py`) | Multi-Stage Optical Pre-Filter + SQPnP + 13-State EKF | Gaussian Blur + Dynamic Peak Threshold + Sub-Pixel Moments | SQPnP Permutation Solve with Fast Cache | 13-State Extended Kalman Filter (`PoseEKF`) | Geometric Classical (~1–2 ms) |
| **ONNX** (`realtime_onnx_latency.py`) | Single-Frame Static ONNX Graph | 256x256 Letterbox Float32 | ONNX Runtime Session | Quaternion Canonicalization | Optimized Static DL (~8–15 ms) |

---

## 2. Directory Layout & Bundled Assets

```
latency_test/
├── README.md                            # This comprehensive manual
├── requirements.txt                     # Unified Python dependencies
├── package_bundle.py                    # 1-Click ZIP distribution creator
├── camera_utils.py                      # Picamera2 low-latency capture & stats tracker
├── run_latency_benchmark.py             # Multi-model comparative orchestrator
├── realtime_cnn_latency.py              # CNN: Temporal-3 YOLO + Pose Head + Causal Kalman
├── realtime_ml_latency.py               # ML: Blob Tracker + Dual TaskMLP + AllFrameFilter
├── realtime_pnp_latency.py              # PnP: Optical Pre-Filter + SQPnP + 13-State PoseEKF
├── realtime_onnx_latency.py             # ONNX: Single-Frame Static ONNX graph
└── models/                              # Bundled weights, calibration & code
    ├── cnn/
    │   ├── best_combined.pt             # Trained temporal-3 checkpoint (33.9 MB)
    │   ├── yolo26n-cls.pt               # Pretrained backbone graph (5.8 MB)
    │   └── bar_pose_regression/         # Model graph & geometry helpers
    ├── ml/
    │   ├── translation.pt               # Translation MLP checkpoint (155 KB)
    │   ├── rotation.pt                  # Rotation MLP checkpoint (155 KB)
    │   ├── provenance.json              # Checkpoint metadata & SHA256 hashes
    │   ├── baseline.py                  # Blob detection & feature computation
    │   ├── tracker.py                   # Hungarian marker tracking
    │   ├── guided_detection.py          # Adaptive local thresholding
    │   ├── evaluate_full_frames.py      # Temporal slot assembling
    │   ├── separate_tasks.py            # TaskMLP PyTorch architecture
    │   └── postprocess_predictions.py   # AllFrameFilter state estimator
    ├── pnp/
    │   ├── camera_calibration.yaml      # Intrinsic matrix (K) & distortion coefficients
    │   └── geometry.yaml                # 3D marker coordinates on the training bar
    └── onnx/
        ├── bar_pose_256.onnx            # Static ONNX pose model (7.1 MB)
        └── model_contract.json          # Input/output schema specification
```

---

## 3. Quickstart & Deployment to Raspberry Pi

### Step A: Package on Development Machine
From this folder, run:
```bash
python package_bundle.py
```
This generates `pi_latency_benchmark_bundle.zip` (~41 MB compressed) containing all models, code, and weights.

### Step B: Transfer & Setup on Raspberry Pi
```bash
# 1. Copy archive to the Pi
scp pi_latency_benchmark_bundle.zip pi@<raspberry_pi_ip>:~/

# 2. SSH into the Pi and extract
ssh pi@<raspberry_pi_ip>
unzip pi_latency_benchmark_bundle.zip
cd latency_test

# 3. Create virtual environment and install dependencies
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> On Raspberry Pi OS (64-bit), install Picamera2 via APT:
> ```bash
> sudo apt update && sudo apt install -y python3-picamera2
> ```

---

## 4. Running Latency Tests

### 1. Run CNN Model
```bash
# Live Picamera2 on Pi
python realtime_cnn_latency.py --camera 0 --fps 72 --shutter-us 1000 --threads 4

# Offline / simulation mode (works on any laptop/desktop)
python realtime_cnn_latency.py --mock --duration 10
```

### 2. Run Classical ML Model
```bash
# Live Picamera2 on Pi
python realtime_ml_latency.py --camera 0 --fps 72 --shutter-us 1000

# With guided adaptive thresholding
python realtime_ml_latency.py --guided-detection

# Offline simulation
python realtime_ml_latency.py --mock --duration 10
```

### 3. Run Classical SQPnP + 13-State EKF Model
```bash
# Live Picamera2 on Pi
python realtime_pnp_latency.py --camera 0 --fps 72 --shutter-us 1000

# Custom EKF smoothing parameters
python realtime_pnp_latency.py --ekf-r-scale 2.0 --ekf-q-scale 1.5

# Offline simulation
python realtime_pnp_latency.py --mock --duration 10
```

### 4. Run Automated Comparative Benchmark
Run all 3 models sequentially and generate a unified comparative latency matrix:
```bash
python run_latency_benchmark.py --benchmark-all --duration 10 --output-report latency_report.md
```

---

## 5. Timing Metrics & Measurement Methodology

All scripts employ Picamera2 zero-queue capture (`queue=True`) to prevent backlog accumulation and measure physical hardware latency:

* **`acquire`**: Time to fetch the latest completed image array from camera memory.
* **`preprocess`**: Time spent on letterboxing / optical pre-filtering / blob extraction.
* **`inference`**: Core model execution (PyTorch forward pass / SQPnP solver).
* **`postprocess`**: State filtering (Causal Kalman / AllFrameFilter / 13-State EKF) and canonicalization.
* **`total`**: End-to-end host processing time (`acquire + preprocess + inference + postprocess`).
* **`frame_age`**: Time the frame sat in the camera driver buffer before processing (`t_acquire - SensorTimestamp`).
* **`sensor_to_prediction`**: True hardware latency from the physical moment of sensor readout to 6-DoF output (`t_done - SensorTimestamp`).
* **`exposure_to_prediction`**: Full photonic-to-prediction latency (`t_done - ExposureStart`).

---

## 6. Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--camera` | `int` | `0` | Picamera2 camera device index. |
| `--width` | `int` | `640` | Video frame width in pixels. |
| `--height` | `int` | `400` | Video frame height in pixels. |
| `--fps` | `float` | `72.0` | Target camera acquisition frame rate. |
| `--shutter-us` | `int` | `1000` | Sensor exposure time in microseconds (`1000` = 1 ms). |
| `--gain` | `float` | `0.0` | Sensor analogue gain (`0.0` for auto gain). |
| `--threads` | `int` | `4` | CPU worker threads for PyTorch / ONNX Runtime. |
| `--duration` | `float` | `0.0` | Seconds to run (`0.0` runs indefinitely until Ctrl+C). |
| `--print-every` | `int` | `1` | Interval between per-frame terminal logs. |
| `--mock` | `flag` | `False` | Force synthetic dark IR stream for offline testing. |
| `--source` | `str` | `None` | Video file path or OpenCV device index (fallback). |
| `--ekf-r-scale` | `float` | `1.0` | (PnP only) Measurement covariance multiplier for heavier smoothing. |
| `--ekf-q-scale` | `float` | `1.0` | (PnP only) Process covariance multiplier for agility. |
| `--no-kalman` | `flag` | `False` | (CNN only) Disable causal Kalman state filtering. |
