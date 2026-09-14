# 6-DoF Real-Time Pose Estimation Latency Benchmark

**Benchmark Timestamp**: `2026-09-14 13:16:24`  
**Test Configuration**: `640x400 @ 72 FPS` | Shutter: `1000 µs` | Threads: `4` | Test Duration: `2.0s per model`

---

## 1. Summary Comparison Matrix

| Pipeline / Model | Total Latency (Mean) | Core Inference | Preprocessing | Postprocessing / Filtering | Sensor to Prediction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CNN** | `13.16 ms` | `10.89 ms` | `2.06 ms` | `0.21 ms` | `14.16 ms` |
| **ML** | `2.49 ms` | `0.60 ms` | `1.76 ms` | `0.08 ms` | `3.49 ms` |
| **PNP** | `1.40 ms` | `0.41 ms` | `0.39 ms` | `0.50 ms` | `2.39 ms` |
| **ONNX** | *Failed / Not Available* | - | - | - | - |
