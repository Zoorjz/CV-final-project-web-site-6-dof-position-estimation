#!/usr/bin/env python3
"""Real-Time Low-Latency 6-DoF Pose Inference & Latency Test: Classical Optical SQPnP + 13-State EKF.

Captures frames with Picamera2 (queue=True, zero backlog) on Raspberry Pi,
executes multi-stage optical pre-filtering (Gaussian blur + dynamic peak thresholding
+ morphological opening), extracts sub-pixel moments centroids with circularity gating,
solves SQPnP correspondence with fast permutation caching, and applies 13-state PoseEKF.

Usage:
  python realtime_pnp_latency.py
  python realtime_pnp_latency.py --mock --duration 10
  python realtime_pnp_latency.py --ekf-r-scale 2.0 --ekf-q-scale 1.5
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
PNP_MODEL_DIR = HERE / "models" / "pnp"

from camera_utils import (  # noqa: E402
    CameraStream,
    LatencyTracker,
    add_standard_camera_args,
    optional_ms,
    pose_text,
)


def filter_and_detect_blobs(
    gray: np.ndarray,
    min_area: float = 2.0,
    max_area: float = 1500.0,
    min_circularity: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies multi-stage optical filtration and sub-pixel blob extraction.

    1. Gaussian Smoothing: Suppresses sensor shot noise and single-pixel hot spots.
    2. Dynamic Peak Thresholding: Isolates highest intensity LED emissions.
    3. Morphological Opening: Eradicates isolated specks and background noise.
    4. Circularity Gating: Eliminates elongated streaks and specular glare.
    5. Sub-pixel Centroid Extraction via Spatial Moments.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
    max_val = float(blurred.max())
    if max_val < 40:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    thresh_val = max(180, int(0.75 * max_val))
    _, bw = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bw_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []

    for c in contours:
        M = cv2.moments(c)
        area = M["m00"]
        if not (min_area < area < max_area):
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
            if circularity < min_circularity:
                continue

        cx = M["m10"] / area
        cy = M["m01"] / area
        pts.append((cx, cy))
        areas.append(area)

    return np.array(pts, dtype=np.float64).reshape(-1, 2), np.array(areas, dtype=np.float64)


def solve_sqpnp_pose(
    img_pts: np.ndarray,
    marker_pos: dict[str, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    preferred_perm: tuple | None = None,
    reproj_limit: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, float, tuple] | None:
    """Solves SQPnP pose over marker permutations with fast-path verification."""
    marker_ids = sorted(marker_pos.keys())
    perms = itertools.permutations(marker_ids)
    if preferred_perm is not None:
        perms = itertools.chain([preferred_perm], perms)

    best = None
    seen = set()
    for perm in perms:
        if perm in seen:
            continue
        seen.add(perm)

        obj = np.array([marker_pos[m] for m in perm], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue

        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.T).T + tvec.ravel()
        if (cam_pts[:, 2] <= 0).any():
            continue

        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean())
        if not np.isfinite(err) or err > reproj_limit:
            continue

        if best is None or err < best[2]:
            best = (rvec, tvec, err, perm)

        # Ultra fast-path: if cached permutation yields < 2px reprojection error, accept immediately
        if preferred_perm is not None and perm == preferred_perm and err < 2.0:
            break

    return best


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_omega(w: np.ndarray, dt: float) -> np.ndarray:
    theta = np.linalg.norm(w) * dt
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = w / np.linalg.norm(w)
    return np.array([np.cos(theta / 2.0), *(axis * np.sin(theta / 2.0))])


def f_state(x: np.ndarray, dt: float) -> np.ndarray:
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    p2 = p + v * dt
    q2 = quat_mult(q, quat_from_omega(w, dt))
    q2 = q2 / np.linalg.norm(q2)
    return np.concatenate([p2, v, q2, w])


def h_state(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[0:3], x[6:10]])


def numeric_jacobian(fn, x: np.ndarray, *args, eps: float = 1e-6) -> np.ndarray:
    fx = fn(x, *args)
    J = np.zeros((len(fx), len(x)))
    for k in range(len(x)):
        dx = np.zeros(len(x))
        dx[k] = eps
        J[:, k] = (fn(x + dx, *args) - fx) / eps
    return J


class PoseEKF:
    """13-state EKF: position(3), velocity(3), quaternion(4), angular velocity(3)."""

    def __init__(
        self,
        dt_ref: float = 1.0 / 72.0,
        r_scale: float = 1.0,
        q_scale: float = 1.0,
        disable_gating: bool = True,
        nis_gate_p: float = 0.9999,
        max_consecutive_rejections: int = 10,
    ):
        self.dt_ref = dt_ref
        self.disable_gating = disable_gating
        self.nis_threshold = chi2.ppf(nis_gate_p, df=7)
        self.max_consecutive_rejections = max_consecutive_rejections

        # Measurement noise covariance R: position std ~ 3.0mm, quaternion std ~ 0.01
        pos_std = 3.0
        quat_std = 0.01
        self.R = np.diag([pos_std**2] * 3 + [quat_std**2] * 4) * r_scale

        # Process noise covariance Q: position, velocity, orientation, angular velocity
        self.Q = np.diag([
            1.0, 1.0, 1.0,
            10.0, 10.0, 10.0,
            1e-5, 1e-5, 1e-5, 1e-5,
            0.1, 0.1, 0.1,
        ]) * q_scale

        self.x = None
        self.P = None
        self._consecutive_rejections = 0
        self.n_updated = 0
        self.n_rejected = 0

    def initialize(self, pos: np.ndarray, quat: np.ndarray):
        self.x = np.concatenate([pos, [0.0, 0.0, 0.0], quat, [0.0, 0.0, 0.0]])
        self.P = np.diag([1, 1, 1, 1e6, 1e6, 1e6, 0.01, 0.01, 0.01, 0.01, 100, 100, 100])
        self._consecutive_rejections = 0

    def predict(self, dt: float):
        if self.x is None:
            return
        F = numeric_jacobian(f_state, self.x, dt)
        self.x = f_state(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q * (dt / self.dt_ref)

    def update(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        if self.x is None:
            self.initialize(pos, quat)
            return True

        z = np.concatenate([pos, quat])
        H = numeric_jacobian(h_state, self.x)
        y = z - h_state(self.x)
        if y[3:] @ self.x[6:10] < 0:
            z = z.copy()
            z[3:] = -z[3:]
            y = z - h_state(self.x)

        S = H @ self.P @ H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))

        force_recover = self._consecutive_rejections >= self.max_consecutive_rejections
        if not self.disable_gating and nis > self.nis_threshold and not force_recover:
            self.n_rejected += 1
            self._consecutive_rejections += 1
            return False

        if force_recover:
            self.initialize(z[:3], z[3:])
            return True

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6:10] /= np.linalg.norm(self.x[6:10])
        self.P = (np.eye(13) - K @ H) @ self.P
        self.n_updated += 1
        self._consecutive_rejections = 0
        return True

    @property
    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[0:3], self.x[6:10]


def rvec_to_quaternion_wxyz(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S])
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S])
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_camera_args(parser)
    parser.add_argument(
        "--calib",
        type=Path,
        default=PNP_MODEL_DIR / "camera_calibration.yaml",
        help="Path to camera calibration YAML",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=PNP_MODEL_DIR / "geometry.yaml",
        help="Path to bar marker geometry YAML",
    )
    parser.add_argument("--fast-threshold", type=float, default=3.5, help="Reprojection error (px) threshold to keep cached permutation (default: 3.5px)")
    parser.add_argument("--reproj-limit", type=float, default=500.0, help="Maximum allowed reprojection error in px")
    parser.add_argument("--ekf-r-scale", type=float, default=1.0, help="EKF measurement noise R scale")
    parser.add_argument("--no-ekf", action="store_true", help="Disable 13-state EKF filter and output raw SQPnP poses directly")
    parser.add_argument("--save-snapshot", type=Path, default=None, help="Save an annotated debug JPEG snapshot of the tracking")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print(" 6-DoF Real-Time Latency Test: Optical SQPnP + 13-State EKF")
    print("=" * 60)
    print(f"Calibration    : {args.calib}")
    print(f"Geometry       : {args.geometry}")
    print(f"Camera         : {args.width}x{args.height} @ {args.fps:g} FPS (shutter={args.shutter_us}us)")
    print(f"Fast Threshold : {args.fast_threshold:.2f} px")
    print(f"EKF State      : {'Disabled (Raw SQPnP)' if args.no_ekf else f'Enabled (R-scale={args.ekf_r_scale:.2f}, Q-scale={args.ekf_q_scale:.2f})'}")
    print(f"Output Mode    : {'QUIET (Final Summary Only)' if args.quiet or args.print_every == 0 else f'Print every {args.print_every} frames'}")
    print("Stop           : Ctrl+C\n")

    # Load calibration & marker geometry
    if not args.calib.is_file():
        raise FileNotFoundError(f"Calibration file missing: {args.calib}")
    if not args.geometry.is_file():
        raise FileNotFoundError(f"Geometry file missing: {args.geometry}")

    cal = yaml.safe_load(args.calib.read_text(encoding="utf-8"))
    orig_w = float(cal.get("image_width", 1280))
    scale = args.width / orig_w if orig_w > 0 else 0.5
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    K[:2, :] *= scale
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

    geom = yaml.safe_load(args.geometry.read_text(encoding="utf-8"))
    marker_pos = {
        m["id"]: np.array(m["position"], dtype=np.float64) for m in geom["markers"]
    }

    ekf = None if args.no_ekf else PoseEKF(
        dt_ref=1.0 / max(1.0, args.fps),
        r_scale=args.ekf_r_scale,
        q_scale=args.ekf_q_scale,
        disable_gating=not args.ekf_gating,
    )
    tracker = LatencyTracker("PnP (Optical SQPnP + 13-State EKF)" if not args.no_ekf else "PnP (Raw Optical SQPnP)")

    stream = CameraStream(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        shutter_us=args.shutter_us,
        gain=args.gain,
        source=args.source,
        use_mock=args.mock,
    )
    stream.start()

    started_ns = time.monotonic_ns()
    last_frame_ns = started_ns
    frame_idx = 0
    cached_perm = None
    saved_snapshot = False

    try:
        while args.duration == 0 or (time.monotonic_ns() - started_ns) / 1e9 < args.duration:
            t0 = time.monotonic_ns()
            req = stream.capture_request()
            metadata = req.get_metadata()
            frame_rgb = req.make_array("main")
            t1 = time.monotonic_ns()

            # Preprocess: Optical pre-filter + moments blob detection
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY) if frame_rgb.ndim == 3 else frame_rgb
            detected_pts, areas = filter_and_detect_blobs(gray)
            t2 = time.monotonic_ns()

            # Inference: SQPnP correspondence solve
            solve_res = None
            reproj_err = 0.0
            if len(detected_pts) >= 4:
                top4_idx = np.argsort(areas)[::-1][:4]
                img_pts_top4 = detected_pts[top4_idx]
                solve_res = solve_sqpnp_pose(
                    img_pts_top4,
                    marker_pos,
                    K,
                    dist,
                    preferred_perm=cached_perm,
                    reproj_limit=args.reproj_limit,
                )

            t3 = time.monotonic_ns()

            # Postprocess: EKF smoothing or raw canonicalization
            dt = max(1e-4, (t3 - last_frame_ns) / 1e9)
            last_frame_ns = t3

            if solve_res is not None:
                rvec, tvec, reproj_err, perm = solve_res
                if reproj_err < args.fast_threshold:
                    cached_perm = perm
                else:
                    cached_perm = None

                t_mm = tvec.ravel()
                q_wxyz = rvec_to_quaternion_wxyz(rvec)

                if ekf is not None:
                    ekf.predict(dt)
                    ekf.update(t_mm, q_wxyz)
                    pos_filt, quat_filt_wxyz = ekf.pose
                    pos_m = pos_filt / 1000.0
                    qw, qx, qy, qz = quat_filt_wxyz
                    pose = np.array([pos_m[0], pos_m[1], pos_m[2], qx, qy, qz, qw], dtype=np.float32)
                else:
                    pos_m = t_mm / 1000.0
                    qw, qx, qy, qz = q_wxyz
                    pose = np.array([pos_m[0], pos_m[1], pos_m[2], qx, qy, qz, qw], dtype=np.float32)


                # Optional debug snapshot save
                if args.save_snapshot and not saved_snapshot and frame_idx > 10:
                    vis = frame_rgb.copy()
                    for cx, cy in detected_pts:
                        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), 2)
                    for cx, cy in img_pts_top4:
                        cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 255), 2)
                    # Project axes
                    R_m, _ = cv2.Rodrigues(rvec)
                    axes_3d = np.array([[80.0, 0, 0], [0, 80.0, 0], [0, 0, 80.0]], dtype=np.float64)
                    origin_3d = np.zeros((1, 3), dtype=np.float64)
                    proj_o, _ = cv2.projectPoints(origin_3d, rvec, tvec, K, dist)
                    proj_a, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
                    po = tuple(proj_o.reshape(-1, 2)[0].astype(int))
                    for pt, col in zip(proj_a.reshape(-1, 2), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        cv2.line(vis, po, tuple(pt.astype(int)), col, 2)
                    cv2.imwrite(str(args.save_snapshot), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    print(f"\n[Snapshot] Saved annotated debug frame to {args.save_snapshot}")
                    saved_snapshot = True
            else:
                cached_perm = None
                if ekf.x is not None:
                    ekf.predict(dt)
                    pos_filt, quat_filt_wxyz = ekf.pose
                    pos_m = pos_filt / 1000.0
                    qw, qx, qy, qz = quat_filt_wxyz
                    pose = np.array([pos_m[0], pos_m[1], pos_m[2], qx, qy, qz, qw], dtype=np.float32)
                else:
                    pose = np.zeros(7, dtype=np.float32)

            t4 = time.monotonic_ns()

            (
                acq_ms,
                pre_ms,
                infer_ms,
                post_ms,
                total_ms,
                sensor_ms,
                age_ms,
            ) = tracker.record_frame(t0, t1, t2, t3, t4, metadata)

            if not args.quiet and args.print_every > 0 and frame_idx % args.print_every == 0:
                exp_text = f"{metadata.get('ExposureTime', args.shutter_us):.0f}us"
                err_text = f"err={reproj_err:4.2f}px" if solve_res is not None else "no_solve"
                print(
                    f"#{frame_idx:06d} "
                    f"total={total_ms:6.2f}ms "
                    f"sensor={optional_ms(sensor_ms)}ms "
                    f"age={optional_ms(age_ms)}ms | "
                    f"acq={acq_ms:5.2f} pre={pre_ms:5.2f} infer={infer_ms:5.2f} post={post_ms:5.2f} | "
                    f"exp={exp_text} blobs={len(detected_pts)} {err_text} | "
                    f"{pose_text(pose)}",
                    flush=True,
                )

            frame_idx += 1


    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        finished_ns = time.monotonic_ns()
        stream.stop()

    elapsed_s = (finished_ns - started_ns) / 1e9
    tracker.print_summary(elapsed_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
