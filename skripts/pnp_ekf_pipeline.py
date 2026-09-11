#!/usr/bin/env python3
"""6-DoF bar tracking: per-frame PnP + EKF smoothing.

Pipeline, per method.md, with one deliberate change from the original notebook version:

    Stages 1-2 (binarize, sub-pixel blob centroids) -> Stage 3/4 combined: for EVERY frame,
    independently brute-force the 4! = 24 assignments of markers M1..M4 to the (up to 4)
    detected 2D points, solve SQPNP for each, and keep the lowest-reprojection-error valid
    one -> Stage 5: EKF smoothing.

Why per-frame, not warm-started/tracked:
    The original notebook only resolved marker correspondence once, at the bootstrap frame,
    then relied on 2D proximity tracking (Hungarian matching) plus a guess-seeded
    useExtrinsicGuess=True solve chained frame-to-frame to keep things fast. That chain has
    no way to self-correct: once it drifts, each new frame's solve starts from a bad seed
    and stays bad.

    Re-deriving the full correspondence + pose from scratch every frame fixes exactly that
    problem: median reprojection error drops from ~20px (guess-seeded chain) to ~0.2-0.3px
    (this version) -- i.e. solvePnP now genuinely explains what the camera sees, essentially
    every frame, confirmed both numerically and by the real-image overlay in
    render_pnp_vs_gt.py. That is a real, confirmed fix, and worth keeping regardless of the
    next point.

    IMPORTANT, still open: this fix does NOT resolve the separate finding that the recovered
    trajectory (raw or EKF-filtered) shows ~0 correlation with the independent ground-truth
    log (labels.csv) -- re-checked on this version across the full 42k-frame dataset
    (32,753 overlapping samples), correlation is +0.02, essentially unchanged from before this
    fix. So the correspondence-tracking bug was real and is now fixed, but it was not the
    (sole) explanation for the ground-truth mismatch investigated earlier. That remains open;
    current leading suspects are calibration_output/geometry.yaml's measured marker positions,
    or a labels.csv<->this specific image set correspondence issue (see render_overlay.py,
    which references a different dataset path, /home/kyong/dataset_cleaned2).

    The cost of the per-frame fix is running solvePnP up to 24x per frame instead of once; at
    ~15-25k solvePnP/s on a desktop (single 4-point solve), even 24x is comfortably real-time
    here (measured: ~350fps end-to-end including detection), but this has NOT yet been
    benchmarked on the actual Raspberry Pi 4 target -- see the --cache-permutation flag below
    for a cheap speed-up if that turns out to be necessary.

The EKF does two jobs: (1) smooth normal frame-to-frame jitter, and (2) actively reject
outlier measurements (e.g. the rotation flips that can occur on a near-planar 4-point
configuration) via an NIS gate, so a single bad frame's pose never enters the filtered
trajectory -- it's treated as a missed detection for that frame instead.

Usage:
    python pnp_ekf_pipeline.py --start 0 --count 42202 --out tracking_result.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import median_filter
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Stage 1-2: binarization + sub-pixel blob extraction
# ---------------------------------------------------------------------------

THRESH = 60
MIN_BLOB_AREA = 2.0


def detect_blobs(gray: np.ndarray, thresh: int = THRESH, min_area: float = MIN_BLOB_AREA):
    """cv2.threshold + cv2.findContours + cv2.moments -> sub-pixel (x, y) centroids."""
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts, areas = [], []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > min_area:
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
            areas.append(M["m00"])
    return np.array(pts), np.array(areas)


# ---------------------------------------------------------------------------
# Stage 3/4: per-frame correspondence + pose, redone independently every frame
# ---------------------------------------------------------------------------

REPROJ_ERR_LIMIT = 500.0  # px; above this (or a point behind the camera) a solve is rejected


def solve_frame_pose(img_pts: np.ndarray, marker_pos: dict, K: np.ndarray, dist: np.ndarray,
                      preferred_perm: tuple | None = None):
    """Try marker<->point assignments and return the best valid (pose, permutation, error).

    If `preferred_perm` is given, it's tried first; if its reprojection error is already
    excellent (<2px) the other 23 permutations are skipped -- a safe speed-up (see
    --cache-permutation) since the correct assignment essentially never produces a bad fit,
    while a wrong one only occasionally produces a good one by chance (and even then, only
    with a corresponding wrong pose, which the EKF's NIS gate is positioned to catch).
    Returns None if no permutation yields a valid solve.
    """
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
        obj = np.array([marker_pos[m] for m in perm])
        ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        cam_pts = (R @ obj.T).T + tvec.ravel()
        if (cam_pts[:, 2] <= 0).any():
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).mean()
        if not np.isfinite(err) or err > REPROJ_ERR_LIMIT:
            continue
        if best is None or err < best[2]:
            best = (rvec, tvec, err, perm)
        if preferred_perm is not None and perm == preferred_perm and err < 2.0:
            break  # cheap path: previous frame's assignment still fits essentially perfectly
    if best is None:
        return None
    rvec, tvec, err, perm = best
    return rvec, tvec, err, perm


# ---------------------------------------------------------------------------
# Stage 5: EKF (position, velocity, quaternion, angular velocity), numeric Jacobians
# ---------------------------------------------------------------------------

def rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S])
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = np.array([(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S])
        elif i == 1:
            S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = np.array([(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S])
        else:
            S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = np.array([(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S])
    if q[0] < 0:
        q = -q  # canonical hemisphere; the EKF update also re-checks this per measurement
    return q / np.linalg.norm(q)


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_omega(w, dt):
    theta = np.linalg.norm(w) * dt
    if theta < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = w / np.linalg.norm(w)
    return np.array([np.cos(theta / 2), *(axis * np.sin(theta / 2))])


def f_state(x, dt):
    """Constant-velocity / constant-angular-velocity process model."""
    p, v, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
    p2 = p + v * dt
    q2 = quat_mult(q, quat_from_omega(w, dt))
    q2 = q2 / np.linalg.norm(q2)
    return np.concatenate([p2, v, q2, w])


def h_state(x):
    """Measurement model: solvePnP gives position + orientation directly (mostly linear)."""
    return np.concatenate([x[0:3], x[6:10]])


def numeric_jacobian(fn, x, *args, eps=1e-6):
    fx = fn(x, *args)
    J = np.zeros((len(fx), len(x)))
    for k in range(len(x)):
        dx = np.zeros(len(x))
        dx[k] = eps
        J[:, k] = (fn(x + dx, *args) - fx) / eps
    return J


class PoseEKF:
    """13-state EKF: position(3), velocity(3), quaternion(4), angular velocity(3).

    Measurement: position(3) + quaternion(4), taken directly from a per-frame solvePnP
    result. Rejects (treats as a missed detection) any update whose Normalized Innovation
    Squared exceeds a chi-square gate -- this is what stops an occasional bad/flipped PnP
    solve from ever entering the filtered trajectory, rather than just damping it afterward.
    """

    # If this many consecutive measurements get gated out, force-accept the next one as a
    # re-initialization instead of gating it too. Without this, the filter has no way back:
    # once it falls behind, its own (falsely confident) prediction makes every subsequent
    # real measurement look like a bigger outlier, which only widens the gap further --
    # observed directly here as NIS climbing for hundreds of frames with zero acceptance.
    MAX_CONSECUTIVE_REJECTIONS = 5

    def __init__(self, process_noise: np.ndarray, measurement_noise: np.ndarray,
                 dt_ref: float, nis_gate_p: float = 0.999):
        self.Q = process_noise  # sized for one step of duration dt_ref; predict() scales it
        self.dt_ref = dt_ref
        self.R = measurement_noise
        self.nis_threshold = chi2.ppf(nis_gate_p, df=7)  # 7-dim measurement
        self.x = None
        self.P = None
        self.last_nis = None
        self.n_rejected = 0
        self.n_updated = 0
        self.n_reinitialized = 0
        self._consecutive_rejections = 0

    def initialize(self, pos: np.ndarray, quat: np.ndarray):
        self.x = np.concatenate([pos, [0.0, 0.0, 0.0], quat, [0.0, 0.0, 0.0]])
        # Velocity and angular velocity are completely unknown from a single measurement --
        # a tight prior there (e.g. plain identity) is what caused the divergence above:
        # the true velocity is essentially guaranteed to look like a rejectable "outlier"
        # against a confident-but-wrong guess of zero.
        self.P = np.diag([1, 1, 1, 1e6, 1e6, 1e6, 0.01, 0.01, 0.01, 0.01, 100, 100, 100])
        self._consecutive_rejections = 0

    def predict(self, dt: float):
        F = numeric_jacobian(f_state, self.x, dt)
        self.x = f_state(self.x, dt)
        # Q was estimated for one ~dt_ref step; scale linearly for longer gaps (missed
        # detections) so a multi-frame gap doesn't leave the filter falsely overconfident.
        self.P = F @ self.P @ F.T + self.Q * (dt / self.dt_ref)

    def update(self, pos: np.ndarray, quat: np.ndarray) -> bool:
        """Returns True if the measurement was accepted (including a forced recovery reset),
        False if gated out as a likely-outlier."""
        z = np.concatenate([pos, quat])
        H = numeric_jacobian(h_state, self.x)
        y = z - h_state(self.x)
        if y[3:] @ self.x[6:10] < 0:  # quaternion double-cover: use the closer sign
            z = z.copy()
            z[3:] = -z[3:]
            y = z - h_state(self.x)

        S = H @ self.P @ H.T + self.R
        nis = float(y @ np.linalg.solve(S, y))
        self.last_nis = nis

        force_recover = self._consecutive_rejections >= self.MAX_CONSECUTIVE_REJECTIONS
        if nis > self.nis_threshold and not force_recover:
            self.n_rejected += 1
            self._consecutive_rejections += 1
            return False

        if force_recover:
            self.initialize(z[:3], z[3:])
            self.n_reinitialized += 1
            return True

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6:10] /= np.linalg.norm(self.x[6:10])
        self.P = (np.eye(13) - K @ H) @ self.P
        self.n_updated += 1
        self._consecutive_rejections = 0
        return True

    @property
    def position(self):
        return self.x[0:3]

    @property
    def velocity(self):
        return self.x[3:6]

    @property
    def quaternion(self):
        return self.x[6:10]

    @property
    def angular_velocity(self):
        return self.x[10:13]


def robust_std(x: np.ndarray, axis=None) -> np.ndarray:
    """MAD-based scale estimate. A handful of bad frames (misidentified markers, detection
    glitches) slip through even a tight reprojection-error gate, and a plain std() over
    positions/velocities/accelerations gets dominated by those outliers -- observed to
    mis-tune Q by roughly 10x when used naively. This is robust to that.
    """
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def estimate_measurement_noise(positions: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Empirical R: residual of each channel from a short median-filtered version of itself,
    rather than a hand-picked guess -- so the EKF's own consistency check (NIS) means something.

    With the per-frame full-search solve, the bulk of frames fit to a fraction of a pixel,
    so the robust (MAD) residual estimate can legitimately come back at ~0 for more than half
    the data -- but R=0 is degenerate for a Kalman filter (any nonzero innovation then looks
    infinitely significant, which is what caused near-total NIS-gate rejection here). Floor it
    at a small physically-plausible value instead of trusting the estimator down to zero.
    """
    pos_std = np.array([robust_std(positions[:, k] - median_filter(positions[:, k], size=5)) for k in range(3)])
    quat_std = np.array([robust_std(quats[:, k] - median_filter(quats[:, k], size=5)) for k in range(4)])
    pos_std = np.maximum(pos_std, 1.0)      # mm: sub-pixel detection noise floor
    quat_std = np.maximum(quat_std, 0.005)  # unitless quaternion component floor
    return np.diag(np.concatenate([pos_std**2, quat_std**2]))


def estimate_process_noise(positions: np.ndarray, quats: np.ndarray, fps: float,
                            frame_gap: np.ndarray) -> np.ndarray:
    """Empirical Q, sized from how much velocity/angular velocity actually change between
    consecutive real frames (i.e. acceleration), not a hand-picked constant. Position/
    orientation process noise is left small since those are mostly explained by integrating
    velocity/angular velocity -- only the velocity terms need to track real dynamics, and a
    too-tight Q there is exactly what silently turns "the bar sped up" into "outlier, reject."

    `frame_gap[j]` is the true frame-index gap between valid sample j and j+1 (length n-1);
    a velocity/angular-velocity estimate is only trusted where that gap is exactly 1, and an
    acceleration estimate only where two such steps in a row are both gap-1.
    """
    dt = 1.0 / fps
    n = len(positions)
    is_consec = frame_gap == 1                       # (n-1,) -- step j is a real single-frame step
    accel_valid = is_consec[:-1] & is_consec[1:]      # (n-2,) -- steps j and j+1 both real

    vel = np.diff(positions, axis=0) * fps            # (n-1, 3)
    accel = np.diff(vel, axis=0)[accel_valid] * fps   # (n_valid_accel, 3)
    q_vel = np.maximum((robust_std(accel, axis=0) * dt) ** 2, 1e-3) if len(accel) > 5 else np.full(3, 1e-1)

    def rel_angle(qa, qb):
        Ra, Rb = quat_to_R(qa), quat_to_R(qb)
        Rrel = Ra.T @ Rb
        return np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))

    angle = np.array([rel_angle(quats[i], quats[i + 1]) for i in range(n - 1)])  # (n-1,)
    angvel = angle * fps                              # (n-1,) -- magnitude only
    angaccel = np.diff(angvel)[accel_valid] * fps      # (n_valid_accel,)
    q_angvel_scalar = max((robust_std(angaccel) * dt) ** 2, 1e-4) if len(angaccel) > 5 else 1e-2

    return np.diag(np.concatenate([
        [1.0, 1.0, 1.0],              # position (mm^2/step): small, mostly from integrated velocity
        q_vel,                        # velocity (mm/s)^2/step: from measured acceleration
        [1e-5, 1e-5, 1e-5, 1e-5],     # quaternion: small, mostly from integrated angular velocity
        [q_angvel_scalar] * 3,        # angular velocity (rad/s)^2/step: from measured angular accel
    ]))


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", type=str, default=str(ROOT / "images"))
    ap.add_argument("--calib", type=str, default=str(ROOT / "camera_calibration_v2.yaml"))
    ap.add_argument("--geometry", type=str, default=str(ROOT / "calibration_output" / "geometry.yaml"))
    ap.add_argument("--timestamps-csv", type=str,
                     default=str(ROOT / "dataset_20260805_223546_032501" /
                                 "dataset_frame_timestamps_20260805_223546_032501.csv"))
    ap.add_argument("--out", type=str, default=str(ROOT / "src_classic" / "tracking_result.csv"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None, help="default: all available frames")
    ap.add_argument("--cache-permutation", action="store_true",
                     help="try the previous frame's marker assignment first and skip the full "
                          "24-permutation search if it already fits to <2px -- a speed optimization "
                          "for constrained hardware (e.g. Raspberry Pi 4); off by default so the "
                          "pipeline matches the validated full-search behavior exactly")
    args = ap.parse_args()

    image_files = sorted(glob.glob(str(Path(args.images_dir) / "frame_*.png")))
    n_available = len(image_files)
    count = args.count if args.count is not None else n_available - args.start
    end = min(args.start + count, n_available)
    frame_indices = list(range(args.start, end))
    print(f"{n_available} frames available; processing {len(frame_indices)} "
          f"(frames {args.start}..{end-1})")

    cal = yaml.safe_load(open(args.calib))
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    K[:2, :] *= 0.5  # calibration images are 1280x800; recording is 640x400 (same FOV, binned)
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)

    geom = yaml.safe_load(open(args.geometry))
    marker_pos = {m["id"]: np.array(m["position"], dtype=np.float64) for m in geom["markers"]}

    ts = pd.read_csv(args.timestamps_csv)
    fps = 1000.0 / ts["elapsed_ms"].diff().median()

    # ---- Stage 1-4: per-frame detection + independent pose solve ----
    print("Detecting markers and solving pose independently for every frame...")
    t0 = time.time()
    raw_pos = np.full((len(frame_indices), 3), np.nan)
    raw_quat = np.full((len(frame_indices), 4), np.nan)
    raw_err = np.full(len(frame_indices), np.nan)
    n_permutation_evals = 0
    prev_perm = None

    for k, i in enumerate(frame_indices):
        gray = cv2.imread(image_files[i], cv2.IMREAD_GRAYSCALE)
        pts, areas = detect_blobs(gray)
        if len(pts) > 4:
            pts = pts[np.argsort(areas)[::-1][:4]]
        if len(pts) != 4:
            prev_perm = None
            continue

        preferred = prev_perm if args.cache_permutation else None
        result = solve_frame_pose(pts.astype(np.float64), marker_pos, K, dist, preferred_perm=preferred)
        if result is None:
            prev_perm = None
            continue

        rvec, tvec, err, perm = result
        raw_pos[k] = tvec.ravel()
        raw_quat[k] = rvec_to_quat(rvec)
        raw_err[k] = err
        prev_perm = perm
        n_permutation_evals += 1

        if k % 5000 == 0:
            print(f"  {k}/{len(frame_indices)} frames ({time.time()-t0:.1f}s elapsed)")

    elapsed = time.time() - t0
    n_valid = np.sum(~np.isnan(raw_pos[:, 0]))
    print(f"done in {elapsed:.1f}s -> {len(frame_indices)/elapsed:.0f} fps "
          f"({n_valid}/{len(frame_indices)} frames got a valid pose, "
          f"median reprojection error {np.nanmedian(raw_err):.1f}px)")

    # ---- Stage 5: EKF ----
    valid_mask = ~np.isnan(raw_pos[:, 0])
    if valid_mask.sum() < 10:
        raise RuntimeError("Too few valid frames to fit measurement noise / run the EKF.")

    valid_frame_idx = np.array(frame_indices)[valid_mask]
    frame_gap = np.diff(valid_frame_idx)  # length = valid_mask.sum() - 1

    R_meas = estimate_measurement_noise(raw_pos[valid_mask], raw_quat[valid_mask])
    Q_proc = estimate_process_noise(raw_pos[valid_mask], raw_quat[valid_mask], fps, frame_gap)
    print(f"measurement noise (R) position std (mm): {np.sqrt(np.diag(R_meas)[:3])}")
    print(f"process noise (Q) velocity std (mm/s/step): {np.sqrt(np.diag(Q_proc)[3:6])}")
    ekf = PoseEKF(process_noise=Q_proc, measurement_noise=R_meas, dt_ref=1.0 / fps)

    filt_pos = np.full((len(frame_indices), 3), np.nan)
    filt_quat = np.full((len(frame_indices), 4), np.nan)
    filt_vel = np.full((len(frame_indices), 3), np.nan)
    filt_angvel = np.full((len(frame_indices), 3), np.nan)

    last_k = None
    for k in range(len(frame_indices)):
        if np.isnan(raw_pos[k, 0]):
            continue
        if ekf.x is None:
            ekf.initialize(raw_pos[k], raw_quat[k])
            last_k = k
        else:
            ekf.predict(dt=(k - last_k) / fps)
            ekf.update(raw_pos[k], raw_quat[k])
            last_k = k
        filt_pos[k] = ekf.position
        filt_quat[k] = ekf.quaternion
        filt_vel[k] = ekf.velocity
        filt_angvel[k] = ekf.angular_velocity

    print(f"EKF: {ekf.n_updated} updates accepted, {ekf.n_rejected} rejected as outliers, "
          f"{ekf.n_reinitialized} forced recoveries "
          f"(NIS gate at p=0.999, threshold={ekf.nis_threshold:.1f})")

    # ---- write output ----
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "time_s",
                    "raw_x_mm", "raw_y_mm", "raw_z_mm", "raw_qw", "raw_qx", "raw_qy", "raw_qz",
                    "raw_reproj_err_px",
                    "filt_x_mm", "filt_y_mm", "filt_z_mm", "filt_qw", "filt_qx", "filt_qy", "filt_qz",
                    "vx_mm_s", "vy_mm_s", "vz_mm_s", "wx_rad_s", "wy_rad_s", "wz_rad_s"])
        for k, i in enumerate(frame_indices):
            w.writerow([i, i / fps,
                        *raw_pos[k], *raw_quat[k], raw_err[k],
                        *filt_pos[k], *filt_quat[k],
                        *(filt_vel[k] * fps), *filt_angvel[k]])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
