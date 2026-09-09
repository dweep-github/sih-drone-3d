"""Geometric computations, camera projections, reprojection errors, and trajectory analysis."""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple
import numpy as np

from reconstruction.validation.model_io import Camera, ReconstructionModel

logger = logging.getLogger(__name__)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Converts a quaternion [qw, qx, qy, qz] (Hamilton convention) to a 3x3 rotation matrix."""
    norm = np.linalg.norm(qvec)
    if norm < 1e-12:
        raise ValueError(f"Invalid near-zero quaternion: {qvec}")
    q = qvec / norm
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    return np.array([
        [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx**2 + qy**2)],
    ], dtype=np.float64)


def compute_camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Calculates camera center in world coordinates: C = -R^T * t.

    COLMAP stores the pose as world-to-camera transformation (x_cam = R * x_world + t).
    Therefore, the camera center C in world coordinates is -R^T * t.
    """
    R = qvec_to_rotmat(qvec)
    return -R.T @ np.asarray(tvec, dtype=np.float64)


def compute_relative_rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Computes relative angular change in degrees between two rotation matrices.

    R_rel = R2 * R1^T
    angle = arccos((trace(R_rel) - 1) / 2)
    """
    R_rel = R2 @ R1.T
    tr = np.trace(R_rel)
    cos_angle = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = math.acos(float(cos_angle))
    return math.degrees(angle_rad)


def project_3d_to_2d(
    xyz: np.ndarray,
    qvec: np.ndarray,
    tvec: np.ndarray,
    camera: Camera,
) -> Tuple[Optional[np.ndarray], bool]:
    """Projects a 3D world point into 2D pixel coordinates using exact camera distortion model.

    Returns:
        (projected_point_2d, is_camera_model_supported)
        If point is behind camera (z_cam <= 0), returns (None, True).
        If camera model is unknown/unsupported, returns (None, False).
    """
    R = qvec_to_rotmat(qvec)
    p_cam = R @ xyz + tvec
    x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]

    # Point must be strictly in front of camera
    if z_c <= 1e-8:
        return None, True

    u = x_c / z_c
    v = y_c / z_c

    model = camera.model_name.upper()
    p = camera.params

    if model == "SIMPLE_PINHOLE":
        # params: [f, cx, cy]
        if len(p) < 3:
            return None, False
        f, cx, cy = p[0], p[1], p[2]
        return np.array([f * u + cx, f * v + cy], dtype=np.float64), True

    elif model == "PINHOLE":
        # params: [fx, fy, cx, cy]
        if len(p) < 4:
            return None, False
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([fx * u + cx, fy * v + cy], dtype=np.float64), True

    elif model == "SIMPLE_RADIAL":
        # params: [f, cx, cy, k1]
        if len(p) < 4:
            return None, False
        f, cx, cy, k1 = p[0], p[1], p[2], p[3]
        r2 = u**2 + v**2
        dist = 1.0 + k1 * r2
        return np.array([f * dist * u + cx, f * dist * v + cy], dtype=np.float64), True

    elif model == "RADIAL":
        # params: [f, cx, cy, k1, k2]
        if len(p) < 5:
            return None, False
        f, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], p[4]
        r2 = u**2 + v**2
        dist = 1.0 + k1 * r2 + k2 * (r2**2)
        return np.array([f * dist * u + cx, f * dist * v + cy], dtype=np.float64), True

    elif model == "OPENCV":
        # params: [fx, fy, cx, cy, k1, k2, p1, p2]
        if len(p) < 8:
            return None, False
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        k1, k2, p1, p2 = p[4], p[5], p[6], p[7]
        r2 = u**2 + v**2
        radial = 1.0 + k1 * r2 + k2 * (r2**2)
        u_dist = u * radial + 2.0 * p1 * u * v + p2 * (r2 + 2.0 * u**2)
        v_dist = v * radial + p1 * (r2 + 2.0 * v**2) + 2.0 * p2 * u * v
        return np.array([fx * u_dist + cx, fy * v_dist + cy], dtype=np.float64), True

    elif model == "OPENCV_FISHEYE":
        # params: [fx, fy, cx, cy, k1, k2, k3, k4]
        if len(p) < 8:
            return None, False
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        k1, k2, k3, k4 = p[4], p[5], p[6], p[7]
        r = math.sqrt(u**2 + v**2)
        if r < 1e-8:
            scale = 1.0
        else:
            theta = math.atan(r)
            theta2 = theta**2
            theta_d = theta * (1.0 + k1 * theta2 + k2 * (theta2**2) + k3 * (theta2**3) + k4 * (theta2**4))
            scale = theta_d / r
        return np.array([fx * scale * u + cx, fy * scale * v + cy], dtype=np.float64), True

    logger.warning("Unsupported camera distortion model: %s", model)
    return None, False


def compute_reprojection_errors(
    model: ReconstructionModel,
) -> Tuple[List[float], str]:
    """Calculates Euclidean reprojection errors (pixels) for all valid 3D observations.

    Returns:
        (list_of_errors_px, status)
        status: 'AVAILABLE' or 'NOT_AVAILABLE'
    """
    errors: List[float] = []

    if not model.images or not model.points3D:
        return errors, "NOT_AVAILABLE"

    for img in model.images.values():
        camera = model.cameras.get(img.camera_id)
        if not camera:
            continue

        qvec, tvec = img.qvec, img.tvec
        for i in range(len(img.point3D_ids)):
            p3d_id = img.point3D_ids[i]
            if p3d_id < 0:
                continue

            pt3d = model.points3D.get(p3d_id)
            if not pt3d:
                continue

            proj, supported = project_3d_to_2d(pt3d.xyz, qvec, tvec, camera)
            if not supported:
                return [], "NOT_AVAILABLE"

            if proj is None:
                continue

            obs_xy = img.xys[i]
            err = float(np.linalg.norm(obs_xy - proj))
            errors.append(err)

    if not errors:
        return [], "NOT_AVAILABLE"

    return errors, "AVAILABLE"


def detect_jumps(
    values: List[float],
    threshold: Optional[float] = None,
    k_mad: float = 3.5,
) -> Tuple[int, float, List[int]]:
    """Detects jumps/outliers in a sequence of consecutive step distances or rotation changes.

    Args:
        values: Sequence of consecutive step values
        threshold: Explicit jump cutoff. If None, computes robust median + k * MAD.
        k_mad: Multiplier for Median Absolute Deviation (default: 3.5).

    Returns:
        (jump_count, max_jump_value, flagged_indices)
    """
    if not values:
        return 0, 0.0, []

    arr = np.array(values, dtype=np.float64)
    max_val = float(np.max(arr))

    if threshold is not None:
        cutoff = float(threshold)
    else:
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        if mad < 1e-7:
            # Low variability; default to 3x median or generous threshold
            cutoff = max(med * 3.0, 1e-4)
        else:
            cutoff = med + k_mad * mad

    flagged = [i for i, v in enumerate(values) if v > cutoff]
    return len(flagged), max_val, flagged
