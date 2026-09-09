"""Similarity Alignment Module (Procrustes / Umeyama Algorithm).

Implements closed-form 3D similarity transformation (scale, rotation, translation):
    X_geo = s * R * X_local + t

Features:
- Minimum correspondence enforcement (N >= 3)
- Degeneracy detection (zero variance, collinearity, non-finite values)
- Reflection guard ensuring det(R) = +1 in SO(3)
- Camera pose transformation (centers and orientations) in OpenCV convention
- Residual analysis (per-point residuals, RMSE, max error)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger("georeferencing.alignment")


@dataclass
class SimilarityTransform:
    """Estimated similarity transformation parameters and quality metrics."""
    scale: float
    rotation_matrix: List[List[float]]
    translation_vector: List[float]
    rmse: float
    max_error: float
    num_correspondences: int
    residuals: List[float]
    status: str  # "PASS", "FAIL", "WARNING"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": round(self.scale, 8),
            "rotation_matrix": [[round(x, 8) for x in row] for row in self.rotation_matrix],
            "translation_vector": [round(x, 4) for x in self.translation_vector],
            "rmse": round(self.rmse, 4),
            "max_error": round(self.max_error, 4),
            "num_correspondences": self.num_correspondences,
            "residuals": [round(r, 4) for r in self.residuals],
            "status": self.status,
            "notes": self.notes,
        }


def check_point_set_validity(
    points: np.ndarray,
    name: str = "Points",
) -> Tuple[bool, Optional[str]]:
    """Checks for non-finite values and geometric degeneracy (collinearity / zero variance)."""
    if not np.all(np.isfinite(points)):
        return False, f"{name} contains NaN or infinite values"

    n = points.shape[0]
    if n < 3:
        return False, f"{name} has {n} points; at least 3 non-collinear points required"

    # Centered points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Variance check
    variance = np.mean(np.sum(centered**2, axis=1))
    if variance < 1e-10:
        return False, f"{name} points are coincident or have near-zero spatial extent (variance={variance:.2e})"

    # SVD for collinearity check
    _, s_vals, _ = np.linalg.svd(centered)
    # If the second singular value is tiny relative to the first, points are collinear
    if s_vals[0] > 0 and (s_vals[1] / s_vals[0]) < 1e-5:
        return False, f"{name} points are collinear (singular value ratio {s_vals[1]/s_vals[0]:.2e} < 1e-5); 3D rotation unconstrained"

    return True, None


def align_umeyama(
    source_points: np.ndarray | List[List[float]],
    target_points: np.ndarray | List[List[float]],
) -> SimilarityTransform:
    """Computes the optimal similarity transformation: target = s * R * source + t.

    Uses Umeyama's least-squares algorithm with reflection guard det(R) = +1.

    Args:
        source_points: (N, 3) coordinates in local reconstruction frame.
        target_points: (N, 3) coordinates in georeferenced metric frame.

    Returns:
        SimilarityTransform dataclass with scale, R, t, residuals, RMSE, max_error.

    Raises:
        ValueError: If correspondences < 3, shapes mismatch, or geometry is degenerate.
    """
    P = np.asarray(source_points, dtype=np.float64)
    Q = np.asarray(target_points, dtype=np.float64)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"source_points must have shape (N, 3), got {P.shape}")
    if Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError(f"target_points must have shape (N, 3), got {Q.shape}")
    if P.shape[0] != Q.shape[0]:
        raise ValueError(f"Point count mismatch: source has {P.shape[0]}, target has {Q.shape[0]}")

    n = P.shape[0]
    if n < 3:
        raise ValueError(f"Insufficient correspondences for 3D alignment: {n} < 3 required")

    # Degeneracy checks
    valid_p, err_p = check_point_set_validity(P, name="Source (local)")
    if not valid_p:
        raise ValueError(f"Degenerate source geometry: {err_p}")

    valid_q, err_q = check_point_set_validity(Q, name="Target (geo)")
    if not valid_q:
        raise ValueError(f"Degenerate target geometry: {err_q}")

    # 1. Centroids
    mu_p = np.mean(P, axis=0)
    mu_q = np.mean(Q, axis=0)

    # 2. Centered coordinates
    P_c = P - mu_p
    Q_c = Q - mu_q

    # 3. Variance of source points
    var_p = float(np.mean(np.sum(P_c**2, axis=1)))
    if var_p <= 1e-12:
        raise ValueError(f"Source variance too small ({var_p:.2e}); degenerate point set")

    # 4. Cross-covariance matrix: Sigma = (1/N) * Q_c^T @ P_c
    Sigma = (Q_c.T @ P_c) / n

    # 5. SVD
    U, S, Vt = np.linalg.svd(Sigma)

    # 6. Reflection check ensuring det(R) = +1
    det_uv = float(np.linalg.det(U @ Vt))
    D = np.eye(3)
    if det_uv < 0:
        D[2, 2] = -1.0

    R = U @ D @ Vt

    # Re-verify rotation properties
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5):
        raise ValueError("Estimated rotation matrix is not orthogonal")
    if abs(np.linalg.det(R) - 1.0) > 1e-4:
        raise ValueError(f"Estimated rotation matrix has invalid determinant: {np.linalg.det(R):.4f}")

    # 7. Scale
    scale = float(np.trace(D @ np.diag(S)) / var_p)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Estimated non-positive or non-finite scale: {scale}")

    # 8. Translation: t = mu_q - s * R * mu_p
    t = mu_q - scale * (R @ mu_p)

    # 9. Compute predictions and residuals
    Q_pred = (scale * (P @ R.T)) + t
    diffs = Q_pred - Q
    residuals = np.linalg.norm(diffs, axis=1)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    max_err = float(np.max(residuals))

    notes = [
        f"Alignment computed with {n} correspondences using Procrustes/Umeyama similarity transform.",
        f"Scale factor: {scale:.6f}.",
        "Results reflect mathematical best-fit; not survey-grade ground truth.",
    ]

    status = "PASS"
    if rmse > 5.0:
        status = "WARNING"
        notes.append(f"High alignment RMSE ({rmse:.2f} m > 5.0 m). Check timestamp sync and GPS quality.")

    return SimilarityTransform(
        scale=scale,
        rotation_matrix=R.tolist(),
        translation_vector=t.tolist(),
        rmse=rmse,
        max_error=max_err,
        num_correspondences=n,
        residuals=residuals.tolist(),
        status=status,
        notes=notes,
    )


def transform_camera_pose(
    local_position: List[float],
    local_rotation_matrix: Optional[List[List[float]]] = None,
    local_quaternion: Optional[List[float]] = None,
    transform: Optional[SimilarityTransform] = None,
) -> Dict[str, Any]:
    """Transforms a camera pose from local reconstruction frame to georeferenced metric frame.

    OpenCV convention:
      X = right, Y = down, Z = forward
      World-to-camera: x = K [R_cam | t_cam] X_world
      Camera center: C = -R_cam^T @ t_cam

    Coordinate change: X_geo = s * R_sim * X_local + t_sim
      Camera center: C_geo = s * R_sim * C_local + t_sim
      Camera rotation: R_cam_geo = R_cam_local @ R_sim^T
      Camera translation: t_cam_geo = - R_cam_geo @ C_geo
      Quaternion: [qw, qx, qy, qz]

    Args:
        local_position: [X, Y, Z] camera center in local coordinates.
        local_rotation_matrix: 3x3 world-to-camera rotation matrix.
        local_quaternion: [qw, qx, qy, qz] rotation quaternion.
        transform: SimilarityTransform object. If None, returns untransformed pose.

    Returns:
        Dictionary with georeferenced position, rotation_matrix, quaternion, and center.
    """
    pos = np.asarray(local_position, dtype=np.float64)

    # Determine local rotation matrix
    if local_rotation_matrix is not None:
        R_cam_local = np.asarray(local_rotation_matrix, dtype=np.float64)
    elif local_quaternion is not None:
        qw, qx, qy, qz = local_quaternion
        # scipy uses [qx, qy, qz, qw]
        rot = Rotation.from_quat([qx, qy, qz, qw])
        R_cam_local = rot.as_matrix()
    else:
        R_cam_local = np.eye(3)

    if transform is None:
        rot_obj = Rotation.from_matrix(R_cam_local)
        q = rot_obj.as_quat()  # [qx, qy, qz, qw]
        return {
            "position": pos.tolist(),
            "rotation_matrix": R_cam_local.tolist(),
            "rotation_quaternion": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
        }

    s = transform.scale
    R_sim = np.asarray(transform.rotation_matrix, dtype=np.float64)
    t_sim = np.asarray(transform.translation_vector, dtype=np.float64)

    # 1. Transform camera center
    C_geo = (s * (R_sim @ pos)) + t_sim

    # 2. Transform world-to-camera rotation: R_cam_geo = R_cam_local @ R_sim^T
    R_cam_geo = R_cam_local @ R_sim.T

    # Ensure valid rotation
    rot_geo = Rotation.from_matrix(R_cam_geo)
    q_geo = rot_geo.as_quat()  # [qx, qy, qz, qw]
    qw_geo, qx_geo, qy_geo, qz_geo = float(q_geo[3]), float(q_geo[0]), float(q_geo[1]), float(q_geo[2])

    return {
        "position": [round(float(v), 4) for v in C_geo],
        "rotation_matrix": [[round(float(v), 6) for v in row] for row in R_cam_geo],
        "rotation_quaternion": [round(qw_geo, 6), round(qx_geo, 6), round(qy_geo, 6), round(qz_geo, 6)],
    }
