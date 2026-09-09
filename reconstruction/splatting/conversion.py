"""Camera coordinate convention conversion module for Nerfstudio Splatfacto (SIH26158 Step 9).

Explicitly bridges:
- Source: OpenCV camera convention (X=right, Y=down, Z=forward)
- Target: Nerfstudio / OpenGL camera convention (X=right, Y=up, Z=backward)

Handles:
- Camera-to-world rotation and 4x4 matrix conversion
- Quaternions [qw, qx, qy, qz] (Hamilton convention) to rotation matrices
- Bidirectional scene normalization (centering and scaling)
- Comprehensive documentation of coordinate transforms
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger("splatting.conversion")

# The reflection/axis-flip matrix converting camera frames:
# OpenCV (X-right, Y-down, Z-forward) -> OpenGL/Nerfstudio (X-right, Y-up, Z-backward)
OPENCV_TO_NERFSTUDIO_MATRIX = np.diag([1.0, -1.0, -1.0])


def get_convention_documentation() -> Dict[str, Any]:
    """Returns detailed documentation of source and target camera conventions."""
    return {
        "source_coordinate_convention": {
            "name": "OpenCV / COLMAP camera convention",
            "x_axis": "Right (+X points right in image plane)",
            "y_axis": "Down (+Y points down in image plane)",
            "z_axis": "Forward (+Z points along optical axis into scene)",
            "camera_center": "C = -R_w2c^T @ t_w2c in world frame",
            "quaternion": "[qw, qx, qy, qz] (Hamilton convention, scalar-first)",
            "camera_to_world": "X_world = R_c2w @ x_cam + C",
        },
        "target_coordinate_convention": {
            "name": "Nerfstudio / OpenGL / NeRF camera convention",
            "x_axis": "Right (+X points right in image plane)",
            "y_axis": "Up (+Y points up in image plane)",
            "z_axis": "Backward (+Z points out of screen; camera looks toward -Z)",
            "transform_matrix": "4x4 camera-to-world matrix [R_c2w_nerf | C]",
        },
        "conversion_matrix": OPENCV_TO_NERFSTUDIO_MATRIX.tolist(),
        "rotation_conversion": "R_c2w_nerf = R_c2w_cv @ diag(1, -1, -1)",
        "translation_conversion": "Camera center C is invariant under local camera axis flip",
        "normalization": "X_norm = scale_factor * (X_world - center_offset)",
    }


def quaternion_to_rotmat(qvec: Union[List[float], np.ndarray]) -> np.ndarray:
    """Converts a Hamilton quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix.

    Args:
        qvec: [qw, qx, qy, qz] with scalar qw first.

    Returns:
        3x3 numpy array float64.
    """
    q = np.asarray(qvec, dtype=np.float64)
    if len(q) != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {len(q)}")

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Cannot convert degenerate zero-norm quaternion to rotation matrix")
    q = q / norm

    qw, qx, qy, qz = q
    # Scipy expects [qx, qy, qz, qw]
    rot = Rotation.from_quat([qx, qy, qz, qw])
    return rot.as_matrix()


def rotmat_to_quaternion(rotmat: np.ndarray) -> List[float]:
    """Converts a 3x3 rotation matrix to a Hamilton quaternion [qw, qx, qy, qz]."""
    R = np.asarray(rotmat, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix must have shape (3, 3), got {R.shape}")

    rot = Rotation.from_matrix(R)
    # Scipy returns [qx, qy, qz, qw]
    qx, qy, qz, qw = rot.as_quat()
    return [float(qw), float(qx), float(qy), float(qz)]


def opencv_c2w_to_nerfstudio_c2w(
    R_c2w_cv: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Converts an OpenCV camera-to-world pose into a 4x4 Nerfstudio camera-to-world matrix.

    In OpenCV:
      Column 0 of R_c2w: Right (+X)
      Column 1 of R_c2w: Down (+Y)
      Column 2 of R_c2w: Forward (+Z)

    In Nerfstudio / OpenGL:
      Column 0 of R_c2w: Right (+X)
      Column 1 of R_c2w: Up (-Y_cv)
      Column 2 of R_c2w: Backward (-Z_cv)

    Args:
        R_c2w_cv: 3x3 camera-to-world rotation matrix in OpenCV convention.
        center: 3-element camera center in world coordinates.

    Returns:
        4x4 camera-to-world transformation matrix in Nerfstudio convention.
    """
    R_cv = np.asarray(R_c2w_cv, dtype=np.float64)
    C = np.asarray(center, dtype=np.float64).reshape(3)

    if R_cv.shape != (3, 3):
        raise ValueError(f"R_c2w_cv must have shape (3, 3), got {R_cv.shape}")

    # Flip Y and Z axes
    R_nerf = R_cv @ OPENCV_TO_NERFSTUDIO_MATRIX

    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = R_nerf
    T_c2w[:3, 3] = C
    return T_c2w


def nerfstudio_c2w_to_opencv_c2w(
    T_c2w_nerf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse conversion: 4x4 Nerfstudio camera-to-world matrix to OpenCV R_c2w and center.

    Returns:
        Tuple of (R_c2w_cv [3x3], center [3]).
    """
    T = np.asarray(T_c2w_nerf, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_c2w_nerf must have shape (4, 4), got {T.shape}")

    R_nerf = T[:3, :3]
    C = T[:3, 3]

    # Inverse axis flip (diag(1, -1, -1) is self-inverse)
    R_cv = R_nerf @ OPENCV_TO_NERFSTUDIO_MATRIX
    return R_cv, C


def camera_record_to_nerfstudio_transform(
    cam_dict: Dict[str, Any],
) -> np.ndarray:
    """Extracts position and orientation from a camera dict and converts to Nerfstudio 4x4 c2w.

    Accepts camera records from:
    - `selected/cameras.json`
    - `selected/georeferenced/cameras.json`
    - Step 2 pose validation outputs

    Args:
        cam_dict: Dictionary containing 'position' and ('rotation_matrix' or 'rotation_quaternion').

    Returns:
        4x4 numpy array camera-to-world matrix in Nerfstudio convention.
    """
    if "position" not in cam_dict:
        raise ValueError(f"Camera record missing required 'position' field: {cam_dict.keys()}")

    center = np.asarray(cam_dict["position"], dtype=np.float64)

    # Determine R_c2w
    if "rotation_matrix" in cam_dict and cam_dict["rotation_matrix"] is not None:
        R_c2w = np.asarray(cam_dict["rotation_matrix"], dtype=np.float64)
    elif "rotation_quaternion" in cam_dict and cam_dict["rotation_quaternion"] is not None:
        R_c2w = quaternion_to_rotmat(cam_dict["rotation_quaternion"])
    else:
        # Default identity rotation if unspecified
        R_c2w = np.eye(3, dtype=np.float64)

    return opencv_c2w_to_nerfstudio_c2w(R_c2w, center)


def compute_scene_normalization(
    camera_centers: np.ndarray,
    target_radius: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """Computes bounding-sphere translation offset and scale factor to center and scale a scene.

    Formula:
        center_offset = mean(camera_centers)
        scale_factor = target_radius / (max_dist_from_center + eps)

    Args:
        camera_centers: (N, 3) coordinates of camera centers.
        target_radius: Desired radius of normalized bounding sphere (default: 1.0).

    Returns:
        Tuple of (center_offset [3], scale_factor float).
    """
    pts = np.asarray(camera_centers, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"camera_centers must have shape (N, 3), got {pts.shape}")

    if len(pts) == 0:
        return np.zeros(3, dtype=np.float64), 1.0

    center_offset = np.mean(pts, axis=0)
    diffs = pts - center_offset
    dists = np.linalg.norm(diffs, axis=1)
    max_dist = float(np.max(dists)) if len(dists) > 0 else 0.0

    if max_dist < 1e-6:
        scale_factor = 1.0
    else:
        scale_factor = float(target_radius / max_dist)

    return center_offset, scale_factor


def apply_scene_normalization_to_poses(
    transforms: List[np.ndarray],
    center_offset: np.ndarray,
    scale_factor: float,
) -> List[np.ndarray]:
    """Applies scene normalization (center shift and scale) to a list of 4x4 c2w matrices.

    The rotation is invariant under uniform scaling and translation; only the camera center
    position C is translated and scaled: C_norm = scale * (C - offset).

    Args:
        transforms: List of 4x4 camera-to-world matrices.
        center_offset: 3-element translation offset.
        scale_factor: Uniform scaling multiplier.

    Returns:
        List of 4x4 normalized camera-to-world matrices.
    """
    offset = np.asarray(center_offset, dtype=np.float64).reshape(3)
    norm_transforms: List[np.ndarray] = []

    for T in transforms:
        T_norm = np.copy(T)
        C = T_norm[:3, 3]
        T_norm[:3, 3] = scale_factor * (C - offset)
        norm_transforms.append(T_norm)

    return norm_transforms


def apply_scene_normalization_to_points(
    points: np.ndarray,
    center_offset: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Transforms 3D points by applying scene normalization: P_norm = scale * (P - offset)."""
    P = np.asarray(points, dtype=np.float64)
    offset = np.asarray(center_offset, dtype=np.float64).reshape(3)
    return scale_factor * (P - offset)


def invert_scene_normalization_points(
    points_norm: np.ndarray,
    center_offset: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Inverse transform: converts normalized coordinates back to the original world/metric frame.

    Formula: P_original = (P_norm / scale_factor) + offset
    """
    P_norm = np.asarray(points_norm, dtype=np.float64)
    offset = np.asarray(center_offset, dtype=np.float64).reshape(3)
    if abs(scale_factor) < 1e-12:
        raise ValueError(f"Invalid near-zero scale_factor for inversion: {scale_factor}")
    return (P_norm / scale_factor) + offset
