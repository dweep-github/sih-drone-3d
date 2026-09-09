"""Validation and reporting module for SIH26158 3D reconstruction."""

from reconstruction.validation.geometry import (
    compute_camera_center,
    compute_relative_rotation_angle_deg,
    compute_reprojection_errors,
    detect_jumps,
    project_3d_to_2d,
    qvec_to_rotmat,
)
from reconstruction.validation.model_io import (
    Camera,
    ImagePose,
    Point3D,
    ReconstructionModel,
    read_colmap_model,
)
from reconstruction.validation.pose_report import (
    count_input_images,
    find_sparse_model_dir,
    generate_reconstruction_report,
    inspect_sparse_model,
    validate_reconstruction,
)

__all__ = [
    "Camera",
    "ImagePose",
    "Point3D",
    "ReconstructionModel",
    "read_colmap_model",
    "compute_camera_center",
    "compute_relative_rotation_angle_deg",
    "compute_reprojection_errors",
    "detect_jumps",
    "project_3d_to_2d",
    "qvec_to_rotmat",
    "count_input_images",
    "find_sparse_model_dir",
    "generate_reconstruction_report",
    "inspect_sparse_model",
    "validate_reconstruction",
]
