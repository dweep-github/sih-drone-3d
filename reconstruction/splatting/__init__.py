"""Gaussian Splatting / Nerfstudio Splatfacto module for SIH26158 (Step 9)."""

from reconstruction.splatting.conversion import (
    apply_scene_normalization_to_points,
    apply_scene_normalization_to_poses,
    camera_record_to_nerfstudio_transform,
    compute_scene_normalization,
    get_convention_documentation,
    invert_scene_normalization_points,
    nerfstudio_c2w_to_opencv_c2w,
    opencv_c2w_to_nerfstudio_c2w,
    quaternion_to_rotmat,
    rotmat_to_quaternion,
)
from reconstruction.splatting.dataset import (
    match_images_and_poses,
    prepare_splatfacto_dataset,
)
from reconstruction.splatting.export import (
    export_splat_bundle,
    validate_gaussian_splat,
    validate_render,
)
from reconstruction.splatting.splatfacto import (
    SplatfactoConfig,
    check_splatfacto_environment,
    run_real_splat_smoke_test,
    run_splatfacto_training,
)

__all__ = [
    "opencv_c2w_to_nerfstudio_c2w",
    "nerfstudio_c2w_to_opencv_c2w",
    "camera_record_to_nerfstudio_transform",
    "quaternion_to_rotmat",
    "rotmat_to_quaternion",
    "compute_scene_normalization",
    "apply_scene_normalization_to_poses",
    "apply_scene_normalization_to_points",
    "invert_scene_normalization_points",
    "get_convention_documentation",
    "match_images_and_poses",
    "prepare_splatfacto_dataset",
    "SplatfactoConfig",
    "check_splatfacto_environment",
    "run_splatfacto_training",
    "run_real_splat_smoke_test",
    "validate_gaussian_splat",
    "export_splat_bundle",
    "validate_render",
]
