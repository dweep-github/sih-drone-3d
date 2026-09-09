"""Dataset preparation module for Nerfstudio Splatfacto (SIH26158 Step 9).

Prepares the standardized dataset required by Splatfacto:
1. Validates and matches images with camera poses by frame ID
2. Checks image readability and dimensions
3. Converts camera poses to Nerfstudio convention
4. Computes and applies coordinate normalization
5. Prepares Step 8 point cloud for Gaussian initialization (normalized and linked)
6. Exports transforms.json and transform.json
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from reconstruction.pointcloud.export import (
    PointCloudData,
    export_point_cloud_ply,
    load_point_cloud,
)
from reconstruction.splatting.conversion import (
    apply_scene_normalization_to_points,
    apply_scene_normalization_to_poses,
    camera_record_to_nerfstudio_transform,
    compute_scene_normalization,
    get_convention_documentation,
)

logger = logging.getLogger("splatting.dataset")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Reads image dimensions (width, height) using PIL."""
    with Image.open(image_path) as img:
        return img.width, img.height


def is_image_readable(image_path: Path) -> bool:
    """Verifies that the image file exists and can be opened and decoded."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_camera_poses(cameras_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads camera pose records from a cameras.json file."""
    path = Path(cameras_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cameras file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Cameras file '{path.name}' must contain a JSON list of camera dictionaries.")

    return data


def match_images_and_poses(
    image_dir: Union[str, Path],
    camera_poses: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Matches images on disk with camera pose records, preserving stable frame IDs.

    Matching strategy:
    1. Exact match on 'image_name' (e.g. 'frame_000127.jpg')
    2. Exact match on stem / 'frame_id' (e.g. 'frame_000127')

    Returns:
        Tuple of (matched_pairs, unreadable_images, missing_pose_images).
    """
    img_dir = Path(image_dir).resolve()
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Build pose lookup tables
    pose_by_name: Dict[str, Dict[str, Any]] = {}
    pose_by_stem: Dict[str, Dict[str, Any]] = {}

    for cam in camera_poses:
        img_name = cam.get("image_name")
        frame_id = cam.get("frame_id")
        if img_name:
            pose_by_name[img_name.lower()] = cam
            pose_by_stem[Path(img_name).stem.lower()] = cam
        if frame_id:
            pose_by_stem[frame_id.lower()] = cam

    matched_pairs: List[Dict[str, Any]] = []
    unreadable_images: List[str] = []
    missing_pose_images: List[str] = []

    # Sort files alphabetically for stable deterministic ordering
    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ])

    for img_file in image_files:
        if not is_image_readable(img_file):
            unreadable_images.append(img_file.name)
            continue

        # Look up pose
        pose = pose_by_name.get(img_file.name.lower()) or pose_by_stem.get(img_file.stem.lower())
        if not pose:
            missing_pose_images.append(img_file.name)
            continue

        frame_id = pose.get("frame_id", img_file.stem)
        matched_pairs.append({
            "image_path": img_file,
            "image_name": img_file.name,
            "frame_id": frame_id,
            "pose": pose,
        })

    return matched_pairs, unreadable_images, missing_pose_images


def prepare_splatfacto_dataset(
    images_dir: Union[str, Path],
    cameras_path: Union[str, Path],
    output_dir: Union[str, Path],
    pointcloud_path: Optional[Union[str, Path]] = None,
    georef_metadata_path: Optional[Union[str, Path]] = None,
    normalize_coords: bool = True,
    target_radius: float = 1.0,
    copy_images: bool = True,
) -> Dict[str, Any]:
    """Prepares the complete dataset required by Nerfstudio Splatfacto.

    Steps:
    1. Match images with camera poses, validating readability and IDs.
    2. Convert camera poses to Nerfstudio OpenGL convention.
    3. Compute scene normalization (center offset and scale) if enabled.
    4. Process Step 8 point cloud for Gaussian initialization (normalize coordinates and link).
    5. Generate transforms.json (Nerfstudio dataset format).
    6. Generate transform.json (georeference & normalization records).

    Args:
        images_dir: Directory containing input flight images.
        cameras_path: Path to cameras.json.
        output_dir: Output dataset directory.
        pointcloud_path: Optional path to Step 8 pointcloud.ply or points3D.txt.
        georef_metadata_path: Optional path to georeferencing metadata.json.
        normalize_coords: Whether to center and scale camera poses and points.
        target_radius: Normalization bounding radius (default: 1.0).
        copy_images: Whether to copy images into dataset_dir/images (recommended for portability).

    Returns:
        Summary dictionary of prepared dataset.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir = out_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load poses and match images
    camera_poses = load_camera_poses(cameras_path)
    matched, unreadable, missing_pose = match_images_and_poses(images_dir, camera_poses)

    if len(matched) == 0:
        raise ValueError(
            f"No valid matched image-pose pairs found between '{images_dir}' and '{cameras_path}'. "
            f"(Unreadable: {len(unreadable)}, Missing pose: {len(missing_pose)})"
        )

    # 2. Extract image dimensions and intrinsics from first readable matched image
    first_img_path = matched[0]["image_path"]
    w, h = get_image_dimensions(first_img_path)

    # Default pinhole intrinsics (~55-60 deg FOV standard for drone sensors)
    fl_x = fl_y = float(1.2 * max(w, h))
    cx = float(w / 2.0)
    cy = float(h / 2.0)

    # Check if cameras.json contains custom intrinsics
    first_pose = matched[0]["pose"]
    if "fl_x" in first_pose and "fl_y" in first_pose:
        fl_x = float(first_pose["fl_x"])
        fl_y = float(first_pose["fl_y"])
    if "cx" in first_pose and "cy" in first_pose:
        cx = float(first_pose["cx"])
        cy = float(first_pose["cy"])

    # 3. Convert poses to Nerfstudio 4x4 matrices
    raw_transforms: List[np.ndarray] = []
    camera_centers: List[np.ndarray] = []

    for item in matched:
        T_c2w = camera_record_to_nerfstudio_transform(item["pose"])
        raw_transforms.append(T_c2w)
        camera_centers.append(T_c2w[:3, 3])

    centers_arr = np.asarray(camera_centers, dtype=np.float64)

    # 4. Coordinate Normalization
    if normalize_coords and len(centers_arr) > 0:
        center_offset, scale_factor = compute_scene_normalization(
            centers_arr, target_radius=target_radius
        )
        norm_transforms = apply_scene_normalization_to_poses(
            raw_transforms, center_offset=center_offset, scale_factor=scale_factor
        )
        norm_applied = True
    else:
        center_offset = np.zeros(3, dtype=np.float64)
        scale_factor = 1.0
        norm_transforms = raw_transforms
        norm_applied = False

    # 5. Handle Point-Cloud Initialization (Step 8)
    pcd_init_info: Dict[str, Any] = {
        "used": False,
        "reason": "No point-cloud initialization path provided.",
    }
    pcd_filename: Optional[str] = None

    if pointcloud_path is not None:
        pcd_path = Path(pointcloud_path).resolve()
        if pcd_path.exists():
            try:
                pcd = load_point_cloud(pcd_path, georef_metadata_path=georef_metadata_path)
                if pcd.num_points > 0:
                    if not np.all(np.isfinite(pcd.points)):
                        raise ValueError("Point cloud contains non-finite coordinates.")

                    # Apply identical scene normalization to point cloud
                    norm_points = apply_scene_normalization_to_points(
                        pcd.points, center_offset=center_offset, scale_factor=scale_factor
                    )

                    normalized_pcd = PointCloudData(
                        points=norm_points,
                        colors=pcd.colors,
                        source_method=f"normalized_{pcd.source_method}",
                        crs=pcd.crs,
                        units="normalized_units" if norm_applied else pcd.units,
                        coordinate_convention="Nerfstudio scene coordinates",
                        is_georeferenced=pcd.is_georeferenced,
                    )

                    pcd_dest = out_dir / "pointcloud.ply"
                    export_point_cloud_ply(normalized_pcd, pcd_dest)
                    pcd_filename = "pointcloud.ply"

                    pcd_init_info = {
                        "used": True,
                        "source_path": str(pcd_path),
                        "num_points": normalized_pcd.num_points,
                        "has_colors": normalized_pcd.has_colors,
                        "ply_file": pcd_filename,
                        "normalized": norm_applied,
                    }
                    logger.info("Point-cloud initialization prepared: %d points written to %s", normalized_pcd.num_points, pcd_dest)
                else:
                    pcd_init_info = {"used": False, "reason": "Point cloud is empty (0 points)."}
            except Exception as exc:
                logger.warning("Point-cloud initialization failed to process: %s", exc)
                pcd_init_info = {"used": False, "reason": f"Processing failed: {exc}"}
        else:
            pcd_init_info = {"used": False, "reason": f"File does not exist: {pcd_path}"}

    # 6. Build frames list and copy images
    frames: List[Dict[str, Any]] = []

    for idx, item in enumerate(matched):
        src_img = item["image_path"]
        frame_id = item["frame_id"]

        if copy_images:
            dst_img = images_out_dir / src_img.name
            if not dst_img.is_file() or dst_img.stat().st_size != src_img.stat().st_size:
                shutil.copy2(src_img, dst_img)
            rel_img_path = f"images/{src_img.name}"
        else:
            # Relative path from out_dir to original image
            try:
                rel_img_path = str(src_img.relative_to(out_dir)).replace("\\", "/")
            except ValueError:
                rel_img_path = str(src_img).replace("\\", "/")

        T_matrix = norm_transforms[idx].tolist()

        frame_entry = {
            "file_path": rel_img_path,
            "transform_matrix": T_matrix,
            "colmap_im_id": idx + 1,
            "frame_id": frame_id,
        }
        frames.append(frame_entry)

    # 7. Write transforms.json (Nerfstudio dataset format)
    transforms_dict: Dict[str, Any] = {
        "camera_model": "OPENCV",
        "fl_x": round(fl_x, 4),
        "fl_y": round(fl_y, 4),
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "w": int(w),
        "h": int(h),
        "frames": frames,
    }

    if pcd_filename is not None:
        transforms_dict["ply_file_path"] = pcd_filename

    transforms_path = out_dir / "transforms.json"
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(transforms_dict, f, indent=2)

    # 8. Write transform.json (georeferencing & normalization metadata)
    # Load Step 7 georeferencing metadata if available
    geo_meta_dict: Optional[Dict[str, Any]] = None
    if georef_metadata_path is not None:
        gpath = Path(georef_metadata_path).resolve()
        if gpath.is_file():
            with open(gpath, "r", encoding="utf-8") as f:
                geo_meta_dict = json.load(f)

    source_crs = geo_meta_dict.get("target_crs") if geo_meta_dict else None
    source_units = "metres" if source_crs else "local_unscaled"

    transform_meta: Dict[str, Any] = {
        "dataset_name": out_dir.name,
        "conventions": get_convention_documentation(),
        "coordinate_system": {
            "source_crs": source_crs,
            "source_units": source_units,
            "is_georeferenced": source_crs is not None,
        },
        "normalization": {
            "applied": norm_applied,
            "center_offset": [round(float(c), 6) for c in center_offset],
            "scale_factor": round(float(scale_factor), 8),
            "target_radius": target_radius,
            "formula": "X_norm = scale_factor * (X_source - center_offset)",
            "inverse_formula": "X_source = (X_norm / scale_factor) + center_offset",
        },
        "georeferencing": geo_meta_dict,
        "pointcloud_initialization": pcd_init_info,
        "summary": {
            "total_images_processed": len(matched),
            "unreadable_images_count": len(unreadable),
            "missing_pose_images_count": len(missing_pose),
        },
    }

    transform_meta_path = out_dir / "transform.json"
    with open(transform_meta_path, "w", encoding="utf-8") as f:
        json.dump(transform_meta, f, indent=2)

    logger.info("Prepared Splatfacto dataset in %s (%d frames)", out_dir, len(frames))

    return {
        "dataset_dir": str(out_dir),
        "transforms_path": str(transforms_path),
        "transform_meta_path": str(transform_meta_path),
        "num_frames": len(frames),
        "image_dimensions": [w, h],
        "pointcloud_initialization": pcd_init_info,
        "normalization": transform_meta["normalization"],
        "unreadable_images": unreadable,
        "missing_pose_images": missing_pose,
    }
