"""VGGT (Visual Geometry Grounded Transformer) Pose Engine Integration and Normalization Adapter.

Integrates upstream VGGT (https://github.com/facebookresearch/vggt, arXiv 2503.11651)
as a feed-forward neural geometry and camera pose estimation backend.
Provides frame selection, pose normalization into project standards (OpenCV world-to-camera,
camera center C = -R^T t, scalar-first quaternion [qw, qx, qy, qz]), COLMAP-compatible
export, photogrammetric validation ingestion, and environment diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add project root and tools/vggt to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VGGT_REPO_DIR = PROJECT_ROOT / "tools" / "vggt"
if VGGT_REPO_DIR.is_dir() and str(VGGT_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(VGGT_REPO_DIR))

from reconstruction.colmap.feature_extraction import validate_image_dir
from reconstruction.colmap.utils import is_cuda_available
from reconstruction.validation.geometry import compute_camera_center, qvec_to_rotmat
from reconstruction.validation.model_io import read_colmap_model
from reconstruction.validation.pose_report import (
    count_input_images,
    find_sparse_model_dir,
    validate_reconstruction,
)

logger = logging.getLogger("vggt")


def get_vggt_repo_dir() -> Path:
    """Returns path to the cloned tools/vggt repository."""
    return VGGT_REPO_DIR


def get_vggt_commit() -> Optional[str]:
    """Retrieves the commit hash of the cloned tools/vggt repository."""
    if not (VGGT_REPO_DIR / ".git").exists():
        return None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(VGGT_REPO_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_vggt_environment_info() -> Dict[str, Any]:
    """Inspects the runtime environment and determines VGGT hardware and library compatibility.

    VGGT is a 1B-parameter transformer requiring PyTorch with CUDA acceleration
    (Compute Capability >= 7.0, Ampere+ >= 8.0 for bfloat16) and substantial VRAM (16-24GB+).
    """
    python_ver = sys.version.split()[0]
    cuda_avail = is_cuda_available()
    cuda_ver = None
    gpu_name = "None"
    pytorch_ver = "Missing"

    try:
        import torch
        pytorch_ver = torch.__version__
        if cuda_avail and torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "None (CPU / non-CUDA host)"
    except ImportError:
        pass

    # Check dependencies
    deps_status: Dict[str, bool] = {}
    for pkg in ["einops", "safetensors", "huggingface_hub", "PIL"]:
        try:
            __import__(pkg)
            deps_status[pkg] = True
        except ImportError:
            deps_status[pkg] = False

    repo_present = (VGGT_REPO_DIR / "vggt").is_dir()
    commit_hash = get_vggt_commit()

    blockers: List[str] = []
    if not repo_present:
        blockers.append("VGGT repository not found at tools/vggt.")
    if not all(deps_status.values()):
        missing_pkgs = [pkg for pkg, ok in deps_status.items() if not ok]
        blockers.append(f"Missing required Python dependencies: {', '.join(missing_pkgs)}.")
    if not cuda_avail:
        blockers.append(
            f"VGGT neural inference requires a CUDA-capable GPU and CUDA-enabled PyTorch "
            f"(current CUDA available: False; Host GPU: {gpu_name})."
        )

    is_supported = len(blockers) == 0

    return {
        "repository": "https://github.com/facebookresearch/vggt",
        "repo_path": str(VGGT_REPO_DIR),
        "commit": commit_hash,
        "license": "Meta VGGT License v1 (July 29, 2025; Commercial/Research permitted, non-military)",
        "default_checkpoint": "facebook/VGGT-1B",
        "checkpoint_license": "Research Use (facebook/VGGT-1B) / Commercial Use with form sign-off (facebook/VGGT-1B-Commercial)",
        "os": sys.platform,
        "python": python_ver,
        "pytorch": pytorch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu": gpu_name,
        "dependencies": deps_status,
        "supported": is_supported,
        "blockers": blockers,
        "blocker_reason": " | ".join(blockers) if blockers else None,
    }


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a scalar-first unit quaternion [qw, qx, qy, qz] with qw >= 0.

    Uses Shepperd's algorithm for numerical stability across all rotation angles.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {R.shape}")

    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * math.sqrt(tr + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    if q[0] < 0.0:
        q = -q
    return q


def select_frame_subset(
    image_files: List[Path],
    max_images: Optional[int] = None,
) -> Tuple[List[Path], Dict[int, Dict[str, Any]]]:
    """Selects a representative image subset with uniform temporal downsampling.

    Preserves chronological order, flight trajectory coverage (first and last frames
    are strictly preserved), and records explicit mapping between VGGT input index,
    original filename, and standardized frame ID.
    """
    total = len(image_files)
    if max_images is None or total <= max_images:
        selected = list(image_files)
    else:
        indices = np.round(np.linspace(0, total - 1, max_images)).astype(int)
        # Deduplicate while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        selected = [image_files[i] for i in unique_indices]

    mapping: Dict[int, Dict[str, Any]] = {}
    for vggt_idx, p in enumerate(selected):
        frame_id = p.stem
        mapping[vggt_idx] = {
            "vggt_index": vggt_idx,
            "image_name": p.name,
            "frame_id": frame_id,
            "original_path": str(p),
        }

    return selected, mapping


class VGGTAdapter:
    """Normalizes VGGT model outputs into standardized pose representations and COLMAP formats."""

    @staticmethod
    def normalize_poses(
        extrinsics: np.ndarray,
        mapping: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Converts raw extrinsics [S, 3, 4] into project standard camera representation.

        Conventions:
        - Coordinate system: OpenCV camera convention (X-right, Y-down, Z-forward)
        - Camera/world convention: Camera-from-world X_cam = R * X_world + t
        - Camera position: C = -R^T * t in world frame
        - Quaternion: Scalar-first [qw, qx, qy, qz] with qw >= 0
        - Units / Scale: Relative geometry (arbitrary scale, up-to-scale, not georeferenced)
        """
        extrinsics = np.asarray(extrinsics, dtype=np.float64)
        num_frames = extrinsics.shape[0]

        normalized: List[Dict[str, Any]] = []
        for i in range(num_frames):
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]

            qvec = rotmat_to_qvec(R)
            center = compute_camera_center(qvec, t)

            meta = mapping.get(i, {
                "frame_id": f"frame_{i:06d}",
                "image_name": f"frame_{i:06d}.jpg",
            })

            normalized.append({
                "frame_id": meta["frame_id"],
                "image_name": meta["image_name"],
                "vggt_index": i,
                "position": [round(float(c), 6) for c in center],
                "rotation_quaternion": [round(float(q), 6) for q in qvec],
                "coordinate_convention": "OpenCV (X-right, Y-down, Z-forward)",
                "pose_type": "camera_from_world (X_cam = R * X_world + t)",
                "scale": "arbitrary_relative_geometry",
            })

        return normalized

    @staticmethod
    def export_colmap_format(
        sparse_dir: Path | str,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        mapping: Dict[int, Dict[str, Any]],
        points_3d: Optional[np.ndarray] = None,
        points_rgb: Optional[np.ndarray] = None,
        image_size: Tuple[int, int] = (518, 518),
    ) -> Path:
        """Exports VGGT intrinsics, extrinsics, and point geometry to COLMAP sparse text format."""
        out_path = Path(sparse_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        num_frames = extrinsics.shape[0]
        width, height = image_size

        # 1. cameras.txt
        # Each frame can have its own PINHOLE camera (fx, fy, cx, cy)
        with open(out_path / "cameras.txt", "w", encoding="utf-8") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {num_frames}\n")
            for i in range(num_frames):
                cam_id = i + 1
                K = intrinsics[i]
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
                f.write(f"{cam_id} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

        # 2. images.txt
        # Image list with two lines per image: pose line and 2D points line
        with open(out_path / "images.txt", "w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {num_frames}\n")
            for i in range(num_frames):
                img_id = i + 1
                cam_id = i + 1
                R = extrinsics[i, :3, :3]
                t = extrinsics[i, :3, 3]
                qvec = rotmat_to_qvec(R)
                img_name = mapping.get(i, {}).get("image_name", f"frame_{i:06d}.jpg")

                f.write(
                    f"{img_id} {qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} "
                    f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {cam_id} {img_name}\n"
                )
                # Second line: 2D feature observations (empty if pure feedforward point map)
                f.write("\n")

        # 3. points3D.txt
        # Point list: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
        with open(out_path / "points3D.txt", "w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            if points_3d is not None and len(points_3d) > 0:
                pts = np.asarray(points_3d).reshape(-1, 3)
                colors = (
                    np.asarray(points_rgb).reshape(-1, 3)
                    if points_rgb is not None
                    else np.full((len(pts), 3), 128, dtype=np.uint8)
                )
                f.write(f"# Number of points: {len(pts)}\n")
                for p_idx in range(len(pts)):
                    pid = p_idx + 1
                    xyz = pts[p_idx]
                    rgb = colors[p_idx]
                    f.write(
                        f"{pid} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                        f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0.0\n"
                    )
            else:
                f.write("# Number of points: 0\n")

        return out_path

    @staticmethod
    def validate(
        model_dir: Path | str,
        image_dir: Optional[Path | str] = None,
        output_report_path: Optional[Path | str] = None,
        trajectory_output_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Runs the standard Step 2 validation engine on exported COLMAP model."""
        return validate_reconstruction(
            model_dir=model_dir,
            image_dir=image_dir,
            output_report_path=output_report_path,
            trajectory_output_path=trajectory_output_path,
        )


def run_vggt(
    images: str | Path,
    output: str | Path,
    checkpoint: str = "facebook/VGGT-1B",
    max_images: Optional[int] = 50,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """Runs VGGT pose estimation and geometry inference or records exact environment blocker.

    Workflow:
    1. Validates input images directory.
    2. Inspects runtime environment (CUDA availability, GPU, dependencies).
    3. Selects representative frame subset according to max_images limit.
    4. Runs VGGT model inference when supported or detects blocker cleanly.
    5. Saves raw predictions under output/predictions/ and metadata.json.
    6. Converts camera predictions into standardized pose representation.
    7. Exports COLMAP-compatible model format into output/sparse/.
    8. Ingests model into existing Step 2 photogrammetric validation engine.
    9. Generates comprehensive vggt_report.json.
    """
    total_start = time.time()
    errors: List[str] = []
    warnings: List[str] = []

    img_path = Path(images).resolve()
    out_path = Path(output).resolve()
    predictions_dir = out_path / "predictions"
    sparse_dir = out_path / "sparse"
    report_path = out_path / "vggt_report.json"
    cameras_path = out_path / "cameras.json"
    mapping_path = out_path / "index_mapping.json"
    metadata_path = out_path / "metadata.json"

    out_path.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate input images directory
    if not img_path.is_dir():
        err = f"Image directory does not exist: {img_path}"
        errors.append(err)
        report = {
            "status": "FAIL",
            "failure_type": "integration_failure",
            "method": "VGGT",
            "model": "VGGT-1B",
            "checkpoint": checkpoint,
            "input_images": 0,
            "processed_images": 0,
            "runtime_seconds": 0.0,
            "peak_gpu_memory_mb": None,
            "camera_predictions": 0,
            "reprojection": {"status": "NOT_AVAILABLE", "reason": err},
            "trajectory": {"status": "NOT_AVAILABLE", "reason": err},
            "warnings": warnings,
            "errors": errors,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        raise FileNotFoundError(err)

    try:
        image_files = validate_image_dir(img_path)
        total_images = len(image_files)
        if total_images == 0:
            raise ValueError(f"No supported images found in directory: {img_path}")
    except ValueError as val_err:
        err = str(val_err)
        errors.append(err)
        report = {
            "status": "FAIL",
            "failure_type": "integration_failure",
            "method": "VGGT",
            "model": "VGGT-1B",
            "checkpoint": checkpoint,
            "input_images": 0,
            "processed_images": 0,
            "runtime_seconds": 0.0,
            "peak_gpu_memory_mb": None,
            "camera_predictions": 0,
            "reprojection": {"status": "NOT_AVAILABLE", "reason": err},
            "trajectory": {"status": "NOT_AVAILABLE", "reason": err},
            "warnings": warnings,
            "errors": errors,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        raise ValueError(err)

    # 2. Select representative frame subset
    selected_files, index_mapping = select_frame_subset(image_files, max_images=max_images)
    num_selected = len(selected_files)

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(index_mapping, f, indent=2)

    # 3. Inspect environment
    env_info = get_vggt_environment_info()
    target_device = device or ("cuda" if env_info["cuda_available"] else "cpu")

    # If not supported and not in mock mode, report BLOCKED
    if not env_info["supported"] and not mock_mode:
        blocker_msg = env_info["blocker_reason"]
        warnings.append(f"Execution blocked: {blocker_msg}")
        report = {
            "status": "BLOCKED",
            "failure_type": "runtime_resource_failure",
            "method": "VGGT",
            "model": "VGGT-1B",
            "checkpoint": checkpoint,
            "input_images": total_images,
            "processed_images": 0,
            "runtime_seconds": 0.0,
            "model_load_time_seconds": 0.0,
            "inference_time_seconds": 0.0,
            "postprocess_time_seconds": 0.0,
            "peak_gpu_memory_mb": None,
            "camera_predictions": 0,
            "blocker_reason": blocker_msg,
            "reprojection": {
                "status": "NOT_AVAILABLE",
                "reason": f"Execution blocked: {blocker_msg}",
            },
            "trajectory": {
                "status": "NOT_AVAILABLE",
                "reason": f"Execution blocked: {blocker_msg}",
            },
            "warnings": warnings,
            "errors": errors,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    # 4. Model Loading & Inference
    load_start = time.time()
    model_load_time = 0.0
    infer_time = 0.0
    peak_gpu_mb: Optional[float] = None

    try:
        import torch
        import torch.nn.functional as F

        if mock_mode:
            # Synthetic predictions for verification and testing
            model_load_time = 0.01
            infer_start = time.time()
            extrinsics = np.zeros((num_selected, 3, 4), dtype=np.float64)
            intrinsics = np.zeros((num_selected, 3, 3), dtype=np.float64)
            depth_maps = np.ones((num_selected, 518, 518), dtype=np.float32) * 10.0
            depth_conf = np.ones((num_selected, 518, 518), dtype=np.float32) * 5.0

            for i in range(num_selected):
                # Identity rotation, stepping 0.5 along X axis
                extrinsics[i, :3, :3] = np.eye(3)
                extrinsics[i, :3, 3] = np.array([-float(i) * 0.5, 0.0, 0.0])
                intrinsics[i] = np.array([
                    [500.0, 0.0, 259.0],
                    [0.0, 500.0, 259.0],
                    [0.0, 0.0, 1.0],
                ])
            points_3d = np.array([
                [0.0, 0.0, 10.0],
                [1.0, 0.0, 10.0],
                [0.0, 1.0, 10.0],
            ], dtype=np.float64)
            points_rgb = np.full((3, 3), 200, dtype=np.uint8)
            infer_time = round(time.time() - infer_start, 4)

        else:
            # Full PyTorch Inference
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images_square
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            from vggt.utils.geometry import unproject_depth_map_to_point_map

            if target_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            else:
                dtype = torch.float32

            model = VGGT()
            # Attempt pretrained load
            try:
                _URL = f"https://huggingface.co/{checkpoint}/resolve/main/model.pt"
                model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, progress=True))
            except Exception:
                model = VGGT.from_pretrained(checkpoint)

            model.eval()
            model = model.to(target_device)
            model_load_time = round(time.time() - load_start, 4)

            # Preprocess images
            infer_start = time.time()
            selected_paths_str = [str(p) for p in selected_files]
            images_tensor, original_coords = load_and_preprocess_images_square(selected_paths_str, 1024)
            images_tensor = images_tensor.to(target_device)

            # Resize to 518x518 for VGGT
            vggt_resolution = 518
            vggt_images = F.interpolate(
                images_tensor, size=(vggt_resolution, vggt_resolution), mode="bilinear", align_corners=False
            )

            with torch.no_grad():
                autocast_ctx = (
                    torch.cuda.amp.autocast(dtype=dtype)
                    if target_device.startswith("cuda") and torch.cuda.is_available()
                    else torch.cpu.amp.autocast(dtype=dtype) if hasattr(torch.cpu.amp, "autocast") else nullcontext()
                )
                with autocast_ctx:
                    vggt_images = vggt_images[None]
                    aggregated_tokens_list, ps_idx = model.aggregator(vggt_images)
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(pose_enc, vggt_images.shape[-2:])
                    depth_map_t, depth_conf_t = model.depth_head(aggregated_tokens_list, vggt_images, ps_idx)

            extrinsics = extrinsic_t.squeeze(0).cpu().numpy()
            intrinsics = intrinsic_t.squeeze(0).cpu().numpy()
            depth_maps = depth_map_t.squeeze(0).cpu().numpy()
            depth_conf = depth_conf_t.squeeze(0).cpu().numpy()

            infer_time = round(time.time() - infer_start, 4)
            if target_device.startswith("cuda") and torch.cuda.is_available():
                peak_gpu_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
                torch.cuda.empty_cache()

            # Unproject point map
            pts_3d_full = unproject_depth_map_to_point_map(depth_maps, extrinsics, intrinsics)
            conf_mask = depth_conf >= 5.0
            points_3d = pts_3d_full[conf_mask][:50000]
            points_rgb = None

        # 5. Save Raw Predictions
        post_start = time.time()
        np.savez_compressed(
            predictions_dir / "predictions.npz",
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth_maps=depth_maps,
            depth_conf=depth_conf,
        )

        metadata = {
            "method": "VGGT",
            "model": "VGGT-1B",
            "checkpoint": checkpoint,
            "commit": env_info["commit"],
            "device": target_device,
            "resolution": [518, 518],
            "input_images_count": total_images,
            "processed_images_count": num_selected,
            "model_load_time_seconds": model_load_time,
            "inference_time_seconds": infer_time,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 6. Normalize Camera Poses
        normalized_poses = VGGTAdapter.normalize_poses(extrinsics, index_mapping)
        with open(cameras_path, "w", encoding="utf-8") as f:
            json.dump(normalized_poses, f, indent=2)

        # 7. Export COLMAP Sparse Model
        VGGTAdapter.export_colmap_format(
            sparse_dir=sparse_dir,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            mapping=index_mapping,
            points_3d=points_3d if 'points_3d' in locals() else None,
            points_rgb=points_rgb if 'points_rgb' in locals() else None,
            image_size=(518, 518),
        )

        # 8. Photogrammetric Validation Ingestion
        val_report_path = out_path / "pose_report.json"
        traj_report_path = out_path / "trajectory.json"
        val_result = VGGTAdapter.validate(
            model_dir=sparse_dir,
            image_dir=img_path,
            output_report_path=val_report_path,
            trajectory_output_path=traj_report_path,
        )

        postprocess_time = round(time.time() - post_start, 4)
        total_runtime = round(time.time() - total_start, 4)

        rep = val_result.get("reprojection", {})
        traj = val_result.get("trajectory", {})

        # Reprojection status handling: Feedforward predictions lack 2D-3D observation tracks
        if rep.get("status") != "AVAILABLE":
            rep_status = {
                "status": "NOT_AVAILABLE",
                "reason": "Feedforward neural model does not produce multi-view feature correspondences across views.",
            }
        else:
            rep_status = rep

        report = {
            "status": "PASS" if len(normalized_poses) > 0 else "FAIL",
            "method": "VGGT",
            "model": "VGGT-1B",
            "checkpoint": checkpoint,
            "input_images": total_images,
            "processed_images": num_selected,
            "runtime_seconds": total_runtime,
            "model_load_time_seconds": model_load_time,
            "inference_time_seconds": infer_time,
            "postprocess_time_seconds": postprocess_time,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "camera_predictions": len(normalized_poses),
            "reprojection": rep_status,
            "trajectory": traj,
            "warnings": warnings + val_result.get("warnings", []),
            "errors": errors + val_result.get("errors", []),
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    except Exception as exc:
        total_runtime = round(time.time() - total_start, 4)
        err = f"VGGT execution failed: {exc}"
        errors.append(err)
        report = {
            "status": "FAIL",
            "failure_type": "model_inference_failure",
            "method": "VGGT",
            "model": "VGGT-1B",
            "checkpoint": checkpoint,
            "input_images": total_images,
            "processed_images": num_selected,
            "runtime_seconds": total_runtime,
            "model_load_time_seconds": model_load_time,
            "inference_time_seconds": infer_time,
            "postprocess_time_seconds": 0.0,
            "peak_gpu_memory_mb": peak_gpu_mb,
            "camera_predictions": 0,
            "reprojection": {"status": "NOT_AVAILABLE", "reason": err},
            "trajectory": {"status": "NOT_AVAILABLE", "reason": err},
            "warnings": warnings,
            "errors": errors,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report


class nullcontext:
    """Context manager for conditional autocasting."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT Pose Engine CLI (Person 1 - Step 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", required=True, type=Path, help="Input images directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reconstruction/test_output/vggt"),
        help="Directory to save VGGT outputs and reports",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="facebook/VGGT-1B",
        help="Hugging Face checkpoint identifier",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum representative images to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda / cpu)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Run mock verification pass",
    )

    args = parser.parse_args()

    try:
        res = run_vggt(
            images=args.images,
            output=args.output,
            checkpoint=args.checkpoint,
            max_images=args.max_images,
            device=args.device,
            mock_mode=args.mock,
        )
        print(f"\n[VGGT] Execution completed with status: {res.get('status')}")
        print(f"  Processed images:    {res.get('processed_images')}")
        print(f"  Camera predictions:  {res.get('camera_predictions')}")
        print(f"  Total runtime:       {res.get('runtime_seconds')} s")
        if res.get("status") == "BLOCKED":
            print(f"  Blocker reason:      {res.get('blocker_reason')}")
        sys.exit(0 if res.get("status") in ["PASS", "BLOCKED"] else 1)
    except Exception as exc:
        print(f"[ERROR] VGGT failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
