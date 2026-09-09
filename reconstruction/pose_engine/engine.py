"""Adaptive Pose Engine and Real Execution Gate for Person 1 (SIH26158).

Orchestrates multi-route Structure-from-Motion and neural geometry backends:
FastMap -> COLMAP Global SfM -> VGGT (with COLMAP Incremental as reference baseline).
Enforces pre-flight data integrity gates, applies configurable photogrammetric
validation policies, short-circuits on the first acceptable candidate, standardizes
selected outputs, and records detailed attempt logs in pose_engine_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.colmap.reconstruct import run_reconstruction as run_colmap_reconstruction
from reconstruction.colmap.utils import is_cuda_available
from reconstruction.pose_engine.fastmap import FastMapAdapter, run_fastmap
from reconstruction.pose_engine.global_sfm import run_global_sfm
from reconstruction.pose_engine.vggt import VGGTAdapter, run_vggt
from reconstruction.validation.geometry import compute_camera_center
from reconstruction.validation.model_io import read_colmap_model
from reconstruction.validation.pose_report import (
    DEFAULT_MAX_REPROJECTION_ERROR,
    DEFAULT_MIN_REGISTRATION_RATE,
    SUPPORTED_IMAGE_EXTS,
    find_sparse_model_dir,
    validate_reconstruction,
)

logger = logging.getLogger("adaptive_engine")


@dataclass
class ValidationPolicy:
    """Configurable photogrammetric validation policy.

    NOTE: These thresholds are provisional engineering gates, NOT survey-grade or scientifically validated.
    """
    min_registration_rate: float = DEFAULT_MIN_REGISTRATION_RATE
    max_reprojection_rmse_px: float = DEFAULT_MAX_REPROJECTION_ERROR
    require_trajectory: bool = True
    min_registered_images: int = 2
    max_position_jump: Optional[float] = None
    max_rotation_jump: Optional[float] = None
    policy_type: str = "provisional_engineering_gate"
    policy_note: str = "Provisional engineering thresholds; not survey-grade or scientifically validated."

    def __init__(
        self,
        min_registration_rate: float = DEFAULT_MIN_REGISTRATION_RATE,
        max_reprojection_rmse: Optional[float] = None,
        max_reprojection_rmse_px: Optional[float] = None,
        require_trajectory: bool = True,
        min_registered_images: int = 2,
        max_position_jump: Optional[float] = None,
        max_rotation_jump: Optional[float] = None,
        policy_type: str = "provisional_engineering_gate",
        policy_note: str = "Provisional engineering thresholds; not survey-grade or scientifically validated.",
    ) -> None:
        self.min_registration_rate = min_registration_rate
        if max_reprojection_rmse_px is not None:
            self.max_reprojection_rmse_px = max_reprojection_rmse_px
        elif max_reprojection_rmse is not None:
            self.max_reprojection_rmse_px = max_reprojection_rmse
        else:
            self.max_reprojection_rmse_px = DEFAULT_MAX_REPROJECTION_ERROR
        self.require_trajectory = require_trajectory
        self.min_registered_images = min_registered_images
        self.max_position_jump = max_position_jump
        self.max_rotation_jump = max_rotation_jump
        self.policy_type = policy_type
        self.policy_note = policy_note

    @property
    def max_reprojection_rmse(self) -> float:
        return self.max_reprojection_rmse_px

    @max_reprojection_rmse.setter
    def max_reprojection_rmse(self, val: float) -> None:
        self.max_reprojection_rmse_px = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_registration_rate": self.min_registration_rate,
            "max_reprojection_rmse_px": self.max_reprojection_rmse_px,
            "require_trajectory": self.require_trajectory,
            "policy_type": self.policy_type,
            "min_registered_images": self.min_registered_images,
            "policy_note": self.policy_note,
        }


@dataclass
class PoseResult:
    """Normalized candidate result across all reconstruction backends."""
    method: str
    status: str  # "PASS", "FAIL", "BLOCKED"
    output_path: Path
    runtime_seconds: float
    validation_status: str  # "PASS", "FAIL", "NOT_AVAILABLE"
    registration_rate: float = 0.0
    registered_images: int = 0
    input_images: int = 0
    num_points3D: int = 0
    reprojection_rmse_px: Optional[float] = None
    reprojection_status: str = "NOT_AVAILABLE"  # "AVAILABLE", "NOT_AVAILABLE", "FAIL"
    trajectory_status: str = "NOT_AVAILABLE"    # "AVAILABLE", "NOT_AVAILABLE", "FAIL", "WARNING"
    cameras_path: Optional[Path] = None
    trajectory_path: Optional[Path] = None
    sparse_dir: Optional[Path] = None
    start_time: str = ""
    end_time: str = ""
    blocker_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_report: Dict[str, Any] = field(default_factory=dict)


def run_preflight_checks(
    image_dir: Path | str,
    min_images: int = 20,
) -> Dict[str, Any]:
    """Validates input image directory integrity before any pose estimation backend runs.

    Checks:
    1. Directory existence.
    2. Image discovery with supported extensions.
    3. Minimum image count threshold (rejects 0 images strictly).
    4. Duplicate filename detection.
    5. Image readability and non-zero dimensions.
    """
    path = Path(image_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {path}")

    # Discover images
    candidates = sorted([
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ])

    total_images = len(candidates)
    if total_images == 0:
        raise ValueError(
            f"Preflight check failed: 0 supported images found in '{path}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}."
        )

    if total_images < min_images:
        raise ValueError(
            f"Preflight check failed: Insufficient images ({total_images} found). "
            f"Minimum required threshold is {min_images} images."
        )

    # Check for duplicate filenames (case-insensitive)
    seen_names: Set[str] = set()
    duplicates: List[str] = []
    for img in candidates:
        low = img.name.lower()
        if low in seen_names:
            duplicates.append(img.name)
        seen_names.add(low)

    if duplicates:
        raise ValueError(f"Preflight check failed: Duplicate image filenames detected: {duplicates}")

    # Check readability and dimensions
    checked_count = 0
    dimensions: Dict[str, int] = {}
    corrupted: List[str] = []

    try:
        from PIL import Image as PILImage
        for img_p in candidates:
            try:
                with PILImage.open(img_p) as img_obj:
                    w, h = img_obj.size
                    if w <= 0 or h <= 0:
                        corrupted.append(f"{img_p.name} (invalid dimensions {w}x{h})")
                    dimensions[f"{w}x{h}"] = dimensions.get(f"{w}x{h}", 0) + 1
                    checked_count += 1
            except Exception as exc:
                corrupted.append(f"{img_p.name} ({exc})")
    except ImportError:
        # Fallback if PIL not available
        checked_count = total_images

    if corrupted:
        raise ValueError(f"Preflight check failed: Corrupted or unreadable images found: {corrupted}")

    return {
        "status": "PASS",
        "image_directory": str(path),
        "total_images": total_images,
        "min_images_required": min_images,
        "formats": sorted(list({f.suffix.lower() for f in candidates})),
        "dimension_distribution": dimensions,
        "verified_readable": checked_count,
    }


def evaluate_candidate(
    result: PoseResult,
    policy: ValidationPolicy,
) -> Tuple[bool, Optional[str]]:
    """Evaluates whether a candidate backend result satisfies the photogrammetric policy.

    Returns:
        (is_acceptable, rejection_reason)
    """
    if result.status != "PASS":
        reason = result.blocker_reason or result.failure_reason or f"Backend status is {result.status}"
        return False, f"Backend execution status '{result.status}': {reason}"

    # 1. Registration rate check
    if result.registered_images < policy.min_registered_images:
        return False, f"Registered images ({result.registered_images}) < minimum required ({policy.min_registered_images})"

    if result.registration_rate < policy.min_registration_rate:
        return False, f"Registration rate ({result.registration_rate:.2%}) < policy threshold ({policy.min_registration_rate:.2%})"

    # 2. Reprojection check
    if result.reprojection_status == "AVAILABLE":
        if result.reprojection_rmse_px is not None and result.reprojection_rmse_px > policy.max_reprojection_rmse:
            return False, f"Reprojection RMSE ({result.reprojection_rmse_px:.2f} px) > policy threshold ({policy.max_reprojection_rmse:.2f} px)"
    elif result.reprojection_status == "FAIL":
        return False, "Reprojection check failed"
    # If reprojection_status == "NOT_AVAILABLE" (e.g. feed-forward VGGT), it does not disqualify the candidate

    # 3. Trajectory check
    if policy.require_trajectory:
        if result.trajectory_status not in ["PASS", "AVAILABLE", "WARNING"]:
            return False, f"Trajectory validation status is '{result.trajectory_status}'"

    return True, None


def _export_normalized_cameras_from_colmap(model_dir: Path, output_file: Path) -> Path:
    """Helper creating standard cameras.json from a COLMAP model directory."""
    sparse_path = find_sparse_model_dir(model_dir) or model_dir
    model = read_colmap_model(sparse_path)

    poses: List[Dict[str, Any]] = []
    for img in sorted(model.images.values(), key=lambda x: x.name):
        center = compute_camera_center(img.qvec, img.tvec)
        poses.append({
            "frame_id": Path(img.name).stem,
            "image_name": img.name,
            "image_id": img.image_id,
            "camera_id": img.camera_id,
            "position": [round(float(c), 6) for c in center],
            "rotation_quaternion": [round(float(q), 6) for q in img.qvec],
            "coordinate_convention": "OpenCV (X-right, Y-down, Z-forward)",
            "pose_type": "camera_from_world (X_cam = R * X_world + t)",
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    return output_file


# ---------------------------------------------------------------------------
# Backend Executors
# ---------------------------------------------------------------------------

def execute_fastmap(
    image_dir: Path,
    output_dir: Path,
    policy: ValidationPolicy,
    device: str = "cuda:0",
) -> PoseResult:
    """Executes FastMap backend and standardizes the result."""
    start_wall = time.time()
    start_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    fastmap_out = output_dir / "fastmap"
    fastmap_out.mkdir(parents=True, exist_ok=True)

    try:
        raw_res = run_fastmap(
            images=image_dir,
            output=fastmap_out,
            device=device,
            headless=True,
        )
        end_wall = time.time()
        end_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        runtime = round(end_wall - start_wall, 4)

        rep = raw_res.get("reprojection", {})
        traj = raw_res.get("trajectory", {})

        sparse_dir = find_sparse_model_dir(fastmap_out / "sparse") or (fastmap_out / "sparse")
        cameras_json = fastmap_out / "cameras.json"
        if not cameras_json.is_file() and sparse_dir.is_dir():
            try:
                _export_normalized_cameras_from_colmap(sparse_dir, cameras_json)
            except Exception:
                pass

        traj_json = fastmap_out / "trajectory.json"

        return PoseResult(
            method="FastMap",
            status=raw_res.get("status", "FAIL"),
            output_path=fastmap_out,
            runtime_seconds=runtime,
            validation_status=raw_res.get("status", "FAIL"),
            registration_rate=raw_res.get("registration_rate", 0.0),
            registered_images=raw_res.get("registered_images", 0),
            input_images=raw_res.get("input_images", 0),
            num_points3D=raw_res.get("num_points3D", 0),
            reprojection_rmse_px=rep.get("rmse_px") if rep.get("status") == "AVAILABLE" else None,
            reprojection_status=rep.get("status", "NOT_AVAILABLE"),
            trajectory_status=traj.get("status", "NOT_AVAILABLE"),
            cameras_path=cameras_json if cameras_json.is_file() else None,
            trajectory_path=traj_json if traj_json.is_file() else None,
            sparse_dir=sparse_dir if sparse_dir.is_dir() else None,
            start_time=start_iso,
            end_time=end_iso,
            blocker_reason=raw_res.get("blocker_reason"),
            warnings=raw_res.get("warnings", []),
            errors=raw_res.get("errors", []),
            raw_report=raw_res,
        )

    except Exception as exc:
        end_wall = time.time()
        return PoseResult(
            method="FastMap",
            status="FAIL",
            output_path=fastmap_out,
            runtime_seconds=round(end_wall - start_wall, 4),
            validation_status="FAIL",
            start_time=start_iso,
            end_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            failure_reason=str(exc),
            errors=[str(exc)],
        )


def execute_global_sfm(
    image_dir: Path,
    output_dir: Path,
    policy: ValidationPolicy,
    camera_model: str = "SIMPLE_RADIAL",
    matcher: str = "sequential",
) -> PoseResult:
    """Executes COLMAP Global SfM backend and standardizes the result."""
    start_wall = time.time()
    start_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    gsfm_out = output_dir / "global_sfm"
    gsfm_out.mkdir(parents=True, exist_ok=True)

    try:
        raw_res = run_global_sfm(
            images=image_dir,
            output=gsfm_out,
            camera_model=camera_model,
            matcher=matcher,
        )
        end_wall = time.time()
        end_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        runtime = round(end_wall - start_wall, 4)

        # Inspect pose_report.json
        val_path = gsfm_out / "pose_report.json"
        val_report: Dict[str, Any] = {}
        if val_path.is_file():
            with open(val_path, "r", encoding="utf-8") as f:
                val_report = json.load(f)

        rep = val_report.get("reprojection", {})
        traj = val_report.get("trajectory", {})

        sparse_dir = find_sparse_model_dir(gsfm_out / "sparse") or (gsfm_out / "sparse")
        cameras_json = gsfm_out / "cameras.json"
        if not cameras_json.is_file() and sparse_dir.is_dir():
            try:
                _export_normalized_cameras_from_colmap(sparse_dir, cameras_json)
            except Exception:
                pass

        traj_json = gsfm_out / "trajectory.json"

        rmse_val = raw_res.get("reprojection_rmse_px")
        if rmse_val is None and rep.get("status") == "AVAILABLE":
            rmse_val = rep.get("rmse_px")

        return PoseResult(
            method="COLMAP_Global_SfM",
            status=raw_res.get("status", "FAIL"),
            output_path=gsfm_out,
            runtime_seconds=runtime,
            validation_status=val_report.get("validation_status", raw_res.get("status", "FAIL")),
            registration_rate=raw_res.get("registration_rate", 0.0),
            registered_images=raw_res.get("registered_images", 0),
            input_images=raw_res.get("input_images", 0),
            num_points3D=raw_res.get("num_points3D", 0),
            reprojection_rmse_px=rmse_val,
            reprojection_status=rep.get("status", "NOT_AVAILABLE"),
            trajectory_status=traj.get("status", "NOT_AVAILABLE"),
            cameras_path=cameras_json if cameras_json.is_file() else None,
            trajectory_path=traj_json if traj_json.is_file() else None,
            sparse_dir=sparse_dir if sparse_dir.is_dir() else None,
            start_time=start_iso,
            end_time=end_iso,
            warnings=raw_res.get("warnings", []),
            errors=raw_res.get("errors", []),
            raw_report=raw_res,
        )

    except Exception as exc:
        end_wall = time.time()
        return PoseResult(
            method="COLMAP_Global_SfM",
            status="FAIL",
            output_path=gsfm_out,
            runtime_seconds=round(end_wall - start_wall, 4),
            validation_status="FAIL",
            start_time=start_iso,
            end_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            failure_reason=str(exc),
            errors=[str(exc)],
        )


def execute_vggt(
    image_dir: Path,
    output_dir: Path,
    policy: ValidationPolicy,
    checkpoint: str = "facebook/VGGT-1B",
    max_images: Optional[int] = 50,
    device: Optional[str] = None,
    mock_mode: bool = False,
) -> PoseResult:
    """Executes VGGT backend and standardizes the result."""
    start_wall = time.time()
    start_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    vggt_out = output_dir / "vggt"
    vggt_out.mkdir(parents=True, exist_ok=True)

    try:
        raw_res = run_vggt(
            images=image_dir,
            output=vggt_out,
            checkpoint=checkpoint,
            max_images=max_images,
            device=device,
            mock_mode=mock_mode,
        )
        end_wall = time.time()
        end_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        runtime = round(end_wall - start_wall, 4)

        rep = raw_res.get("reprojection", {})
        traj = raw_res.get("trajectory", {})

        sparse_dir = vggt_out / "sparse"
        cameras_json = vggt_out / "cameras.json"
        traj_json = vggt_out / "trajectory.json"

        inp_count = raw_res.get("input_images", 0)
        reg_count = raw_res.get("camera_predictions", 0)
        reg_rate = round(reg_count / max(inp_count, 1), 4) if inp_count > 0 else 0.0

        return PoseResult(
            method="VGGT",
            status=raw_res.get("status", "FAIL"),
            output_path=vggt_out,
            runtime_seconds=runtime,
            validation_status=raw_res.get("status", "FAIL"),
            registration_rate=reg_rate,
            registered_images=reg_count,
            input_images=inp_count,
            num_points3D=raw_res.get("num_points3D", 0),
            reprojection_rmse_px=rep.get("rmse_px") if rep.get("status") == "AVAILABLE" else None,
            reprojection_status=rep.get("status", "NOT_AVAILABLE"),
            trajectory_status=traj.get("status", "NOT_AVAILABLE"),
            cameras_path=cameras_json if cameras_json.is_file() else None,
            trajectory_path=traj_json if traj_json.is_file() else None,
            sparse_dir=sparse_dir if sparse_dir.is_dir() else None,
            start_time=start_iso,
            end_time=end_iso,
            blocker_reason=raw_res.get("blocker_reason"),
            warnings=raw_res.get("warnings", []),
            errors=raw_res.get("errors", []),
            raw_report=raw_res,
        )

    except Exception as exc:
        end_wall = time.time()
        return PoseResult(
            method="VGGT",
            status="FAIL",
            output_path=vggt_out,
            runtime_seconds=round(end_wall - start_wall, 4),
            validation_status="FAIL",
            start_time=start_iso,
            end_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            failure_reason=str(exc),
            errors=[str(exc)],
        )


def execute_colmap_incremental(
    image_dir: Path,
    output_dir: Path,
    policy: ValidationPolicy,
    camera_model: str = "SIMPLE_RADIAL",
    matcher: str = "sequential",
) -> PoseResult:
    """Executes COLMAP Incremental reference backend and standardizes the result."""
    start_wall = time.time()
    start_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    colmap_out = output_dir / "colmap_incremental"
    colmap_out.mkdir(parents=True, exist_ok=True)

    try:
        report = run_colmap_reconstruction(
            image_dir=image_dir,
            output_dir=colmap_out,
            camera_model=camera_model,
            matcher=matcher,
        )
        end_wall = time.time()
        end_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        runtime = round(end_wall - start_wall, 4)

        # Run Step 2 validation
        sparse_dir = colmap_out / "sparse"
        val_path = colmap_out / "pose_report.json"
        traj_path = colmap_out / "trajectory.json"
        val_report = validate_reconstruction(
            model_dir=sparse_dir,
            image_dir=image_dir,
            output_report_path=val_path,
            trajectory_output_path=traj_path,
            min_registration_rate=policy.min_registration_rate,
            max_reprojection_error=policy.max_reprojection_rmse,
        )

        rep = val_report.get("reprojection", {})
        traj = val_report.get("trajectory", {})

        cameras_json = colmap_out / "cameras.json"
        if not cameras_json.is_file() and sparse_dir.is_dir():
            try:
                _export_normalized_cameras_from_colmap(sparse_dir, cameras_json)
            except Exception:
                pass

        return PoseResult(
            method="COLMAP_Incremental",
            status=val_report.get("validation_status", "FAIL"),
            output_path=colmap_out,
            runtime_seconds=runtime,
            validation_status=val_report.get("validation_status", "FAIL"),
            registration_rate=report.get("registration_rate", 0.0),
            registered_images=report.get("registered_images", 0),
            input_images=report.get("input_images", 0),
            num_points3D=report.get("num_points3D", 0),
            reprojection_rmse_px=rep.get("rmse_px") if rep.get("status") == "AVAILABLE" else None,
            reprojection_status=rep.get("status", "NOT_AVAILABLE"),
            trajectory_status=traj.get("status", "NOT_AVAILABLE"),
            cameras_path=cameras_json if cameras_json.is_file() else None,
            trajectory_path=traj_path if traj_path.is_file() else None,
            sparse_dir=sparse_dir if sparse_dir.is_dir() else None,
            start_time=start_iso,
            end_time=end_iso,
            warnings=val_report.get("warnings", []),
            errors=val_report.get("errors", []),
            raw_report=val_report,
        )

    except Exception as exc:
        end_wall = time.time()
        return PoseResult(
            method="COLMAP_Incremental",
            status="FAIL",
            output_path=colmap_out,
            runtime_seconds=round(end_wall - start_wall, 4),
            validation_status="FAIL",
            start_time=start_iso,
            end_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            failure_reason=str(exc),
            errors=[str(exc)],
        )


# ---------------------------------------------------------------------------
# Adaptive Engine Orchestrator
# ---------------------------------------------------------------------------

def run_adaptive_engine(
    images: str | Path,
    output: str | Path,
    method: str = "auto",
    min_images: int = 20,
    policy: Optional[ValidationPolicy] = None,
    device: str = "cuda:0",
    mock_backends: Optional[Dict[str, Callable[[], PoseResult]]] = None,
    gps_path: Optional[str | Path] = None,
    imu_path: Optional[str | Path] = None,
    max_time_diff_s: float = 1.0,
    target_crs: Optional[str] = None,
    altitude_reference: str = "WGS84_ellipsoidal",
) -> Dict[str, Any]:
    """Runs the adaptive pose engine with preflight validation, fallback orchestration, and output publication.

    Args:
        images: Input directory containing drone flight images.
        output: Directory to save adaptive output and reports.
        method: Selection mode ('auto', 'fastmap', 'global_sfm', 'vggt', 'colmap').
        min_images: Minimum image count preflight gate (default 20).
        policy: Photogrammetric validation criteria.
        device: Execution device for GPU backends.
        mock_backends: Optional mock callable map for unit test orchestration verification.
        gps_path: Optional path to GPS CSV file for georeferencing.
        imu_path: Optional path to IMU data file.
        max_time_diff_s: Maximum timestamp difference tolerance for GPS synchronization.
        target_crs: Target coordinate reference system (e.g. 'EPSG:32643').
        altitude_reference: Vertical datum reference name.
    """
    total_start = time.time()
    val_policy = policy or ValidationPolicy()
    img_path = Path(images).resolve()
    out_path = Path(output).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    report_path = out_path / "pose_engine_report.json"
    selected_dir = out_path / "selected"

    # 1. Pre-flight checks
    preflight_info = run_preflight_checks(img_path, min_images=min_images)

    # 2. Determine execution order
    valid_methods = ["auto", "fastmap", "global_sfm", "vggt", "colmap"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Choose from: {', '.join(valid_methods)}")

    if method == "auto":
        execution_plan = ["fastmap", "global_sfm", "vggt"]
    else:
        execution_plan = [method]

    # 3. Method execution loop
    attempts: List[Dict[str, Any]] = []
    selected_result: Optional[PoseResult] = None
    cumulative_runtime = 0.0

    for candidate_name in execution_plan:
        logger.info("Executing candidate backend: %s", candidate_name)

        # Check for mocks (used strictly for orchestration unit tests)
        if mock_backends and candidate_name in mock_backends:
            res = mock_backends[candidate_name]()
        elif candidate_name == "fastmap":
            res = execute_fastmap(img_path, out_path, val_policy, device=device)
        elif candidate_name == "global_sfm":
            res = execute_global_sfm(img_path, out_path, val_policy)
        elif candidate_name == "vggt":
            res = execute_vggt(img_path, out_path, val_policy, device=device)
        elif candidate_name == "colmap":
            res = execute_colmap_incremental(img_path, out_path, val_policy)
        else:
            continue

        cumulative_runtime += res.runtime_seconds

        # Evaluate candidate against photogrammetric policy
        is_pass, reject_reason = evaluate_candidate(res, val_policy)

        attempt_log = {
            "method": res.method,
            "status": res.status,
            "decision": "ACCEPT" if is_pass else "REJECT",
            "start_time": res.start_time,
            "end_time": res.end_time,
            "runtime_seconds": res.runtime_seconds,
            "validation": {
                "validation_status": res.validation_status,
                "registration_rate": res.registration_rate,
                "registered_images": res.registered_images,
                "reprojection_rmse_px": res.reprojection_rmse_px,
                "reprojection_status": res.reprojection_status,
                "trajectory_status": res.trajectory_status,
            },
            "rejection_reason": reject_reason,
            "blocker_reason": res.blocker_reason,
            "warnings": res.warnings,
            "errors": res.errors,
        }
        attempts.append(attempt_log)

        if is_pass:
            selected_result = res
            logger.info("Candidate %s passed validation. Short-circuiting remaining methods.", res.method)
            break
        else:
            logger.warning("Candidate %s rejected: %s. Falling back to next backend.", res.method, reject_reason)

    # 4. Standardize Selected Output & Georeferencing
    geo_meta: Optional[Dict[str, Any]] = None
    geo_error: Optional[str] = None
    pointcloud_meta: Optional[Dict[str, Any]] = None
    pointcloud_error: Optional[str] = None
    splat_meta: Optional[Dict[str, Any]] = None
    splat_error: Optional[str] = None

    if selected_result is not None:
        selected_dir.mkdir(parents=True, exist_ok=True)

        # Copy cameras.json
        if selected_result.cameras_path and selected_result.cameras_path.is_file():
            shutil.copy2(selected_result.cameras_path, selected_dir / "cameras.json")

        # Copy trajectory.json
        if selected_result.trajectory_path and selected_result.trajectory_path.is_file():
            shutil.copy2(selected_result.trajectory_path, selected_dir / "trajectory.json")

        # Copy sparse reconstruction files
        if selected_result.sparse_dir and selected_result.sparse_dir.is_dir():
            target_sparse = selected_dir / "sparse"
            target_sparse.mkdir(parents=True, exist_ok=True)
            for fname in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
                src_file = selected_result.sparse_dir / fname
                if src_file.is_file():
                    shutil.copy2(src_file, target_sparse / fname)

        # Copy validation report
        val_src = selected_result.output_path / "pose_report.json"
        if val_src.is_file():
            shutil.copy2(val_src, selected_dir / "pose_report.json")

        # Georeferencing if GPS is provided
        if gps_path is not None:
            from reconstruction.georeferencing.georeference import run_georeferencing
            geo_out = selected_dir / "georeferenced"
            try:
                geo_meta = run_georeferencing(
                    cameras_path=selected_dir / "cameras.json",
                    gps_path=gps_path,
                    output_dir=geo_out,
                    imu_path=imu_path,
                    max_time_diff_s=max_time_diff_s,
                    target_crs=target_crs,
                    altitude_reference=altitude_reference,
                    source_pose_method=selected_result.method,
                )
                logger.info("Georeferencing completed successfully in %s", geo_out)
            except Exception as exc:
                logger.error("Georeferencing failed: %s", exc)
                geo_error = str(exc)

        # Point-Cloud Processing (Step 8)
        sparse_points_txt = selected_dir / "sparse" / "points3D.txt"
        sparse_points_bin = selected_dir / "sparse" / "points3D.bin"

        if sparse_points_txt.is_file() or sparse_points_bin.is_file():
            from reconstruction.pointcloud.processing import run_point_cloud_pipeline
            pcd_in = sparse_points_txt if sparse_points_txt.is_file() else sparse_points_bin
            pcd_out = selected_dir / "pointcloud"
            geo_meta_path = selected_dir / "georeferenced" / "metadata.json"
            try:
                _, pcd_stats = run_point_cloud_pipeline(
                    input_cloud=pcd_in,
                    output_dir=pcd_out,
                    georef_metadata_path=geo_meta_path if geo_meta_path.is_file() else None,
                    source_method=selected_result.method,
                )
                pointcloud_meta = {
                    "output_dir": str(pcd_out),
                    "ply_file": str(pcd_out / "pointcloud.ply"),
                    "input_points": pcd_stats.get("input_points", 0),
                    "final_points": pcd_stats.get("final_points", 0),
                    "retention_rate": pcd_stats.get("retention_rate", 0.0),
                    "coordinate_system": pcd_stats.get("coordinate_system"),
                    "coordinate_units": pcd_stats.get("coordinate_units"),
                    "is_georeferenced": pcd_stats.get("coordinate_system") is not None,
                }
                logger.info("Point-cloud processing completed successfully in %s", pcd_out)
            except Exception as exc:
                logger.error("Point-cloud processing failed: %s", exc)
                pointcloud_error = str(exc)

        # Gaussian Splatting Dataset Preparation & Splatfacto (Step 9)
        cams_candidate = (
            selected_dir / "georeferenced" / "cameras.json"
            if (selected_dir / "georeferenced" / "cameras.json").is_file()
            else selected_dir / "cameras.json"
        )
        pcd_candidate = (
            selected_dir / "pointcloud" / "pointcloud.ply"
            if (selected_dir / "pointcloud" / "pointcloud.ply").is_file()
            else None
        )
        geo_meta_candidate = (
            selected_dir / "georeferenced" / "metadata.json"
            if (selected_dir / "georeferenced" / "metadata.json").is_file()
            else None
        )

        if cams_candidate.is_file() and img_path.is_dir():
            from reconstruction.splatting.dataset import prepare_splatfacto_dataset
            from reconstruction.splatting.splatfacto import run_splatfacto_training
            splat_out = selected_dir / "splat"
            dset_out = splat_out / "dataset"
            try:
                dset_info = prepare_splatfacto_dataset(
                    images_dir=img_path,
                    cameras_path=cams_candidate,
                    output_dir=dset_out,
                    pointcloud_path=pcd_candidate,
                    georef_metadata_path=geo_meta_candidate,
                    copy_images=False,
                )
                mock_splat = mock_backends.get("splatfacto") if mock_backends else None
                train_info = run_splatfacto_training(
                    dataset_dir=dset_out,
                    output_dir=splat_out / "training",
                    mock_runner=mock_splat,
                )
                splat_meta = {
                    "dataset_dir": str(dset_out),
                    "transforms_file": dset_info["transforms_path"],
                    "frames_count": dset_info["num_frames"],
                    "pointcloud_initialization": dset_info["pointcloud_initialization"],
                    "training_status": train_info.get("status", "NOT_AVAILABLE"),
                    "training_report": str(splat_out / "training" / "training_report.json"),
                }
                logger.info("Gaussian Splatting dataset prepared in %s (training: %s)", dset_out, train_info.get("status"))
            except Exception as exc:
                logger.error("Gaussian Splatting preparation failed: %s", exc)
                splat_error = str(exc)

    # 5. Generate Final Decision Report
    if selected_result is None:
        overall_status = "FAIL"
    elif gps_path is not None and geo_meta is None:
        # GPS was explicitly requested/required but georeferencing failed
        overall_status = "FAIL"
    else:
        overall_status = "PASS"

    policy_dict = val_policy.to_dict()
    decision_report = {
        "selected_method": selected_result.method if selected_result else None,
        "overall_status": overall_status,
        "mode": method,
        "total_runtime_seconds": round(cumulative_runtime, 4),
        "wall_clock_seconds": round(time.time() - total_start, 4),
        "preflight": preflight_info,
        "validation_policy": policy_dict,
        "policy": policy_dict,
        "georeferencing": geo_meta,
        "georeference_error": geo_error,
        "pointcloud": pointcloud_meta,
        "pointcloud_error": pointcloud_error,
        "splatting": splat_meta,
        "splatting_error": splat_error,
        "attempts": attempts,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(decision_report, f, indent=2)

    return decision_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Pose Engine CLI (Person 1 - Step 6 & 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", required=True, type=Path, help="Input images directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reconstruction/test_output/adaptive"),
        help="Directory to save adaptive reports and selected output",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "fastmap", "global_sfm", "vggt", "colmap"],
        default="auto",
        help="Execution mode (auto fallback sequence or single target backend)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=20,
        help="Minimum image count required by preflight data gate",
    )
    parser.add_argument(
        "--min-registration-rate",
        type=float,
        default=DEFAULT_MIN_REGISTRATION_RATE,
        help="Minimum acceptable camera registration rate (provisional engineering gate)",
    )
    parser.add_argument(
        "--max-reprojection-rmse",
        type=float,
        default=DEFAULT_MAX_REPROJECTION_ERROR,
        help="Maximum acceptable reprojection RMSE in pixels (provisional engineering gate)",
    )
    parser.add_argument(
        "--require-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require trajectory continuity validation (use --no-require-trajectory to disable)",
    )
    parser.add_argument(
        "--gps",
        type=Path,
        default=None,
        help="Optional path to GPS CSV file for georeferencing",
    )
    parser.add_argument(
        "--imu",
        type=Path,
        default=None,
        help="Optional path to IMU data file",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=1.0,
        help="Maximum time difference for GPS-pose synchronization (seconds)",
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default=None,
        help="Target coordinate reference system (e.g. EPSG:32643)",
    )
    parser.add_argument(
        "--altitude-reference",
        type=str,
        default="WGS84_ellipsoidal",
        help="Altitude reference datum",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Compute device for neural/accelerated backends",
    )

    args = parser.parse_args()

    policy = ValidationPolicy(
        min_registration_rate=args.min_registration_rate,
        max_reprojection_rmse_px=args.max_reprojection_rmse,
        require_trajectory=args.require_trajectory,
    )

    try:
        report = run_adaptive_engine(
            images=args.images,
            output=args.output,
            method=args.method,
            min_images=args.min_images,
            policy=policy,
            device=args.device,
            gps_path=args.gps,
            imu_path=args.imu,
            max_time_diff_s=args.max_time_diff,
            target_crs=args.target_crs,
            altitude_reference=args.altitude_reference,
        )

        print("\n=====================================================================")
        print("                 ADAPTIVE POSE ENGINE DECISION REPORT                ")
        print("=====================================================================")
        print(f"Overall Status:        {report.get('overall_status')}")
        print(f"Selected Backend:      {report.get('selected_method')}")
        print(f"Total Attempt Runtime: {report.get('total_runtime_seconds')} s")
        print(f"Attempts Count:        {len(report.get('attempts', []))}")
        for idx, att in enumerate(report.get("attempts", []), 1):
            print(f"  [{idx}] {att['method']:<18} -> {att['status']:<8} (Decision: {att['decision']})")
            if att.get("rejection_reason"):
                print(f"      Reason: {att['rejection_reason']}")
        if report.get("georeferencing"):
            print(f"Georeferencing:        PASS (Target CRS: {report['georeferencing'].get('target_crs')})")
        elif report.get("georeference_error"):
            print(f"Georeferencing Error:  {report['georeference_error']}")
        if report.get("pointcloud"):
            pcd_info = report["pointcloud"]
            print(f"Point Cloud:           PASS ({pcd_info.get('final_points')} pts, {pcd_info.get('coordinate_units')}, CRS: {pcd_info.get('coordinate_system') or 'None (local)'})")
        elif report.get("pointcloud_error"):
            print(f"Point Cloud Error:     {report['pointcloud_error']}")
        if report.get("splatting"):
            s_info = report["splatting"]
            print(f"Gaussian Splatting:    {s_info.get('training_status')} ({s_info.get('frames_count')} frames, init: {s_info.get('pointcloud_initialization', {}).get('used')})")
        elif report.get("splatting_error"):
            print(f"Gaussian Splat Error:  {report['splatting_error']}")
        print("=====================================================================\n")

        sys.exit(0 if report.get("overall_status") == "PASS" else 1)

    except Exception as exc:
        print(f"\n[ERROR] Adaptive engine halted: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

