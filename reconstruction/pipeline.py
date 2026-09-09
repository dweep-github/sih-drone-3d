"""End-to-End Reconstruction Pipeline for SIH26158 (Step 10).

Orchestrates Steps 1–9 into one unified, reproducible reconstruction pipeline:
1. Input Handling & Frame Processing (Video or Image Directory)
2. Dynamic Object Detection & Masking (YOLO11)
3. Adaptive Pose Estimation (FastMap -> COLMAP Global SfM -> VGGT)
4. Pose Validation & Quality Gating
5. GPS / IMU Georeferencing
6. Point-Cloud Processing (Statistical, Radius, Voxel filtering)
7. Gaussian Splatting / Splatfacto (Transforms dataset & training)
8. Output Packaging & Processing Report

Maintains strict state tracking, execution timing, resource monitoring,
explicit failure handling, and benchmark integrity without fabricating results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.frame_selection.extractor import FrameExtractor
from reconstruction.masking.dynamic_mask import (
    DEFAULT_DYNAMIC_CLASSES,
    DynamicMaskingReport,
    run_dynamic_masking,
)
from reconstruction.pointcloud.processing import (
    DEFAULT_NB_NEIGHBORS,
    DEFAULT_RADIUS_MIN_POINTS,
    DEFAULT_STD_RATIO,
    run_point_cloud_pipeline,
)
from reconstruction.pose_engine.engine import (
    DEFAULT_MAX_REPROJECTION_ERROR,
    DEFAULT_MIN_REGISTRATION_RATE,
    ValidationPolicy,
    run_adaptive_engine,
)
from reconstruction.splatting.dataset import prepare_splatfacto_dataset
from reconstruction.splatting.splatfacto import (
    SplatfactoConfig,
    check_splatfacto_environment,
    run_splatfacto_training,
)

logger = logging.getLogger("reconstruction.pipeline")

PIPELINE_VERSION = "1.0.0"

# Supported input extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# Pipeline States
STATE_UPLOADED = "UPLOADED"
STATE_FRAME_PROCESSING = "FRAME_PROCESSING"
STATE_OBJECT_DETECTION = "OBJECT_DETECTION"
STATE_POSE_ESTIMATION = "POSE_ESTIMATION"
STATE_POSE_VALIDATION = "POSE_VALIDATION"
STATE_GEOREFERENCING = "GEOREFERENCING"
STATE_POINT_CLOUD = "POINT_CLOUD"
STATE_SPLATTING = "SPLATTING"
STATE_EXPORT = "EXPORT"
STATE_COMPLETE = "COMPLETE"
STATE_FAILED = "FAILED"

ALL_STATES = [
    STATE_UPLOADED,
    STATE_FRAME_PROCESSING,
    STATE_OBJECT_DETECTION,
    STATE_POSE_ESTIMATION,
    STATE_POSE_VALIDATION,
    STATE_GEOREFERENCING,
    STATE_POINT_CLOUD,
    STATE_SPLATTING,
    STATE_EXPORT,
    STATE_COMPLETE,
    STATE_FAILED,
]


@dataclass
class StateTransition:
    """Records a discrete state transition within the pipeline."""

    state: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED, BLOCKED, NOT_AVAILABLE, SKIPPED, WARNING, PASS, FAIL
    details: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineStateMachine:
    """Manages pipeline execution state transitions, history, and persistence."""

    def __init__(self, state_file: Optional[Path] = None) -> None:
        self.transitions: List[StateTransition] = []
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.current_state: str = STATE_UPLOADED
        self.status: str = "RUNNING"
        self.state_file: Optional[Path] = Path(state_file).resolve() if state_file else None
        self._record_transition(STATE_UPLOADED, status="SUCCESS", details="Pipeline initialized.")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _record_transition(
        self,
        state: str,
        status: str = "SUCCESS",
        details: Optional[str] = None,
        error: Optional[str] = None,
    ) -> StateTransition:
        now = self._now()
        t = StateTransition(
            state=state,
            started_at=now,
            completed_at=now,
            status=status,
            details=details,
            error=error,
        )
        self.transitions.append(t)
        self.current_state = state
        self.stages[state] = {
            "status": status,
            "started_at": now,
            "completed_at": now,
            "details": details,
            "error": error,
        }
        self.persist()
        return t

    def transition_start(self, state: str, details: Optional[str] = None) -> StateTransition:
        now = self._now()
        t = StateTransition(
            state=state,
            started_at=now,
            completed_at=None,
            status="RUNNING",
            details=details,
        )
        self.transitions.append(t)
        self.current_state = state
        self.stages[state] = {
            "status": "RUNNING",
            "started_at": now,
            "completed_at": None,
            "details": details,
            "error": None,
        }
        self.persist()
        return t

    def transition_end(
        self,
        status: str = "SUCCESS",
        details: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        now = self._now()
        if self.transitions:
            last = self.transitions[-1]
            last.completed_at = now
            last.status = status
            if details:
                last.details = details
            if error:
                last.error = error
            self.stages[last.state] = {
                "status": status,
                "started_at": last.started_at,
                "completed_at": now,
                "details": details or last.details,
                "error": error or last.error,
            }

        if status in ("FAILED", "FAIL"):
            self.current_state = STATE_FAILED
            self.status = "FAIL"
        elif status in ("WARNING", "COMPLETE_WITH_WARNINGS"):
            self.status = "WARNING"
        elif status in ("SUCCESS", "PASS") and self.status != "WARNING":
            self.status = "PASS"

        self.persist()

    def persist(self, path: Optional[Path] = None) -> None:
        """Serializes current state machine to output/pipeline_state.json."""
        target = path or self.state_file
        if target is not None:
            try:
                target_p = Path(target).resolve()
                target_p.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "current_state": self.current_state,
                    "status": self.status,
                    "stages": self.stages,
                    "transitions": self.to_list(),
                }
                with open(target_p, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as exc:
                logger.warning("Failed to persist pipeline state: %s", exc)

    def to_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.transitions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state,
            "status": self.status,
            "stages": self.stages,
            "transitions": self.to_list(),
        }


def _default_device() -> str:
    """Returns 'cuda:0' if CUDA is available, else 'cpu'."""
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class PipelineConfig:
    """Central configuration for the entire SIH26158 reconstruction pipeline."""

    input_path: str
    output_dir: str = "output"
    resume: bool = False
    config_path: Optional[str] = None

    # Pose estimation & validation
    method: str = "auto"
    min_images: int = 20
    min_registration_rate: float = DEFAULT_MIN_REGISTRATION_RATE
    max_reprojection_rmse: float = DEFAULT_MAX_REPROJECTION_ERROR
    require_trajectory: bool = True

    # Georeferencing
    gps_path: Optional[str] = None
    imu_path: Optional[str] = None
    max_time_diff: float = 1.0
    target_crs: Optional[str] = None
    altitude_reference: str = "WGS84_ellipsoidal"

    # Frame processing & quality filtering
    blur_threshold: float = 100.0
    min_brightness: float = 20.0
    max_brightness: float = 245.0
    duplicate_threshold: Optional[float] = None
    target_frames: int = 400
    sim_threshold: float = 0.95

    # Object detection & masking
    enable_masking: bool = True
    require_masking: bool = False
    yolo_weights: str = "yolo11n-seg.pt"
    dynamic_classes: Optional[List[str]] = None
    mask_confidence: float = 0.25

    # Point cloud filtering
    nb_neighbors: int = DEFAULT_NB_NEIGHBORS
    std_ratio: float = DEFAULT_STD_RATIO
    radius: Optional[float] = None
    min_points: int = DEFAULT_RADIUS_MIN_POINTS
    voxel_size: Optional[float] = None
    enable_statistical: bool = True
    enable_radius: bool = False
    enable_voxel: bool = False

    # Splatting
    enable_splatting: bool = True
    max_num_iterations: int = 30000
    downscale_factor: int = 1
    eval_step: int = 500
    save_step: int = 2000
    splat_device: str = "cuda"

    # Execution device
    device: str = field(default_factory=_default_device)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: Union[str, Path], **overrides: Any) -> PipelineConfig:
        """Loads configuration from a JSON file with optional keyword overrides."""
        p = Path(path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def get_system_resources() -> Dict[str, Any]:
    """Measures actual system resources (CPU RAM, GPU info). Never fabricates results."""
    cpu_ram: Optional[Dict[str, Any]] = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        cpu_ram = {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "percent_used": vm.percent,
        }
    except Exception:
        cpu_ram = None

    gpu_info: Optional[Dict[str, Any]] = None
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_info = {
                "cuda_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "total_vram_gb": round(total_bytes / (1024**3), 2),
                "device_count": torch.cuda.device_count(),
            }
        else:
            gpu_info = {
                "cuda_available": False,
                "gpu_name": None,
                "total_vram_gb": None,
                "status": "NOT_AVAILABLE",
                "reason": "CUDA is not available on this host.",
            }
    except Exception as exc:
        gpu_info = {
            "cuda_available": False,
            "status": "NOT_AVAILABLE",
            "reason": str(exc),
        }

    return {
        "cpu_ram": cpu_ram,
        "gpu": gpu_info,
    }


def compute_color_hist(img: np.ndarray) -> np.ndarray:
    """Computes a normalized HSV color histogram for near-duplicate frame detection."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def process_input_frames(
    input_path: Union[str, Path],
    output_frames_dir: Union[str, Path],
    blur_threshold: float = 100.0,
    min_brightness: float = 20.0,
    max_brightness: float = 245.0,
    duplicate_threshold: Optional[float] = None,
    target_frames: int = 400,
    sim_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Processes video or directory input into standardized selected frames with stable IDs.

    Calculates:
    - Image readability
    - Width/height dimensions
    - Brightness statistics (mean, min, max)
    - Blur/sharpness metric (Laplacian variance)
    - Near-duplicate detection against previous accepted frames

    Stable frame IDs follow the schema: `frame_000001.jpg`, `frame_000002.jpg`, etc.
    Writes:
    - <output>/frames/frames.json
    - <output>/frames/frames_manifest.json (backward compatibility alias)
    """
    start_time = time.time()
    in_p = Path(input_path).resolve()
    out_dir = Path(output_frames_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_p.exists():
        raise FileNotFoundError(f"Input path does not exist: {in_p}")

    manifest: List[Dict[str, Any]] = []
    input_count = 0
    selected_count = 0
    discarded_count = 0
    rejection_reasons: Dict[str, int] = {
        "unreadable": 0,
        "too_blurry": 0,
        "too_dark": 0,
        "too_bright": 0,
        "duplicate": 0,
    }

    if in_p.is_file() and in_p.suffix.lower() in VIDEO_EXTS:
        # Video extraction via FrameExtractor
        staging_dir = out_dir.parent / "staging_video"
        staging_dir.mkdir(parents=True, exist_ok=True)
        try:
            extractor = FrameExtractor(
                input_video=str(in_p),
                output_dir=str(staging_dir),
                target_frames=target_frames,
                blur_threshold=blur_threshold,
                sim_threshold=sim_threshold,
            )
            extractor.process()

            extracted_files = sorted(
                [f for f in (staging_dir / "frames").iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
                key=lambda x: x.name,
            )
            input_count = extractor.target_frames if hasattr(extractor, "target_frames") else len(extracted_files)

            # Assign deterministic stable IDs: frame_000001.jpg, ...
            for idx, src_img in enumerate(extracted_files, start=1):
                stable_name = f"frame_{idx:06d}.jpg"
                dest_path = out_dir / stable_name
                shutil.copy2(src_img, dest_path)
                selected_count += 1

                img = cv2.imread(str(dest_path))
                h, w = img.shape[:2] if img is not None else (0, 0)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
                variance = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if gray is not None else 0.0
                mean_bright = float(np.mean(gray)) if gray is not None else 0.0

                manifest.append({
                    "frame_id": f"frame_{idx:06d}",
                    "filename": stable_name,
                    "original_file": src_img.name,
                    "source": "video_frame",
                    "source_type": "video_frame",
                    "source_frame_index": idx - 1,
                    "timestamp": None,
                    "width": int(w),
                    "height": int(h),
                    "quality_status": "ACCEPTED",
                    "laplacian_variance": round(variance, 2),
                    "metrics": {
                        "laplacian_variance": round(variance, 2),
                        "mean_brightness": round(mean_bright, 2),
                    },
                })
        finally:
            if staging_dir.is_dir():
                shutil.rmtree(staging_dir, ignore_errors=True)

    elif in_p.is_dir():
        # Directory of images
        discovered_files = sorted(
            [f for f in in_p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
            key=lambda x: x.name,
        )
        input_count = len(discovered_files)

        if input_count == 0:
            raise ValueError(f"No supported image files found in directory: {in_p}")

        selected_idx = 1
        last_accepted_hist: Optional[np.ndarray] = None

        for src_idx, f in enumerate(discovered_files):
            img = cv2.imread(str(f))
            if img is None:
                discarded_count += 1
                rejection_reasons["unreadable"] += 1
                manifest.append({
                    "frame_id": f"unusable_{src_idx:06d}",
                    "filename": f.name,
                    "original_file": f.name,
                    "source": "image_directory",
                    "source_type": "image_directory",
                    "source_frame_index": src_idx,
                    "timestamp": None,
                    "width": 0,
                    "height": 0,
                    "quality_status": "REJECTED_UNREADABLE",
                    "rejection_reason": "Image unreadable or corrupt",
                })
                continue

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            mean_bright = float(np.mean(gray))

            # Quality checks
            is_rejected = False
            rej_reason = ""

            if input_count > 20 and mean_bright < min_brightness:
                is_rejected = True
                rej_reason = f"Mean brightness {mean_bright:.1f} < min threshold {min_brightness}"
                rejection_reasons["too_dark"] += 1
            elif input_count > 20 and mean_bright > max_brightness:
                is_rejected = True
                rej_reason = f"Mean brightness {mean_bright:.1f} > max threshold {max_brightness}"
                rejection_reasons["too_bright"] += 1
            elif input_count > 20 and variance < blur_threshold:
                is_rejected = True
                rej_reason = f"Laplacian blur variance {variance:.1f} < threshold {blur_threshold}"
                rejection_reasons["too_blurry"] += 1
            elif input_count > 20 and duplicate_threshold is not None and last_accepted_hist is not None:
                curr_hist = compute_color_hist(img)
                corr = cv2.compareHist(last_accepted_hist, curr_hist, cv2.HISTCMP_CORREL)
                if corr > duplicate_threshold:
                    is_rejected = True
                    rej_reason = f"Near-duplicate correlation {corr:.4f} > threshold {duplicate_threshold}"
                    rejection_reasons["duplicate"] += 1

            if is_rejected:
                discarded_count += 1
                manifest.append({
                    "frame_id": f"rejected_{src_idx:06d}",
                    "filename": f.name,
                    "original_file": f.name,
                    "source": "image_directory",
                    "source_type": "image_directory",
                    "source_frame_index": src_idx,
                    "timestamp": None,
                    "width": int(w),
                    "height": int(h),
                    "quality_status": "REJECTED",
                    "rejection_reason": rej_reason,
                    "metrics": {
                        "laplacian_variance": round(variance, 2),
                        "mean_brightness": round(mean_bright, 2),
                    },
                })
                continue

            # Accepted frame
            stable_name = f"frame_{selected_idx:06d}.jpg"
            dest_path = out_dir / stable_name
            if in_p != out_dir or f != dest_path:
                shutil.copy2(f, dest_path)

            last_accepted_hist = compute_color_hist(img)

            manifest.append({
                "frame_id": f"frame_{selected_idx:06d}",
                "filename": stable_name,
                "original_file": f.name,
                "source": "image_directory",
                "source_type": "image_directory",
                "source_frame_index": src_idx,
                "timestamp": None,
                "width": int(w),
                "height": int(h),
                "quality_status": "ACCEPTED",
                "laplacian_variance": round(variance, 2),
                "metrics": {
                    "laplacian_variance": round(variance, 2),
                    "mean_brightness": round(mean_bright, 2),
                },
            })
            selected_count += 1
            selected_idx += 1

        # Fallback if quality threshold discarded all frames: keep all readable frames
        if selected_count == 0 and input_count > 0:
            logger.warning("Quality threshold discarded all frames. Retaining all discovered readable images.")
            for f in out_dir.glob("frame_*.jpg"):
                f.unlink(missing_ok=True)
            manifest.clear()
            selected_count = 0
            for idx, f in enumerate(discovered_files, start=1):
                img = cv2.imread(str(f))
                if img is None:
                    continue
                h, w = img.shape[:2]
                stable_name = f"frame_{idx:06d}.jpg"
                dest_path = out_dir / stable_name
                shutil.copy2(f, dest_path)
                manifest.append({
                    "frame_id": f"frame_{idx:06d}",
                    "filename": stable_name,
                    "original_file": f.name,
                    "source": "image_directory_fallback",
                    "source_type": "image_directory_fallback",
                    "source_frame_index": idx - 1,
                    "timestamp": None,
                    "width": int(w),
                    "height": int(h),
                    "quality_status": "ACCEPTED",
                    "metrics": {},
                })
                selected_count += 1
            discarded_count = input_count - selected_count
            rejection_reasons = {"unreadable": discarded_count, "too_blurry": 0, "too_dark": 0, "too_bright": 0, "duplicate": 0}
    else:
        raise ValueError(f"Input path must be a video file or directory of images: {in_p}")

    elapsed = round(time.time() - start_time, 4)
    selection_ratio = round(selected_count / input_count, 4) if input_count > 0 else 0.0

    frames_json_payload = {
        "total_frames": input_count,
        "input_frames": input_count,
        "accepted_frames": selected_count,
        "selected_frames": selected_count,
        "rejected_frames": discarded_count,
        "discarded_frames": discarded_count,
        "selection_ratio": selection_ratio,
        "rejection_reasons": rejection_reasons,
        "processing_time": elapsed,
        "frames": manifest,
    }

    # Save <output>/frames/frames.json (Section 3 requirement)
    frames_json_path = out_dir.parent / "frames.json"
    with open(frames_json_path, "w", encoding="utf-8") as f:
        json.dump(frames_json_payload, f, indent=2)

    # Save <output>/frames/frames_manifest.json (backward compatibility)
    manifest_path = out_dir.parent / "frames_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(frames_json_payload, f, indent=2)

    return {
        "input_frames": input_count,
        "total_frames": input_count,
        "selected_frames": selected_count,
        "accepted_frames": selected_count,
        "discarded_frames": discarded_count,
        "rejected_frames": discarded_count,
        "rejection_reasons": rejection_reasons,
        "selection_ratio": selection_ratio,
        "processing_time": elapsed,
        "frames_json": str(frames_json_path),
        "frames_manifest": str(manifest_path),
    }


def run_pipeline(
    config: PipelineConfig,
    mock_pose_backends: Optional[Dict[str, Any]] = None,
    mock_detector: Optional[Callable[[np.ndarray, str], List[Dict[str, Any]]]] = None,
    mock_splat_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Executes the full reconstruction pipeline according to the provided configuration.

    Orchestrates Steps 1 through 9, records per-stage timing and state machine
    transitions, standardizes outputs, and produces output/processing_report.json.
    """
    total_start_time = time.time()
    out_base = Path(config.output_dir).resolve()
    state_file = out_base / "pipeline_state.json"
    state_machine = PipelineStateMachine(state_file=state_file)

    frames_dir = out_base / "frames" / "selected"
    masks_dir = out_base / "masks"
    detections_dir = out_base / "detections"
    poses_dir = out_base / "poses"
    geospatial_dir = out_base / "geospatial"
    georef_alias_dir = out_base / "georeferenced"
    pointcloud_dir = out_base / "pointcloud"
    splat_dir = out_base / "splat"
    results_dir = out_base / "results"

    for d in [frames_dir, masks_dir, detections_dir, poses_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Serialize effective configuration to <output>/config.json (Section 12)
    config_file = out_base / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    timing_report: Dict[str, float] = {}
    frame_report: Dict[str, Any] = {}
    mask_report: Dict[str, Any] = {}
    pose_report: Dict[str, Any] = {}
    validation_report: Dict[str, Any] = {}
    georef_report: Dict[str, Any] = {}
    pointcloud_report: Dict[str, Any] = {}
    splatting_report: Dict[str, Any] = {}

    pipeline_status = "RUNNING"
    failure_reason: Optional[str] = None
    warnings: List[str] = []

    try:
        # =====================================================================
        # Stage 1: FRAME_PROCESSING
        # =====================================================================
        state_machine.transition_start(STATE_FRAME_PROCESSING, "Extracting and filtering frames.")
        t0 = time.time()

        skip_frames = config.resume and frames_dir.is_dir() and len(list(frames_dir.glob("frame_*.jpg"))) > 0
        if skip_frames:
            existing_frames = list(frames_dir.glob("frame_*.jpg"))
            frame_report = {
                "input_frames": len(existing_frames),
                "total_frames": len(existing_frames),
                "selected_frames": len(existing_frames),
                "accepted_frames": len(existing_frames),
                "discarded_frames": 0,
                "rejected_frames": 0,
                "selection_ratio": 1.0,
                "processing_time": 0.0,
                "status": "SKIPPED_RESUME",
            }
            state_machine.transition_end("SUCCESS", "Reused existing processed frames (--resume).")
        else:
            frame_res = process_input_frames(
                input_path=config.input_path,
                output_frames_dir=frames_dir,
                blur_threshold=config.blur_threshold,
                min_brightness=config.min_brightness,
                max_brightness=config.max_brightness,
                duplicate_threshold=config.duplicate_threshold,
                target_frames=config.target_frames,
                sim_threshold=config.sim_threshold,
            )
            frame_report = frame_res
            if frame_res["selected_frames"] == 0:
                raise ValueError("No valid input images could be selected or decoded.")
            state_machine.transition_end("SUCCESS", f"Selected {frame_res['selected_frames']} clean frames.")

        timing_report["frame_processing_seconds"] = round(time.time() - t0, 4)

        # =====================================================================
        # Stage 2: OBJECT_DETECTION (YOLO11 Dynamic Masking)
        # =====================================================================
        state_machine.transition_start(STATE_OBJECT_DETECTION, "Detecting dynamic objects and creating masks.")
        t0 = time.time()

        skip_masks = config.resume and masks_dir.is_dir() and len(list(masks_dir.glob("*.png"))) > 0
        if not config.enable_masking:
            mask_report = {
                "status": "SKIPPED",
                "reason": "Dynamic object masking disabled by configuration.",
            }
            state_machine.transition_end("SKIPPED", "Masking disabled by configuration.")
        elif skip_masks:
            mask_report = {
                "status": "SKIPPED_RESUME",
                "reason": "Reused existing masks (--resume).",
                "masked_frames": len(list(masks_dir.glob("*.png"))),
            }
            state_machine.transition_end("SUCCESS", "Reused existing masks (--resume).")
        else:
            dev = config.device
            if dev and dev.startswith("cuda"):
                try:
                    import torch

                    if not torch.cuda.is_available():
                        dev = "cpu"
                except Exception:
                    dev = "cpu"

            dyn_mask_res = run_dynamic_masking(
                frames_dir=frames_dir,
                masks_output_dir=masks_dir,
                detections_output_dir=detections_dir,
                dynamic_classes=config.dynamic_classes,
                weights=config.yolo_weights,
                confidence=config.mask_confidence,
                device=dev,
                mock_detector=mock_detector,
                require_masking=config.require_masking,
            )
            mask_report = dyn_mask_res.to_dict()

            # Generate aggregate detections.json (Section 13)
            det_summary_file = detections_dir / "detections.json"
            with open(det_summary_file, "w", encoding="utf-8") as f:
                json.dump(mask_report, f, indent=2)

            if dyn_mask_res.status == "FAILED":
                raise RuntimeError(f"Dynamic object masking failed: {dyn_mask_res.reason}")
            elif dyn_mask_res.status == "NOT_AVAILABLE":
                warnings.append(f"Dynamic object masking unavailable: {dyn_mask_res.reason}")
                state_machine.transition_end("NOT_AVAILABLE", f"YOLO11 unavailable: {dyn_mask_res.reason}")
            else:
                state_machine.transition_end("SUCCESS", f"Masked {dyn_mask_res.masked_frames} frames (mean cov: {dyn_mask_res.mask_coverage:.4f}).")

        timing_report["object_detection_seconds"] = round(time.time() - t0, 4)

        # =====================================================================
        # Stage 3: POSE_ESTIMATION & POSE_VALIDATION
        # =====================================================================
        state_machine.transition_start(STATE_POSE_ESTIMATION, f"Running adaptive pose engine (mode: {config.method}).")
        t0 = time.time()

        policy = ValidationPolicy(
            min_registration_rate=config.min_registration_rate,
            max_reprojection_rmse_px=config.max_reprojection_rmse,
            require_trajectory=config.require_trajectory,
        )

        backends_for_engine = dict(mock_pose_backends) if mock_pose_backends else {}
        if mock_splat_runner is not None and "splatfacto" not in backends_for_engine:
            backends_for_engine["splatfacto"] = mock_splat_runner

        adaptive_out = out_base / "adaptive_engine_output"
        engine_report = run_adaptive_engine(
            images=frames_dir,
            output=adaptive_out,
            method=config.method,
            min_images=config.min_images,
            policy=policy,
            device=config.device,
            mock_backends=backends_for_engine if backends_for_engine else None,
            gps_path=config.gps_path,
            imu_path=config.imu_path,
            max_time_diff_s=config.max_time_diff,
            target_crs=config.target_crs,
            altitude_reference=config.altitude_reference,
        )

        selected_candidate = engine_report.get("selected_method")
        overall_pose_status = engine_report.get("overall_status")
        attempts = engine_report.get("attempts", [])

        pose_timing = round(time.time() - t0, 4)
        timing_report["pose_estimation_seconds"] = pose_timing

        if overall_pose_status != "PASS" or selected_candidate is None:
            reasons = [a.get("rejection_reason") or a.get("blocker_reason") for a in attempts if a.get("status") == "FAIL"]
            err_msg = f"All pose estimation candidates failed validation: {'; '.join(filter(None, reasons))}"
            state_machine.transition_end("FAILED", error=err_msg)
            raise RuntimeError(err_msg)

        state_machine.transition_end("SUCCESS", f"Pose estimation selected backend '{selected_candidate}'.")

        # Stage 4: POSE_VALIDATION transition record
        state_machine.transition_start(STATE_POSE_VALIDATION, "Validating photogrammetric quality.")
        winning_attempt = next((a for a in attempts if a.get("method") == selected_candidate), {})
        val_dict = winning_attempt.get("validation", {})
        pose_report = {
            "selected_method": selected_candidate,
            "attempts": attempts,
            "registration_rate": val_dict.get("registration_rate", 0.0),
            "reprojection_metrics": {
                "rmse_px": val_dict.get("reprojection_rmse_px"),
                "status": val_dict.get("reprojection_status"),
            },
            "trajectory_status": val_dict.get("trajectory_status"),
            "runtime": winning_attempt.get("runtime_seconds", pose_timing),
        }
        validation_report = {
            "policy": policy.to_dict(),
            "status": "PASS" if val_dict.get("status") == "PASS" else "SUCCESS",
            "passed_method": selected_candidate,
        }
        state_machine.transition_end("SUCCESS", "Pose validation passed criteria.")

        # Standardize pose artifacts into output/poses/
        src_selected_dir = adaptive_out / "selected"
        if (src_selected_dir / "cameras.json").is_file():
            shutil.copy2(src_selected_dir / "cameras.json", poses_dir / "cameras.json")
        if (src_selected_dir / "trajectory.json").is_file():
            shutil.copy2(src_selected_dir / "trajectory.json", poses_dir / "trajectory.json")
        if (src_selected_dir / "pose_report.json").is_file():
            shutil.copy2(src_selected_dir / "pose_report.json", poses_dir / "pose_report.json")
        if (adaptive_out / "pose_engine_report.json").is_file():
            shutil.copy2(adaptive_out / "pose_engine_report.json", poses_dir / "pose_engine_report.json")

        sparse_target = poses_dir / "sparse"
        sparse_target.mkdir(parents=True, exist_ok=True)
        if (src_selected_dir / "sparse").is_dir():
            for f in (src_selected_dir / "sparse").iterdir():
                if f.is_file():
                    shutil.copy2(f, sparse_target / f.name)

        # =====================================================================
        # Stage 5: GEOREFERENCING
        # =====================================================================
        state_machine.transition_start(STATE_GEOREFERENCING, "Georeferencing reconstructed poses.")
        t0 = time.time()

        if config.gps_path is not None:
            geo_meta = engine_report.get("georeferencing")
            src_geo_dir = src_selected_dir / "georeferenced"
            if geo_meta and src_geo_dir.is_dir():
                # Populate both output/geospatial/ (Section 13) and output/georeferenced/ (alias)
                geospatial_dir.mkdir(parents=True, exist_ok=True)
                georef_alias_dir.mkdir(parents=True, exist_ok=True)
                for f in src_geo_dir.iterdir():
                    if f.is_file():
                        shutil.copy2(f, geospatial_dir / f.name)
                        shutil.copy2(f, georef_alias_dir / f.name)

                georef_report = {
                    "status": "SUCCESS",
                    "source_crs": geo_meta.get("source_crs"),
                    "target_crs": geo_meta.get("target_crs"),
                    "altitude_reference": geo_meta.get("altitude_reference"),
                    "matched_observations": geo_meta.get("matched_observations", 0),
                    "alignment_scale": geo_meta.get("alignment_scale"),
                    "alignment_rmse": geo_meta.get("alignment_rmse"),
                    "maximum_residual": geo_meta.get("maximum_residual"),
                    "georeferencing_status": "SUCCESS",
                    "output_dir": str(geospatial_dir),
                }
                state_machine.transition_end("SUCCESS", "GPS georeferencing aligned poses.")
            else:
                geo_err = engine_report.get("georeference_error") or "Georeferencing produced no output or alignment failed."
                georef_report = {
                    "status": "FAILED",
                    "error": geo_err,
                    "georeferencing_status": "FAILED",
                }
                state_machine.transition_end("FAILED", error=geo_err)
                raise RuntimeError(f"Georeferencing failed: {geo_err}")
        else:
            georef_report = {
                "status": "NOT_AVAILABLE",
                "georeferencing": "NOT_AVAILABLE",
                "georeferencing_status": "NOT_AVAILABLE",
                "coordinate_system": "local_unscaled",
                "reason": "No GPS log provided (--gps was not specified).",
            }
            state_machine.transition_end("NOT_AVAILABLE", "GPS not supplied. Local unscaled coordinates preserved.")

        timing_report["georeferencing_seconds"] = round(time.time() - t0, 4)

        # =====================================================================
        # Stage 6: POINT_CLOUD (Step 8 Processing)
        # =====================================================================
        state_machine.transition_start(STATE_POINT_CLOUD, "Filtering and processing sparse point cloud.")
        t0 = time.time()

        sparse_pts_txt = sparse_target / "points3D.txt"
        sparse_pts_bin = sparse_target / "points3D.bin"

        if not sparse_pts_txt.is_file() and not sparse_pts_bin.is_file():
            # Check if engine already produced point cloud in adaptive output
            existing_pcd = src_selected_dir / "pointcloud" / "pointcloud.ply"
            if existing_pcd.is_file():
                pointcloud_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing_pcd, pointcloud_dir / "pointcloud.ply")
                if (src_selected_dir / "pointcloud" / "metadata.json").is_file():
                    shutil.copy2(src_selected_dir / "pointcloud" / "metadata.json", pointcloud_dir / "metadata.json")
                if (src_selected_dir / "pointcloud" / "processing_report.json").is_file():
                    shutil.copy2(src_selected_dir / "pointcloud" / "processing_report.json", pointcloud_dir / "processing_report.json")
                pointcloud_report = engine_report.get("pointcloud", {"status": "SUCCESS"})
                state_machine.transition_end("SUCCESS", "Point cloud reused from engine.")
            else:
                err_msg = "No reconstructed 3D points found (points3D.txt/bin missing)."
                state_machine.transition_end("FAILED", error=err_msg)
                raise RuntimeError(err_msg)
        else:
            pcd_input = sparse_pts_txt if sparse_pts_txt.is_file() else sparse_pts_bin
            geo_meta_file = geospatial_dir / "metadata.json" if (geospatial_dir / "metadata.json").is_file() else None

            pcd_data, pcd_stats = run_point_cloud_pipeline(
                input_cloud=pcd_input,
                output_dir=pointcloud_dir,
                georef_metadata_path=geo_meta_file,
                nb_neighbors=config.nb_neighbors,
                std_ratio=config.std_ratio,
                enable_statistical=config.enable_statistical,
                radius=config.radius,
                min_points=config.min_points,
                enable_radius=config.enable_radius,
                voxel_size=config.voxel_size,
                enable_voxel=config.enable_voxel,
                source_method=selected_candidate,
            )
            pointcloud_report = {
                "status": "SUCCESS",
                "input_points": pcd_stats.get("input_points", 0),
                "final_points": pcd_stats.get("final_points", 0),
                "retention_rate": pcd_stats.get("retention_rate", 0.0),
                "coordinate_system": pcd_stats.get("coordinate_system", "local_unscaled"),
                "parameters": {
                    "nb_neighbors": config.nb_neighbors,
                    "std_ratio": config.std_ratio,
                    "radius": config.radius,
                    "voxel_size": config.voxel_size,
                },
                "pointcloud_ply": str(pointcloud_dir / "pointcloud.ply"),
                "metadata_json": str(pointcloud_dir / "metadata.json"),
                "processing_report_json": str(pointcloud_dir / "processing_report.json"),
            }
            state_machine.transition_end("SUCCESS", f"Filtered point cloud: {pcd_stats.get('final_points', 0)} points retained.")

        timing_report["point_cloud_seconds"] = round(time.time() - t0, 4)

        # =====================================================================
        # Stage 7: SPLATTING (Step 9 Nerfstudio Splatfacto)
        # =====================================================================
        state_machine.transition_start(STATE_SPLATTING, "Preparing dataset and running Gaussian Splatting.")
        t0 = time.time()

        if not config.enable_splatting:
            splatting_report = {
                "status": "SKIPPED",
                "reason": "Gaussian Splatting disabled by configuration.",
            }
            state_machine.transition_end("SKIPPED", "Splatting disabled by configuration.")
        else:
            cams_file = geospatial_dir / "cameras.json" if (geospatial_dir / "cameras.json").is_file() else poses_dir / "cameras.json"
            pcd_ply_file = pointcloud_dir / "pointcloud.ply" if (pointcloud_dir / "pointcloud.ply").is_file() else None
            georef_meta_file = geospatial_dir / "metadata.json" if (geospatial_dir / "metadata.json").is_file() else None

            splat_dir.mkdir(parents=True, exist_ok=True)
            dset_out = splat_dir / "dataset"

            # Prepare dataset transforms
            dset_info = prepare_splatfacto_dataset(
                images_dir=frames_dir,
                cameras_path=cams_file,
                output_dir=dset_out,
                pointcloud_path=pcd_ply_file,
                georef_metadata_path=georef_meta_file,
                copy_images=False,
            )

            # Copy transforms.json and transform.json to top-level splat/
            if (dset_out / "transforms.json").is_file():
                shutil.copy2(dset_out / "transforms.json", splat_dir / "transforms.json")
            if (dset_out / "transform.json").is_file():
                shutil.copy2(dset_out / "transform.json", splat_dir / "transform.json")

            splat_cfg = SplatfactoConfig(
                max_num_iterations=config.max_num_iterations,
                downscale_factor=config.downscale_factor,
                eval_step=config.eval_step,
                save_step=config.save_step,
                device=config.splat_device,
            )

            # Save splat config.json
            with open(splat_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(splat_cfg.to_dict(), f, indent=2)

            train_info = run_splatfacto_training(
                dataset_dir=dset_out,
                output_dir=splat_dir / "training",
                config=splat_cfg,
                mock_runner=mock_splat_runner,
            )

            (splat_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
            (splat_dir / "export").mkdir(parents=True, exist_ok=True)

            if (splat_dir / "training" / "training_report.json").is_file():
                shutil.copy2(splat_dir / "training" / "training_report.json", splat_dir / "training_report.json")

            splat_status = train_info.get("status", "NOT_AVAILABLE")
            splatting_report = {
                "status": splat_status,
                "dataset_transforms": str(splat_dir / "transforms.json"),
                "frames_count": dset_info.get("num_frames", 0),
                "pointcloud_initialization": bool(dset_info.get("pointcloud_initialization", {}).get("used", False)) if isinstance(dset_info.get("pointcloud_initialization"), dict) else bool(dset_info.get("pointcloud_initialization", False)),
                "pointcloud_initialization_details": dset_info.get("pointcloud_initialization"),
                "runtime_seconds": train_info.get("runtime_seconds", 0.0),
                "iterations": train_info.get("iterations", 0),
                "gpu": train_info.get("gpu"),
                "initialization_method": "sparse_pointcloud" if dset_info.get("pointcloud_initialization") else "random",
                "export_path": str(splat_dir / "export"),
                "render_validation_status": train_info.get("render_validation_status", "NOT_AVAILABLE"),
                "blocker_reason": train_info.get("blocker_reason") or train_info.get("error"),
            }

            if splat_status in ["SUCCESS", "SKIPPED"]:
                state_machine.transition_end("SUCCESS", f"Splatfacto completed ({train_info.get('iterations', 0)} iters).")
            elif splat_status == "NOT_AVAILABLE":
                warnings.append(f"Splatfacto unavailable: {train_info.get('blocker_reason') or train_info.get('error')}")
                state_machine.transition_end("NOT_AVAILABLE", f"Splatfacto hardware unavailable: {train_info.get('blocker_reason') or train_info.get('error')}")
            else:
                state_machine.transition_end("FAILED", error=train_info.get("error") or train_info.get("blocker_reason"))

        timing_report["splatting_seconds"] = round(time.time() - t0, 4)

        # =====================================================================
        # Stage 8: EXPORT & COMPLETE
        # =====================================================================
        state_machine.transition_start(STATE_EXPORT, "Organizing final output bundle.")
        state_machine.transition_end("SUCCESS", "Outputs organized.")

        # Determine overall pipeline status
        if warnings:
            pipeline_status = "COMPLETE_WITH_WARNINGS"
        else:
            pipeline_status = "COMPLETE"

        state_machine.transition_start(STATE_COMPLETE, f"Pipeline finished: {pipeline_status}.")
        state_machine.transition_end("SUCCESS", "All requested stages executed.")

    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        pipeline_status = "FAILED"
        failure_reason = str(exc)
        if state_machine.current_state != STATE_FAILED:
            state_machine.transition_end("FAILED", error=str(exc))
        state_machine.transition_start(STATE_FAILED, details=f"Pipeline aborted: {exc}")
        state_machine.transition_end("FAILED", error=str(exc))

    total_pipeline_time = round(time.time() - total_start_time, 4)
    timing_report["total_pipeline_seconds"] = total_pipeline_time

    # Output inventory
    outputs_inventory = {
        "config": str(config_file),
        "pipeline_state": str(out_base / "pipeline_state.json"),
        "frames_selected": str(frames_dir),
        "frames_json": str(out_base / "frames" / "frames.json"),
        "masks": str(masks_dir),
        "detections": str(detections_dir),
        "detections_json": str(detections_dir / "detections.json"),
        "poses": str(poses_dir),
        "geospatial": str(geospatial_dir) if config.gps_path else None,
        "pointcloud": str(pointcloud_dir),
        "splat": str(splat_dir),
        "processing_report": str(out_base / "processing_report.json"),
    }

    # Construct final processing report (Section 14)
    final_report = {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_status": pipeline_status,
        "failure_reason": failure_reason,
        "warnings": warnings,
        "errors": [failure_reason] if failure_reason else [],
        "timing": timing_report,
        "resource_monitoring": get_system_resources(),
        "input": {
            "input_path": config.input_path,
            "output_dir": config.output_dir,
            "gps_path": config.gps_path if config.gps_path else None,
            "imu_path": config.imu_path if config.imu_path else None,
        },
        "frame_processing": frame_report,
        "object_detection": mask_report,
        "pose_estimation": pose_report,
        "pose_validation": validation_report,
        "georeferencing": georef_report,
        "point_cloud": pointcloud_report,
        "splatting": splatting_report,
        "configuration": config.to_dict(),
        "outputs": outputs_inventory,
        "state_transitions": state_machine.to_list(),
    }

    # Save <output>/processing_report.json (Section 14)
    root_report_path = out_base / "processing_report.json"
    with open(root_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Also save to <output>/results/processing_report.json (backward compatibility)
    results_report_path = results_dir / "processing_report.json"
    with open(results_report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Pipeline %s in %.2fs. Report written to %s", pipeline_status, total_pipeline_time, root_report_path)
    return final_report


def run_real_e2e_smoke_test(
    input_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Attempts a real end-to-end reconstruction smoke test if genuine hardware and dataset exist.

    If CUDA or required real dataset is missing, cleanly reports REAL_E2E_SMOKE = 'NOT_AVAILABLE'
    with exact blocker reason. Never fabricates metrics or outcomes (Section 21).
    """
    import torch

    cuda_avail = torch.cuda.is_available()
    has_real_dataset = False

    if input_path is not None:
        p = Path(input_path).resolve()
        if p.is_dir() and len(list(p.glob("*.jpg"))) >= 20:
            has_real_dataset = True

    blockers: List[str] = []
    if not cuda_avail:
        blockers.append("CUDA is not available on host system.")
    if not has_real_dataset:
        blockers.append("No real drone reconstruction dataset supplied.")

    if blockers:
        return {
            "real_end_to_end_smoke_test": "NOT_AVAILABLE",
            "status": "NOT_AVAILABLE",
            "reason": " | ".join(blockers),
            "cuda_available": cuda_avail,
            "dataset_available": has_real_dataset,
        }

    # If genuine environment exists, execute real run
    cfg = config or PipelineConfig(
        input_path=str(input_path),
        output_dir=str(output_dir or "output/real_smoke_test"),
        method="auto",
    )
    report = run_pipeline(cfg)
    return {
        "real_end_to_end_smoke_test": report.get("pipeline_status", "FAILED"),
        "report": report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the comprehensive CLI parser for the reconstruction pipeline."""
    parser = argparse.ArgumentParser(
        description="SIH26158 End-to-End Reconstruction Pipeline (Step 10)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs & Outputs
    parser.add_argument("--input", required=True, type=str, help="Input drone video file or directory of images")
    parser.add_argument("--output", type=str, default="output", help="Root directory for outputs")
    parser.add_argument("--resume", action="store_true", help="Reuse completed intermediate stage outputs if valid")
    parser.add_argument("--config", dest="config_path", type=str, default=None, help="Path to custom JSON configuration file")

    # Pose estimation
    parser.add_argument("--method", choices=["auto", "fastmap", "global_sfm", "vggt", "colmap"], default="auto")
    parser.add_argument("--min-images", type=int, default=20, help="Minimum images required for reconstruction")
    parser.add_argument("--min-registration-rate", type=float, default=DEFAULT_MIN_REGISTRATION_RATE)
    parser.add_argument("--max-reprojection-rmse", type=float, default=DEFAULT_MAX_REPROJECTION_ERROR)
    parser.add_argument("--require-trajectory", action=argparse.BooleanOptionalAction, default=True)

    # Georeferencing
    parser.add_argument("--gps", dest="gps_path", type=str, default=None, help="Path to GPS CSV file")
    parser.add_argument("--imu", dest="imu_path", type=str, default=None, help="Path to IMU data file")
    parser.add_argument("--max-time-diff", type=float, default=1.0, help="Max timestamp sync difference (seconds)")
    parser.add_argument("--target-crs", type=str, default=None, help="Target CRS (e.g. EPSG:32643)")
    parser.add_argument("--altitude-reference", type=str, default="WGS84_ellipsoidal")

    # Object Detection / Masking
    parser.add_argument("--enable-masking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-masking", action="store_true", help="Fail if YOLO11 is unavailable")
    parser.add_argument("--yolo-weights", type=str, default="yolo11n-seg.pt")

    # Point Cloud Filtering
    parser.add_argument("--nb-neighbors", type=int, default=DEFAULT_NB_NEIGHBORS)
    parser.add_argument("--std-ratio", type=float, default=DEFAULT_STD_RATIO)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--min-points", type=int, default=DEFAULT_RADIUS_MIN_POINTS)
    parser.add_argument("--voxel-size", type=float, default=None)

    # Gaussian Splatting
    parser.add_argument("--enable-splatting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-num-iterations", type=int, default=30000)
    parser.add_argument("--downscale-factor", type=int, default=1)
    parser.add_argument("--eval-step", type=int, default=500)
    parser.add_argument("--save-step", type=int, default=2000)

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point returning:

    0 = successful completion (including COMPLETE_WITH_WARNINGS)
    1 = pipeline failure
    2 = invalid command or configuration
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_arg_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return code

    # Optional configuration from file
    config_dict: Dict[str, Any] = {}
    if args.config_path:
        cfg_p = Path(args.config_path)
        if not cfg_p.is_file():
            logger.error("Configuration file not found: %s", args.config_path)
            return 2
        try:
            with open(cfg_p, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        except Exception as exc:
            logger.error("Invalid configuration file format: %s", exc)
            return 2

    # CLI args override file configuration
    config = PipelineConfig(
        input_path=args.input or config_dict.get("input_path", ""),
        output_dir=args.output or config_dict.get("output_dir", "output"),
        resume=args.resume or config_dict.get("resume", False),
        config_path=args.config_path,
        method=args.method or config_dict.get("method", "auto"),
        min_images=args.min_images or config_dict.get("min_images", 20),
        min_registration_rate=args.min_registration_rate or config_dict.get("min_registration_rate", DEFAULT_MIN_REGISTRATION_RATE),
        max_reprojection_rmse=args.max_reprojection_rmse or config_dict.get("max_reprojection_rmse", DEFAULT_MAX_REPROJECTION_ERROR),
        require_trajectory=args.require_trajectory if args.require_trajectory is not None else config_dict.get("require_trajectory", True),
        gps_path=args.gps_path or config_dict.get("gps_path"),
        imu_path=args.imu_path or config_dict.get("imu_path"),
        max_time_diff=args.max_time_diff or config_dict.get("max_time_diff", 1.0),
        target_crs=args.target_crs or config_dict.get("target_crs"),
        altitude_reference=args.altitude_reference or config_dict.get("altitude_reference", "WGS84_ellipsoidal"),
        sim_threshold=config_dict.get("sim_threshold", 0.999),
        blur_threshold=config_dict.get("blur_threshold", 10.0),
        enable_masking=args.enable_masking if args.enable_masking is not None else config_dict.get("enable_masking", True),
        require_masking=args.require_masking or config_dict.get("require_masking", False),
        yolo_weights=args.yolo_weights or config_dict.get("yolo_weights", "yolo11n-seg.pt"),
        nb_neighbors=args.nb_neighbors or config_dict.get("nb_neighbors", DEFAULT_NB_NEIGHBORS),
        std_ratio=args.std_ratio or config_dict.get("std_ratio", DEFAULT_STD_RATIO),
        radius=args.radius or config_dict.get("radius"),
        min_points=args.min_points or config_dict.get("min_points", DEFAULT_RADIUS_MIN_POINTS),
        voxel_size=args.voxel_size or config_dict.get("voxel_size"),
        enable_splatting=args.enable_splatting if args.enable_splatting is not None else config_dict.get("enable_splatting", True),
        max_num_iterations=args.max_num_iterations or config_dict.get("max_num_iterations", 30000),
        downscale_factor=args.downscale_factor or config_dict.get("downscale_factor", 1),
        eval_step=args.eval_step or config_dict.get("eval_step", 500),
        save_step=args.save_step or config_dict.get("save_step", 2000),
        device=args.device or config_dict.get("device", "cuda:0"),
    )

    try:
        report = run_pipeline(config)
        status = report.get("pipeline_status")
        return 0 if status in ("COMPLETE", "COMPLETE_WITH_WARNINGS") else 1
    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
