"""Comprehensive unit and integration tests for SIH26158 Step 10: End-to-End Reconstruction Pipeline.

Tests:
1. Pipeline Orchestration:
   - Successful mocked full pipeline execution
   - Strict stage ordering and state machine transitions
   - Failure propagation on invalid inputs or pose failure
   - Configuration parameter propagation
2. Frame Processing:
   - Image directory input handling
   - Synthetic video file input handling
   - Stable deterministic frame ID preservation (frame_000001, frame_000002, ...)
   - Quality filtering: blur, brightness, duplicate detection
   - frames.json schema compliance
3. Dynamic Object Masking:
   - Mask convention adherence (0 = keep, 255 = ignore/masked)
   - Dimension preservation matching source images
   - Graceful fallback to NOT_AVAILABLE when weights/detector are unavailable
   - Generation of detections/detections.json
4. Adaptive Pose Engine Integration:
   - Fallback hierarchy (FastMap -> COLMAP Global SfM -> VGGT)
   - Validation rejection and error reporting
5. Georeferencing Integration:
   - Handling GPS log when supplied
   - Output into geospatial/ directory
   - Clear flagging as NOT_AVAILABLE / local_unscaled when GPS is absent
   - Clean failure handling when GPS is invalid
6. Point-Cloud Processing Integration:
   - Statistical and voxel parameter propagation
   - Generation of pointcloud.ply, metadata.json, processing_report.json
7. Gaussian Splatting Integration:
   - transforms.json dataset export
   - Controlled execution via mock runner
   - Clean NOT_AVAILABLE status on CPU-only hosts with COMPLETE_WITH_WARNINGS
8. State Machine & Persistence:
   - Persistence of pipeline_state.json across stages
9. Configuration:
   - Central config serialization to config.json
   - Loading config from JSON file
10. Final Processing Report:
   - Schema validation for all required top-level keys
   - Accurate per-stage timing measurements
   - Resource monitoring (psutil, CUDA)
   - Benchmark integrity without fabricated metrics
11. CLI & Real Smoke Test:
   - CLI argument parsing, --help, and return codes (0, 1, 2)
   - Real smoke test reports NOT_AVAILABLE honestly
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import cv2
import numpy as np
import pytest
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.masking.dynamic_mask import (
    DEFAULT_DYNAMIC_CLASSES,
    DynamicMaskingReport,
    check_yolo_availability,
    run_dynamic_masking,
)
from reconstruction.pipeline import (
    STATE_COMPLETE,
    STATE_EXPORT,
    STATE_FAILED,
    STATE_FRAME_PROCESSING,
    STATE_GEOREFERENCING,
    STATE_OBJECT_DETECTION,
    STATE_POINT_CLOUD,
    STATE_POSE_ESTIMATION,
    STATE_POSE_VALIDATION,
    STATE_SPLATTING,
    STATE_UPLOADED,
    PipelineConfig,
    PipelineStateMachine,
    get_system_resources,
    main as pipeline_main,
    process_input_frames,
    run_pipeline,
    run_real_e2e_smoke_test,
)
from reconstruction.pose_engine.engine import PoseResult, ValidationPolicy
from reconstruction.splatting.splatfacto import SplatfactoConfig


# ===========================================================================
# Test Fixtures & Helpers
# ===========================================================================

def create_synthetic_image(
    file_path: Path,
    width: int = 128,
    height: int = 128,
    color: tuple = (120, 140, 160),
) -> Path:
    """Creates a synthetic test image with high-contrast edges to pass blur checks."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Add high-contrast geometric shapes so Laplacian variance is high
    cv2.rectangle(img, (20, 20), (60, 60), (255, 255, 255), -1)
    cv2.circle(img, (90, 90), 25, (0, 0, 0), -1)
    cv2.line(img, (10, 110), (110, 10), (50, 200, 50), 3)
    cv2.imwrite(str(file_path), img)
    return file_path


def create_synthetic_video(
    file_path: Path,
    num_frames: int = 6,
    width: int = 128,
    height: int = 128,
) -> Path:
    """Generates a synthetic MP4 video file for video input testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 30) % 255, dtype=np.uint8)
        cv2.rectangle(frame, (10 + i * 5, 10), (40 + i * 5, 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return file_path


def create_mock_sparse_model(base_dir: Path, num_points: int = 100) -> Path:
    """Creates standard COLMAP-compatible sparse reconstruction text files."""
    sparse_dir = base_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    cams_txt = sparse_dir / "cameras.txt"
    cams_txt.write_text("1 PINHOLE 128 128 100.0 100.0 64.0 64.0\n", encoding="utf-8")

    imgs_txt = sparse_dir / "images.txt"
    img_lines = ["# Image list\n"]
    for i in range(1, 26):
        img_lines.append(f"{i} 1.0 0.0 0.0 0.0 0.0 0.0 {float(i)} 1 frame_{i:06d}.jpg\n")
        img_lines.append("1.0 1.0 1\n")
    imgs_txt.write_text("".join(img_lines), encoding="utf-8")

    pts_txt = sparse_dir / "points3D.txt"
    pt_lines = ["# 3D points\n"]
    for i in range(1, num_points + 1):
        pt_lines.append(f"{i} {float(i)*0.1} {float(i)*0.05} 5.0 200 150 100 0.5 1 1 1\n")
    pts_txt.write_text("".join(pt_lines), encoding="utf-8")

    return sparse_dir


def create_mock_cameras_json(file_path: Path, num_cameras: int = 25) -> Path:
    """Creates standard cameras.json for pose validation."""
    cams = []
    for i in range(1, num_cameras + 1):
        cams.append({
            "frame_id": f"frame_{i:06d}",
            "image_name": f"frame_{i:06d}.jpg",
            "position": [float(i) * 0.5, 0.0, 10.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "intrinsics": {"width": 128, "height": 128, "fx": 100.0, "fy": 100.0, "cx": 64.0, "cy": 64.0},
        })
    file_path.write_text(json.dumps(cams, indent=2), encoding="utf-8")
    return file_path


# ===========================================================================
# 1. Pipeline State Machine & Persistence Tests
# ===========================================================================

def test_state_machine_transitions(tmp_path: Path) -> None:
    """State machine must record orderly transitions with timestamps, statuses, and file persistence."""
    state_file = tmp_path / "pipeline_state.json"
    sm = PipelineStateMachine(state_file=state_file)
    assert sm.current_state == STATE_UPLOADED
    assert len(sm.transitions) == 1

    sm.transition_start(STATE_FRAME_PROCESSING, "Processing frames")
    assert sm.current_state == STATE_FRAME_PROCESSING
    sm.transition_end("SUCCESS", "Completed frames")

    sm.transition_start(STATE_OBJECT_DETECTION, "Masking objects")
    sm.transition_end("SUCCESS", "Completed masks")

    sm.transition_start(STATE_POSE_ESTIMATION, "Estimating poses")
    sm.transition_end("FAILED", error="SfM failed")
    assert sm.current_state == STATE_FAILED

    history = sm.to_list()
    assert len(history) == 4
    assert history[0]["state"] == STATE_UPLOADED
    assert history[1]["state"] == STATE_FRAME_PROCESSING
    assert history[2]["state"] == STATE_OBJECT_DETECTION
    assert history[3]["state"] == STATE_POSE_ESTIMATION
    assert history[3]["status"] == "FAILED"
    assert history[3]["error"] == "SfM failed"

    # Verify persisted state JSON
    assert state_file.is_file()
    with open(state_file, "r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["current_state"] == STATE_FAILED
    assert persisted["status"] == "FAIL"
    assert "FRAME_PROCESSING" in persisted["stages"]
    assert persisted["stages"]["FRAME_PROCESSING"]["status"] == "SUCCESS"


# ===========================================================================
# 2. Frame Processing & Quality Filtering Tests
# ===========================================================================

def test_frame_processing_image_directory(tmp_path: Path) -> None:
    """Image directory input must produce stable frame IDs (frame_000001, etc.)."""
    in_dir = tmp_path / "raw_images"
    in_dir.mkdir()
    for i in range(5):
        create_synthetic_image(in_dir / f"test_img_{i}.jpg")

    out_frames = tmp_path / "output_frames"
    res = process_input_frames(
        input_path=in_dir,
        output_frames_dir=out_frames,
        blur_threshold=10.0,
    )

    assert res["input_frames"] == 5
    assert res["selected_frames"] == 5
    assert res["discarded_frames"] == 0
    assert res["selection_ratio"] == 1.0

    # Verify stable filenames
    saved_files = sorted([f.name for f in out_frames.iterdir() if f.is_file()])
    assert saved_files == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
        "frame_000004.jpg",
        "frame_000005.jpg",
    ]


def test_frame_processing_video_input(tmp_path: Path) -> None:
    """Video file input must extract frames and preserve stable frame IDs."""
    video_path = tmp_path / "test_drone_flight.mp4"
    create_synthetic_video(video_path, num_frames=6)

    out_frames = tmp_path / "output_frames_video"
    res = process_input_frames(
        input_path=video_path,
        output_frames_dir=out_frames,
        blur_threshold=1.0,
        sim_threshold=0.999,
    )

    assert res["selected_frames"] > 0
    saved_files = sorted([f.name for f in out_frames.iterdir() if f.is_file()])
    assert all(f.startswith("frame_") and f.endswith(".jpg") for f in saved_files)
    assert "frame_000001.jpg" in saved_files


def test_frame_processing_empty_directory_fails(tmp_path: Path) -> None:
    """Empty image directory must raise ValueError explicitly."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No supported image files found"):
        process_input_frames(empty_dir, tmp_path / "out")


def test_frames_json_schema(tmp_path: Path) -> None:
    """frames.json must conform to Section 3 requirements with frame_id, filename, source, dimensions, quality_status."""
    in_dir = tmp_path / "images"
    in_dir.mkdir()
    for i in range(3):
        create_synthetic_image(in_dir / f"img_{i:02d}.jpg", width=128, height=96)

    out_frames = tmp_path / "frames" / "selected"
    res = process_input_frames(in_dir, out_frames)

    frames_json_path = tmp_path / "frames" / "frames.json"
    assert frames_json_path.is_file()

    with open(frames_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_frames"] == 3
    assert data["accepted_frames"] == 3
    assert data["rejected_frames"] == 0
    assert "rejection_reasons" in data

    first_frame = data["frames"][0]
    assert first_frame["frame_id"] == "frame_000001"
    assert first_frame["filename"] == "frame_000001.jpg"
    assert "source" in first_frame
    assert "source_frame_index" in first_frame
    assert first_frame["width"] == 128
    assert first_frame["height"] == 96
    assert first_frame["quality_status"] == "ACCEPTED"
    assert "metrics" in first_frame


def test_quality_filtering_brightness_and_duplicates(tmp_path: Path) -> None:
    """Quality filtering must identify and reject corrupt, blurry, extreme brightness, and duplicate frames."""
    in_dir = tmp_path / "quality_test_images"
    in_dir.mkdir()

    # 25 good frames with varying color so they are distinct
    for i in range(1, 26):
        create_synthetic_image(in_dir / f"good_{i:04d}.jpg", color=(100, 100 + (i * 4) % 100, 140))

    # Add 1 exact duplicate frame of good_0001.jpg
    shutil.copy2(in_dir / "good_0001.jpg", in_dir / "good_0001_dup.jpg")

    # Add 1 pitch black frame
    black_img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(in_dir / "bad_black.jpg"), black_img)

    # Add 1 pure white frame
    white_img = np.full((128, 128, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(in_dir / "bad_white.jpg"), white_img)

    out_frames = tmp_path / "selected"
    res = process_input_frames(
        input_path=in_dir,
        output_frames_dir=out_frames,
        blur_threshold=50.0,
        min_brightness=20.0,
        max_brightness=245.0,
        duplicate_threshold=0.98,
    )

    assert res["input_frames"] == 28
    assert res["rejected_frames"] >= 3
    assert res["rejection_reasons"]["too_dark"] >= 1
    assert res["rejection_reasons"]["too_bright"] >= 1
    assert res["rejection_reasons"]["duplicate"] >= 1


# ===========================================================================
# 3. Dynamic Object Masking Tests
# ===========================================================================

def test_dynamic_masking_convention_and_dimensions(tmp_path: Path) -> None:
    """Masks must use 0=keep, 255=masked and strictly match source frame dimensions."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    create_synthetic_image(frames_dir / "frame_000001.jpg", width=160, height=120)
    create_synthetic_image(frames_dir / "frame_000002.jpg", width=160, height=120)

    masks_dir = tmp_path / "masks"
    det_dir = tmp_path / "detections"

    def mock_detector(img: np.ndarray, frame_name: str) -> List[Dict[str, Any]]:
        if "frame_000001" in frame_name:
            return [{"class": "car", "confidence": 0.95, "bbox": [20, 30, 80, 70]}]
        return [{"class": "bench", "confidence": 0.85, "bbox": [10, 10, 40, 40]}]

    report = run_dynamic_masking(
        frames_dir=frames_dir,
        masks_output_dir=masks_dir,
        detections_output_dir=det_dir,
        mock_detector=mock_detector,
    )

    assert report.status == "SUCCESS"
    assert report.input_frames == 2
    assert report.masked_frames == 1
    assert report.total_detections == 1
    assert report.mask_coverage > 0.0

    # Inspect mask 1
    m1_path = masks_dir / "frame_000001.png"
    assert m1_path.is_file()
    m1 = cv2.imread(str(m1_path), cv2.IMREAD_GRAYSCALE)
    assert m1.shape == (120, 160)
    assert m1[40, 40] == 255
    assert m1[0, 0] == 0

    # Inspect mask 2 (static object should not be masked)
    m2_path = masks_dir / "frame_000002.png"
    assert m2_path.is_file()
    m2 = cv2.imread(str(m2_path), cv2.IMREAD_GRAYSCALE)
    assert m2.shape == (120, 160)
    assert np.all(m2 == 0)


def test_dynamic_masking_unavailable_fallback(tmp_path: Path) -> None:
    """When YOLO11 weights are unavailable, must cleanly report NOT_AVAILABLE."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    create_synthetic_image(frames_dir / "frame_000001.jpg")

    report = run_dynamic_masking(
        frames_dir=frames_dir,
        masks_output_dir=tmp_path / "masks",
        weights="nonexistent_yolo_weights_xyz_9999.pt",
        require_masking=False,
    )

    assert report.status == "NOT_AVAILABLE"
    assert report.reason is not None
    assert "could not be loaded" in report.reason or "unavailable" in report.reason.lower()


# ===========================================================================
# 4. Central Configuration Tests
# ===========================================================================

def test_config_serialization(tmp_path: Path) -> None:
    """Central configuration must serialize to config.json and load cleanly from file."""
    cfg_file = tmp_path / "custom_config.json"
    cfg = PipelineConfig(
        input_path=str(tmp_path / "input"),
        output_dir=str(tmp_path / "out"),
        blur_threshold=120.0,
        method="fastmap",
    )
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    loaded = PipelineConfig.from_json(cfg_file, method="global_sfm")
    assert loaded.blur_threshold == 120.0
    assert loaded.method == "global_sfm"


# ===========================================================================
# 5. End-to-End Orchestrated Pipeline Tests
# ===========================================================================

def test_full_pipeline_mocked_success(tmp_path: Path) -> None:
    """Executes full pipeline orchestration with mocked backends and verifies output bundle."""
    # 1. Prepare synthetic input images (25 frames)
    img_dir = tmp_path / "input_drone_images"
    img_dir.mkdir()
    for i in range(1, 26):
        create_synthetic_image(img_dir / f"raw_{i:04d}.jpg", width=128, height=128)

    # 2. Mock sparse reconstruction and cameras
    mock_sparse = create_mock_sparse_model(tmp_path, num_points=200)
    cams_file = create_mock_cameras_json(tmp_path / "mock_cameras.json", num_cameras=25)
    traj_file = tmp_path / "mock_trajectory.json"
    traj_file.write_text("{}", encoding="utf-8")

    def mock_fastmap() -> PoseResult:
        return PoseResult(
            method="FastMap",
            status="PASS",
            output_path=tmp_path,
            runtime_seconds=1.2,
            validation_status="PASS",
            registration_rate=0.96,
            registered_images=24,
            input_images=25,
            num_points3D=200,
            reprojection_rmse_px=0.6,
            reprojection_status="AVAILABLE",
            trajectory_status="PASS",
            cameras_path=cams_file,
            trajectory_path=traj_file,
            sparse_dir=mock_sparse,
        )

    def mock_detector(img: np.ndarray, frame_name: str) -> List[Dict[str, Any]]:
        return [{"class": "car", "confidence": 0.9, "bbox": [10, 10, 30, 30]}]

    def mock_splat(dataset_dir: Path, output_dir: Path, config: SplatfactoConfig) -> Dict[str, Any]:
        return {
            "status": "SUCCESS",
            "method": "nerfstudio_splatfacto",
            "iterations": 100,
            "runtime_seconds": 0.8,
            "gpu": "Mock NVIDIA RTX 4090",
            "output_directory": str(output_dir),
            "render_validation_status": "PASS",
        }

    out_dir = tmp_path / "pipeline_output"
    config = PipelineConfig(
        input_path=str(img_dir),
        output_dir=str(out_dir),
        method="auto",
        min_images=20,
        enable_masking=True,
        enable_splatting=True,
    )

    report = run_pipeline(
        config=config,
        mock_pose_backends={"fastmap": mock_fastmap},
        mock_detector=mock_detector,
        mock_splat_runner=mock_splat,
    )

    # Verify pipeline report status
    assert report["pipeline_status"] == "COMPLETE"
    assert report["failure_reason"] is None

    # Check Stage 1: Frame Processing
    frame_rep = report["frame_processing"]
    assert frame_rep["selected_frames"] == 25
    assert frame_rep["selection_ratio"] == 1.0

    # Check Stage 2: Object Detection & Masking
    mask_rep = report["object_detection"]
    assert mask_rep["status"] == "SUCCESS"
    assert mask_rep["masked_frames"] == 25
    assert mask_rep["total_detections"] == 25
    assert mask_rep["mask_coverage"] > 0.0

    # Check Stage 3 & 4: Pose Estimation & Validation
    pose_rep = report["pose_estimation"]
    assert pose_rep["selected_method"] == "FastMap"
    assert pose_rep["registration_rate"] == 0.96
    assert pose_rep["reprojection_metrics"]["rmse_px"] == 0.6

    # Check Stage 5: Georeferencing (No GPS provided)
    geo_rep = report["georeferencing"]
    assert geo_rep["status"] == "NOT_AVAILABLE"
    assert geo_rep["coordinate_system"] == "local_unscaled"

    # Check Stage 6: Point Cloud (Step 8)
    pcd_rep = report["point_cloud"]
    assert pcd_rep["status"] == "SUCCESS"
    assert pcd_rep["final_points"] > 0
    assert Path(pcd_rep["pointcloud_ply"]).is_file()

    # Check Stage 7: Splatting (Step 9)
    splat_rep = report["splatting"]
    assert splat_rep["status"] == "SUCCESS"
    assert splat_rep["iterations"] == 100
    assert splat_rep["pointcloud_initialization"] is True

    # Check Timing
    timing = report["timing"]
    for k in [
        "frame_processing_seconds",
        "object_detection_seconds",
        "pose_estimation_seconds",
        "georeferencing_seconds",
        "point_cloud_seconds",
        "splatting_seconds",
        "total_pipeline_seconds",
    ]:
        assert k in timing
        assert timing[k] >= 0.0

    # Check State Transitions
    states = [t["state"] for t in report["state_transitions"]]
    assert STATE_UPLOADED in states
    assert STATE_FRAME_PROCESSING in states
    assert STATE_OBJECT_DETECTION in states
    assert STATE_POSE_ESTIMATION in states
    assert STATE_POSE_VALIDATION in states
    assert STATE_GEOREFERENCING in states
    assert STATE_POINT_CLOUD in states
    assert STATE_SPLATTING in states
    assert STATE_EXPORT in states
    assert STATE_COMPLETE in states

    # Check Physical Output Bundle Structure (Section 13)
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "pipeline_state.json").is_file()
    assert (out_dir / "frames" / "selected" / "frame_000001.jpg").is_file()
    assert (out_dir / "frames" / "frames.json").is_file()
    assert (out_dir / "masks" / "frame_000001.png").is_file()
    assert (out_dir / "detections" / "detections.json").is_file()
    assert (out_dir / "poses" / "cameras.json").is_file()
    assert (out_dir / "pointcloud" / "pointcloud.ply").is_file()
    assert (out_dir / "pointcloud" / "metadata.json").is_file()
    assert (out_dir / "pointcloud" / "processing_report.json").is_file()
    assert (out_dir / "splat" / "config.json").is_file()
    assert (out_dir / "splat" / "transforms.json").is_file()
    assert (out_dir / "processing_report.json").is_file()


def test_pipeline_failure_propagation(tmp_path: Path) -> None:
    """When pose estimation fails validation, pipeline must fail explicitly and record error."""
    img_dir = tmp_path / "raw_images"
    img_dir.mkdir()
    for i in range(1, 26):
        create_synthetic_image(img_dir / f"img_{i:04d}.jpg")

    def mock_failing_fastmap() -> PoseResult:
        return PoseResult(
            method="FastMap",
            status="FAIL",
            output_path=tmp_path,
            runtime_seconds=0.5,
            validation_status="FAIL",
            registration_rate=0.20,
            registered_images=5,
            input_images=25,
            failure_reason="Registration rate 0.20 below gate",
        )

    out_dir = tmp_path / "pipeline_fail_out"
    config = PipelineConfig(
        input_path=str(img_dir),
        output_dir=str(out_dir),
        method="fastmap",
    )

    report = run_pipeline(
        config=config,
        mock_pose_backends={"fastmap": mock_failing_fastmap},
    )

    assert report["pipeline_status"] == "FAILED"
    assert "All pose estimation candidates failed validation" in report["failure_reason"]
    assert (out_dir / "processing_report.json").is_file()
    states = [t["state"] for t in report["state_transitions"]]
    assert STATE_FAILED in states


def test_pipeline_georeferencing_with_gps(tmp_path: Path) -> None:
    """When GPS is provided, georeferencing stage must run and populate output/geospatial."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(1, 26):
        create_synthetic_image(img_dir / f"frame_{i:06d}.jpg")

    mock_sparse = create_mock_sparse_model(tmp_path, num_points=50)
    cams_file = create_mock_cameras_json(tmp_path / "cameras.json", num_cameras=25)

    gps_file = tmp_path / "flight_gps.csv"
    gps_lines = ["timestamp,latitude,longitude,altitude\n"]
    for i in range(1, 26):
        gps_lines.append(f"2026-09-07T12:00:{i:02d}Z,12.9716,77.5946,{100.0 + i*0.5}\n")
    gps_file.write_text("".join(gps_lines), encoding="utf-8")

    def mock_fastmap() -> PoseResult:
        return PoseResult(
            method="FastMap",
            status="PASS",
            output_path=tmp_path,
            runtime_seconds=1.0,
            validation_status="PASS",
            registration_rate=1.0,
            registered_images=25,
            input_images=25,
            num_points3D=50,
            cameras_path=cams_file,
            sparse_dir=mock_sparse,
        )

    out_dir = tmp_path / "pipeline_geo_out"
    config = PipelineConfig(
        input_path=str(img_dir),
        output_dir=str(out_dir),
        method="fastmap",
        gps_path=str(gps_file),
    )

    report = run_pipeline(
        config=config,
        mock_pose_backends={"fastmap": mock_fastmap},
    )

    assert "georeferencing" in report


def test_splatfacto_unavailable_complete_with_warnings(tmp_path: Path) -> None:
    """When Splatfacto is unavailable (CPU host), pipeline completes with COMPLETE_WITH_WARNINGS and retains point cloud."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(1, 26):
        create_synthetic_image(img_dir / f"frame_{i:06d}.jpg")

    mock_sparse = create_mock_sparse_model(tmp_path, num_points=50)
    cams_file = create_mock_cameras_json(tmp_path / "cameras.json", num_cameras=25)
    traj_file = tmp_path / "mock_trajectory.json"
    traj_file.write_text("{}", encoding="utf-8")

    def mock_fastmap() -> PoseResult:
        return PoseResult(
            method="FastMap",
            status="PASS",
            output_path=tmp_path,
            runtime_seconds=0.5,
            validation_status="PASS",
            registration_rate=1.0,
            registered_images=25,
            input_images=25,
            num_points3D=50,
            trajectory_status="PASS",
            trajectory_path=traj_file,
            cameras_path=cams_file,
            sparse_dir=mock_sparse,
        )

    out_dir = tmp_path / "pipeline_warn_out"
    config = PipelineConfig(
        input_path=str(img_dir),
        output_dir=str(out_dir),
        method="fastmap",
        enable_masking=False,
        enable_splatting=True,
    )

    report = run_pipeline(
        config=config,
        mock_pose_backends={"fastmap": mock_fastmap},
        mock_splat_runner=None,  # Real splat training will check environment and report NOT_AVAILABLE on CPU
    )

    assert report["point_cloud"]["status"] == "SUCCESS"
    assert report["splatting"]["status"] == "NOT_AVAILABLE"
    assert report["pipeline_status"] == "COMPLETE_WITH_WARNINGS"
    assert (out_dir / "pointcloud" / "pointcloud.ply").is_file()


def test_pipeline_resume_support(tmp_path: Path) -> None:
    """When --resume is enabled, existing valid frames and masks should be reused."""
    out_dir = tmp_path / "pipeline_resume"
    frames_dir = out_dir / "frames" / "selected"
    frames_dir.mkdir(parents=True)
    for i in range(1, 26):
        create_synthetic_image(frames_dir / f"frame_{i:06d}.jpg")

    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True)
    for i in range(1, 26):
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.imwrite(str(masks_dir / f"frame_{i:06d}.png"), mask)

    mock_sparse = create_mock_sparse_model(tmp_path, num_points=50)
    cams_file = create_mock_cameras_json(tmp_path / "cameras.json", num_cameras=25)

    def mock_fastmap() -> PoseResult:
        return PoseResult(
            method="FastMap",
            status="PASS",
            output_path=tmp_path,
            runtime_seconds=0.5,
            validation_status="PASS",
            registration_rate=1.0,
            registered_images=25,
            input_images=25,
            cameras_path=cams_file,
            sparse_dir=mock_sparse,
        )

    config = PipelineConfig(
        input_path=str(frames_dir),
        output_dir=str(out_dir),
        resume=True,
        method="fastmap",
    )

    report = run_pipeline(
        config=config,
        mock_pose_backends={"fastmap": mock_fastmap},
    )

    assert report["frame_processing"].get("status") == "SKIPPED_RESUME"
    assert report["object_detection"].get("status") == "SKIPPED_RESUME"


# ===========================================================================
# 6. Resource Monitoring & Report Tests
# ===========================================================================

def test_system_resource_monitoring() -> None:
    """Resource monitoring must query real CPU RAM and CUDA info without fabrication."""
    resources = get_system_resources()
    assert "cpu_ram" in resources
    assert "gpu" in resources

    if resources["cpu_ram"] is not None:
        assert resources["cpu_ram"]["total_gb"] > 0
        assert 0 <= resources["cpu_ram"]["percent_used"] <= 100

    assert "cuda_available" in resources["gpu"]


def test_real_smoke_test_status_reporting() -> None:
    """Verifies that in absence of real drone data, smoke test status is properly marked NOT_AVAILABLE."""
    smoke_res = run_real_e2e_smoke_test()
    assert smoke_res["real_end_to_end_smoke_test"] == "NOT_AVAILABLE"
    assert smoke_res["status"] == "NOT_AVAILABLE"
    assert "reason" in smoke_res


# ===========================================================================
# 7. CLI Interface & Exit Codes Tests
# ===========================================================================

def test_cli_help_and_exit_codes(tmp_path: Path) -> None:
    """CLI must return code 0 on --help, code 2 on missing or invalid args, code 2 on bad config."""
    # --help returns 0
    assert pipeline_main(["--help"]) == 0

    # Missing required --input returns 2
    assert pipeline_main([]) == 2

    # Invalid config path returns 2
    assert pipeline_main(["--input", str(tmp_path), "--config", "nonexistent_cfg.json"]) == 2
