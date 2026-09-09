"""Comprehensive unit and integration tests for SIH26158 Step 6: Adaptive Pose Engine."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.pose_engine.benchmark import benchmark_adaptive, run_benchmark
from reconstruction.pose_engine.engine import (
    PoseResult,
    ValidationPolicy,
    evaluate_candidate,
    run_adaptive_engine,
    run_preflight_checks,
)


def helper_create_dummy_images(tmpdir: Path, num_images: int = 25) -> Path:
    """Creates valid minimal JPEG image files for preflight testing."""
    img_dir = tmpdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    jpeg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x01, 0x00, 0x48, 0x00, 0x48, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F,
        0x00, 0xBF, 0x00, 0xFF, 0xD9,
    ])
    for i in range(num_images):
        (img_dir / f"frame_{i:06d}.jpg").write_bytes(jpeg_bytes)
    return img_dir


def helper_create_mock_pose_result(
    method: str,
    status: str = "PASS",
    registration_rate: float = 0.95,
    reprojection_rmse: float = 0.65,
    runtime: float = 5.0,
    out_dir: Optional[Path] = None,
) -> PoseResult:
    """Creates a mock PoseResult for unit testing orchestration state machines."""
    base_dir = out_dir or Path(tempfile.gettempdir())
    cam_file = base_dir / f"{method.lower()}_cameras.json"
    cam_file.write_text("[]", encoding="utf-8")
    traj_file = base_dir / f"{method.lower()}_trajectory.json"
    traj_file.write_text("{}", encoding="utf-8")
    sparse = base_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    (sparse / "cameras.txt").write_text("# cameras", encoding="utf-8")

    return PoseResult(
        method=method,
        status=status,
        output_path=base_dir,
        runtime_seconds=runtime,
        validation_status=status,
        registration_rate=registration_rate,
        registered_images=20 if status == "PASS" else 0,
        input_images=25,
        num_points3D=500 if status == "PASS" else 0,
        reprojection_rmse_px=reprojection_rmse if status == "PASS" else None,
        reprojection_status="AVAILABLE" if status == "PASS" else "NOT_AVAILABLE",
        trajectory_status="AVAILABLE" if status == "PASS" else "NOT_AVAILABLE",
        cameras_path=cam_file,
        trajectory_path=traj_file,
        sparse_dir=sparse,
        start_time="2026-09-07 19:00:00",
        end_time="2026-09-07 19:00:05",
        failure_reason=None if status == "PASS" else f"Mock failure in {method}",
    )


# ---------------------------------------------------------------------------
# Preflight Tests
# ---------------------------------------------------------------------------

def test_preflight_nonexistent_directory():
    """Missing image directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_preflight_checks("nonexistent_path_xyz_123")


def test_preflight_empty_directory():
    """Empty image directory must be strictly rejected with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="0 supported images found"):
            run_preflight_checks(tmpdir, min_images=20)


def test_preflight_insufficient_images():
    """Directory with fewer images than min_images must be rejected with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = helper_create_dummy_images(Path(tmpdir), num_images=5)
        with pytest.raises(ValueError, match="Insufficient images"):
            run_preflight_checks(img_dir, min_images=20)


def test_preflight_valid_dataset():
    """Directory with >= min_images readable files must pass preflight checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = helper_create_dummy_images(Path(tmpdir), num_images=25)
        info = run_preflight_checks(img_dir, min_images=20)
        assert info["status"] == "PASS"
        assert info["total_images"] == 25
        assert info["min_images_required"] == 20
        assert ".jpg" in info["formats"]


# ---------------------------------------------------------------------------
# Orchestration State Machine Tests (Using Mocks)
# ---------------------------------------------------------------------------

def test_adaptive_short_circuit_fastmap_pass():
    """When FastMap passes validation, Global SfM and VGGT must NOT be called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=22)
        out_dir = tmp_path / "adaptive_out"

        called = {"fastmap": 0, "global_sfm": 0, "vggt": 0}

        def mock_fastmap():
            called["fastmap"] += 1
            return helper_create_mock_pose_result("FastMap", status="PASS", runtime=4.0, out_dir=tmp_path)

        def mock_global_sfm():
            called["global_sfm"] += 1
            return helper_create_mock_pose_result("COLMAP_Global_SfM", status="PASS", out_dir=tmp_path)

        def mock_vggt():
            called["vggt"] += 1
            return helper_create_mock_pose_result("VGGT", status="PASS", out_dir=tmp_path)

        report = run_adaptive_engine(
            images=img_dir,
            output=out_dir,
            method="auto",
            min_images=20,
            mock_backends={
                "fastmap": mock_fastmap,
                "global_sfm": mock_global_sfm,
                "vggt": mock_vggt,
            },
        )

        assert report["overall_status"] == "PASS"
        assert report["selected_method"] == "FastMap"
        assert called["fastmap"] == 1
        assert called["global_sfm"] == 0
        assert called["vggt"] == 0
        assert len(report["attempts"]) == 1
        assert report["attempts"][0]["decision"] == "ACCEPT"


def test_adaptive_fallback_fastmap_fail_to_global_sfm():
    """When FastMap fails validation, Global SfM is called and selected upon passing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=22)
        out_dir = tmp_path / "adaptive_out"

        called = {"fastmap": 0, "global_sfm": 0, "vggt": 0}

        def mock_fastmap():
            called["fastmap"] += 1
            return helper_create_mock_pose_result("FastMap", status="BLOCKED", runtime=1.0, out_dir=tmp_path)

        def mock_global_sfm():
            called["global_sfm"] += 1
            return helper_create_mock_pose_result("COLMAP_Global_SfM", status="PASS", runtime=8.0, out_dir=tmp_path)

        def mock_vggt():
            called["vggt"] += 1
            return helper_create_mock_pose_result("VGGT", status="PASS", out_dir=tmp_path)

        report = run_adaptive_engine(
            images=img_dir,
            output=out_dir,
            method="auto",
            min_images=20,
            mock_backends={
                "fastmap": mock_fastmap,
                "global_sfm": mock_global_sfm,
                "vggt": mock_vggt,
            },
        )

        assert report["overall_status"] == "PASS"
        assert report["selected_method"] == "COLMAP_Global_SfM"
        assert called["fastmap"] == 1
        assert called["global_sfm"] == 1
        assert called["vggt"] == 0
        assert len(report["attempts"]) == 2
        assert report["attempts"][0]["decision"] == "REJECT"
        assert report["attempts"][1]["decision"] == "ACCEPT"
        # Cumulative runtime must include both attempted backends
        assert report["total_runtime_seconds"] == 9.0


def test_adaptive_fallback_to_vggt():
    """When FastMap and Global SfM fail, VGGT is called and selected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=22)
        out_dir = tmp_path / "adaptive_out"

        called = {"fastmap": 0, "global_sfm": 0, "vggt": 0}

        def mock_fastmap():
            called["fastmap"] += 1
            return helper_create_mock_pose_result("FastMap", status="BLOCKED", runtime=1.0, out_dir=tmp_path)

        def mock_global_sfm():
            called["global_sfm"] += 1
            return helper_create_mock_pose_result("COLMAP_Global_SfM", status="FAIL", runtime=3.0, out_dir=tmp_path)

        def mock_vggt():
            called["vggt"] += 1
            res = helper_create_mock_pose_result("VGGT", status="PASS", runtime=6.0, out_dir=tmp_path)
            # Feedforward VGGT may report reprojection as NOT_AVAILABLE
            res.reprojection_status = "NOT_AVAILABLE"
            res.reprojection_rmse_px = None
            return res

        report = run_adaptive_engine(
            images=img_dir,
            output=out_dir,
            method="auto",
            min_images=20,
            mock_backends={
                "fastmap": mock_fastmap,
                "global_sfm": mock_global_sfm,
                "vggt": mock_vggt,
            },
        )

        assert report["overall_status"] == "PASS"
        assert report["selected_method"] == "VGGT"
        assert called["fastmap"] == 1
        assert called["global_sfm"] == 1
        assert called["vggt"] == 1
        assert len(report["attempts"]) == 3
        assert report["total_runtime_seconds"] == 10.0


def test_adaptive_all_methods_fail():
    """When all methods fail, engine reports overall FAIL and selected_method is None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=22)
        out_dir = tmp_path / "adaptive_out"

        def mock_fail(method_name: str):
            return helper_create_mock_pose_result(method_name, status="FAIL", runtime=2.0, out_dir=tmp_path)

        report = run_adaptive_engine(
            images=img_dir,
            output=out_dir,
            method="auto",
            min_images=20,
            mock_backends={
                "fastmap": lambda: mock_fail("FastMap"),
                "global_sfm": lambda: mock_fail("COLMAP_Global_SfM"),
                "vggt": lambda: mock_fail("VGGT"),
            },
        )

        assert report["overall_status"] == "FAIL"
        assert report["selected_method"] is None
        assert len(report["attempts"]) == 3
        assert all(a["decision"] == "REJECT" for a in report["attempts"])


def test_explicit_method_selection():
    """When a specific method is requested via --method, only that candidate is executed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=22)
        out_dir = tmp_path / "adaptive_out"

        called = {"fastmap": 0, "vggt": 0}

        def mock_vggt():
            called["vggt"] += 1
            return helper_create_mock_pose_result("VGGT", status="PASS", out_dir=tmp_path)

        report = run_adaptive_engine(
            images=img_dir,
            output=out_dir,
            method="vggt",
            min_images=20,
            mock_backends={"vggt": mock_vggt},
        )

        assert report["selected_method"] == "VGGT"
        assert called["vggt"] == 1
        assert called["fastmap"] == 0
        assert len(report["attempts"]) == 1


def test_standardized_selected_output():
    """Successful candidate output files are standardized under output/selected/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=22)
        out_dir = tmp_path / "adaptive_out"

        def mock_fastmap():
            res = helper_create_mock_pose_result("FastMap", status="PASS", out_dir=tmp_path)
            # Create dummy pose_report
            (res.output_path / "pose_report.json").write_text('{"validation_status": "PASS"}', encoding="utf-8")
            return res

        report = run_adaptive_engine(
            images=img_dir,
            output=out_dir,
            method="fastmap",
            min_images=20,
            mock_backends={"fastmap": mock_fastmap},
        )

        selected_dir = out_dir / "selected"
        assert selected_dir.is_dir()
        assert (selected_dir / "cameras.json").is_file()
        assert (selected_dir / "trajectory.json").is_file()
        assert (selected_dir / "sparse").is_dir()
        assert (selected_dir / "pose_report.json").is_file()
        assert (out_dir / "pose_engine_report.json").is_file()


# ---------------------------------------------------------------------------
# Benchmark 5-Way Integration Test
# ---------------------------------------------------------------------------

def test_run_5way_benchmark():
    """5-way benchmark must evaluate all 5 engines and export all 6 JSON reports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=5)
        bench_out = tmp_path / "benchmark"

        # Mock adaptive engine candidate for benchmark verification
        mock_res = helper_create_mock_pose_result("FastMap", status="PASS", runtime=3.0, out_dir=tmp_path)

        comparison = run_benchmark(
            images=img_dir,
            output=bench_out,
            mock_vggt=True,
            min_images=1,  # Lower gate for quick unit test
            mock_backends={"fastmap": lambda: mock_res},
        )

        assert "colmap_incremental" in comparison
        assert "fastmap" in comparison
        assert "colmap_global_sfm" in comparison
        assert "vggt" in comparison
        assert "adaptive" in comparison
        assert "adaptive_engine" in comparison

        assert (bench_out / "colmap_result.json").is_file()
        assert (bench_out / "fastmap_result.json").is_file()
        assert (bench_out / "global_sfm_result.json").is_file()
        assert (bench_out / "vggt_result.json").is_file()
        assert (bench_out / "adaptive_result.json").is_file()
        assert (bench_out / "comparison.json").is_file()

        adap = comparison["adaptive"]
        assert adap["backend"] == "Adaptive_Engine"
        assert adap["selected_method"] == "FastMap"
        assert adap["status"] == "PASS"
