"""Comprehensive unit and integration tests for SIH26158 Step 3: FastMap Integration & Benchmark."""

import json
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.pose_engine.fastmap import (
    FastMapAdapter,
    get_fastmap_commit,
    get_fastmap_environment_info,
    run_fastmap,
)
from reconstruction.pose_engine.benchmark import run_benchmark
from reconstruction.validation.geometry import project_3d_to_2d
from reconstruction.validation.model_io import Camera


def helper_create_synthetic_model(tmpdir: Path) -> Path:
    """Helper creating a synthetic COLMAP-format sparse model."""
    model_dir = tmpdir / "sparse" / "0"
    model_dir.mkdir(parents=True)

    with open(model_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list\n1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n")

    points = [(1, np.array([0.0, 0.0, 10.0]))]
    with open(model_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D points\n1 0.0 0.0 10.0 255 255 255 0.5 1 0\n")

    cam = Camera(1, "PINHOLE", 1920, 1080, np.array([1000.0, 1000.0, 960.0, 540.0]))
    with open(model_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list\n")
        for i in range(3):
            q = np.array([1.0, 0.0, 0.0, 0.0])
            t = np.array([-float(i) * 0.5, 0.0, 0.0])
            f.write(f"{i+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 frame_{i+1:06d}.jpg\n")
            proj, _ = project_3d_to_2d(points[0][1], q, t, cam)
            f.write(f"{proj[0]:.4f} {proj[1]:.4f} 1\n")

    return model_dir


def test_fastmap_environment_detection():
    """Environment inspection must accurately detect repository, commit, and platform support."""
    env = get_fastmap_environment_info()
    assert "repository" in env
    assert env["repository"] == "https://github.com/pals-ttic/fastmap"
    assert env["repo_path"] is not None

    commit = get_fastmap_commit()
    assert commit is not None
    assert len(commit) == 40  # Valid full git SHA-1

    # On Windows without CUDA, FastMap must report supported = False with specific blockers
    if sys.platform == "win32":
        assert env["supported"] is False
        assert any("Linux only" in b for b in env["blockers"])


def test_fastmap_adapter_normalize_poses():
    """FastMap adapter should extract normalized poses from COLMAP-format output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = helper_create_synthetic_model(Path(tmpdir))
        poses = FastMapAdapter.normalize_poses(model_dir)

        assert len(poses) == 3
        assert poses[0]["frame_id"] == "frame_000001"
        assert poses[0]["image_name"] == "frame_000001.jpg"
        assert poses[0]["position"] == [0.0, 0.0, 0.0]
        assert poses[1]["position"] == [0.5, 0.0, 0.0]
        assert poses[0]["rotation_quaternion"] == [1.0, 0.0, 0.0, 0.0]


def test_fastmap_adapter_validation():
    """FastMap adapter should feed models into Step 2 validation engine."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_dir = helper_create_synthetic_model(tmp_path)
        report_file = tmp_path / "fastmap_val.json"
        traj_file = tmp_path / "fastmap_traj.json"

        report = FastMapAdapter.validate(
            model_dir=model_dir,
            output_report_path=report_file,
            trajectory_output_path=traj_file,
        )

        assert report["validation_status"] == "PASS"
        assert report_file.is_file()
        assert traj_file.is_file()


def test_run_fastmap_invalid_input_dir():
    """Non-existent image directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_fastmap(images="nonexistent_folder_xyz", output="reconstruction/test_output/fastmap")


def test_run_fastmap_empty_input_dir():
    """Empty image directory must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No supported images found"):
            run_fastmap(images=tmpdir, output="reconstruction/test_output/fastmap")


def test_run_fastmap_blocked_on_incompatible_env():
    """On Windows/No-CUDA host, run_fastmap should return status BLOCKED without crashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(3):
            (img_dir / f"frame_{i:04d}.jpg").write_bytes(b"dummy")

        out_dir = tmp_path / "fastmap_out"
        res = run_fastmap(images=img_dir, output=out_dir)

        # On this environment (Windows / AMD Radeon / CPU torch), status should be BLOCKED
        if sys.platform == "win32":
            assert res["status"] == "BLOCKED"
            assert "blocker_reason" in res
            assert (out_dir / "fastmap_report.json").is_file()


def test_run_benchmark_schema():
    """Benchmark runner should produce colmap_result.json, fastmap_result.json, and comparison.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(2):
            (img_dir / f"frame_{i:04d}.jpg").write_bytes(b"dummy")

        bench_out = tmp_path / "benchmark"
        comparison = run_benchmark(images=img_dir, output=bench_out)

        assert "dataset" in comparison
        assert "colmap" in comparison
        assert "fastmap" in comparison
        assert (bench_out / "colmap_result.json").is_file()
        assert (bench_out / "fastmap_result.json").is_file()
        assert (bench_out / "comparison.json").is_file()
