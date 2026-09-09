"""Comprehensive unit and integration tests for SIH26158 Step 4: COLMAP Global SfM & 3-Way Benchmark."""

import json
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.pose_engine.global_sfm import (
    is_global_mapper_available,
    is_view_graph_calibrator_available,
    run_global_sfm,
)
from reconstruction.pose_engine.benchmark import run_benchmark


def test_global_mapper_available():
    """Verify that the installed COLMAP binary supports the native global_mapper command."""
    available = is_global_mapper_available()
    assert available is True


def test_view_graph_calibrator_available():
    """Verify that view_graph_calibrator is supported in the installed COLMAP version."""
    available = is_view_graph_calibrator_available()
    assert available is True


def test_run_global_sfm_invalid_input_dir():
    """Non-existent image directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_global_sfm(
            images="nonexistent_images_folder_xyz",
            output="reconstruction/test_output/global_sfm",
        )


def test_run_global_sfm_empty_input_dir():
    """Empty image directory must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No supported images found"):
            run_global_sfm(
                images=tmpdir,
                output="reconstruction/test_output/global_sfm",
            )


def test_run_3way_benchmark():
    """3-way benchmark must evaluate COLMAP Incr, FastMap, and Global SfM, generating all 4 JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(2):
            (img_dir / f"frame_{i:04d}.jpg").write_bytes(b"dummy")

        bench_out = tmp_path / "benchmark"
        comparison = run_benchmark(images=img_dir, output=bench_out)

        assert "experiment" in comparison
        assert "colmap_incremental" in comparison
        assert "fastmap" in comparison
        assert "colmap_global_sfm" in comparison

        assert (bench_out / "colmap_result.json").is_file()
        assert (bench_out / "fastmap_result.json").is_file()
        assert (bench_out / "global_sfm_result.json").is_file()
        assert (bench_out / "comparison.json").is_file()
