"""Comprehensive unit and integration tests for SIH26158 Step 5: VGGT Integration & 4-Way Benchmark."""

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

from reconstruction.pose_engine.benchmark import benchmark_vggt, run_benchmark
from reconstruction.pose_engine.vggt import (
    VGGTAdapter,
    get_vggt_commit,
    get_vggt_environment_info,
    rotmat_to_qvec,
    run_vggt,
    select_frame_subset,
)
from reconstruction.validation.geometry import compute_camera_center, qvec_to_rotmat


def helper_create_dummy_images(tmpdir: Path, num_images: int = 5) -> Path:
    """Creates dummy JPEG image files for testing."""
    img_dir = tmpdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # Minimal 1x1 valid JPEG bytes
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


def test_vggt_environment_detection():
    """Environment inspection must accurately detect repository, commit, python, and CUDA status."""
    env = get_vggt_environment_info()
    assert "repository" in env
    assert Path(env["repo_path"]).name == "vggt"
    assert "commit" in env
    assert "license" in env
    assert "Meta VGGT License" in env["license"]
    assert "default_checkpoint" in env
    assert env["default_checkpoint"] == "facebook/VGGT-1B"

    # On Windows without CUDA, must report supported = False with specific blockers
    if sys.platform == "win32" and not env["cuda_available"]:
        assert env["supported"] is False
        assert any("CUDA" in b for b in env["blockers"])


def test_vggt_commit_detection():
    """Commit hash of tools/vggt should be detected as a 40-char git SHA."""
    commit = get_vggt_commit()
    assert commit is not None
    assert len(commit) == 40


def test_rotmat_to_qvec_conversion():
    """Shepperd rotation matrix to quaternion conversion must match qvec_to_rotmat."""
    test_quats = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.70710678, 0.70710678, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.9238795, 0.0, 0.3826834, 0.0]),
    ]
    for q_orig in test_quats:
        q_orig = q_orig / np.linalg.norm(q_orig)
        R = qvec_to_rotmat(q_orig)
        q_recovered = rotmat_to_qvec(R)

        # Check orientation equivalence (q and -q represent identical rotations)
        dot = abs(float(np.dot(q_orig, q_recovered)))
        assert math.isclose(dot, 1.0, abs_tol=1e-5)
        # Canonicalization check: qw must be non-negative
        assert q_recovered[0] >= 0.0


def test_select_frame_subset():
    """Frame selection must preserve chronological order, start/end bounds, and index mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        files = [tmp_path / f"img_{i:04d}.jpg" for i in range(20)]
        for f in files:
            f.touch()

        # Selection with limit 5
        selected, mapping = select_frame_subset(files, max_images=5)
        assert len(selected) == 5
        assert selected[0].name == "img_0000.jpg"
        assert selected[-1].name == "img_0019.jpg"

        # Verify mapping contents
        assert len(mapping) == 5
        assert mapping[0]["vggt_index"] == 0
        assert mapping[0]["image_name"] == "img_0000.jpg"
        assert mapping[0]["frame_id"] == "img_0000"


def test_vggt_adapter_normalize_poses():
    """VGGTAdapter.normalize_poses converts extrinsics into project standard camera representation."""
    extrinsics = np.zeros((3, 3, 4), dtype=np.float64)
    for i in range(3):
        extrinsics[i, :3, :3] = np.eye(3)
        extrinsics[i, :3, 3] = np.array([-float(i) * 1.0, 0.0, 0.0])

    mapping = {
        0: {"image_name": "frame_000000.jpg", "frame_id": "frame_000000"},
        1: {"image_name": "frame_000001.jpg", "frame_id": "frame_000001"},
        2: {"image_name": "frame_000002.jpg", "frame_id": "frame_000002"},
    }

    normalized = VGGTAdapter.normalize_poses(extrinsics, mapping)
    assert len(normalized) == 3
    assert normalized[0]["frame_id"] == "frame_000000"
    assert normalized[0]["position"] == [0.0, 0.0, 0.0]
    assert normalized[1]["position"] == [1.0, 0.0, 0.0]
    assert normalized[2]["position"] == [2.0, 0.0, 0.0]
    assert normalized[0]["rotation_quaternion"] == [1.0, 0.0, 0.0, 0.0]


def test_vggt_adapter_colmap_export_and_validation():
    """VGGTAdapter.export_colmap_format produces valid COLMAP files and ingests into validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sparse_dir = tmp_path / "sparse"

        extrinsics = np.zeros((3, 3, 4), dtype=np.float64)
        intrinsics = np.zeros((3, 3, 3), dtype=np.float64)
        for i in range(3):
            extrinsics[i, :3, :3] = np.eye(3)
            extrinsics[i, :3, 3] = np.array([-float(i) * 0.5, 0.0, 0.0])
            intrinsics[i] = np.array([[500.0, 0.0, 259.0], [0.0, 500.0, 259.0], [0.0, 0.0, 1.0]])

        mapping = {
            i: {"image_name": f"frame_{i:06d}.jpg", "frame_id": f"frame_{i:06d}"}
            for i in range(3)
        }
        points_3d = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]], dtype=np.float64)

        exported_dir = VGGTAdapter.export_colmap_format(
            sparse_dir=sparse_dir,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            mapping=mapping,
            points_3d=points_3d,
        )

        assert (exported_dir / "cameras.txt").is_file()
        assert (exported_dir / "images.txt").is_file()
        assert (exported_dir / "points3D.txt").is_file()

        # Run validation
        val_report_path = tmp_path / "pose_report.json"
        traj_report_path = tmp_path / "trajectory.json"
        val = VGGTAdapter.validate(
            model_dir=exported_dir,
            output_report_path=val_report_path,
            trajectory_output_path=traj_report_path,
        )

        assert val_report_path.is_file()
        assert traj_report_path.is_file()
        assert "trajectory" in val
        assert val["trajectory"]["status"] == "AVAILABLE"
        assert math.isclose(val["trajectory"]["mean_step"], 0.5, abs_tol=1e-3)


def test_run_vggt_invalid_input_dir():
    """Nonexistent image directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_vggt(
            images="nonexistent_images_folder_abc",
            output="reconstruction/test_output/vggt",
        )


def test_run_vggt_empty_input_dir():
    """Empty image directory must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No supported images found"):
            run_vggt(
                images=tmpdir,
                output="reconstruction/test_output/vggt",
            )


def test_run_vggt_blocked_on_incompatible_env():
    """When CUDA is not available, run_vggt records BLOCKED status without throwing unhandled exceptions."""
    env = get_vggt_environment_info()
    if not env["supported"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_dir = helper_create_dummy_images(tmp_path, num_images=3)
            out_dir = tmp_path / "vggt_out"

            res = run_vggt(images=img_dir, output=out_dir, mock_mode=False)
            assert res["status"] == "BLOCKED"
            assert "blocker_reason" in res
            assert (out_dir / "vggt_report.json").is_file()

            with open(out_dir / "vggt_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            assert report["status"] == "BLOCKED"
            assert report["reprojection"]["status"] == "NOT_AVAILABLE"


def test_run_vggt_mock_mode():
    """In mock mode, run_vggt performs end-to-end normalization, export, and validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=4)
        out_dir = tmp_path / "vggt_out"

        res = run_vggt(images=img_dir, output=out_dir, mock_mode=True, max_images=3)
        assert res["status"] == "PASS"
        assert res["processed_images"] == 3
        assert res["camera_predictions"] == 3
        assert (out_dir / "vggt_report.json").is_file()
        assert (out_dir / "cameras.json").is_file()
        assert (out_dir / "metadata.json").is_file()
        assert (out_dir / "index_mapping.json").is_file()
        assert (out_dir / "predictions" / "predictions.npz").is_file()
        assert (out_dir / "sparse" / "cameras.txt").is_file()


def test_run_4way_benchmark():
    """4-way benchmark must evaluate COLMAP, FastMap, Global SfM, and VGGT, exporting all 5 JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = helper_create_dummy_images(tmp_path, num_images=3)
        bench_out = tmp_path / "benchmark"

        comparison = run_benchmark(
            images=img_dir,
            output=bench_out,
            mock_vggt=True,
        )

        assert "experiment" in comparison
        assert "colmap_incremental" in comparison
        assert "fastmap" in comparison
        assert "colmap_global_sfm" in comparison
        assert "vggt" in comparison

        assert (bench_out / "colmap_result.json").is_file()
        assert (bench_out / "fastmap_result.json").is_file()
        assert (bench_out / "global_sfm_result.json").is_file()
        assert (bench_out / "vggt_result.json").is_file()
        assert (bench_out / "comparison.json").is_file()

        # Check vggt result structure
        vggt_res = comparison["vggt"]
        assert vggt_res["backend"] == "VGGT"
        assert vggt_res["status"] == "PASS"
        assert vggt_res["registered_images"] == 3
