"""Unit and integration tests for SIH26158 Step 1: COLMAP baseline and environment setup."""

import json
import struct
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.check_environment import check_environment
from reconstruction.colmap.utils import (
    find_colmap_binary,
    get_colmap_version,
    is_cuda_available,
)
from reconstruction.colmap.feature_extraction import validate_image_dir
from reconstruction.colmap.feature_matching import match_features
from reconstruction.validation.pose_report import (
    count_input_images,
    find_sparse_model_dir,
    generate_reconstruction_report,
    inspect_sparse_model,
    inspect_sparse_model_fallback,
)


def test_environment_check_runs():
    """Environment check should complete with exit code 0 when core tools are present."""
    code = check_environment(require_cuda=False)
    assert code == 0


def test_find_colmap_binary():
    """COLMAP binary should be found on the system or in tools/colmap/bin."""
    binary = find_colmap_binary()
    assert binary is not None
    assert binary.is_file()

    version = get_colmap_version(binary)
    assert "COLMAP" in version


def test_validate_image_dir_nonexistent():
    """Validating a non-existent image directory must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        validate_image_dir(Path("nonexistent_directory_xyz_123"))


def test_validate_image_dir_empty():
    """Validating an empty image directory must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No supported images found"):
            validate_image_dir(Path(tmpdir))


def test_validate_image_dir_with_images():
    """Validating an image directory with image files should find and count them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "img1.jpg").write_bytes(b"dummy")
        (tmp_path / "img2.PNG").write_bytes(b"dummy")
        (tmp_path / "ignore.txt").write_bytes(b"ignore")

        images = validate_image_dir(tmp_path)
        assert len(images) == 2
        assert count_input_images(tmp_path) == 2


def test_feature_matching_invalid_matcher():
    """Invalid matcher string should raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmpdb:
        with pytest.raises(ValueError, match="Unsupported matcher"):
            match_features(database_path=tmpdb.name, matcher="invalid_matcher_name")


def test_feature_matching_missing_database():
    """Missing database file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        match_features(database_path="nonexistent_database.db", matcher="sequential")


def test_sparse_model_inspector_fallback_binary():
    """Inspect sparse model fallback should correctly parse COLMAP binary headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "0"
        model_dir.mkdir()

        # Write mock binary files with 64-bit counts
        num_cams = 2
        num_reg_imgs = 15
        num_pts = 350

        with open(model_dir / "cameras.bin", "wb") as f:
            f.write(struct.pack("<Q", num_cams))
        with open(model_dir / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", num_reg_imgs))
        with open(model_dir / "points3D.bin", "wb") as f:
            f.write(struct.pack("<Q", num_pts))

        found_dir = find_sparse_model_dir(Path(tmpdir))
        assert found_dir == model_dir

        stats = inspect_sparse_model_fallback(model_dir)
        assert stats["exists"] is True
        assert stats["num_cameras"] == num_cams
        assert stats["registered_images"] == num_reg_imgs
        assert stats["num_points3D"] == num_pts


def test_sparse_model_inspector_fallback_text():
    """Inspect sparse model fallback should correctly parse COLMAP text headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        # Text format: cameras.txt, images.txt, points3D.txt
        with open(model_dir / "cameras.txt", "w", encoding="utf-8") as f:
            f.write("# Camera list\n1 SIMPLE_RADIAL 1000 1000 500 500 0.1\n")

        with open(model_dir / "images.txt", "w", encoding="utf-8") as f:
            # 2 lines per image
            f.write("# Image list\n1 0 0 0 1 0 0 0 1 img1.jpg\n0 0 -1\n2 0 0 0 1 0 0 0 1 img2.jpg\n0 0 -1\n")

        with open(model_dir / "points3D.txt", "w", encoding="utf-8") as f:
            f.write("# 3D point list\n1 0 0 0 255 255 255 0.1 1 0\n2 1 1 1 255 255 255 0.1 1 1\n3 2 2 2 255 255 255 0.1 1 2\n")

        stats = inspect_sparse_model_fallback(model_dir)
        assert stats["exists"] is True
        assert stats["num_cameras"] == 1
        assert stats["registered_images"] == 2
        assert stats["num_points3D"] == 3


def test_generate_reconstruction_report():
    """Report generator should produce a compliant JSON structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(10):
            (img_dir / f"img_{i:03d}.jpg").write_bytes(b"dummy")

        sparse_dir = tmp_path / "sparse" / "0"
        sparse_dir.mkdir(parents=True)
        with open(sparse_dir / "cameras.bin", "wb") as f:
            f.write(struct.pack("<Q", 1))
        with open(sparse_dir / "images.bin", "wb") as f:
            f.write(struct.pack("<Q", 8))
        with open(sparse_dir / "points3D.bin", "wb") as f:
            f.write(struct.pack("<Q", 1200))

        report_file = tmp_path / "reconstruction_report.json"
        report = generate_reconstruction_report(
            image_dir=img_dir,
            sparse_dir=tmp_path / "sparse",
            output_report_path=report_file,
            colmap_version="COLMAP 4.2.0-test",
            processing_time_seconds=12.5,
        )

        assert report["reconstruction_success"] is True
        assert report["input_images"] == 10
        assert report["registered_images"] == 8
        assert report["registration_rate"] == 0.8
        assert report["num_cameras"] == 1
        assert report["num_points3D"] == 1200
        assert report["processing_time_seconds"] == 12.5
        assert report["colmap_version"] == "COLMAP 4.2.0-test"
        assert report["errors"] == []

        # Check saved JSON on disk
        assert report_file.is_file()
        with open(report_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == report
