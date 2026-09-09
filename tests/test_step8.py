"""Comprehensive unit and integration tests for SIH26158 Step 8: Point-Cloud Processing.

Tests:
1. Loading:
   - Valid COLMAP points3D text
   - Valid COLMAP points3D binary
   - Empty point cloud rejection
   - Malformed data handling
   - Non-finite coordinates rejection (no NaN/Inf)
   - RGB color preservation
2. Statistical Outlier Removal:
   - Synthetic outlier removal
   - Parameter behavior (varying nb_neighbors and std_ratio)
3. Radius Outlier Removal:
   - Isolated-point removal
   - Sparse cloud handling
4. Voxel Downsampling:
   - Deterministic point reduction
   - Coordinate unit handling (metres vs local unscaled)
5. Processing Pipeline & Statistics:
   - All stages enabled
   - Individual stages disabled
   - Section 11 statistics schema verification
   - Section 14 metadata schema verification
   - Retention rate calculations and warnings
6. Export:
   - Standard PLY export and read-back
   - RGB channel preservation in PLY
   - Complete bundle export (PLY, metadata.json, processing_report.json)
7. Failure & Boundary Handling:
   - Nonexistent input path
   - Empty file
   - Invalid filter parameters (negative values)
8. Pipeline Integration:
   - End-to-end integration with adaptive pose engine and georeferencing transform
   - CRS preservation in UTM metres
   - Local unscaled fallback with crs=None
9. Real-Data Benchmark Integrity:
   - Ensures all synthetic test artifacts are labeled as synthetic software tests
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.pointcloud.export import (
    PointCloudData,
    apply_georeferencing_transform,
    export_point_cloud_bundle,
    export_point_cloud_ply,
    load_colmap_points3d_bin,
    load_colmap_points3d_txt,
    load_point_cloud,
)
from reconstruction.pointcloud.processing import (
    downsample_voxels,
    filter_ground_extension,
    filter_radius_outliers,
    filter_statistical_outliers,
    run_point_cloud_pipeline,
    validate_point_cloud,
)
from reconstruction.pose_engine.engine import (
    PoseResult,
    run_adaptive_engine,
)


# ===========================================================================
# Helpers for Synthetic Point Clouds
# ===========================================================================

def create_synthetic_points3d_txt(
    file_path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    errors: Optional[np.ndarray] = None,
) -> Path:
    """Creates a valid COLMAP points3D.txt file with synthetic test data."""
    n = len(points)
    cols = colors if colors is not None else np.full((n, 3), 128, dtype=np.uint8)
    errs = errors if errors is not None else np.zeros(n, dtype=np.float64)

    lines = [
        "# 3D point list with one line of data per point (SYNTHETIC SOFTWARE TEST):",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for i in range(n):
        pid = i + 1
        x, y, z = points[i]
        r, g, b = cols[i]
        err = errs[i]
        # Dummy track with 2 observations
        track_str = f"1 {i} 2 {i}"
        lines.append(f"{pid} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {err:.4f} {track_str}")

    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_path


def create_synthetic_points3d_bin(
    file_path: Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    errors: Optional[np.ndarray] = None,
) -> Path:
    """Creates a valid COLMAP points3D.bin file with synthetic test data."""
    n = len(points)
    cols = colors if colors is not None else np.full((n, 3), 128, dtype=np.uint8)
    errs = errors if errors is not None else np.zeros(n, dtype=np.float64)

    with open(file_path, "wb") as f:
        # num_points uint64
        f.write(struct.pack("<Q", n))
        for i in range(n):
            pid = i + 1
            x, y, z = points[i]
            r, g, b = cols[i]
            err = errs[i]
            # point3D_id (uint64), xyz (3d), rgb (3B), error (1d)
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<3d", x, y, z))
            f.write(struct.pack("<3B", int(r), int(g), int(b)))
            f.write(struct.pack("<d", err))
            # track_len uint64: 2 observations (img_id uint32, p2d_idx uint32)
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<II", 1, i))
            f.write(struct.pack("<II", 2, i))

    return file_path


# ===========================================================================
# 1. Loading Tests
# ===========================================================================

def test_load_valid_colmap_points3d_txt(tmp_path: Path) -> None:
    """Tests loading a well-formed COLMAP points3D.txt file."""
    txt_file = tmp_path / "points3D.txt"
    pts = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    cols = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ], dtype=np.uint8)

    create_synthetic_points3d_txt(txt_file, pts, colors=cols)

    pcd = load_colmap_points3d_txt(txt_file)
    assert pcd.num_points == 3
    assert np.allclose(pcd.points, pts)
    assert np.array_equal(pcd.colors, cols)
    assert pcd.has_colors is True
    assert pcd.crs is None
    assert pcd.units == "local_unscaled"
    assert pcd.is_georeferenced is False


def test_load_valid_colmap_points3d_bin(tmp_path: Path) -> None:
    """Tests loading a well-formed COLMAP points3D.bin file."""
    bin_file = tmp_path / "points3D.bin"
    pts = np.array([
        [10.5, 20.5, 30.5],
        [40.5, 50.5, 60.5],
    ], dtype=np.float64)
    cols = np.array([
        [100, 150, 200],
        [50, 60, 70],
    ], dtype=np.uint8)

    create_synthetic_points3d_bin(bin_file, pts, colors=cols)

    pcd = load_colmap_points3d_bin(bin_file)
    assert pcd.num_points == 2
    assert np.allclose(pcd.points, pts)
    assert np.array_equal(pcd.colors, cols)
    assert pcd.is_georeferenced is False


def test_load_empty_point_cloud_raises_error(tmp_path: Path) -> None:
    """Tests that an empty points3D file raises a clear ValueError."""
    empty_txt = tmp_path / "empty_points3D.txt"
    empty_txt.write_text("# Empty COLMAP point cloud\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_colmap_points3d_txt(empty_txt)


def test_load_malformed_point_data_raises_error(tmp_path: Path) -> None:
    """Tests that non-numeric or truncated point lines raise ValueError."""
    bad_txt = tmp_path / "bad_points3D.txt"
    bad_txt.write_text("1 2.0 3.0 not_a_number 255 255 255 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Non-numeric"):
        load_colmap_points3d_txt(bad_txt)


def test_load_non_finite_coordinates_raises_error(tmp_path: Path) -> None:
    """Tests that NaN or Inf coordinates are rejected without silent fabrication."""
    nan_txt = tmp_path / "nan_points3D.txt"
    nan_txt.write_text("1 nan 2.0 3.0 255 255 255 0.5 1 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Non-finite XYZ"):
        load_colmap_points3d_txt(nan_txt)

    inf_txt = tmp_path / "inf_points3D.txt"
    inf_txt.write_text("1 1.0 inf 3.0 255 255 255 0.5 1 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Non-finite XYZ"):
        load_colmap_points3d_txt(inf_txt)


def test_load_rgb_preservation(tmp_path: Path) -> None:
    """Tests that RGB color attributes are preserved faithfully."""
    txt_file = tmp_path / "rgb_points3D.txt"
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    cols = np.array([[12, 34, 56], [78, 90, 123]], dtype=np.uint8)

    create_synthetic_points3d_txt(txt_file, pts, cols)
    pcd = load_colmap_points3d_txt(txt_file)

    assert pcd.colors[0, 0] == 12
    assert pcd.colors[0, 1] == 34
    assert pcd.colors[0, 2] == 56
    assert pcd.colors[1, 0] == 78
    assert pcd.colors[1, 1] == 90
    assert pcd.colors[1, 2] == 123


# ===========================================================================
# 2. Quality Validation Tests
# ===========================================================================

def test_validate_point_cloud_metrics() -> None:
    """Tests quality validation statistics calculation."""
    pts = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 10.0],
    ], dtype=np.float64)
    cols = np.zeros((4, 3), dtype=np.uint8)

    pcd = PointCloudData(points=pts, colors=cols)
    metrics = validate_point_cloud(pcd)

    assert metrics["status"] == "PASS"
    assert metrics["num_points"] == 4
    assert metrics["unique_points"] == 4
    assert metrics["duplicate_points"] == 0
    assert metrics["xyz_min"] == [0.0, 0.0, 0.0]
    assert metrics["xyz_max"] == [10.0, 10.0, 10.0]
    assert metrics["extents"] == [10.0, 10.0, 10.0]
    assert metrics["bounding_box_volume"] == 1000.0


def test_validate_point_cloud_duplicate_detection() -> None:
    """Tests detection and reporting of duplicate coordinate points."""
    pts = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],  # Duplicate
        [2.0, 2.0, 2.0],
    ], dtype=np.float64)

    pcd = PointCloudData(points=pts)
    metrics = validate_point_cloud(pcd)

    assert metrics["duplicate_points"] == 1
    assert metrics["unique_points"] == 2
    assert any("duplicate" in w.lower() for w in metrics["warnings"])


# ===========================================================================
# 3. Statistical Outlier Removal Tests
# ===========================================================================

def test_statistical_outlier_removal_synthetic_outlier() -> None:
    """Tests that isolated statistical outliers are removed from a dense cluster."""
    np.random.seed(42)
    # Dense cluster of 100 points around origin
    cluster = np.random.normal(loc=0.0, scale=0.2, size=(100, 3))
    # 2 distant outliers
    outliers = np.array([[50.0, 50.0, 50.0], [-50.0, -50.0, -50.0]], dtype=np.float64)
    all_pts = np.vstack([cluster, outliers])

    pcd = PointCloudData(points=all_pts)
    filtered, report = filter_statistical_outliers(
        pcd,
        nb_neighbors=16,
        std_ratio=1.5,
        enabled=True,
    )

    assert report["status"] == "COMPLETED"
    assert report["removed_points"] >= 2
    # Verify that the distant outliers were removed
    max_dist = np.max(np.linalg.norm(filtered.points, axis=1))
    assert max_dist < 10.0  # Far away points eliminated


def test_statistical_outlier_parameter_behavior() -> None:
    """Tests that a stricter std_ratio removes more or equal points."""
    np.random.seed(123)
    pts = np.random.normal(loc=0.0, scale=1.0, size=(150, 3))
    pcd = PointCloudData(points=pts)

    _, report_loose = filter_statistical_outliers(pcd, nb_neighbors=20, std_ratio=3.0)
    _, report_strict = filter_statistical_outliers(pcd, nb_neighbors=20, std_ratio=1.0)

    assert report_strict["removed_points"] >= report_loose["removed_points"]


# ===========================================================================
# 4. Radius Outlier Removal Tests
# ===========================================================================

def test_radius_outlier_removal_isolated_point() -> None:
    """Tests that points with fewer than min_points in radius are removed."""
    # Cluster of 15 points within radius 0.5
    cluster = np.random.uniform(low=0.0, high=0.3, size=(15, 3))
    # Isolated point at distance 10.0
    isolated = np.array([[10.0, 10.0, 10.0]])
    all_pts = np.vstack([cluster, isolated])

    pcd = PointCloudData(points=all_pts)
    filtered, report = filter_radius_outliers(
        pcd,
        radius=0.5,
        min_points=8,
        enabled=True,
    )

    assert report["status"] == "COMPLETED"
    assert report["removed_points"] >= 1
    # Check that the isolated point was removed
    assert not any(np.allclose(p, [10.0, 10.0, 10.0]) for p in filtered.points)


def test_radius_outlier_removal_disabled_by_default() -> None:
    """Tests that radius filtering is skipped when disabled."""
    pts = np.array([[1.0, 2.0, 3.0], [100.0, 100.0, 100.0]], dtype=np.float64)
    pcd = PointCloudData(points=pts)

    filtered, report = filter_radius_outliers(pcd, radius=0.5, enabled=False)
    assert report["status"] == "SKIPPED"
    assert filtered.num_points == 2


# ===========================================================================
# 5. Voxel Downsampling Tests
# ===========================================================================

def test_voxel_downsampling_deterministic_reduction() -> None:
    """Tests that dense grid points inside the same voxel are merged."""
    # 8 points in a 0.05m cube
    pts = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],
        [0.03, 0.03, 0.03],
        [0.04, 0.04, 0.04],
        [1.01, 1.01, 1.01],
        [1.02, 1.02, 1.02],
        [1.03, 1.03, 1.03],
        [1.04, 1.04, 1.04],
    ], dtype=np.float64)
    pcd = PointCloudData(points=pts, units="metres", is_georeferenced=True)

    downsampled, report = downsample_voxels(pcd, voxel_size=0.5, enabled=True)

    assert report["status"] == "COMPLETED"
    assert downsampled.num_points == 2  # 2 distinct voxels
    assert report["coordinate_units"] == "metres"
    assert report["voxel_size"] == 0.5


def test_voxel_downsampling_unscaled_units_warning() -> None:
    """Tests that unscaled local point clouds generate a clear warning on voxel downsampling."""
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0]], dtype=np.float64)
    pcd = PointCloudData(points=pts, units="local_unscaled", is_georeferenced=False)

    _, report = downsample_voxels(pcd, voxel_size=0.2, enabled=True)
    assert report["warning"] is not None
    assert "local unscaled" in report["warning"].lower()


# ===========================================================================
# 6. Pipeline & Statistics Generation Tests
# ===========================================================================

def test_pipeline_all_stages_enabled(tmp_path: Path) -> None:
    """Tests running the entire pipeline with all stages enabled."""
    np.random.seed(99)
    cluster = np.random.normal(loc=0.0, scale=0.5, size=(100, 3))
    outlier = np.array([[30.0, 30.0, 30.0]])
    pts = np.vstack([cluster, outlier])
    cols = np.full((101, 3), 200, dtype=np.uint8)

    pcd = PointCloudData(points=pts, colors=cols, units="metres", is_georeferenced=True, crs="EPSG:32643")
    out_dir = tmp_path / "pcd_out"

    cleaned, stats = run_point_cloud_pipeline(
        input_cloud=pcd,
        output_dir=out_dir,
        nb_neighbors=15,
        std_ratio=2.0,
        enable_statistical=True,
        radius=1.5,
        min_points=5,
        enable_radius=True,
        voxel_size=0.2,
        enable_voxel=True,
    )

    # Verify Section 11 required stats keys
    assert "input_points" in stats
    assert "after_statistical_filter" in stats
    assert "after_radius_filter" in stats
    assert "after_voxel_downsample" in stats
    assert "final_points" in stats
    assert "removed_points" in stats
    assert "retention_rate" in stats

    assert stats["input_points"] == 101
    assert stats["final_points"] == cleaned.num_points
    assert stats["removed_points"] == 101 - cleaned.num_points
    assert 0.0 < stats["retention_rate"] <= 1.0

    # Verify Section 14 metadata schema in output directory
    meta_path = out_dir / "metadata.json"
    assert meta_path.is_file()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "source" in meta
    assert "coordinate_system" in meta
    assert meta["coordinate_system"]["crs"] == "EPSG:32643"
    assert meta["coordinate_system"]["units"] == "metres"
    assert meta["processing"]["statistical_outlier_removal"] is True
    assert meta["processing"]["radius_outlier_removal"] is True
    assert meta["processing"]["voxel_downsampling"] is True
    assert meta["point_counts"]["input"] == 101
    assert meta["point_counts"]["output"] == cleaned.num_points


def test_pipeline_individual_stages_disabled() -> None:
    """Tests that individual stages can be selectively disabled."""
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    pcd = PointCloudData(points=pts)

    cleaned, stats = run_point_cloud_pipeline(
        input_cloud=pcd,
        enable_statistical=False,
        enable_radius=False,
        enable_voxel=False,
    )

    assert stats["after_statistical_filter"] == 3
    assert stats["after_radius_filter"] == 3
    assert stats["after_voxel_downsample"] == 3
    assert stats["final_points"] == 3
    assert stats["removed_points"] == 0
    assert stats["retention_rate"] == 1.0


# ===========================================================================
# 7. Export Tests
# ===========================================================================

def test_export_valid_ply_and_read_back(tmp_path: Path) -> None:
    """Tests that exported PLY is standard, valid, and preserves coordinates and RGB."""
    import open3d as o3d

    pts = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float64)
    cols = np.array([[255, 128, 0], [0, 64, 128]], dtype=np.uint8)
    pcd = PointCloudData(points=pts, colors=cols)

    ply_path = tmp_path / "test.ply"
    export_point_cloud_ply(pcd, ply_path)

    assert ply_path.is_file()
    assert ply_path.stat().st_size > 0

    # Read back with Open3D to verify format standard compliance
    read_pcd = o3d.io.read_point_cloud(str(ply_path))
    assert len(read_pcd.points) == 2
    read_pts = np.asarray(read_pcd.points)
    assert np.allclose(read_pts, pts, atol=1e-5)

    read_cols = (np.asarray(read_pcd.colors) * 255.0).astype(np.uint8)
    assert np.allclose(read_cols, cols, atol=2)


def test_export_point_cloud_bundle_files(tmp_path: Path) -> None:
    """Tests that export_point_cloud_bundle creates all 3 standard Step 8 files."""
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    pcd = PointCloudData(points=pts, crs=None, units="local_unscaled")
    out_dir = tmp_path / "bundle_out"

    stats = {
        "input_points": 2,
        "final_points": 2,
        "parameters": {
            "statistical": {"enabled": True},
            "radius": {"enabled": False},
            "voxel": {"enabled": False},
        },
    }

    bundle = export_point_cloud_bundle(pcd, out_dir, stats)
    assert bundle["ply"].is_file()
    assert bundle["metadata"].is_file()
    assert bundle["processing_report"].is_file()

    with open(bundle["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["coordinate_system"]["crs"] is None
    assert meta["coordinate_system"]["units"] == "local_unscaled"


# ===========================================================================
# 8. Failure & Boundary Handling Tests
# ===========================================================================

def test_failure_nonexistent_input() -> None:
    """Tests that nonexistent input paths raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_point_cloud(Path("nonexistent_directory/sparse/points3D.txt"))


def test_failure_invalid_processing_parameters() -> None:
    """Tests that invalid parameter values raise clear ValueErrors."""
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    pcd = PointCloudData(points=pts)

    with pytest.raises(ValueError, match="nb_neighbors"):
        filter_statistical_outliers(pcd, nb_neighbors=-5)

    with pytest.raises(ValueError, match="std_ratio"):
        filter_statistical_outliers(pcd, std_ratio=-1.0)

    with pytest.raises(ValueError, match="radius"):
        filter_radius_outliers(pcd, radius=-0.5, enabled=True)

    with pytest.raises(ValueError, match="voxel_size"):
        downsample_voxels(pcd, voxel_size=-0.1, enabled=True)


# ===========================================================================
# 9. Georeferencing Transformation Tests
# ===========================================================================

def test_apply_georeferencing_transform() -> None:
    """Tests applying similarity transform (s, R, t) to point cloud coordinates."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    pcd = PointCloudData(points=pts, crs=None, units="local_unscaled")

    # Georeference metadata with scale 2.0, identity rotation, translation [100, 200, 300]
    geo_meta = {
        "target_crs": "EPSG:32643",
        "coordinate_system": "WGS 84 / UTM zone 43N",
        "alignment": {
            "scale": 2.0,
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "translation_vector": [100.0, 200.0, 300.0],
        },
    }

    geo_pcd = apply_georeferencing_transform(pcd, geo_meta)
    assert geo_pcd.is_georeferenced is True
    assert geo_pcd.crs == "EPSG:32643"
    assert geo_pcd.units == "metres"

    expected_pts = np.array([
        [100.0, 200.0, 300.0],
        [102.0, 200.0, 300.0],
    ])
    assert np.allclose(geo_pcd.points, expected_pts)


# ===========================================================================
# 10. Pipeline Integration Tests
# ===========================================================================

def test_adaptive_engine_with_point_cloud_processing(tmp_path: Path) -> None:
    """Tests full pipeline integration: Pose Engine -> Georeferencing -> Point-Cloud Processing."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    # Minimal JPEG bytes for pre-flight check
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
    for i in range(25):
        (img_dir / f"frame_{i:04d}.jpg").write_bytes(jpeg_bytes)

    out_dir = tmp_path / "engine_out"
    mock_sparse_dir = tmp_path / "mock_sparse"
    mock_sparse_dir.mkdir(parents=True)

    # Create synthetic sparse points3D.txt
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.5],
    ], dtype=np.float64)
    create_synthetic_points3d_txt(mock_sparse_dir / "points3D.txt", pts)

    # Synthetic cameras
    cams = [
        {"frame_id": "f0", "timestamp": 100.0, "position": [0.0, 0.0, 0.0]},
        {"frame_id": "f1", "timestamp": 101.0, "position": [10.0, 0.0, 0.0]},
        {"frame_id": "f2", "timestamp": 102.0, "position": [0.0, 10.0, 0.0]},
        {"frame_id": "f3", "timestamp": 103.0, "position": [10.0, 10.0, 5.0]},
    ]
    cams_file = tmp_path / "cams.json"
    cams_file.write_text(json.dumps(cams), encoding="utf-8")

    # Synthetic GPS
    gps_file = tmp_path / "gps.csv"
    gps_content = (
        "timestamp,latitude,longitude,altitude\n"
        "100.0,12.971500,77.594500,920.0\n"
        "101.0,12.971590,77.594500,920.0\n"
        "102.0,12.971500,77.594590,920.0\n"
        "103.0,12.971590,77.594590,925.0\n"
    )
    gps_file.write_text(gps_content, encoding="utf-8")

    def mock_backend() -> PoseResult:
        return PoseResult(
            method="FastMap",
            status="PASS",
            output_path=tmp_path,
            runtime_seconds=1.0,
            validation_status="PASS",
            registration_rate=0.92,
            registered_images=23,
            input_images=25,
            reprojection_rmse_px=0.6,
            reprojection_status="AVAILABLE",
            trajectory_status="PASS",
            cameras_path=cams_file,
            sparse_dir=mock_sparse_dir,
        )

    report = run_adaptive_engine(
        images=img_dir,
        output=out_dir,
        method="fastmap",
        gps_path=gps_file,
        mock_backends={"fastmap": mock_backend},
    )

    assert report["overall_status"] == "PASS"
    assert report["georeferencing"] is not None
    assert report["pointcloud"] is not None

    pcd_summary = report["pointcloud"]
    assert pcd_summary["final_points"] == 4
    assert pcd_summary["coordinate_units"] == "metres"
    assert pcd_summary["is_georeferenced"] is True

    # Check exported bundle files on disk
    pcd_dir = out_dir / "selected" / "pointcloud"
    assert (pcd_dir / "pointcloud.ply").is_file()
    assert (pcd_dir / "metadata.json").is_file()
    assert (pcd_dir / "processing_report.json").is_file()

    with open(pcd_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["coordinate_system"]["units"] == "metres"
    assert meta["coordinate_system"]["crs"] is not None
