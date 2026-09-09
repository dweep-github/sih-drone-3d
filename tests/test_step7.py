"""Tests for Step 7: Georeferencing + Validation Policy Revision.

Covers:
- GPS CSV parsing, validation, error handling, duplicate and missing value detection
- Timestamp synchronization (exact, nearest, tolerance rejection, unmatched reporting)
- Coordinate transformations via pyproj (WGS84 <-> UTM, RFC 7946 ordering)
- Umeyama similarity alignment (known ground truth recovery, degeneracy rejection, RMSE)
- GeoJSON FeatureCollection generation and coordinate conventions
- Configurable provisional engineering ValidationPolicy and serialization
- Adaptive engine pipeline integration and clean failure handling
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.georeferencing.alignment import (
    SimilarityTransform,
    align_umeyama,
    check_point_set_validity,
    transform_camera_pose,
)
from reconstruction.georeferencing.coordinate_transform import (
    CoordinateTransformer,
    determine_utm_crs,
)
from reconstruction.georeferencing.georeference import (
    run_georeferencing,
)
from reconstruction.georeferencing.gps import (
    GPSRecord,
    extract_timestamp_from_frame,
    load_gps_csv,
    parse_timestamp,
    synchronize_poses_with_gps,
)
from reconstruction.pose_engine.engine import (
    PoseResult,
    ValidationPolicy,
    run_adaptive_engine,
)


# ===========================================================================
# 1. GPS Loader & Validator Tests
# ===========================================================================

def test_gps_valid_csv(tmp_path: Path) -> None:
    """Tests loading a well-formed GPS CSV with synthetic drone flight data."""
    csv_file = tmp_path / "valid_gps.csv"
    # Synthetic flight records
    content = (
        "timestamp,latitude,longitude,altitude\n"
        "1725700000.0,12.971598,77.594562,920.5\n"
        "1725700001.0,12.971610,77.594580,921.0\n"
        "1725700002.0,12.971625,77.594600,921.4\n"
    )
    csv_file.write_text(content, encoding="utf-8")

    records, meta = load_gps_csv(csv_file)
    assert len(records) == 3
    assert meta["total_valid_records"] == 3
    assert meta["duplicate_timestamps"] == 0
    assert meta["altitude_reference"] == "WGS84_ellipsoidal"
    assert records[0].timestamp == 1725700000.0
    assert records[0].latitude == 12.971598
    assert records[0].longitude == 77.594562
    assert records[0].altitude == 920.5


def test_gps_missing_columns(tmp_path: Path) -> None:
    """Tests that GPS CSV without required columns raises a descriptive ValueError."""
    csv_file = tmp_path / "missing_col.csv"
    content = (
        "timestamp,latitude,altitude\n"  # Missing longitude
        "1725700000.0,12.971598,920.5\n"
    )
    csv_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column 'longitude'"):
        load_gps_csv(csv_file)


def test_gps_invalid_coordinates(tmp_path: Path) -> None:
    """Tests that out-of-range coordinates are rejected."""
    csv_file = tmp_path / "invalid_coords.csv"
    content = (
        "timestamp,latitude,longitude,altitude\n"
        "1725700000.0,95.0,77.594562,920.5\n"  # Lat > 90
        "1725700001.0,12.971610,-195.0,921.0\n" # Lon < -180
    )
    csv_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="No valid GPS records could be loaded"):
        load_gps_csv(csv_file)


def test_gps_timestamp_parsing() -> None:
    """Tests diverse timestamp representations (numeric, ISO8601 UTC)."""
    assert parse_timestamp(1725700000) == 1725700000.0
    assert parse_timestamp("1725700000.5") == 1725700000.5
    assert parse_timestamp("2026-09-07T12:00:00Z") == 1788782400.0
    with pytest.raises(ValueError):
        parse_timestamp("invalid-date-string")


def test_gps_duplicate_timestamps(tmp_path: Path) -> None:
    """Tests detection and reporting of duplicate timestamps in GPS data."""
    csv_file = tmp_path / "dup_gps.csv"
    content = (
        "timestamp,latitude,longitude,altitude\n"
        "1725700000.0,12.971598,77.594562,920.5\n"
        "1725700000.0,12.971610,77.594580,921.0\n"  # Duplicate
        "1725700001.0,12.971625,77.594600,921.4\n"
    )
    csv_file.write_text(content, encoding="utf-8")

    records, meta = load_gps_csv(csv_file)
    assert meta["duplicate_timestamps"] == 1
    assert len(records) == 3


# ===========================================================================
# 2. Timestamp Synchronization Tests
# ===========================================================================

def test_sync_exact_and_nearest_match() -> None:
    """Tests exact and nearest-neighbor timestamp matching between poses and GPS."""
    gps_records = [
        GPSRecord(timestamp=100.0, latitude=12.0, longitude=77.0, altitude=900.0),
        GPSRecord(timestamp=101.0, latitude=12.1, longitude=77.1, altitude=905.0),
        GPSRecord(timestamp=102.0, latitude=12.2, longitude=77.2, altitude=910.0),
    ]

    poses = [
        {"frame_id": "f0", "timestamp": 100.0, "position": [0.0, 0.0, 0.0]},
        {"frame_id": "f1", "timestamp": 101.1, "position": [1.0, 0.0, 0.0]},  # diff 0.1s
        {"frame_id": "f2", "timestamp": 105.0, "position": [2.0, 0.0, 0.0]},  # diff 3.0s (out of bounds)
    ]

    matched, summary = synchronize_poses_with_gps(
        poses=poses,
        gps_records=gps_records,
        max_time_diff_s=0.5,
    )

    assert len(matched) == 2
    assert summary["matched_observations"] == 2
    assert summary["unmatched_poses_count"] == 1
    assert summary["unmatched_poses"][0]["frame_id"] == "f2"
    assert matched[0]["time_difference_s"] == 0.0
    assert matched[1]["time_difference_s"] == 0.1


def test_sync_tolerance_rejection() -> None:
    """Tests that timestamps exceeding tolerance are rejected without false matches."""
    gps_records = [
        GPSRecord(timestamp=100.0, latitude=12.0, longitude=77.0, altitude=900.0),
    ]
    poses = [
        {"frame_id": "f0", "timestamp": 102.0, "position": [0.0, 0.0, 0.0]},
    ]
    matched, summary = synchronize_poses_with_gps(
        poses=poses,
        gps_records=gps_records,
        max_time_diff_s=1.0,
    )
    assert len(matched) == 0
    assert summary["unmatched_poses_count"] == 1


# ===========================================================================
# 3. Coordinate Transformation Tests
# ===========================================================================

def test_determine_utm_crs() -> None:
    """Tests automatic UTM zone detection for northern and southern hemispheres."""
    # Bengaluru, India (77.59 E, 12.97 N) -> UTM Zone 43N (EPSG:32643)
    assert determine_utm_crs(77.59, 12.97) == "EPSG:32643"
    # Sydney, Australia (151.2 E, -33.8 S) -> UTM Zone 56S (EPSG:32756)
    assert determine_utm_crs(151.2, -33.8) == "EPSG:32756"


def test_wgs84_to_utm_and_inverse() -> None:
    """Tests bidirectional transformation between WGS84 and UTM with sub-millimeter invertibility."""
    transformer = CoordinateTransformer(
        source_crs="EPSG:4326",
        reference_point=(77.5946, 12.9716),
        altitude_reference="WGS84_ellipsoidal",
    )
    assert transformer.target_crs == "EPSG:32643"

    lon, lat, alt = 77.594562, 12.971598, 920.5
    x, y, z = transformer.wgs84_to_metric(lon, lat, alt)

    # Invert back to WGS84
    lon_inv, lat_inv, alt_inv = transformer.metric_to_wgs84(x, y, z)
    assert abs(lon_inv - lon) < 1e-7
    assert abs(lat_inv - lat) < 1e-7
    assert abs(alt_inv - alt) < 1e-3


def test_geojson_coordinate_ordering() -> None:
    """Tests that format_geojson_point strictly outputs [longitude, latitude, altitude] (RFC 7946)."""
    transformer = CoordinateTransformer(
        source_crs="EPSG:4326",
        target_crs="EPSG:32643",
    )
    pt = transformer.format_geojson_point(lon=77.59, lat=12.97, alt=920.5)
    assert pt == [77.59, 12.97, 920.5]
    # Verify order is [lon, lat, alt]
    assert pt[0] == 77.59
    assert pt[1] == 12.97
    assert pt[2] == 920.5


# ===========================================================================
# 4. Umeyama Similarity Alignment Tests
# ===========================================================================

def test_alignment_known_ground_truth_recovery() -> None:
    """Tests exact recovery of known scale, rotation, and translation (synthetic mathematical verification)."""
    np.random.seed(42)
    # Synthetic local points
    P = np.random.randn(8, 3) * 10.0

    # Synthetic ground truth similarity transform
    s_gt = 4.25
    rot_gt = Rotation.from_euler("zyx", [25.0, -15.0, 40.0], degrees=True)
    R_gt = rot_gt.as_matrix()
    t_gt = np.array([500.0, -250.0, 100.0])

    # Transform to target: Q = s * P @ R^T + t
    Q = (s_gt * (P @ R_gt.T)) + t_gt

    res = align_umeyama(P, Q)

    assert abs(res.scale - s_gt) < 1e-6
    assert np.allclose(np.asarray(res.rotation_matrix), R_gt, atol=1e-5)
    assert np.allclose(np.asarray(res.translation_vector), t_gt, atol=1e-4)
    assert res.rmse < 1e-4
    assert res.max_error < 1e-4
    assert res.num_correspondences == 8
    assert res.status == "PASS"


def test_alignment_insufficient_correspondences() -> None:
    """Tests that alignment with N < 3 points raises ValueError."""
    P = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    Q = [[10.0, 20.0, 30.0], [11.0, 20.0, 30.0]]
    with pytest.raises(ValueError, match="Insufficient correspondences.*3 required"):
        align_umeyama(P, Q)


def test_alignment_degenerate_collinear_points() -> None:
    """Tests that collinear points are detected and rejected as degenerate geometry."""
    # Points lying strictly along a 1D line in 3D
    t = np.linspace(0, 10, 6)
    P = np.column_stack([t, 2*t + 1, -t + 3])
    Q = P + 10.0

    with pytest.raises(ValueError, match="collinear"):
        align_umeyama(P, Q)


def test_alignment_degenerate_coincident_points() -> None:
    """Tests that coincident points (near-zero variance) are rejected."""
    P = np.ones((5, 3)) * 2.0
    Q = np.ones((5, 3)) * 10.0

    with pytest.raises(ValueError, match="coincident or have near-zero spatial extent"):
        align_umeyama(P, Q)


def test_transform_camera_pose_opencv_convention() -> None:
    """Tests transforming camera center and orientation according to OpenCV conventions."""
    # Synthetic transform: scale=2.0, translation=[100, 200, 300], R=Identity
    transform = SimilarityTransform(
        scale=2.0,
        rotation_matrix=np.eye(3).tolist(),
        translation_vector=[100.0, 200.0, 300.0],
        rmse=0.01,
        max_error=0.02,
        num_correspondences=5,
        residuals=[0.01] * 5,
        status="PASS",
    )

    local_pos = [5.0, -2.0, 10.0]
    local_quat = [1.0, 0.0, 0.0, 0.0]  # [qw, qx, qy, qz]

    geo_pose = transform_camera_pose(
        local_position=local_pos,
        local_quaternion=local_quat,
        transform=transform,
    )

    # C_geo = 2.0 * [5, -2, 10] + [100, 200, 300] = [110, 196, 320]
    assert geo_pose["position"] == [110.0, 196.0, 320.0]
    # R_cam_geo = R_cam_local @ R_sim^T = I
    assert geo_pose["rotation_quaternion"] == [1.0, 0.0, 0.0, 0.0]


# ===========================================================================
# 5. Georeferencing Pipeline & GeoJSON Tests
# ===========================================================================

def test_run_georeferencing_end_to_end(tmp_path: Path) -> None:
    """Tests complete georeferencing pipeline generating all 4 output files."""
    out_dir = tmp_path / "georeferenced"

    # 1. Create synthetic cameras.json
    cams = [
        {"frame_id": "f0", "timestamp": 100.0, "position": [0.0, 0.0, 0.0], "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
        {"frame_id": "f1", "timestamp": 101.0, "position": [10.0, 0.0, 0.0], "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
        {"frame_id": "f2", "timestamp": 102.0, "position": [0.0, 10.0, 0.0], "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
        {"frame_id": "f3", "timestamp": 103.0, "position": [10.0, 10.0, 5.0], "rotation_quaternion": [1.0, 0.0, 0.0, 0.0]},
    ]
    cams_file = tmp_path / "cameras.json"
    cams_file.write_text(json.dumps(cams, indent=2), encoding="utf-8")

    # 2. Create synthetic GPS CSV with known non-collinear coordinates
    gps_rows = [
        ("timestamp", "latitude", "longitude", "altitude"),
        ("100.0", "12.971500", "77.594500", "920.0"),
        ("101.0", "12.971590", "77.594500", "920.0"),
        ("102.0", "12.971500", "77.594590", "920.0"),
        ("103.0", "12.971590", "77.594590", "925.0"),
    ]
    gps_file = tmp_path / "flight_gps.csv"
    with open(gps_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(gps_rows)

    # 3. Run georeferencing
    meta = run_georeferencing(
        cameras_path=cams_file,
        gps_path=gps_file,
        output_dir=out_dir,
    )

    # Verify return metadata
    assert meta["source_crs"] == "EPSG:4326"
    assert "EPSG:" in meta["target_crs"]
    assert meta["matched_observations"] == 4
    assert meta["scale"] > 0
    assert meta["alignment_rmse"] is not None
    assert meta["gps_available"] is True
    assert meta["imu_available"] is False

    # Verify generated output files
    assert (out_dir / "cameras.json").is_file()
    assert (out_dir / "trajectory.json").is_file()
    assert (out_dir / "trajectory.geojson").is_file()
    assert (out_dir / "metadata.json").is_file()

    # Verify GeoJSON validity
    with open(out_dir / "trajectory.geojson", "r", encoding="utf-8") as f:
        geojson = json.load(f)

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 5  # 1 LineString + 4 Points

    # Check LineString coordinates: list of [lon, lat, alt]
    line_feat = geojson["features"][0]
    assert line_feat["geometry"]["type"] == "LineString"
    coords = line_feat["geometry"]["coordinates"]
    assert len(coords) == 4
    for pt in coords:
        assert len(pt) == 3
        lon, lat, alt = pt
        assert 70.0 <= lon <= 80.0
        assert 10.0 <= lat <= 15.0
        assert alt > 900.0  # Altitude preserved


def test_georeferencing_failure_clean_handling(tmp_path: Path) -> None:
    """Tests that pipeline fails cleanly when GPS data is missing or incompatible."""
    cams_file = tmp_path / "empty_cams.json"
    cams_file.write_text("[]", encoding="utf-8")
    gps_file = tmp_path / "gps.csv"
    gps_file.write_text("timestamp,latitude,longitude,altitude\n", encoding="utf-8")

    with pytest.raises(ValueError):
        run_georeferencing(
            cameras_path=cams_file,
            gps_path=gps_file,
            output_dir=tmp_path / "out",
        )


# ===========================================================================
# 6. Validation Policy Revision Tests
# ===========================================================================

def test_validation_policy_provisional_defaults() -> None:
    """Tests that ValidationPolicy retains required provisional engineering defaults."""
    policy = ValidationPolicy()
    assert policy.min_registration_rate == 0.80
    assert policy.max_reprojection_rmse_px == 1.0
    assert policy.max_reprojection_rmse == 1.0
    assert policy.require_trajectory is True
    assert policy.policy_type == "provisional_engineering_gate"

    p_dict = policy.to_dict()
    assert p_dict["min_registration_rate"] == 0.80
    assert p_dict["max_reprojection_rmse_px"] == 1.0
    assert p_dict["require_trajectory"] is True
    assert p_dict["policy_type"] == "provisional_engineering_gate"


def test_validation_policy_custom_thresholds() -> None:
    """Tests configuring custom thresholds and disabling trajectory requirement."""
    policy = ValidationPolicy(
        min_registration_rate=0.70,
        max_reprojection_rmse_px=1.5,
        require_trajectory=False,
    )
    assert policy.min_registration_rate == 0.70
    assert policy.max_reprojection_rmse_px == 1.5
    assert policy.require_trajectory is False


VALID_JPEG_BYTES = bytes([
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


def test_validation_policy_serialized_in_report(tmp_path: Path) -> None:
    """Tests that pose_engine_report.json records the exact policy used."""
    img_dir = tmp_path / "synthetic_images"
    img_dir.mkdir(parents=True)
    for i in range(25):
        (img_dir / f"frame_{i:04d}.jpg").write_bytes(VALID_JPEG_BYTES)

    out_dir = tmp_path / "adaptive_out"

    # Mock FastMap passing candidate
    def mock_pass() -> PoseResult:
        cam_file = tmp_path / "fastmap_cam.json"
        cam_file.write_text(json.dumps([{"frame_id": "f0", "position": [0, 0, 0]}]))
        return PoseResult(
            method="FastMap",
            status="PASS",
            output_path=tmp_path,
            runtime_seconds=1.2,
            validation_status="PASS",
            registration_rate=0.95,
            registered_images=24,
            input_images=25,
            reprojection_rmse_px=0.6,
            reprojection_status="AVAILABLE",
            trajectory_status="PASS",
            cameras_path=cam_file,
        )

    policy = ValidationPolicy(
        min_registration_rate=0.85,
        max_reprojection_rmse_px=0.8,
        require_trajectory=True,
    )

    report = run_adaptive_engine(
        images=img_dir,
        output=out_dir,
        method="fastmap",
        policy=policy,
        mock_backends={"fastmap": mock_pass},
    )

    assert "validation_policy" in report
    val_policy_meta = report["validation_policy"]
    assert val_policy_meta["min_registration_rate"] == 0.85
    assert val_policy_meta["max_reprojection_rmse_px"] == 0.8
    assert val_policy_meta["require_trajectory"] is True
    assert val_policy_meta["policy_type"] == "provisional_engineering_gate"

    # Check file on disk
    with open(out_dir / "pose_engine_report.json", "r", encoding="utf-8") as f:
        saved_report = json.load(f)

    assert saved_report["validation_policy"]["policy_type"] == "provisional_engineering_gate"


def test_adaptive_engine_with_gps_georeferencing(tmp_path: Path) -> None:
    """Tests full pipeline integration: Input -> Pre-flight -> Pose Engine -> Georeferencing."""
    img_dir = tmp_path / "flight_images"
    img_dir.mkdir(parents=True)
    for i in range(25):
        (img_dir / f"frame_{i:04d}.jpg").write_bytes(VALID_JPEG_BYTES)

    out_dir = tmp_path / "pipeline_out"

    # Synthetic cameras
    cams = [
        {"frame_id": "f0", "timestamp": 100.0, "position": [0.0, 0.0, 0.0]},
        {"frame_id": "f1", "timestamp": 101.0, "position": [10.0, 0.0, 0.0]},
        {"frame_id": "f2", "timestamp": 102.0, "position": [0.0, 10.0, 0.0]},
        {"frame_id": "f3", "timestamp": 103.0, "position": [10.0, 10.0, 5.0]},
    ]
    cams_file = tmp_path / "mock_cams.json"
    cams_file.write_text(json.dumps(cams), encoding="utf-8")

    # Synthetic GPS CSV
    gps_file = tmp_path / "flight_gps.csv"
    gps_content = (
        "timestamp,latitude,longitude,altitude\n"
        "100.0,12.971500,77.594500,920.0\n"
        "101.0,12.971590,77.594500,920.0\n"
        "102.0,12.971500,77.594590,920.0\n"
        "103.0,12.971590,77.594590,925.0\n"
    )
    gps_file.write_text(gps_content, encoding="utf-8")

    def mock_pass() -> PoseResult:
        return PoseResult(
            method="FastMap",
            status="PASS",
            output_path=tmp_path,
            runtime_seconds=1.5,
            validation_status="PASS",
            registration_rate=0.92,
            registered_images=23,
            input_images=25,
            reprojection_rmse_px=0.7,
            reprojection_status="AVAILABLE",
            trajectory_status="PASS",
            cameras_path=cams_file,
        )

    report = run_adaptive_engine(
        images=img_dir,
        output=out_dir,
        method="fastmap",
        gps_path=gps_file,
        mock_backends={"fastmap": mock_pass},
    )

    assert report["overall_status"] == "PASS"
    assert report["georeferencing"] is not None
    assert report["georeferencing"]["matched_observations"] == 4

    geo_dir = out_dir / "selected" / "georeferenced"
    assert (geo_dir / "cameras.json").is_file()
    assert (geo_dir / "trajectory.json").is_file()
    assert (geo_dir / "trajectory.geojson").is_file()
    assert (geo_dir / "metadata.json").is_file()
