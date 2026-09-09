"""End-to-End Georeferencing Pipeline and CLI.

Transforms local reconstruction camera poses into a geospatial coordinate system
using GPS/IMU data, automatic UTM zone selection, and Umeyama similarity alignment.

Outputs:
  output/georeferenced/
    ├── cameras.json
    ├── trajectory.json
    ├── trajectory.geojson
    └── metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.georeferencing.alignment import (
    SimilarityTransform,
    align_umeyama,
    transform_camera_pose,
)
from reconstruction.georeferencing.coordinate_transform import (
    CoordinateTransformer,
    determine_utm_crs,
)
from reconstruction.georeferencing.gps import (
    GPSRecord,
    load_gps_csv,
    parse_timestamp,
    synchronize_poses_with_gps,
)

logger = logging.getLogger("georeferencing")


def load_imu_data(imu_path: Optional[str | Path]) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """Optionally loads and parses IMU data (CSV or JSON).

    Exposes orientation and timestamp data for validation; does not perform full Kalman fusion.
    """
    if not imu_path:
        return False, [], {"imu_available": False, "reason": "No IMU path specified"}

    path = Path(imu_path).resolve()
    if not path.is_file():
        return False, [], {"imu_available": False, "reason": f"IMU file not found: {path}"}

    try:
        records: List[Dict[str, Any]] = []
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and "records" in data:
                    records = data["records"]
        else:
            import csv
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(dict(row))

        return True, records, {
            "imu_available": True,
            "source_file": str(path),
            "total_records": len(records),
        }
    except Exception as exc:
        logger.warning("Failed to parse IMU data: %s", exc)
        return False, [], {"imu_available": False, "reason": f"Parsing failed: {exc}"}


def run_georeferencing(
    cameras_path: str | Path,
    gps_path: str | Path,
    output_dir: str | Path,
    imu_path: Optional[str | Path] = None,
    max_time_diff_s: float = 1.0,
    target_crs: Optional[str] = None,
    altitude_reference: str = "WGS84_ellipsoidal",
    source_pose_method: str = "selected_reconstruction",
) -> Dict[str, Any]:
    """Executes georeferencing pipeline from local camera poses and GPS data.

    Steps:
      1. Load and validate GPS records.
      2. Load local camera poses from cameras.json.
      3. Synchronize camera poses with GPS observations.
      4. Determine target CRS (UTM or user-specified) and transform GPS coordinates.
      5. Estimate 3D similarity transformation via Umeyama algorithm.
      6. Georeference camera poses and trajectory.
      7. Export cameras.json, trajectory.json, trajectory.geojson, metadata.json.

    Returns:
      Comprehensive metadata dictionary.

    Raises:
      FileNotFoundError, ValueError: If inputs are missing, malformed, or alignment fails.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_path = Path(cameras_path).resolve()
    # Support passing either cameras.json directly or directory containing cameras.json
    if cam_path.is_dir():
        candidate = cam_path / "cameras.json"
        if candidate.is_file():
            cam_path = candidate
        else:
            raise FileNotFoundError(f"Directory '{cam_path}' does not contain cameras.json")

    if not cam_path.is_file():
        raise FileNotFoundError(f"Local cameras.json not found: {cam_path}")

    # 1. Load local cameras
    with open(cam_path, "r", encoding="utf-8") as f:
        local_cameras = json.load(f)

    if not isinstance(local_cameras, list) or len(local_cameras) == 0:
        raise ValueError(f"Cameras file '{cam_path}' must contain a non-empty list of camera dictionaries")

    # 2. Load GPS records
    gps_records, gps_meta = load_gps_csv(gps_path, altitude_reference=altitude_reference)

    # 3. Synchronize poses with GPS
    matched_pairs, sync_summary = synchronize_poses_with_gps(
        poses=local_cameras,
        gps_records=gps_records,
        max_time_diff_s=max_time_diff_s,
    )

    if len(matched_pairs) < 3:
        raise ValueError(
            f"Insufficient synchronized GPS-camera correspondences: {len(matched_pairs)} matched, "
            f"minimum 3 required for 3D similarity alignment. (Tolerance: {max_time_diff_s}s)"
        )

    # 4. Determine coordinate projection
    mean_lon = float(np.mean([p["longitude"] for p in matched_pairs]))
    mean_lat = float(np.mean([p["latitude"] for p in matched_pairs]))

    transformer = CoordinateTransformer(
        source_crs="EPSG:4326",
        target_crs=target_crs,
        reference_point=(mean_lon, mean_lat),
        altitude_reference=altitude_reference,
    )

    # Transform matched GPS points to projected metric coordinates
    local_pts = []
    geo_pts = []
    for pair in matched_pairs:
        local_pts.append(pair["local_position"])
        mx, my, mz = transformer.wgs84_to_metric(
            pair["longitude"], pair["latitude"], pair["altitude"]
        )
        geo_pts.append([mx, my, mz])

    # 5. Procrustes / Umeyama similarity alignment
    alignment_result = align_umeyama(local_pts, geo_pts)

    # Build match lookup by frame_id
    match_lookup: Dict[str, Dict[str, Any]] = {}
    for idx, pair in enumerate(matched_pairs):
        f_id = pair["frame_id"]
        res_val = alignment_result.residuals[idx] if idx < len(alignment_result.residuals) else None
        match_lookup[f_id] = {
            "gps_timestamp": pair["gps_timestamp"],
            "pose_timestamp": pair["pose_timestamp"],
            "time_diff_s": pair["time_difference_s"],
            "wgs84": [pair["longitude"], pair["latitude"], pair["altitude"]],
            "metric_target": geo_pts[idx],
            "residual_m": round(res_val, 4) if res_val is not None else None,
        }

    # 6. Georeference all local camera poses
    georeferenced_cameras: List[Dict[str, Any]] = []
    georeferenced_trajectory: List[Dict[str, Any]] = []
    geojson_features: List[Dict[str, Any]] = []
    linestring_coords: List[List[float]] = []

    for idx, cam in enumerate(local_cameras):
        frame_id = cam.get("frame_id", f"frame_{idx:06d}")
        img_name = cam.get("image_name", "")
        local_pos = cam.get("position", [0.0, 0.0, 0.0])
        local_rot = cam.get("rotation_matrix")
        local_quat = cam.get("rotation_quaternion")

        # Pose transformation
        geo_pose = transform_camera_pose(
            local_position=local_pos,
            local_rotation_matrix=local_rot,
            local_quaternion=local_quat,
            transform=alignment_result,
        )

        # Compute WGS84 coordinates from metric position
        metric_pos = geo_pose["position"]
        lon_wgs, lat_wgs, alt_wgs = transformer.metric_to_wgs84(
            metric_pos[0], metric_pos[1], metric_pos[2]
        )
        geojson_pt = transformer.format_geojson_point(lon_wgs, lat_wgs, alt_wgs)
        linestring_coords.append(geojson_pt)

        match_info = match_lookup.get(frame_id)
        residual_m = match_info.get("residual_m") if match_info else None

        # Camera record
        cam_rec = {
            "frame_id": frame_id,
            "image_name": img_name,
            "position": metric_pos,
            "rotation_matrix": geo_pose["rotation_matrix"],
            "rotation_quaternion": geo_pose["rotation_quaternion"],
            "coordinate_system": transformer.target_crs,
            "source_pose_method": source_pose_method,
            "wgs84": {
                "longitude": round(lon_wgs, 7),
                "latitude": round(lat_wgs, 7),
                "altitude": round(alt_wgs, 3),
            },
        }
        georeferenced_cameras.append(cam_rec)

        # Trajectory record
        traj_rec = {
            "frame_id": frame_id,
            "image_name": img_name,
            "local_position": local_pos,
            "georeferenced_position": metric_pos,
            "wgs84_position": {
                "longitude": round(lon_wgs, 7),
                "latitude": round(lat_wgs, 7),
                "altitude": round(alt_wgs, 3),
            },
            "timestamp": cam.get("timestamp") or (match_info.get("pose_timestamp") if match_info else None),
            "matched_gps": match_info,
            "residual_m": residual_m,
        }
        georeferenced_trajectory.append(traj_rec)

        # GeoJSON Point feature
        pt_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": geojson_pt,  # [longitude, latitude, altitude]
            },
            "properties": {
                "frame_id": frame_id,
                "image_name": img_name,
                "altitude_m": round(alt_wgs, 3),
                "residual_m": residual_m,
                "matched_gps": match_info is not None,
            },
        }
        geojson_features.append(pt_feature)

    # Add trajectory LineString feature
    if len(linestring_coords) >= 2:
        linestring_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": linestring_coords,  # List of [lon, lat, alt]
            },
            "properties": {
                "name": "Camera Flight Trajectory",
                "total_frames": len(linestring_coords),
                "source_crs": transformer.source_crs,
                "target_crs": transformer.target_crs,
            },
        }
        # Place line feature at beginning
        geojson_features.insert(0, linestring_feature)

    geojson_collection = {
        "type": "FeatureCollection",
        "name": "Georeferenced Camera Trajectory",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": geojson_features,
    }

    # 7. Optional IMU handling
    imu_available, imu_records, imu_meta = load_imu_data(imu_path)

    # 8. Validation checks
    val_status = "PASS"
    val_checks = {
        "correspondences_check": "PASS" if len(matched_pairs) >= 3 else "FAIL",
        "synchronization_rate": sync_summary.get("synchronization_rate", 0.0),
        "synchronization_check": "PASS" if sync_summary.get("synchronization_rate", 0.0) >= 0.5 else "WARNING",
        "degeneracy_check": "PASS",
        "rmse_check": alignment_result.status,
    }
    if alignment_result.status == "WARNING" or val_checks["synchronization_check"] == "WARNING":
        val_status = "WARNING"

    # 9. Build metadata
    metadata = {
        "source_crs": transformer.source_crs,
        "target_crs": transformer.target_crs,
        "coordinate_system": transformer.coordinate_system_name,
        "alignment_method": "similarity_transform",
        "scale": round(alignment_result.scale, 8),
        "matched_observations": len(matched_pairs),
        "alignment_rmse": round(alignment_result.rmse, 4),
        "alignment_max_error": round(alignment_result.max_error, 4),
        "altitude_reference": altitude_reference,
        "gps_available": True,
        "imu_available": imu_available,
        "opencv_convention": {
            "axes": "X=right, Y=down, Z=forward",
            "camera_center": "C = -R^T t",
            "quaternion": "[qw, qx, qy, qz]",
        },
        "geojson_coordinate_order": "[longitude, latitude, altitude] (RFC 7946)",
        "synchronization": sync_summary,
        "alignment": alignment_result.to_dict(),
        "imu": imu_meta,
        "validation": {
            "status": val_status,
            "checks": val_checks,
            "note": "Internal software validation gates; not survey-grade ground truth.",
        },
    }

    # 10. Write outputs
    cameras_out = out_dir / "cameras.json"
    traj_out = out_dir / "trajectory.json"
    geojson_out = out_dir / "trajectory.geojson"
    meta_out = out_dir / "metadata.json"

    with open(cameras_out, "w", encoding="utf-8") as f:
        json.dump(georeferenced_cameras, f, indent=2)

    with open(traj_out, "w", encoding="utf-8") as f:
        json.dump(georeferenced_trajectory, f, indent=2)

    with open(geojson_out, "w", encoding="utf-8") as f:
        json.dump(geojson_collection, f, indent=2)

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Georeferencing complete: %d cameras exported to %s", len(georeferenced_cameras), out_dir)

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPS/IMU Georeferencing CLI (Person 1 - Step 7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cameras",
        required=True,
        type=Path,
        help="Path to local cameras.json or selected reconstruction directory",
    )
    parser.add_argument(
        "--gps",
        required=True,
        type=Path,
        help="Path to GPS CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reconstruction/test_output/georeferenced"),
        help="Directory to save georeferenced cameras, trajectory, and metadata",
    )
    parser.add_argument(
        "--imu",
        type=Path,
        default=None,
        help="Optional path to IMU data file (CSV or JSON)",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=1.0,
        help="Maximum timestamp synchronization difference in seconds",
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default=None,
        help="Target projected CRS (e.g. EPSG:32643). If omitted, automatically determined from GPS coordinates.",
    )
    parser.add_argument(
        "--altitude-reference",
        type=str,
        default="WGS84_ellipsoidal",
        help="Altitude reference datum (e.g. WGS84_ellipsoidal, EGM96_geoid)",
    )

    args = parser.parse_args()

    try:
        meta = run_georeferencing(
            cameras_path=args.cameras,
            gps_path=args.gps,
            output_dir=args.output,
            imu_path=args.imu,
            max_time_diff_s=args.max_time_diff,
            target_crs=args.target_crs,
            altitude_reference=args.altitude_reference,
        )

        print("\n=====================================================================")
        print("                   GEOREFERENCING EXECUTION REPORT                   ")
        print("=====================================================================")
        print(f"Validation Status:    {meta['validation']['status']}")
        print(f"Source CRS:           {meta['source_crs']}")
        print(f"Target CRS:           {meta['target_crs']} ({meta['coordinate_system']})")
        print(f"Matched Observations: {meta['matched_observations']}")
        print(f"Scale Factor:         {meta['scale']}")
        print(f"Alignment RMSE:       {meta['alignment_rmse']} m")
        print(f"Max Alignment Error:  {meta['alignment_max_error']} m")
        print(f"Altitude Reference:   {meta['altitude_reference']}")
        print(f"GPS Available:        {meta['gps_available']}")
        print(f"IMU Available:        {meta['imu_available']}")
        print(f"Output Directory:     {args.output}")
        print("=====================================================================\n")

        sys.exit(0 if meta["validation"]["status"] in ["PASS", "WARNING"] else 1)

    except Exception as exc:
        print(f"\n[ERROR] Georeferencing failed: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
