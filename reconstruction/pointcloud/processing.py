"""Point cloud processing algorithms, quality validation, and pipeline module (SIH26158 Step 8).

Implements:
1. Input point cloud quality validation (geometry, bounds, finite coords, density)
2. Statistical outlier removal (Open3D, neighbourhood statistics)
3. Optional radius-based filtering
4. Optional voxel grid downsampling with coordinate unit awareness
5. Clean ground/non-ground segmentation extension point (disabled by default)
6. Complete configurable pipeline and statistics reporting
7. Standalone CLI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.pointcloud.export import (
    PointCloudData,
    export_point_cloud_bundle,
    export_point_cloud_ply,
    load_point_cloud,
)

logger = logging.getLogger("pointcloud.processing")

# Engineering defaults (clearly marked as provisional project gates, not universally optimal)
DEFAULT_NB_NEIGHBORS = 20
DEFAULT_STD_RATIO = 2.0
DEFAULT_RADIUS_MIN_POINTS = 10
EXCESSIVE_POINT_REMOVAL_THRESHOLD = 0.30  # Warning if retention < 30%


def validate_point_cloud(pcd: PointCloudData) -> Dict[str, Any]:
    """Validates point cloud geometry, finiteness, bounds, and density.

    Checks:
    - Finite coordinates (no NaN or Inf)
    - Non-empty cloud (points > 0)
    - Valid bounding-box extents
    - Detect duplicate coordinates
    - Point density metrics

    Raises:
        ValueError: If cloud is empty or contains non-finite coordinates.
    """
    if pcd.num_points == 0:
        raise ValueError("Point cloud validation failed: Point cloud is empty (0 points).")

    points = pcd.points
    if not np.all(np.isfinite(points)):
        raise ValueError("Point cloud validation failed: Coordinates contain NaN or Infinite values.")

    xyz_min = [float(v) for v in np.min(points, axis=0)]
    xyz_max = [float(v) for v in np.max(points, axis=0)]
    extents = [float(xyz_max[i] - xyz_min[i]) for i in range(3)]

    # Check for degenerate bounding box (flat or zero extent)
    warnings: List[str] = []
    if any(e <= 1e-6 for e in extents):
        warnings.append(f"Near-zero spatial extent in one or more dimensions: extents={extents}")

    # Volume and density calculation
    bbox_volume = extents[0] * extents[1] * extents[2]
    point_density = (pcd.num_points / bbox_volume) if bbox_volume > 1e-12 else 0.0

    # Duplicate coordinate check
    unique_points_count = len(np.unique(points, axis=0))
    duplicate_count = pcd.num_points - unique_points_count
    if duplicate_count > 0:
        warnings.append(f"Detected {duplicate_count} duplicate coordinate points ({duplicate_count / pcd.num_points * 100:.1f}%).")

    # RGB validation if present
    rgb_valid = False
    if pcd.has_colors:
        if np.all((pcd.colors >= 0) & (pcd.colors <= 255)):
            rgb_valid = True
        else:
            warnings.append("RGB color array contains values outside [0, 255].")

    status = "WARNING" if warnings else "PASS"

    return {
        "status": status,
        "num_points": pcd.num_points,
        "unique_points": unique_points_count,
        "duplicate_points": duplicate_count,
        "xyz_min": [round(v, 6) for v in xyz_min],
        "xyz_max": [round(v, 6) for v in xyz_max],
        "extents": [round(v, 6) for v in extents],
        "bounding_box_volume": round(bbox_volume, 6),
        "point_density_pts_per_unit3": round(point_density, 4),
        "rgb_available": pcd.has_colors,
        "rgb_valid": rgb_valid,
        "crs": pcd.crs,
        "units": pcd.units,
        "is_georeferenced": pcd.is_georeferenced,
        "warnings": warnings,
    }


def filter_statistical_outliers(
    pcd: PointCloudData,
    nb_neighbors: int = DEFAULT_NB_NEIGHBORS,
    std_ratio: float = DEFAULT_STD_RATIO,
    enabled: bool = True,
) -> Tuple[PointCloudData, Dict[str, Any]]:
    """Applies statistical outlier filtering to remove sparse noise points.

    Algorithm:
    1. Computes mean distance from each point to its k nearest neighbours.
    2. Computes the global mean and standard deviation of these neighbour distances.
    3. Retains only points whose mean neighbour distance is within mean + std_ratio * std.

    Args:
        pcd: Input PointCloudData
        nb_neighbors: Number of nearest neighbours (engineering default = 20)
        std_ratio: Standard deviation multiplier threshold (engineering default = 2.0)
        enabled: Whether to execute this stage

    Returns:
        Tuple of (filtered PointCloudData, stage report dict)
    """
    start_time = time.time()
    n_in = pcd.num_points

    if not enabled:
        return pcd, {
            "enabled": False,
            "status": "SKIPPED",
            "initial_points": n_in,
            "remaining_points": n_in,
            "removed_points": 0,
            "retention_rate": 1.0,
            "time_seconds": 0.0,
            "parameters": {
                "nb_neighbors": nb_neighbors,
                "std_ratio": std_ratio,
                "note": "Stage disabled by user configuration.",
            },
        }

    if nb_neighbors <= 0:
        raise ValueError(f"nb_neighbors must be a positive integer, got {nb_neighbors}")
    if std_ratio <= 0.0:
        raise ValueError(f"std_ratio must be a positive float, got {std_ratio}")

    if n_in <= nb_neighbors:
        logger.warning("Point cloud has fewer points (%d) than nb_neighbors (%d); skipping filter.", n_in, nb_neighbors)
        return pcd, {
            "enabled": True,
            "status": "SKIPPED_TOO_FEW_POINTS",
            "initial_points": n_in,
            "remaining_points": n_in,
            "removed_points": 0,
            "retention_rate": 1.0,
            "time_seconds": round(time.time() - start_time, 4),
            "parameters": {"nb_neighbors": nb_neighbors, "std_ratio": std_ratio},
        }

    o3d_pcd = pcd.to_open3d()
    cl, inlier_indices = o3d_pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    filtered_pcd = PointCloudData.from_open3d(cl, source_data=pcd, inlier_indices=inlier_indices)
    elapsed = time.time() - start_time
    n_out = filtered_pcd.num_points
    n_removed = n_in - n_out
    retention = (n_out / n_in) if n_in > 0 else 0.0

    report = {
        "enabled": True,
        "status": "COMPLETED",
        "initial_points": n_in,
        "remaining_points": n_out,
        "removed_points": n_removed,
        "retention_rate": round(retention, 4),
        "time_seconds": round(elapsed, 4),
        "parameters": {
            "nb_neighbors": nb_neighbors,
            "std_ratio": std_ratio,
            "parameter_type": "provisional_engineering_default",
        },
    }
    logger.info("Statistical filter: %d -> %d points (-%d, %.1f%% retained) in %.3fs", n_in, n_out, n_removed, retention * 100, elapsed)
    return filtered_pcd, report


def filter_radius_outliers(
    pcd: PointCloudData,
    radius: Optional[float] = None,
    min_points: int = DEFAULT_RADIUS_MIN_POINTS,
    enabled: bool = False,
) -> Tuple[PointCloudData, Dict[str, Any]]:
    """Applies optional radius-based outlier filtering.

    A point is retained only when at least min_points neighbours exist within the radius sphere.

    Args:
        pcd: Input PointCloudData
        radius: Sphere radius in point-cloud coordinate units
        min_points: Minimum number of neighbouring points required
        enabled: Whether to execute this stage (disabled by default)

    Returns:
        Tuple of (filtered PointCloudData, stage report dict)
    """
    start_time = time.time()
    n_in = pcd.num_points

    if not enabled or radius is None:
        return pcd, {
            "enabled": False,
            "status": "SKIPPED",
            "initial_points": n_in,
            "remaining_points": n_in,
            "removed_points": 0,
            "retention_rate": 1.0,
            "time_seconds": 0.0,
            "parameters": {
                "radius": radius,
                "min_points": min_points,
                "note": "Optional radius filtering not enabled or radius not specified.",
            },
        }

    if radius <= 0.0:
        raise ValueError(f"radius must be a positive float, got {radius}")
    if min_points <= 0:
        raise ValueError(f"min_points must be a positive integer, got {min_points}")

    if n_in < min_points:
        return pcd, {
            "enabled": True,
            "status": "SKIPPED_TOO_FEW_POINTS",
            "initial_points": n_in,
            "remaining_points": n_in,
            "removed_points": 0,
            "retention_rate": 1.0,
            "time_seconds": round(time.time() - start_time, 4),
            "parameters": {"radius": radius, "min_points": min_points},
        }

    o3d_pcd = pcd.to_open3d()
    cl, inlier_indices = o3d_pcd.remove_radius_outlier(
        nb_points=min_points,
        radius=radius,
    )

    filtered_pcd = PointCloudData.from_open3d(cl, source_data=pcd, inlier_indices=inlier_indices)
    elapsed = time.time() - start_time
    n_out = filtered_pcd.num_points
    n_removed = n_in - n_out
    retention = (n_out / n_in) if n_in > 0 else 0.0

    report = {
        "enabled": True,
        "status": "COMPLETED",
        "initial_points": n_in,
        "remaining_points": n_out,
        "removed_points": n_removed,
        "retention_rate": round(retention, 4),
        "time_seconds": round(elapsed, 4),
        "coordinate_units": pcd.units,
        "parameters": {
            "radius": radius,
            "min_points": min_points,
            "radius_units": pcd.units,
        },
    }
    logger.info("Radius filter (r=%.3f %s): %d -> %d points in %.3fs", radius, pcd.units, n_in, n_out, elapsed)
    return filtered_pcd, report


def downsample_voxels(
    pcd: PointCloudData,
    voxel_size: Optional[float] = None,
    enabled: bool = False,
) -> Tuple[PointCloudData, Dict[str, Any]]:
    """Applies optional voxel grid downsampling with coordinate-unit awareness.

    Points falling into the same 3D voxel grid bin are averaged into their centroid.

    Important:
    - If the cloud is georeferenced, the units are metres.
    - If unscaled/local COLMAP, voxel size applies to arbitrary local units.

    Args:
        pcd: Input PointCloudData
        voxel_size: Edge length of each cubic voxel grid cell
        enabled: Whether to execute this stage (disabled by default)

    Returns:
        Tuple of (downsampled PointCloudData, stage report dict)
    """
    start_time = time.time()
    n_in = pcd.num_points

    if not enabled or voxel_size is None:
        return pcd, {
            "enabled": False,
            "status": "SKIPPED",
            "initial_points": n_in,
            "remaining_points": n_in,
            "removed_points": 0,
            "retention_rate": 1.0,
            "time_seconds": 0.0,
            "coordinate_system": pcd.crs,
            "coordinate_units": pcd.units,
            "voxel_size": None,
            "parameters": {
                "voxel_size": voxel_size,
                "note": "Optional voxel downsampling not enabled.",
            },
        }

    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be a positive float, got {voxel_size}")

    warning_msg: Optional[str] = None
    if not pcd.is_georeferenced or pcd.units == "local_unscaled":
        warning_msg = (
            f"Voxel size ({voxel_size}) applied directly in local unscaled COLMAP units, "
            "not physical metres. Point cloud is not georeferenced."
        )
        logger.warning(warning_msg)

    o3d_pcd = pcd.to_open3d()
    down_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Note: Voxel downsampling merges points so point IDs are not preserved 1-to-1
    result_pcd = PointCloudData.from_open3d(down_pcd, source_data=pcd)
    elapsed = time.time() - start_time
    n_out = result_pcd.num_points
    n_removed = n_in - n_out
    retention = (n_out / n_in) if n_in > 0 else 0.0

    report = {
        "enabled": True,
        "status": "COMPLETED",
        "initial_points": n_in,
        "remaining_points": n_out,
        "removed_points": n_removed,
        "retention_rate": round(retention, 4),
        "time_seconds": round(elapsed, 4),
        "coordinate_system": pcd.crs,
        "coordinate_units": pcd.units,
        "voxel_size": voxel_size,
        "warning": warning_msg,
        "parameters": {
            "voxel_size": voxel_size,
            "voxel_size_units": pcd.units,
        },
    }
    logger.info("Voxel downsampling (size=%.4f %s): %d -> %d points in %.3fs", voxel_size, pcd.units, n_in, n_out, elapsed)
    return result_pcd, report


def filter_ground_extension(
    pcd: PointCloudData,
    enabled: bool = False,
    distance_threshold: float = 0.2,
) -> Tuple[PointCloudData, Dict[str, Any]]:
    """Clean extension point for future terrain/ground classification.

    Per Step 8 Section 8:
    - Disabled by default
    - Preserves all geometry, structures, buildings, terrain, and vegetation
    - Returns original cloud untouched unless explicitly enabled
    """
    if not enabled:
        return pcd, {
            "enabled": False,
            "status": "SKIPPED",
            "note": "Extension point for future terrain classification; disabled by default to preserve all geometry and structural edges.",
        }

    # Optional RANSAC plane segmentation placeholder when explicitly requested
    start_time = time.time()
    try:
        o3d_pcd = pcd.to_open3d()
        plane_model, inliers = o3d_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        return pcd, {
            "enabled": True,
            "status": "ANALYZED_NOT_FILTERED",
            "plane_equation": [round(float(c), 6) for c in plane_model],
            "ground_inliers_count": len(inliers),
            "ground_fraction": round(len(inliers) / pcd.num_points, 4),
            "time_seconds": round(time.time() - start_time, 4),
            "note": "Ground plane detected for spatial analysis without discarding non-ground points.",
        }
    except Exception as exc:
        return pcd, {
            "enabled": True,
            "status": "FAILED",
            "error": str(exc),
            "note": "Ground analysis encountered an error; original point cloud preserved.",
        }


def run_point_cloud_pipeline(
    input_cloud: Union[PointCloudData, str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    georef_metadata_path: Optional[Union[str, Path]] = None,
    nb_neighbors: int = DEFAULT_NB_NEIGHBORS,
    std_ratio: float = DEFAULT_STD_RATIO,
    enable_statistical: bool = True,
    radius: Optional[float] = None,
    min_points: int = DEFAULT_RADIUS_MIN_POINTS,
    enable_radius: bool = False,
    voxel_size: Optional[float] = None,
    enable_voxel: bool = False,
    enable_ground: bool = False,
    source_method: Optional[str] = None,
) -> Tuple[PointCloudData, Dict[str, Any]]:
    """Executes the full 7-stage Point-Cloud Processing Pipeline:

    Input Point Cloud
            ↓
    Validate Input
            ↓
    Statistical Outlier Removal
            ↓
    Optional Radius Filtering
            ↓
    Optional Voxel Downsampling
            ↓
    Validate Output
            ↓
    Export

    Args:
        input_cloud: PointCloudData object or path to points3D.txt/points3D.bin/directory.
        output_dir: Optional directory to export PLY and reports.
        georef_metadata_path: Optional path to Step 7 georeferencing metadata.json.
        nb_neighbors: Statistical filter neighbour count.
        std_ratio: Statistical filter standard deviation multiplier.
        enable_statistical: Enable/disable statistical outlier removal.
        radius: Radius filter distance in point cloud units.
        min_points: Radius filter minimum points.
        enable_radius: Enable/disable radius outlier removal.
        voxel_size: Voxel downsampling grid size.
        enable_voxel: Enable/disable voxel downsampling.
        enable_ground: Enable/disable ground separation extension.
        source_method: Optional source method label.

    Returns:
        Tuple of (Cleaned PointCloudData, comprehensive processing statistics dict).
    """
    total_start = time.time()
    timing: Dict[str, float] = {}

    # 1. Load Point Cloud
    t0 = time.time()
    if isinstance(input_cloud, PointCloudData):
        pcd = input_cloud.copy()
        if source_method:
            pcd.source_method = source_method
    else:
        pcd = load_point_cloud(
            input_path=input_cloud,
            georef_metadata_path=georef_metadata_path,
            source_method=source_method,
        )
    timing["loading"] = round(time.time() - t0, 4)

    input_point_count = pcd.num_points

    # 2. Validate Input
    t0 = time.time()
    input_validation = validate_point_cloud(pcd)
    timing["input_validation"] = round(time.time() - t0, 4)

    # 3. Statistical Outlier Removal
    current_pcd, stat_report = filter_statistical_outliers(
        pcd=pcd,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
        enabled=enable_statistical,
    )
    timing["statistical_filter"] = stat_report.get("time_seconds", 0.0)
    after_stat_count = current_pcd.num_points

    # 4. Optional Radius Outlier Removal
    current_pcd, radius_report = filter_radius_outliers(
        pcd=current_pcd,
        radius=radius,
        min_points=min_points,
        enabled=enable_radius,
    )
    timing["radius_filter"] = radius_report.get("time_seconds", 0.0)
    after_radius_count = current_pcd.num_points

    # 5. Optional Voxel Downsampling
    current_pcd, voxel_report = downsample_voxels(
        pcd=current_pcd,
        voxel_size=voxel_size,
        enabled=enable_voxel,
    )
    timing["voxel_downsample"] = voxel_report.get("time_seconds", 0.0)
    after_voxel_count = current_pcd.num_points

    # 6. Optional Ground Separation Extension
    current_pcd, ground_report = filter_ground_extension(
        pcd=current_pcd,
        enabled=enable_ground,
    )

    # 7. Validate Output
    t0 = time.time()
    output_validation = validate_point_cloud(current_pcd)
    timing["output_validation"] = round(time.time() - t0, 4)

    final_point_count = current_pcd.num_points
    removed_point_count = input_point_count - final_point_count
    retention_rate = (final_point_count / input_point_count) if input_point_count > 0 else 0.0

    # Quality and retention check
    warnings: List[str] = list(output_validation.get("warnings", []))
    if retention_rate < EXCESSIVE_POINT_REMOVAL_THRESHOLD:
        msg = (
            f"Excessive point removal warning: retention rate is {retention_rate * 100:.1f}% "
            f"(below {EXCESSIVE_POINT_REMOVAL_THRESHOLD * 100:.0f}% threshold). Check filter thresholds."
        )
        warnings.append(msg)
        logger.warning(msg)

    # Coordinate integrity check: verify CRS and units were not lost
    if pcd.is_georeferenced:
        if current_pcd.crs != pcd.crs or current_pcd.units != pcd.units:
            err = f"Coordinate integrity compromised: CRS changed from {pcd.crs} to {current_pcd.crs} during filtering."
            warnings.append(err)
            logger.error(err)

    # Total elapsed time
    total_time = round(time.time() - total_start, 4)
    timing["total"] = total_time

    # 8. Statistics Dictionary (strictly conforming to Step 8 Section 11)
    stats: Dict[str, Any] = {
        "input_points": input_point_count,
        "after_statistical_filter": after_stat_count,
        "after_radius_filter": after_radius_count,
        "after_voxel_downsample": after_voxel_count,
        "final_points": final_point_count,
        "removed_points": removed_point_count,
        "retention_rate": round(retention_rate, 4),
        "retention_percentage": round(retention_rate * 100.0, 2),
        "processing_times_seconds": timing,
        "parameters": {
            "statistical": {
                "enabled": enable_statistical,
                "nb_neighbors": nb_neighbors,
                "std_ratio": std_ratio,
                "type": "provisional_engineering_default",
            },
            "radius": {
                "enabled": enable_radius,
                "radius": radius,
                "min_points": min_points,
            },
            "voxel": {
                "enabled": enable_voxel,
                "voxel_size": voxel_size,
                "voxel_units": current_pcd.units,
            },
            "ground_separation": ground_report,
        },
        "coordinate_system": current_pcd.crs,
        "coordinate_units": current_pcd.units,
        "coordinate_convention": current_pcd.coordinate_convention,
        "bounding_box": {
            "min_xyz": output_validation["xyz_min"],
            "max_xyz": output_validation["xyz_max"],
            "extents": output_validation["extents"],
            "volume": output_validation["bounding_box_volume"],
        },
        "xyz_min": output_validation["xyz_min"],
        "xyz_max": output_validation["xyz_max"],
        "rgb_available": current_pcd.has_colors,
        "stages": {
            "statistical_outliers": stat_report,
            "radius_outliers": radius_report,
            "voxel_downsample": voxel_report,
            "ground_classification": ground_report,
        },
        "validation": {
            "status": "WARNING" if warnings else "PASS",
            "warnings": warnings,
            "input_metrics": input_validation,
            "output_metrics": output_validation,
        },
    }

    # 9. Optional Export
    if output_dir is not None:
        t0 = time.time()
        source_info = {
            "method": current_pcd.source_method,
            "path": current_pcd.source_path,
        }
        export_point_cloud_bundle(
            pcd=current_pcd,
            output_dir=output_dir,
            processing_stats=stats,
            source_info=source_info,
        )
        timing["export"] = round(time.time() - t0, 4)
        stats["processing_times_seconds"]["export"] = timing["export"]

    return current_pcd, stats


def print_summary(stats: Dict[str, Any]) -> None:
    """Prints a clean human-readable summary of the point cloud processing run."""
    print("\n========================================")
    print("POINT-CLOUD PROCESSING SUMMARY")
    print("========================================")
    print(f"Input Points:        {stats['input_points']}")
    print(f"After Statistical:   {stats['after_statistical_filter']}")
    print(f"After Radius:        {stats['after_radius_filter']}")
    print(f"After Voxel:         {stats['after_voxel_downsample']}")
    print(f"Final Points:        {stats['final_points']}")
    print(f"Removed Points:      {stats['removed_points']}")
    print(f"Retention Rate:      {stats['retention_percentage']:.1f}%")
    print("----------------------------------------")
    print(f"Coordinate System:   {stats.get('coordinate_system') or 'None (local unscaled)'}")
    print(f"Coordinate Units:    {stats.get('coordinate_units')}")
    print(f"RGB Colors:          {'Available' if stats.get('rgb_available') else 'Not Available'}")
    print(f"Total Runtime:       {stats['processing_times_seconds'].get('total', 0.0):.3f}s")
    val = stats.get("validation", {})
    print(f"Validation Status:   {val.get('status', 'PASS')}")
    if val.get("warnings"):
        for w in val["warnings"]:
            print(f"  [WARN] {w}")
    print("========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIH26158 Point-Cloud Processing CLI (Step 8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to points3D.txt, points3D.bin, pointcloud.ply, or reconstruction directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to save cleaned pointcloud.ply, metadata.json, and processing_report.json",
    )
    parser.add_argument(
        "--georef-metadata",
        type=Path,
        default=None,
        help="Optional path to Step 7 georeferencing metadata.json to transform coordinates",
    )
    parser.add_argument(
        "--nb-neighbors",
        type=int,
        default=DEFAULT_NB_NEIGHBORS,
        help="Number of nearest neighbours for statistical outlier removal (engineering default: 20)",
    )
    parser.add_argument(
        "--std-ratio",
        type=float,
        default=DEFAULT_STD_RATIO,
        help="Standard deviation threshold for statistical outlier removal (engineering default: 2.0)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Radius distance for radius-based outlier filtering in cloud coordinate units",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=DEFAULT_RADIUS_MIN_POINTS,
        help="Minimum required neighbours within radius for radius outlier filtering",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel size for downsampling (in coordinate units: metres if georeferenced, local units otherwise)",
    )
    parser.add_argument(
        "--disable-statistical",
        action="store_true",
        help="Disable statistical outlier removal stage",
    )
    parser.add_argument(
        "--disable-radius",
        action="store_true",
        help="Explicitly disable radius outlier filtering (disabled by default)",
    )
    parser.add_argument(
        "--disable-voxel",
        action="store_true",
        help="Explicitly disable voxel downsampling (disabled by default)",
    )
    parser.add_argument(
        "--ground-filter",
        action="store_true",
        help="Enable experimental ground plane analysis (disabled by default)",
    )

    args = parser.parse_args()

    # Determine stage enablement
    enable_stat = not args.disable_statistical
    enable_rad = (args.radius is not None) and (not args.disable_radius)
    enable_vox = (args.voxel_size is not None) and (not args.disable_voxel)

    try:
        _, stats = run_point_cloud_pipeline(
            input_cloud=args.input,
            output_dir=args.output,
            georef_metadata_path=args.georef_metadata,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
            enable_statistical=enable_stat,
            radius=args.radius,
            min_points=args.min_points,
            enable_radius=enable_rad,
            voxel_size=args.voxel_size,
            enable_voxel=enable_vox,
            enable_ground=args.ground_filter,
        )
        print_summary(stats)
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] Point-cloud processing failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
