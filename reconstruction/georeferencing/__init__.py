"""Georeferencing Module for Drone 3D Reconstruction (SIH26158 Person 1).

Exposes:
- GPS loader, validator, and timestamp synchronization (GPSRecord, load_gps_csv, synchronize_poses_with_gps)
- Coordinate transformation between WGS84 and metric UTM/projected systems (CoordinateTransformer, determine_utm_crs)
- Umeyama 3D similarity alignment and camera pose transformation (align_umeyama, transform_camera_pose, SimilarityTransform)
- End-to-end georeferencing pipeline runner (run_georeferencing)
"""

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
    load_imu_data,
    run_georeferencing,
)
from reconstruction.georeferencing.gps import (
    GPSRecord,
    extract_timestamp_from_frame,
    load_gps_csv,
    parse_timestamp,
    synchronize_poses_with_gps,
)

__all__ = [
    "GPSRecord",
    "load_gps_csv",
    "parse_timestamp",
    "extract_timestamp_from_frame",
    "synchronize_poses_with_gps",
    "determine_utm_crs",
    "CoordinateTransformer",
    "SimilarityTransform",
    "check_point_set_validity",
    "align_umeyama",
    "transform_camera_pose",
    "load_imu_data",
    "run_georeferencing",
]
