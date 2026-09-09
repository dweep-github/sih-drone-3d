"""GPS Data Loader, Validator, and Pose Synchronization Module.

Parses GPS CSV files, enforces column and coordinate constraints, validates altitude references,
detects duplicate/missing values, sorts chronologically, and performs nearest-neighbor
timestamp synchronization with reconstruction camera poses.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

logger = logging.getLogger("georeferencing.gps")


@dataclass
class GPSRecord:
    """Individual validated GPS observation."""
    timestamp: float  # Unix epoch timestamp in seconds
    latitude: float   # WGS84 latitude in degrees [-90.0, 90.0]
    longitude: float  # WGS84 longitude in degrees [-180.0, 180.0]
    altitude: float   # Height above reference in meters
    raw_timestamp: str = ""
    altitude_reference: str = "WGS84_ellipsoidal"


def parse_timestamp(val: Any) -> float:
    """Parses a timestamp value into Unix epoch seconds (float).

    Supports:
    - Numeric values (float, int representing Unix epoch seconds)
    - ISO8601 strings (e.g., '2026-09-07T14:30:00.123Z', '2026-09-07 14:30:00')
    - Standard datetime formats
    """
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        raise ValueError("Empty timestamp string")

    # Try direct numeric conversion
    try:
        return float(s)
    except ValueError:
        pass

    # Try ISO8601 / common date-time patterns
    clean_s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean_s)
        if dt.tzinfo is None:
            # Assume UTC if naive
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    raise ValueError(f"Unsupported timestamp format: '{val}'")


def load_gps_csv(
    csv_path: str | Path,
    altitude_reference: str = "WGS84_ellipsoidal",
) -> Tuple[List[GPSRecord], Dict[str, Any]]:
    """Loads and validates a GPS CSV file.

    Requirements:
    - Must contain columns for timestamp, latitude, longitude, and altitude.
    - Latitude in [-90.0, 90.0], Longitude in [-180.0, 180.0].
    - Altitude must be finite.
    - Sorts chronologically.
    - Detects and reports duplicate timestamps and missing values.
    """
    path = Path(csv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GPS CSV file not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise ValueError(f"GPS CSV file is completely empty: {path}")

    headers_lower = [h.strip().lower() for h in raw_headers]

    # Column mapping aliases
    col_aliases = {
        "timestamp": ["timestamp", "time", "utc_time", "datetime", "epoch", "t"],
        "latitude": ["latitude", "lat", "y"],
        "longitude": ["longitude", "lon", "lng", "long", "x"],
        "altitude": ["altitude", "alt", "height", "elevation", "z"],
    }

    col_idx: Dict[str, int] = {}
    for col_name, aliases in col_aliases.items():
        found = False
        for alias in aliases:
            if alias in headers_lower:
                col_idx[col_name] = headers_lower.index(alias)
                found = True
                break
        if not found:
            raise ValueError(
                f"GPS CSV missing required column '{col_name}'. Found headers: {raw_headers}. "
                f"Expected one of {aliases}."
            )

    records: List[GPSRecord] = []
    warnings: List[str] = []
    errors: List[str] = []
    seen_timestamps: Set[float] = set()
    duplicate_count = 0
    missing_value_rows = 0

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row_num, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue

            max_needed = max(col_idx.values())
            if len(row) <= max_needed:
                errors.append(f"Row {row_num}: incomplete columns (expected > {max_needed}, got {len(row)})")
                continue

            raw_t = row[col_idx["timestamp"]].strip()
            raw_lat = row[col_idx["latitude"]].strip()
            raw_lon = row[col_idx["longitude"]].strip()
            raw_alt = row[col_idx["altitude"]].strip()

            if any(v.lower() in ["", "nan", "null", "none"] for v in [raw_t, raw_lat, raw_lon, raw_alt]):
                missing_value_rows += 1
                warnings.append(f"Row {row_num}: skipped due to missing or null values")
                continue

            try:
                t_val = parse_timestamp(raw_t)
            except ValueError as exc:
                errors.append(f"Row {row_num}: invalid timestamp '{raw_t}': {exc}")
                continue

            try:
                lat_val = float(raw_lat)
                if not (-90.0 <= lat_val <= 90.0):
                    errors.append(f"Row {row_num}: latitude out of range [-90, 90]: {lat_val}")
                    continue
            except ValueError:
                errors.append(f"Row {row_num}: invalid numeric latitude '{raw_lat}'")
                continue

            try:
                lon_val = float(raw_lon)
                if not (-180.0 <= lon_val <= 180.0):
                    errors.append(f"Row {row_num}: longitude out of range [-180, 180]: {lon_val}")
                    continue
            except ValueError:
                errors.append(f"Row {row_num}: invalid numeric longitude '{raw_lon}'")
                continue

            try:
                alt_val = float(raw_alt)
                if not math.isfinite(alt_val):
                    errors.append(f"Row {row_num}: altitude is not finite: {alt_val}")
                    continue
            except ValueError:
                errors.append(f"Row {row_num}: invalid numeric altitude '{raw_alt}'")
                continue

            if t_val in seen_timestamps:
                duplicate_count += 1
                warnings.append(f"Row {row_num}: duplicate timestamp detected ({t_val})")
            else:
                seen_timestamps.add(t_val)

            records.append(GPSRecord(
                timestamp=t_val,
                latitude=lat_val,
                longitude=lon_val,
                altitude=alt_val,
                raw_timestamp=raw_t,
                altitude_reference=altitude_reference,
            ))

    if not records:
        err_summary = "; ".join(errors[:5])
        raise ValueError(f"No valid GPS records could be loaded from '{path}'. Errors: {err_summary}")

    # Sort chronologically
    records.sort(key=lambda r: r.timestamp)

    metadata = {
        "source_file": str(path),
        "total_valid_records": len(records),
        "duplicate_timestamps": duplicate_count,
        "missing_value_rows": missing_value_rows,
        "altitude_reference": altitude_reference,
        "time_start": records[0].timestamp,
        "time_end": records[-1].timestamp,
        "duration_seconds": round(records[-1].timestamp - records[0].timestamp, 3),
        "warnings": warnings,
        "errors": errors,
    }

    return records, metadata


def extract_timestamp_from_frame(
    frame_dict: Dict[str, Any],
    timestamps_map: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Extracts a Unix epoch timestamp from a camera frame dictionary or lookup table."""
    frame_id = str(frame_dict.get("frame_id", ""))
    img_name = str(frame_dict.get("image_name", ""))

    # 1. Lookup table
    if timestamps_map:
        if frame_id in timestamps_map:
            return float(timestamps_map[frame_id])
        if img_name in timestamps_map:
            return float(timestamps_map[img_name])

    # 2. Direct key in dictionary
    if "timestamp" in frame_dict and frame_dict["timestamp"] is not None:
        try:
            return parse_timestamp(frame_dict["timestamp"])
        except ValueError:
            pass

    # 3. Regex search for timestamp in image filename or frame ID
    # Pattern 1: float timestamp like 1725712345.123
    match = re.search(r"(\d{10}(?:\.\d+)?)", img_name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def synchronize_poses_with_gps(
    poses: List[Dict[str, Any]],
    gps_records: List[GPSRecord],
    max_time_diff_s: float = 1.0,
    frame_timestamps: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synchronizes camera poses with GPS observations using nearest-neighbor timestamp matching.

    Matches each frame to the closest GPS record if the difference is within max_time_diff_s.
    Reports matched observations, unmatched poses, unmatched GPS records, and synchronization metrics.
    """
    if not gps_records:
        raise ValueError("Cannot synchronize poses: GPS records list is empty")
    if not poses:
        raise ValueError("Cannot synchronize poses: camera poses list is empty")

    gps_times = np.array([r.timestamp for r in gps_records], dtype=np.float64)
    matched_pairs: List[Dict[str, Any]] = []
    unmatched_poses: List[Dict[str, Any]] = []
    matched_gps_indices: Set[int] = set()
    time_diffs: List[float] = []

    for idx, pose in enumerate(poses):
        frame_id = pose.get("frame_id", f"frame_{idx:06d}")
        pose_t = extract_timestamp_from_frame(pose, frame_timestamps)

        if pose_t is None:
            unmatched_poses.append({
                "frame_id": frame_id,
                "image_name": pose.get("image_name", ""),
                "reason": "Timestamp could not be determined for frame",
            })
            continue

        # Nearest neighbor search
        diffs = np.abs(gps_times - pose_t)
        nearest_idx = int(np.argmin(diffs))
        min_diff = float(diffs[nearest_idx])

        if min_diff <= max_time_diff_s:
            gps_match = gps_records[nearest_idx]
            matched_gps_indices.add(nearest_idx)
            time_diffs.append(min_diff)

            matched_pairs.append({
                "frame_id": frame_id,
                "image_name": pose.get("image_name", ""),
                "local_position": pose.get("position", [0.0, 0.0, 0.0]),
                "local_rotation_quaternion": pose.get("rotation_quaternion", [1.0, 0.0, 0.0, 0.0]),
                "pose_timestamp": pose_t,
                "gps_timestamp": gps_match.timestamp,
                "time_difference_s": round(min_diff, 4),
                "latitude": gps_match.latitude,
                "longitude": gps_match.longitude,
                "altitude": gps_match.altitude,
                "altitude_reference": gps_match.altitude_reference,
            })
        else:
            unmatched_poses.append({
                "frame_id": frame_id,
                "image_name": pose.get("image_name", ""),
                "pose_timestamp": pose_t,
                "nearest_gps_timestamp": float(gps_times[nearest_idx]),
                "time_difference_s": round(min_diff, 4),
                "reason": f"Nearest GPS record exceeds tolerance ({min_diff:.4f}s > {max_time_diff_s:.4f}s)",
            })

    unmatched_gps_count = len(gps_records) - len(matched_gps_indices)
    mean_diff = float(np.mean(time_diffs)) if time_diffs else 0.0
    max_diff = float(np.max(time_diffs)) if time_diffs else 0.0

    sync_summary = {
        "total_poses": len(poses),
        "total_gps_records": len(gps_records),
        "matched_observations": len(matched_pairs),
        "unmatched_poses_count": len(unmatched_poses),
        "unmatched_gps_count": unmatched_gps_count,
        "max_time_tolerance_s": max_time_diff_s,
        "observed_mean_time_diff_s": round(mean_diff, 4),
        "observed_max_time_diff_s": round(max_diff, 4),
        "synchronization_rate": round(len(matched_pairs) / max(len(poses), 1), 4),
        "unmatched_poses": unmatched_poses,
    }

    return matched_pairs, sync_summary
