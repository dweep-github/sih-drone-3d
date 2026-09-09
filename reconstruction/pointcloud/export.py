"""Point cloud loading and export module (SIH26158 Step 8).

Handles:
- Parsing COLMAP points3D (text and binary formats)
- Applying Step 7 georeferencing similarity transformation when available
- Preserving RGB colors, point IDs, and CRS metadata
- Writing standard PLY files
- Generating metadata.json and processing_report.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("pointcloud.export")


@dataclass
class PointCloudData:
    """Represents a 3D point cloud with optional colors, IDs, and georeference metadata."""

    points: np.ndarray  # (N, 3) float64
    colors: Optional[np.ndarray] = None  # (N, 3) uint8 [0..255]
    point_ids: Optional[np.ndarray] = None  # (N,) int64
    errors: Optional[np.ndarray] = None  # (N,) float64 reprojection errors
    source_method: str = "unknown"
    source_path: Optional[str] = None
    crs: Optional[str] = None  # e.g., "EPSG:32643" or None
    units: str = "local_unscaled"  # "metres" or "local_unscaled"
    coordinate_convention: str = "COLMAP local reconstruction coordinates"
    is_georeferenced: bool = False

    @property
    def num_points(self) -> int:
        return len(self.points) if self.points is not None else 0

    @property
    def has_colors(self) -> bool:
        return self.colors is not None and len(self.colors) == self.num_points

    def copy(self) -> PointCloudData:
        """Returns a deep copy of PointCloudData."""
        return PointCloudData(
            points=self.points.copy() if self.points is not None else np.empty((0, 3)),
            colors=self.colors.copy() if self.colors is not None else None,
            point_ids=self.point_ids.copy() if self.point_ids is not None else None,
            errors=self.errors.copy() if self.errors is not None else None,
            source_method=self.source_method,
            source_path=self.source_path,
            crs=self.crs,
            units=self.units,
            coordinate_convention=self.coordinate_convention,
            is_georeferenced=self.is_georeferenced,
        )

    def to_open3d(self) -> Any:
        """Converts to an Open3D PointCloud object."""
        try:
            import open3d as o3d
        except ImportError as err:
            raise RuntimeError("Open3D is required for point-cloud geometry processing.") from err

        pcd = o3d.geometry.PointCloud()
        if self.num_points > 0:
            pcd.points = o3d.utility.Vector3dVector(self.points)
            if self.has_colors:
                # Open3D expects colors normalized to [0.0, 1.0]
                norm_colors = self.colors.astype(np.float64) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(norm_colors)
        return pcd

    @classmethod
    def from_open3d(
        cls,
        o3d_pcd: Any,
        source_data: Optional[PointCloudData] = None,
        inlier_indices: Optional[List[int] | np.ndarray] = None,
    ) -> PointCloudData:
        """Constructs PointCloudData from an Open3D PointCloud object."""
        pts = np.asarray(o3d_pcd.points, dtype=np.float64)
        if len(o3d_pcd.colors) > 0:
            raw_colors = np.asarray(o3d_pcd.colors, dtype=np.float64)
            # Clip and scale to [0, 255] uint8
            cols = np.clip(raw_colors * 255.0, 0, 255).astype(np.uint8)
        else:
            cols = None

        new_ids = None
        new_errors = None

        if source_data is not None:
            if inlier_indices is not None and len(inlier_indices) == len(pts):
                idx = np.asarray(inlier_indices, dtype=np.int64)
                if source_data.point_ids is not None and len(source_data.point_ids) > 0:
                    new_ids = source_data.point_ids[idx]
                if source_data.errors is not None and len(source_data.errors) > 0:
                    new_errors = source_data.errors[idx]

            return cls(
                points=pts,
                colors=cols,
                point_ids=new_ids,
                errors=new_errors,
                source_method=source_data.source_method,
                source_path=source_data.source_path,
                crs=source_data.crs,
                units=source_data.units,
                coordinate_convention=source_data.coordinate_convention,
                is_georeferenced=source_data.is_georeferenced,
            )

        return cls(points=pts, colors=cols)


def load_colmap_points3d_txt(file_path: Union[str, Path]) -> PointCloudData:
    """Parses COLMAP points3D.txt.

    Format per point:
        POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty, malformed, or coordinates are non-finite.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"points3D.txt not found at: {path}")

    point_ids: List[int] = []
    xyz_list: List[List[float]] = []
    rgb_list: List[List[int]] = []
    error_list: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                raise ValueError(
                    f"Malformed line {line_num} in {path.name}: expected at least 8 elements, got {len(parts)}"
                )

            try:
                pid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                r = int(parts[4])
                g = int(parts[5])
                b = int(parts[6])
                err = float(parts[7])
            except ValueError as exc:
                raise ValueError(f"Non-numeric coordinate or color on line {line_num} in {path.name}: {exc}") from exc

            # Validate finite coordinates
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                raise ValueError(
                    f"Non-finite XYZ coordinate on line {line_num} in {path.name}: [{x}, {y}, {z}]"
                )

            # Validate RGB bounds
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError(f"RGB color out of range [0, 255] on line {line_num} in {path.name}: [{r}, {g}, {b}]")

            point_ids.append(pid)
            xyz_list.append([x, y, z])
            rgb_list.append([r, g, b])
            error_list.append(err)

    if len(xyz_list) == 0:
        raise ValueError(f"Point cloud file '{path.name}' is empty (0 valid 3D points).")

    points = np.asarray(xyz_list, dtype=np.float64)
    colors = np.asarray(rgb_list, dtype=np.uint8)
    p_ids = np.asarray(point_ids, dtype=np.int64)
    errors = np.asarray(error_list, dtype=np.float64)

    return PointCloudData(
        points=points,
        colors=colors,
        point_ids=p_ids,
        errors=errors,
        source_method="colmap_sparse_text",
        source_path=str(path),
        crs=None,
        units="local_unscaled",
        coordinate_convention="COLMAP local reconstruction coordinates",
        is_georeferenced=False,
    )


def load_colmap_points3d_bin(file_path: Union[str, Path]) -> PointCloudData:
    """Parses COLMAP points3D.bin.

    Reuses binary model parser from reconstruction.validation.model_io.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or contains non-finite coordinates.
    """
    from reconstruction.validation.model_io import read_points3D_binary

    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"points3D.bin not found at: {path}")

    points3D_dict = read_points3D_binary(path)
    if not points3D_dict:
        raise ValueError(f"Point cloud file '{path.name}' is empty (0 valid 3D points).")

    point_ids: List[int] = []
    xyz_list: List[np.ndarray] = []
    rgb_list: List[np.ndarray] = []
    error_list: List[float] = []

    for pid, p in points3D_dict.items():
        if not np.all(np.isfinite(p.xyz)):
            raise ValueError(f"Non-finite XYZ coordinate in {path.name} for point ID {pid}: {p.xyz}")
        point_ids.append(pid)
        xyz_list.append(p.xyz)
        rgb_list.append(p.rgb)
        error_list.append(p.error)

    points = np.asarray(xyz_list, dtype=np.float64)
    colors = np.asarray(rgb_list, dtype=np.uint8)
    p_ids = np.asarray(point_ids, dtype=np.int64)
    errors = np.asarray(error_list, dtype=np.float64)

    return PointCloudData(
        points=points,
        colors=colors,
        point_ids=p_ids,
        errors=errors,
        source_method="colmap_sparse_binary",
        source_path=str(path),
        crs=None,
        units="local_unscaled",
        coordinate_convention="COLMAP local reconstruction coordinates",
        is_georeferenced=False,
    )


def apply_georeferencing_transform(
    pcd: PointCloudData,
    georef_metadata: Dict[str, Any],
) -> PointCloudData:
    """Applies Step 7 similarity transformation to transform point coordinates to metric CRS.

    Transformation: X_geo = s * (P @ R^T) + t
    where s is scale, R is 3x3 rotation matrix, t is translation vector.

    Args:
        pcd: Local unscaled PointCloudData
        georef_metadata: Metadata dictionary from Step 7 georeferencing containing 'alignment'
                         and 'target_crs'.

    Returns:
        New PointCloudData in georeferenced metric coordinate system.
    """
    alignment = georef_metadata.get("alignment")
    target_crs = georef_metadata.get("target_crs")
    coord_sys_name = georef_metadata.get("coordinate_system", target_crs or "Projected Metric")

    if not alignment:
        logger.warning("Georeferencing metadata does not contain 'alignment' section; preserving local coordinates.")
        return pcd

    try:
        scale = float(alignment["scale"])
        R = np.asarray(alignment["rotation_matrix"], dtype=np.float64)
        t = np.asarray(alignment["translation_vector"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed alignment parameters in georeferencing metadata: {exc}") from exc

    if scale <= 0 or not np.isfinite(scale):
        raise ValueError(f"Invalid georeferencing scale factor: {scale}")

    # Transform coordinates: X_geo = s * (P @ R^T) + t
    P = pcd.points
    geo_points = (scale * (P @ R.T)) + t

    new_pcd = pcd.copy()
    new_pcd.points = geo_points
    new_pcd.crs = target_crs
    new_pcd.units = "metres"
    new_pcd.coordinate_convention = f"Georeferenced projected metric coordinates ({coord_sys_name})"
    new_pcd.is_georeferenced = True

    logger.info(
        "Applied georeferencing similarity transform: scale=%.6f, target_crs=%s (%d points)",
        scale,
        target_crs,
        new_pcd.num_points,
    )
    return new_pcd


def load_point_cloud(
    input_path: Union[str, Path],
    georef_metadata_path: Optional[Union[str, Path]] = None,
    source_method: Optional[str] = None,
) -> PointCloudData:
    """Unified point-cloud loader supporting files or reconstruction directories.

    Args:
        input_path: Path to points3D.txt, points3D.bin, .ply, or reconstruction directory.
        georef_metadata_path: Optional path to Step 7 georeferencing metadata.json.
        source_method: Optional label for source reconstruction method.

    Returns:
        PointCloudData loaded and optionally georeferenced.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Point cloud input path not found: {path}")

    # Handle directory input
    if path.is_dir():
        txt_candidate = path / "sparse" / "points3D.txt"
        bin_candidate = path / "sparse" / "points3D.bin"
        direct_txt = path / "points3D.txt"
        direct_bin = path / "points3D.bin"
        ply_candidate = path / "pointcloud.ply"

        if txt_candidate.is_file():
            target_file = txt_candidate
        elif bin_candidate.is_file():
            target_file = bin_candidate
        elif direct_txt.is_file():
            target_file = direct_txt
        elif direct_bin.is_file():
            target_file = direct_bin
        elif ply_candidate.is_file():
            target_file = ply_candidate
        else:
            raise FileNotFoundError(
                f"No points3D.txt, points3D.bin, or pointcloud.ply found in {path}"
            )

        # Auto-detect georeferencing metadata in directory if not explicitly provided
        if georef_metadata_path is None:
            auto_geo_meta = path / "georeferenced" / "metadata.json"
            if auto_geo_meta.is_file():
                georef_metadata_path = auto_geo_meta
    else:
        target_file = path
        # If pointing to selected/sparse/points3D.txt, check selected/georeferenced/metadata.json
        if georef_metadata_path is None:
            parent_dir = target_file.parent
            if parent_dir.name == "sparse":
                candidate = parent_dir.parent / "georeferenced" / "metadata.json"
                if candidate.is_file():
                    georef_metadata_path = candidate

    # Load by file extension
    suffix = target_file.suffix.lower()
    if suffix == ".txt":
        pcd = load_colmap_points3d_txt(target_file)
    elif suffix == ".bin":
        pcd = load_colmap_points3d_bin(target_file)
    elif suffix == ".ply":
        pcd = _load_ply_file(target_file)
    else:
        # Fall back to trying txt then bin
        try:
            pcd = load_colmap_points3d_txt(target_file)
        except Exception:
            pcd = load_colmap_points3d_bin(target_file)

    if source_method:
        pcd.source_method = source_method

    # Apply georeferencing if metadata is found/provided
    if georef_metadata_path is not None:
        meta_p = Path(georef_metadata_path).resolve()
        if meta_p.is_file():
            with open(meta_p, "r", encoding="utf-8") as f:
                geo_meta = json.load(f)
            pcd = apply_georeferencing_transform(pcd, geo_meta)
        else:
            logger.warning("Georeference metadata path provided but does not exist: %s", meta_p)

    return pcd


def _load_ply_file(file_path: Path) -> PointCloudData:
    """Loads a point cloud from PLY format via Open3D."""
    try:
        import open3d as o3d
    except ImportError as err:
        raise RuntimeError("Open3D is required for PLY loading.") from err

    o3d_pcd = o3d.io.read_point_cloud(str(file_path))
    if len(o3d_pcd.points) == 0:
        raise ValueError(f"PLY file '{file_path.name}' is empty (0 points).")

    pts = np.asarray(o3d_pcd.points, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"PLY file '{file_path.name}' contains non-finite coordinates.")

    cols = None
    if len(o3d_pcd.colors) > 0:
        raw_colors = np.asarray(o3d_pcd.colors, dtype=np.float64)
        cols = np.clip(raw_colors * 255.0, 0, 255).astype(np.uint8)

    return PointCloudData(
        points=pts,
        colors=cols,
        source_method="ply_import",
        source_path=str(file_path),
        crs=None,
        units="local_unscaled",
        coordinate_convention="PLY local coordinates",
        is_georeferenced=False,
    )


def export_point_cloud_ply(
    pcd: PointCloudData,
    output_path: Union[str, Path],
    write_ascii: bool = False,
) -> Path:
    """Exports PointCloudData to a standard PLY file with RGB color support.

    Args:
        pcd: PointCloudData instance
        output_path: Target path for .ply file
        write_ascii: If True writes ASCII PLY; otherwise binary PLY.

    Returns:
        Resolved Path to written PLY file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if pcd.num_points == 0:
        raise ValueError("Cannot export empty point cloud (0 points) to PLY.")

    try:
        import open3d as o3d
        o3d_pcd = pcd.to_open3d()
        success = o3d.io.write_point_cloud(
            str(out_file),
            o3d_pcd,
            write_ascii=write_ascii,
            compressed=False,
        )
        if not success:
            raise RuntimeError(f"Open3D returned False when writing PLY to {out_file}")
    except Exception as exc:
        logger.warning("Open3D PLY export failed (%s); using direct PLY writer fallback.", exc)
        _write_ply_direct(pcd, out_file, ascii_mode=write_ascii)

    return out_file


def _write_ply_direct(pcd: PointCloudData, out_path: Path, ascii_mode: bool = True) -> None:
    """Direct standalone PLY writer fallback without third-party dependencies."""
    n_pts = pcd.num_points
    has_colors = pcd.has_colors

    header = [
        "ply",
        "format ascii 1.0" if ascii_mode else "format binary_little_endian 1.0",
        f"comment Exported by SIH26158 PointCloud Module",
        f"element vertex {n_pts}",
        "property double x",
        "property double y",
        "property double z",
    ]
    if has_colors:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])
    header.append("end_header\n")
    header_str = "\n".join(header)

    if ascii_mode:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header_str)
            for i in range(n_pts):
                x, y, z = pcd.points[i]
                if has_colors:
                    r, g, b = pcd.colors[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    else:
        import struct
        with open(out_path, "wb") as f:
            f.write(header_str.encode("utf-8"))
            for i in range(n_pts):
                x, y, z = pcd.points[i]
                f.write(struct.pack("<3d", x, y, z))
                if has_colors:
                    r, g, b = pcd.colors[i]
                    f.write(struct.pack("<3B", int(r), int(g), int(b)))


def export_point_cloud_bundle(
    pcd: PointCloudData,
    output_dir: Union[str, Path],
    processing_stats: Dict[str, Any],
    source_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Exports standardized Step 8 bundle:

    output/pointcloud/
    ├── pointcloud.ply
    ├── metadata.json
    └── processing_report.json

    Returns:
        Dict mapping keys 'ply', 'metadata', 'processing_report' to their Paths.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = out_dir / "pointcloud.ply"
    meta_path = out_dir / "metadata.json"
    report_path = out_dir / "processing_report.json"

    # 1. Export PLY
    export_point_cloud_ply(pcd, ply_path)

    # 2. Build metadata conforming strictly to Step 8 Section 14
    src_info = source_info or {}
    source_method = src_info.get("method", pcd.source_method)
    source_path = src_info.get("path", pcd.source_path)

    params = processing_stats.get("parameters", {})
    stat_enabled = params.get("statistical", {}).get("enabled", True)
    radius_enabled = params.get("radius", {}).get("enabled", False)
    voxel_enabled = params.get("voxel", {}).get("enabled", False)

    input_count = processing_stats.get("input_points", pcd.num_points)
    output_count = processing_stats.get("final_points", pcd.num_points)

    metadata: Dict[str, Any] = {
        "source": {
            "method": source_method,
            "path": source_path,
        },
        "coordinate_system": {
            "crs": pcd.crs if pcd.is_georeferenced else None,
            "units": pcd.units,
            "coordinate_convention": pcd.coordinate_convention,
        },
        "processing": {
            "statistical_outlier_removal": stat_enabled,
            "radius_outlier_removal": radius_enabled,
            "voxel_downsampling": voxel_enabled,
        },
        "point_counts": {
            "input": input_count,
            "output": output_count,
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Write detailed processing report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(processing_stats, f, indent=2)

    logger.info("Point cloud bundle exported successfully to %s", out_dir)
    return {
        "ply": ply_path,
        "metadata": meta_path,
        "processing_report": report_path,
    }
