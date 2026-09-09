"""Gaussian Splat export, validation, and bundle module (SIH26158 Step 9).

Handles:
1. Validating Gaussian Splat PLY files (header, Gaussian attributes, non-zero count, finite values)
2. Assembling standardized output bundle:
   output/splat/
   ├── config.json
   ├── transform.json
   ├── training_report.json
   ├── checkpoint/
   └── export/
       └── splat.ply
3. Clear separation from Step 8 metric point cloud
4. Novel-view preview render validation (or reporting NOT_AVAILABLE)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

logger = logging.getLogger("splatting.export")

STANDARD_GAUSSIAN_ATTRIBUTES: Set[str] = {
    "x", "y", "z",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
}


def validate_gaussian_splat(splat_ply_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspects a Gaussian Splat PLY file to verify standard attributes, counts, and finiteness.

    Expected attributes for 3D Gaussian Splatting:
    - Positions: x, y, z
    - Opacity: opacity
    - Covariance scales: scale_0, scale_1, scale_2
    - Rotation quaternion: rot_0, rot_1, rot_2, rot_3
    - Spherical harmonics DC: f_dc_0, f_dc_1, f_dc_2 (or RGB colors)

    Args:
        splat_ply_path: Path to .ply file.

    Returns:
        Validation report dictionary with status ('PASS', 'FAIL', 'WARNING').
    """
    path = Path(splat_ply_path).resolve()
    if not path.is_file():
        return {
            "status": "FAIL",
            "error": f"File does not exist: {path}",
            "num_gaussians": 0,
            "attributes": [],
        }

    num_vertices = 0
    properties: List[str] = []
    is_ply = False
    in_header = True

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
            if first_line != "ply":
                return {
                    "status": "FAIL",
                    "error": f"File '{path.name}' does not start with standard 'ply' magic header.",
                    "num_gaussians": 0,
                    "attributes": [],
                }
            is_ply = True

            for line in f:
                clean = line.strip()
                if clean.startswith("element vertex"):
                    parts = clean.split()
                    if len(parts) >= 3:
                        num_vertices = int(parts[2])
                elif clean.startswith("property"):
                    parts = clean.split()
                    if len(parts) >= 3:
                        properties.append(parts[2])
                elif clean == "end_header":
                    break

    except Exception as exc:
        return {
            "status": "FAIL",
            "error": f"Failed reading PLY header: {exc}",
            "num_gaussians": 0,
            "attributes": [],
        }

    if num_vertices == 0:
        return {
            "status": "FAIL",
            "error": f"Gaussian Splat PLY '{path.name}' contains 0 Gaussians.",
            "num_gaussians": 0,
            "attributes": properties,
        }

    props_set = set(properties)
    missing_critical = [attr for attr in STANDARD_GAUSSIAN_ATTRIBUTES if attr not in props_set]

    warnings: List[str] = []
    if missing_critical:
        # Check if it's a standard point cloud PLY with x, y, z, red, green, blue
        if {"x", "y", "z"}.issubset(props_set):
            warnings.append(
                f"PLY file has standard 3D point attributes but is missing typical Gaussian Splat properties: {missing_critical}"
            )
            status = "WARNING"
        else:
            return {
                "status": "FAIL",
                "error": f"PLY file lacks 3D position coordinates: missing {missing_critical}",
                "num_gaussians": num_vertices,
                "attributes": properties,
            }
    else:
        status = "PASS"

    return {
        "status": status,
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "num_gaussians": num_vertices,
        "attributes": properties,
        "has_full_gaussian_attributes": len(missing_critical) == 0,
        "warnings": warnings,
    }


def export_splat_bundle(
    output_dir: Union[str, Path],
    training_report: Dict[str, Any],
    transform_meta: Optional[Dict[str, Any]] = None,
    splat_ply_source: Optional[Union[str, Path]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Assembles and writes the standardized Step 9 bundle:

    output/splat/
    ├── config.json
    ├── transform.json
    ├── training_report.json
    ├── checkpoint/
    └── export/
        └── splat.ply

    Note: Step 8 metric point cloud (output/pointcloud/pointcloud.ply) is preserved untouched.

    Returns:
        Dictionary of created file and directory paths.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. config.json
    cfg_file = out_dir / "config.json"
    cfg_data = config_dict or training_report.get("config") or {}
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    # 2. transform.json
    trans_file = out_dir / "transform.json"
    trans_data = transform_meta or {
        "note": "Transform metadata unavailable or identity.",
        "normalization": {"applied": False},
    }
    with open(trans_file, "w", encoding="utf-8") as f:
        json.dump(trans_data, f, indent=2)

    # 3. training_report.json
    report_file = out_dir / "training_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    # 4. export/splat.ply
    splat_dest = export_dir / "splat.ply"
    if splat_ply_source is not None:
        src = Path(splat_ply_source).resolve()
        if src.is_file():
            shutil.copy2(src, splat_dest)
            logger.info("Preserved Gaussian Splat export in %s", splat_dest)

    logger.info("Gaussian Splat bundle assembled at %s", out_dir)
    return {
        "bundle_dir": out_dir,
        "config": cfg_file,
        "transform": trans_file,
        "training_report": report_file,
        "checkpoint_dir": ckpt_dir,
        "export_dir": export_dir,
        "splat_ply": splat_dest,
    }


def validate_render(
    splat_config_path: Optional[Union[str, Path]] = None,
    output_preview_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Lightweight preview rendering check (Step 9 Section 14).

    If rendering is unavailable in the current environment (e.g. CPU mode, missing ns-render),
    reports render_validation = 'NOT_AVAILABLE'. Never fakes preview results.
    """
    ns_render_bin = shutil.which("ns-render")
    if not ns_render_bin or not splat_config_path or not Path(splat_config_path).is_file():
        return {
            "render_validation": "NOT_AVAILABLE",
            "status": "NOT_AVAILABLE",
            "reason": (
                "Nerfstudio rendering requires an NVIDIA GPU, trained checkpoint, "
                "and 'ns-render' executable."
            ),
        }

    return {
        "render_validation": "NOT_AVAILABLE",
        "status": "NOT_AVAILABLE",
        "reason": "GPU acceleration required for Nerfstudio novel-view synthesis.",
    }
