"""COLMAP Global SfM (global_mapper) integration for Person 1 (SIH26158).

Provides a clean, modular interface to run the native global_mapper pipeline
in current COLMAP versions (>=4.0), including:
- Input image and database validation
- Feature extraction and sequential/exhaustive matching
- Optional view-graph calibration prior to global positioning
- COLMAP global_mapper execution (rotation averaging + global positioning + bundle adjustment)
- Ingestion into the existing Step 2 photogrammetric validation engine
- Output report export (global_sfm_report.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.colmap.feature_extraction import extract_features, validate_image_dir
from reconstruction.colmap.feature_matching import match_features
from reconstruction.colmap.utils import (
    find_colmap_binary,
    get_colmap_version,
    is_cuda_available,
    run_colmap,
)
from reconstruction.validation.pose_report import (
    count_input_images,
    find_sparse_model_dir,
    validate_reconstruction,
)

logger = logging.getLogger("global_sfm")


def is_global_mapper_available(colmap_bin: Optional[str | Path] = None) -> bool:
    """Checks whether the installed COLMAP binary supports the 'global_mapper' command."""
    binary = find_colmap_binary(colmap_bin)
    if not binary:
        return False
    try:
        res = run_colmap("global_mapper", ["-h"], colmap_bin=binary)
        output = res.stdout + res.stderr
        return "global_mapper" in output.lower() or "options" in output.lower()
    except Exception as exc:
        logger.debug("global_mapper check failed: %s", exc)
        return False


def is_view_graph_calibrator_available(colmap_bin: Optional[str | Path] = None) -> bool:
    """Checks whether the installed COLMAP binary supports 'view_graph_calibrator'."""
    binary = find_colmap_binary(colmap_bin)
    if not binary:
        return False
    try:
        res = run_colmap("view_graph_calibrator", ["-h"], colmap_bin=binary)
        output = res.stdout + res.stderr
        return "view_graph_calibrator" in output.lower() or "cross-validate" in output.lower()
    except Exception:
        return False


def run_global_sfm(
    images: str | Path,
    output: str | Path,
    camera_model: str = "SIMPLE_RADIAL",
    matcher: str = "sequential",
    overlap: int = 10,
    calibrate_view_graph: bool = False,
    colmap_bin: Optional[str | Path] = None,
    extra_mapper_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs COLMAP global SfM using the native global_mapper pipeline.

    Workflow:
    1. Validates input images.
    2. Initializes isolated output directory and database.
    3. Runs feature extraction and matching.
    4. Optionally runs view_graph_calibrator.
    5. Runs global_mapper.
    6. Ingests model into existing Step 2 validation engine.
    7. Exports global_sfm_report.json.
    """
    start_time = time.time()
    errors: List[str] = []
    warnings: List[str] = []

    img_path = Path(images).resolve()
    out_path = Path(output).resolve()
    db_path = out_path / "database.db"
    sparse_dir = out_path / "sparse"
    report_path = out_path / "global_sfm_report.json"
    pose_report_path = out_path / "pose_report.json"
    traj_path = out_path / "trajectory.json"

    # 1. Validate COLMAP executable and global_mapper availability
    binary = find_colmap_binary(colmap_bin)
    if not binary:
        err = "COLMAP executable could not be found."
        errors.append(err)
        report = {
            "status": "FAIL",
            "method": "colmap_global_sfm",
            "colmap_version": "Not found",
            "input_images": 0,
            "registered_images": 0,
            "registration_rate": 0.0,
            "num_cameras": 0,
            "num_points3D": 0,
            "runtime_seconds": 0.0,
            "reprojection_rmse_px": None,
            "errors": errors,
            "warnings": warnings,
        }
        out_path.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        raise FileNotFoundError(err)

    colmap_ver = get_colmap_version(binary)
    if not is_global_mapper_available(binary):
        err = f"COLMAP binary ({colmap_ver}) does not support the 'global_mapper' command."
        errors.append(err)
        report = {
            "status": "BLOCKED",
            "method": "colmap_global_sfm",
            "colmap_version": colmap_ver,
            "input_images": 0,
            "registered_images": 0,
            "registration_rate": 0.0,
            "num_cameras": 0,
            "num_points3D": 0,
            "runtime_seconds": 0.0,
            "reprojection_rmse_px": None,
            "errors": errors,
            "warnings": warnings,
        }
        out_path.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        raise RuntimeError(err)

    # 2. Validate input images
    valid_images = validate_image_dir(img_path)
    input_count = len(valid_images)

    # 3. Setup output directory & database
    out_path.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    cuda_avail = is_cuda_available()

    # 4. Feature extraction
    logger.info("Extracting features for Global SfM...")
    extract_features(
        image_dir=img_path,
        database_path=db_path,
        camera_model=camera_model,
        single_camera=True,
        use_gpu=cuda_avail,
        colmap_bin=binary,
    )

    # 5. Feature matching
    logger.info("Matching features for Global SfM (%s)...", matcher)
    match_features(
        database_path=db_path,
        matcher=matcher,
        overlap=overlap,
        use_gpu=cuda_avail,
        colmap_bin=binary,
    )

    # 6. Optional View-Graph Calibration
    if calibrate_view_graph:
        if is_view_graph_calibrator_available(binary):
            logger.info("Running view_graph_calibrator...")
            calib_args = ["--database_path", str(db_path)]
            try:
                run_colmap("view_graph_calibrator", calib_args, colmap_bin=binary)
            except Exception as exc:
                warn = f"view_graph_calibrator encountered an issue: {exc}"
                logger.warning(warn)
                warnings.append(warn)
        else:
            warnings.append("view_graph_calibrator is not supported by this COLMAP binary.")

    # 7. Global Mapper Execution
    logger.info("Running COLMAP global_mapper...")
    mapper_args = [
        "--database_path", str(db_path),
        "--image_path", str(img_path),
        "--output_path", str(sparse_dir),
        "--GlobalMapper.gp_use_gpu", "1" if cuda_avail else "0",
        "--GlobalMapper.ba_ceres_use_gpu", "1" if cuda_avail else "0",
    ]
    if extra_mapper_args:
        mapper_args.extend(extra_mapper_args)

    try:
        run_colmap("global_mapper", mapper_args, colmap_bin=binary)
    except Exception as exc:
        err = f"COLMAP global_mapper failed: {exc}"
        logger.error(err)
        errors.append(err)

    elapsed = round(time.time() - start_time, 4)

    # 8. Ingest into Step 2 Validation Engine
    model_dir = find_sparse_model_dir(sparse_dir)
    if model_dir:
        val_report = validate_reconstruction(
            model_dir=model_dir,
            image_dir=img_path,
            output_report_path=pose_report_path,
            trajectory_output_path=traj_path,
        )
        status = val_report.get("validation_status", "FAIL")
        registered_count = val_report.get("input", {}).get("registered_images", 0)
        reg_rate = val_report.get("input", {}).get("registration_rate", 0.0)
        num_cams = val_report.get("reconstruction", {}).get("num_cameras", 0)
        num_pts = val_report.get("reconstruction", {}).get("num_3d_points", 0)
        rmse_px = val_report.get("reprojection", {}).get("rmse_px")
        errors.extend(val_report.get("errors", []))
        warnings.extend(val_report.get("warnings", []))
    else:
        status = "FAIL"
        registered_count = 0
        reg_rate = 0.0
        num_cams = 0
        num_pts = 0
        rmse_px = None
        if not errors:
            errors.append("global_mapper finished but produced no valid sparse model in output directory.")

    report = {
        "status": status,
        "method": "colmap_global_sfm",
        "colmap_version": colmap_ver,
        "input_images": input_count,
        "registered_images": registered_count,
        "registration_rate": reg_rate,
        "num_cameras": num_cams,
        "num_points3D": num_pts,
        "runtime_seconds": elapsed,
        "reprojection_rmse_px": rmse_px,
        "errors": errors,
        "warnings": warnings,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Global SfM finished: Status=%s, Registered=%d/%d", status, registered_count, input_count)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIH26158 Person 1: COLMAP Global SfM (global_mapper) Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", type=Path, default=None, help="Path to input images directory")
    parser.add_argument("--output", type=Path, default=None, help="Path to output directory")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL", help="Camera model")
    parser.add_argument("--matcher", default="sequential", choices=["sequential", "exhaustive", "spatial", "vocab_tree"])
    parser.add_argument("--overlap", type=int, default=10, help="Frame overlap window for sequential matching")
    parser.add_argument("--calibrate-view-graph", action="store_true", help="Run view_graph_calibrator before global mapping")
    parser.add_argument("--colmap-bin", type=Path, default=None, help="Custom COLMAP executable path")
    parser.add_argument("--check-available", action="store_true", help="Verify if global_mapper is supported and exit")

    args = parser.parse_args()

    if args.check_available:
        avail = is_global_mapper_available(args.colmap_bin)
        ver = get_colmap_version(find_colmap_binary(args.colmap_bin))
        print(f"COLMAP Version: {ver}")
        print(f"global_mapper:  {'AVAILABLE' if avail else 'UNAVAILABLE'}")
        sys.exit(0 if avail else 1)

    if not args.images or not args.output:
        parser.error("--images and --output are required unless --check-available is specified.")

    try:
        report = run_global_sfm(
            images=args.images,
            output=args.output,
            camera_model=args.camera_model,
            matcher=args.matcher,
            overlap=args.overlap,
            calibrate_view_graph=args.calibrate_view_graph,
            colmap_bin=args.colmap_bin,
        )
        print(f"Global SfM Result: Status={report['status']}, Registered={report['registered_images']}/{report['input_images']}")
        sys.exit(0 if report["status"] == "PASS" else 1)
    except Exception as exc:
        print(f"[ERROR] Global SfM execution terminated with exception: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
