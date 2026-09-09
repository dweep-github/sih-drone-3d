#!/usr/bin/env python3
"""Main COLMAP reconstruction pipeline for Person 1 (SIH26158 Step 1).

Executes end-to-end baseline reconstruction:
1. Environment and input verification
2. Database preparation
3. Feature extraction
4. Feature matching (sequential default)
5. Sparse reconstruction (mapper)
6. Output verification and reconstruction_report.json export
"""

from __future__ import annotations

import argparse
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
    find_sparse_model_dir,
    generate_reconstruction_report,
    inspect_sparse_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reconstruct")


def run_reconstruction(
    image_dir: str | Path,
    output_dir: str | Path,
    camera_model: str = "SIMPLE_RADIAL",
    matcher: str = "sequential",
    overlap: int = 10,
    colmap_bin: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Executes the 5-stage COLMAP baseline reconstruction pipeline."""
    start_time = time.time()
    errors: List[str] = []

    image_path = Path(image_dir).resolve()
    out_path = Path(output_dir).resolve()
    db_path = out_path / "database.db"
    sparse_dir = out_path / "sparse"
    report_path = out_path / "reconstruction_report.json"

    # [1/5] Checking environment
    print("\n[1/5] Checking environment")
    binary = find_colmap_binary(colmap_bin)
    if not binary:
        err = (
            "COLMAP executable could not be found.\n"
            "Remediation: Install COLMAP, add its bin directory to PATH, "
            "or specify --colmap-bin <path>."
        )
        print(f"[ERROR] {err}", file=sys.stderr)
        errors.append(err)
        generate_reconstruction_report(
            image_dir=image_path,
            sparse_dir=sparse_dir,
            output_report_path=report_path,
            colmap_version="Not found",
            processing_time_seconds=time.time() - start_time,
            errors=errors,
        )
        raise FileNotFoundError(err)

    colmap_ver = get_colmap_version(binary)
    cuda_avail = is_cuda_available()
    gpu_desc = "CUDA Enabled" if cuda_avail else "Disabled (CPU Mode)"

    print(f"  Input images:     {image_path}")
    print(f"  Output directory: {out_path}")
    print(f"  COLMAP version:   {colmap_ver}")
    print(f"  GPU:              {gpu_desc}")
    print(f"  Camera model:     {camera_model}")
    print(f"  Matcher:          {matcher}")

    # Validate image directory
    try:
        valid_images = validate_image_dir(image_path)
        print(f"  Found {len(valid_images)} valid images.")
    except Exception as exc:
        err = f"Image validation failed: {exc}"
        print(f"[ERROR] {err}", file=sys.stderr)
        errors.append(err)
        generate_reconstruction_report(
            image_dir=image_path,
            sparse_dir=sparse_dir,
            output_report_path=report_path,
            colmap_version=colmap_ver,
            processing_time_seconds=time.time() - start_time,
            errors=errors,
        )
        raise

    # [2/5] Preparing COLMAP database
    print("\n[2/5] Preparing COLMAP database")
    out_path.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        logger.info("Removing existing database: %s", db_path)
        db_path.unlink()
    print(f"  Database initialized at {db_path}")

    # [3/5] Extracting features
    print("\n[3/5] Extracting features")
    try:
        extract_features(
            image_dir=image_path,
            database_path=db_path,
            camera_model=camera_model,
            single_camera=True,
            use_gpu=cuda_avail,
            colmap_bin=binary,
        )
        print("  Feature extraction completed successfully.")
    except Exception as exc:
        err = (
            f"Feature extraction failed during Stage 3.\n"
            f"Reason: {exc}\n"
            f"Remediation: Verify image file formats and permissions, or run check_environment.py."
        )
        print(f"[ERROR] {err}", file=sys.stderr)
        errors.append(err)
        generate_reconstruction_report(
            image_dir=image_path,
            sparse_dir=sparse_dir,
            output_report_path=report_path,
            colmap_version=colmap_ver,
            processing_time_seconds=time.time() - start_time,
            errors=errors,
        )
        raise RuntimeError(err) from exc

    # [4/5] Matching features
    print("\n[4/5] Matching features")
    try:
        match_features(
            database_path=db_path,
            matcher=matcher,
            overlap=overlap,
            use_gpu=cuda_avail,
            colmap_bin=binary,
        )
        print(f"  Feature matching ({matcher}) completed successfully.")
    except Exception as exc:
        err = (
            f"Feature matching failed during Stage 4.\n"
            f"Reason: {exc}\n"
            f"Remediation: Check matcher options or try '--matcher exhaustive' if overlap is insufficient."
        )
        print(f"[ERROR] {err}", file=sys.stderr)
        errors.append(err)
        generate_reconstruction_report(
            image_dir=image_path,
            sparse_dir=sparse_dir,
            output_report_path=report_path,
            colmap_version=colmap_ver,
            processing_time_seconds=time.time() - start_time,
            errors=errors,
        )
        raise RuntimeError(err) from exc

    # [5/5] Running sparse reconstruction
    print("\n[5/5] Running sparse reconstruction (mapper)")
    mapper_args = [
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--output_path", str(sparse_dir),
    ]

    try:
        run_colmap("mapper", mapper_args, colmap_bin=binary)
        print("  COLMAP mapper finished.")
    except Exception as exc:
        err = (
            f"Sparse reconstruction failed during Stage 5.\n"
            f"Reason: {exc}\n"
            f"Remediation: Check if feature matches are sufficient or inspect COLMAP logs."
        )
        print(f"[ERROR] {err}", file=sys.stderr)
        errors.append(err)

    # Verification & Report Generation
    model_dir = find_sparse_model_dir(sparse_dir)
    if not model_dir:
        msg = f"Sparse reconstruction did not produce valid model files in {sparse_dir}"
        print(f"[WARNING] {msg}", file=sys.stderr)
        errors.append(msg)
    else:
        print(f"  Found reconstruction model in: {model_dir}")

    elapsed = time.time() - start_time
    report = generate_reconstruction_report(
        image_dir=image_path,
        sparse_dir=sparse_dir,
        output_report_path=report_path,
        colmap_version=colmap_ver,
        processing_time_seconds=elapsed,
        errors=errors,
    )

    print("\n========================================")
    print("RECONSTRUCTION REPORT SUMMARY")
    print("========================================")
    print(f"Status:             {'SUCCESS' if report['reconstruction_success'] else 'FAILED'}")
    print(f"Input images:       {report['input_images']}")
    print(f"Registered images:  {report['registered_images']}")
    print(f"Registration rate:  {report['registration_rate'] * 100:.1f}%")
    print(f"Cameras:            {report['num_cameras']}")
    print(f"3D points:          {report['num_points3D']}")
    print(f"Processing time:    {report['processing_time_seconds']:.2f}s")
    print(f"Report file:        {report_path}")
    print("========================================")

    if not report["reconstruction_success"]:
        raise RuntimeError("Reconstruction failed or produced 0 registered images.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIH26158 Person 1: COLMAP Baseline Reconstruction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Path to directory containing input drone images",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to directory where sparse model and reports will be saved",
    )
    parser.add_argument(
        "--camera-model",
        default="SIMPLE_RADIAL",
        help=(
            "COLMAP camera model (e.g., SIMPLE_RADIAL, PINHOLE, RADIAL, OPENCV). "
            "SIMPLE_RADIAL is standard for single moving-UAV flight with unknown radial distortion."
        ),
    )
    parser.add_argument(
        "--matcher",
        default="sequential",
        choices=["sequential", "exhaustive", "spatial", "vocab_tree"],
        help="Feature matching method. Sequential is recommended for ordered UAV flights.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Consecutive frame overlap window for sequential matcher",
    )
    parser.add_argument(
        "--colmap-bin",
        type=Path,
        default=None,
        help="Optional custom path to COLMAP executable or directory",
    )

    args = parser.parse_args()

    try:
        run_reconstruction(
            image_dir=args.images,
            output_dir=args.output,
            camera_model=args.camera_model,
            matcher=args.matcher,
            overlap=args.overlap,
            colmap_bin=args.colmap_bin,
        )
        sys.exit(0)
    except Exception as exc:
        print(f"\nPipeline terminated with error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
