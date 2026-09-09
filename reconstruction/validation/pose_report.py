"""Pose validation and report generation module for Person 1 (SIH26158).

Performs rigorous photogrammetric validation on COLMAP sparse reconstruction outputs:
1. Registration rate validation against configurable threshold
2. Reprojection error computation (Mean, Median, RMSE, Max px) using exact camera distortion models
3. Trajectory continuity analysis (Camera center C = -R^T*t, translation steps, rotation changes, jump detection)
4. Exports standardized pose_report.json and trajectory.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.validation.model_io import (
    ImagePose,
    ReconstructionModel,
    read_colmap_model,
)
from reconstruction.validation.geometry import (
    compute_camera_center,
    compute_relative_rotation_angle_deg,
    compute_reprojection_errors,
    detect_jumps,
    qvec_to_rotmat,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_MIN_REGISTRATION_RATE = 0.80
DEFAULT_MAX_REPROJECTION_ERROR = 1.0


# =====================================================================
# Step 1 Compatibility Functions (Preserved for reconstruct.py)
# =====================================================================

def count_input_images(image_dir: Path | str) -> int:
    """Counts valid image files in the input directory."""
    path = Path(image_dir)
    if not path.is_dir():
        return 0
    return len([
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ])


def find_sparse_model_dir(sparse_root: Path | str) -> Optional[Path]:
    """Locates the directory containing COLMAP cameras, images, and points3D files.

    COLMAP mapper outputs models into subdirectories like 'sparse/0', or directly in 'sparse'.
    """
    root = Path(sparse_root)
    if not root.is_dir():
        return None

    # Check if files exist directly in root
    bin_files = (root / "cameras.bin", root / "images.bin", root / "points3D.bin")
    txt_files = (root / "cameras.txt", root / "images.txt", root / "points3D.txt")
    if all(f.is_file() for f in bin_files) or all(f.is_file() for f in txt_files):
        return root

    # Check numbered subdirectories ('0', '1', ...)
    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    for subdir in subdirs:
        bin_files = (subdir / "cameras.bin", subdir / "images.bin", subdir / "points3D.bin")
        txt_files = (subdir / "cameras.txt", subdir / "images.txt", subdir / "points3D.txt")
        if all(f.is_file() for f in bin_files) or all(f.is_file() for f in txt_files):
            return subdir

    return None


def inspect_sparse_model_fallback(model_dir: Path) -> Dict[str, Any]:
    """Inspects a COLMAP sparse model directory using direct binary/text parsing."""
    stats = {
        "num_cameras": 0,
        "registered_images": 0,
        "num_points3D": 0,
        "exists": False,
    }

    # Binary check
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    points3d_bin = model_dir / "points3D.bin"

    if cameras_bin.is_file() and images_bin.is_file() and points3d_bin.is_file():
        try:
            with open(cameras_bin, "rb") as f:
                stats["num_cameras"] = struct.unpack("<Q", f.read(8))[0]
            with open(images_bin, "rb") as f:
                stats["registered_images"] = struct.unpack("<Q", f.read(8))[0]
            with open(points3d_bin, "rb") as f:
                stats["num_points3D"] = struct.unpack("<Q", f.read(8))[0]
            stats["exists"] = True
            return stats
        except Exception as exc:
            logger.warning("Binary parse failed on %s: %s", model_dir, exc)

    # Text check
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"
    points3d_txt = model_dir / "points3D.txt"

    if cameras_txt.is_file() and images_txt.is_file() and points3d_txt.is_file():
        try:
            num_cams = 0
            with open(cameras_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        num_cams += 1
            stats["num_cameras"] = num_cams

            num_imgs = 0
            with open(images_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        num_imgs += 1
            stats["registered_images"] = num_imgs // 2

            num_pts = 0
            with open(points3d_txt, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        num_pts += 1
            stats["num_points3D"] = num_pts
            stats["exists"] = True
            return stats
        except Exception as exc:
            logger.warning("Text parse failed on %s: %s", model_dir, exc)

    return stats


def inspect_sparse_model(sparse_dir: Path | str) -> Dict[str, Any]:
    """Inspects a COLMAP sparse model directory (text or binary)."""
    model_dir = find_sparse_model_dir(sparse_dir)
    if not model_dir:
        return {
            "num_cameras": 0,
            "registered_images": 0,
            "num_points3D": 0,
            "mean_reprojection_error": None,
            "exists": False,
            "model_path": None,
        }

    try:
        import pycolmap
        rec = pycolmap.Reconstruction(str(model_dir))
        mean_error = None
        try:
            mean_error = round(float(rec.compute_mean_reprojection_error()), 4)
        except Exception:
            pass

        return {
            "num_cameras": rec.num_cameras(),
            "registered_images": rec.num_reg_images(),
            "num_points3D": rec.num_points3D(),
            "mean_reprojection_error": mean_error,
            "exists": True,
            "model_path": str(model_dir),
        }
    except Exception as exc:
        logger.debug("pycolmap inspection failed (%s); falling back to direct parse.", exc)

    stats = inspect_sparse_model_fallback(model_dir)
    stats["mean_reprojection_error"] = None
    stats["model_path"] = str(model_dir) if stats["exists"] else None
    return stats


def generate_reconstruction_report(
    image_dir: Path | str,
    sparse_dir: Path | str,
    output_report_path: Path | str,
    colmap_version: str,
    processing_time_seconds: float,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validates baseline reconstruction outputs and writes reconstruction_report.json (Step 1)."""
    errors_list = list(errors or [])
    input_images = count_input_images(image_dir)

    model_stats = inspect_sparse_model(sparse_dir)
    registered_images = model_stats["registered_images"]
    num_cameras = model_stats["num_cameras"]
    num_points = model_stats["num_points3D"]

    if input_images > 0:
        registration_rate = round(registered_images / input_images, 4)
    else:
        registration_rate = 0.0

    success = model_stats["exists"] and registered_images > 0 and not errors_list

    if not model_stats["exists"]:
        errors_list.append("COLMAP sparse reconstruction files (cameras, images, points3D) not found.")

    report: Dict[str, Any] = {
        "reconstruction_success": success,
        "input_images": input_images,
        "registered_images": registered_images,
        "registration_rate": registration_rate,
        "num_cameras": num_cameras,
        "num_points3D": num_points,
        "processing_time_seconds": round(processing_time_seconds, 2),
        "colmap_version": colmap_version,
        "errors": errors_list,
    }

    report_path = Path(output_report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# =====================================================================
# Step 2: Full Pose Validation & Trajectory Extraction
# =====================================================================

def extract_frame_sequence_key(image_name: str) -> Tuple[int, str]:
    """Extracts chronological sequence number from image filename (e.g., frame_000127.jpg)."""
    # Look for trailing or prominent integer digits
    matches = re.findall(r"\d+", image_name)
    if matches:
        # Use the last numeric group (standard for frame_000127 or IMG_0127)
        return (int(matches[-1]), image_name)
    return (-1, image_name)


def order_registered_images(images: Dict[int, ImagePose]) -> Tuple[List[ImagePose], str]:
    """Orders registered images chronologically by frame number, or lexicographically."""
    if not images:
        return [], "none"

    img_list = list(images.values())
    keys = [extract_frame_sequence_key(img.name) for img in img_list]
    has_all_numbers = all(k[0] >= 0 for k in keys)

    if has_all_numbers:
        # Sort by extracted integer frame number
        sorted_images = [img for _, img in sorted(zip(keys, img_list), key=lambda item: item[0])]
        return sorted_images, "frame_filename"
    else:
        # Fall back to filename alphabetical ordering
        sorted_images = sorted(img_list, key=lambda img: img.name)
        return sorted_images, "lexicographical"


def validate_reconstruction(
    model_dir: str | Path,
    image_dir: Optional[str | Path] = None,
    output_report_path: Optional[str | Path] = None,
    trajectory_output_path: Optional[str | Path] = None,
    min_registration_rate: float = DEFAULT_MIN_REGISTRATION_RATE,
    max_reprojection_error: float = DEFAULT_MAX_REPROJECTION_ERROR,
    max_position_jump: Optional[float] = None,
    max_rotation_jump: Optional[float] = None,
) -> Dict[str, Any]:
    """Executes full pose, reprojection, and trajectory validation on a COLMAP reconstruction."""
    warnings: List[str] = []
    errors: List[str] = []

    model_path = Path(model_dir).resolve()
    actual_model_dir = find_sparse_model_dir(model_path) or model_path

    # 1. Read Model
    try:
        model = read_colmap_model(actual_model_dir)
    except Exception as exc:
        err_msg = f"Failed to read COLMAP model from {actual_model_dir}: {exc}"
        errors.append(err_msg)
        report = {
            "validation_status": "FAIL",
            "input": {
                "total_images": 0,
                "registered_images": 0,
                "unregistered_images": 0,
                "registration_rate": 0.0,
            },
            "reconstruction": {"num_cameras": 0, "num_3d_points": 0, "num_observations": 0},
            "reprojection": {"status": "NOT_AVAILABLE", "reason": err_msg},
            "trajectory": {"status": "NOT_AVAILABLE", "reason": err_msg},
            "checks": {
                "registration": {"status": "FAIL"},
                "reprojection": {"status": "FAIL"},
                "trajectory": {"status": "FAIL"},
            },
            "thresholds": {
                "min_registration_rate": min_registration_rate,
                "max_reprojection_error_px": max_reprojection_error,
                "threshold_note": "This is an initial project validation threshold, not a universal photogrammetric accuracy standard.",
            },
            "warnings": warnings,
            "errors": errors,
        }
        if output_report_path:
            out_p = Path(output_report_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        return report

    # 2. Registration Validation
    reg_images_count = model.num_images
    if image_dir and Path(image_dir).is_dir():
        total_images_count = count_input_images(image_dir)
    else:
        # If no image dir given, assume registered count as minimum baseline
        total_images_count = reg_images_count

    unregistered_count = max(0, total_images_count - reg_images_count)
    if total_images_count > 0:
        reg_rate = round(reg_images_count / total_images_count, 4)
    else:
        reg_rate = 0.0

    reg_passed = reg_rate >= min_registration_rate and reg_images_count > 0
    reg_status = "PASS" if reg_passed else "FAIL"
    if not reg_passed:
        warnings.append(
            f"Registration rate ({reg_rate * 100:.1f}%) is below minimum threshold ({min_registration_rate * 100:.1f}%)."
        )

    # 3. Reprojection Error Validation
    rep_errors, rep_status_str = compute_reprojection_errors(model)
    reprojection_section: Dict[str, Any] = {"status": rep_status_str}
    rep_passed = False

    if rep_status_str == "AVAILABLE" and rep_errors:
        err_arr = np.array(rep_errors, dtype=np.float64)
        mean_px = round(float(np.mean(err_arr)), 4)
        median_px = round(float(np.median(err_arr)), 4)
        rmse_px = round(float(np.sqrt(np.mean(err_arr**2))), 4)
        max_px = round(float(np.max(err_arr)), 4)
        obs_used = len(rep_errors)

        reprojection_section.update({
            "mean_px": mean_px,
            "median_px": median_px,
            "rmse_px": rmse_px,
            "max_px": max_px,
            "observations_used": obs_used,
        })

        rep_passed = rmse_px <= max_reprojection_error
        reproj_check_status = "PASS" if rep_passed else "FAIL"
        if not rep_passed:
            warnings.append(
                f"Reprojection RMSE ({rmse_px} px) exceeds maximum threshold ({max_reprojection_error} px)."
            )
    else:
        reprojection_section["reason"] = (
            "No valid 3D point observations found or camera model distortion is unsupported."
        )
        reproj_check_status = "NOT_AVAILABLE"
        warnings.append("Reprojection error could not be computed.")

    # 4. Trajectory Extraction & Analysis
    ordered_images, order_type = order_registered_images(model.images)
    trajectory_section: Dict[str, Any] = {
        "status": "AVAILABLE" if len(ordered_images) >= 2 else "NOT_AVAILABLE",
        "ordered_by": order_type,
    }

    positions: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    trajectory_frames: List[Dict[str, Any]] = []

    for img in ordered_images:
        center = compute_camera_center(img.qvec, img.tvec)
        positions.append(center)
        rotations.append(qvec_to_rotmat(img.qvec))

        # Frame ID extraction
        seq_num, _ = extract_frame_sequence_key(img.name)
        frame_id = f"frame_{seq_num:06d}" if seq_num >= 0 else Path(img.name).stem

        # Store in trajectory frames (local coordinates)
        trajectory_frames.append({
            "frame_id": frame_id,
            "image_name": img.name,
            "image_id": img.image_id,
            "position": [round(float(coord), 6) for coord in center],
            "rotation_quaternion": [round(float(q), 6) for q in img.qvec],
        })

    traj_passed = False
    if len(ordered_images) < 2:
        trajectory_section["reason"] = "Fewer than 2 registered images; cannot compute trajectory steps."
        traj_check_status = "FAIL" if reg_images_count > 0 else "NOT_AVAILABLE"
        warnings.append("Insufficient registered images for trajectory analysis.")
    else:
        # Consecutive step distances
        step_distances = [
            float(np.linalg.norm(positions[i] - positions[i - 1]))
            for i in range(1, len(positions))
        ]
        step_arr = np.array(step_distances, dtype=np.float64)

        # Consecutive relative rotation angles
        rot_angles_deg = [
            compute_relative_rotation_angle_deg(rotations[i - 1], rotations[i])
            for i in range(1, len(rotations))
        ]
        rot_arr = np.array(rot_angles_deg, dtype=np.float64)

        # Jump detections
        pos_jumps, max_pos_jump, flagged_pos = detect_jumps(
            step_distances, threshold=max_position_jump
        )
        rot_jumps, max_rot_jump, flagged_rot = detect_jumps(
            rot_angles_deg, threshold=max_rotation_jump
        )

        trajectory_section.update({
            "mean_step": round(float(np.mean(step_arr)), 4),
            "median_step": round(float(np.median(step_arr)), 4),
            "max_step": round(float(np.max(step_arr)), 4),
            "position_jump_count": pos_jumps,
            "flagged_position_transitions": flagged_pos,
            "mean_rotation_change_deg": round(float(np.mean(rot_arr)), 2),
            "median_rotation_change_deg": round(float(np.median(rot_arr)), 2),
            "max_rotation_change_deg": round(float(np.max(rot_arr)), 2),
            "rotation_jump_count": rot_jumps,
            "flagged_rotation_transitions": flagged_rot,
        })

        # Trajectory passes if no abnormal jumps when explicit thresholds are set, or ordering is established
        traj_passed = (pos_jumps == 0 if max_position_jump is not None else True) and \
                      (rot_jumps == 0 if max_rotation_jump is not None else True) and \
                      (order_type != "none")
        traj_check_status = "PASS" if traj_passed else "WARNING"

    # 5. Overall Decision
    # Overall is PASS if registration passed, reprojection passed, and trajectory passed
    overall_passed = reg_passed and rep_passed and (traj_check_status == "PASS")
    validation_status = "PASS" if overall_passed else "FAIL"

    # Count total 2D observations
    total_obs = sum(len(img.xys) for img in model.images.values())

    report = {
        "validation_status": validation_status,
        "input": {
            "total_images": total_images_count,
            "registered_images": reg_images_count,
            "unregistered_images": unregistered_count,
            "registration_rate": reg_rate,
        },
        "reconstruction": {
            "num_cameras": model.num_cameras,
            "num_3d_points": model.num_points3D,
            "num_observations": total_obs,
        },
        "reprojection": reprojection_section,
        "trajectory": trajectory_section,
        "checks": {
            "registration": {"status": reg_status},
            "reprojection": {"status": reproj_check_status},
            "trajectory": {"status": traj_check_status},
        },
        "thresholds": {
            "min_registration_rate": min_registration_rate,
            "max_reprojection_error_px": max_reprojection_error,
            "threshold_note": "This is an initial project validation threshold, not a universal photogrammetric accuracy standard.",
        },
        "warnings": warnings,
        "errors": errors,
    }

    # 6. Export Reports
    if output_report_path:
        out_p = Path(output_report_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved pose validation report to %s", out_p)

    if trajectory_output_path:
        traj_p = Path(trajectory_output_path).resolve()
        traj_p.parent.mkdir(parents=True, exist_ok=True)
        traj_data = {
            "coordinate_convention": "COLMAP local reconstruction coordinates (camera center C = -R^T * t)",
            "georeferenced": False,
            "quaternion_convention": "[qw, qx, qy, qz] (Hamilton convention)",
            "num_frames": len(trajectory_frames),
            "frames": trajectory_frames,
        }
        with open(traj_p, "w", encoding="utf-8") as f:
            json.dump(traj_data, f, indent=2)
        logger.info("Saved trajectory data to %s", traj_p)

    return report


def print_human_readable_summary(report: Dict[str, Any]) -> None:
    """Prints a clean, standardized human-readable validation summary."""
    inp = report.get("input", {})
    rep = report.get("reprojection", {})
    traj = report.get("trajectory", {})
    checks = report.get("checks", {})

    print("\n========================================")
    print("COLMAP POSE VALIDATION")
    print("========================================")

    print("\nREGISTRATION")
    print(f"  Total:       {inp.get('total_images', 0)}")
    print(f"  Registered:  {inp.get('registered_images', 0)}")
    rate_pct = inp.get("registration_rate", 0.0) * 100.0
    print(f"  Rate:        {rate_pct:.1f}%")
    print(f"  Status:      {checks.get('registration', {}).get('status', 'N/A')}")

    print("\nREPROJECTION")
    if rep.get("status") == "AVAILABLE":
        print(f"  Mean:        {rep.get('mean_px', 0.0):.2f} px")
        print(f"  Median:      {rep.get('median_px', 0.0):.2f} px")
        print(f"  RMSE:        {rep.get('rmse_px', 0.0):.2f} px")
        print(f"  Maximum:     {rep.get('max_px', 0.0):.2f} px")
    else:
        print(f"  Status:      {rep.get('status', 'NOT_AVAILABLE')}")
        if "reason" in rep:
            print(f"  Reason:      {rep['reason']}")
    print(f"  Status:      {checks.get('reprojection', {}).get('status', 'N/A')}")

    print("\nTRAJECTORY")
    if traj.get("status") == "AVAILABLE":
        print(f"  Mean step:   {traj.get('mean_step', 0.0):.4f}")
        print(f"  Median step: {traj.get('median_step', 0.0):.4f}")
        print(f"  Maximum step:{traj.get('max_step', 0.0):.4f}")
        print(f"  Rot mean:    {traj.get('mean_rotation_change_deg', 0.0):.1f} deg")
        print(f"  Rot max:     {traj.get('max_rotation_change_deg', 0.0):.1f} deg")
    else:
        print(f"  Status:      {traj.get('status', 'NOT_AVAILABLE')}")
        if "reason" in traj:
            print(f"  Reason:      {traj['reason']}")
    print(f"  Status:      {checks.get('trajectory', {}).get('status', 'N/A')}")

    print("\n----------------------------------------")
    print(f"OVERALL: {report.get('validation_status', 'FAIL')}")
    print("========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIH26158 Person 1: COLMAP Pose Validation & Trajectory Extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to COLMAP sparse model directory (e.g., test_output/sparse/0)",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Optional path to input image directory to determine total input image count",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reconstruction/test_output/pose_report.json"),
        help="Path to output pose_report.json",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=Path("reconstruction/test_output/trajectory.json"),
        help="Path to output trajectory.json",
    )
    parser.add_argument(
        "--min-registration-rate",
        type=float,
        default=DEFAULT_MIN_REGISTRATION_RATE,
        help="Minimum required registration rate [0.0 - 1.0]",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=DEFAULT_MAX_REPROJECTION_ERROR,
        help="Maximum allowed reprojection error in pixels (RMSE)",
    )
    parser.add_argument(
        "--max-position-jump",
        type=float,
        default=None,
        help="Maximum allowed translation step between consecutive frames (COLMAP units)",
    )
    parser.add_argument(
        "--max-rotation-jump",
        type=float,
        default=None,
        help="Maximum allowed rotation angle between consecutive frames (degrees)",
    )

    args = parser.parse_args()

    try:
        report = validate_reconstruction(
            model_dir=args.model,
            image_dir=args.images,
            output_report_path=args.output,
            trajectory_output_path=args.trajectory_output,
            min_registration_rate=args.min_registration_rate,
            max_reprojection_error=args.max_reprojection_error,
            max_position_jump=args.max_position_jump,
            max_rotation_jump=args.max_rotation_jump,
        )
        print_human_readable_summary(report)
        exit_code = 0 if report["validation_status"] == "PASS" else 1
        sys.exit(exit_code)
    except Exception as exc:
        print(f"[ERROR] Pose validation terminated with exception: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
