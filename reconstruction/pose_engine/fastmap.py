"""FastMap Pose Engine Integration and Normalization Adapter.

Integrates the upstream FastMap Structure-from-Motion pipeline (https://github.com/pals-ttic/fastmap),
manages input database preparation using COLMAP matchers, executes FastMap where platform and
hardware support permit, adapts FastMap output into standardized pose representations,
and interfaces seamlessly with the Step 2 validation engine.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.colmap.feature_extraction import extract_features, validate_image_dir
from reconstruction.colmap.feature_matching import match_features
from reconstruction.colmap.utils import is_cuda_available
from reconstruction.validation.geometry import compute_camera_center
from reconstruction.validation.model_io import read_colmap_model
from reconstruction.validation.pose_report import (
    count_input_images,
    find_sparse_model_dir,
    validate_reconstruction,
)

logger = logging.getLogger("fastmap")

FASTMAP_REPO_DIR = PROJECT_ROOT / "tools" / "fastmap"


def get_fastmap_commit() -> Optional[str]:
    """Retrieves the commit hash of the cloned FastMap repository."""
    if not (FASTMAP_REPO_DIR / ".git").exists():
        return None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(FASTMAP_REPO_DIR),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def get_fastmap_environment_info() -> Dict[str, Any]:
    """Inspects the runtime environment and determines FastMap compatibility.

    FastMap upstream officially documents Linux-only support and requires PyTorch
    CUDA synchronization and custom CUDA extension kernels.
    """
    python_ver = sys.version.split()[0]
    cuda_avail = is_cuda_available()
    cuda_ver = None
    gpu_name = "None"

    try:
        import torch
        pytorch_ver = torch.__version__
        if cuda_avail:
            cuda_ver = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "None (CPU only)"
    except ImportError:
        pytorch_ver = "Missing"

    is_linux = sys.platform.startswith("linux")
    repo_present = (FASTMAP_REPO_DIR / "run.py").is_file()
    commit_hash = get_fastmap_commit()

    blockers: List[str] = []
    if not repo_present:
        blockers.append("FastMap repository not found at tools/fastmap.")
    if not is_linux:
        blockers.append(f"FastMap upstream officially supports Linux only (current OS: {sys.platform}).")
    if not cuda_avail:
        blockers.append("FastMap requires a CUDA-capable GPU and CUDA-enabled PyTorch (current CUDA available: False).")

    is_supported = len(blockers) == 0

    return {
        "repository": "https://github.com/pals-ttic/fastmap",
        "repo_path": str(FASTMAP_REPO_DIR),
        "commit": commit_hash,
        "os": sys.platform,
        "is_linux": is_linux,
        "python": python_ver,
        "pytorch": pytorch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu": gpu_name,
        "supported": is_supported,
        "blockers": blockers,
        "blocker_reason": " | ".join(blockers) if blockers else None,
    }


class FastMapAdapter:
    """Normalizes FastMap outputs into the project's standard pose representation."""

    @staticmethod
    def normalize_poses(model_dir: Path | str) -> List[Dict[str, Any]]:
        """Extracts camera poses in standard representation: position [x, y, z] and quaternion [qw, qx, qy, qz]."""
        path = Path(model_dir).resolve()
        sparse_dir = find_sparse_model_dir(path) or path
        model = read_colmap_model(sparse_dir)

        normalized_frames: List[Dict[str, Any]] = []
        for img in model.images.values():
            center = compute_camera_center(img.qvec, img.tvec)
            normalized_frames.append({
                "frame_id": Path(img.name).stem,
                "image_name": img.name,
                "image_id": img.image_id,
                "camera_id": img.camera_id,
                "position": [round(float(c), 6) for c in center],
                "rotation_quaternion": [round(float(q), 6) for q in img.qvec],
            })

        return normalized_frames

    @staticmethod
    def validate(
        model_dir: Path | str,
        image_dir: Optional[Path | str] = None,
        output_report_path: Optional[Path | str] = None,
        trajectory_output_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """Runs the standard Step 2 validation engine on FastMap output."""
        return validate_reconstruction(
            model_dir=model_dir,
            image_dir=image_dir,
            output_report_path=output_report_path,
            trajectory_output_path=trajectory_output_path,
        )


def run_fastmap(
    images: str | Path,
    output: str | Path,
    device: str = "cuda:0",
    config: Optional[str | Path] = None,
    headless: bool = True,
    colmap_bin: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Executes FastMap SfM pipeline or reports structured blocker information.

    Workflow:
    1. Validates input images directory.
    2. Verifies environment compatibility (Linux + CUDA).
    3. Prepares COLMAP feature database (FastMap requirement).
    4. Runs FastMap backend.
    5. Normalizes poses and runs validation.
    """
    start_time = time.time()
    img_path = Path(images).resolve()
    out_path = Path(output).resolve()
    db_path = out_path / "database.db"
    fastmap_report_path = out_path / "fastmap_report.json"
    pose_report_path = out_path / "pose_report.json"
    traj_path = out_path / "trajectory.json"

    # 1. Validate inputs
    valid_images = validate_image_dir(img_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # 2. Check environment compatibility
    env_info = get_fastmap_environment_info()

    if not env_info["supported"]:
        elapsed = time.time() - start_time
        blocker_msg = env_info["blocker_reason"]
        logger.warning("FastMap cannot execute: %s", blocker_msg)

        report = {
            "status": "BLOCKED",
            "runtime_seconds": round(elapsed, 4),
            "input_images": len(valid_images),
            "registered_images": 0,
            "registration_rate": 0.0,
            "environment": env_info,
            "reprojection": {"status": "NOT_AVAILABLE", "reason": blocker_msg},
            "trajectory": {"status": "NOT_AVAILABLE", "reason": blocker_msg},
            "errors": env_info["blockers"],
            "blocker_reason": blocker_msg,
        }

        with open(fastmap_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    # 3. Prepare COLMAP database (FastMap's expected input)
    logger.info("Extracting features for FastMap...")
    extract_features(
        image_dir=img_path,
        database_path=db_path,
        camera_model="SIMPLE_RADIAL",
        use_gpu=env_info["cuda_available"],
        colmap_bin=colmap_bin,
    )

    logger.info("Matching features for FastMap...")
    match_features(
        database_path=db_path,
        matcher="sequential",
        overlap=10,
        use_gpu=env_info["cuda_available"],
        colmap_bin=colmap_bin,
    )

    # 4. Execute FastMap
    fastmap_run_script = FASTMAP_REPO_DIR / "run.py"
    cmd = [
        sys.executable,
        str(fastmap_run_script),
        "--database", str(db_path),
        "--image_dir", str(img_path),
        "--output_dir", str(out_path),
        "--device", device,
    ]
    if headless:
        cmd.append("--headless")
    if config:
        cmd.extend(["--config", str(Path(config).resolve())])

    logger.info("Running FastMap command: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(FASTMAP_REPO_DIR),
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start_time

    if proc.returncode != 0:
        err_msg = (
            f"FastMap execution failed (code {proc.returncode}).\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
        logger.error(err_msg)
        report = {
            "status": "FAIL",
            "runtime_seconds": round(elapsed, 4),
            "input_images": len(valid_images),
            "registered_images": 0,
            "registration_rate": 0.0,
            "environment": env_info,
            "errors": [err_msg],
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        with open(fastmap_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    # 5. Adapt and Validate Output
    validation_result = FastMapAdapter.validate(
        model_dir=out_path,
        image_dir=img_path,
        output_report_path=pose_report_path,
        trajectory_output_path=traj_path,
    )

    report = {
        "status": validation_result["validation_status"],
        "runtime_seconds": round(elapsed, 4),
        "input_images": len(valid_images),
        "registered_images": validation_result["input"]["registered_images"],
        "registration_rate": validation_result["input"]["registration_rate"],
        "reprojection": validation_result["reprojection"],
        "trajectory": validation_result["trajectory"],
        "environment": env_info,
        "validation_report": validation_result,
        "errors": validation_result.get("errors", []),
    }

    with open(fastmap_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIH26158 Person 1: FastMap Pose Engine Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", type=Path, default=None, help="Input images directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Execution device (e.g. cuda:0, cpu)")
    parser.add_argument("--config", type=Path, default=None, help="Optional FastMap YAML config")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without visualization GUI")
    parser.add_argument("--check-env", action="store_true", help="Print FastMap environment diagnostics and exit")

    args = parser.parse_args()

    if args.check_env:
        env = get_fastmap_environment_info()
        print(json.dumps(env, indent=2))
        sys.exit(0)

    if not args.images or not args.output:
        parser.error("--images and --output are required unless --check-env is specified.")

    try:
        res = run_fastmap(
            images=args.images,
            output=args.output,
            device=args.device,
            config=args.config,
            headless=args.headless,
        )
        print(f"FastMap Result: Status={res['status']}")
        if res["status"] == "BLOCKED":
            print(f"Blocker: {res.get('blocker_reason')}")
            sys.exit(2)
        elif res["status"] != "PASS":
            sys.exit(1)
        sys.exit(0)
    except Exception as exc:
        print(f"[ERROR] FastMap execution terminated with exception: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
