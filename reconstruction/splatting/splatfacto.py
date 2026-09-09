"""Nerfstudio Splatfacto training wrapper, configuration, and preflight module (SIH26158 Step 9).

Handles:
1. GPU / CUDA and Nerfstudio / Splatfacto preflight checks
2. Configurable training parameters (iterations, downscale, eval/save intervals)
3. Controlled execution wrapper capturing stdout/stderr, runtime, checkpoints
4. Real smoke test execution (or reporting NOT_AVAILABLE if hardware/software missing)
5. Standalone CLI with --check-available mode
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("splatting.splatfacto")


def check_splatfacto_environment() -> Dict[str, Any]:
    """Inspects the local system for Python, PyTorch, CUDA, and Nerfstudio/Splatfacto availability.

    Never fabricates results.
    """
    python_ver = sys.version.split()[0]
    pytorch_ver = torch.__version__
    cuda_avail = bool(torch.cuda.is_available())
    cuda_ver = torch.version.cuda if cuda_avail else None
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None

    gpu_vram_gb: Optional[float] = None
    if cuda_avail:
        try:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            gpu_vram_gb = round(total_bytes / (1024**3), 2)
        except Exception:
            gpu_vram_gb = None

    # Check Nerfstudio Python package
    nerfstudio_avail = False
    nerfstudio_ver = None
    try:
        import nerfstudio
        nerfstudio_avail = True
        nerfstudio_ver = getattr(nerfstudio, "__version__", "unknown")
    except ImportError:
        nerfstudio_avail = False

    # Check ns-train executable
    ns_train_bin = shutil.which("ns-train")
    splatfacto_avail = nerfstudio_avail or (ns_train_bin is not None)

    blockers: List[str] = []
    if not cuda_avail:
        blockers.append("CUDA is not available (PyTorch running in CPU mode). Splatfacto requires an NVIDIA GPU.")
    if not nerfstudio_avail and ns_train_bin is None:
        blockers.append("Nerfstudio is not installed in the active environment ('ns-train' executable not found).")

    is_supported = (len(blockers) == 0)

    return {
        "python": python_ver,
        "pytorch": pytorch_ver,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_ver,
        "gpu": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "nerfstudio_available": nerfstudio_avail,
        "nerfstudio_version": nerfstudio_ver,
        "splatfacto_available": splatfacto_avail,
        "ns_train_path": ns_train_bin,
        "supported": is_supported,
        "blockers": blockers,
        "blocker_reason": " | ".join(blockers) if blockers else None,
    }


@dataclass
class SplatfactoConfig:
    """Configurable training parameters for Nerfstudio Splatfacto."""

    max_num_iterations: int = 30000  # Default full training; use 100 for smoke testing
    downscale_factor: int = 1
    eval_step: int = 500
    save_step: int = 2000
    device: str = "cuda"
    vis: str = "None"  # Headless mode for automated pipelines
    experiment_name: str = "splatfacto"

    def to_cli_args(self, data_dir: Path, output_dir: Path) -> List[str]:
        """Translates config into CLI arguments for ns-train splatfacto."""
        args = [
            "splatfacto",
            "--data", str(data_dir),
            "--output-dir", str(output_dir),
            "--experiment-name", self.experiment_name,
            "--max-num-iterations", str(self.max_num_iterations),
            "--pipeline.datamanager.dataparser.downscale-factor", str(self.downscale_factor),
            "--vis", self.vis,
        ]
        return args

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_splatfacto_training(
    dataset_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[SplatfactoConfig] = None,
    mock_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Executes Nerfstudio Splatfacto training on a prepared dataset directory.

    If CUDA or Splatfacto is unavailable and no mock runner is supplied:
    - Fails cleanly with status = "NOT_AVAILABLE" (or "FAILED" if forced)
    - Records explicit hardware/software reason
    - Never fabricates results

    Args:
        dataset_dir: Path to prepared dataset containing transforms.json.
        output_dir: Target directory for checkpoints, config, and exports.
        config: SplatfactoConfig object.
        mock_runner: Optional mock function for unit tests without GPU.

    Returns:
        Standardized training report dictionary.
    """
    start_time = time.time()
    cfg = config or SplatfactoConfig()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dset_dir = Path(dataset_dir).resolve()

    env_info = check_splatfacto_environment()

    # If mock runner provided (e.g. in automated unit tests)
    if mock_runner is not None:
        logger.info("Executing Splatfacto via mock runner for software test verification.")
        mock_res = mock_runner(dataset_dir=dset_dir, output_dir=out_dir, config=cfg)
        mock_res["runtime_seconds"] = round(time.time() - start_time, 4)
        return mock_res

    # Preflight validation
    if not env_info["supported"]:
        elapsed = round(time.time() - start_time, 4)
        reason = env_info["blocker_reason"] or "Environment incompatible with Splatfacto"
        logger.warning("Splatfacto training skipped: %s", reason)

        report = {
            "status": "NOT_AVAILABLE",
            "method": "nerfstudio_splatfacto",
            "iterations": 0,
            "runtime_seconds": elapsed,
            "device": cfg.device,
            "gpu": env_info.get("gpu"),
            "environment": env_info,
            "error": reason,
            "checkpoint_path": None,
            "output_directory": str(out_dir),
            "pointcloud_initialization": {
                "used": False,
                "reason": "Training not executed due to unavailable environment.",
            },
            "coordinate_transform": {
                "applied": False,
                "source_crs": None,
                "normalization": None,
            },
        }

        # Save report
        with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    # Formulate ns-train command
    ns_train_bin = env_info["ns_train_path"] or "ns-train"
    cmd = [ns_train_bin] + cfg.to_cli_args(data_dir=dset_dir, output_dir=out_dir)

    logger.info("Launching Splatfacto training: %s", " ".join(cmd))
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(out_dir),
            check=False,
        )
        elapsed = round(time.time() - start_time, 4)

        if proc.returncode != 0:
            err_msg = f"ns-train splatfacto failed with return code {proc.returncode}.\nSTDERR: {proc.stderr[:1000]}"
            logger.error(err_msg)
            report = {
                "status": "FAILED",
                "method": "nerfstudio_splatfacto",
                "iterations": 0,
                "runtime_seconds": elapsed,
                "device": cfg.device,
                "gpu": env_info.get("gpu"),
                "environment": env_info,
                "error": err_msg,
                "checkpoint_path": None,
                "output_directory": str(out_dir),
            }
        else:
            # Locate saved config / checkpoint
            ckpt_dirs = list(out_dir.glob("**/nerfstudio_models"))
            ckpt_path = str(ckpt_dirs[0]) if ckpt_dirs else None

            report = {
                "status": "SUCCESS",
                "method": "nerfstudio_splatfacto",
                "iterations": cfg.max_num_iterations,
                "runtime_seconds": elapsed,
                "device": cfg.device,
                "gpu": env_info.get("gpu"),
                "environment": env_info,
                "checkpoint_path": ckpt_path,
                "output_directory": str(out_dir),
                "config": cfg.to_dict(),
            }

    except Exception as exc:
        elapsed = round(time.time() - start_time, 4)
        err_msg = f"Failed to execute Splatfacto subprocess: {exc}"
        logger.error(err_msg)
        report = {
            "status": "FAILED",
            "method": "nerfstudio_splatfacto",
            "iterations": 0,
            "runtime_seconds": elapsed,
            "device": cfg.device,
            "gpu": env_info.get("gpu"),
            "environment": env_info,
            "error": err_msg,
            "checkpoint_path": None,
            "output_directory": str(out_dir),
        }

    # Save training_report.json
    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def run_real_splat_smoke_test(
    images_dir: Union[str, Path],
    cameras_path: Union[str, Path],
    output_dir: Union[str, Path],
    pointcloud_path: Optional[Union[str, Path]] = None,
    max_iterations: int = 50,
) -> Dict[str, Any]:
    """Attempts ONE real small smoke test if GPU and Splatfacto are available (Step 9 Section 19).

    If GPU or Splatfacto is unavailable, reports status = "NOT_AVAILABLE" with genuine reason.
    Never fabricates metrics.
    """
    env = check_splatfacto_environment()
    if not env["supported"]:
        return {
            "real_splat_smoke_test": "NOT_AVAILABLE",
            "status": "NOT_AVAILABLE",
            "reason": env["blocker_reason"],
            "environment": env,
        }

    # If supported, prepare mini dataset and run
    from reconstruction.splatting.dataset import prepare_splatfacto_dataset

    smoke_out = Path(output_dir).resolve() / "smoke_test"
    smoke_out.mkdir(parents=True, exist_ok=True)
    dset_dir = smoke_out / "dataset"

    dset_info = prepare_splatfacto_dataset(
        images_dir=images_dir,
        cameras_path=cameras_path,
        output_dir=dset_dir,
        pointcloud_path=pointcloud_path,
    )

    cfg = SplatfactoConfig(
        max_num_iterations=max_iterations,
        experiment_name="smoke_test",
        vis="None",
    )

    train_report = run_splatfacto_training(
        dataset_dir=dset_dir,
        output_dir=smoke_out / "training",
        config=cfg,
    )

    return {
        "real_splat_smoke_test": train_report.get("status", "FAILED"),
        "training_report": train_report,
        "dataset_summary": dset_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nerfstudio Splatfacto Wrapper CLI (SIH26158 Step 9)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--check-available", action="store_true", help="Check CUDA and Splatfacto availability and exit")
    parser.add_argument("--images", type=Path, help="Path to images directory")
    parser.add_argument("--cameras", type=Path, help="Path to cameras.json")
    parser.add_argument("--pointcloud", type=Path, default=None, help="Optional path to Step 8 point cloud (pointcloud.ply)")
    parser.add_argument("--output", type=Path, help="Output directory for dataset and training")
    parser.add_argument("--max-num-iterations", type=int, default=30000, help="Maximum training iterations")
    parser.add_argument("--downscale-factor", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--eval-step", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--save-step", type=int, default=2000, help="Checkpoint save interval")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda or cpu)")

    args = parser.parse_args()

    if args.check_available:
        env = check_splatfacto_environment()
        print("\n========================================")
        print("SPLATFACTO ENVIRONMENT PREFLIGHT CHECK")
        print("========================================")
        print(f"Python Version:       {env['python']}")
        print(f"PyTorch Version:      {env['pytorch']}")
        print(f"CUDA Available:       {env['cuda_available']}")
        print(f"CUDA Version:         {env['cuda_version'] or 'N/A'}")
        print(f"GPU Device:           {env['gpu'] or 'None (CPU Mode)'}")
        print(f"GPU VRAM:             {env['gpu_vram_gb']} GB" if env['gpu_vram_gb'] else "GPU VRAM:             N/A")
        print(f"Nerfstudio Available: {env['nerfstudio_available']} ({env['nerfstudio_version'] or 'Not installed'})")
        print(f"ns-train Binary:      {env['ns_train_path'] or 'Not found'}")
        print(f"Splatfacto Support:   {'SUPPORTED' if env['supported'] else 'NOT AVAILABLE'}")
        if env.get("blocker_reason"):
            print(f"Blocker Reason:       {env['blocker_reason']}")
        print("========================================\n")
        sys.exit(0)

    if not args.images or not args.cameras or not args.output:
        parser.error("--images, --cameras, and --output are required when not using --check-available.")

    # Dataset preparation and execution
    from reconstruction.splatting.dataset import prepare_splatfacto_dataset

    dset_dir = args.output / "dataset"
    train_dir = args.output / "train"

    try:
        print("\n[1/2] Preparing Splatfacto dataset...")
        dset_info = prepare_splatfacto_dataset(
            images_dir=args.images,
            cameras_path=args.cameras,
            output_dir=dset_dir,
            pointcloud_path=args.pointcloud,
        )
        print(f"  Processed {dset_info['num_frames']} frames.")
        print(f"  Pointcloud init: {dset_info['pointcloud_initialization']['used']}")

        print("\n[2/2] Running Splatfacto training wrapper...")
        cfg = SplatfactoConfig(
            max_num_iterations=args.max_num_iterations,
            downscale_factor=args.downscale_factor,
            eval_step=args.eval_step,
            save_step=args.save_step,
            device=args.device,
        )
        report = run_splatfacto_training(
            dataset_dir=dset_dir,
            output_dir=train_dir,
            config=cfg,
        )
        print(f"  Status: {report['status']}")
        if report.get("error"):
            print(f"  Reason: {report['error']}")
        sys.exit(0 if report["status"] in ("SUCCESS", "NOT_AVAILABLE") else 1)

    except Exception as exc:
        print(f"\n[ERROR] Execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
