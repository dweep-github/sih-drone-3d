"""Benchmarking framework comparing COLMAP incremental, FastMap, COLMAP global SfM, VGGT, and Adaptive Engine.

Executes side-by-side evaluation on identical datasets, measuring:
- Input / registered images and registration rate
- Execution runtime (wall-clock seconds)
- Memory usage (RAM and peak GPU memory)
- Reprojection error statistics (Mean, Median, RMSE, Max px)
- 3D points count and trajectory continuity metrics
- Exports colmap_result.json, fastmap_result.json, global_sfm_result.json, vggt_result.json, adaptive_result.json, and comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.colmap.feature_extraction import validate_image_dir
from reconstruction.colmap.reconstruct import run_reconstruction as run_colmap_reconstruction
from reconstruction.colmap.utils import is_cuda_available
from reconstruction.pose_engine.engine import ValidationPolicy, run_adaptive_engine
from reconstruction.pose_engine.fastmap import get_fastmap_environment_info, run_fastmap
from reconstruction.pose_engine.global_sfm import run_global_sfm
from reconstruction.pose_engine.vggt import get_vggt_environment_info, run_vggt
from reconstruction.validation.pose_report import validate_reconstruction

logger = logging.getLogger("benchmark")


def measure_peak_gpu_mb() -> Any:
    """Measures peak GPU memory allocated by PyTorch if CUDA is available."""
    if is_cuda_available():
        try:
            import torch
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        except Exception:
            return "N/A"
    return "N/A (No CUDA GPU)"


def get_current_ram_mb() -> float:
    """Returns current process RAM usage in MB."""
    try:
        proc = psutil.Process()
        return round(proc.memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


def benchmark_colmap(
    image_dir: Path,
    output_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    matcher: str = "sequential",
) -> Dict[str, Any]:
    """Runs and benchmarks the baseline COLMAP incremental pipeline."""
    colmap_out = output_dir / "colmap_run"
    colmap_out.mkdir(parents=True, exist_ok=True)

    ram_before = get_current_ram_mb()
    start_time = time.time()
    try:
        report = run_colmap_reconstruction(
            image_dir=image_dir,
            output_dir=colmap_out,
            camera_model=camera_model,
            matcher=matcher,
        )
        runtime = round(time.time() - start_time, 4)
        ram_after = get_current_ram_mb()

        # Run validation
        val_report = validate_reconstruction(
            model_dir=colmap_out / "sparse",
            image_dir=image_dir,
            output_report_path=colmap_out / "pose_report.json",
        )

        rep = val_report.get("reprojection", {})
        traj = val_report.get("trajectory", {})

        return {
            "backend": "COLMAP_Incremental",
            "status": val_report.get("validation_status", "FAIL"),
            "runtime_seconds": runtime,
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": report.get("input_images", 0),
            "registered_images": report.get("registered_images", 0),
            "registration_rate": report.get("registration_rate", 0.0),
            "num_points3D": report.get("num_points3D", 0),
            "reprojection_mean_px": rep.get("mean_px", "N/A"),
            "reprojection_median_px": rep.get("median_px", "N/A"),
            "reprojection_rmse_px": rep.get("rmse_px", "N/A"),
            "reprojection_max_px": rep.get("max_px", "N/A"),
            "trajectory_status": traj.get("status", "NOT_AVAILABLE"),
            "trajectory_mean_step": traj.get("mean_step", "N/A"),
            "trajectory_max_step": traj.get("max_step", "N/A"),
            "trajectory_mean_rotation_deg": traj.get("mean_rotation_change_deg", "N/A"),
            "trajectory_max_rotation_deg": traj.get("max_rotation_change_deg", "N/A"),
            "errors": val_report.get("errors", []),
        }

    except Exception as exc:
        runtime = round(time.time() - start_time, 4)
        return {
            "backend": "COLMAP_Incremental",
            "status": "FAIL",
            "runtime_seconds": runtime,
            "peak_ram_mb": get_current_ram_mb(),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": 0,
            "registered_images": 0,
            "registration_rate": 0.0,
            "num_points3D": 0,
            "reprojection_mean_px": "N/A",
            "reprojection_median_px": "N/A",
            "reprojection_rmse_px": "N/A",
            "reprojection_max_px": "N/A",
            "trajectory_status": "NOT_AVAILABLE",
            "trajectory_mean_step": "N/A",
            "trajectory_max_step": "N/A",
            "trajectory_mean_rotation_deg": "N/A",
            "trajectory_max_rotation_deg": "N/A",
            "errors": [str(exc)],
        }


def benchmark_fastmap(
    image_dir: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Runs and benchmarks FastMap or records exact environment blocker."""
    fastmap_out = output_dir / "fastmap_run"
    fastmap_out.mkdir(parents=True, exist_ok=True)

    ram_before = get_current_ram_mb()
    try:
        res = run_fastmap(
            images=image_dir,
            output=fastmap_out,
            device=device,
            headless=True,
        )
        ram_after = get_current_ram_mb()

        rep = res.get("reprojection", {})
        traj = res.get("trajectory", {})

        return {
            "backend": "FastMap",
            "status": res.get("status", "FAIL"),
            "runtime_seconds": res.get("runtime_seconds", 0.0),
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": res.get("input_images", 0),
            "registered_images": res.get("registered_images", 0),
            "registration_rate": res.get("registration_rate", 0.0),
            "num_points3D": res.get("num_points3D", 0),
            "reprojection_mean_px": rep.get("mean_px", "N/A"),
            "reprojection_median_px": rep.get("median_px", "N/A"),
            "reprojection_rmse_px": rep.get("rmse_px", "N/A"),
            "reprojection_max_px": rep.get("max_px", "N/A"),
            "trajectory_status": traj.get("status", "NOT_AVAILABLE"),
            "trajectory_mean_step": traj.get("mean_step", "N/A"),
            "trajectory_max_step": traj.get("max_step", "N/A"),
            "trajectory_mean_rotation_deg": traj.get("mean_rotation_change_deg", "N/A"),
            "trajectory_max_rotation_deg": traj.get("max_rotation_change_deg", "N/A"),
            "blocker_reason": res.get("blocker_reason"),
            "errors": res.get("errors", []),
        }
    except Exception as exc:
        ram_after = get_current_ram_mb()
        env_info = get_fastmap_environment_info()
        return {
            "backend": "FastMap",
            "status": "BLOCKED" if not env_info["supported"] else "FAIL",
            "runtime_seconds": 0.0,
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": 0,
            "registered_images": 0,
            "registration_rate": 0.0,
            "num_points3D": 0,
            "reprojection_mean_px": "N/A",
            "reprojection_median_px": "N/A",
            "reprojection_rmse_px": "N/A",
            "reprojection_max_px": "N/A",
            "trajectory_status": "NOT_AVAILABLE",
            "trajectory_mean_step": "N/A",
            "trajectory_max_step": "N/A",
            "trajectory_mean_rotation_deg": "N/A",
            "trajectory_max_rotation_deg": "N/A",
            "blocker_reason": env_info.get("blocker_reason"),
            "errors": [str(exc)],
        }


def benchmark_global_sfm(
    image_dir: Path,
    output_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    matcher: str = "sequential",
) -> Dict[str, Any]:
    """Runs and benchmarks COLMAP Global SfM (global_mapper)."""
    gsfm_out = output_dir / "global_sfm_run"
    gsfm_out.mkdir(parents=True, exist_ok=True)

    ram_before = get_current_ram_mb()
    try:
        res = run_global_sfm(
            images=image_dir,
            output=gsfm_out,
            camera_model=camera_model,
            matcher=matcher,
        )
        ram_after = get_current_ram_mb()

        # Check pose report generated by global_sfm
        val_path = gsfm_out / "pose_report.json"
        if val_path.is_file():
            with open(val_path, "r", encoding="utf-8") as f:
                val_report = json.load(f)
            rep = val_report.get("reprojection", {})
            traj = val_report.get("trajectory", {})
        else:
            rep = {}
            traj = {}

        return {
            "backend": "COLMAP_Global_SfM",
            "status": res.get("status", "FAIL"),
            "runtime_seconds": res.get("runtime_seconds", 0.0),
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": res.get("input_images", 0),
            "registered_images": res.get("registered_images", 0),
            "registration_rate": res.get("registration_rate", 0.0),
            "num_points3D": res.get("num_points3D", 0),
            "reprojection_mean_px": rep.get("mean_px", "N/A"),
            "reprojection_median_px": rep.get("median_px", "N/A"),
            "reprojection_rmse_px": res.get("reprojection_rmse_px") or rep.get("rmse_px", "N/A"),
            "reprojection_max_px": rep.get("max_px", "N/A"),
            "trajectory_status": traj.get("status", "NOT_AVAILABLE"),
            "trajectory_mean_step": traj.get("mean_step", "N/A"),
            "trajectory_max_step": traj.get("max_step", "N/A"),
            "trajectory_mean_rotation_deg": traj.get("mean_rotation_change_deg", "N/A"),
            "trajectory_max_rotation_deg": traj.get("max_rotation_change_deg", "N/A"),
            "errors": res.get("errors", []),
            "warnings": res.get("warnings", []),
        }

    except Exception as exc:
        ram_after = get_current_ram_mb()
        return {
            "backend": "COLMAP_Global_SfM",
            "status": "FAIL",
            "runtime_seconds": 0.0,
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": 0,
            "registered_images": 0,
            "registration_rate": 0.0,
            "num_points3D": 0,
            "reprojection_mean_px": "N/A",
            "reprojection_median_px": "N/A",
            "reprojection_rmse_px": "N/A",
            "reprojection_max_px": "N/A",
            "trajectory_status": "NOT_AVAILABLE",
            "trajectory_mean_step": "N/A",
            "trajectory_max_step": "N/A",
            "trajectory_mean_rotation_deg": "N/A",
            "trajectory_max_rotation_deg": "N/A",
            "errors": [str(exc)],
        }


def benchmark_vggt(
    image_dir: Path,
    output_dir: Path,
    checkpoint: str = "facebook/VGGT-1B",
    max_images: Optional[int] = 50,
    device: Optional[str] = None,
    mock_mode: bool = False,
) -> Dict[str, Any]:
    """Runs and benchmarks VGGT or records exact environment blocker."""
    vggt_out = output_dir / "vggt_run"
    vggt_out.mkdir(parents=True, exist_ok=True)

    ram_before = get_current_ram_mb()
    try:
        res = run_vggt(
            images=image_dir,
            output=vggt_out,
            checkpoint=checkpoint,
            max_images=max_images,
            device=device,
            mock_mode=mock_mode,
        )
        ram_after = get_current_ram_mb()

        rep = res.get("reprojection", {})
        traj = res.get("trajectory", {})

        peak_gpu = res.get("peak_gpu_memory_mb")
        gpu_str = f"{peak_gpu} MB" if peak_gpu is not None else measure_peak_gpu_mb()

        reg_count = res.get("camera_predictions", 0)
        inp_count = res.get("input_images", 0)
        reg_rate = round(reg_count / max(inp_count, 1), 4) if inp_count > 0 else 0.0

        return {
            "backend": "VGGT",
            "status": res.get("status", "FAIL"),
            "runtime_seconds": res.get("runtime_seconds", 0.0),
            "model_load_time_seconds": res.get("model_load_time_seconds", 0.0),
            "inference_time_seconds": res.get("inference_time_seconds", 0.0),
            "postprocess_time_seconds": res.get("postprocess_time_seconds", 0.0),
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": gpu_str,
            "input_images": inp_count,
            "registered_images": reg_count,
            "registration_rate": reg_rate,
            "num_points3D": res.get("num_points3D", 0),
            "reprojection_mean_px": rep.get("mean_px", "N/A"),
            "reprojection_median_px": rep.get("median_px", "N/A"),
            "reprojection_rmse_px": rep.get("rmse_px", "N/A"),
            "reprojection_max_px": rep.get("max_px", "N/A"),
            "trajectory_status": traj.get("status", "NOT_AVAILABLE"),
            "trajectory_mean_step": traj.get("mean_step", "N/A"),
            "trajectory_max_step": traj.get("max_step", "N/A"),
            "trajectory_mean_rotation_deg": traj.get("mean_rotation_change_deg", "N/A"),
            "trajectory_max_rotation_deg": traj.get("max_rotation_change_deg", "N/A"),
            "blocker_reason": res.get("blocker_reason"),
            "errors": res.get("errors", []),
            "warnings": res.get("warnings", []),
        }
    except Exception as exc:
        ram_after = get_current_ram_mb()
        env_info = get_vggt_environment_info()
        return {
            "backend": "VGGT",
            "status": "BLOCKED" if not env_info["supported"] else "FAIL",
            "runtime_seconds": 0.0,
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": 0,
            "registered_images": 0,
            "registration_rate": 0.0,
            "num_points3D": 0,
            "reprojection_mean_px": "N/A",
            "reprojection_median_px": "N/A",
            "reprojection_rmse_px": "N/A",
            "reprojection_max_px": "N/A",
            "trajectory_status": "NOT_AVAILABLE",
            "trajectory_mean_step": "N/A",
            "trajectory_max_step": "N/A",
            "trajectory_mean_rotation_deg": "N/A",
            "trajectory_max_rotation_deg": "N/A",
            "blocker_reason": env_info.get("blocker_reason"),
            "errors": [str(exc)],
        }


def benchmark_adaptive(
    image_dir: Path,
    output_dir: Path,
    device: str = "cuda:0",
    min_images: int = 20,
    mock_backends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs and benchmarks the Adaptive Pose Engine."""
    adaptive_out = output_dir / "adaptive_run"
    adaptive_out.mkdir(parents=True, exist_ok=True)

    ram_before = get_current_ram_mb()
    try:
        report = run_adaptive_engine(
            images=image_dir,
            output=adaptive_out,
            method="auto",
            min_images=min_images,
            device=device,
            mock_backends=mock_backends,
        )
        ram_after = get_current_ram_mb()

        attempts = report.get("attempts", [])
        selected_method = report.get("selected_method")
        overall_status = report.get("overall_status", "FAIL")

        winning_att = next((att for att in attempts if att.get("decision") == "ACCEPT"), None)
        val = winning_att.get("validation", {}) if winning_att else {}

        reg_count = val.get("registered_images", 0)
        inp_count = report.get("preflight", {}).get("total_images", 0)
        reg_rate = val.get("registration_rate", 0.0)

        return {
            "backend": "Adaptive_Engine",
            "status": overall_status,
            "selected_method": selected_method,
            "attempts_count": len(attempts),
            "runtime_seconds": report.get("total_runtime_seconds", 0.0),
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": inp_count,
            "registered_images": reg_count,
            "registration_rate": reg_rate,
            "num_points3D": 0,
            "reprojection_mean_px": "N/A",
            "reprojection_median_px": "N/A",
            "reprojection_rmse_px": val.get("reprojection_rmse_px") or "N/A",
            "reprojection_max_px": "N/A",
            "trajectory_status": val.get("trajectory_status", "NOT_AVAILABLE"),
            "trajectory_mean_step": "N/A",
            "trajectory_max_step": "N/A",
            "trajectory_mean_rotation_deg": "N/A",
            "trajectory_max_rotation_deg": "N/A",
            "attempts": attempts,
            "errors": [att.get("rejection_reason") for att in attempts if att.get("rejection_reason")] if overall_status != "PASS" else [],
        }
    except Exception as exc:
        ram_after = get_current_ram_mb()
        return {
            "backend": "Adaptive_Engine",
            "status": "FAIL",
            "selected_method": None,
            "attempts_count": 0,
            "runtime_seconds": 0.0,
            "peak_ram_mb": max(ram_before, ram_after),
            "peak_gpu_memory": measure_peak_gpu_mb(),
            "input_images": 0,
            "registered_images": 0,
            "registration_rate": 0.0,
            "num_points3D": 0,
            "reprojection_mean_px": "N/A",
            "reprojection_median_px": "N/A",
            "reprojection_rmse_px": "N/A",
            "reprojection_max_px": "N/A",
            "trajectory_status": "NOT_AVAILABLE",
            "trajectory_mean_step": "N/A",
            "trajectory_max_step": "N/A",
            "trajectory_mean_rotation_deg": "N/A",
            "trajectory_max_rotation_deg": "N/A",
            "errors": [str(exc)],
        }


def print_comparison_table(
    colmap_res: Dict[str, Any],
    fastmap_res: Dict[str, Any],
    gsfm_res: Dict[str, Any],
    vggt_res: Dict[str, Any],
    adaptive_res: Dict[str, Any],
) -> None:
    """Prints a clean, formatted 5-way comparison table to terminal."""
    metrics = [
        ("Status", "status"),
        ("Input images", "input_images"),
        ("Registered images", "registered_images"),
        ("Registration rate", "registration_rate"),
        ("3D points", "num_points3D"),
        ("Runtime (s)", "runtime_seconds"),
        ("Peak RAM (MB)", "peak_ram_mb"),
        ("Peak GPU memory", "peak_gpu_memory"),
        ("Reprojection RMSE", "reprojection_rmse_px"),
        ("Reprojection Max", "reprojection_max_px"),
        ("Trajectory status", "trajectory_status"),
        ("Mean step", "trajectory_mean_step"),
        ("Max step", "trajectory_max_step"),
    ]

    print("\n================================================================================================")
    print("STRUCTURE FROM MOTION 5-WAY BENCHMARK: INCR vs FASTMAP vs GLOBAL vs VGGT vs ADAPTIVE")
    print("================================================================================================")
    header = f"{'Metric':<20} | {'COLMAP':<11} | {'FastMap':<10} | {'Global':<10} | {'VGGT':<10} | {'Adaptive':<10}"
    print(header)
    print("-" * len(header))

    for label, key in metrics:
        c_val = colmap_res.get(key, "N/A")
        f_val = fastmap_res.get(key, "N/A")
        g_val = gsfm_res.get(key, "N/A")
        v_val = vggt_res.get(key, "N/A")
        a_val = adaptive_res.get(key, "N/A")

        c_str = f"{c_val:.2f}" if isinstance(c_val, float) else str(c_val)
        f_str = f"{f_val:.2f}" if isinstance(f_val, float) else str(f_val)
        g_str = f"{g_val:.2f}" if isinstance(g_val, float) else str(g_val)
        v_str = f"{v_val:.2f}" if isinstance(v_val, float) else str(v_val)
        a_str = f"{a_val:.2f}" if isinstance(a_val, float) else str(a_val)

        print(f"{label:<20} | {c_str:<11} | {f_str:<10} | {g_str:<10} | {v_str:<10} | {a_str:<10}")

    print("================================================================================================\n")


def run_benchmark(
    images: str | Path,
    output: str | Path,
    device: str = "cuda:0",
    vggt_checkpoint: str = "facebook/VGGT-1B",
    vggt_max_images: Optional[int] = 50,
    mock_vggt: bool = False,
    min_images: int = 20,
    mock_backends: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs comparative benchmark on identical dataset across all 5 reconstruction routes."""
    img_path = Path(images).resolve()
    out_path = Path(output).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate image directory
    try:
        valid_images = validate_image_dir(img_path)
        img_count = len(valid_images)
    except Exception:
        img_count = 0

    colmap_res = benchmark_colmap(img_path, out_path)
    fastmap_res = benchmark_fastmap(img_path, out_path, device=device)
    gsfm_res = benchmark_global_sfm(img_path, out_path)
    vggt_res = benchmark_vggt(
        img_path,
        out_path,
        checkpoint=vggt_checkpoint,
        max_images=vggt_max_images,
        device=device,
        mock_mode=mock_vggt,
    )
    adaptive_res = benchmark_adaptive(
        img_path,
        out_path,
        device=device,
        min_images=min_images,
        mock_backends=mock_backends,
    )

    comparison = {
        "dataset": {
            "image_path": str(img_path),
            "image_count": img_count,
        },
        "experiment": {
            "dataset": str(img_path),
            "image_count": img_count,
            "machine": sys.platform,
            "python": sys.version.split()[0],
            "cuda_available": is_cuda_available(),
        },
        "colmap": colmap_res,
        "colmap_incremental": colmap_res,
        "fastmap": fastmap_res,
        "global_sfm": gsfm_res,
        "colmap_global_sfm": gsfm_res,
        "vggt": vggt_res,
        "adaptive": adaptive_res,
        "adaptive_engine": adaptive_res,
    }

    # Save outputs
    with open(out_path / "colmap_result.json", "w", encoding="utf-8") as f:
        json.dump(colmap_res, f, indent=2)

    with open(out_path / "fastmap_result.json", "w", encoding="utf-8") as f:
        json.dump(fastmap_res, f, indent=2)

    with open(out_path / "global_sfm_result.json", "w", encoding="utf-8") as f:
        json.dump(gsfm_res, f, indent=2)

    with open(out_path / "vggt_result.json", "w", encoding="utf-8") as f:
        json.dump(vggt_res, f, indent=2)

    with open(out_path / "adaptive_result.json", "w", encoding="utf-8") as f:
        json.dump(adaptive_res, f, indent=2)

    with open(out_path / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print_comparison_table(colmap_res, fastmap_res, gsfm_res, vggt_res, adaptive_res)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIH26158 Person 1: Incremental vs FastMap vs Global SfM vs VGGT vs Adaptive Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", required=True, type=Path, help="Input images directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reconstruction/test_output/benchmark"),
        help="Directory to save benchmark reports",
    )
    parser.add_argument("--device", default="cuda:0", help="Execution device for FastMap and VGGT")
    parser.add_argument(
        "--vggt-checkpoint",
        default="facebook/VGGT-1B",
        help="Hugging Face checkpoint identifier for VGGT",
    )
    parser.add_argument(
        "--vggt-max-images",
        type=int,
        default=50,
        help="Maximum representative images for VGGT",
    )
    parser.add_argument(
        "--mock-vggt",
        action="store_true",
        default=False,
        help="Run VGGT in mock mode for verification",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=20,
        help="Minimum image count for preflight gate",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            images=args.images,
            output=args.output,
            device=args.device,
            vggt_checkpoint=args.vggt_checkpoint,
            vggt_max_images=args.vggt_max_images,
            mock_vggt=args.mock_vggt,
            min_images=args.min_images,
        )
        sys.exit(0)
    except Exception as exc:
        print(f"[ERROR] Benchmark execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
