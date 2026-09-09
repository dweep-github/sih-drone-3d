#!/usr/bin/env python3
"""Environment diagnostic and verification script for SIH26158 3D Reconstruction Pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so reconstruction module can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_environment(require_cuda: bool = False) -> int:
    """Checks and prints the current runtime environment for 3D reconstruction."""
    missing_required: list[str] = []

    # 1. Python
    python_version = sys.version.split()[0]

    # 2. PyTorch & CUDA
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "None (CPU only)"
            if require_cuda:
                missing_required.append("CUDA-capable GPU")
    except ImportError:
        pytorch_version = "Missing"
        cuda_available = False
        cuda_version = None
        gpu_name = "None"
        missing_required.append("torch")

    # 3. OpenCV
    try:
        import cv2
        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = "Missing"
        missing_required.append("opencv-python")

    # 4. COLMAP
    try:
        from reconstruction.colmap.utils import find_colmap_binary, get_colmap_version
        colmap_bin = find_colmap_binary()
        if colmap_bin:
            colmap_version = get_colmap_version(colmap_bin)
        else:
            colmap_version = "Not found"
            missing_required.append("COLMAP executable")
    except Exception as exc:
        colmap_version = f"Error detecting: {exc}"
        missing_required.append("COLMAP executable")

    # Output formatted report
    print("========================================")
    print("SIH26158 RECONSTRUCTION ENVIRONMENT")
    print("========================================")
    print(f"Python:         {python_version}")
    print(f"PyTorch:        {pytorch_version}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA version:   {cuda_version if cuda_version is not None else 'None'}")
    print(f"GPU:            {gpu_name}")
    print(f"OpenCV:         {opencv_version}")
    print(f"COLMAP:         {colmap_version}")
    print("========================================")

    if missing_required:
        print(f"\n[ERROR] Missing required dependencies: {', '.join(missing_required)}")
        return 1

    if not cuda_available:
        print("\n[NOTE] Running without CUDA GPU acceleration. Reconstruction and feature extraction will run on CPU.")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SIH26158 3D reconstruction environment")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA GPU is not available")
    args = parser.parse_args()

    exit_code = check_environment(require_cuda=args.require_cuda)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
