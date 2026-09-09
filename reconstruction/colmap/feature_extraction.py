"""COLMAP Feature Extraction wrapper for SIH26158 3D reconstruction."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.colmap.utils import (
    find_colmap_binary,
    is_cuda_available,
    run_colmap,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_image_dir(image_dir: Path) -> List[Path]:
    """Validates that the image directory exists and contains valid image files.

    Raises:
        FileNotFoundError: If image_dir does not exist or is not a directory.
        ValueError: If image_dir contains no valid image files.
    """
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]

    if not images:
        raise ValueError(
            f"No supported images found in '{image_dir}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
        )

    return sorted(images)


def extract_features(
    image_dir: str | Path,
    database_path: str | Path,
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    use_gpu: Optional[bool] = None,
    colmap_bin: Optional[str | Path] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs COLMAP feature extraction on the given image directory.

    Args:
        image_dir: Path to directory containing input images.
        database_path: Path where the SQLite database will be written.
        camera_model: Camera model type (default: 'SIMPLE_RADIAL').
        single_camera: If True, assumes all images share a single camera intrinsic.
        use_gpu: Force GPU usage on or off. If None, auto-detects CUDA availability.
        colmap_bin: Explicit path to COLMAP executable.
        extra_args: Additional command-line arguments to forward to COLMAP.

    Returns:
        Dict summarizing feature extraction parameters and results.
    """
    image_path = Path(image_dir).resolve()
    db_path = Path(database_path).resolve()
    binary = find_colmap_binary(colmap_bin)

    if not binary:
        raise FileNotFoundError("COLMAP executable not found.")

    # 1. Validate images
    images = validate_image_dir(image_path)
    logger.info("Found %d valid images in %s", len(images), image_path)

    # 2. Ensure database directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Determine GPU setting
    if use_gpu is None:
        gpu_enabled = is_cuda_available()
    else:
        gpu_enabled = use_gpu

    # 4. Construct COLMAP CLI arguments
    from reconstruction.colmap.utils import get_colmap_version
    ver_str = get_colmap_version(binary)
    is_colmap4 = "COLMAP 4" in ver_str or "COLMAP 5" in ver_str

    args = [
        "--database_path", str(db_path),
        "--image_path", str(image_path),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" if single_camera else "0",
    ]
    if is_colmap4:
        args.extend(["--FeatureExtraction.use_gpu", "1" if gpu_enabled else "0"])
    else:
        args.extend(["--SiftExtraction.use_gpu", "1" if gpu_enabled else "0"])

    if extra_args:
        args.extend(extra_args)

    logger.info("Extracting features with COLMAP (%s)...", binary.name)
    run_colmap("feature_extractor", args, colmap_bin=binary)

    return {
        "num_images": len(images),
        "database_path": str(db_path),
        "camera_model": camera_model,
        "single_camera": single_camera,
        "use_gpu": gpu_enabled,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Extract features using COLMAP")
    parser.add_argument("--images", required=True, type=Path, help="Path to input images directory")
    parser.add_argument("--database", required=True, type=Path, help="Path to output database.db")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL", help="Camera model (default: SIMPLE_RADIAL)")
    parser.add_argument("--gpu", action="store_true", default=None, help="Enable GPU SIFT extraction")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU SIFT extraction")
    parser.add_argument("--colmap-bin", type=Path, default=None, help="Custom path to COLMAP binary")

    args = parser.parse_args()

    try:
        res = extract_features(
            image_dir=args.images,
            database_path=args.database,
            camera_model=args.camera_model,
            use_gpu=args.gpu,
            colmap_bin=args.colmap_bin,
        )
        print(f"Feature extraction succeeded. {res['num_images']} images processed.")
    except Exception as exc:
        print(f"Feature extraction failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
