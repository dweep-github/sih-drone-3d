"""COLMAP Feature Matching wrapper for SIH26158 3D reconstruction."""

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

SUPPORTED_MATCHERS = {
    "sequential": "sequential_matcher",
    "exhaustive": "exhaustive_matcher",
    "spatial": "spatial_matcher",
    "vocab_tree": "vocab_tree_matcher",
}


def match_features(
    database_path: str | Path,
    matcher: str = "sequential",
    overlap: int = 10,
    use_gpu: Optional[bool] = None,
    colmap_bin: Optional[str | Path] = None,
    extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs COLMAP feature matching on the given database.

    Args:
        database_path: Path to the SQLite database containing extracted features.
        matcher: Matching strategy ('sequential', 'exhaustive', 'spatial', 'vocab_tree').
        overlap: Consecutive frame overlap window for sequential matching.
        use_gpu: Force GPU usage on or off. If None, auto-detects CUDA availability.
        colmap_bin: Explicit path to COLMAP executable.
        extra_args: Additional command-line arguments for COLMAP matcher.

    Returns:
        Dict summarizing matching parameters and execution status.
    """
    db_path = Path(database_path).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"COLMAP database does not exist: {db_path}")

    matcher_key = matcher.lower().strip()
    if matcher_key not in SUPPORTED_MATCHERS:
        supported_names = ", ".join(sorted(SUPPORTED_MATCHERS.keys()))
        raise ValueError(
            f"Unsupported matcher '{matcher}'. Supported options: {supported_names}"
        )

    subcommand = SUPPORTED_MATCHERS[matcher_key]
    binary = find_colmap_binary(colmap_bin)
    if not binary:
        raise FileNotFoundError("COLMAP executable not found.")

    if use_gpu is None:
        gpu_enabled = is_cuda_available()
    else:
        gpu_enabled = use_gpu

    from reconstruction.colmap.utils import get_colmap_version
    ver_str = get_colmap_version(binary)
    is_colmap4 = "COLMAP 4" in ver_str or "COLMAP 5" in ver_str

    args = ["--database_path", str(db_path)]
    if is_colmap4:
        args.extend(["--FeatureMatching.use_gpu", "1" if gpu_enabled else "0"])
    else:
        args.extend(["--SiftMatching.use_gpu", "1" if gpu_enabled else "0"])

    if matcher_key == "sequential":
        args.extend(["--SequentialMatching.overlap", str(overlap)])

    if extra_args:
        args.extend(extra_args)

    logger.info(
        "Running feature matching using '%s' (GPU=%s)...",
        subcommand,
        gpu_enabled,
    )
    run_colmap(subcommand, args, colmap_bin=binary)

    return {
        "database_path": str(db_path),
        "matcher": matcher_key,
        "overlap": overlap if matcher_key == "sequential" else None,
        "use_gpu": gpu_enabled,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Match features using COLMAP")
    parser.add_argument("--database", required=True, type=Path, help="Path to database.db")
    parser.add_argument(
        "--matcher",
        default="sequential",
        choices=list(SUPPORTED_MATCHERS.keys()),
        help="Feature matching algorithm (default: sequential)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Frame overlap window for sequential matching (default: 10)",
    )
    parser.add_argument("--gpu", action="store_true", default=None, help="Enable GPU matching")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu", help="Disable GPU matching")
    parser.add_argument("--colmap-bin", type=Path, default=None, help="Custom path to COLMAP binary")

    args = parser.parse_args()

    try:
        res = match_features(
            database_path=args.database,
            matcher=args.matcher,
            overlap=args.overlap,
            use_gpu=args.gpu,
            colmap_bin=args.colmap_bin,
        )
        print(f"Feature matching succeeded using {res['matcher']}.")
    except Exception as exc:
        print(f"Feature matching failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
