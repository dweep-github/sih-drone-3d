"""Utilities for COLMAP binary discovery, environment setup, and subprocess execution."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Checks if PyTorch has CUDA available for GPU acceleration."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def find_colmap_binary(explicit_path: Optional[str | Path] = None) -> Optional[Path]:
    """Finds the COLMAP executable across explicit args, env vars, PATH, and local tools."""
    # 1. Explicit path passed by caller
    if explicit_path:
        p = Path(explicit_path).resolve()
        if p.is_file():
            return p
        if p.is_dir():
            candidate = p / ("colmap.exe" if sys.platform == "win32" else "colmap")
            if candidate.is_file():
                return candidate

    # 2. COLMAP_EXE environment variable
    env_path = os.environ.get("COLMAP_EXE")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return p

    # 3. System PATH
    system_colmap = shutil.which("colmap")
    if system_colmap:
        return Path(system_colmap).resolve()

    # 4. Project-bundled binary in tools/colmap/bin
    current_file = Path(__file__).resolve()
    # Go up from reconstruction/colmap/utils.py to project root
    project_root = current_file.parents[2]
    local_colmap_bin = project_root / "tools" / "colmap" / "bin" / (
        "colmap.exe" if sys.platform == "win32" else "colmap"
    )
    if local_colmap_bin.is_file():
        return local_colmap_bin

    return None


def get_colmap_env(colmap_bin: Optional[Path] = None) -> Dict[str, str]:
    """Prepares execution environment for COLMAP, ensuring companion DLLs/libraries are found."""
    env = os.environ.copy()
    if colmap_bin and colmap_bin.exists():
        bin_dir = colmap_bin.parent
        # Prepend binary directory to PATH for shared library / DLL discovery
        current_path = env.get("PATH", "")
        env["PATH"] = str(bin_dir) + os.pathsep + current_path

        # If Qt plugins directory exists beside bin, set QT_PLUGIN_PATH
        plugins_dir = bin_dir.parent / "plugins"
        if plugins_dir.is_dir():
            env["QT_PLUGIN_PATH"] = str(plugins_dir)

    return env


def get_colmap_version(colmap_bin: Optional[Path] = None) -> str:
    """Queries COLMAP for its version string."""
    binary = colmap_bin or find_colmap_binary()
    if not binary:
        return "Not found"

    env = get_colmap_env(binary)
    try:
        # colmap -h prints version on the first line
        result = subprocess.run(
            [str(binary), "-h"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        output = result.stdout or result.stderr
        for line in output.splitlines():
            clean = line.strip()
            if clean.startswith("COLMAP"):
                return clean
        return "COLMAP (unknown version)"
    except Exception as exc:
        logger.warning("Failed to execute colmap to determine version: %s", exc)
        return "Execution failed"


def run_colmap(
    subcommand: str,
    args: List[str],
    colmap_bin: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Executes a COLMAP subcommand with provided arguments and environment.

    Args:
        subcommand: The COLMAP command (e.g., 'feature_extractor', 'sequential_matcher', 'mapper')
        args: Command-line arguments for the subcommand
        colmap_bin: Explicit path to colmap binary
        cwd: Working directory for execution

    Returns:
        CompletedProcess object

    Raises:
        FileNotFoundError: If colmap executable cannot be found
        RuntimeError: If colmap subcommand execution returns non-zero code
    """
    binary = colmap_bin or find_colmap_binary()
    if not binary:
        raise FileNotFoundError(
            "COLMAP executable could not be found. Please install COLMAP, add it to PATH, "
            "or set the COLMAP_EXE environment variable."
        )

    env = get_colmap_env(binary)
    cmd = [str(binary), subcommand] + args

    logger.debug("Running COLMAP command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
    )

    if result.returncode != 0:
        error_msg = (
            f"COLMAP command '{subcommand}' failed with return code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return result
