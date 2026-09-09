"""COLMAP sparse reconstruction model I/O reader (binary and text formats).

Provides robust, self-contained parsing of cameras, images, and points3D
from COLMAP sparse reconstruction directories.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_MODEL_IDS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_NAMES: Dict[str, Tuple[int, int]] = {
    name: (mid, n_params) for mid, (name, n_params) in CAMERA_MODEL_IDS.items()
}


@dataclass
class Camera:
    camera_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray  # float64 array of intrinsic parameters


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray  # [qw, qx, qy, qz]
    tvec: np.ndarray  # [tx, ty, tz]
    camera_id: int
    name: str
    xys: np.ndarray  # Nx2 array of 2D feature coordinates
    point3D_ids: np.ndarray  # N array of point3D IDs (-1 if unassociated)


@dataclass
class Point3D:
    point3D_id: int
    xyz: np.ndarray  # [x, y, z]
    rgb: np.ndarray  # [r, g, b]
    error: float
    image_ids: List[int] = field(default_factory=list)
    point2D_idxs: List[int] = field(default_factory=list)


@dataclass
class ReconstructionModel:
    cameras: Dict[int, Camera] = field(default_factory=dict)
    images: Dict[int, ImagePose] = field(default_factory=dict)
    points3D: Dict[int, Point3D] = field(default_factory=dict)
    model_dir: Optional[Path] = None

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_points3D(self) -> int:
        return len(self.points3D)


def read_cameras_binary(path_to_model_file: Path) -> Dict[int, Camera]:
    """Reads cameras from COLMAP cameras.bin."""
    cameras: Dict[int, Camera] = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", fid.read(24))
            if model_id in CAMERA_MODEL_IDS:
                model_name, num_params = CAMERA_MODEL_IDS[model_id]
            else:
                model_name = f"UNKNOWN_{model_id}"
                num_params = 0

            params = struct.unpack(f"<{num_params}d", fid.read(8 * num_params))
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model_name=model_name,
                width=width,
                height=height,
                params=np.array(params, dtype=np.float64),
            )
    return cameras


def read_images_binary(path_to_model_file: Path) -> Dict[int, ImagePose]:
    """Reads registered images from COLMAP images.bin."""
    images: Dict[int, ImagePose] = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            image_id = struct.unpack("<I", fid.read(4))[0]
            qvec = np.array(struct.unpack("<4d", fid.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", fid.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", fid.read(4))[0]

            image_name_chars = []
            while True:
                char = fid.read(1)
                if char == b"\x00" or not char:
                    break
                image_name_chars.append(char.decode("utf-8", errors="replace"))
            image_name = "".join(image_name_chars)

            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            if num_points2D > 0:
                raw_points = fid.read(num_points2D * 24)
                # Each point: 2 doubles (x, y) + 1 uint64 (point3D_id)
                xys = np.empty((num_points2D, 2), dtype=np.float64)
                point3D_ids = np.empty(num_points2D, dtype=np.int64)

                fmt = "<2dQ"
                entry_size = 24
                for i in range(num_points2D):
                    offset = i * entry_size
                    x, y, p3d_id = struct.unpack_from(fmt, raw_points, offset)
                    xys[i, 0] = x
                    xys[i, 1] = y
                    # COLMAP uses (1 << 64) - 1 or -1 for unassociated points
                    point3D_ids[i] = -1 if p3d_id == 0xFFFFFFFFFFFFFFFF else int(p3d_id)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point3D_ids = np.empty(0, dtype=np.int64)

            images[image_id] = ImagePose(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path_to_model_file: Path) -> Dict[int, Point3D]:
    """Reads 3D points from COLMAP points3D.bin."""
    points3D: Dict[int, Point3D] = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", fid.read(8))[0]
            xyz = np.array(struct.unpack("<3d", fid.read(24)), dtype=np.float64)
            rgb = np.array(struct.unpack("<3B", fid.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", fid.read(8))[0]
            track_len = struct.unpack("<Q", fid.read(8))[0]

            image_ids: List[int] = []
            point2D_idxs: List[int] = []
            if track_len > 0:
                track_raw = fid.read(track_len * 8)
                for i in range(track_len):
                    img_id, p2d_idx = struct.unpack_from("<II", track_raw, i * 8)
                    image_ids.append(img_id)
                    point2D_idxs.append(p2d_idx)

            points3D[point3D_id] = Point3D(
                point3D_id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def read_cameras_text(path_to_model_file: Path) -> Dict[int, Camera]:
    """Reads cameras from COLMAP cameras.txt."""
    cameras: Dict[int, Camera] = {}
    with open(path_to_model_file, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float64)
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model_name=model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_images_text(path_to_model_file: Path) -> Dict[int, ImagePose]:
    """Reads images from COLMAP images.txt."""
    images: Dict[int, ImagePose] = {}
    with open(path_to_model_file, "r", encoding="utf-8") as fid:
        lines = [line.rstrip("\r\n") for line in fid if not line.startswith("#")]

    idx = 0
    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
        line1 = lines[idx].split()
        idx += 1
        line2 = lines[idx].split() if idx < len(lines) else []
        idx += 1

        image_id = int(line1[0])
        qvec = np.array([float(x) for x in line1[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in line1[5:8]], dtype=np.float64)
        camera_id = int(line1[8])
        name = line1[9]

        num_points = len(line2) // 3
        xys = np.empty((num_points, 2), dtype=np.float64)
        point3D_ids = np.empty(num_points, dtype=np.int64)

        for i in range(num_points):
            xys[i, 0] = float(line2[i * 3])
            xys[i, 1] = float(line2[i * 3 + 1])
            point3D_ids[i] = int(line2[i * 3 + 2])

        images[image_id] = ImagePose(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            xys=xys,
            point3D_ids=point3D_ids,
        )
    return images


def read_points3D_text(path_to_model_file: Path) -> Dict[int, Point3D]:
    """Reads 3D points from COLMAP points3D.txt."""
    points3D: Dict[int, Point3D] = {}
    with open(path_to_model_file, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point3D_id = int(parts[0])
            xyz = np.array([float(x) for x in parts[1:4]], dtype=np.float64)
            rgb = np.array([int(x) for x in parts[4:7]], dtype=np.uint8)
            error = float(parts[7])

            image_ids: List[int] = []
            point2D_idxs: List[int] = []
            track_parts = parts[8:]
            for i in range(0, len(track_parts), 2):
                image_ids.append(int(track_parts[i]))
                point2D_idxs.append(int(track_parts[i + 1]))

            points3D[point3D_id] = Point3D(
                point3D_id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def read_colmap_model(model_dir: str | Path) -> ReconstructionModel:
    """Loads a COLMAP sparse model from binary or text format.

    Args:
        model_dir: Directory containing cameras/images/points3D (.bin or .txt)

    Returns:
        ReconstructionModel dataclass

    Raises:
        FileNotFoundError: If model files are missing
    """
    path = Path(model_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")

    # Check for binary files first
    cameras_bin = path / "cameras.bin"
    images_bin = path / "images.bin"
    points3d_bin = path / "points3D.bin"

    if cameras_bin.is_file() and images_bin.is_file() and points3d_bin.is_file():
        cameras = read_cameras_binary(cameras_bin)
        images = read_images_binary(images_bin)
        points3D = read_points3D_binary(points3d_bin)
        return ReconstructionModel(cameras=cameras, images=images, points3D=points3D, model_dir=path)

    # Check for text files
    cameras_txt = path / "cameras.txt"
    images_txt = path / "images.txt"
    points3d_txt = path / "points3D.txt"

    if cameras_txt.is_file() and images_txt.is_file() and points3d_txt.is_file():
        cameras = read_cameras_text(cameras_txt)
        images = read_images_text(images_txt)
        points3D = read_points3D_text(points3d_txt)
        return ReconstructionModel(cameras=cameras, images=images, points3D=points3D, model_dir=path)

    raise FileNotFoundError(
        f"Could not find valid COLMAP model files (cameras.bin/txt, images.bin/txt, "
        f"points3D.bin/txt) in {path}"
    )
