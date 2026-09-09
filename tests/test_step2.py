"""Comprehensive unit and integration tests for SIH26158 Step 2: COLMAP Pose Validation."""

import json
import math
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reconstruction.validation.geometry import (
    compute_camera_center,
    compute_relative_rotation_angle_deg,
    compute_reprojection_errors,
    detect_jumps,
    project_3d_to_2d,
    qvec_to_rotmat,
)
from reconstruction.validation.model_io import (
    Camera,
    ImagePose,
    Point3D,
    ReconstructionModel,
    read_colmap_model,
)
from reconstruction.validation.pose_report import (
    extract_frame_sequence_key,
    order_registered_images,
    validate_reconstruction,
)


# =====================================================================
# 1. Pure Geometry & Mathematical Unit Tests
# =====================================================================

def test_qvec_to_rotmat_identity():
    """Identity quaternion [1, 0, 0, 0] must yield the 3x3 identity matrix."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    R = qvec_to_rotmat(q_id)
    assert np.allclose(R, np.eye(3), atol=1e-10)


def test_qvec_to_rotmat_rotation_x():
    """90 degree rotation about the X axis."""
    # q = [cos(pi/4), sin(pi/4), 0, 0] = [sqrt(2)/2, sqrt(2)/2, 0, 0]
    val = math.sqrt(2.0) / 2.0
    q = np.array([val, val, 0.0, 0.0], dtype=np.float64)
    R = qvec_to_rotmat(q)

    # Point on Y axis [0, 1, 0] rotated 90 deg around X should become [0, 0, 1]
    p_y = np.array([0.0, 1.0, 0.0])
    p_rot = R @ p_y
    assert np.allclose(p_rot, np.array([0.0, 0.0, 1.0]), atol=1e-7)


def test_qvec_to_rotmat_invalid_zero():
    """Invalid near-zero quaternion must raise ValueError."""
    q_zero = np.array([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="Invalid near-zero quaternion"):
        qvec_to_rotmat(q_zero)


def test_compute_camera_center():
    """Verify camera center calculation C = -R^T * t.
    
    By definition, transforming the camera center C into camera coordinates
    must yield the origin: R * C + t == [0, 0, 0].
    """
    # 90 deg rotation around Z and translation [10, -5, 2]
    val = math.sqrt(2.0) / 2.0
    q = np.array([val, 0.0, 0.0, val], dtype=np.float64)
    t = np.array([10.0, -5.0, 2.0], dtype=np.float64)

    C = compute_camera_center(q, t)
    R = qvec_to_rotmat(q)

    # In camera coordinates: R * C + t == 0
    p_cam = R @ C + t
    assert np.allclose(p_cam, np.zeros(3), atol=1e-7)


def test_compute_relative_rotation_angle_deg():
    """Verify relative rotation angle computation in degrees."""
    R_id = np.eye(3)
    # Relative angle between identical orientations is 0.0 deg
    assert compute_relative_rotation_angle_deg(R_id, R_id) == pytest.approx(0.0, abs=1e-6)

    # 45 deg rotation around Z
    theta = math.radians(45.0)
    R_45 = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    angle = compute_relative_rotation_angle_deg(R_id, R_45)
    assert angle == pytest.approx(45.0, abs=1e-5)


# =====================================================================
# 2. Camera Model Projection Tests
# =====================================================================

def test_project_simple_pinhole():
    """Test SIMPLE_PINHOLE projection: f, cx, cy."""
    cam = Camera(camera_id=1, model_name="SIMPLE_PINHOLE", width=1920, height=1080,
                 params=np.array([1000.0, 960.0, 540.0], dtype=np.float64))
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    t = np.array([0.0, 0.0, 0.0])       # At origin
    # Point at (X=1, Y=2, Z=10)
    # u = 1/10 = 0.1, v = 2/10 = 0.2
    # x = 1000 * 0.1 + 960 = 1060.0
    # y = 1000 * 0.2 + 540 = 740.0
    pt3d = np.array([1.0, 2.0, 10.0])
    proj, supported = project_3d_to_2d(pt3d, q, t, cam)
    assert supported is True
    assert proj is not None
    assert np.allclose(proj, np.array([1060.0, 740.0]), atol=1e-6)


def test_project_simple_radial():
    """Test SIMPLE_RADIAL projection with distortion parameter k1."""
    cam = Camera(camera_id=1, model_name="SIMPLE_RADIAL", width=1920, height=1080,
                 params=np.array([1000.0, 960.0, 540.0, 0.1], dtype=np.float64))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t = np.array([0.0, 0.0, 0.0])
    # Point at (X=1, Y=0, Z=10) -> u=0.1, v=0, r2=0.01, dist = 1 + 0.1*0.01 = 1.001
    # x = 1000 * 1.001 * 0.1 + 960 = 100.1 + 960 = 1060.1
    # y = 1000 * 1.001 * 0 + 540 = 540.0
    pt3d = np.array([1.0, 0.0, 10.0])
    proj, supported = project_3d_to_2d(pt3d, q, t, cam)
    assert supported is True
    assert proj is not None
    assert np.allclose(proj, np.array([1060.1, 540.0]), atol=1e-6)


def test_project_point_behind_camera():
    """Points behind the camera (z_cam <= 0) must return None without error."""
    cam = Camera(camera_id=1, model_name="PINHOLE", width=1920, height=1080,
                 params=np.array([1000.0, 1000.0, 960.0, 540.0], dtype=np.float64))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t = np.array([0.0, 0.0, 0.0])
    pt_behind = np.array([0.0, 0.0, -5.0])
    proj, supported = project_3d_to_2d(pt_behind, q, t, cam)
    assert supported is True
    assert proj is None


def test_project_unsupported_camera_model():
    """Unsupported camera models must report supported=False."""
    cam = Camera(camera_id=1, model_name="UNKNOWN_EXOTIC_MODEL", width=1920, height=1080,
                 params=np.array([1.0, 2.0], dtype=np.float64))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    t = np.array([0.0, 0.0, 0.0])
    proj, supported = project_3d_to_2d(np.array([1.0, 1.0, 5.0]), q, t, cam)
    assert supported is False
    assert proj is None


# =====================================================================
# 3. Trajectory & Jump Detection Tests
# =====================================================================

def test_detect_jumps_explicit_threshold():
    """Explicit threshold must flag only values exceeding the cutoff."""
    steps = [0.5, 0.4, 0.6, 5.0, 0.5]  # index 3 is a jump
    count, max_jump, flagged = detect_jumps(steps, threshold=2.0)
    assert count == 1
    assert max_jump == pytest.approx(5.0)
    assert flagged == [3]


def test_detect_jumps_mad_outlier():
    """Statistical MAD detection must flag significant outliers."""
    # Normal distribution with one huge spike
    steps = [1.0, 1.05, 0.98, 1.02, 1.01, 10.0, 0.99, 1.0]
    count, max_jump, flagged = detect_jumps(steps, threshold=None, k_mad=3.5)
    assert count >= 1
    assert 5 in flagged
    assert max_jump == pytest.approx(10.0)


def test_extract_frame_sequence_key():
    """Verify chronological sequence extraction from drone frame filenames."""
    assert extract_frame_sequence_key("frame_000127.jpg") == (127, "frame_000127.jpg")
    assert extract_frame_sequence_key("DJI_0452.JPG") == (452, "DJI_0452.JPG")
    assert extract_frame_sequence_key("000001.png") == (1, "000001.png")
    assert extract_frame_sequence_key("no_number_here.png") == (-1, "no_number_here.png")


def test_order_registered_images():
    """Images with sequence numbers must be ordered chronologically regardless of ID."""
    images = {
        10: ImagePose(image_id=10, qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                      camera_id=1, name="frame_000003.jpg", xys=np.empty((0, 2)), point3D_ids=np.empty(0)),
        20: ImagePose(image_id=20, qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                      camera_id=1, name="frame_000001.jpg", xys=np.empty((0, 2)), point3D_ids=np.empty(0)),
        30: ImagePose(image_id=30, qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
                      camera_id=1, name="frame_000002.jpg", xys=np.empty((0, 2)), point3D_ids=np.empty(0)),
    }
    ordered, order_type = order_registered_images(images)
    assert order_type == "frame_filename"
    assert [img.name for img in ordered] == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
    ]


# =====================================================================
# 4. End-to-End Validation Logic Tests
# =====================================================================

def helper_create_synthetic_model(tmpdir: Path, num_frames: int = 5, noise_px: float = 0.0) -> Path:
    """Helper creating a synthetic COLMAP text model with known geometry."""
    model_dir = tmpdir / "sparse" / "0"
    model_dir.mkdir(parents=True)

    # 1. cameras.txt (PINHOLE: fx=1000, fy=1000, cx=960, cy=540)
    with open(model_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list\n")
        f.write("1 PINHOLE 1920 1080 1000.0 1000.0 960.0 540.0\n")

    # Create 4 3D points
    points = [
        (1, np.array([-1.0, -1.0, 10.0])),
        (2, np.array([1.0, -1.0, 10.0])),
        (3, np.array([1.0, 1.0, 10.0])),
        (4, np.array([-1.0, 1.0, 10.0])),
    ]

    with open(model_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D points\n")
        for pid, pt in points:
            f.write(f"{pid} {pt[0]} {pt[1]} {pt[2]} 255 255 255 0.5 1 0\n")

    # Create images moving along X axis: C = [i*0.5, 0, 0]
    # In camera coordinates with identity rotation: t = -C = [-i*0.5, 0, 0]
    cam = Camera(1, "PINHOLE", 1920, 1080, np.array([1000.0, 1000.0, 960.0, 540.0]))
    with open(model_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list\n")
        for i in range(num_frames):
            img_id = i + 1
            name = f"frame_{i+1:06d}.jpg"
            q = np.array([1.0, 0.0, 0.0, 0.0])
            t = np.array([-float(i) * 0.5, 0.0, 0.0])
            f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} 1 {name}\n")

            # Project each 3D point and write observations
            obs_parts = []
            for pid, pt in points:
                proj, _ = project_3d_to_2d(pt, q, t, cam)
                x = proj[0] + noise_px
                y = proj[1] + noise_px
                obs_parts.append(f"{x:.4f} {y:.4f} {pid}")
            f.write(" ".join(obs_parts) + "\n")

    return model_dir


def test_validate_reconstruction_passes_clean_model():
    """A clean synthetic reconstruction should achieve validation_status = PASS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_dir = helper_create_synthetic_model(tmp_path, num_frames=5, noise_px=0.0)

        report_file = tmp_path / "pose_report.json"
        traj_file = tmp_path / "trajectory.json"

        report = validate_reconstruction(
            model_dir=model_dir,
            output_report_path=report_file,
            trajectory_output_path=traj_file,
            min_registration_rate=0.80,
            max_reprojection_error=1.0,
        )

        assert report["validation_status"] == "PASS"
        assert report["checks"]["registration"]["status"] == "PASS"
        assert report["checks"]["reprojection"]["status"] == "PASS"
        assert report["checks"]["trajectory"]["status"] == "PASS"
        assert report["reprojection"]["rmse_px"] == pytest.approx(0.0, abs=1e-3)
        assert report["trajectory"]["mean_step"] == pytest.approx(0.5, abs=1e-3)

        # Verify exported files
        assert report_file.is_file()
        assert traj_file.is_file()

        with open(traj_file, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
        assert traj_data["georeferenced"] is False
        assert traj_data["num_frames"] == 5
        assert traj_data["frames"][0]["frame_id"] == "frame_000001"
        assert traj_data["frames"][0]["position"] == [0.0, 0.0, 0.0]
        assert traj_data["frames"][1]["position"] == [0.5, 0.0, 0.0]


def test_validate_reconstruction_fails_high_reprojection_error():
    """Excessive reprojection error must trigger validation_status = FAIL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Inject 5.0 px noise per axis -> ~7.07 px reprojection error
        model_dir = helper_create_synthetic_model(tmp_path, num_frames=5, noise_px=5.0)

        report = validate_reconstruction(
            model_dir=model_dir,
            max_reprojection_error=1.0,  # Strict threshold
        )

        assert report["validation_status"] == "FAIL"
        assert report["checks"]["reprojection"]["status"] == "FAIL"
        assert report["reprojection"]["rmse_px"] > 1.0


def test_validate_reconstruction_fails_low_registration_rate():
    """Registration rate below threshold must trigger validation_status = FAIL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        model_dir = helper_create_synthetic_model(tmp_path, num_frames=3, noise_px=0.0)

        # Create image folder with 10 images (only 3 registered -> 30% rate)
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(10):
            (img_dir / f"frame_{i+1:06d}.jpg").write_bytes(b"dummy")

        report = validate_reconstruction(
            model_dir=model_dir,
            image_dir=img_dir,
            min_registration_rate=0.80,  # Requires 80%
        )

        assert report["validation_status"] == "FAIL"
        assert report["input"]["registration_rate"] == 0.3
        assert report["checks"]["registration"]["status"] == "FAIL"


def test_validate_reconstruction_nonexistent_model():
    """Nonexistent model directory must fail gracefully without unhandled crash."""
    report = validate_reconstruction(model_dir="nonexistent_sparse_directory_xyz")
    assert report["validation_status"] == "FAIL"
    assert len(report["errors"]) > 0
    assert report["checks"]["registration"]["status"] == "FAIL"
