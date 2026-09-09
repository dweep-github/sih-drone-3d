"""YOLO11 Dynamic Object Detection and Mask Generation (SIH26158 Step 10).

Detects dynamic/moving objects (e.g. vehicles, pedestrians, animals) in selected
reconstruction frames and produces binary ignore masks following the convention:
- 0   = keep/use (background, static structures, terrain)
- 255 = ignore/masked (dynamic objects)

Masks strictly preserve source image dimensions (H, W).
If YOLO11 or weights are unavailable, the module cleanly reports NOT_AVAILABLE
without manufacturing detections.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("masking.dynamic_mask")

# COCO dynamic classes typically subject to movement in drone surveys
DEFAULT_DYNAMIC_CLASSES: Set[str] = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DynamicMaskingReport:
    """Standardized report for dynamic object detection and masking."""

    status: str  # "SUCCESS", "NOT_AVAILABLE", "FAILED", "SKIPPED"
    reason: Optional[str] = None
    input_frames: int = 0
    masked_frames: int = 0
    total_detections: int = 0
    mask_coverage: float = 0.0  # mean fraction of pixels masked across all frames
    dynamic_classes: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    masks_dir: Optional[str] = None
    detections_dir: Optional[str] = None
    per_frame: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_yolo_availability(
    weights: str = "yolo11n-seg.pt",
) -> Dict[str, Any]:
    """Inspects whether Ultralytics YOLO is importable and weights are resolvable."""
    try:
        import ultralytics
        from ultralytics import YOLO

        ultralytics_ver = getattr(ultralytics, "__version__", "unknown")
    except ImportError as exc:
        return {
            "available": False,
            "version": None,
            "reason": f"Ultralytics is not installed: {exc}",
        }

    # Check if weights exist locally on disk
    weights_path = Path(weights)
    if weights_path.is_file():
        return {
            "available": True,
            "version": ultralytics_ver,
            "weights_path": str(weights_path.resolve()),
            "reason": None,
        }

    return {
        "available": False,
        "version": ultralytics_ver,
        "reason": f"YOLO11 weights '{weights}' unavailable (not found locally on disk).",
    }


def run_dynamic_masking(
    frames_dir: Union[str, Path],
    masks_output_dir: Union[str, Path],
    detections_output_dir: Optional[Union[str, Path]] = None,
    dynamic_classes: Optional[Sequence[str]] = None,
    weights: str = "yolo11n-seg.pt",
    confidence: float = 0.25,
    device: Optional[str] = None,
    mock_detector: Optional[Callable[[np.ndarray, str], List[Dict[str, Any]]]] = None,
    require_masking: bool = False,
) -> DynamicMaskingReport:
    """Executes dynamic object detection and mask generation on selected frames.

    Args:
        frames_dir: Directory containing selected frame images.
        masks_output_dir: Directory where binary mask PNGs will be saved.
        detections_output_dir: Optional directory to store per-frame detection JSONs.
        dynamic_classes: Collection of class names to mask. Defaults to DEFAULT_DYNAMIC_CLASSES.
        weights: YOLO weights filename or path.
        confidence: Minimum detection confidence threshold.
        device: Device to run inference on (e.g. 'cpu', 'cuda:0').
        mock_detector: Optional callable `(image_bgr, frame_name) -> list_of_detections`
            used in testing without real neural model inference.
        require_masking: If True, unavailability of YOLO raises or marks FAILED.

    Returns:
        DynamicMaskingReport with complete execution statistics.
    """
    start_time = time.time()
    frames_path = Path(frames_dir).resolve()
    masks_path = Path(masks_output_dir).resolve()
    masks_path.mkdir(parents=True, exist_ok=True)

    det_path = Path(detections_output_dir).resolve() if detections_output_dir else None
    if det_path:
        det_path.mkdir(parents=True, exist_ok=True)

    target_classes = set(dynamic_classes) if dynamic_classes else DEFAULT_DYNAMIC_CLASSES

    # Discover frame images
    image_files = sorted(
        [p for p in frames_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS],
        key=lambda p: p.name,
    )

    if not image_files:
        elapsed = round(time.time() - start_time, 4)
        return DynamicMaskingReport(
            status="SUCCESS",
            reason="No input frames to mask.",
            input_frames=0,
            processing_time_seconds=elapsed,
            dynamic_classes=sorted(target_classes),
            masks_dir=str(masks_path),
            detections_dir=str(det_path) if det_path else None,
        )

    # Resolve detector
    model = None
    if mock_detector is None:
        avail_info = check_yolo_availability(weights)
        if not avail_info["available"]:
            elapsed = round(time.time() - start_time, 4)
            status = "FAILED" if require_masking else "NOT_AVAILABLE"
            logger.warning("Dynamic object masking unavailable: %s", avail_info["reason"])
            return DynamicMaskingReport(
                status=status,
                reason=avail_info["reason"],
                input_frames=len(image_files),
                processing_time_seconds=elapsed,
                dynamic_classes=sorted(target_classes),
                masks_dir=str(masks_path),
                detections_dir=str(det_path) if det_path else None,
            )

        try:
            from ultralytics import YOLO

            model = YOLO(weights)
        except Exception as exc:
            elapsed = round(time.time() - start_time, 4)
            status = "FAILED" if require_masking else "NOT_AVAILABLE"
            return DynamicMaskingReport(
                status=status,
                reason=f"Failed to load YOLO model: {exc}",
                input_frames=len(image_files),
                processing_time_seconds=elapsed,
                dynamic_classes=sorted(target_classes),
                masks_dir=str(masks_path),
                detections_dir=str(det_path) if det_path else None,
            )

    total_detections = 0
    masked_frames = 0
    coverage_sum = 0.0
    per_frame: Dict[str, Any] = {}

    for img_p in image_files:
        frame_name = img_p.name
        frame_stem = img_p.stem

        img = cv2.imread(str(img_p))
        if img is None:
            logger.warning("Could not read frame %s", img_p)
            continue

        h, w = img.shape[:2]
        # Mask initialization: 0 = keep/use, 255 = ignore/masked
        mask = np.zeros((h, w), dtype=np.uint8)
        frame_detections: List[Dict[str, Any]] = []

        if mock_detector is not None:
            # Mock detector returns list of dicts: {"class": str, "bbox": [x1,y1,x2,y2], "mask": Optional[np.ndarray], "confidence": float}
            dets = mock_detector(img, frame_name)
            for d in dets:
                cls_name = d.get("class", "").lower()
                conf = float(d.get("confidence", 1.0))
                if cls_name in target_classes and conf >= confidence:
                    frame_detections.append(d)
                    if "mask" in d and d["mask"] is not None:
                        # Direct mask
                        m_arr = np.asarray(d["mask"], dtype=np.uint8)
                        if m_arr.shape != (h, w):
                            m_arr = cv2.resize(m_arr, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask[m_arr > 0] = 255
                    elif "bbox" in d:
                        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        mask[y1:y2, x1:x2] = 255
        elif model is not None:
            predict_kwargs: Dict[str, Any] = {
                "source": img,
                "conf": confidence,
                "verbose": False,
            }
            if device:
                safe_dev = str(device).lower()
                if safe_dev.startswith("cuda"):
                    try:
                        import torch

                        if not torch.cuda.is_available():
                            safe_dev = "cpu"
                    except Exception:
                        safe_dev = "cpu"
                predict_kwargs["device"] = safe_dev

            results = model.predict(**predict_kwargs)
            if results and len(results) > 0:
                res = results[0]
                names = res.names or {}

                has_masks = hasattr(res, "masks") and res.masks is not None
                boxes = res.boxes if hasattr(res, "boxes") and res.boxes is not None else []

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    cls_name = str(names.get(cls_id, cls_id)).lower()
                    conf = float(box.conf[0].item())

                    if cls_name in target_classes:
                        xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                        frame_detections.append({
                            "class": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": xyxy,
                        })

                        if has_masks and i < len(res.masks.data):
                            # Extract segmentation mask
                            raw_mask = res.masks.data[i].cpu().numpy()
                            if raw_mask.shape != (h, w):
                                raw_mask = cv2.resize(raw_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                            mask[raw_mask > 0.5] = 255
                        else:
                            # Fill bounding box as fallback
                            x1, y1, x2, y2 = [int(v) for v in xyxy]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            mask[y1:y2, x1:x2] = 255

        # Record metrics for frame
        num_dets = len(frame_detections)
        total_detections += num_dets
        masked_pixels = int(np.count_nonzero(mask))
        coverage = masked_pixels / float(h * w) if (h * w) > 0 else 0.0
        coverage_sum += coverage

        if num_dets > 0 and masked_pixels > 0:
            masked_frames += 1

        # Write mask PNG (0 = keep, 255 = masked)
        mask_filename = f"{frame_stem}.png"
        mask_path = masks_path / mask_filename
        cv2.imwrite(str(mask_path), mask)

        # Write detection metadata if requested
        if det_path:
            det_file = det_path / f"{frame_stem}.json"
            with open(det_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "frame": frame_name,
                        "frame_stem": frame_stem,
                        "dimensions": {"width": w, "height": h},
                        "detections_count": num_dets,
                        "mask_coverage": round(coverage, 6),
                        "detections": frame_detections,
                    },
                    f,
                    indent=2,
                )

        per_frame[frame_name] = {
            "mask_path": str(mask_path),
            "detections_count": num_dets,
            "mask_coverage": round(coverage, 6),
        }

    elapsed = round(time.time() - start_time, 4)
    mean_coverage = round(coverage_sum / len(image_files), 6) if image_files else 0.0

    logger.info(
        "Dynamic object masking finished: %d frames, %d with dynamic masks, %d detections, mean coverage %.4f in %.2fs",
        len(image_files),
        masked_frames,
        total_detections,
        mean_coverage,
        elapsed,
    )

    return DynamicMaskingReport(
        status="SUCCESS",
        reason=None,
        input_frames=len(image_files),
        masked_frames=masked_frames,
        total_detections=total_detections,
        mask_coverage=mean_coverage,
        dynamic_classes=sorted(target_classes),
        processing_time_seconds=elapsed,
        masks_dir=str(masks_path),
        detections_dir=str(det_path) if det_path else None,
        per_frame=per_frame,
    )
