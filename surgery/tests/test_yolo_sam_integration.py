"""
Integration smoke test — Step 5: YOLO → SAM pipeline.

Tests that YOLO detections flow correctly into SAM as prompts,
and that SAM produces continuous masks even between YOLO detection frames.

Run: pytest surgery/tests/test_yolo_sam_integration.py -v -s
"""

import sys
import os
import time
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.yolo_detection_op import YOLODetector, Detection
from operators.sam2_inference_op import SAM2Segmenter, HAS_SAM2

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(not HAS_SAM2, reason="SAM 2 not installed"),
]

DEVICE = "cuda:0"
CHECKPOINT = "models/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@pytest.fixture(scope="module")
def yolo():
    return YOLODetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.3,
        detect_every_n_frames=15,
        device=DEVICE,
    )


@pytest.fixture(scope="module")
def sam(request):
    if not os.path.exists(CHECKPOINT):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")
    return SAM2Segmenter(
        checkpoint=CHECKPOINT,
        model_cfg=MODEL_CFG,
        device=DEVICE,
        dtype="bfloat16",
        vos_optimized=False,
        max_objects=3,
    )


def make_synthetic_frames(n=30, width=854, height=480):
    """Generate N synthetic surgical-like frames with slight motion."""
    import cv2
    frames = []
    for i in range(n):
        frame = np.full((height, width, 3), [40, 20, 25], dtype=np.uint8)
        cx = 427 + int(20 * np.sin(i * 0.1))
        cy = 240 + int(10 * np.cos(i * 0.15))
        cv2.ellipse(frame, (cx, cy), (80, 50), 30, 0, 360, (60, 120, 50), -1)
        tx = 600 + int(30 * np.sin(i * 0.2))
        cv2.rectangle(frame, (tx - 50, 280), (tx + 50, 320), (180, 180, 190), -1)
        frames.append(torch.from_numpy(frame).to(DEVICE))
    return frames


class TestYOLOSAMIntegration:
    """Test the YOLO → SAM pipeline end-to-end."""

    def test_full_pipeline_30_frames(self, yolo, sam):
        """Process 30 frames through YOLO → SAM, verify continuous masks."""
        yolo.reset()
        sam.reset()
        frames = make_synthetic_frames(30)

        detection_frame_count = 0
        mask_frame_count = 0
        total_mask_pixels = 0

        for i, frame in enumerate(frames):
            # Step 1: YOLO detection
            detections = yolo.detect(frame)
            if detections:
                detection_frame_count += 1

            # If no YOLO detections on first frame, provide manual fallback
            # (YOLO might not detect synthetic shapes)
            if i == 0 and not detections:
                detections = [
                    Detection(class_name="gallbladder",
                              bbox=[347, 190, 507, 290],
                              confidence=0.9, class_id=0),
                ]

            # Step 2: SAM segmentation
            masks = sam.segment_frame(frame, detections if detections else None)

            if masks:
                mask_frame_count += 1
                for name, mask in masks.items():
                    total_mask_pixels += mask.sum().item()

        print(f"\n  === Integration Results (30 frames) ===")
        print(f"  YOLO detection frames: {detection_frame_count}")
        print(f"  Frames with masks: {mask_frame_count}")
        print(f"  Total mask pixels: {total_mask_pixels:.0f}")

        # Key assertion: masks should exist on ALL frames after first prompt
        assert mask_frame_count >= 29, \
            f"Expected masks on ~all frames, got {mask_frame_count}/30"

    def test_mask_quality(self, yolo, sam):
        """Verify masks are non-trivially sized (not just a few pixels)."""
        sam.reset()
        frame = make_synthetic_frames(1)[0]

        detections = [
            Detection(class_name="gallbladder",
                      bbox=[347, 190, 507, 290],
                      confidence=0.9, class_id=0),
        ]

        masks = sam.segment_frame(frame, detections)
        assert "gallbladder" in masks
        mask_area = masks["gallbladder"].sum().item()
        print(f"  Mask area: {mask_area:.0f} pixels")
        assert mask_area > 100, f"Mask too small: {mask_area}"

    def test_pipeline_speed(self, yolo, sam):
        """Measure end-to-end latency for the YOLO→SAM pipeline."""
        yolo.reset()
        sam.reset()
        frames = make_synthetic_frames(5)

        # Warm up
        detections = [Detection("gallbladder", [347, 190, 507, 290], 0.9)]
        sam.segment_frame(frames[0], detections)
        yolo.detect(frames[0])

        yolo.reset()
        sam.reset()

        torch.cuda.synchronize()
        start = time.perf_counter()

        for i, frame in enumerate(frames):
            dets = yolo.detect(frame)
            if i == 0 and not dets:
                dets = [Detection("gallbladder", [347, 190, 507, 290], 0.9)]
            sam.segment_frame(frame, dets if dets else None)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fps = len(frames) / elapsed

        print(f"  Pipeline: {elapsed*1000:.0f}ms for {len(frames)} frames = {fps:.1f} FPS")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
