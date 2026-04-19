"""
Smoke test for sam2_inference_op.py — Step 3 of the build.

Tests SAM 2.1 loading, box-prompt segmentation, mask output format.
Run: pytest surgery/tests/test_sam2_inference_op.py -v
"""

import sys
import os
import time
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.sam2_inference_op import SAM2Segmenter, HAS_SAM2
from operators.yolo_detection_op import Detection

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(not HAS_SAM2, reason="SAM 2 not installed"),
]

DEVICE = "cuda:0"
CHECKPOINT = "models/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@pytest.fixture(scope="module")
def segmenter():
    """Load SAM 2.1 once for all tests."""
    if not os.path.exists(CHECKPOINT):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")
    seg = SAM2Segmenter(
        checkpoint=CHECKPOINT,
        model_cfg=MODEL_CFG,
        device=DEVICE,
        dtype="bfloat16",
        vos_optimized=False,  # Skip compile for faster test startup
        max_objects=3,
    )
    return seg


def make_test_frame(width=854, height=480) -> torch.Tensor:
    """Create a synthetic surgical-like frame."""
    frame = np.full((height, width, 3), [40, 20, 25], dtype=np.uint8)
    # Bright ellipse simulating an organ
    import cv2
    cv2.ellipse(frame, (427, 240), (80, 50), 30, 0, 360, (60, 120, 50), -1)
    # Tool-like rectangle
    cv2.rectangle(frame, (500, 280), (700, 320), (180, 180, 190), -1)
    return torch.from_numpy(frame).to(DEVICE)


class TestSAM2Segmenter:
    """Test the standalone SAM 2.1 segmentation engine."""

    def test_model_loads(self, segmenter):
        """SAM 2.1 should load without errors."""
        assert segmenter.predictor is not None
        print(f"  SAM 2.1 loaded on {segmenter.device}")

    def test_segment_with_box_prompt(self, segmenter):
        """Box prompt should produce a non-empty mask."""
        segmenter.reset()
        frame = make_test_frame()

        # Provide a bounding box around the "organ" ellipse
        detections = [
            Detection(class_name="gallbladder", bbox=[347, 190, 507, 290],
                      confidence=0.8, class_id=0)
        ]

        masks = segmenter.segment_frame(frame, detections)

        assert "gallbladder" in masks
        mask = masks["gallbladder"]
        assert mask.shape == (480, 854)
        assert mask.sum() > 100, "Mask should be non-trivially filled"
        assert mask.device.type == "cuda"
        print(f"  Gallbladder mask: {mask.sum().item():.0f} pixels")

    def test_segment_multiple_objects(self, segmenter):
        """Multiple box prompts should produce multiple masks."""
        segmenter.reset()
        frame = make_test_frame()

        detections = [
            Detection(class_name="gallbladder", bbox=[347, 190, 507, 290],
                      confidence=0.8, class_id=0),
            Detection(class_name="grasper", bbox=[490, 270, 710, 330],
                      confidence=0.7, class_id=1),
        ]

        masks = segmenter.segment_frame(frame, detections)

        assert len(masks) == 2
        assert "gallbladder" in masks
        assert "grasper" in masks

    def test_stack_masks(self, segmenter):
        """Masks should stack into (N, H, W) tensor."""
        segmenter.reset()
        frame = make_test_frame()

        detections = [
            Detection(class_name="gallbladder", bbox=[347, 190, 507, 290],
                      confidence=0.8, class_id=0),
        ]

        masks = segmenter.segment_frame(frame, detections)
        stacked = segmenter.get_mask_tensor(masks, 480, 854)

        assert stacked.shape == (1, 480, 854)
        assert stacked.device.type == "cuda"

    def test_empty_detections_use_cache(self, segmenter):
        """With no new detections, should return cached masks."""
        segmenter.reset()
        frame = make_test_frame()

        # First: provide detection
        detections = [
            Detection(class_name="gallbladder", bbox=[347, 190, 507, 290],
                      confidence=0.8, class_id=0),
        ]
        masks1 = segmenter.segment_frame(frame, detections)
        assert "gallbladder" in masks1

        # Second: no detection (simulate between-YOLO frames)
        masks2 = segmenter.segment_frame(frame, detections=None)
        assert "gallbladder" in masks2, "Should use cached mask"

    def test_empty_masks_output(self, segmenter):
        """With no detections and no cache, should return empty tensor."""
        segmenter.reset()
        frame = make_test_frame()

        masks = segmenter.segment_frame(frame, detections=[])
        stacked = segmenter.get_mask_tensor(masks, 480, 854)

        assert stacked.shape[0] == 0  # No masks
        assert stacked.shape[1:] == (480, 854)

    def test_inference_speed(self, segmenter):
        """Single-frame SAM inference should complete in reasonable time."""
        segmenter.reset()
        frame = make_test_frame()
        detections = [
            Detection(class_name="gallbladder", bbox=[347, 190, 507, 290],
                      confidence=0.8, class_id=0),
        ]

        # Warm up
        segmenter.segment_frame(frame, detections)
        segmenter.reset()

        torch.cuda.synchronize()
        start = time.perf_counter()
        segmenter.segment_frame(frame, detections)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"  SAM 2.1 single-frame latency: {elapsed_ms:.1f}ms")
        # Allow up to 500ms for tiny model without torch.compile
        assert elapsed_ms < 500, f"SAM too slow: {elapsed_ms:.1f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
