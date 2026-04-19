"""
Smoke test for yolo_detection_op.py — Step 2 of the build.

Tests YOLOv8n loading, detection on synthetic frames, skip-frame logic.
Run: pytest surgery/tests/test_yolo_detection_op.py -v
"""

import sys
import os
import time
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.yolo_detection_op import YOLODetector, Detection

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for smoke tests"
)

DEVICE = "cuda:0"


@pytest.fixture(scope="module")
def detector():
    """Load YOLOv8n once for all tests in this module."""
    det = YOLODetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.3,
        detect_every_n_frames=15,
        device=DEVICE,
    )
    return det


def make_synthetic_frame(width=854, height=480) -> torch.Tensor:
    """Create a synthetic frame with a colored rectangle (detectable by YOLO)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[50:200, 100:300] = [255, 0, 0]     # Red rectangle (top-left)
    frame[250:400, 400:700] = [0, 255, 0]     # Green rectangle (center-right)
    frame[100:150, 500:800] = [200, 200, 200] # Grey bar (tool-like)
    return torch.from_numpy(frame).to(DEVICE)


class TestYOLODetector:
    """Test the standalone YOLO detection engine."""

    def test_model_loads(self, detector):
        """YOLOv8n should load without errors."""
        assert detector.model is not None
        assert len(detector.class_names) > 0
        print(f"  YOLO classes: {len(detector.class_names)} loaded")

    def test_detection_output_format(self, detector):
        """Detections should have correct format."""
        detector.reset()
        frame = make_synthetic_frame()
        detections = detector.detect(frame)

        # May or may not detect anything on synthetic frames, but format should be valid
        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert isinstance(det.class_name, str)
            assert len(det.bbox) == 4
            assert all(isinstance(v, float) for v in det.bbox)
            assert 0.0 <= det.confidence <= 1.0
            assert isinstance(det.class_id, int)

    def test_skip_frame_logic(self, detector):
        """YOLO should only run every N frames."""
        detector.reset()
        frame = make_synthetic_frame()

        detection_frames = []
        for i in range(30):
            was_detection_frame = detector.should_detect()
            detections = detector.detect(frame)

            if was_detection_frame:
                detection_frames.append(i)

        # With detect_every_n_frames=15, frames 0 and 15 should be detection frames
        # (frame_count is incremented inside detect(), so should_detect checks
        # frame_count before increment — frame 0 hits at count=0, frame 15 at count=15)
        assert 0 in detection_frames or 1 in detection_frames, \
            f"First frame should be a detection frame, got: {detection_frames}"
        assert len(detection_frames) == 2, \
            f"Expected 2 detection frames in 30, got {len(detection_frames)}: {detection_frames}"

    def test_gpu_performance(self, detector):
        """Single YOLO inference should be fast (<50ms on GPU)."""
        detector.reset()
        frame = make_synthetic_frame()

        # Warm up
        detector.detect(frame)
        detector.frame_count = 0  # Reset to force detection

        # Time a single detection
        torch.cuda.synchronize()
        start = time.perf_counter()
        detector.detect(frame)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"  YOLO single-frame latency: {elapsed_ms:.1f}ms")
        assert elapsed_ms < 50, f"YOLO too slow: {elapsed_ms:.1f}ms (target <50ms)"

    def test_real_image_detection(self, detector):
        """Test with a more realistic image (person-like shape, COCO trained)."""
        detector.reset()
        # Create a frame with a rough "person" shape that COCO-trained YOLO might detect
        frame = np.full((480, 854, 3), 128, dtype=np.uint8)
        # Head
        frame[50:100, 400:450] = [220, 180, 150]
        # Body
        frame[100:250, 380:470] = [50, 50, 150]
        # Legs
        frame[250:400, 380:420] = [50, 50, 100]
        frame[250:400, 430:470] = [50, 50, 100]

        tensor = torch.from_numpy(frame).to(DEVICE)
        detections = detector.detect(tensor)

        # We can't guarantee detection on synthetic data, but should not crash
        assert isinstance(detections, list)
        print(f"  Detections on synthetic person: {len(detections)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
