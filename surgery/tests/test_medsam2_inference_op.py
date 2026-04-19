"""
Smoke test for medsam2_inference_op.py.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.medsam2_inference_op import HAS_MEDSAM2, MedSAM2Segmenter
from operators.yolo_detection_op import Detection

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(not HAS_MEDSAM2, reason="MedSAM2 not installed"),
]

DEVICE = "cuda:0"
CHECKPOINT = "models/MedSAM2_latest.pt"
MODEL_CFG = "configs/sam2.1_hiera_t512.yaml"


def make_test_frame(width=854, height=480) -> torch.Tensor:
    frame = np.full((height, width, 3), [40, 20, 25], dtype=np.uint8)
    import cv2

    cv2.ellipse(frame, (427, 240), (80, 50), 30, 0, 360, (60, 120, 50), -1)
    cv2.rectangle(frame, (500, 280), (700, 320), (180, 180, 190), -1)
    return torch.from_numpy(frame).to(DEVICE)


@pytest.fixture(scope="module")
def segmenter():
    if not os.path.exists(CHECKPOINT):
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")
    return MedSAM2Segmenter(
        checkpoint=CHECKPOINT,
        model_cfg=MODEL_CFG,
        device=DEVICE,
        dtype="bfloat16",
        max_objects=3,
        use_temporal_memory=False,
    )


def test_model_loads(segmenter):
    assert segmenter.predictor is not None


def test_segment_with_box_prompt(segmenter):
    segmenter.reset()
    frame = make_test_frame()
    detections = [
        Detection("gallbladder", [347, 190, 507, 290], 0.9, 0),
    ]
    masks = segmenter.segment_frame(frame, detections, frame_idx=0)
    assert "gallbladder" in masks
    assert masks["gallbladder"].sum() > 100


def test_empty_detections_use_cache(segmenter):
    segmenter.reset()
    frame = make_test_frame()
    detections = [Detection("gallbladder", [347, 190, 507, 290], 0.9, 0)]
    assert segmenter.segment_frame(frame, detections, frame_idx=0)
    cached = segmenter.segment_frame(frame, None, frame_idx=1)
    assert "gallbladder" in cached


def test_inference_speed(segmenter):
    segmenter.reset()
    frame = make_test_frame()
    detections = [Detection("gallbladder", [347, 190, 507, 290], 0.9, 0)]

    segmenter.segment_frame(frame, detections, frame_idx=0)
    segmenter.reset()

    torch.cuda.synchronize()
    start = time.perf_counter()
    segmenter.segment_frame(frame, detections, frame_idx=0)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 750
