"""
Tests for the offline staged-video evaluation workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.convert_video import VideoAsset
from evaluation.offline_evaluator import OfflineVideoEvaluator
from operators.overlay_compositor_op import OverlayCompositor
from operators.yolo_detection_op import Detection


class FakeDetector:
    def __init__(self):
        self.device = "cpu"
        self.detect_every_n_frames = 2
        self.frame_idx = 0

    def reset(self):
        self.frame_idx = 0

    def detect(self, frame):
        self.frame_idx += 1
        return []


class FakeSegmenter:
    def __init__(self):
        self.device = "cpu"
        self.use_temporal_memory = True
        self.last_masks = {}
        self.prepared_video = None

    def reset(self):
        self.last_masks = {}
        self.prepared_video = None

    def prepare_video(self, video_source, propagation_window=15):
        self.prepared_video = (video_source, propagation_window)

    def segment_frame(self, frame, detections=None, frame_idx=None):
        if detections:
            mask = torch.zeros(frame.shape[0], frame.shape[1], dtype=torch.float32)
            mask[5:15, 10:20] = 1.0
            self.last_masks = {detections[0].class_name: mask}
            return dict(self.last_masks)
        return dict(self.last_masks)

    def get_mask_tensor(self, masks, height, width):
        if not masks:
            return torch.zeros(0, height, width)
        return torch.stack(list(masks.values()), dim=0)

    def get_mask_labels(self, masks):
        return list(masks.keys())


def test_offline_evaluator_uses_seed_prompts_and_writes_artifacts(tmp_path):
    frames = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    bundle_path = tmp_path / "video01.gxf_entities.npy"
    np.save(bundle_path, frames)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in range(3):
        (frames_dir / f"{idx:05d}.jpg").write_bytes(b"fake")

    asset = VideoAsset(
        name="video01",
        source_path=str(tmp_path / "video01.mp4"),
        output_dir=str(tmp_path),
        bundle_path=str(bundle_path),
        frames_dir=str(frames_dir),
        metadata_path=str(tmp_path / "video01_metadata.json"),
        frame_count=3,
        fps=25,
        resolution=[32, 32],
    )

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "video01.json").write_text(
        json.dumps(
            {
                "detections_by_frame": {
                    "0": [
                        {
                            "class_name": "gallbladder",
                            "bbox": [5, 5, 20, 20],
                            "confidence": 1.0,
                            "class_id": 0,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    evaluator = OfflineVideoEvaluator(
        detector=FakeDetector(),
        segmenter=FakeSegmenter(),
        overlay=OverlayCompositor(device="cpu", glow_effect=False),
        output_dir=str(tmp_path / "results"),
        max_frames=3,
        save_overlays=False,
        save_masks=True,
        prompt_dir=str(prompt_dir),
    )

    [metrics] = evaluator.evaluate_assets([asset])

    assert metrics["frames_processed"] == 3
    assert metrics["frames_using_seed_prompts"] == 1
    assert metrics["frames_with_non_empty_masks"] == 3
    assert (tmp_path / "results" / "video01" / "metrics.json").exists()
    assert (tmp_path / "results" / "video01" / "masks" / "frame_00000.npz").exists()
