"""
Tests for batch video extraction used by the Cholec80 evaluation workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from data.convert_video import extract_video_batch


def _make_test_video(path: Path, frame_count: int = 5, size=(96, 64)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for idx in range(frame_count):
        frame = np.full((size[1], size[0], 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_video_batch_creates_assets(tmp_path):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir()

    _make_test_video(raw_dir / "video01.mp4")
    _make_test_video(raw_dir / "video02.mp4")

    assets = extract_video_batch(
        video_dir=str(raw_dir),
        output_dir=str(out_dir),
        video_glob="video*.mp4",
        fps=10,
        resolution=(80, 48),
        max_frames=3,
        save_numpy=True,
        save_frames=True,
    )

    assert len(assets) == 2
    for asset in assets:
        asset_dir = out_dir / asset.name
        assert asset_dir.exists()
        assert Path(asset.bundle_path).exists()
        assert Path(asset.frames_dir).exists()
        assert Path(asset.metadata_path).exists()
        assert asset.frame_count == 3
        assert len(list(Path(asset.frames_dir).glob("*.jpg"))) == 3

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2
