"""
Full pipeline smoke test - Step 6.

Runs the complete YOLO -> SAM -> Overlay pipeline on synthetic frames
without Holoscan by exercising the standalone engines directly.
"""

import os
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.overlay_compositor_op import OverlayCompositor
from operators.sam2_inference_op import HAS_SAM2, SAM2Segmenter
from operators.yolo_detection_op import Detection, YOLODetector

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
    pytest.mark.skipif(not HAS_SAM2, reason="SAM 2 not installed"),
]

DEVICE = "cuda:0"
CHECKPOINT = "models/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def make_surgical_frames(n=30):
    import cv2

    frames = []
    for i in range(n):
        frame = np.full((480, 854, 3), [40, 20, 25], dtype=np.uint8)
        cx = 350 + int(30 * np.sin(i * 0.1))
        cy = 240 + int(15 * np.cos(i * 0.12))
        cv2.ellipse(frame, (cx, cy), (90, 55), 25, 0, 360, (55, 115, 45), -1)
        dx = 550 + int(20 * np.sin(i * 0.08))
        cv2.ellipse(frame, (dx, 200), (35, 12), -15, 0, 360, (75, 75, 35), -1)
        tx = 600 + int(40 * np.sin(i * 0.15))
        cv2.rectangle(frame, (tx - 55, 310), (tx + 55, 340), (175, 175, 185), -1)
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frames.append(cv2.add(frame, noise))
    return frames


class TestFullPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        if not os.path.exists(CHECKPOINT):
            pytest.skip(f"Checkpoint not found: {CHECKPOINT}")

        yolo = YOLODetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.3,
            detect_every_n_frames=15,
            device=DEVICE,
        )
        sam = SAM2Segmenter(
            checkpoint=CHECKPOINT,
            model_cfg=MODEL_CFG,
            device=DEVICE,
            dtype="bfloat16",
            vos_optimized=False,
            max_objects=3,
        )
        overlay = OverlayCompositor(device=DEVICE, glow_effect=True)

        return {"yolo": yolo, "sam": sam, "overlay": overlay}

    def test_end_to_end_30_frames(self, pipeline):
        yolo = pipeline["yolo"]
        sam = pipeline["sam"]
        overlay = pipeline["overlay"]

        yolo.reset()
        sam.reset()

        frames = make_surgical_frames(30)
        composited_frames = []

        torch.cuda.synchronize()
        start = time.perf_counter()

        for i, frame_np in enumerate(frames):
            frame_gpu = torch.from_numpy(frame_np).to(DEVICE)
            detections = yolo.detect(frame_gpu)

            if i == 0 and not detections:
                detections = [
                    Detection("gallbladder", [260, 185, 440, 295], 0.9, 0),
                    Detection("cystic_duct", [515, 188, 585, 212], 0.8, 1),
                ]

            masks = sam.segment_frame(frame_gpu, detections if detections else None)
            composited_frames.append(overlay.composite(frame_gpu, masks))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        fps = len(frames) / elapsed

        print("\n  === Full Pipeline Results ===")
        print(f"  Frames processed: {len(frames)}")
        print(f"  Total time: {elapsed * 1000:.0f}ms")
        print(f"  Average FPS: {fps:.1f}")
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        assert len(composited_frames) == 30
        for frame in composited_frames:
            assert frame.shape == (480, 854, 4)
            assert frame.dtype == torch.uint8

    def test_overlay_has_colors(self, pipeline):
        sam = pipeline["sam"]
        overlay = pipeline["overlay"]
        sam.reset()

        frame_gpu = torch.from_numpy(make_surgical_frames(1)[0]).to(DEVICE)
        detections = [Detection("gallbladder", [260, 185, 440, 295], 0.9, 0)]
        masks = sam.segment_frame(frame_gpu, detections)
        composited = overlay.composite(frame_gpu, masks)

        original_sum = frame_gpu.float().sum().item()
        composited_sum = composited[:, :, :3].float().sum().item()
        assert composited_sum != original_sum

    def test_save_sample_output(self, pipeline):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sam = pipeline["sam"]
        overlay = pipeline["overlay"]
        sam.reset()

        for i, frame_np in enumerate(make_surgical_frames(5)):
            frame_gpu = torch.from_numpy(frame_np).to(DEVICE)
            detections = (
                [
                    Detection("gallbladder", [260, 185, 440, 295], 0.9, 0),
                    Detection("cystic_duct", [515, 188, 585, 212], 0.8, 1),
                ]
                if i == 0
                else None
            )

            masks = sam.segment_frame(frame_gpu, detections)
            composited = overlay.composite(frame_gpu, masks)

            try:
                import cv2

                bgra = cv2.cvtColor(composited.cpu().numpy(), cv2.COLOR_RGBA2BGRA)
                path = os.path.join(OUTPUT_DIR, f"pipeline_frame_{i:03d}.png")
                cv2.imwrite(path, bgra)
            except ImportError:
                pass

        print(f"  [ok] Sample frames saved to {OUTPUT_DIR}/")

    def test_vram_stability(self, pipeline):
        yolo = pipeline["yolo"]
        sam = pipeline["sam"]
        overlay = pipeline["overlay"]

        yolo.reset()
        sam.reset()
        torch.cuda.reset_peak_memory_stats()

        for i, frame_np in enumerate(make_surgical_frames(30)):
            frame_gpu = torch.from_numpy(frame_np).to(DEVICE)
            detections = yolo.detect(frame_gpu)
            if i == 0 and not detections:
                detections = [Detection("gallbladder", [260, 185, 440, 295], 0.9)]

            masks = sam.segment_frame(frame_gpu, detections if detections else None)
            overlay.composite(frame_gpu, masks)

        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM after 30 frames: {peak_vram_gb:.2f} GB")
        assert peak_vram_gb < 8.0, f"VRAM too high: {peak_vram_gb:.2f} GB (limit 8 GB)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
