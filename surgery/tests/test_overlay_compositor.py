"""
Smoke test for overlay_compositor_op.py - Step 4 of the build.

Tests color mapping, alpha blending, glow effect, contour drawing,
and saving a visual PNG for inspection.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.overlay_compositor_op import OverlayCompositor

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for smoke tests",
)

DEVICE = "cuda:0"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "output")


@pytest.fixture(scope="module")
def compositor():
    return OverlayCompositor(device=DEVICE, glow_effect=True, glow_radius=8)


def make_test_frame(width=854, height=480) -> torch.Tensor:
    frame = np.full((height, width, 3), [40, 20, 25], dtype=np.uint8)
    return torch.from_numpy(frame).to(DEVICE)


def make_circle_mask(height=480, width=854, cx=427, cy=240, r=60) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    mask = ((x - cx) ** 2 + (y - cy) ** 2 < r ** 2).float()
    return mask.to(DEVICE)


def make_rect_mask(height=480, width=854, x1=500, y1=280, x2=700, y2=320) -> torch.Tensor:
    mask = torch.zeros(height, width, device=DEVICE)
    mask[y1:y2, x1:x2] = 1.0
    return mask


class TestOverlayCompositor:
    def test_output_shape(self, compositor):
        frame = make_test_frame()
        masks = {"gallbladder": make_circle_mask()}
        result = compositor.composite(frame, masks)

        assert result.shape == (480, 854, 4)
        assert result.dtype == torch.uint8
        assert result.device.type == "cuda"

    def test_unmasked_region_preserved(self, compositor):
        frame = make_test_frame()
        masks = {"gallbladder": make_circle_mask()}
        result = compositor.composite(frame, masks)

        far_pixel = result[0, 0, :3].cpu().float()
        orig_pixel = frame[0, 0].cpu().float()
        diff = (far_pixel - orig_pixel).abs().max()
        assert diff < 30, f"Far pixel changed too much: {diff}"

    def test_masked_region_colored(self, compositor):
        frame = make_test_frame()
        masks = {"gallbladder": make_circle_mask()}
        result = compositor.composite(frame, masks)

        center_pixel = result[240, 427, :3].cpu().float()
        assert center_pixel[1] > center_pixel[0], "Green should dominate for gallbladder"

    def test_multiple_masks(self, compositor):
        frame = make_test_frame()
        masks = {
            "gallbladder": make_circle_mask(cx=200, cy=200, r=50),
            "grasper": make_rect_mask(x1=500, y1=280, x2=700, y2=320),
        }
        result = compositor.composite(frame, masks)
        assert result.shape == (480, 854, 4)

        gb_pixel = result[200, 200, :3].cpu()
        gr_pixel = result[300, 600, :3].cpu()
        assert not torch.equal(gb_pixel, gr_pixel)

    def test_empty_masks(self, compositor):
        frame = make_test_frame()
        result = compositor.composite(frame, {})

        assert result.shape == (480, 854, 4)
        assert torch.equal(result[:, :, :3].cpu(), frame.cpu())

    def test_glow_effect_visible(self, compositor):
        frame = make_test_frame()
        masks = {"gallbladder": make_circle_mask(cx=427, cy=240, r=60)}
        result = compositor.composite(frame, masks)

        bg_brightness = result[10, 10, :3].cpu().float().sum()
        glow_brightness = result[240, 427 + 65, :3].cpu().float().sum()

        print(f"  Background brightness: {bg_brightness:.0f}")
        print(f"  Glow region brightness: {glow_brightness:.0f}")
        assert glow_brightness > bg_brightness, "Glow should be brighter than background"

    def test_simple_composite_api(self, compositor):
        frame = make_test_frame()
        mask_tensor = torch.stack(
            [
                make_circle_mask(cx=200, cy=200),
                make_rect_mask(x1=500, y1=280, x2=700, y2=320),
            ]
        )
        labels = ["gallbladder", "grasper"]

        result = compositor.composite_simple(frame, mask_tensor, labels)
        assert result.shape == (480, 854, 4)

    def test_save_visual_output(self, compositor):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        frame = make_test_frame()
        masks = {
            "gallbladder": make_circle_mask(cx=300, cy=240, r=70),
            "cystic_duct": make_circle_mask(cx=450, cy=200, r=30),
            "grasper": make_rect_mask(x1=500, y1=300, x2=750, y2=330),
        }
        result = compositor.composite(frame, masks)

        result_np = result.cpu().numpy()
        try:
            import cv2

            bgra = cv2.cvtColor(result_np, cv2.COLOR_RGBA2BGRA)
            out_path = os.path.join(OUTPUT_DIR, "overlay_smoke_test.png")
            cv2.imwrite(out_path, bgra)
            print(f"  [ok] Visual output saved: {out_path}")
        except ImportError:
            print("  [warn] OpenCV not available for PNG save (non-fatal)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
