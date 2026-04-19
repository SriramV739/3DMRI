"""
Smoke test for format_utils.py — Step 1 of the build.

Tests tensor conversions, GPU resize, and SAM2 normalization.
Run: pytest surgery/tests/test_format_utils.py -v
"""

import sys
import os
import numpy as np
import torch
import pytest

# Add parent to path so we can import operators
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.format_utils import (
    holoscan_to_torch,
    torch_to_numpy,
    resize_tensor_gpu,
    normalize_for_sam2,
    denormalize_from_sam2,
    create_color_mask,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# Skip all tests if no CUDA GPU available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA GPU required for smoke tests"
)

DEVICE = "cuda:0"


class TestHoloscanToTorch:
    """Test tensor conversion from various formats to PyTorch GPU tensor."""

    def test_numpy_to_gpu(self):
        """Numpy array (CPU) should be copied to GPU."""
        arr = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
        tensor = holoscan_to_torch(arr, device=DEVICE)
        assert tensor.device.type == "cuda"
        assert tensor.shape == (480, 854, 3)
        assert torch.equal(tensor.cpu(), torch.from_numpy(arr))

    def test_torch_cpu_to_gpu(self):
        """PyTorch CPU tensor should be moved to GPU."""
        cpu_tensor = torch.randn(480, 854, 3)
        gpu_tensor = holoscan_to_torch(cpu_tensor, device=DEVICE)
        assert gpu_tensor.device.type == "cuda"
        assert gpu_tensor.shape == (480, 854, 3)

    def test_torch_gpu_passthrough(self):
        """PyTorch GPU tensor should pass through without copy."""
        gpu_tensor = torch.randn(480, 854, 3, device=DEVICE)
        result = holoscan_to_torch(gpu_tensor, device=DEVICE)
        assert result.data_ptr() == gpu_tensor.data_ptr(), "Should be zero-copy"

    def test_dlpack_roundtrip(self):
        """DLPack conversion should be zero-copy."""
        original = torch.randn(480, 854, 3, device=DEVICE)
        # Simulate DLPack capsule
        dlpack_capsule = torch.utils.dlpack.to_dlpack(original)
        recovered = torch.from_dlpack(dlpack_capsule)
        assert torch.equal(original, recovered)


class TestTorchToNumpy:
    """Test conversion from PyTorch to numpy."""

    def test_gpu_to_numpy(self):
        """GPU tensor should be detached and moved to CPU."""
        gpu_tensor = torch.randn(480, 854, 3, device=DEVICE)
        arr = torch_to_numpy(gpu_tensor)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (480, 854, 3)


class TestResizeTensorGPU:
    """Test GPU-accelerated resize."""

    def test_hwc_resize(self):
        """Resize (H, W, C) tensor."""
        tensor = torch.randint(0, 255, (480, 854, 3), device=DEVICE, dtype=torch.uint8)
        resized = resize_tensor_gpu(tensor, target_size=(1024, 1024))
        assert resized.shape == (1024, 1024, 3)
        assert resized.device.type == "cuda"

    def test_chw_resize(self):
        """Resize (C, H, W) tensor."""
        tensor = torch.randn(3, 480, 854, device=DEVICE)
        resized = resize_tensor_gpu(tensor, target_size=(1024, 1024))
        assert resized.shape == (3, 1024, 1024)

    def test_batch_resize(self):
        """Resize (B, C, H, W) tensor."""
        tensor = torch.randn(2, 3, 480, 854, device=DEVICE)
        resized = resize_tensor_gpu(tensor, target_size=(256, 256))
        assert resized.shape == (2, 3, 256, 256)

    def test_preserves_device(self):
        """Output should stay on same GPU."""
        tensor = torch.randn(480, 854, 3, device=DEVICE)
        resized = resize_tensor_gpu(tensor, target_size=(256, 256))
        assert resized.device == tensor.device


class TestNormalization:
    """Test SAM 2.1 ImageNet normalization."""

    def test_normalize_uint8_hwc(self):
        """uint8 (H,W,3) → normalized float32 (3,H,W)."""
        img = torch.randint(0, 255, (480, 854, 3), device=DEVICE, dtype=torch.uint8)
        normed = normalize_for_sam2(img)
        assert normed.shape == (3, 480, 854)
        assert normed.dtype == torch.float32
        # Check rough range: ImageNet normalized values typically in [-2.5, 2.5]
        assert normed.min() > -3.0
        assert normed.max() < 3.0

    def test_normalize_float_chw(self):
        """float (3,H,W) in [0,1] → normalized."""
        img = torch.rand(3, 480, 854, device=DEVICE)
        normed = normalize_for_sam2(img)
        assert normed.shape == (3, 480, 854)

    def test_roundtrip_normalization(self):
        """Normalize → denormalize should approximately recover original."""
        img = torch.randint(0, 255, (480, 854, 3), device=DEVICE, dtype=torch.uint8)
        normed = normalize_for_sam2(img)
        recovered = denormalize_from_sam2(normed)
        assert recovered.shape == (480, 854, 3)
        assert recovered.dtype == torch.uint8
        # Allow small rounding errors
        diff = (img.float() - recovered.float()).abs().mean()
        assert diff < 2.0, f"Roundtrip error too large: {diff}"


class TestCreateColorMask:
    """Test color mask creation."""

    def test_basic_color_mask(self):
        """Binary mask + color → RGBA output."""
        mask = torch.zeros(480, 854, device=DEVICE)
        mask[100:200, 100:200] = 1.0  # 100x100 square
        color = [0, 255, 128, 160]  # Green with alpha

        rgba = create_color_mask(mask, color, device=DEVICE)
        assert rgba.shape == (480, 854, 4)
        assert rgba.dtype == torch.float32

        # Check masked region has the right color
        assert rgba[150, 150, 0].item() == pytest.approx(0 / 255, abs=0.01)  # R
        assert rgba[150, 150, 1].item() == pytest.approx(255 / 255, abs=0.01)  # G
        assert rgba[150, 150, 2].item() == pytest.approx(128 / 255, abs=0.01)  # B
        assert rgba[150, 150, 3].item() == pytest.approx(160 / 255, abs=0.01)  # A

        # Check unmasked region is transparent
        assert rgba[0, 0, 3].item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
