"""
format_utils.py — GPU tensor conversion utilities for the Holoscan ↔ PyTorch bridge.

Provides zero-copy (where possible) conversions between Holoscan GXF tensors
and PyTorch tensors, plus GPU-accelerated image processing helpers needed
by the SAM 2.1 and overlay operators.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ImageNet normalization constants (used by SAM 2.1)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def holoscan_to_torch(tensor_data, device: str = "cuda:0") -> torch.Tensor:
    """
    Convert a Holoscan tensor (or numpy array) to a PyTorch GPU tensor.

    Attempts zero-copy via DLPack / __cuda_array_interface__ when the source
    is already on GPU. Falls back to copying from CPU if necessary.

    Args:
        tensor_data: Holoscan tensor, numpy array, or any DLPack-compatible object
        device: Target CUDA device

    Returns:
        torch.Tensor on the specified device
    """
    # Path 1: Already a PyTorch tensor
    if isinstance(tensor_data, torch.Tensor):
        if tensor_data.device.type == "cuda":
            return tensor_data
        return tensor_data.to(device)

    # Path 2: DLPack (zero-copy from GPU)
    if hasattr(tensor_data, "__dlpack__"):
        return torch.from_dlpack(tensor_data).to(device)

    # Path 3: CUDA array interface (zero-copy from GPU)
    if hasattr(tensor_data, "__cuda_array_interface__"):
        return torch.as_tensor(tensor_data, device=device)

    # Path 4: Numpy array (CPU → GPU copy)
    if isinstance(tensor_data, np.ndarray):
        return torch.from_numpy(tensor_data).to(device)

    # Path 5: Try generic conversion
    return torch.tensor(tensor_data, device=device)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to numpy array (CPU).

    Args:
        tensor: PyTorch tensor (any device)

    Returns:
        numpy array on CPU
    """
    return tensor.detach().cpu().numpy()


def resize_tensor_gpu(tensor: torch.Tensor, target_size: tuple) -> torch.Tensor:
    """
    GPU-accelerated resize using bilinear interpolation.

    Args:
        tensor: Input tensor of shape (H, W, C) or (C, H, W) or (B, C, H, W)
        target_size: (height, width) tuple

    Returns:
        Resized tensor in the same format as input
    """
    original_shape_len = len(tensor.shape)

    # Normalize to (B, C, H, W) for F.interpolate
    if original_shape_len == 3:
        if tensor.shape[2] <= 4:
            # (H, W, C) → (1, C, H, W)
            t = tensor.permute(2, 0, 1).unsqueeze(0).float()
            hwc = True
        else:
            # (C, H, W) → (1, C, H, W)
            t = tensor.unsqueeze(0).float()
            hwc = False
    elif original_shape_len == 4:
        t = tensor.float()
        hwc = False
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    resized = F.interpolate(t, size=target_size, mode="bilinear",
                            align_corners=False)

    # Restore original format
    if original_shape_len == 3:
        resized = resized.squeeze(0)
        if hwc:
            resized = resized.permute(1, 2, 0)

    return resized.to(tensor.dtype)


def normalize_for_sam2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization expected by SAM 2.1.

    Args:
        tensor: RGB image tensor of shape (H, W, 3) or (3, H, W), uint8 or float.
                If uint8, values are expected in [0, 255].
                If float, values are expected in [0.0, 1.0].

    Returns:
        Normalized float32 tensor in (3, H, W) format
    """
    t = tensor.float()

    # Convert uint8 [0, 255] → float [0.0, 1.0]
    if t.max() > 1.0:
        t = t / 255.0

    # Ensure (3, H, W) format
    if t.shape[-1] == 3:
        t = t.permute(2, 0, 1)

    # Normalize with ImageNet stats
    mean = IMAGENET_MEAN.to(t.device).view(3, 1, 1)
    std = IMAGENET_STD.to(t.device).view(3, 1, 1)
    t = (t - mean) / std

    return t


def denormalize_from_sam2(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse ImageNet normalization.

    Args:
        tensor: Normalized tensor in (3, H, W) format

    Returns:
        Denormalized uint8 tensor in (H, W, 3) format, values in [0, 255]
    """
    mean = IMAGENET_MEAN.to(tensor.device).view(3, 1, 1)
    std = IMAGENET_STD.to(tensor.device).view(3, 1, 1)

    t = tensor * std + mean
    t = (t * 255.0).clamp(0, 255).byte()
    t = t.permute(1, 2, 0)  # (3, H, W) → (H, W, 3)

    return t


def create_color_mask(mask: torch.Tensor, color: list,
                      device: str = "cuda:0") -> torch.Tensor:
    """
    Apply a solid RGBA color to a binary mask.

    Args:
        mask: Binary mask tensor (H, W), values 0 or 1
        color: [R, G, B, A] list with values 0-255

    Returns:
        RGBA tensor (H, W, 4), float32, values 0.0-1.0
    """
    h, w = mask.shape
    rgba = torch.zeros(h, w, 4, device=device, dtype=torch.float32)

    color_t = torch.tensor(color, device=device, dtype=torch.float32) / 255.0

    mask_bool = mask.bool()
    rgba[mask_bool] = color_t

    return rgba
