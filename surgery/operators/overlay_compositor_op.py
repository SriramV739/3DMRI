"""
overlay_compositor_op.py — AR overlay compositor with glowing edge effects.

Takes the original video frame + segmentation masks from SAM 2.1 and composites
a beautiful, color-coded, glowing AR overlay for each tracked structure.

This is the "artist" of the pipeline — it makes the output look stunning.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import holoscan.core
    from holoscan.core import Operator, OperatorSpec
    HAS_HOLOSCAN = True
except ImportError:
    HAS_HOLOSCAN = False


# Default color palette: structure_name → [R, G, B, A] (0-255)
DEFAULT_COLORS = {
    "gallbladder":  [0, 255, 128, 160],     # Green
    "cystic_duct":  [255, 165, 0, 140],     # Orange
    "cystic_artery": [255, 50, 50, 140],    # Red
    "grasper":      [100, 149, 237, 120],   # Steel blue
    "hook":         [186, 85, 211, 120],    # Purple
}

# Fallback colors for unknown classes (cycle through)
FALLBACK_COLORS = [
    [255, 215, 0, 130],    # Gold
    [0, 191, 255, 130],    # Deep sky blue
    [255, 105, 180, 130],  # Hot pink
    [50, 205, 50, 130],    # Lime green
    [255, 127, 80, 130],   # Coral
]


class OverlayCompositor:
    """
    Standalone AR overlay compositor (decoupled from Holoscan for testability).

    Produces color-coded overlays with optional glowing edge effects.
    """

    def __init__(self, colors: Optional[Dict[str, List[int]]] = None,
                 blend_alpha: float = 0.45,
                 glow_effect: bool = True,
                 glow_radius: int = 8,
                 contour_thickness: int = 2,
                 device: str = "cuda:0"):
        self.colors = colors or DEFAULT_COLORS
        self.blend_alpha = blend_alpha
        self.glow_effect = glow_effect
        self.glow_radius = glow_radius
        self.contour_thickness = contour_thickness
        self.device = device
        self._fallback_idx = 0
        self._assigned_fallbacks: Dict[str, List[int]] = {}

        # Pre-compute Gaussian kernel for glow effect
        if self.glow_effect:
            self.glow_kernel = self._make_gaussian_kernel(glow_radius).to(device)

    def _make_gaussian_kernel(self, radius: int) -> torch.Tensor:
        """Create a 2D Gaussian kernel for glow blurring."""
        size = 2 * radius + 1
        x = torch.arange(size, dtype=torch.float32) - radius
        kernel_1d = torch.exp(-x ** 2 / (2 * (radius / 2.5) ** 2))
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        # Shape for F.conv2d: (1, 1, H, W)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def _get_color(self, class_name: str) -> List[int]:
        """Get RGBA color for a class, with fallback for unknown classes."""
        if class_name in self.colors:
            return self.colors[class_name]
        if class_name not in self._assigned_fallbacks:
            color = FALLBACK_COLORS[self._fallback_idx % len(FALLBACK_COLORS)]
            self._assigned_fallbacks[class_name] = color
            self._fallback_idx += 1
        return self._assigned_fallbacks[class_name]

    def _compute_contour(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract contour (edge) pixels from a binary mask using morphological gradient.

        Args:
            mask: Binary mask (H, W), float, values 0 or 1

        Returns:
            Contour mask (H, W), float
        """
        # Dilate using max pooling
        k = self.contour_thickness * 2 + 1
        mask_4d = mask.unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(mask_4d, kernel_size=k, stride=1,
                               padding=k // 2)
        dilated = dilated.squeeze(0).squeeze(0)

        # Contour = dilated - original
        contour = (dilated - mask).clamp(0, 1)
        return contour

    def _compute_glow(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Create a soft glow around the mask boundary.

        Args:
            mask: Binary mask (H, W), float

        Returns:
            Glow mask (H, W), float, values 0-1 (brightest at boundary)
        """
        contour = self._compute_contour(mask)

        # Blur the contour to create glow
        contour_4d = contour.unsqueeze(0).unsqueeze(0)
        glow = F.conv2d(contour_4d, self.glow_kernel.to(mask.device),
                        padding=self.glow_radius)
        glow = glow.squeeze(0).squeeze(0)

        # Normalize to 0-1 range
        if glow.max() > 0:
            glow = glow / glow.max()

        return glow

    def composite(self, frame: torch.Tensor,
                  masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Composite AR overlays onto the video frame.

        Args:
            frame: RGB tensor (H, W, 3), uint8, on GPU
            masks: Dict of class_name → binary mask tensor (H, W)

        Returns:
            Composited RGBA tensor (H, W, 4), uint8, on GPU
        """
        h, w = frame.shape[:2]

        # Start with original frame as float, add alpha channel
        result = torch.zeros(h, w, 4, device=frame.device, dtype=torch.float32)
        result[:, :, :3] = frame.float()
        result[:, :, 3] = 255.0  # Fully opaque background

        for class_name, mask in masks.items():
            if mask.sum() == 0:
                continue

            color = self._get_color(class_name)
            r, g, b, a = [c / 255.0 for c in color]
            alpha = a * self.blend_alpha

            mask_f = mask.float()

            # --- Filled overlay ---
            mask_bool = mask_f > 0.5
            result[:, :, 0][mask_bool] = (
                result[:, :, 0][mask_bool] * (1 - alpha) + r * 255 * alpha
            )
            result[:, :, 1][mask_bool] = (
                result[:, :, 1][mask_bool] * (1 - alpha) + g * 255 * alpha
            )
            result[:, :, 2][mask_bool] = (
                result[:, :, 2][mask_bool] * (1 - alpha) + b * 255 * alpha
            )

            # --- Contour outline ---
            if self.contour_thickness > 0:
                contour = self._compute_contour(mask_f)
                contour_bool = contour > 0.5
                brightness = min(1.0, r * 1.5), min(1.0, g * 1.5), min(1.0, b * 1.5)
                result[:, :, 0][contour_bool] = brightness[0] * 255
                result[:, :, 1][contour_bool] = brightness[1] * 255
                result[:, :, 2][contour_bool] = brightness[2] * 255

            # --- Glow effect ---
            if self.glow_effect:
                glow = self._compute_glow(mask_f)
                glow_alpha = glow * 0.6  # Glow intensity
                result[:, :, 0] += glow_alpha * r * 255
                result[:, :, 1] += glow_alpha * g * 255
                result[:, :, 2] += glow_alpha * b * 255

        # Clamp and convert to uint8
        result = result.clamp(0, 255).byte()
        return result

    def composite_simple(self, frame: torch.Tensor,
                         mask_tensor: torch.Tensor,
                         labels: List[str]) -> torch.Tensor:
        """
        Convenience method: composite from stacked mask tensor + label list.

        Args:
            frame: RGB tensor (H, W, 3)
            mask_tensor: Stacked masks (N, H, W)
            labels: List of N class names

        Returns:
            Composited RGBA tensor (H, W, 4), uint8
        """
        masks = {}
        for i, label in enumerate(labels):
            if i < mask_tensor.shape[0]:
                masks[label] = mask_tensor[i]
        return self.composite(frame, masks)


# --- Holoscan Operator wrapper ---
if HAS_HOLOSCAN:
    class OverlayCompositorOp(Operator):
        """
        Holoscan operator for AR overlay compositing.

        Inputs:
            rgb_tensor: Original video frame (H, W, 3)
            mask_tensor: Stacked masks (N, H, W) from SAM
            mask_labels: List of class names

        Outputs:
            composited_frame: AR overlay result (H, W, 4) RGBA
        """

        def setup(self, spec: OperatorSpec):
            spec.input("rgb_tensor")
            spec.input("mask_tensor")
            spec.input("mask_labels")
            spec.output("composited_frame")

        def start(self):
            self.compositor = OverlayCompositor(device="cuda:0")
            print("[ok] Overlay compositor initialized")

        def compute(self, op_input, op_output, context):
            from operators.format_utils import holoscan_to_torch
            frame = holoscan_to_torch(op_input.receive("rgb_tensor"))
            mask_tensor = holoscan_to_torch(op_input.receive("mask_tensor"))
            labels = op_input.receive("mask_labels")

            composited = self.compositor.composite_simple(
                frame, mask_tensor, labels
            )
            op_output.emit(composited, "composited_frame")

        def stop(self):
            del self.compositor
