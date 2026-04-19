"""
sam2_inference_op.py - SAM 2.1 segmentation backend.
"""

from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch

from operators.sam2_bootstrap import (
    ensure_vendored_sam2,
    import_sam2_symbols,
    normalize_sam_config_name,
    resolve_repo_path,
)

try:
    import holoscan.core
    from holoscan.core import Operator, OperatorSpec

    HAS_HOLOSCAN = True
except ImportError:
    HAS_HOLOSCAN = False

try:
    build_sam2, SAM2ImagePredictor = import_sam2_symbols()

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False


class SAM2Segmenter:
    """Standalone SAM 2.1 segmentation engine."""

    def __init__(
        self,
        checkpoint: str,
        model_cfg: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        vos_optimized: bool = True,
        max_objects: int = 5,
        prompt_classes: Optional[List[str]] = None,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
        self.device_type = torch.device(self.device).type
        self.max_objects = max_objects
        self.prompt_classes = set(prompt_classes) if prompt_classes else None
        self.frame_count = 0
        self.active_tracks: Dict[str, torch.Tensor] = {}

        if not HAS_SAM2:
            raise ImportError(
                "SAM 2 required: pip install -e . from facebookresearch/sam2 repo"
            )

        self.torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        checkpoint_path = resolve_repo_path(checkpoint)
        model_cfg_name = normalize_sam_config_name(model_cfg)

        print(f"  Loading SAM 2.1 from {checkpoint_path}...")
        ensure_vendored_sam2(reset_hydra=True)
        sam2_model = build_sam2(model_cfg_name, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        if vos_optimized and hasattr(torch, "compile"):
            try:
                self.predictor.model = torch.compile(
                    self.predictor.model, mode="reduce-overhead"
                )
                print("  [ok] torch.compile applied for SAM 2.1")
            except Exception as exc:
                print(f"  [warn] torch.compile failed (non-fatal): {exc}")

        print(f"  [ok] SAM 2.1 loaded on {self.device} ({dtype})")

    def _filter_detections(self, detections: List) -> List:
        if self.prompt_classes is None:
            return detections
        return [det for det in detections if det.class_name in self.prompt_classes]

    def segment_frame(
        self,
        frame: torch.Tensor,
        detections: Optional[List] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        self.frame_count += 1
        h, w = frame.shape[:2]

        frame_np = frame.detach().cpu().numpy()
        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

        autocast_ctx = (
            torch.autocast(device_type=self.device_type, dtype=self.torch_dtype)
            if self.device_type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            self.predictor.set_image(frame_np)
            masks_out: Dict[str, torch.Tensor] = {}

            if detections:
                filtered = self._filter_detections(detections)
                for det in filtered[: self.max_objects]:
                    x1, y1, x2, y2 = det.bbox
                    box = np.array(
                        [
                            max(0.0, min(float(x1), w - 1)),
                            max(0.0, min(float(y1), h - 1)),
                            max(0.0, min(float(x2), w - 1)),
                            max(0.0, min(float(y2), h - 1)),
                        ],
                        dtype=np.float32,
                    )

                    masks, _, _ = self.predictor.predict(
                        box=box,
                        multimask_output=False,
                    )
                    if masks is not None and len(masks) > 0:
                        best_mask = torch.from_numpy(masks[0] > 0).to(
                            device=self.device,
                            dtype=torch.float32,
                        )
                        masks_out[det.class_name] = best_mask
                        self.active_tracks[det.class_name] = best_mask
            elif self.active_tracks:
                masks_out = dict(self.active_tracks)

        return masks_out

    def get_mask_tensor(
        self,
        masks: Dict[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        if not masks:
            return torch.zeros(0, height, width, device=self.device)
        return torch.stack(list(masks.values()), dim=0)

    def get_mask_labels(self, masks: Dict[str, torch.Tensor]) -> List[str]:
        return list(masks.keys())

    def reset(self):
        self.frame_count = 0
        self.active_tracks.clear()


if HAS_HOLOSCAN:

    class SAM2InferenceOp(Operator):
        """Holoscan operator wrapper for SAM 2.1."""

        def setup(self, spec: OperatorSpec):
            spec.input("rgb_tensor")
            spec.input("bboxes")
            spec.output("mask_tensor")
            spec.output("mask_labels")
            spec.param("checkpoint")
            spec.param("model_cfg")
            spec.param("device", default_value="cuda:0")
            spec.param("dtype", default_value="bfloat16")
            spec.param("vos_optimized", default_value=True)
            spec.param("max_objects", default_value=5)
            spec.param("prompt_classes", default_value=None)

        def start(self):
            self.segmenter = SAM2Segmenter(
                checkpoint=self.checkpoint,
                model_cfg=self.model_cfg,
                device=self.device,
                dtype=self.dtype,
                vos_optimized=self.vos_optimized,
                max_objects=self.max_objects,
                prompt_classes=self.prompt_classes,
            )

        def compute(self, op_input, op_output, context):
            from operators.format_utils import holoscan_to_torch

            frame = holoscan_to_torch(op_input.receive("rgb_tensor"))
            detections = op_input.receive("bboxes")
            masks = self.segmenter.segment_frame(frame, detections)
            h, w = frame.shape[:2]

            op_output.emit(self.segmenter.get_mask_tensor(masks, h, w), "mask_tensor")
            op_output.emit(self.segmenter.get_mask_labels(masks), "mask_labels")

        def stop(self):
            self.segmenter.reset()
            del self.segmenter
            torch.cuda.empty_cache()
