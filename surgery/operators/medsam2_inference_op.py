"""
medsam2_inference_op.py - MedSAM2 segmentation backend.

Provides:
- an image-predictor path that preserves the current streaming operator contract
- an optional video-predictor path for offline staged-video evaluation
"""

from __future__ import annotations

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
    build_sam2, build_sam2_video_predictor, SAM2ImagePredictor = import_sam2_symbols(
        include_video_predictor=True
    )

    HAS_MEDSAM2 = True
except ImportError:
    HAS_MEDSAM2 = False


class MedSAM2Segmenter:
    """MedSAM2 segmentation engine with optional staged-video temporal propagation."""

    def __init__(
        self,
        checkpoint: str,
        model_cfg: str = "configs/sam2.1_hiera_t512.yaml",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_objects: int = 5,
        use_temporal_memory: bool = True,
        prompt_classes: Optional[List[str]] = None,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        if not HAS_MEDSAM2:
            raise ImportError(
                "MedSAM2 is required: pip install -e from https://github.com/bowang-lab/MedSAM2.git"
            )

        self.device_type = torch.device(self.device).type
        self.max_objects = max_objects
        self.use_temporal_memory = use_temporal_memory
        self.prompt_classes = set(prompt_classes) if prompt_classes else None
        self.torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.checkpoint = resolve_repo_path(checkpoint)
        self.model_cfg = normalize_sam_config_name(model_cfg)

        self.frame_count = 0
        self.active_tracks: Dict[str, torch.Tensor] = {}
        self.video_mask_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.object_id_by_label: Dict[str, int] = {}
        self.label_by_object_id: Dict[int, str] = {}
        self.next_object_id = 1
        self.video_predictor = None
        self.inference_state = None
        self.video_source = None
        self.propagation_window = 1

        print(f"  Loading MedSAM2 image predictor from {self.checkpoint}...")
        ensure_vendored_sam2(reset_hydra=True)
        model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        print(f"  [ok] MedSAM2 image predictor loaded on {self.device} ({dtype})")

    def _get_or_create_object_id(self, label: str) -> int:
        if label not in self.object_id_by_label:
            obj_id = self.next_object_id
            self.next_object_id += 1
            self.object_id_by_label[label] = obj_id
            self.label_by_object_id[obj_id] = label
        return self.object_id_by_label[label]

    def _maybe_build_video_predictor(self):
        if self.video_predictor is not None:
            return
        print(f"  Loading MedSAM2 video predictor from {self.checkpoint}...")
        ensure_vendored_sam2(reset_hydra=True)
        self.video_predictor = build_sam2_video_predictor(
            config_file=self.model_cfg,
            ckpt_path=self.checkpoint,
            device=self.device,
        )
        print(f"  [ok] MedSAM2 video predictor loaded on {self.device}")

    def prepare_video(self, video_source: str, propagation_window: int = 15):
        """Initialize staged-video inference for temporal propagation."""
        if not self.use_temporal_memory:
            return

        self._maybe_build_video_predictor()
        self.video_source = video_source
        self.propagation_window = max(1, propagation_window)
        self.inference_state = self.video_predictor.init_state(
            video_path=video_source,
            async_loading_frames=False,
            offload_video_to_cpu=self.device_type != "cuda",
            offload_state_to_cpu=self.device_type != "cuda",
        )
        self.video_mask_cache.clear()
        self.object_id_by_label.clear()
        self.label_by_object_id.clear()
        self.next_object_id = 1

    def _image_segment_frame(
        self,
        frame: torch.Tensor,
        detections: Optional[List] = None,
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
            masks_out = {}

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
                    masks, _, _ = self.predictor.predict(box=box, multimask_output=False)
                    if masks is not None and len(masks) > 0:
                        best_mask = torch.from_numpy(masks[0] > 0).to(
                            device=self.device, dtype=torch.float32
                        )
                        if det.class_name in masks_out:
                            masks_out[det.class_name] = torch.max(masks_out[det.class_name], best_mask)
                        else:
                            masks_out[det.class_name] = best_mask

                        self.active_tracks[det.class_name] = masks_out[det.class_name]
            elif self.active_tracks:
                masks_out = dict(self.active_tracks)

        return masks_out

    def _convert_video_output_to_masks(self, obj_ids, out_mask_logits) -> Dict[str, torch.Tensor]:
        masks = {}
        if out_mask_logits is None:
            return masks

        for idx, obj_id in enumerate(obj_ids):
            label = self.label_by_object_id.get(int(obj_id), f"obj_{obj_id}")
            mask = (out_mask_logits[idx] > 0).to(device=self.device, dtype=torch.float32)
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            masks[label] = mask
        if masks:
            self.active_tracks = dict(masks)
        return masks

    def _video_segment_frame(
        self,
        detections: Optional[List],
        frame_idx: int,
    ) -> Dict[str, torch.Tensor]:
        if self.inference_state is None or self.video_predictor is None:
            return {}

        if detections:
            filtered = self._filter_detections(detections)
            current_masks = {}
            for det in filtered[: self.max_objects]:
                obj_id = self._get_or_create_object_id(det.class_name)
                _, obj_ids, mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=np.array(det.bbox, dtype=np.float32),
                )
                current_masks = self._convert_video_output_to_masks(obj_ids, mask_logits)

            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=self.propagation_window,
                reverse=False,
            ):
                self.video_mask_cache[out_frame_idx] = self._convert_video_output_to_masks(
                    out_obj_ids,
                    out_mask_logits,
                )

            return self.video_mask_cache.get(frame_idx, current_masks)

        if frame_idx in self.video_mask_cache:
            cached = self.video_mask_cache[frame_idx]
            self.active_tracks = dict(cached)
            return cached

        if self.active_tracks:
            return dict(self.active_tracks)
        return {}

    def _filter_detections(self, detections: List) -> List:
        if self.prompt_classes is None:
            return detections
        return [
            det
            for det in detections
            if det.class_name in self.prompt_classes or getattr(det, "source_model", "") == "vlm_prompt_box"
        ]

    def segment_frame(
        self,
        frame: torch.Tensor,
        detections: Optional[List] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Segment a frame using MedSAM2."""
        if (
            self.use_temporal_memory
            and self.inference_state is not None
            and self.video_predictor is not None
            and frame_idx is not None
        ):
            return self._video_segment_frame(detections, frame_idx)
        return self._image_segment_frame(frame, detections)

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
        self.video_mask_cache.clear()
        self.object_id_by_label.clear()
        self.label_by_object_id.clear()
        self.next_object_id = 1
        if self.video_predictor is not None and self.inference_state is not None:
            self.video_predictor.reset_state(self.inference_state)
        self.inference_state = None
        self.video_source = None


if HAS_HOLOSCAN:

    class MedSAM2InferenceOp(Operator):
        """Holoscan operator wrapper for MedSAM2."""

        def setup(self, spec: OperatorSpec):
            spec.input("rgb_tensor")
            spec.input("bboxes")
            spec.output("mask_tensor")
            spec.output("mask_labels")
            spec.param("checkpoint")
            spec.param("model_cfg", default_value="configs/sam2.1_hiera_t512.yaml")
            spec.param("device", default_value="cuda:0")
            spec.param("dtype", default_value="bfloat16")
            spec.param("max_objects", default_value=5)
            spec.param("use_temporal_memory", default_value=True)
            spec.param("prompt_classes", default_value=None)

        def start(self):
            self.segmenter = MedSAM2Segmenter(
                checkpoint=self.checkpoint,
                model_cfg=self.model_cfg,
                device=self.device,
                dtype=self.dtype,
                max_objects=self.max_objects,
                use_temporal_memory=self.use_temporal_memory,
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
