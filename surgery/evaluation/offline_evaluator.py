"""
Offline MedSAM2 evaluation workflow for staged surgical videos.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from data.convert_video import VideoAsset
from operators.overlay_compositor_op import OverlayCompositor
from operators.scene_copilot_op import SurgicalSceneCopilot
from operators.vlm_prompt_op import AnatomyVLMGuide
from operators.yolo_detection_op import Detection, YOLODetector


def load_seed_prompts(prompt_dir: Optional[str], video_name: str) -> Dict[int, List[Detection]]:
    """Load optional per-video fallback prompts from JSON."""
    if not prompt_dir:
        return {}

    prompt_path = Path(prompt_dir) / f"{video_name}.json"
    if not prompt_path.exists():
        return {}

    with open(prompt_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    detections_by_frame = raw.get("detections_by_frame", raw)
    parsed: Dict[int, List[Detection]] = {}
    for frame_key, detection_items in detections_by_frame.items():
        frame_idx = int(frame_key)
        parsed[frame_idx] = [
            Detection(
                class_name=item["class_name"],
                bbox=[float(v) for v in item["bbox"]],
                confidence=float(item.get("confidence", 1.0)),
                class_id=int(item.get("class_id", 0)),
            )
            for item in detection_items
        ]
    return parsed


class OfflineVideoEvaluator:
    """Run YOLO -> segmenter -> overlay over staged videos and save artifacts."""

    def __init__(
        self,
        detector: YOLODetector,
        segmenter,
        vlm_guide: Optional[AnatomyVLMGuide],
        scene_copilot: Optional[SurgicalSceneCopilot],
        overlay: OverlayCompositor,
        output_dir: str,
        max_frames: Optional[int] = 600,
        save_overlays: bool = False,
        save_masks: bool = False,
        prompt_dir: Optional[str] = None,
        overlay_sample_stride: int = 100,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.vlm_guide = vlm_guide
        self.scene_copilot = scene_copilot
        self.overlay = overlay
        self.output_dir = Path(output_dir)
        self.max_frames = max_frames
        self.save_overlays = save_overlays
        self.save_masks = save_masks
        self.prompt_dir = prompt_dir
        self.overlay_sample_stride = overlay_sample_stride

    def evaluate_assets(self, assets: List[VideoAsset]) -> List[dict]:
        results = [self.evaluate_asset(asset) for asset in assets]
        summary_path = self.output_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results

    def evaluate_asset(self, asset: VideoAsset) -> dict:
        if not asset.bundle_path:
            raise RuntimeError(f"Bundle path missing for asset {asset.name}")

        frames = np.load(asset.bundle_path)
        total_frames = frames.shape[0]
        if self.max_frames is not None:
            total_frames = min(total_frames, self.max_frames)

        prompt_overrides = load_seed_prompts(self.prompt_dir, asset.name)
        result_dir = self.output_dir / asset.name
        overlay_dir = result_dir / "overlay_frames"
        mask_dir = result_dir / "masks"
        analysis_path = result_dir / "scene_analysis.jsonl"
        result_dir.mkdir(parents=True, exist_ok=True)
        if self.scene_copilot is not None and analysis_path.exists():
            analysis_path.unlink()

        overlay_writer = None
        if self.save_overlays:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            try:
                import cv2

                overlay_video_path = result_dir / "overlay.mp4"
                writer_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                width, height = asset.resolution
                overlay_writer = cv2.VideoWriter(
                    str(overlay_video_path),
                    writer_fourcc,
                    asset.fps,
                    (width, height),
                )
            except ImportError:
                overlay_writer = None

        if self.save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        self.detector.reset()
        if self.vlm_guide is not None:
            self.vlm_guide.reset()
        if self.scene_copilot is not None:
            self.scene_copilot.reset()
        if hasattr(self.segmenter, "reset"):
            self.segmenter.reset()
        if getattr(self.segmenter, "use_temporal_memory", False) and getattr(asset, "frames_dir", None):
            self.segmenter.prepare_video(asset.frames_dir, propagation_window=self.detector.detect_every_n_frames)

        latencies_ms: List[float] = []
        frames_with_yolo_detections = 0
        frames_with_prompts = 0
        frames_with_masks = 0
        total_masks_emitted = 0
        total_mask_pixels = 0.0
        frames_using_seed_prompts = 0
        frames_with_vlm_targets = 0
        scene_analysis_refreshes = 0
        scene_analysis_latencies_ms: List[float] = []

        processing_device = getattr(self.segmenter, "device", getattr(self.detector, "device", "cpu"))
        if str(processing_device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for frame_idx in range(total_frames):
            frame_np = frames[frame_idx]
            frame_t = torch.from_numpy(frame_np).to(processing_device)

            start = time.perf_counter()
            raw_detections = self.detector.detect(frame_t)
            detections = raw_detections
            if raw_detections:
                frames_with_yolo_detections += 1
                frames_with_prompts += 1
            else:
                detections = prompt_overrides.get(frame_idx, [])
                if detections:
                    frames_using_seed_prompts += 1
                    frames_with_prompts += 1

            if self.vlm_guide is not None:
                selection = self.vlm_guide.select_prompts(frame_t, detections, frame_idx=frame_idx)
                detections = selection.filtered_detections
                if selection.target_labels:
                    frames_with_vlm_targets += 1

            masks = self.segmenter.segment_frame(
                frame_t,
                detections if detections else None,
                frame_idx=frame_idx,
            )
            scene_analysis = None
            if self.scene_copilot is not None:
                analysis_start = time.perf_counter()
                scene_analysis = self.scene_copilot.analyze(
                    frame_t,
                    detections=raw_detections or detections,
                    masks=masks,
                    frame_idx=frame_idx,
                )
                scene_analysis_latencies_ms.append((time.perf_counter() - analysis_start) * 1000.0)
                if scene_analysis.refreshed:
                    scene_analysis_refreshes += 1
                    with open(analysis_path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(scene_analysis.to_dict()) + "\n")
            composited = self.overlay.composite(frame_t, masks)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)

            if masks:
                frames_with_masks += 1
                total_masks_emitted += len(masks)
                total_mask_pixels += sum(mask.sum().item() for mask in masks.values())

            if self.save_masks:
                mask_tensor = self.segmenter.get_mask_tensor(masks, frame_np.shape[0], frame_np.shape[1])
                labels = self.segmenter.get_mask_labels(masks)
                np.savez_compressed(
                    mask_dir / f"frame_{frame_idx:05d}.npz",
                    mask_tensor=mask_tensor.detach().cpu().numpy(),
                    labels=np.array(labels, dtype=object),
                )

            if self.save_overlays:
                result_np = composited.detach().cpu().numpy()
                try:
                    import cv2

                    if frame_idx % self.overlay_sample_stride == 0:
                        bgra = cv2.cvtColor(result_np, cv2.COLOR_RGBA2BGRA)
                        cv2.imwrite(str(overlay_dir / f"frame_{frame_idx:05d}.png"), bgra)
                    if overlay_writer is not None:
                        bgr = cv2.cvtColor(result_np[:, :, :3], cv2.COLOR_RGB2BGR)
                        overlay_writer.write(bgr)
                except ImportError:
                    pass

        if overlay_writer is not None:
            overlay_writer.release()

        latencies = np.asarray(latencies_ms, dtype=np.float32)
        peak_vram_gb = 0.0
        if str(processing_device).startswith("cuda") and torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        metrics = {
            "video_name": asset.name,
            "source_path": asset.source_path,
            "frames_processed": total_frames,
            "frames_with_yolo_detections": frames_with_yolo_detections,
            "frames_with_prompts": frames_with_prompts,
            "frames_with_non_empty_masks": frames_with_masks,
            "frames_using_seed_prompts": frames_using_seed_prompts,
            "frames_with_vlm_targets": frames_with_vlm_targets,
            "scene_analysis_refreshes": scene_analysis_refreshes,
            "average_masks_per_frame": float(total_masks_emitted / max(total_frames, 1)),
            "average_mask_pixels_per_masked_frame": float(total_mask_pixels / max(frames_with_masks, 1)),
            "latency_ms_median": float(np.median(latencies)) if len(latencies) else 0.0,
            "latency_ms_p95": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
            "scene_analysis_latency_ms_median": float(np.median(scene_analysis_latencies_ms)) if scene_analysis_latencies_ms else 0.0,
            "peak_vram_gb": peak_vram_gb,
            "overlay_artifacts_dir": str(overlay_dir) if self.save_overlays else None,
            "mask_artifacts_dir": str(mask_dir) if self.save_masks else None,
            "scene_analysis_path": str(analysis_path) if self.scene_copilot is not None else None,
        }

        with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"[ok] Evaluated {asset.name}: {total_frames} frames, "
              f"{frames_with_masks} frames with masks, p95 {metrics['latency_ms_p95']:.1f}ms")
        return metrics
