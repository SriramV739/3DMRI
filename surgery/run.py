"""
run.py - CLI entry point for the Surgical AR Pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from app import HAS_HOLOSCAN, SurgicalARApp, load_app_config
from data.convert_video import extract_frames, extract_video_batch
from evaluation.offline_evaluator import OfflineVideoEvaluator
from operators.medsam2_inference_op import MedSAM2Segmenter
from operators.overlay_compositor_op import OverlayCompositor
from operators.roboflow_detection_op import RoboflowHostedDetector
from operators.sam2_inference_op import SAM2Segmenter
from operators.scene_copilot_op import SurgicalSceneCopilot
from operators.vlm_prompt_op import AnatomyVLMGuide
from operators.yolo_detection_op import CombinedYOLODetector, YOLODetector


def convert_video_if_needed(video_path: str, output_dir: str = "data/converted"):
    """Convert raw MP4 to a GXF-like numpy bundle if not already done."""
    if os.path.exists(os.path.join(output_dir, "surgery_video.gxf_entities.npy")):
        print(f"[ok] Converted video already exists in {output_dir}")
        return

    print(f"[info] Converting {video_path} -> GXF entities...")
    extract_frames(video_path, output_dir)


def build_segmenter_from_config(cfg):
    backend = cfg.get("segmenter", {}).get("backend", "medsam2").lower()
    prompt_classes = cfg.get("segmenter", {}).get("prompt_classes")
    if backend == "medsam2":
        seg_cfg = cfg["medsam2"]
        return MedSAM2Segmenter(
            checkpoint=seg_cfg["checkpoint"],
            model_cfg=seg_cfg["model_cfg"],
            device=seg_cfg["device"],
            dtype=seg_cfg["dtype"],
            max_objects=seg_cfg["max_objects"],
            use_temporal_memory=seg_cfg["use_temporal_memory"],
            prompt_classes=prompt_classes,
        )

    if backend == "sam2":
        seg_cfg = cfg["sam2"]
        return SAM2Segmenter(
            checkpoint=seg_cfg["checkpoint"],
            model_cfg=seg_cfg["model_cfg"],
            device=seg_cfg["device"],
            dtype=seg_cfg["dtype"],
            vos_optimized=seg_cfg["vos_optimized"],
            max_objects=seg_cfg["max_objects"],
            prompt_classes=prompt_classes,
        )

    raise ValueError(f"Unsupported segmenter backend: {backend}")


def build_vlm_guide_from_config(cfg):
    vlm_cfg = cfg.get("vlm", {})
    return AnatomyVLMGuide(
        enabled=vlm_cfg.get("enabled", False),
        provider=vlm_cfg.get("provider", "rule_based"),
        user_query=vlm_cfg.get("user_query", ""),
        candidate_labels=vlm_cfg.get("candidate_labels"),
        anatomy_aliases=vlm_cfg.get("anatomy_aliases"),
        prompt_every_n_frames=vlm_cfg.get("prompt_every_n_frames", 30),
        max_image_size=vlm_cfg.get("max_image_size", 512),
        api_url=vlm_cfg.get("api_url", ""),
        api_key=vlm_cfg.get("api_key", ""),
        api_key_env=vlm_cfg.get("api_key_env", "VLM_API_KEY"),
        model=vlm_cfg.get("model", ""),
    )


def build_scene_copilot_from_config(cfg, fps: Optional[float] = None, force_refresh_every_frame: bool = False):
    copilot_cfg = cfg.get("scene_copilot", {})
    if not copilot_cfg.get("enabled", False):
        return None

    if force_refresh_every_frame:
        refresh_every_n_frames = 1
    else:
        refresh_seconds = copilot_cfg.get("refresh_interval_seconds", 2.0)
        if fps is not None:
            refresh_every_n_frames = max(1, int(round(refresh_seconds * fps)))
        else:
            refresh_every_n_frames = max(1, copilot_cfg.get("refresh_every_n_frames", 30))

    return SurgicalSceneCopilot(
        enabled=copilot_cfg.get("enabled", True),
        provider=copilot_cfg.get("provider", "rule_based"),
        user_query=copilot_cfg.get("user_query", ""),
        refresh_every_n_frames=refresh_every_n_frames,
        max_history_frames=copilot_cfg.get("max_history_frames", 90),
        max_image_size=copilot_cfg.get("max_image_size", 512),
        api_url=copilot_cfg.get("api_url", ""),
        api_key=copilot_cfg.get("api_key", ""),
        api_key_env=copilot_cfg.get("api_key_env", "GROQ_API_KEY"),
        model=copilot_cfg.get("model", ""),
        ontology_version=copilot_cfg.get("ontology_version", "lap_chole_v1"),
        conservative_mode=copilot_cfg.get("conservative_mode", True),
        output_path=copilot_cfg.get("output_path", ""),
        assistant_modes=copilot_cfg.get("assistant_modes"),
    )


def build_detector_and_overlay(cfg):
    detector_backend = cfg.get("detector", {}).get("backend", "local_yolo").lower()
    tools_cfg = cfg.get("yolo_tools")
    anatomy_cfg = cfg.get("yolo_anatomy")
    legacy_cfg = cfg.get("yolo", {})
    if detector_backend == "roboflow_hosted":
        rf_cfg = cfg["roboflow_laparoscopy"]
        detector = RoboflowHostedDetector(
            model_id=rf_cfg["model_id"],
            api_url=rf_cfg["api_url"],
            api_key=rf_cfg.get("api_key", ""),
            api_key_env=rf_cfg.get("api_key_env", "ROBOFLOW_API_KEY"),
            confidence_threshold=rf_cfg["confidence_threshold"],
            detect_every_n_frames=rf_cfg["detect_every_n_frames"],
            target_classes=rf_cfg.get("target_classes"),
            class_name_map=rf_cfg.get("class_name_map"),
        )
        overlay_device = cfg["medsam2"]["device"] if cfg.get("segmenter", {}).get("backend") == "medsam2" else cfg["sam2"]["device"]
    else:
        tools_ready = tools_cfg and Path(tools_cfg["model_path"]).exists()
        anatomy_ready = anatomy_cfg and Path(anatomy_cfg["model_path"]).exists()

        if tools_ready and anatomy_ready:
            detector = CombinedYOLODetector(
                [
                    YOLODetector(
                        model_path=tools_cfg["model_path"],
                        confidence_threshold=tools_cfg["confidence_threshold"],
                        detect_every_n_frames=tools_cfg["detect_every_n_frames"],
                        device=tools_cfg["device"],
                        target_classes=tools_cfg.get("target_classes"),
                        class_name_map=tools_cfg.get("class_name_map"),
                    ),
                    YOLODetector(
                        model_path=anatomy_cfg["model_path"],
                        confidence_threshold=anatomy_cfg["confidence_threshold"],
                        detect_every_n_frames=anatomy_cfg["detect_every_n_frames"],
                        device=anatomy_cfg["device"],
                        target_classes=anatomy_cfg.get("target_classes"),
                        class_name_map=anatomy_cfg.get("class_name_map"),
                    ),
                ]
            )
            overlay_device = tools_cfg["device"]
        else:
            if tools_cfg and anatomy_cfg and not (tools_ready and anatomy_ready):
                print("[warn] Dual YOLO weights not found yet. Falling back to legacy yolo.model_path.")
            detector = YOLODetector(
                model_path=legacy_cfg["model_path"],
                confidence_threshold=legacy_cfg["confidence_threshold"],
                detect_every_n_frames=legacy_cfg["detect_every_n_frames"],
                device=legacy_cfg["device"],
                target_classes=legacy_cfg.get("target_classes"),
                class_name_map=legacy_cfg.get("class_name_map"),
            )
            overlay_device = legacy_cfg["device"]

    overlay = OverlayCompositor(
        colors=cfg["overlay"]["colors"],
        blend_alpha=cfg["overlay"]["blend_alpha"],
        glow_effect=cfg["overlay"]["glow_effect"],
        glow_radius=cfg["overlay"]["glow_radius"],
        contour_thickness=cfg["overlay"]["contour_thickness"],
        device=overlay_device,
    )
    return detector, overlay


def run_pipeline(config_path: str, headless: bool = False):
    """Launch the full Holoscan pipeline."""
    if not HAS_HOLOSCAN:
        print("[error] Holoscan SDK not available.")
        print("   This must run on Linux inside the Holoscan Docker container.")
        print("   Run: docker-compose up")
        sys.exit(1)

    app = SurgicalARApp(config_path=config_path, headless=headless)
    app.config(config_path)
    app.run()


def run_batch_evaluation(
    config_path: str,
    video_dir: str,
    video_glob: str,
    output_dir: str,
    max_frames: int,
    save_overlays: bool,
    save_masks: bool,
    prompt_dir: str | None,
    fps: int,
    width: int,
    height: int,
):
    """Run the offline evaluation workflow on a batch of staged videos."""
    cfg = load_app_config(config_path)

    assets = extract_video_batch(
        video_dir=video_dir,
        output_dir=output_dir,
        video_glob=video_glob,
        fps=fps,
        resolution=(width, height),
        max_frames=max_frames,
        save_numpy=True,
        save_frames=True,
    )

    detector, overlay = build_detector_and_overlay(cfg)
    segmenter = build_segmenter_from_config(cfg)
    vlm_guide = build_vlm_guide_from_config(cfg)
    scene_copilot = build_scene_copilot_from_config(cfg, fps=fps)

    evaluator = OfflineVideoEvaluator(
        detector=detector,
        segmenter=segmenter,
        vlm_guide=vlm_guide,
        scene_copilot=scene_copilot,
        overlay=overlay,
        output_dir=output_dir,
        max_frames=max_frames,
        save_overlays=save_overlays,
        save_masks=save_masks,
        prompt_dir=prompt_dir,
    )
    evaluator.evaluate_assets(assets)


def run_smoke_test():
    """Quick smoke test: synthetic video + format/overlay checks."""
    print("=== Smoke Test Mode ===\n")

    print("1. Generating synthetic test frames...")
    from data.convert_video import generate_synthetic_video

    generate_synthetic_video("data/converted", num_frames=30)
    print()

    print("2. Testing format utilities...")
    import numpy as np
    import torch

    from operators.format_utils import holoscan_to_torch, normalize_for_sam2, resize_tensor_gpu

    dummy = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
    if torch.cuda.is_available():
        tensor = holoscan_to_torch(dummy, "cuda:0")
        resized = resize_tensor_gpu(tensor, (1024, 1024))
        normed = normalize_for_sam2(resized)
        print(
            f"   Format utils: ok ({tensor.device}, resize {resized.shape}, norm {normed.shape})"
        )
    else:
        print("   Format utils: no GPU, skipping GPU checks")
    print()

    print("3. Testing overlay compositor...")
    if torch.cuda.is_available():
        from operators.overlay_compositor_op import OverlayCompositor

        comp = OverlayCompositor(device="cuda:0")
        frame = torch.from_numpy(dummy).to("cuda:0")
        mask = torch.zeros(480, 854, device="cuda:0")
        mask[190:290, 347:507] = 1.0
        result = comp.composite(frame, {"gallbladder": mask})
        print(f"   Overlay: ok output shape {result.shape}")
    else:
        print("   Overlay: no GPU, skipping compositor check")
    print()

    print("=== Smoke Test Complete ===")
    print("For full MedSAM2 evaluation, use --video-dir on a Linux GPU machine.")


def main():
    parser = argparse.ArgumentParser(
        description="Surgical AR Pipeline - MedSAM2 + YOLO surgical segmentation"
    )
    parser.add_argument("--config", type=str, default="config/app_config.yaml", help="Path to app_config.yaml")
    parser.add_argument("--video", type=str, default=None, help="Path to single MP4 video for the Holoscan path")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing staged local Cholec80 videos")
    parser.add_argument("--video-glob", type=str, default="video*.mp4", help="Glob used under --video-dir")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for batch ingestion/evaluation")
    parser.add_argument("--prompt-dir", type=str, default=None, help="Optional directory containing per-video seed bbox JSON files")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled frames per video during evaluation")
    parser.add_argument("--save-overlays", action="store_true", help="Save overlay video and sampled PNG frames")
    parser.add_argument("--save-masks", action="store_true", help="Save per-frame mask artifacts as NPZ files")
    parser.add_argument("--fps", type=int, default=25, help="Sampling FPS for ingestion/evaluation")
    parser.add_argument("--width", type=int, default=854, help="Frame width for ingestion/evaluation")
    parser.add_argument("--height", type=int, default=480, help="Frame height for ingestion/evaluation")
    parser.add_argument("--headless", action="store_true", help="Run Holoscan without display")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test without Holoscan")
    parser.add_argument("--benchmark", action="store_true", help="Reserved flag for future performance benchmarking")
    parser.add_argument("--frames", type=int, default=250, help="Reserved benchmark frame count")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    cfg = load_app_config(args.config)

    if args.video_dir:
        eval_cfg = cfg.get("evaluation", {})
        output_dir = args.output_dir or eval_cfg.get("processed_dir", "data/cholec80/processed")
        prompt_dir = args.prompt_dir or eval_cfg.get("prompt_dir")
        max_frames = args.max_frames if args.max_frames is not None else eval_cfg.get("default_max_frames", 600)
        run_batch_evaluation(
            config_path=args.config,
            video_dir=args.video_dir,
            video_glob=args.video_glob or eval_cfg.get("video_glob", "video*.mp4"),
            output_dir=output_dir,
            max_frames=max_frames,
            save_overlays=args.save_overlays,
            save_masks=args.save_masks,
            prompt_dir=prompt_dir,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        return

    if args.video:
        convert_video_if_needed(args.video)

    run_pipeline(args.config, headless=args.headless)


if __name__ == "__main__":
    main()
