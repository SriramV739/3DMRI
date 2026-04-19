import argparse
import glob
import os
import sys
from pathlib import Path
import json

import cv2
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import load_app_config
from operators.overlay_compositor_op import OverlayCompositor
from operators.medsam2_inference_op import MedSAM2Segmenter
from operators.vlm_prompt_op import AnatomyVLMGuide
from operators.yolo_detection_op import YOLODetector, Detection
from operators.roboflow_detection_op import RoboflowHostedDetector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/app_config.yaml")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()

    cfg = load_app_config(args.config)

    print("Initializing pipeline...")
    # 1. Setup Detector
    detector_backend = cfg.get("detector", {}).get("backend", "local_yolo").lower()
    if detector_backend == "roboflow_hosted":
        rf_cfg = cfg["roboflow_laparoscopy"]
        detector = RoboflowHostedDetector(
            model_id=rf_cfg["model_id"],
            api_url=rf_cfg["api_url"],
            api_key=rf_cfg.get("api_key", ""),
            api_key_env=rf_cfg.get("api_key_env", "ROBOFLOW_API_KEY"),
            confidence_threshold=rf_cfg["confidence_threshold"],
            detect_every_n_frames=1, # Detect on every image
            target_classes=rf_cfg.get("target_classes"),
            class_name_map=rf_cfg.get("class_name_map"),
        )
    else:
        legacy_cfg = cfg.get("yolo", {})
        detector = YOLODetector(
            model_path=legacy_cfg["model_path"],
            confidence_threshold=legacy_cfg["confidence_threshold"],
            detect_every_n_frames=1,
            device=legacy_cfg.get("device", "cuda:0"),
            target_classes=legacy_cfg.get("target_classes"),
            class_name_map=legacy_cfg.get("class_name_map"),
        )

    # 2. Setup VLM
    vlm_cfg = cfg.get("vlm", {})
    vlm_guide = AnatomyVLMGuide(
        enabled=True,
        provider=vlm_cfg.get("provider", "openai_compatible"),
        user_query=args.prompt,
        candidate_labels=vlm_cfg.get("candidate_labels"),
        anatomy_aliases=vlm_cfg.get("anatomy_aliases"),
        prompt_every_n_frames=1, # Always prompt on static images
        max_image_size=vlm_cfg.get("max_image_size", 512),
        api_url=vlm_cfg.get("api_url", ""),
        api_key=vlm_cfg.get("api_key", ""),
        api_key_env=vlm_cfg.get("api_key_env", "VLM_API_KEY"),
        model=vlm_cfg.get("model", ""),
    )

    # 3. Setup MedSAM2
    seg_cfg = cfg["medsam2"]
    segmenter = MedSAM2Segmenter(
        checkpoint=seg_cfg["checkpoint"],
        model_cfg=seg_cfg["model_cfg"],
        device=seg_cfg["device"],
        dtype=seg_cfg["dtype"],
        max_objects=seg_cfg["max_objects"],
        use_temporal_memory=False, # No temporal propagation for standalone images
    )

    # 4. Setup Overlay
    overlay = OverlayCompositor(
        colors=cfg["overlay"]["colors"],
        blend_alpha=cfg["overlay"]["blend_alpha"],
        glow_effect=cfg["overlay"]["glow_effect"],
        glow_radius=cfg["overlay"]["glow_radius"],
        contour_thickness=cfg["overlay"]["contour_thickness"],
        device=seg_cfg["device"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(args.image_dir, "*.png")))

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    processed = 0
    for img_path in image_paths:
        if processed >= args.max_images:
            break

        print(f"\n[{processed+1}/{args.max_images}] Processing {os.path.basename(img_path)}...")

        frame_bgr = cv2.imread(img_path)
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_t = torch.from_numpy(frame_rgb).to(seg_cfg["device"])

        # Force a refresh of the VLM since it's a new image
        vlm_guide.reset()
        detector.reset()

        # 1. Detect
        detections = detector.detect(frame_t)

        # 2. Query VLM
        selection = vlm_guide.select_prompts(frame_t, detections, frame_idx=processed)

        # 3. Segment
        masks = segmenter.segment_frame(frame_t, selection.filtered_detections, frame_idx=processed)

        # 4. Overlay
        composited = overlay.composite(frame_t, masks)

        out_name = Path(img_path).stem
        out_bgr = cv2.cvtColor(composited.cpu().numpy(), cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(os.path.join(args.output_dir, f"{out_name}_highlighted.png"), out_bgr)

        with open(os.path.join(args.output_dir, f"{out_name}_response.json"), "w") as f:
            json.dump({
                "prompt": args.prompt,
                "rationale": selection.rationale,
                "selected_labels": selection.target_labels,
                "filtered_boxes": [{"class": d.class_name, "bbox": d.bbox} for d in selection.filtered_detections]
            }, f, indent=2)

        print(f" -> Rationale: {selection.rationale}")
        print(f" -> Selected targets: {selection.target_labels}")
        print(f" -> Output saved to {args.output_dir}")

        processed += 1

if __name__ == "__main__":
    main()
