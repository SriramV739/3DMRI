"""
live_streaming_agent.py - Live video agent with an asynchronous terminal for real-time prompting.
Falls back to native OpenCV if NVIDIA Holoscan is not installed.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Add the repo root to sys.path so we can import from app
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import HAS_HOLOSCAN, SurgicalARApp, load_app_config
from operators.format_utils import holoscan_to_torch

# We can reuse the builder functions from the interactive script to save boilerplate
from scripts.interactive_vlm import build_detector, build_segmenter, build_scene_copilot
from operators.vlm_prompt_op import AnatomyVLMGuide
from operators.overlay_compositor_op import OverlayCompositor


class OpenCVFallbackPipeline:
    """A pure Python/OpenCV implementation of the live AR pipeline."""
    def __init__(self, cfg):
        print("[info] Initializing Native OpenCV Fallback Pipeline...")
        self.cfg = cfg
        self.detector = build_detector(cfg)

        vlm_cfg = cfg.get("vlm", {})
        self.vlm_guide = AnatomyVLMGuide(
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

        self.scene_copilot = build_scene_copilot(cfg)
        self.segmenter = build_segmenter(cfg)
        self.device = getattr(self.segmenter, "device", "cpu")

        self.overlay = OverlayCompositor(
            colors=cfg["overlay"]["colors"],
            blend_alpha=cfg["overlay"]["blend_alpha"],
            glow_effect=cfg["overlay"]["glow_effect"],
            glow_radius=cfg["overlay"]["glow_radius"],
            contour_thickness=cfg["overlay"]["contour_thickness"],
            device=self.device,
        )

        # Resolve the video path
        video_dir = cfg["replayer"]["directory"]
        basename = cfg["replayer"]["basename"]
        # The frames are stored in directory/parent_of_basename/frames
        base_path = Path(video_dir) / basename
        self.frames_dir = base_path.parent / "frames"

        self.running = False
        print("[ok] Native OpenCV Pipeline Initialized.")

    def run(self):
        if not self.frames_dir.exists():
            print(f"[error] Frames directory not found: {self.frames_dir}")
            return

        frames = sorted(self.frames_dir.glob("*.jpg"))
        if not frames:
            print(f"[error] No frames found in {self.frames_dir}")
            return

        self.running = True
        frame_idx = 0
        total_frames = len(frames)

        if getattr(self.segmenter, "use_temporal_memory", False):
            print(f"[info] Preparing video temporal memory for {self.frames_dir}")
            self.segmenter.prepare_video(str(self.frames_dir), propagation_window=self.cfg["yolo"].get("detect_every_n_frames", 15))

        # We will loop continuously until stopped or window is closed
        while self.running:
            frame_path = frames[frame_idx % total_frames]

            # Read frame and convert BGR -> RGB for inference
            bgr_img = cv2.imread(str(frame_path))
            if bgr_img is None:
                break
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            frame_t = torch.from_numpy(rgb_img).to(self.device)

            # Pipeline Execution
            detections = self.detector.detect(frame_t)

            if self.vlm_guide is not None:
                selection = self.vlm_guide.select_prompts(frame_t, detections, frame_idx=frame_idx)
                filtered_detections = selection.filtered_detections
            else:
                filtered_detections = []

            if hasattr(self.segmenter, "segment_frame"):
                masks = self.segmenter.segment_frame(frame_t, filtered_detections, frame_idx=frame_idx)
            else:
                masks = {}

            if self.scene_copilot is not None:
                self.scene_copilot.analyze(frame_t, detections, masks, frame_idx=frame_idx)

            composited = self.overlay.composite(frame_t, masks)

            # Convert back to BGR for display
            composited_rgb = composited[:, :, :3].cpu().numpy().astype(np.uint8)
            composited_bgr = cv2.cvtColor(composited_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Live Surgical AR (Native Fallback)", composited_bgr)

            # Wait for 1ms and check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                self.running = False
                break

            frame_idx += 1

            # Simulated frame rate pacing (approx 25fps = 40ms per frame)
            # time.sleep(0.04)

        cv2.destroyAllWindows()


def run_terminal_listener(app_or_pipeline):
    """
    Background thread that listens for terminal input and injects it directly
    into the running pipeline without pausing the video stream.
    """
    time.sleep(2.0)
    print("\n" + "="*80)
    print("LIVE STREAMING AGENT READY")
    print("The surgical video is now streaming live.")
    print("Type a command below to instantly update the AR overlay or ask a scene question.")
    print("Examples: 'Highlight the cystic duct', 'What tool is that?', 'Is the CVS safe?'")
    print("Type 'exit' or press Ctrl+C to quit.")
    print("="*80 + "\n")

    while True:
        try:
            cmd = input("Live Surgical Command > ").strip()
            if not cmd:
                continue
            if cmd.lower() in ("exit", "quit", "q"):
                print("[info] Exiting terminal listener...")
                if hasattr(app_or_pipeline, "running"):
                    app_or_pipeline.running = False
                break

            print(f"\n[agent] Processing live command: '{cmd}'")

            # Handle Holoscan app
            if HAS_HOLOSCAN and isinstance(app_or_pipeline, SurgicalARApp):
                if hasattr(app_or_pipeline, "vlm_op"):
                    app_or_pipeline.vlm_op.set_query(cmd)
                if hasattr(app_or_pipeline, "copilot_op"):
                    app_or_pipeline.copilot_op.set_query(cmd)
            # Handle OpenCV Fallback
            else:
                if hasattr(app_or_pipeline, "vlm_guide") and app_or_pipeline.vlm_guide is not None:
                    app_or_pipeline.vlm_guide.set_query(cmd)
                if hasattr(app_or_pipeline, "scene_copilot") and app_or_pipeline.scene_copilot is not None:
                    app_or_pipeline.scene_copilot.set_query(cmd)

            print("[agent] Command injected! The overlay/copilot will update on the next frame tick.\n")

        except EOFError:
            break
        except Exception as e:
            print(f"[error] Failed to inject command: {e}")


def main():
    parser = argparse.ArgumentParser(description="Live Streaming Surgical Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="config/app_config.yaml",
        help="Path to the application configuration file.",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(REPO_ROOT, config_path)

    print(f"[info] Loading configuration from: {config_path}")
    cfg = load_app_config(config_path)

    if HAS_HOLOSCAN:
        print("[info] Holoscan detected. Initializing NVIDIA Holoscan pipeline...")
        app = SurgicalARApp(cfg, headless=False)
        listener_thread = threading.Thread(target=run_terminal_listener, args=(app,), daemon=True)
        listener_thread.start()
        print("[info] Starting Holoscan execution graph...")
        app.run()
    else:
        print("[warn] NVIDIA Holoscan is not installed or importable.")
        print("[info] Falling back to native OpenCV playback loop...")
        pipeline = OpenCVFallbackPipeline(cfg)
        listener_thread = threading.Thread(target=run_terminal_listener, args=(pipeline,), daemon=True)
        listener_thread.start()
        print("[info] Starting OpenCV playback loop...")
        pipeline.run()

    print("[info] Live streaming agent finished.")


if __name__ == "__main__":
    main()
