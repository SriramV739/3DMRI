"""
app.py - Main Holoscan application for the Surgical AR Pipeline.

Wires together the processing graph:
  VideoReplay -> Format -> YOLO -> Segmenter -> Overlay -> Display
"""

from pathlib import Path
from typing import Any, Dict

import yaml

try:
    import holoscan.core
    from holoscan.core import Application
    from holoscan.operators import FormatConverterOp, HolovizOp, VideoStreamReplayerOp

    HAS_HOLOSCAN = True
except ImportError:
    HAS_HOLOSCAN = False
    Application = object

if HAS_HOLOSCAN:
    from operators.medsam2_inference_op import MedSAM2InferenceOp
    from operators.overlay_compositor_op import OverlayCompositorOp
    from operators.roboflow_detection_op import RoboflowDetectionOp
    from operators.sam2_inference_op import SAM2InferenceOp
    from operators.scene_copilot_op import SurgicalSceneCopilotOp
    from operators.vlm_prompt_op import VLMAnatomyPromptOp
    from operators.yolo_detection_op import YOLODetectionOp


def load_app_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and resolve repo-relative file paths."""
    config_file = Path(config_path).resolve()
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for section, key in (
        ("replayer", "directory"),
        ("yolo", "model_path"),
        ("yolo_tools", "model_path"),
        ("yolo_anatomy", "model_path"),
        ("roboflow_laparoscopy", "dataset_dir"),
        ("sam2", "checkpoint"),
        ("medsam2", "checkpoint"),
        ("evaluation", "raw_video_dir"),
        ("evaluation", "prompt_dir"),
        ("evaluation", "processed_dir"),
        ("scene_copilot", "output_path"),
    ):
        raw_value = config.get(section, {}).get(key)
        if isinstance(raw_value, str):
            config[section][key] = str((config_file.parent / raw_value).resolve())

    return config


class SurgicalARApp(Application if HAS_HOLOSCAN else object):
    """Holoscan application: YOLO + segmentation backend + AR overlay."""

    def __init__(self, config_path: str, headless: bool = False):
        if HAS_HOLOSCAN:
            super().__init__()
        self.config_path = config_path
        self.headless = headless
        self.app_config = load_app_config(config_path)

    def compose(self):
        if not HAS_HOLOSCAN:
            raise RuntimeError(
                "Holoscan SDK not available. This must run on Linux with the Holoscan container."
            )

        cfg = self.app_config
        backend = cfg.get("segmenter", {}).get("backend", "medsam2").lower()
        detector_backend = cfg.get("detector", {}).get("backend", "local_yolo").lower()
        vlm_enabled = cfg.get("vlm", {}).get("enabled", False)
        scene_copilot_enabled = cfg.get("scene_copilot", {}).get("enabled", False)

        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=cfg["replayer"]["directory"],
            basename=cfg["replayer"]["basename"],
            realtime=cfg["replayer"]["realtime"],
            repeat=cfg["replayer"]["repeat"],
            frame_rate=cfg["replayer"]["frame_rate"],
        )

        formatter = FormatConverterOp(self, name="formatter", output_type="rgb888")

        if detector_backend == "roboflow_hosted":
            rf_cfg = cfg["roboflow_laparoscopy"]
            yolo = RoboflowDetectionOp(
                self,
                name="roboflow_detector",
                model_id=rf_cfg["model_id"],
                api_url=rf_cfg["api_url"],
                api_key=rf_cfg.get("api_key", ""),
                api_key_env=rf_cfg.get("api_key_env", "ROBOFLOW_API_KEY"),
                confidence_threshold=rf_cfg["confidence_threshold"],
                detect_every_n_frames=rf_cfg["detect_every_n_frames"],
                target_classes=rf_cfg.get("target_classes"),
                class_name_map=rf_cfg.get("class_name_map"),
            )
        else:
            legacy_yolo_cfg = cfg.get("yolo", {})
            tools_cfg = cfg.get("yolo_tools")
            anatomy_cfg = cfg.get("yolo_anatomy")

            secondary_model_path = ""
            secondary_confidence_threshold = 0.4
            secondary_target_classes = None
            secondary_class_name_map = None
            model_path = legacy_yolo_cfg.get("model_path", "yolov8n.pt")
            confidence_threshold = legacy_yolo_cfg.get("confidence_threshold", 0.4)
            detect_every_n_frames = legacy_yolo_cfg.get("detect_every_n_frames", 15)
            device = legacy_yolo_cfg.get("device", "cuda:0")
            target_classes = legacy_yolo_cfg.get("target_classes")
            class_name_map = legacy_yolo_cfg.get("class_name_map")

            tools_ready = tools_cfg and Path(tools_cfg["model_path"]).exists()
            anatomy_ready = anatomy_cfg and Path(anatomy_cfg["model_path"]).exists()

            if tools_ready and anatomy_ready:
                model_path = tools_cfg["model_path"]
                confidence_threshold = tools_cfg["confidence_threshold"]
                detect_every_n_frames = tools_cfg["detect_every_n_frames"]
                device = tools_cfg["device"]
                target_classes = tools_cfg.get("target_classes")
                class_name_map = tools_cfg.get("class_name_map")
                secondary_model_path = anatomy_cfg["model_path"]
                secondary_confidence_threshold = anatomy_cfg["confidence_threshold"]
                secondary_target_classes = anatomy_cfg.get("target_classes")
                secondary_class_name_map = anatomy_cfg.get("class_name_map")
            elif tools_cfg and anatomy_cfg:
                print("[warn] Dual YOLO weights not found. Falling back to legacy yolo.model_path.")

            yolo = YOLODetectionOp(
                self,
                name="yolo_detector",
                model_path=model_path,
                secondary_model_path=secondary_model_path,
                confidence_threshold=confidence_threshold,
                secondary_confidence_threshold=secondary_confidence_threshold,
                detect_every_n_frames=detect_every_n_frames,
                device=device,
                target_classes=target_classes,
                secondary_target_classes=secondary_target_classes,
                class_name_map=class_name_map,
                secondary_class_name_map=secondary_class_name_map,
            )

        if backend == "medsam2":
            segmenter = MedSAM2InferenceOp(
                self,
                name="medsam2_segmenter",
                checkpoint=cfg["medsam2"]["checkpoint"],
                model_cfg=cfg["medsam2"]["model_cfg"],
                device=cfg["medsam2"]["device"],
                dtype=cfg["medsam2"]["dtype"],
                max_objects=cfg["medsam2"]["max_objects"],
                use_temporal_memory=cfg["medsam2"]["use_temporal_memory"],
                prompt_classes=cfg.get("segmenter", {}).get("prompt_classes"),
            )
        elif backend == "sam2":
            segmenter = SAM2InferenceOp(
                self,
                name="sam2_segmenter",
                checkpoint=cfg["sam2"]["checkpoint"],
                model_cfg=cfg["sam2"]["model_cfg"],
                device=cfg["sam2"]["device"],
                dtype=cfg["sam2"]["dtype"],
                vos_optimized=cfg["sam2"]["vos_optimized"],
                max_objects=cfg["sam2"]["max_objects"],
                prompt_classes=cfg.get("segmenter", {}).get("prompt_classes"),
            )
        else:
            raise ValueError(f"Unsupported segmenter backend: {backend}")

        overlay = OverlayCompositorOp(self, name="overlay")

        self.add_flow(replayer, formatter)
        self.add_flow(formatter, yolo, {("output", "rgb_tensor")})
        segmenter_input_source = yolo

        if vlm_enabled:
            vlm_cfg = cfg["vlm"]
            vlm = VLMAnatomyPromptOp(
                self,
                name="vlm_prompt_guide",
                enabled=vlm_cfg["enabled"],
                provider=vlm_cfg["provider"],
                user_query=vlm_cfg["user_query"],
                candidate_labels=vlm_cfg["candidate_labels"],
                anatomy_aliases=vlm_cfg["anatomy_aliases"],
                prompt_every_n_frames=vlm_cfg["prompt_every_n_frames"],
                max_image_size=vlm_cfg["max_image_size"],
                api_url=vlm_cfg.get("api_url", ""),
                api_key=vlm_cfg.get("api_key", ""),
                api_key_env=vlm_cfg.get("api_key_env", "VLM_API_KEY"),
                model=vlm_cfg.get("model", ""),
            )
            self.vlm_op = vlm
            self.add_flow(formatter, vlm, {("output", "rgb_tensor")})
            self.add_flow(yolo, vlm, {("bboxes", "bboxes")})
            segmenter_input_source = vlm

        self.add_flow(segmenter_input_source, segmenter, {("bboxes", "bboxes")})
        self.add_flow(formatter, segmenter, {("output", "rgb_tensor")})
        self.add_flow(segmenter, overlay, {("mask_tensor", "mask_tensor")})
        self.add_flow(segmenter, overlay, {("mask_labels", "mask_labels")})
        self.add_flow(formatter, overlay, {("output", "rgb_tensor")})

        if scene_copilot_enabled:
            copilot_cfg = cfg["scene_copilot"]
            refresh_seconds = copilot_cfg.get("refresh_interval_seconds", 2.0)
            refresh_every_n_frames = max(
                1,
                int(round(refresh_seconds * cfg["replayer"]["frame_rate"])),
            )
            scene_copilot = SurgicalSceneCopilotOp(
                self,
                name="scene_copilot",
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
            self.copilot_op = scene_copilot
            self.add_flow(formatter, scene_copilot, {("output", "rgb_tensor")})
            self.add_flow(yolo, scene_copilot, {("bboxes", "bboxes")})
            self.add_flow(segmenter, scene_copilot, {("mask_labels", "mask_labels")})

        if not self.headless:
            holoviz = HolovizOp(
                self,
                name="holoviz",
                width=cfg["holoviz"]["width"],
                height=cfg["holoviz"]["height"],
                fullscreen=cfg["holoviz"]["fullscreen"],
            )
            self.add_flow(overlay, holoviz, {("composited_frame", "receivers")})

        print("[ok] Pipeline graph assembled:")
        if vlm_enabled and scene_copilot_enabled:
            print(
                f"   Replay -> Format -> Detector -> VLM -> {backend} -> Overlay -> Display"
                " (+ parallel SceneCopilot branch)"
            )
        elif vlm_enabled:
            print(f"   Replay -> Format -> Detector -> VLM -> {backend} -> Overlay -> Display")
        elif scene_copilot_enabled:
            print(
                f"   Replay -> Format -> Detector -> {backend} -> Overlay -> Display"
                " (+ parallel SceneCopilot branch)"
            )
        else:
            print(f"   Replay -> Format -> Detector -> {backend} -> Overlay -> Display")
