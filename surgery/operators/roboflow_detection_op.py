"""
roboflow_detection_op.py - Hosted Roboflow detector backend for POC use.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch

from operators.yolo_detection_op import Detection

try:
    import holoscan.core
    from holoscan.core import Operator, OperatorSpec

    HAS_HOLOSCAN = True
except ImportError:
    HAS_HOLOSCAN = False

try:
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient

    HAS_ROBOFLOW = True
except ImportError:
    HAS_ROBOFLOW = False


class RoboflowHostedDetector:
    """Run hosted Roboflow inference and adapt predictions to the local detector API."""

    def __init__(
        self,
        model_id: str,
        api_url: str = "https://serverless.roboflow.com",
        api_key: Optional[str] = None,
        api_key_env: str = "ROBOFLOW_API_KEY",
        confidence_threshold: float = 0.35,
        detect_every_n_frames: int = 15,
        target_classes: Optional[List[str]] = None,
        class_name_map: Optional[dict] = None,
    ):
        if not HAS_ROBOFLOW:
            raise ImportError("inference-sdk is required: pip install inference-sdk")

        self.model_id = model_id
        self.api_url = api_url
        # Treat an empty config value the same as "not provided" so env vars work.
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.confidence_threshold = confidence_threshold
        self.detect_every_n_frames = max(1, detect_every_n_frames)
        self.target_classes = target_classes
        self.class_name_map = class_name_map or {}
        self.frame_count = 0
        self.last_detections: List[Detection] = []
        self.client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
        self.client.configure(
            InferenceConfiguration(confidence_threshold=self.confidence_threshold)
        )

    def should_detect(self) -> bool:
        return self.frame_count % self.detect_every_n_frames == 0

    def _normalize_class_name(self, class_name: str) -> str:
        return self.class_name_map.get(class_name, class_name)

    def _to_bgr(self, frame) -> np.ndarray:
        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().cpu().numpy()
        else:
            frame_np = np.asarray(frame)

        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    def _parse_result(self, result: dict) -> List[Detection]:
        detections: List[Detection] = []
        for item in result.get("predictions", []):
            class_name = self._normalize_class_name(item["class"])
            if self.target_classes is not None and class_name not in self.target_classes:
                continue

            x_center = float(item["x"])
            y_center = float(item["y"])
            width = float(item["width"])
            height = float(item["height"])
            x1 = x_center - width / 2.0
            y1 = y_center - height / 2.0
            x2 = x_center + width / 2.0
            y2 = y_center + height / 2.0
            detections.append(
                Detection(
                    class_name=class_name,
                    bbox=[x1, y1, x2, y2],
                    confidence=float(item.get("confidence", 0.0)),
                    class_id=int(item.get("class_id", 0)),
                    source_model=f"roboflow:{self.model_id}",
                )
            )
        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def detect(self, frame) -> List[Detection]:
        run_detection = self.should_detect()
        self.frame_count += 1
        if not run_detection:
            return []

        image_bgr = self._to_bgr(frame)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            temp_path = handle.name
        try:
            cv2.imwrite(temp_path, image_bgr)
            result = self.client.infer(temp_path, model_id=self.model_id)
            detections = self._parse_result(result)
            self.last_detections = detections
            return detections
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def reset(self):
        self.frame_count = 0
        self.last_detections = []


if HAS_HOLOSCAN:

    class RoboflowDetectionOp(Operator):
        """Holoscan wrapper around the hosted Roboflow detector."""

        def setup(self, spec: OperatorSpec):
            spec.input("rgb_tensor")
            spec.output("bboxes")
            spec.param("model_id")
            spec.param("api_url", default_value="https://serverless.roboflow.com")
            spec.param("api_key", default_value="")
            spec.param("api_key_env", default_value="ROBOFLOW_API_KEY")
            spec.param("confidence_threshold", default_value=0.35)
            spec.param("detect_every_n_frames", default_value=15)
            spec.param("target_classes", default_value=None)
            spec.param("class_name_map", default_value=None)

        def start(self):
            self.detector = RoboflowHostedDetector(
                model_id=self.model_id,
                api_url=self.api_url,
                api_key=self.api_key,
                api_key_env=self.api_key_env,
                confidence_threshold=self.confidence_threshold,
                detect_every_n_frames=self.detect_every_n_frames,
                target_classes=self.target_classes,
                class_name_map=self.class_name_map,
            )
            print(f"[ok] Roboflow detector ready for model {self.model_id}")

        def compute(self, op_input, op_output, context):
            from operators.format_utils import holoscan_to_torch

            frame = holoscan_to_torch(op_input.receive("rgb_tensor"))
            detections = self.detector.detect(frame)
            op_output.emit(detections, "bboxes")

        def stop(self):
            del self.detector
