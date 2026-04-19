"""
yolo_detection_op.py - YOLO detection helpers for surgical tools and anatomy.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

try:
    import holoscan.core
    from holoscan.core import Operator, OperatorSpec

    HAS_HOLOSCAN = True
except ImportError:
    HAS_HOLOSCAN = False

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


@dataclass
class Detection:
    """A single YOLO detection result."""

    class_name: str
    bbox: List[float]
    confidence: float
    class_id: int = 0
    source_model: str = "yolo"


def _resolve_model_path(model_path: str) -> str:
    """Resolve a model path from common repo-relative locations."""
    candidate = Path(model_path)
    if candidate.exists():
        return str(candidate)

    repo_root = Path(__file__).resolve().parent.parent
    fallback = repo_root / "models" / candidate.name
    if fallback.exists():
        return str(fallback)

    return model_path


class YOLODetector:
    """Standalone YOLO detection engine."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        detect_every_n_frames: int = 15,
        device: str = "cuda:0",
        target_classes: Optional[List[str]] = None,
        class_name_map: Optional[dict] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.detect_every_n_frames = max(1, detect_every_n_frames)
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
        self.target_classes = target_classes
        self.class_name_map = class_name_map or {}
        self.frame_count = 0
        self.last_detections: List[Detection] = []
        self.model_path = _resolve_model_path(model_path)
        self.model_name = Path(self.model_path).stem

        if not HAS_ULTRALYTICS:
            raise ImportError("ultralytics package required: pip install ultralytics")

        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        self.class_names = self.model.names

    def should_detect(self) -> bool:
        return self.frame_count % self.detect_every_n_frames == 0

    def detect(self, frame: torch.Tensor) -> List[Detection]:
        run_detection = self.should_detect()
        self.frame_count += 1

        if not run_detection:
            return []

        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().cpu().numpy()
        else:
            frame_np = np.asarray(frame)

        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

        results = self.model.predict(
            source=frame_np,
            verbose=False,
            conf=self.confidence_threshold,
            device=self.device,
        )

        detections: List[Detection] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                bbox = [float(v) for v in boxes.xyxy[i].tolist()]
                conf = float(boxes.conf[i].item())
                cls_id = int(boxes.cls[i].item())
                if isinstance(self.class_names, dict):
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                else:
                    cls_name = self.class_names[cls_id]
                cls_name = self.class_name_map.get(cls_name, cls_name)

                if self.target_classes is not None and cls_name not in self.target_classes:
                    continue

                detections.append(
                    Detection(
                        class_name=cls_name,
                        bbox=bbox,
                        confidence=conf,
                        class_id=cls_id,
                        source_model=self.model_name,
                    )
                )

        self.last_detections = detections
        return detections

    def reset(self):
        self.frame_count = 0
        self.last_detections = []


class CombinedYOLODetector:
    """Merge multiple YOLO detectors behind one detector API."""

    def __init__(self, detectors: Iterable[YOLODetector]):
        self.detectors = list(detectors)
        if not self.detectors:
            raise ValueError("CombinedYOLODetector requires at least one detector")

        self.device = self.detectors[0].device
        self.detect_every_n_frames = self.detectors[0].detect_every_n_frames
        self.last_detections: List[Detection] = []

    def should_detect(self) -> bool:
        return self.detectors[0].should_detect()

    def detect(self, frame: torch.Tensor) -> List[Detection]:
        detections: List[Detection] = []
        for detector in self.detectors:
            detections.extend(detector.detect(frame))
        detections.sort(key=lambda det: det.confidence, reverse=True)
        self.last_detections = detections
        return detections

    def reset(self):
        for detector in self.detectors:
            detector.reset()
        self.last_detections = []


if HAS_HOLOSCAN:

    class YOLODetectionOp(Operator):
        """Holoscan operator wrapping one or two YOLO checkpoints."""

        def setup(self, spec: OperatorSpec):
            spec.input("rgb_tensor")
            spec.output("bboxes")
            spec.param("model_path", default_value="yolov8n.pt")
            spec.param("secondary_model_path", default_value="")
            spec.param("confidence_threshold", default_value=0.4)
            spec.param("secondary_confidence_threshold", default_value=0.4)
            spec.param("detect_every_n_frames", default_value=15)
            spec.param("device", default_value="cuda:0")
            spec.param("target_classes", default_value=None)
            spec.param("secondary_target_classes", default_value=None)
            spec.param("class_name_map", default_value=None)
            spec.param("secondary_class_name_map", default_value=None)

        def start(self):
            primary = YOLODetector(
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold,
                detect_every_n_frames=self.detect_every_n_frames,
                device=self.device,
                target_classes=self.target_classes,
                class_name_map=self.class_name_map,
            )
            if self.secondary_model_path:
                secondary = YOLODetector(
                    model_path=self.secondary_model_path,
                    confidence_threshold=self.secondary_confidence_threshold,
                    detect_every_n_frames=self.detect_every_n_frames,
                    device=self.device,
                    target_classes=self.secondary_target_classes,
                    class_name_map=self.secondary_class_name_map,
                )
                self.detector = CombinedYOLODetector([primary, secondary])
                print(f"[ok] Dual YOLO detectors loaded on {self.device}")
            else:
                self.detector = primary
                print(f"[ok] YOLO detector loaded on {self.device}")

        def compute(self, op_input, op_output, context):
            from operators.format_utils import holoscan_to_torch

            tensor = holoscan_to_torch(op_input.receive("rgb_tensor"))
            detections = self.detector.detect(tensor)
            op_output.emit(detections, "bboxes")

        def stop(self):
            del self.detector
            torch.cuda.empty_cache()
