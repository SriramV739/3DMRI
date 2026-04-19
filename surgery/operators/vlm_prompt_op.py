"""
vlm_prompt_op.py - VLM-guided anatomy prompt selection for MedSAM2.

Production-ready for Groq Llama 4 Scout (meta-llama/llama-4-scout-17b-16e-instruct).
Also works with any OpenAI-compatible vision-chat endpoint.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import requests
import torch

from operators.yolo_detection_op import Detection

try:
    import holoscan.core
    from holoscan.core import Operator, OperatorSpec

    HAS_HOLOSCAN = True
except ImportError:
    HAS_HOLOSCAN = False


@dataclass
class PromptSelection:
    """VLM anatomy-selection result for a frame or a prompt window."""

    target_labels: List[str]
    filtered_detections: List[Detection]
    rationale: str = ""
    provider: str = "rule_based"


class AnatomyVLMGuide:
    """Select anatomy prompts for MedSAM2 from a user query and detections."""

    def __init__(
        self,
        enabled: bool = False,
        provider: str = "rule_based",
        user_query: str = "",
        candidate_labels: Optional[Sequence[str]] = None,
        anatomy_aliases: Optional[Dict[str, List[str]]] = None,
        prompt_every_n_frames: int = 30,
        max_image_size: int = 512,
        api_url: str = "",
        api_key: str = "",
        api_key_env: str = "VLM_API_KEY",
        model: str = "",
    ):
        self.enabled = enabled
        self.provider = provider
        self.user_query = user_query
        self.candidate_labels = list(candidate_labels or [])
        self.anatomy_aliases = anatomy_aliases or {}
        self.prompt_every_n_frames = max(1, prompt_every_n_frames)
        self.max_image_size = max(64, max_image_size)
        self.api_url = api_url
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.model = model
        self.frame_count = 0
        self.last_selection = PromptSelection(target_labels=[], filtered_detections=[])

    def reset(self):
        self.frame_count = 0
        self.last_selection = PromptSelection(target_labels=[], filtered_detections=[])

    def set_query(self, new_query: str):
        """Update the active query and force the next prompt selection to refresh."""
        self.user_query = new_query
        self.reset()

    def get_requested_labels(self, query: Optional[str] = None) -> List[str]:
        query_text = (query if query is not None else self.user_query).lower()
        matches: List[str] = []
        for label in self.candidate_labels:
            aliases = [
                label.lower(),
                label.lower().replace("_", " "),
                *[alias.lower() for alias in self.anatomy_aliases.get(label, [])],
            ]
            if any(alias in query_text for alias in aliases):
                matches.append(label)
        return matches

    def should_refresh(self) -> bool:
        return self.frame_count % self.prompt_every_n_frames == 0

    def select_prompts(
        self,
        frame,
        detections: Optional[List[Detection]] = None,
        frame_idx: Optional[int] = None,
    ) -> PromptSelection:
        detections = list(detections or [])
        if not self.enabled:
            self.frame_count += 1
            selection = PromptSelection(
                target_labels=[],
                filtered_detections=detections,
                rationale="VLM disabled",
                provider="disabled",
            )
            self.last_selection = selection
            return selection

        refresh = self.should_refresh() or not self.last_selection.target_labels
        self.frame_count += 1

        if refresh:
            target_labels, rationale = self._infer_target_labels(frame, detections, frame_idx)
        else:
            target_labels = list(self.last_selection.target_labels)
            rationale = self.last_selection.rationale

        filtered = self._filter_detections(detections, target_labels)
        selection = PromptSelection(
            target_labels=target_labels,
            filtered_detections=filtered,
            rationale=rationale,
            provider=self.provider,
        )
        self.last_selection = selection
        return selection

    def _infer_target_labels(
        self,
        frame,
        detections: List[Detection],
        frame_idx: Optional[int],
    ) -> tuple[List[str], str]:
        explicit_matches = self._extract_query_matches()
        if explicit_matches:
            return self._resolve_explicit_query_targets(explicit_matches, detections)

        if self.provider == "openai_compatible":
            try:
                labels, rationale = self._infer_with_openai_compatible(frame, detections, frame_idx)
                if labels:
                    return labels, rationale
            except Exception as exc:
                return self._infer_with_rules(detections, fallback_reason=f"VLM fallback after error: {exc}")
        return self._infer_with_rules(detections)

    def _extract_query_matches(self) -> List[str]:
        return self.get_requested_labels()

    def _resolve_explicit_query_targets(
        self,
        requested_labels: Sequence[str],
        detections: List[Detection],
    ) -> tuple[List[str], str]:
        detected_labels = {det.class_name for det in detections}
        supported = [label for label in requested_labels if label in detected_labels]
        missing = [label for label in requested_labels if label not in detected_labels]

        if supported and missing:
            return (
                list(requested_labels),
                "Direct query match. Requested labels not supported by current detections: "
                + ", ".join(missing),
            )
        if supported:
            return supported, "Direct query match from explicit user request."
        return list(requested_labels), (
            "Explicit user request kept for VLM prompt-box localization because no detector box is available yet."
        )

    def _infer_with_rules(
        self,
        detections: List[Detection],
        fallback_reason: Optional[str] = None,
    ) -> tuple[List[str], str]:
        matches = self._extract_query_matches()

        if not matches:
            for det in detections:
                if det.class_name in self.candidate_labels and det.class_name not in matches:
                    matches.append(det.class_name)

        rationale = fallback_reason or "Rule-based query matching"
        return matches, rationale

    # Groq rate-limit retry settings
    _MAX_RETRIES = 3
    _RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry
    _GROQ_BASE64_LIMIT_BYTES = 4 * 1024 * 1024  # 4 MB

    def _infer_with_openai_compatible(
        self,
        frame,
        detections: List[Detection],
        frame_idx: Optional[int],
    ) -> tuple[List[str], str]:
        if not self.api_url or not self.api_key or not self.model:
            missing = []
            if not self.api_url:
                missing.append("api_url")
            if not self.api_key:
                missing.append(f"api_key (set env ${self.api_key_env if hasattr(self, 'api_key_env') else 'GROQ_API_KEY'})")
            if not self.model:
                missing.append("model")
            raise RuntimeError(
                f"VLM openai_compatible config incomplete — missing: {', '.join(missing)}. "
                f"For Groq, set GROQ_API_KEY and use model 'meta-llama/llama-4-scout-17b-16e-instruct'."
            )

        image_b64 = self._frame_to_base64(frame)

        # Validate base64 payload size (Groq limit: 4 MB for base64 images)
        b64_size = len(image_b64)
        if b64_size > self._GROQ_BASE64_LIMIT_BYTES:
            print(f"[warn] VLM image payload {b64_size / 1e6:.1f} MB exceeds Groq 4 MB limit; "
                  f"reduce vlm.max_image_size (currently {self.max_image_size}px)")

        detection_summary = [
            {
                "class_name": det.class_name,
                "bbox": [round(v, 1) for v in det.bbox],
                "confidence": round(det.confidence, 3),
            }
            for det in detections[:12]
        ]
        prompt = (
            "You are an expert surgical AI assistant for laparoscopic cholecystectomy. "
            "You are given a live laparoscopic video frame, a surgeon's intent query, "
            "current object-detector results, and a list of candidate labels.\n\n"
            "Your job: select ONLY the labels from the candidate list that the "
            "surgeon needs highlighted RIGHT NOW based on their query and what you see "
            "in the frame.\n\n"
            "Return strict JSON with exactly two keys:\n"
            '  "target_labels": [list of selected label strings from candidate_labels]\n'
            '  "rationale": "brief explanation of why you chose these labels"\n\n'
            "Rules:\n"
            "- Use ONLY labels from the provided candidate_labels list.\n"
            "- Labels may include anatomy or tools. Select only what is relevant to the current user instruction.\n"
            "- If a candidate label is clearly visible or referenced in the query, include it.\n"
            "- If no candidate label is relevant right now, return an empty target_labels list.\n"
            "- Be concise in your rationale."
        )
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "user_query": self.user_query,
                                    "frame_idx": frame_idx,
                                    "candidate_labels": self.candidate_labels,
                                    "detections": detection_summary,
                                }
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Retry loop with exponential backoff for Groq rate limits (HTTP 429)
        last_error: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else self._RETRY_BACKOFF_BASE * (2 ** attempt)
                    print(f"[vlm] Groq rate-limited (429), retrying in {wait:.1f}s (attempt {attempt + 1}/{self._MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.Timeout:
                last_error = TimeoutError(f"VLM request timed out after 30s (attempt {attempt + 1})")
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(self._RETRY_BACKOFF_BASE * (2 ** attempt))
                continue
        else:
            raise RuntimeError(
                f"VLM request failed after {self._MAX_RETRIES} attempts: {last_error}"
            )

        data = response.json()

        # Parse response — handle both clean JSON and markdown-wrapped JSON
        content = data["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            # Strip markdown code fences if the model wraps its output
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            print(f"[vlm] Failed to parse VLM JSON response: {exc}\nRaw content: {content[:500]}")
            return [], f"JSON parse error: {exc}"

        labels = [label for label in parsed.get("target_labels", []) if label in self.candidate_labels]
        rationale = parsed.get("rationale", "Groq Llama 4 Scout VLM")
        labels, rationale = self._postprocess_vlm_labels(labels, detections, rationale)
        return labels, rationale

    def _postprocess_vlm_labels(
        self,
        labels: Sequence[str],
        detections: List[Detection],
        rationale: str,
    ) -> tuple[List[str], str]:
        requested = self._extract_query_matches()
        detected_labels = {det.class_name for det in detections}

        if requested:
            grounded_requested = [label for label in requested if label in detected_labels]
            if grounded_requested:
                return list(requested), (
                    "Explicit query grounding overrides free-form VLM selection. " + rationale
                )
            return list(requested), (
                "Explicit query targets kept for VLM prompt-box localization despite missing detector support. "
                + rationale
            )

        grounded = [label for label in labels if label in detected_labels]
        if grounded:
            return grounded, rationale
        return [], "VLM returned labels without detector support, so no labels were selected."

    def localize_prompt_boxes(
        self,
        frame,
        requested_labels: Sequence[str],
        existing_detections: Optional[List[Detection]] = None,
        frame_idx: Optional[int] = None,
    ) -> List[Detection]:
        """Ask the VLM for approximate prompt boxes when the detector has no box.

        MedSAM2 is promptable rather than open-vocabulary. This method bridges that
        gap by letting the vision model provide coarse boxes for arbitrary requested
        labels, which MedSAM2 can then refine into masks.
        """
        labels = [str(label).strip().lower().replace(" ", "_") for label in requested_labels if str(label).strip()]
        if not labels or self.provider != "openai_compatible" or not self.api_url or not self.api_key or not self.model:
            return []

        existing_detections = list(existing_detections or [])
        frame_np = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
        height, width = frame_np.shape[:2]
        image_b64 = self._frame_to_base64(frame)
        detection_summary = [
            {
                "class_name": det.class_name,
                "bbox": [round(float(v), 1) for v in det.bbox],
                "confidence": round(float(det.confidence), 3),
            }
            for det in existing_detections[:16]
        ]
        prompt = (
            "You are localizing surgical video regions for a promptable segmentation model. "
            "Return approximate bounding boxes for the requested visible targets only. "
            "Boxes may be coarse; they will be refined by MedSAM2. "
            "If a target is not visible, omit it. Return strict JSON only."
        )
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 450,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "frame_idx": frame_idx,
                                    "image_width": width,
                                    "image_height": height,
                                    "requested_labels": labels,
                                    "existing_detections": detection_summary,
                                    "required_json_schema": {
                                        "boxes": [
                                            {
                                                "label": "requested label",
                                                "bbox": [0, 0, width, height],
                                                "confidence": 0.0,
                                                "rationale": "brief visual reason",
                                            }
                                        ]
                                    },
                                },
                                ensure_ascii=True,
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
        }
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = "\n".join(line for line in content.splitlines() if not line.strip().startswith("```"))
        parsed = json.loads(content)

        label_set = set(labels)
        localized: List[Detection] = []
        for item in parsed.get("boxes", []) or []:
            label = str(item.get("label", "")).strip().lower().replace(" ", "_")
            bbox = item.get("bbox") or []
            if label not in label_set or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1 = max(0.0, min(x1, width - 1.0))
            y1 = max(0.0, min(y1, height - 1.0))
            x2 = max(0.0, min(x2, width - 1.0))
            y2 = max(0.0, min(y2, height - 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            localized.append(
                Detection(
                    class_name=label,
                    bbox=[x1, y1, x2, y2],
                    confidence=float(item.get("confidence", 0.35) or 0.35),
                    source_model="vlm_prompt_box",
                )
            )
        return localized

    def _frame_to_base64(self, frame) -> str:
        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().cpu().numpy()
        else:
            frame_np = np.asarray(frame)

        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_image_size / float(max(h, w)))
        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("Failed to encode frame for VLM request")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _filter_detections(self, detections: List[Detection], target_labels: Sequence[str]) -> List[Detection]:
        if not target_labels:
            return []
        allowed = set(target_labels)
        return [det for det in detections if det.class_name in allowed]


if HAS_HOLOSCAN:

    class VLMAnatomyPromptOp(Operator):
        """Holoscan operator that filters detections using a VLM anatomy query."""

        def setup(self, spec: OperatorSpec):
            spec.input("rgb_tensor")
            spec.input("bboxes")
            spec.output("bboxes")
            spec.output("target_labels")
            spec.param("enabled", default_value=False)
            spec.param("provider", default_value="rule_based")
            spec.param("user_query", default_value="")
            spec.param("candidate_labels", default_value=None)
            spec.param("anatomy_aliases", default_value=None)
            spec.param("prompt_every_n_frames", default_value=30)
            spec.param("max_image_size", default_value=512)
            spec.param("api_url", default_value="")
            spec.param("api_key", default_value="")
            spec.param("api_key_env", default_value="VLM_API_KEY")
            spec.param("model", default_value="")

        def start(self):
            self.guide = AnatomyVLMGuide(
                enabled=self.enabled,
                provider=self.provider,
                user_query=self.user_query,
                candidate_labels=self.candidate_labels,
                anatomy_aliases=self.anatomy_aliases,
                prompt_every_n_frames=self.prompt_every_n_frames,
                max_image_size=self.max_image_size,
                api_url=self.api_url,
                api_key=self.api_key,
                api_key_env=self.api_key_env,
                model=self.model,
            )
            print(f"[ok] VLM prompt guide ready ({self.provider}, enabled={self.enabled})")

        def set_query(self, new_query: str):
            """Dynamically update the target query from the live terminal thread."""
            self.user_query = new_query
            if hasattr(self, "guide"):
                self.guide.set_query(new_query)

        def compute(self, op_input, op_output, context):
            from operators.format_utils import holoscan_to_torch

            frame = holoscan_to_torch(op_input.receive("rgb_tensor"))
            detections = op_input.receive("bboxes")
            selection = self.guide.select_prompts(frame, detections)
            op_output.emit(selection.filtered_detections, "bboxes")
            op_output.emit(selection.target_labels, "target_labels")

        def stop(self):
            del self.guide
