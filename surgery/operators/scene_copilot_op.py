"""
scene_copilot_op.py - Conservative lap-chole scene copilot with deterministic assist modes.
"""

from __future__ import annotations

import json
import os
import base64
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

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


ANATOMY_LABELS = [
    "gallbladder",
    "cystic_duct",
    "cystic_artery",
    "cystic_plate",
    "hepatocystic_triangle",
    "liver",
]

DEFAULT_WORKFLOW_PHASES = [
    "exposure_and_retraction",
    "calot_dissection",
    "clipping_or_cutting_preparation",
    "uncertain",
]

QUERY_ALIASES = {
    "gallbladder": ["gallbladder", "gall bladder", "galbladder", "gb"],
    "cystic_duct": ["cystic duct", "duct"],
    "cystic_artery": ["cystic artery", "artery"],
    "cystic_plate": ["cystic plate", "plate"],
    "hepatocystic_triangle": ["calot", "calots triangle", "hepatocystic triangle"],
    "liver": ["liver", "hepatic"],
    "grasper": ["grasper", "forceps"],
    "bipolar": ["bipolar"],
    "hook": ["hook", "cautery"],
    "scissors": ["scissors"],
    "clipper": ["clipper", "clips"],
    "irrigator": ["irrigator", "suction"],
    "specimen_bag": ["specimen bag", "bag"],
}

ASSISTANT_MODE_DEFAULTS: Dict[str, object] = {
    "enabled_modes": [
        "overlay_update",
        "safety_check",
        "structure_report",
        "anatomy_verification",
        "workflow_status",
        "instrument_activity",
        "general_scene_question",
    ],
    "default_uncertainty_policy": "fail_closed",
    "show_mode_badge": True,
    "allow_general_scene_mode": True,
    "structure_report_requires_target": True,
    "safety_check_requires_grounded_evidence": True,
}


@dataclass
class SceneAnalysis:
    assistant_mode: str
    mode_payload: Dict[str, object]
    scene_summary: str
    surgeon_response: str
    reasoning_summary: str
    visible_structures: List[str]
    visible_tools: List[str]
    workflow_phase: str
    critical_view_status: str
    observed_risks: List[str]
    uncertainties: List[str]
    recommended_attention_targets: List[str]
    qa_response: str
    confidence: float
    frame_idx: int
    timestamp: float = 0.0
    provider: str = "rule_based"
    refreshed: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SceneObservation:
    frame_idx: int
    detections: List[str] = field(default_factory=list)
    mask_labels: List[str] = field(default_factory=list)


class SceneStateStore:
    """Maintain a rolling window of observations to smooth scene analysis."""

    def __init__(self, max_history_frames: int = 90):
        self.max_history_frames = max(1, max_history_frames)
        self.history: Deque[SceneObservation] = deque(maxlen=self.max_history_frames)
        self.last_analysis: Optional[SceneAnalysis] = None

    def reset(self):
        self.history.clear()
        self.last_analysis = None

    def add_observation(
        self,
        frame_idx: int,
        detections: Sequence[Detection],
        mask_labels: Sequence[str],
    ):
        self.history.append(
            SceneObservation(
                frame_idx=frame_idx,
                detections=[det.class_name for det in detections],
                mask_labels=list(mask_labels),
            )
        )

    def stable_labels(self, min_ratio: float = 0.35) -> List[str]:
        if not self.history:
            return []
        counter: Counter[str] = Counter()
        for item in self.history:
            for label in set(item.detections + item.mask_labels):
                counter[label] += 1
        threshold = max(1, int(round(len(self.history) * min_ratio)))
        return sorted(label for label, count in counter.items() if count >= threshold)

    def latest_labels(self) -> List[str]:
        if not self.history:
            return []
        latest = self.history[-1]
        return sorted(set(latest.detections + latest.mask_labels))


class SurgicalSceneCopilot:
    """Structured scene-analysis copilot for laparoscopic cholecystectomy."""

    def __init__(
        self,
        enabled: bool = True,
        provider: str = "rule_based",
        user_query: str = "",
        refresh_every_n_frames: int = 30,
        max_history_frames: int = 90,
        max_image_size: int = 512,
        api_url: str = "",
        api_key: str = "",
        api_key_env: str = "GROQ_API_KEY",
        model: str = "",
        ontology_version: str = "lap_chole_v1",
        conservative_mode: bool = True,
        output_path: str = "",
        assistant_modes: Optional[Dict[str, object]] = None,
    ):
        self.enabled = enabled
        self.provider = provider
        self.user_query = user_query
        self.refresh_every_n_frames = max(1, refresh_every_n_frames)
        self.max_image_size = max(64, max_image_size)
        self.api_url = api_url
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.api_key_env = api_key_env
        self.model = model
        self.ontology_version = ontology_version
        self.conservative_mode = conservative_mode
        self.output_path = output_path
        self.assistant_modes = self._normalize_assistant_modes(assistant_modes)
        self.frame_count = 0
        self.state = SceneStateStore(max_history_frames=max_history_frames)

    def reset(self):
        self.frame_count = 0
        self.state.reset()

    def set_query(self, new_query: str):
        self.user_query = new_query

    def should_refresh(self) -> bool:
        return self.frame_count % self.refresh_every_n_frames == 0

    def _normalize_assistant_modes(self, assistant_modes: Optional[Dict[str, object]]) -> Dict[str, object]:
        cfg = dict(ASSISTANT_MODE_DEFAULTS)
        if assistant_modes:
            cfg.update(assistant_modes)
        enabled = cfg.get("enabled_modes", ASSISTANT_MODE_DEFAULTS["enabled_modes"])
        cfg["enabled_modes"] = list(enabled) if isinstance(enabled, list) else list(ASSISTANT_MODE_DEFAULTS["enabled_modes"])
        return cfg

    def _mode_enabled(self, mode: str) -> bool:
        return mode in set(self.assistant_modes.get("enabled_modes", []))

    def analyze(
        self,
        frame,
        detections: Optional[Sequence[Detection]] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        frame_idx: int = 0,
        user_query: Optional[str] = None,
        conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> SceneAnalysis:
        detections = list(detections or [])
        mask_labels = list((masks or {}).keys())
        if user_query is not None:
            self.set_query(user_query)

        self.state.add_observation(frame_idx, detections, mask_labels)

        if not self.enabled:
            analysis = self._infer_with_rules(detections, mask_labels, frame_idx, provider="disabled")
            self.state.last_analysis = analysis
            self.frame_count += 1
            return analysis

        refresh = self.should_refresh() or self.state.last_analysis is None
        self.frame_count += 1
        if refresh:
            if self.provider == "openai_compatible":
                try:
                    analysis = self._infer_with_openai(
                        frame,
                        detections,
                        mask_labels,
                        frame_idx,
                        conversation_history=conversation_history,
                    )
                except Exception as exc:
                    analysis = self._infer_with_rules(
                        detections,
                        mask_labels,
                        frame_idx,
                        provider="rule_based",
                        fallback_reason=f"VLM fallback after error: {exc}",
                    )
            else:
                analysis = self._infer_with_rules(detections, mask_labels, frame_idx, provider=self.provider)
            self.state.last_analysis = analysis
            self._write_output(analysis)
            return analysis

        cached = self.state.last_analysis
        assert cached is not None
        return SceneAnalysis(
            assistant_mode=cached.assistant_mode,
            mode_payload=dict(cached.mode_payload),
            scene_summary=cached.scene_summary,
            surgeon_response=cached.surgeon_response,
            reasoning_summary=cached.reasoning_summary,
            visible_structures=list(cached.visible_structures),
            visible_tools=list(cached.visible_tools),
            workflow_phase=cached.workflow_phase,
            critical_view_status=cached.critical_view_status,
            observed_risks=list(cached.observed_risks),
            uncertainties=list(cached.uncertainties),
            recommended_attention_targets=list(cached.recommended_attention_targets),
            qa_response=cached.qa_response,
            confidence=cached.confidence,
            frame_idx=cached.frame_idx,
            timestamp=cached.timestamp,
            provider=cached.provider,
            refreshed=False,
        )

    def _write_output(self, analysis: SceneAnalysis):
        if not self.output_path:
            return
        output = Path(self.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(analysis.to_dict(), handle, indent=2)

    def _infer_with_rules(
        self,
        detections: Sequence[Detection],
        mask_labels: Sequence[str],
        frame_idx: int,
        provider: str = "rule_based",
        fallback_reason: str = "",
    ) -> SceneAnalysis:
        evidence = self._collect_grounded_evidence(detections, mask_labels)
        mode = self._resolve_assistant_mode(self.user_query)
        return self._build_analysis_from_mode(
            assistant_mode=mode,
            evidence=evidence,
            frame_idx=frame_idx,
            provider=provider,
            fallback_reason=fallback_reason,
        )

    def _collect_grounded_evidence(
        self,
        detections: Sequence[Detection],
        mask_labels: Sequence[str],
    ) -> Dict[str, object]:
        stable_labels = set(self.state.stable_labels())
        current_labels = set(self.state.latest_labels())
        combined = stable_labels | current_labels | set(mask_labels)
        visible_structures = sorted(label for label in combined if label in ANATOMY_LABELS)
        visible_tools = sorted(label for label in combined if label not in ANATOMY_LABELS)
        requested_labels = self._extract_query_focus_labels(self.user_query)
        workflow_phase = self._infer_workflow_phase(visible_structures, visible_tools)
        critical_view_status = self._infer_critical_view_status(visible_structures)
        observed_risks = self._infer_risks(visible_structures, visible_tools)
        uncertainties = self._infer_uncertainties(visible_structures, visible_tools)
        recommended_attention_targets = self._infer_attention_targets(
            visible_structures,
            visible_tools,
            self.user_query,
        )
        confidence = self._infer_confidence(visible_structures, visible_tools, uncertainties)
        return {
            "visible_structures": visible_structures,
            "visible_tools": visible_tools,
            "requested_labels": requested_labels,
            "mask_labels": list(mask_labels),
            "workflow_phase": workflow_phase,
            "critical_view_status": critical_view_status,
            "observed_risks": observed_risks,
            "uncertainties": uncertainties,
            "recommended_attention_targets": recommended_attention_targets,
            "confidence": confidence,
        }

    def _resolve_assistant_mode(self, user_query: str) -> str:
        mode = self._classify_query(user_query)
        if not self._mode_enabled(mode):
            if self.assistant_modes.get("allow_general_scene_mode", True) and self._mode_enabled("general_scene_question"):
                return "general_scene_question"
            return "uncertainty"
        if mode == "general_scene_question" and not self.assistant_modes.get("allow_general_scene_mode", True):
            return "uncertainty"
        return mode

    def _build_analysis_from_mode(
        self,
        assistant_mode: str,
        evidence: Dict[str, object],
        frame_idx: int,
        provider: str,
        fallback_reason: str = "",
    ) -> SceneAnalysis:
        if assistant_mode == "overlay_update":
            mode_payload = self._build_overlay_update_payload(evidence)
        elif assistant_mode == "safety_check":
            mode_payload = self._build_safety_check_payload(evidence)
        elif assistant_mode == "structure_report":
            mode_payload = self._build_structure_report_payload(evidence)
        elif assistant_mode == "anatomy_verification":
            mode_payload = self._build_anatomy_verification_payload(evidence)
        elif assistant_mode == "workflow_status":
            mode_payload = self._build_workflow_status_payload(evidence)
        elif assistant_mode == "instrument_activity":
            mode_payload = self._build_instrument_activity_payload(evidence)
        elif assistant_mode == "general_scene_question":
            mode_payload = self._build_general_scene_payload(evidence)
        else:
            mode_payload = self._build_uncertainty_payload(
                reason="The current request could not be routed to an enabled grounded assistant mode.",
                evidence=evidence,
            )
            assistant_mode = "uncertainty"

        if self.assistant_modes.get("default_uncertainty_policy") == "fail_closed":
            assistant_mode, mode_payload = self._apply_uncertainty_guardrail(assistant_mode, mode_payload, evidence)

        scene_summary = self._build_scene_summary(
            evidence["visible_structures"],
            evidence["visible_tools"],
            evidence["workflow_phase"],
            evidence["critical_view_status"],
        )
        surgeon_response = self._render_mode_payload(assistant_mode, mode_payload)
        reasoning_summary = self._build_mode_reasoning_summary(
            assistant_mode=assistant_mode,
            mode_payload=mode_payload,
            evidence=evidence,
            fallback_reason=fallback_reason,
        )
        return SceneAnalysis(
            assistant_mode=assistant_mode,
            mode_payload=mode_payload,
            scene_summary=scene_summary,
            surgeon_response=surgeon_response,
            reasoning_summary=reasoning_summary,
            visible_structures=list(evidence["visible_structures"]),
            visible_tools=list(evidence["visible_tools"]),
            workflow_phase=str(evidence["workflow_phase"]),
            critical_view_status=str(evidence["critical_view_status"]),
            observed_risks=list(evidence["observed_risks"]),
            uncertainties=list(evidence["uncertainties"]),
            recommended_attention_targets=list(evidence["recommended_attention_targets"]),
            qa_response=surgeon_response,
            confidence=float(mode_payload.get("confidence", evidence["confidence"])),
            frame_idx=frame_idx,
            timestamp=float(frame_idx),
            provider=provider,
            refreshed=True,
        )

    def _infer_workflow_phase(self, visible_structures: Sequence[str], visible_tools: Sequence[str]) -> str:
        structure_set = set(visible_structures)
        tool_set = set(visible_tools)
        if {"clipper", "scissors"} & tool_set and {"cystic_duct", "cystic_artery"} & structure_set:
            return "clipping_or_cutting_preparation"
        if {"hook", "bipolar"} & tool_set and {"hepatocystic_triangle", "gallbladder"} & structure_set:
            return "calot_dissection"
        if {"grasper"} & tool_set and {"gallbladder", "liver"} & structure_set:
            return "exposure_and_retraction"
        return "uncertain"

    def _infer_critical_view_status(self, visible_structures: Sequence[str]) -> str:
        structure_set = set(visible_structures)
        if {"cystic_duct", "cystic_artery", "hepatocystic_triangle"} <= structure_set:
            return "partially_visualized_but_not_confirmed"
        if {"cystic_duct", "hepatocystic_triangle"} <= structure_set:
            return "developing_but_incomplete"
        return "not_yet_clearly_visualized"

    def _infer_risks(self, visible_structures: Sequence[str], visible_tools: Sequence[str]) -> List[str]:
        risks: List[str] = []
        structure_set = set(visible_structures)
        tool_set = set(visible_tools)
        if {"hook", "bipolar"} & tool_set and "cystic_duct" not in structure_set:
            risks.append("Energy instrument visible while the cystic duct is not confidently identified.")
        if {"clipper", "scissors"} & tool_set and "cystic_artery" not in structure_set:
            risks.append("Cutting or clipping workflow may be starting before both key tubular structures are clearly seen.")
        if "hepatocystic_triangle" not in structure_set:
            risks.append("Calot's triangle is not consistently visualized in the current window.")
        return risks

    def _infer_uncertainties(self, visible_structures: Sequence[str], visible_tools: Sequence[str]) -> List[str]:
        uncertainties: List[str] = []
        if "cystic_duct" not in visible_structures:
            uncertainties.append("The cystic duct is not confidently visible.")
        if "cystic_artery" not in visible_structures:
            uncertainties.append("The cystic artery is not confidently visible.")
        if "hepatocystic_triangle" not in visible_structures:
            uncertainties.append("The hepatocystic triangle is not yet clearly delineated.")
        if not visible_tools:
            uncertainties.append("No instrument class is stable enough to summarize confidently.")
        return uncertainties

    def _infer_attention_targets(
        self,
        visible_structures: Sequence[str],
        visible_tools: Sequence[str],
        user_query: str,
    ) -> List[str]:
        query = (user_query or "").lower()
        targets: List[str] = []
        preferred = [
            ("gallbladder", ["gallbladder", "gb"]),
            ("cystic_duct", ["cystic duct", "duct"]),
            ("cystic_artery", ["cystic artery", "artery"]),
            ("hepatocystic_triangle", ["calot", "hepatocystic triangle"]),
            ("liver", ["liver"]),
            ("grasper", ["grasper", "tool", "forceps"]),
            ("hook", ["hook", "cautery"]),
            ("clipper", ["clipper", "clip"]),
        ]
        for label, aliases in preferred:
            if any(alias in query for alias in aliases):
                targets.append(label)
        if not targets:
            for label in ("cystic_duct", "cystic_artery", "hepatocystic_triangle", "gallbladder"):
                if label not in visible_structures:
                    targets.append(label)
        if not targets and visible_tools:
            targets.extend(list(visible_tools)[:2])
        deduped: List[str] = []
        for item in targets:
            if item not in deduped:
                deduped.append(item)
        return deduped[:4]

    def _infer_confidence(
        self,
        visible_structures: Sequence[str],
        visible_tools: Sequence[str],
        uncertainties: Sequence[str],
    ) -> float:
        evidence = len(visible_structures) * 0.15 + len(visible_tools) * 0.05
        penalty = len(uncertainties) * 0.12
        return float(max(0.1, min(0.95, 0.55 + evidence - penalty)))

    def _build_scene_summary(
        self,
        visible_structures: Sequence[str],
        visible_tools: Sequence[str],
        workflow_phase: str,
        critical_view_status: str,
    ) -> str:
        structures = ", ".join(visible_structures) if visible_structures else "no key anatomy"
        tools = ", ".join(visible_tools) if visible_tools else "no stable tools"
        return (
            f"Current lap-chole scene suggests {workflow_phase.replace('_', ' ')}. "
            f"Visible anatomy: {structures}. Visible tools: {tools}. "
            f"Critical view status is {critical_view_status.replace('_', ' ')}."
        )

    def _humanize_label(self, label: str) -> str:
        return label.replace("_", " ")

    def _join_labels(self, labels: Sequence[str]) -> str:
        if not labels:
            return "none"
        return ", ".join(self._humanize_label(label) for label in labels)

    def _is_overlay_update_question(self, user_query: str) -> bool:
        query = (user_query or "").lower()
        return any(token in query for token in ("highlight", "segment", "outline", "mark", "overlay", "show"))

    def _is_tool_question(self, user_query: str) -> bool:
        query = (user_query or "").lower()
        return any(token in query for token in ("tool", "instrument", "grasper", "hook", "clipper", "scissors", "bipolar"))

    def _is_safety_check_question(self, user_query: str) -> bool:
        query = (user_query or "").lower()
        return any(
            token in query
            for token in ("critical view", "cvs", "safe", "safety", "risk", "careful", "clipping readiness", "exposed enough")
        )

    def _is_phase_question(self, user_query: str) -> bool:
        query = (user_query or "").lower()
        return any(token in query for token in ("what is happening", "what phase", "what stage", "what is going on", "dissection"))

    def _is_anatomy_verification_question(self, user_query: str) -> bool:
        query = (user_query or "").lower()
        verification_tokens = ("is this", "can you confirm", "is ", "do you see", "visible", "confirm")
        return bool(self._extract_query_focus_labels(user_query)) and any(token in query for token in verification_tokens)

    def _is_structure_report_question(self, user_query: str) -> bool:
        query = (user_query or "").lower()
        report_tokens = (
            "full report",
            "report on",
            "diagnosis",
            "diagnostic",
            "full diagnosis",
            "give a report",
        )
        return bool(self._extract_query_focus_labels(user_query)) and any(token in query for token in report_tokens)

    def _extract_query_focus_labels(self, user_query: str) -> List[str]:
        query = (user_query or "").lower()
        labels: List[str] = []
        for label, aliases in QUERY_ALIASES.items():
            if any(alias in query for alias in aliases):
                labels.append(label)
        return labels

    def _classify_query(self, user_query: str) -> str:
        if self._is_overlay_update_question(user_query):
            return "overlay_update"
        if self._is_structure_report_question(user_query):
            return "structure_report"
        if self._is_safety_check_question(user_query):
            return "safety_check"
        if self._is_anatomy_verification_question(user_query):
            return "anatomy_verification"
        if self._is_tool_question(user_query):
            return "instrument_activity"
        if self._is_phase_question(user_query):
            return "workflow_status"
        if self._extract_query_focus_labels(user_query):
            return "anatomy_verification"
        return "general_scene_question"

    def _describe_tool_activity(self, visible_tools: Sequence[str], workflow_phase: str) -> str:
        if not visible_tools:
            return "I cannot confidently identify a stable instrument class in this frame."

        primary_tool = visible_tools[0]
        pretty_tool = self._humanize_label(primary_tool)
        if primary_tool == "grasper":
            action = "providing traction or retraction to improve exposure"
        elif primary_tool in {"hook", "bipolar"}:
            action = "being used for fine dissection or energy-assisted tissue work"
        elif primary_tool in {"clipper", "scissors"}:
            action = "being used in a clipping or division phase"
        else:
            action = f"participating in the current {workflow_phase.replace('_', ' ')} step"
        return f"The most clearly supported instrument impression is a {pretty_tool} that appears to be {action}."

    def _format_bulleted_sentence(self, heading: str, body: str) -> str:
        return f"**{heading}:** {body}"

    def _build_overlay_update_payload(self, evidence: Dict[str, object]) -> Dict[str, object]:
        updated_targets = list(evidence["mask_labels"])
        requested_targets = list(evidence["requested_labels"])
        status = "updated" if updated_targets else "no_supported_targets"
        return {
            "status": status,
            "requested_targets": requested_targets,
            "updated_targets": updated_targets,
            "confidence": float(evidence["confidence"]) if updated_targets else 0.2,
        }

    def _build_safety_check_payload(self, evidence: Dict[str, object]) -> Dict[str, object]:
        missing_evidence: List[str] = []
        visible_structures = list(evidence["visible_structures"])
        if "cystic_duct" not in visible_structures:
            missing_evidence.append("cystic duct not confidently identified")
        if "cystic_artery" not in visible_structures:
            missing_evidence.append("cystic artery not confidently identified")
        if "hepatocystic_triangle" not in visible_structures:
            missing_evidence.append("hepatocystic triangle not clearly delineated")

        risk_flags = list(evidence["observed_risks"])
        if self.assistant_modes.get("safety_check_requires_grounded_evidence", True) and missing_evidence:
            safety_status = "indeterminate_attention_required"
        elif risk_flags:
            safety_status = "attention_required"
        else:
            safety_status = "no_immediate_flag_but_incomplete"

        return {
            "safety_status": safety_status,
            "risk_flags": risk_flags,
            "missing_evidence": missing_evidence,
            "recommended_attention_targets": list(evidence["recommended_attention_targets"]),
            "confidence": min(float(evidence["confidence"]), 0.65),
        }

    def _build_structure_report_payload(self, evidence: Dict[str, object]) -> Dict[str, object]:
        requested = [label for label in evidence["requested_labels"] if label in ANATOMY_LABELS]
        confirmed = [label for label in requested if label in evidence["visible_structures"]]
        not_confirmed = [label for label in requested if label not in evidence["visible_structures"]]
        if confirmed and not not_confirmed:
            visibility_status = "supported"
            confidence = 0.8
        elif confirmed:
            visibility_status = "partially_supported"
            confidence = 0.55
        elif requested:
            visibility_status = "indeterminate"
            confidence = 0.25
        else:
            visibility_status = "no_target"
            confidence = 0.15
        grounded_findings = [f"{self._humanize_label(label)} is supported in the grounded evidence." for label in confirmed]
        indeterminate_findings = [
            f"{self._humanize_label(label)} is not sufficiently grounded in the current evidence."
            for label in not_confirmed
        ]
        diagnostic_limits = [
            "A full diagnosis cannot be made from a single still frame.",
            "Only requested structure visibility is reported in this mode.",
        ]
        return {
            "requested_structure": requested[0] if len(requested) == 1 else None,
            "requested_structures": requested,
            "visibility_status": visibility_status,
            "grounded_findings": grounded_findings,
            "indeterminate_findings": indeterminate_findings,
            "diagnostic_limits": diagnostic_limits,
            "confidence": confidence,
        }

    def _build_anatomy_verification_payload(self, evidence: Dict[str, object]) -> Dict[str, object]:
        requested = [label for label in evidence["requested_labels"] if label in ANATOMY_LABELS]
        confirmed = [label for label in requested if label in evidence["visible_structures"]]
        not_confirmed = [label for label in requested if label not in evidence["visible_structures"]]
        if confirmed and not not_confirmed:
            verification_status = "supported"
            confidence = 0.8
        elif confirmed:
            verification_status = "partially_supported"
            confidence = 0.5
        else:
            verification_status = "indeterminate"
            confidence = 0.25
        supporting_evidence = [f"Grounded structure label present: {self._humanize_label(label)}." for label in confirmed]
        limitations = [f"{self._humanize_label(label)} is not grounded in the current frame evidence." for label in not_confirmed]
        return {
            "requested_structure": requested[0] if len(requested) == 1 else None,
            "requested_structures": requested,
            "verification_status": verification_status,
            "supporting_evidence": supporting_evidence,
            "limitations": limitations or list(evidence["uncertainties"]),
            "confidence": confidence,
        }

    def _build_workflow_status_payload(self, evidence: Dict[str, object]) -> Dict[str, object]:
        return {
            "workflow_phase": evidence["workflow_phase"],
            "supporting_findings": [
                f"Visible anatomy: {self._join_labels(evidence['visible_structures'])}.",
                f"Visible instruments: {self._join_labels(evidence['visible_tools'])}.",
            ],
            "safety_context": f"Critical-view status: {str(evidence['critical_view_status']).replace('_', ' ')}.",
            "confidence": min(float(evidence["confidence"]), 0.75),
        }

    def _build_instrument_activity_payload(self, evidence: Dict[str, object]) -> Dict[str, object]:
        visible_tools = list(evidence["visible_tools"])
        instrument_label = visible_tools[0] if visible_tools else "indeterminate"
        limitations = [
            "This activity assessment is grounded to stable instrument labels and coarse workflow context only.",
            "Adjacent anatomy is not confirmed here unless separately requested and grounded.",
        ]
        if not visible_tools:
            limitations.insert(0, "No instrument label is stable enough for a confident activity call in this frame.")
        return {
            "instrument_label": instrument_label,
            "activity_assessment": self._describe_tool_activity(visible_tools, str(evidence["workflow_phase"])),
            "supporting_context": f"Visible anatomy context: {self._join_labels(evidence['visible_structures'])}.",
            "limitations": limitations,
            "confidence": min(float(evidence["confidence"]), 0.7),
        }

    def _build_general_scene_payload(self, evidence: Dict[str, object]) -> Dict[str, object]:
        return {
            "summary": self._build_scene_summary(
                evidence["visible_structures"],
                evidence["visible_tools"],
                evidence["workflow_phase"],
                evidence["critical_view_status"],
            ),
            "grounded_findings": [
                f"Visible anatomy: {self._join_labels(evidence['visible_structures'])}.",
                f"Visible instruments: {self._join_labels(evidence['visible_tools'])}.",
            ],
            "uncertainty_summary": list(evidence["uncertainties"]),
            "confidence": float(evidence["confidence"]),
        }

    def _build_uncertainty_payload(self, reason: str, evidence: Dict[str, object]) -> Dict[str, object]:
        return {
            "reason": reason,
            "requested_targets": list(evidence["requested_labels"]),
            "limitations": list(evidence["uncertainties"]),
            "confidence": 0.2,
        }

    def _apply_uncertainty_guardrail(
        self,
        assistant_mode: str,
        mode_payload: Dict[str, object],
        evidence: Dict[str, object],
    ) -> tuple[str, Dict[str, object]]:
        if assistant_mode == "structure_report" and self.assistant_modes.get("structure_report_requires_target", True):
            if not mode_payload.get("requested_structures"):
                return "uncertainty", self._build_uncertainty_payload(
                    "Structure report mode requires an explicit anatomy target.",
                    evidence,
                )
        if assistant_mode == "anatomy_verification" and not mode_payload.get("requested_structures"):
            return "uncertainty", self._build_uncertainty_payload(
                "Anatomy verification requires a requested anatomy target.",
                evidence,
            )
        if assistant_mode == "safety_check" and self.assistant_modes.get("safety_check_requires_grounded_evidence", True):
            if mode_payload.get("missing_evidence"):
                return "uncertainty", self._build_uncertainty_payload(
                    "Safety check cannot be concluded confidently because required grounded evidence is incomplete.",
                    evidence,
                )
        return assistant_mode, mode_payload

    def _render_mode_payload(self, assistant_mode: str, mode_payload: Dict[str, object]) -> str:
        if assistant_mode == "overlay_update":
            if mode_payload.get("status") == "updated":
                return (
                    f"**Mode:** Overlay Update\n\n"
                    f"**Assessment:** Updated the overlay for {self._join_labels(mode_payload.get('updated_targets', []))}."
                )
            return (
                f"**Mode:** Overlay Update\n\n"
                f"**Assessment:** No supported overlay targets were grounded for this request."
            )
        if assistant_mode == "safety_check":
            risk_flags = mode_payload.get("risk_flags") or ["No immediate grounded risk flag was triggered."]
            missing = mode_payload.get("missing_evidence") or ["No additional missing evidence recorded."]
            return (
                f"**Mode:** Safety Check\n\n"
                f"**Assessment:** {str(mode_payload.get('safety_status', 'indeterminate')).replace('_', ' ')}.\n\n"
                f"**Risk Flags:** {' '.join(risk_flags)}\n\n"
                f"**Missing Evidence:** {' '.join(missing)}\n\n"
                f"**Limits:** This safety check is constrained to grounded detector and mask evidence."
            )
        if assistant_mode == "structure_report":
            requested = mode_payload.get("requested_structures") or []
            grounded = mode_payload.get("grounded_findings") or ["No grounded findings available."]
            indeterminate = mode_payload.get("indeterminate_findings") or ["No indeterminate findings recorded."]
            limits = mode_payload.get("diagnostic_limits") or []
            return (
                f"**Mode:** Structure Report\n\n"
                f"**Requested Structure:** {self._join_labels(requested)}\n\n"
                f"**Visibility:** {str(mode_payload.get('visibility_status', 'indeterminate')).replace('_', ' ')}\n\n"
                f"**Grounded Findings:** {' '.join(grounded)}\n\n"
                f"**Diagnostic Limits:** {' '.join(indeterminate + limits)}"
            )
        if assistant_mode == "anatomy_verification":
            requested = mode_payload.get("requested_structures") or []
            supporting = mode_payload.get("supporting_evidence") or ["No supporting evidence available."]
            limits = mode_payload.get("limitations") or ["No additional limits recorded."]
            return (
                f"**Mode:** Anatomy Verification\n\n"
                f"**Verification:** {str(mode_payload.get('verification_status', 'indeterminate')).replace('_', ' ')} for {self._join_labels(requested)}.\n\n"
                f"**Supporting Evidence:** {' '.join(supporting)}\n\n"
                f"**Limits:** {' '.join(limits)}"
            )
        if assistant_mode == "workflow_status":
            findings = mode_payload.get("supporting_findings") or []
            return (
                f"**Mode:** Workflow Status\n\n"
                f"**Operative Impression:** {str(mode_payload.get('workflow_phase', 'uncertain')).replace('_', ' ')}.\n\n"
                f"**Supporting Findings:** {' '.join(findings)}\n\n"
                f"**Safety Context:** {mode_payload.get('safety_context', 'Not available.')}"
            )
        if assistant_mode == "instrument_activity":
            return (
                f"**Mode:** Instrument Activity\n\n"
                f"**Instrument Impression:** {self._humanize_label(str(mode_payload.get('instrument_label', 'indeterminate')))}.\n\n"
                f"**Observed Activity:** {mode_payload.get('activity_assessment', 'No grounded activity assessment available.')}\n\n"
                f"**Limits:** {' '.join(mode_payload.get('limitations', []))}"
            )
        if assistant_mode == "general_scene_question":
            return (
                f"**Mode:** General Scene Question\n\n"
                f"**Assessment:** {mode_payload.get('summary', 'No grounded summary available.')}\n\n"
                f"**Grounded Findings:** {' '.join(mode_payload.get('grounded_findings', []))}\n\n"
                f"**Limits:** {' '.join(mode_payload.get('uncertainty_summary', []))}"
            )
        return (
            f"**Mode:** Uncertainty / Escalation\n\n"
            f"**Assessment:** {mode_payload.get('reason', 'The frame is insufficient for a confident grounded answer.')}\n\n"
            f"**Limits:** {' '.join(mode_payload.get('limitations', []))}"
        )

    def _build_mode_reasoning_summary(
        self,
        assistant_mode: str,
        mode_payload: Dict[str, object],
        evidence: Dict[str, object],
        fallback_reason: str,
    ) -> str:
        parts = [
            f"Assistant mode: {assistant_mode}.",
            f"Visible structures: {self._join_labels(evidence['visible_structures'])}.",
            f"Visible tools: {self._join_labels(evidence['visible_tools'])}.",
            f"Requested labels: {self._join_labels(evidence['requested_labels'])}.",
            f"Critical-view status: {str(evidence['critical_view_status']).replace('_', ' ')}.",
        ]
        if fallback_reason:
            parts.append(f"Fallback note: {fallback_reason}")
        if "confidence" in mode_payload:
            parts.append(f"Mode confidence: {float(mode_payload['confidence']):.2f}.")
        return " ".join(parts)

    def _infer_with_openai(
        self,
        frame,
        detections: Sequence[Detection],
        mask_labels: Sequence[str],
        frame_idx: int,
        conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> SceneAnalysis:
        if not self.api_url or not self.api_key or not self.model:
            raise RuntimeError("Scene copilot VLM config is incomplete.")

        evidence = self._collect_grounded_evidence(detections, mask_labels)
        assistant_mode = self._resolve_assistant_mode(self.user_query)
        image_b64 = self._frame_to_base64(frame)
        detection_summary = [
            {
                "class_name": det.class_name,
                "bbox": [round(float(value), 1) for value in det.bbox],
                "confidence": round(float(det.confidence), 3),
            }
            for det in detections[:12]
        ]
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "max_tokens": 700,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a conservative intraoperative AI copilot for laparoscopic cholecystectomy. "
                        "Answer the surgical team in 2-3 clear sentences using the image and grounded detector/mask evidence. "
                        "Do not provide autonomous surgical instructions or claim certainty beyond the evidence. "
                        "Also produce a formal clinical_log_note suitable for an AI-generated draft surgery log."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "user_query": self.user_query,
                                    "assistant_mode": assistant_mode,
                                    "frame_idx": frame_idx,
                                    "detections": detection_summary,
                                    "mask_labels": list(mask_labels),
                                    "rolling_evidence": evidence,
                                    "conversation_history": list(conversation_history or [])[-8:],
                                    "required_json_schema": {
                                        "surgeon_response": "2-3 sentence response to the surgical team",
                                        "clinical_log_note": "formal documentation-style note for the surgery log",
                                        "scene_summary": "one sentence summary",
                                        "observed_risks": ["risk strings"],
                                        "uncertainties": ["uncertainty strings"],
                                        "recommended_attention_targets": [
                                            "visible labels or short snake_case region names worth temporary highlighting"
                                        ],
                                        "confidence": 0.0,
                                    },
                                },
                                ensure_ascii=True,
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
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = "\n".join(line for line in content.splitlines() if not line.strip().startswith("```"))
        parsed = json.loads(content)

        visible_structures = list(evidence["visible_structures"])
        visible_tools = list(evidence["visible_tools"])
        scene_summary = parsed.get("scene_summary") or self._build_scene_summary(
            visible_structures,
            visible_tools,
            str(evidence["workflow_phase"]),
            str(evidence["critical_view_status"]),
        )
        surgeon_response = parsed.get("surgeon_response") or scene_summary
        clinical_log_note = parsed.get("clinical_log_note") or self._build_mode_reasoning_summary(
            assistant_mode,
            {"confidence": parsed.get("confidence", evidence["confidence"])},
            evidence,
            "",
        )

        analysis = SceneAnalysis(
            assistant_mode=assistant_mode,
            mode_payload={
                "provider": "openai_compatible",
                "clinical_log_note": clinical_log_note,
                "confidence": float(parsed.get("confidence", evidence["confidence"])),
            },
            scene_summary=str(scene_summary),
            surgeon_response=str(surgeon_response),
            reasoning_summary=str(clinical_log_note),
            visible_structures=visible_structures,
            visible_tools=visible_tools,
            workflow_phase=str(evidence["workflow_phase"]),
            critical_view_status=str(evidence["critical_view_status"]),
            observed_risks=list(parsed.get("observed_risks") or evidence["observed_risks"]),
            uncertainties=list(parsed.get("uncertainties") or evidence["uncertainties"]),
            recommended_attention_targets=list(
                parsed.get("recommended_attention_targets") or evidence["recommended_attention_targets"]
            ),
            qa_response=str(surgeon_response),
            confidence=float(parsed.get("confidence", evidence["confidence"])),
            frame_idx=frame_idx,
            timestamp=float(frame_idx),
            provider="openai_compatible",
            refreshed=True,
        )
        return analysis

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
            raise RuntimeError("Failed to encode scene copilot frame")
        return base64.b64encode(encoded.tobytes()).decode("ascii")


if HAS_HOLOSCAN:

    class SurgicalSceneCopilotOp(Operator):
        """Holoscan sink operator that writes the latest scene analysis to a side-channel file."""

        def setup(self, spec: OperatorSpec):
            spec.input("rgb_tensor")
            spec.input("bboxes")
            spec.input("mask_labels")
            spec.param("enabled", default_value=True)
            spec.param("provider", default_value="rule_based")
            spec.param("user_query", default_value="")
            spec.param("refresh_every_n_frames", default_value=30)
            spec.param("max_history_frames", default_value=90)
            spec.param("max_image_size", default_value=512)
            spec.param("api_url", default_value="")
            spec.param("api_key", default_value="")
            spec.param("api_key_env", default_value="GROQ_API_KEY")
            spec.param("model", default_value="")
            spec.param("ontology_version", default_value="lap_chole_v1")
            spec.param("conservative_mode", default_value=True)
            spec.param("output_path", default_value="")
            spec.param("assistant_modes", default_value=dict(ASSISTANT_MODE_DEFAULTS))

        def start(self):
            self.frame_idx = 0
            self.copilot = SurgicalSceneCopilot(
                enabled=self.enabled,
                provider=self.provider,
                user_query=self.user_query,
                refresh_every_n_frames=self.refresh_every_n_frames,
                max_history_frames=self.max_history_frames,
                max_image_size=self.max_image_size,
                api_url=self.api_url,
                api_key=self.api_key,
                api_key_env=self.api_key_env,
                model=self.model,
                ontology_version=self.ontology_version,
                conservative_mode=self.conservative_mode,
                output_path=self.output_path,
                assistant_modes=self.assistant_modes,
            )

        def set_query(self, new_query: str):
            """Dynamically update the target query from the live terminal thread."""
            self.user_query = new_query
            if hasattr(self, "copilot"):
                self.copilot.set_query(new_query)

        def compute(self, op_input, op_output, context):
            from operators.format_utils import holoscan_to_torch

            frame = holoscan_to_torch(op_input.receive("rgb_tensor"))
            detections = op_input.receive("bboxes")
            mask_labels = op_input.receive("mask_labels")
            masks = {
                label: torch.empty(0, device=frame.device)
                for label in mask_labels
            }
            self.copilot.analyze(
                frame,
                detections=detections,
                masks=masks,
                frame_idx=self.frame_idx,
            )
            self.frame_idx += 1

        def stop(self):
            self.copilot.reset()
