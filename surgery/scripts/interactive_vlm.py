"""
interactive_vlm.py - Live Streamlit Web App with MJPEG Server.
Provides a 2-column layout: Left (Live Video), Right (Chat Interface).
"""

from __future__ import annotations

import sys
import threading
import time
import re
import html
from dataclasses import dataclass
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import quote, urlparse

import cv2
import numpy as np
import torch
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app import load_app_config
from operators.vlm_prompt_op import AnatomyVLMGuide
from operators.overlay_compositor_op import FALLBACK_COLORS, OverlayCompositor

# To save boilerplate, import builders from a safe place.
# Wait, we might have circular imports if we import from interactive_vlm itself.
# We need to manually rewrite the builder functions here.
from operators.roboflow_detection_op import RoboflowHostedDetector
from operators.yolo_detection_op import YOLODetector
from operators.yolo_detection_op import Detection
from operators.sam2_inference_op import SAM2Segmenter
from operators.medsam2_inference_op import MedSAM2Segmenter
from operators.scene_copilot_op import SurgicalSceneCopilot
from session import SurgeryReportGenerator, SurgerySessionLog

LIVE_OVERLAY_DEFAULTS = {
    "enabled": True,
    "update_every_n_frames": 5,
    "max_inference_fps": 5.0,
    "mask_stale_after_frames": 12,
    "mask_stale_after_seconds": 0.5,
    "replace_on_new_query": True,
    "short_term_seconds": 6.0,
}
OVERLAY_UPDATE_TOKENS = ("highlight", "segment", "outline", "show", "mark")
OVERLAY_CLEAR_TOKENS = ("clear overlay", "remove overlay", "stop highlighting")
OVERLAY_NEGATIVE_TOKENS = (
    "turn off",
    "remove",
    "clear",
    "stop",
    "hide",
    "disable",
    "ignore",
)
OVERLAY_SWITCH_TOKENS = ("switch", "change", "instead")
OVERLAY_EXTRA_ALIASES = {
    "gallbladder": ["galbladder", "gall bladder", "gb"],
    "grasper": ["surgical tool", "surgeon tool", "surgery tool", "instrument", "tool", "forceps"],
}
OVERLAY_FREEFORM_STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "please",
    "for",
    "to",
    "with",
    "in",
    "on",
    "of",
    "segmentation",
    "overlay",
    "mask",
    "masks",
    "area",
    "region",
    "part",
}


@dataclass
class OverlayCommand:
    target_labels: list[str]
    remove_labels: list[str]
    clear_all: bool = False

def resolve_repo_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def encode_mjpeg_frame(bgr_img: np.ndarray) -> bytes:
    ret, jpeg = cv2.imencode(".jpg", bgr_img)
    if not ret:
        return b""
    return b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"


def build_loading_frame(text: str) -> bytes:
    loading_frame = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(loading_frame, text, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return encode_mjpeg_frame(loading_frame)


def get_live_overlay_cfg(cfg: dict) -> dict:
    merged = dict(LIVE_OVERLAY_DEFAULTS)
    merged.update(cfg.get("live_overlay", {}))
    return merged


def is_overlay_clear_query(query: str) -> bool:
    query_l = (query or "").strip().lower()
    return any(token in query_l for token in OVERLAY_CLEAR_TOKENS)


def is_overlay_update_query(query: str) -> bool:
    query_l = (query or "").strip().lower()
    return any(token in query_l for token in (*OVERLAY_UPDATE_TOKENS, *OVERLAY_NEGATIVE_TOKENS, *OVERLAY_SWITCH_TOKENS))


def _merged_label_aliases(candidate_labels: list[str], anatomy_aliases: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for label in candidate_labels:
        aliases = [label, *anatomy_aliases.get(label, []), *OVERLAY_EXTRA_ALIASES.get(label, [])]
        deduped: list[str] = []
        for alias in aliases:
            alias_l = alias.lower()
            if alias_l not in deduped:
                deduped.append(alias_l)
        merged[label] = sorted(deduped, key=len, reverse=True)
    return merged


def _find_label_mentions(query: str, label_aliases: dict[str, list[str]]) -> list[tuple[str, int, int]]:
    query_l = (query or "").lower()
    mentions: list[tuple[str, int, int]] = []
    for label, aliases in label_aliases.items():
        for alias in aliases:
            start = query_l.find(alias)
            while start != -1:
                end = start + len(alias)
                mentions.append((label, start, end))
                start = query_l.find(alias, end)
    mentions.sort(key=lambda item: (item[1], item[2] - item[1]))

    deduped: list[tuple[str, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for label, start, end in mentions:
        if any(start >= occ_start and end <= occ_end for occ_start, occ_end in occupied):
            continue
        deduped.append((label, start, end))
        occupied.append((start, end))
    return deduped


def _append_unique(items: list[str], label: str):
    if label not in items:
        items.append(label)


def normalize_overlay_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", (label or "").strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def humanize_overlay_label(label: str) -> str:
    return (label or "").replace("_", " ").strip() or "unknown"


def extract_freeform_overlay_labels(query: str) -> list[str]:
    query_l = (query or "").lower()
    trigger_match = re.search(r"\b(highlight|segment|outline|show|mark)\b", query_l)
    if not trigger_match:
        return []
    phrase = query_l[trigger_match.end() :]
    phrase = re.split(r"[.;,]|\band\b|\bthen\b|\bwhile\b|\bbecause\b|\bso\b", phrase, maxsplit=1)[0]
    phrase = re.sub(r"\b(turn|off|remove|clear|stop|hide|disable|ignore)\b.*", "", phrase)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9']+", phrase)
        if token not in OVERLAY_FREEFORM_STOPWORDS and len(token) > 1
    ]
    if not tokens:
        return []
    label = normalize_overlay_label(" ".join(tokens[:5]))
    return [label] if label else []


def resolve_requested_labels(query: str, candidate_labels: list[str], anatomy_aliases: dict[str, list[str]]) -> list[str]:
    query_l = (query or "").lower()
    matches: list[str] = []
    for label, aliases in _merged_label_aliases(candidate_labels, anatomy_aliases).items():
        if any(alias in query_l for alias in aliases):
            matches.append(label)
    return matches


def parse_overlay_command(query: str, cfg: dict, active_labels: list[str] | None = None) -> OverlayCommand:
    vlm_cfg = cfg.get("vlm", {})
    candidate_labels = list(vlm_cfg.get("candidate_labels", []) or [])
    anatomy_aliases = dict(vlm_cfg.get("anatomy_aliases", {}) or {})
    label_aliases = _merged_label_aliases(candidate_labels, anatomy_aliases)
    query_l = (query or "").strip().lower()
    active_labels = list(active_labels or [])
    mentions = _find_label_mentions(query_l, label_aliases)
    positive: list[str] = []
    negative: list[str] = []

    if "switch" in query_l and " to " in query_l:
        to_idx = query_l.rfind(" to ")
        for label, start, _ in mentions:
            if start < to_idx:
                _append_unique(negative, label)
            else:
                _append_unique(positive, label)
    else:
        clause_start = 0
        separators = (" and ", " then ", ",", ";", ".")
        for label, start, end in mentions:
            previous_separators = [
                pos + len(separator)
                for separator in separators
                if (pos := query_l.rfind(separator, 0, start)) != -1
            ]
            next_clause_start = max(previous_separators, default=0)
            clause_start = max(clause_start, next_clause_start)
            next_clause_end_candidates = [
                pos for separator in separators if (pos := query_l.find(separator, end)) != -1
            ]
            clause_end = min(next_clause_end_candidates) if next_clause_end_candidates else len(query_l)
            clause = query_l[clause_start:clause_end]
            prefix = query_l[max(clause_start, start - 48):start]

            is_negative = any(token in clause for token in OVERLAY_NEGATIVE_TOKENS) or any(
                token in prefix for token in OVERLAY_NEGATIVE_TOKENS
            )
            is_positive = any(token in clause for token in (*OVERLAY_UPDATE_TOKENS, *OVERLAY_SWITCH_TOKENS))
            if is_negative and not is_positive:
                _append_unique(negative, label)
            elif is_negative and any(token in prefix for token in OVERLAY_NEGATIVE_TOKENS):
                _append_unique(negative, label)
            else:
                _append_unique(positive, label)

    for label in negative:
        if label in positive:
            positive.remove(label)

    clear_all = is_overlay_clear_query(query_l) and not mentions
    if not positive and negative and active_labels:
        positive = [label for label in active_labels if label not in set(negative)]
        clear_all = not positive
    if not positive and not negative and is_overlay_update_query(query_l):
        positive = extract_freeform_overlay_labels(query_l)

    return OverlayCommand(target_labels=positive, remove_labels=negative, clear_all=clear_all)


def resolve_overlay_target_labels(query: str, cfg: dict) -> list[str]:
    return parse_overlay_command(query, cfg).target_labels


def build_visible_response_text(result: dict, scene_analysis: dict | None = None) -> str:
    selected_labels = result.get("selected_labels") or []
    overlay_updated = bool(result.get("overlay_updated"))
    if overlay_updated and selected_labels:
        return f"Overlay targets: {', '.join(selected_labels)}."
    if scene_analysis and scene_analysis.get("scene_summary"):
        return str(scene_analysis["scene_summary"])
    return "Overlay unchanged for this question."

def build_detector(cfg):
    det_backend = cfg.get("detector", {}).get("backend", "local_yolo")
    if det_backend == "roboflow_hosted":
        r_cfg = cfg["roboflow_laparoscopy"]
        return RoboflowHostedDetector(
            model_id=r_cfg["model_id"],
            api_url=r_cfg["api_url"],
            api_key=r_cfg["api_key"],
            api_key_env=r_cfg["api_key_env"],
            confidence_threshold=r_cfg["confidence_threshold"],
            detect_every_n_frames=r_cfg.get("detect_every_n_frames", 15),
            target_classes=r_cfg.get("target_classes"),
            class_name_map=r_cfg.get("class_name_map"),
        )
    y_cfg = cfg["yolo"]
    return YOLODetector(
        model_path=str(resolve_repo_path(y_cfg["model_path"])),
        confidence_threshold=y_cfg["confidence_threshold"],
        device=y_cfg.get("device", "cuda:0"),
        target_classes=y_cfg.get("target_classes"),
        class_name_map=y_cfg.get("class_name_map"),
    )

def build_segmenter(cfg):
    seg_backend = cfg.get("segmenter", {}).get("backend", "medsam2")
    if seg_backend == "medsam2":
        s_cfg = cfg["medsam2"]
        return MedSAM2Segmenter(
            checkpoint=str(resolve_repo_path(s_cfg["checkpoint"])),
            model_cfg=s_cfg["model_cfg"],
            device=s_cfg["device"],
            dtype=s_cfg.get("dtype", "bfloat16"),
            max_objects=s_cfg.get("max_objects", 5),
            use_temporal_memory=s_cfg.get("use_temporal_memory", True),
            prompt_classes=cfg.get("segmenter", {}).get("prompt_classes"),
        )
    s_cfg = cfg["sam2"]
    return SAM2Segmenter(
        checkpoint=str(resolve_repo_path(s_cfg["checkpoint"])),
        model_cfg=s_cfg["model_cfg"],
        device=s_cfg["device"],
    )


def build_vlm_guide(cfg):
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

def build_scene_copilot(cfg):
    c_cfg = cfg.get("scene_copilot")
    if not c_cfg or not c_cfg.get("enabled", False):
        return None
    return SurgicalSceneCopilot(
        enabled=c_cfg["enabled"],
        provider=c_cfg.get("provider", "rule_based"),
        user_query=c_cfg.get("user_query", ""),
        refresh_every_n_frames=c_cfg.get("refresh_every_n_frames", 30),
        max_history_frames=c_cfg.get("max_history_frames", 90),
        max_image_size=c_cfg.get("max_image_size", 512),
        api_url=c_cfg.get("api_url", ""),
        api_key=c_cfg.get("api_key", ""),
        api_key_env=c_cfg.get("api_key_env", "GROQ_API_KEY"),
        model=c_cfg.get("model", ""),
        ontology_version=c_cfg.get("ontology_version", "lap_chole_v1"),
        conservative_mode=c_cfg.get("conservative_mode", True),
        output_path="",
        assistant_modes=c_cfg.get("assistant_modes"),
    )


@dataclass
class LivePipeline:
    detector: object | None
    vlm_guide: AnatomyVLMGuide | None
    scene_copilot: SurgicalSceneCopilot | None
    segmenter: object | None
    overlay: OverlayCompositor | None
    device: str = "cpu"


# Global state for communication between Streamlit and MJPEG thread
class StreamState:
    def __init__(self, initial_video: str):
        self.latest_scene_analysis = None
        self.running = True
        self.target_video = initial_video
        self.current_video = None
        self.last_frame_bgr: np.ndarray | None = None
        self.last_source_frame_bgr: np.ndarray | None = None
        self.last_source_frame_idx = -1
        self.pipeline: LivePipeline | None = None
        self.pipeline_status = "initializing"
        self.pipeline_error = ""
        self.active_overlay_query = ""
        self.active_target_labels: list[str] = []
        self.overlay_status = "idle"
        self.overlay_error = ""
        self.overlay_job_running = False
        self.overlay_generation = 0
        self.overlay_last_completed_generation = 0
        self.overlay_target_modes: dict[str, str] = {}
        self.overlay_temporary_expires_at: dict[str, float] = {}
        self.latest_overlay_masks: dict[str, torch.Tensor] = {}
        self.latest_overlay_frame_idx = -1
        self.latest_overlay_updated_at = 0.0
        self._pending_overlay_frame_idx = -1
        self._pending_overlay_source_bgr: np.ndarray | None = None
        self._pending_overlay_generation = 0
        self.chat_status = "idle"
        self.chat_active_query = ""
        self.chat_error = ""
        self.chat_response = ""
        self.chat_started_at = 0.0
        self.chat_finished_at = 0.0
        self.chat_generation = 0
        self.chat_result_generation = 0
        self.surgery_log: SurgerySessionLog | None = None
        self.report_status = "idle"
        self.report_error = ""
        self.report_text = ""
        self.report_path = ""
        self.report_generation = 0
        self.last_logged_observation_frame_idx = -1
        self.last_keyframe_frame_idx = -1
        self._inference_lock = threading.Lock()
        self._lock = threading.Lock()

    def set_pipeline(self, pipeline: LivePipeline, errors: list[str]):
        with self._lock:
            self.pipeline = pipeline
            self.pipeline_status = "ready" if not errors else "degraded"
            self.pipeline_error = "; ".join(errors)
        self.append_log_event(
            "pipeline_ready" if not errors else "pipeline_degraded",
            status="ready" if not errors else "degraded",
            errors=errors,
        )

    def record_runtime_error(self, source: str, exc: Exception):
        message = f"{source}: {exc}"
        print(f"[warn] {message}")
        with self._lock:
            self.pipeline_status = "degraded"
            if message not in self.pipeline_error:
                self.pipeline_error = f"{self.pipeline_error}; {message}".strip("; ")

    def get_pipeline_snapshot(self) -> tuple[LivePipeline | None, str, str]:
        with self._lock:
            return self.pipeline, self.pipeline_status, self.pipeline_error

    def set_scene_analysis(self, analysis):
        with self._lock:
            self.latest_scene_analysis = analysis

    def set_surgery_log(self, surgery_log: SurgerySessionLog | None):
        with self._lock:
            self.surgery_log = surgery_log

    def append_log_event(self, event_type: str, **payload):
        with self._lock:
            surgery_log = self.surgery_log
        if surgery_log is None:
            return None
        try:
            return surgery_log.append_event(event_type, **payload)
        except Exception as exc:
            self.record_runtime_error("surgery log append failed", exc)
            return None

    def save_keyframe(self, frame_bgr: np.ndarray | None, *, frame_idx: int, reason: str, metadata: dict | None = None):
        with self._lock:
            surgery_log = self.surgery_log
        if surgery_log is None or frame_bgr is None:
            return None
        return surgery_log.save_keyframe(frame_bgr, frame_idx=frame_idx, reason=reason, metadata=metadata)

    def get_surgery_log_snapshot(self):
        with self._lock:
            if self.surgery_log is None:
                return None, "", 0
            return self.surgery_log, str(self.surgery_log.log_path), self.surgery_log.event_count

    def maybe_log_frame_observation(
        self,
        *,
        frame_idx: int,
        observation_every_n_frames: int,
        keyframe_every_n_frames: int,
        frame_bgr: np.ndarray,
    ):
        with self._lock:
            should_log = (
                self.surgery_log is not None
                and observation_every_n_frames > 0
                and (
                    self.last_logged_observation_frame_idx < 0
                    or frame_idx - self.last_logged_observation_frame_idx >= observation_every_n_frames
                )
            )
            active_targets = list(self.active_target_labels)
            latest_scene = dict(self.latest_scene_analysis or {})
            if should_log:
                self.last_logged_observation_frame_idx = frame_idx
            should_keyframe = (
                should_log
                and keyframe_every_n_frames > 0
                and (
                    self.last_keyframe_frame_idx < 0
                    or frame_idx - self.last_keyframe_frame_idx >= keyframe_every_n_frames
                )
            )
            if should_keyframe:
                self.last_keyframe_frame_idx = frame_idx
        if not should_log:
            return
        self.append_log_event(
            "frame_observation",
            frame_idx=frame_idx,
            active_overlay_targets=active_targets,
            visible_structures=latest_scene.get("visible_structures", []),
            visible_tools=latest_scene.get("visible_tools", []),
            workflow_phase=latest_scene.get("workflow_phase", ""),
            uncertainties=latest_scene.get("uncertainties", []),
        )
        if should_keyframe:
            self.save_keyframe(
                frame_bgr,
                frame_idx=frame_idx,
                reason="periodic",
                metadata={"active_overlay_targets": active_targets},
            )

    def start_report_generation(self):
        with self._lock:
            self.report_generation += 1
            self.report_status = "running"
            self.report_error = ""
            self.report_text = ""
            self.report_path = ""
            return self.report_generation

    def finish_report_generation(self, *, generation: int, report_text: str = "", report_path: str = "", error: str = ""):
        with self._lock:
            if generation != self.report_generation:
                return
            self.report_error = error
            self.report_text = report_text
            self.report_path = report_path
            self.report_status = "error" if error else "done"

    def get_report_snapshot(self):
        with self._lock:
            return (
                self.report_status,
                self.report_error,
                self.report_text,
                self.report_path,
                self.report_generation,
            )

    def set_last_frame(self, frame_bgr: np.ndarray):
        with self._lock:
            self.last_frame_bgr = frame_bgr.copy()

    def get_last_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.last_frame_bgr is None:
                return None
            return self.last_frame_bgr.copy()

    def set_last_source_frame(self, frame_bgr: np.ndarray):
        with self._lock:
            self.last_source_frame_bgr = frame_bgr.copy()

    def set_last_source_frame_snapshot(self, frame_bgr: np.ndarray, frame_idx: int):
        with self._lock:
            self.last_source_frame_bgr = frame_bgr.copy()
            self.last_source_frame_idx = frame_idx

    def get_last_source_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.last_source_frame_bgr is None:
                return None
            return self.last_source_frame_bgr.copy()

    def get_last_source_frame_snapshot(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self.last_source_frame_bgr is None:
                return None, -1
            return self.last_source_frame_bgr.copy(), self.last_source_frame_idx

    def _purge_expired_temporary_overlays_locked(self):
        now = time.time()
        expired = [
            label
            for label, expires_at in self.overlay_temporary_expires_at.items()
            if expires_at <= now
        ]
        if not expired:
            return
        expired_set = set(expired)
        self.active_target_labels = [label for label in self.active_target_labels if label not in expired_set]
        for label in expired:
            self.overlay_target_modes.pop(label, None)
            self.overlay_temporary_expires_at.pop(label, None)
            self.latest_overlay_masks.pop(label, None)
        if not self.active_target_labels:
            self.active_overlay_query = ""
            self.overlay_status = "idle"
            self.latest_overlay_frame_idx = -1
            self.latest_overlay_updated_at = 0.0
            self._pending_overlay_frame_idx = -1
            self._pending_overlay_source_bgr = None
            self.overlay_generation += 1
        else:
            self.overlay_generation += 1
            self._pending_overlay_generation = self.overlay_generation

    def _set_active_overlay_labels_locked(
        self,
        *,
        query: str,
        target_labels: list[str],
        remove_labels: list[str] | None = None,
        replace: bool = False,
        mode: str = "persistent",
        temporary_seconds: float | None = None,
    ):
        remove_set = {normalize_overlay_label(label) for label in (remove_labels or []) if normalize_overlay_label(label)}
        existing = [] if replace else [label for label in self.active_target_labels if label not in remove_set]
        labels: list[str] = []
        for label in [*existing, *target_labels]:
            normalized = normalize_overlay_label(label)
            if normalized and normalized not in labels:
                labels.append(normalized)
        self.active_overlay_query = query if labels else ""
        self.active_target_labels = labels
        for label in remove_set:
            self.overlay_target_modes.pop(label, None)
            self.overlay_temporary_expires_at.pop(label, None)
            self.latest_overlay_masks.pop(label, None)
        for label in target_labels:
            normalized = normalize_overlay_label(label)
            if not normalized:
                continue
            self.overlay_target_modes[normalized] = mode
            if temporary_seconds is not None:
                self.overlay_temporary_expires_at[normalized] = time.time() + max(0.5, float(temporary_seconds))
            else:
                self.overlay_temporary_expires_at.pop(normalized, None)
        self.overlay_status = "running" if labels else "idle"
        self.overlay_error = ""
        self.overlay_job_running = False
        self.overlay_generation += 1
        self.latest_overlay_masks = {
            label: mask for label, mask in self.latest_overlay_masks.items() if label in set(labels)
        }
        self.latest_overlay_frame_idx = -1
        self.latest_overlay_updated_at = 0.0
        self._pending_overlay_frame_idx = -1
        self._pending_overlay_source_bgr = None
        self._pending_overlay_generation = self.overlay_generation

    def start_overlay_query(
        self,
        query: str,
        target_labels: list[str],
        *,
        remove_labels: list[str] | None = None,
        replace: bool = False,
        mode: str = "persistent",
        temporary_seconds: float | None = None,
    ):
        with self._lock:
            self._purge_expired_temporary_overlays_locked()
            self._set_active_overlay_labels_locked(
                query=query,
                target_labels=target_labels,
                remove_labels=remove_labels,
                replace=replace,
                mode=mode,
                temporary_seconds=temporary_seconds,
            )

    def remove_overlay_labels(self, labels: list[str]):
        normalized_remove = [normalize_overlay_label(label) for label in labels]
        with self._lock:
            self._purge_expired_temporary_overlays_locked()
            remove_set = set(normalized_remove)
            self.active_target_labels = [label for label in self.active_target_labels if label not in remove_set]
            for label in remove_set:
                self.overlay_target_modes.pop(label, None)
                self.overlay_temporary_expires_at.pop(label, None)
                self.latest_overlay_masks.pop(label, None)
            self.overlay_generation += 1
            self.overlay_error = ""
            self.overlay_job_running = False
            self.latest_overlay_frame_idx = -1
            self.latest_overlay_updated_at = 0.0
            self._pending_overlay_frame_idx = -1
            self._pending_overlay_source_bgr = None
            self._pending_overlay_generation = self.overlay_generation
            if self.active_target_labels:
                self.active_overlay_query = f"Removed overlay labels: {', '.join(normalized_remove)}"
                self.overlay_status = "running"
            else:
                self.active_overlay_query = ""
                self.overlay_status = "idle"

    def clear_overlay(self):
        with self._lock:
            self.active_overlay_query = ""
            self.active_target_labels = []
            self.overlay_status = "idle"
            self.overlay_error = ""
            self.overlay_job_running = False
            self.overlay_generation += 1
            self.overlay_target_modes = {}
            self.overlay_temporary_expires_at = {}
            self.latest_overlay_masks = {}
            self.latest_overlay_frame_idx = -1
            self.latest_overlay_updated_at = 0.0
            self._pending_overlay_frame_idx = -1
            self._pending_overlay_source_bgr = None
            self._pending_overlay_generation = self.overlay_generation

    def get_overlay_snapshot(self) -> tuple[str, list[str], str, str, bool, int, int, float]:
        with self._lock:
            self._purge_expired_temporary_overlays_locked()
            return (
                self.active_overlay_query,
                list(self.active_target_labels),
                self.overlay_status,
                self.overlay_error,
                self.overlay_job_running,
                self.overlay_generation,
                self.overlay_last_completed_generation,
                self.latest_overlay_updated_at,
            )

    def offer_overlay_frame(self, frame_idx: int, frame_bgr: np.ndarray, *, update_every_n_frames: int):
        with self._lock:
            self._purge_expired_temporary_overlays_locked()
            if not self.active_target_labels:
                return
            if update_every_n_frames > 1 and frame_idx % update_every_n_frames != 0:
                return
            self._pending_overlay_frame_idx = frame_idx
            self._pending_overlay_source_bgr = frame_bgr.copy()
            self._pending_overlay_generation = self.overlay_generation

    def claim_overlay_job(self) -> tuple[int, np.ndarray, int, str, list[str]] | None:
        with self._lock:
            self._purge_expired_temporary_overlays_locked()
            if self.overlay_job_running or not self.active_target_labels or self._pending_overlay_source_bgr is None:
                return None
            self.overlay_job_running = True
            job = (
                self._pending_overlay_frame_idx,
                self._pending_overlay_source_bgr.copy(),
                self._pending_overlay_generation,
                self.active_overlay_query,
                list(self.active_target_labels),
            )
            self._pending_overlay_frame_idx = -1
            self._pending_overlay_source_bgr = None
            return job

    def finish_overlay_job(
        self,
        *,
        generation: int,
        frame_idx: int,
        masks: dict[str, torch.Tensor] | None = None,
        error: str = "",
    ):
        event_payload = None
        with self._lock:
            self.overlay_job_running = False
            if generation != self.overlay_generation:
                return
            self.overlay_last_completed_generation = generation
            self.overlay_error = error
            active_query = self.active_overlay_query
            active_targets = list(self.active_target_labels)
            if error:
                self.overlay_status = "error"
                self.latest_overlay_masks = {}
                self.latest_overlay_frame_idx = -1
                self.latest_overlay_updated_at = time.time()
                event_payload = {
                    "event_type": "overlay_failed",
                    "prompt": active_query,
                    "overlay_targets": active_targets,
                    "frame_idx": frame_idx,
                    "error": error,
                }
            masks = dict(masks or {})
            if not error and masks:
                self.overlay_status = "active"
                self.latest_overlay_masks = masks
                self.latest_overlay_frame_idx = frame_idx
                self.latest_overlay_updated_at = time.time()
                event_payload = {
                    "event_type": "overlay_updated",
                    "prompt": active_query,
                    "overlay_targets": active_targets,
                    "mask_labels": list(masks.keys()),
                    "frame_idx": frame_idx,
                }
            elif not error:
                self.overlay_status = "ungrounded"
                self.latest_overlay_masks = {}
                self.latest_overlay_frame_idx = frame_idx
                self.latest_overlay_updated_at = time.time()
                event_payload = {
                    "event_type": "overlay_ungrounded",
                    "prompt": active_query,
                    "overlay_targets": active_targets,
                    "frame_idx": frame_idx,
                }
        if event_payload:
            event_type = event_payload.pop("event_type")
            self.append_log_event(event_type, **event_payload)

    def get_latest_overlay_masks(self) -> tuple[dict[str, torch.Tensor], int, float]:
        with self._lock:
            self._purge_expired_temporary_overlays_locked()
            return dict(self.latest_overlay_masks), self.latest_overlay_frame_idx, self.latest_overlay_updated_at

    def get_overlay_key_items(self) -> list[dict[str, str]]:
        with self._lock:
            self._purge_expired_temporary_overlays_locked()
            labels_with_masks = set(self.latest_overlay_masks.keys())
            items = []
            for label in self.active_target_labels:
                items.append(
                    {
                        "label": label,
                        "name": humanize_overlay_label(label),
                        "mode": self.overlay_target_modes.get(label, "persistent"),
                        "status": "active" if label in labels_with_masks else "pending",
                    }
                )
            return items

    def start_chat_query(self, query: str):
        with self._lock:
            self.chat_generation += 1
            self.chat_status = "running"
            self.chat_active_query = query
            self.chat_error = ""
            self.chat_response = ""
            self.chat_started_at = time.time()
            self.chat_finished_at = 0.0
            return self.chat_generation

    def finish_chat_query(self, *, generation: int, response: str = "", error: str = ""):
        with self._lock:
            if generation != self.chat_generation:
                return
            self.chat_status = "error" if error else "done"
            self.chat_error = error
            self.chat_response = response
            self.chat_finished_at = time.time()
            self.chat_result_generation = generation

    def get_chat_snapshot(self) -> tuple[str, str, str, str, float, int]:
        with self._lock:
            return (
                self.chat_status,
                self.chat_active_query,
                self.chat_error,
                self.chat_response,
                self.chat_finished_at,
                self.chat_result_generation,
            )

    def inference_lock(self):
        return self._inference_lock


def build_live_pipeline(cfg) -> tuple[LivePipeline, list[str]]:
    errors: list[str] = []

    detector = None
    try:
        detector = build_detector(cfg)
    except Exception as exc:
        errors.append(f"detector unavailable ({exc})")
        print(f"[warn] detector initialization failed: {exc}")

    try:
        vlm_guide = build_vlm_guide(cfg)
    except Exception as exc:
        vlm_guide = None
        errors.append(f"VLM guide unavailable ({exc})")
        print(f"[warn] VLM guide initialization failed: {exc}")

    try:
        scene_copilot = build_scene_copilot(cfg)
    except Exception as exc:
        scene_copilot = None
        errors.append(f"scene copilot unavailable ({exc})")
        print(f"[warn] scene copilot initialization failed: {exc}")

    try:
        segmenter = build_segmenter(cfg)
    except Exception as exc:
        segmenter = None
        errors.append(f"segmenter unavailable ({exc})")
        print(f"[warn] segmenter initialization failed: {exc}")

    device = getattr(segmenter, "device", "cpu")
    try:
        overlay = OverlayCompositor(
            colors=cfg["overlay"]["colors"],
            blend_alpha=cfg["overlay"]["blend_alpha"],
            glow_effect=cfg["overlay"]["glow_effect"],
            glow_radius=cfg["overlay"]["glow_radius"],
            contour_thickness=cfg["overlay"]["contour_thickness"],
            device=device,
        )
    except Exception as exc:
        overlay = None
        errors.append(f"overlay unavailable ({exc})")
        print(f"[warn] overlay initialization failed: {exc}")

    return LivePipeline(
        detector=detector,
        vlm_guide=vlm_guide,
        scene_copilot=scene_copilot,
        segmenter=segmenter,
        overlay=overlay,
        device=device,
    ), errors


def start_pipeline_bootstrap(cfg, state: StreamState):
    def worker():
        pipeline, errors = build_live_pipeline(cfg)
        state.set_pipeline(pipeline, errors)
        if errors:
            print("[warn] Live AR pipeline started in degraded mode")
        else:
            print("[ok] Live AR pipeline ready")

    threading.Thread(target=worker, daemon=True).start()


def build_overlay_fallback_detections(
    *,
    requested_labels: list[str],
    frame_shape: tuple[int, ...],
    existing_detections: list[Detection] | None = None,
) -> list[Detection]:
    present_labels = {det.class_name for det in (existing_detections or [])}
    height, width = frame_shape[:2]
    fallback_detections: list[Detection] = []

    for label in requested_labels:
        if label in present_labels:
            continue

        if label == "liver":
            fallback_detections.extend(
                [
                    Detection(
                        class_name="liver",
                        bbox=[0.0, 0.0, width * 0.43, height - 1.0],
                        confidence=0.2,
                        source_model="paused_prompt_fallback",
                    ),
                    Detection(
                        class_name="liver",
                        bbox=[width * 0.58, height * 0.02, width - 1.0, height - 1.0],
                        confidence=0.2,
                        source_model="paused_prompt_fallback",
                    ),
                ]
            )
        elif label == "gallbladder":
            fallback_detections.append(
                Detection(
                    class_name="gallbladder",
                    bbox=[width * 0.35, 0.0, width * 0.60, height * 0.35],
                    confidence=0.2,
                    source_model="paused_prompt_fallback",
                )
            )

    return fallback_detections


def reset_inference_components(pipeline: LivePipeline, *component_names: str):
    names = component_names or ("detector", "segmenter", "scene_copilot")
    for component_name in names:
        component = getattr(pipeline, component_name, None)
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def infer_overlay_masks(
    *,
    rgb_img: np.ndarray,
    bgr_img: np.ndarray,
    frame_idx: int,
    pipeline: LivePipeline,
    target_labels: list[str],
) -> tuple[list[Detection], list[Detection], dict[str, torch.Tensor]]:
    frame_t = torch.from_numpy(rgb_img).to(pipeline.device)
    detections: list[Detection] = []
    if pipeline.detector is not None:
        detections = pipeline.detector.detect(frame_t)

    filtered_detections = [det for det in detections if det.class_name in set(target_labels)]
    detected_labels = {det.class_name for det in filtered_detections}
    missing_labels = [label for label in target_labels if label not in detected_labels]
    if missing_labels and pipeline.vlm_guide is not None and hasattr(pipeline.vlm_guide, "localize_prompt_boxes"):
        try:
            localized_detections = pipeline.vlm_guide.localize_prompt_boxes(
                frame_t,
                missing_labels,
                existing_detections=detections,
                frame_idx=frame_idx,
            )
            if localized_detections:
                labels = ", ".join(sorted({det.class_name for det in localized_detections}))
                print(f"[overlay] using VLM prompt boxes for labels: {labels}", flush=True)
                filtered_detections = [*filtered_detections, *localized_detections]
        except Exception as exc:
            print(f"[overlay] VLM prompt-box localization failed: {exc}", flush=True)

    fallback_detections = build_overlay_fallback_detections(
        requested_labels=target_labels,
        frame_shape=bgr_img.shape,
        existing_detections=filtered_detections,
    )
    if fallback_detections:
        labels = ", ".join(sorted({det.class_name for det in fallback_detections}))
        print(f"[overlay] using fallback boxes for labels: {labels}", flush=True)
        filtered_detections = [*filtered_detections, *fallback_detections]

    masks: dict[str, torch.Tensor] = {}
    if pipeline.segmenter is not None and hasattr(pipeline.segmenter, "segment_frame") and filtered_detections:
        masks = pipeline.segmenter.segment_frame(
            frame_t,
            filtered_detections,
            frame_idx=frame_idx,
        )
    return detections, filtered_detections, masks


def compose_overlay_frame(
    *,
    rgb_img: np.ndarray,
    bgr_img: np.ndarray,
    pipeline: LivePipeline,
    masks: dict[str, torch.Tensor],
) -> np.ndarray:
    if pipeline.overlay is None or not masks:
        return bgr_img
    frame_t = torch.from_numpy(rgb_img).to(pipeline.device)
    composited = pipeline.overlay.composite(frame_t, masks)
    composited_rgb = composited[:, :, :3].cpu().numpy().astype(np.uint8)
    return cv2.cvtColor(composited_rgb, cv2.COLOR_RGB2BGR)


def overlay_color_for_label(label: str, cfg: dict, pipeline: LivePipeline | None = None) -> list[int]:
    overlay = pipeline.overlay if pipeline is not None else None
    if overlay is not None and hasattr(overlay, "_get_color"):
        return list(overlay._get_color(label))
    configured = cfg.get("overlay", {}).get("colors", {})
    if label in configured:
        return list(configured[label])
    fallback_idx = abs(hash(label)) % max(1, len(FALLBACK_COLORS))
    return list(FALLBACK_COLORS[fallback_idx])


def rgba_to_css(rgba: list[int]) -> str:
    r, g, b, a = [int(v) for v in rgba[:4]]
    return f"rgba({r}, {g}, {b}, {max(0.15, min(a / 255.0, 1.0)):.2f})"


def render_chat_history_html(messages: list[dict[str, str]]) -> str:
    rows = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = html.escape(str(msg.get("content", ""))).replace("\n", "<br>")
        rows.append(
            f"<div class='chat-row chat-row-{role}'>"
            f"<div class='chat-bubble chat-bubble-{role}'>{content}</div>"
            "</div>"
        )
    return "<div class='chat-scroll'>" + "".join(rows) + "</div>"


def start_overlay_worker(
    *,
    cfg: dict,
    state: StreamState,
):
    overlay_cfg = get_live_overlay_cfg(cfg)
    min_job_interval = 1.0 / max(float(overlay_cfg["max_inference_fps"]), 0.1)

    def worker():
        last_job_started_at = 0.0
        while state.running:
            pipeline, pipeline_status, _ = state.get_pipeline_snapshot()
            if not overlay_cfg.get("enabled", True) or pipeline is None or pipeline_status == "initializing":
                time.sleep(0.05)
                continue

            overlay_query, target_labels, _, _, _, _, _, _ = state.get_overlay_snapshot()
            if not overlay_query or not target_labels:
                time.sleep(0.02)
                continue

            job = state.claim_overlay_job()
            if job is None:
                time.sleep(0.01)
                continue

            frame_idx, source_bgr, generation, query, target_labels = job
            wait_for = min_job_interval - (time.time() - last_job_started_at)
            if wait_for > 0:
                time.sleep(wait_for)
            last_job_started_at = time.time()

            try:
                with state.inference_lock():
                    reset_inference_components(pipeline, "detector", "segmenter")
                    rgb_img = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
                    _, _, masks = infer_overlay_masks(
                        rgb_img=rgb_img,
                        bgr_img=source_bgr,
                        frame_idx=frame_idx,
                        pipeline=pipeline,
                        target_labels=target_labels,
                    )
                state.finish_overlay_job(generation=generation, frame_idx=frame_idx, masks=masks)
            except Exception as exc:
                state.record_runtime_error("overlay worker failed", exc)
                state.finish_overlay_job(generation=generation, frame_idx=frame_idx, error=str(exc))

    threading.Thread(target=worker, daemon=True).start()


def start_scene_query_worker(
    *,
    state: StreamState,
    pipeline: LivePipeline,
    frame_idx: int,
    query: str,
    source_bgr: np.ndarray,
    conversation_history: list[dict[str, str]] | None = None,
    temporary_highlight_seconds: float = 6.0,
):
    generation = state.start_chat_query(query)
    print(f"[chat] background scene analysis started for query: {query}", flush=True)

    def worker():
        try:
            rgb_img = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
            with state.inference_lock():
                reset_inference_components(pipeline, "detector", "scene_copilot")
                frame_t = torch.from_numpy(rgb_img).to(pipeline.device)
                detections = pipeline.detector.detect(frame_t) if pipeline.detector is not None else []
                masks, _, _ = state.get_latest_overlay_masks()
                analysis = None
                if pipeline.scene_copilot is not None:
                    pipeline.scene_copilot.set_query(query)
                    analysis = pipeline.scene_copilot.analyze(
                        frame_t,
                        detections,
                        masks,
                        frame_idx=frame_idx,
                        user_query=query,
                        conversation_history=conversation_history,
                    )
            if analysis:
                state.set_scene_analysis(analysis.to_dict())
                response = analysis.surgeon_response or analysis.scene_summary
                temporary_targets = []
                for target in analysis.recommended_attention_targets:
                    normalized = normalize_overlay_label(str(target))
                    if normalized and normalized not in temporary_targets:
                        temporary_targets.append(normalized)
                if temporary_targets:
                    state.start_overlay_query(
                        f"Temporary explanation focus: {query}",
                        temporary_targets[:3],
                        mode="temporary",
                        temporary_seconds=temporary_highlight_seconds,
                    )
                    state.append_log_event(
                        "temporary_overlay_requested",
                        frame_idx=frame_idx,
                        prompt=query,
                        overlay_targets=temporary_targets[:3],
                        duration_seconds=temporary_highlight_seconds,
                    )
                state.append_log_event(
                    "assistant_response",
                    frame_idx=frame_idx,
                    user_text=query,
                    response_text=response,
                    clinical_log_note=analysis.reasoning_summary,
                    assistant_mode=analysis.assistant_mode,
                    visible_structures=analysis.visible_structures,
                    visible_tools=analysis.visible_tools,
                    workflow_phase=analysis.workflow_phase,
                    critical_view_status=analysis.critical_view_status,
                    observed_risks=analysis.observed_risks,
                    uncertainties=analysis.uncertainties,
                    recommended_attention_targets=analysis.recommended_attention_targets,
                    confidence=analysis.confidence,
                )
            else:
                response = "Scene analysis is unavailable for this request."
                state.append_log_event(
                    "assistant_response",
                    frame_idx=frame_idx,
                    user_text=query,
                    response_text=response,
                    assistant_mode="unavailable",
                    confidence=0.0,
                )
            print(f"[chat] background scene analysis finished for query: {query}", flush=True)
            state.finish_chat_query(generation=generation, response=response)
        except Exception as exc:
            state.record_runtime_error("scene query worker failed", exc)
            print(f"[chat] scene analysis failed for query '{query}': {exc}", flush=True)
            state.finish_chat_query(generation=generation, error=str(exc))

    threading.Thread(target=worker, daemon=True).start()


def start_report_worker(*, state: StreamState, cfg: dict):
    generation = state.start_report_generation()
    surgery_log, log_path, _ = state.get_surgery_log_snapshot()
    if surgery_log is None:
        state.finish_report_generation(generation=generation, error="No active surgery log is available.")
        return

    def worker():
        try:
            state.append_log_event("session_ended", reason="end_surgery_button")
            events = surgery_log.read_events()
            keyframes = surgery_log.keyframes()
            copilot_cfg = cfg.get("scene_copilot", {})
            log_cfg = cfg.get("surgery_log", {})
            generator = SurgeryReportGenerator(
                provider=copilot_cfg.get("provider", "rule_based"),
                api_url=copilot_cfg.get("api_url", ""),
                api_key=copilot_cfg.get("api_key", ""),
                api_key_env=copilot_cfg.get("api_key_env", "GROQ_API_KEY"),
                model=copilot_cfg.get("model", ""),
                max_keyframes=log_cfg.get("report_max_keyframes", 6),
            )
            report_text = generator.generate(
                events=events,
                keyframe_paths=keyframes,
                metadata={
                    "video_name": surgery_log.video_name,
                    "session_id": surgery_log.session_id,
                    "log_path": log_path,
                    "keyframe_count": len(keyframes),
                },
            )
            report_path = surgery_log.session_dir / "draft_report.md"
            report_path.write_text(report_text, encoding="utf-8")
            state.append_log_event(
                "final_report_generated",
                report_path=str(report_path),
                keyframe_count=len(keyframes),
            )
            state.finish_report_generation(
                generation=generation,
                report_text=report_text,
                report_path=str(report_path),
            )
        except Exception as exc:
            state.append_log_event("final_report_failed", error=str(exc))
            state.finish_report_generation(generation=generation, error=str(exc))

    threading.Thread(target=worker, daemon=True).start()


def run_mjpeg_loop(cfg, state: StreamState):
    print("[info] Starting MJPEG Pipeline Thread...")

    loading_bytes = build_loading_frame("Loading AI Models and Video... Please wait.")
    if loading_bytes:
        yield loading_bytes

    video_dir = resolve_repo_path(cfg["replayer"]["directory"])
    live_overlay_cfg = get_live_overlay_cfg(cfg)
    surgery_log_cfg = cfg.get("surgery_log", {})
    observation_every_n_frames = int(surgery_log_cfg.get("observation_every_n_frames", 75))
    keyframe_every_n_frames = int(surgery_log_cfg.get("keyframe_every_n_frames", 250))
    replayer_cfg = cfg.get("replayer", {})
    realtime = replayer_cfg.get("realtime", True)
    frame_rate = float(replayer_cfg.get("frame_rate", 25.0) or 25.0)
    frame_interval = 1.0 / max(frame_rate, 0.1)
    frames = []
    frame_idx = 0
    total_frames = 1
    next_frame_deadline = time.perf_counter()

    while state.running:
        if state.target_video != state.current_video:
            loading_bytes = build_loading_frame(f"Loading video {state.target_video}...")
            if loading_bytes:
                yield loading_bytes

            base_path = video_dir / state.target_video / state.target_video
            frames_dir = resolve_repo_path(base_path.parent / "frames")

            if not frames_dir.exists():
                err_bytes = build_loading_frame(f"Error: Frames not found for {state.target_video}")
                if err_bytes:
                    yield err_bytes
                time.sleep(1)
                continue

            frames = sorted(frames_dir.glob("*.jpg"))
            if not frames:
                err_bytes = build_loading_frame(f"Error: No JPEGs in {state.target_video}")
                if err_bytes:
                    yield err_bytes
                time.sleep(1)
                continue

            frame_idx = 0
            total_frames = len(frames)
            next_frame_deadline = time.perf_counter()
            state.current_video = state.target_video
            state.set_scene_analysis(None)
            state.clear_overlay()
            state.append_log_event(
                "video_selected",
                video_name=state.target_video,
                frame_count=total_frames,
            )
            continue

        if not state.current_video or not frames:
            time.sleep(0.1)
            continue

        frame_path = frames[frame_idx % total_frames]
        bgr_img = cv2.imread(str(frame_path))
        if bgr_img is None:
            time.sleep(0.1)
            continue
        state.set_last_source_frame_snapshot(bgr_img, frame_idx)
        state.offer_overlay_frame(
            frame_idx,
            bgr_img,
            update_every_n_frames=int(live_overlay_cfg["update_every_n_frames"]),
        )
        if surgery_log_cfg.get("enabled", True):
            state.maybe_log_frame_observation(
                frame_idx=frame_idx,
                observation_every_n_frames=observation_every_n_frames,
                keyframe_every_n_frames=keyframe_every_n_frames,
                frame_bgr=bgr_img,
            )

        display_bgr = bgr_img
        pipeline, _, _ = state.get_pipeline_snapshot()
        overlay_query, _, overlay_status, _, _, _, _, _ = state.get_overlay_snapshot()
        latest_masks, overlay_frame_idx, overlay_updated_at = state.get_latest_overlay_masks()
        if (
            overlay_query
            and overlay_status == "active"
            and latest_masks
            and pipeline is not None
            and time.time() - overlay_updated_at <= float(live_overlay_cfg["mask_stale_after_seconds"])
            and abs(frame_idx - overlay_frame_idx) <= int(live_overlay_cfg["mask_stale_after_frames"])
        ):
            try:
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                display_bgr = compose_overlay_frame(
                    rgb_img=rgb_img,
                    bgr_img=bgr_img,
                    pipeline=pipeline,
                    masks=latest_masks,
                )
            except Exception as exc:
                state.record_runtime_error("live overlay composition failed", exc)
                display_bgr = bgr_img

        frame_bytes = encode_mjpeg_frame(display_bgr)
        if frame_bytes:
            state.set_last_frame(display_bgr)
            yield frame_bytes

        frame_idx += 1
        if realtime:
            next_frame_deadline += frame_interval
            sleep_for = next_frame_deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame_deadline = time.perf_counter()


@st.cache_resource
def start_video_server(config_path: str, groq_key: str, roboflow_key: str, initial_video: str):
    cfg = load_app_config(str(resolve_repo_path(config_path)))
    cfg.setdefault("live_overlay", {}).update(get_live_overlay_cfg(cfg))
    if groq_key:
        cfg.setdefault("vlm", {})["api_key"] = groq_key
        cfg.setdefault("scene_copilot", {})["api_key"] = groq_key
    if roboflow_key:
        cfg.setdefault("roboflow_laparoscopy", {})["api_key"] = roboflow_key

    state = StreamState(initial_video)
    surgery_log_cfg = cfg.get("surgery_log", {})
    if surgery_log_cfg.get("enabled", True):
        log_dir = resolve_repo_path(surgery_log_cfg.get("output_dir", "data/surgery_logs"))
        try:
            state.set_surgery_log(SurgerySessionLog(log_dir, initial_video))
            state.append_log_event(
                "app_started",
                config_path=str(resolve_repo_path(config_path)),
                initial_video=initial_video,
            )
        except Exception as exc:
            state.record_runtime_error("surgery log initialization failed", exc)
    start_pipeline_bootstrap(cfg, state)
    start_overlay_worker(cfg=cfg, state=state)

    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_path = urlparse(self.path).path
            if request_path == '/video_feed':
                self.send_response(200)
                self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    for frame_bytes in run_mjpeg_loop(cfg, state):
                        self.wfile.write(frame_bytes)
                        self.wfile.flush()
                except Exception as e:
                    print(f"Stream interrupted: {e}")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass # Suppress logging

    server = ThreadingHTTPServer(('127.0.0.1', 8503), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return state, server


def main():
    st.set_page_config(page_title="Surgical VLM Live", page_icon=":movie_camera:", layout="wide")
    st.markdown(
        """
        <style>
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            height: 100vh !important;
            overflow: hidden !important;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"], footer {display: none !important;}
        .block-container {
            max-width: 100vw !important;
            height: 100vh !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        .stApp {background: radial-gradient(circle at 20% 20%, #12221f 0, #05070a 42%, #020304 100%);}
        :root {
            --copilot-width: min(390px, calc(100vw - 2rem));
            --stage-pad: 1rem;
            --topbar-clearance: 5rem;
        }
        .full-bleed-stage {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background: #000;
        }
        [data-testid="stHorizontalBlock"] {
            position: fixed !important;
            inset: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            gap: 0 !important;
            align-items: stretch;
            flex-wrap: nowrap !important;
            overflow: hidden !important;
            pointer-events: none;
        }
        [data-testid="stHorizontalBlock"] > div:nth-of-type(1) {
            flex: 1 1 100vw !important;
            min-width: 0 !important;
            max-width: 100vw !important;
            display: flex !important;
            align-items: stretch !important;
            pointer-events: auto;
        }
        [data-testid="stHorizontalBlock"] > div:nth-of-type(2) {
            position: fixed !important;
            top: 1rem !important;
            right: 1rem !important;
            bottom: 1rem !important;
            width: var(--copilot-width) !important;
            min-width: 0 !important;
            max-width: var(--copilot-width) !important;
            z-index: 20 !important;
            pointer-events: auto;
        }
        [data-testid="stHorizontalBlock"] > div:nth-of-type(2) > [data-testid="stVerticalBlock"] {
            background: linear-gradient(145deg, rgba(9, 14, 18, 0.55), rgba(9, 14, 18, 0.28));
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            backdrop-filter: blur(24px) saturate(1.28);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.42), inset 0 1px 0 rgba(255,255,255,0.08);
            padding: 0.9rem;
            height: calc(100vh - 2rem);
            overflow: hidden;
        }
        .surgical-video-shell {
            position: relative;
            width: 100vw;
            height: 100vh;
            margin: 0;
            box-sizing: border-box;
            padding: var(--topbar-clearance) calc(var(--copilot-width) + 2rem) var(--stage-pad) var(--stage-pad);
            border-radius: 0;
            overflow: hidden;
            background:
                radial-gradient(circle at 34% 42%, rgba(92, 35, 25, 0.42), transparent 42%),
                radial-gradient(circle at 78% 66%, rgba(44, 16, 13, 0.58), transparent 36%),
                linear-gradient(135deg, #090303 0%, #160706 48%, #050202 100%);
            border: 0;
            box-shadow: none;
        }
        .surgical-video-shell img {
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            background: rgba(0,0,0,0.88);
            border-radius: 20px;
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 24px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .surgical-video-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background:
                linear-gradient(90deg, rgba(0,0,0,0.12), transparent 24%, transparent 68%, rgba(0,0,0,0.34)),
                linear-gradient(180deg, rgba(0,0,0,0.18), transparent 18%, rgba(0,0,0,0.22));
        }
        .video-topbar {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.75rem 1rem;
            color: rgba(255,255,255,0.86);
            background: linear-gradient(180deg, rgba(9,12,16,0.94), rgba(9,12,16,0.56));
            font-size: 0.9rem;
            letter-spacing: 0.01em;
            position: absolute;
            top: 1rem;
            left: 1rem;
            right: calc(var(--copilot-width) + 2rem);
            z-index: 5;
            border: 1px solid rgba(255,255,255,0.13);
            border-radius: 16px;
            backdrop-filter: blur(18px) saturate(1.2);
            box-shadow: 0 18px 50px rgba(0,0,0,0.26), inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .overlay-key-card {
            position: absolute;
            left: 1rem;
            bottom: 1rem;
            max-width: min(430px, calc(100vw - var(--copilot-width) - 4rem));
            padding: 0.75rem 0.85rem;
            color: white;
            background: linear-gradient(145deg, rgba(8, 12, 16, 0.58), rgba(8, 12, 16, 0.32));
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 18px;
            backdrop-filter: blur(20px) saturate(1.22);
            box-shadow: 0 18px 52px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.08);
            font-size: 0.82rem;
            z-index: 8;
        }
        .overlay-key-title {font-weight: 700; margin-bottom: 0.45rem;}
        .overlay-key-row {
            display: grid;
            grid-template-columns: 0.9rem 1fr auto;
            align-items: center;
            gap: 0.5rem;
            margin: 0.35rem 0;
            padding: 0.38rem 0.45rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
        }
        .overlay-swatch {width: 0.9rem; height: 0.9rem; border-radius: 999px; border: 1px solid rgba(255,255,255,0.65);}
        .overlay-mode {color: rgba(255,255,255,0.58); font-size: 0.75rem;}
        .overlay-toggle {
            color: rgba(255,255,255,0.82);
            font-size: 0.72rem;
            padding: 0.24rem 0.55rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.15);
            text-decoration: none;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 22px rgba(0,0,0,0.18);
        }
        .overlay-toggle:hover {
            color: white;
            background: rgba(255,255,255,0.18);
            border-color: rgba(255,255,255,0.28);
        }
        .copilot-title {
            font-size: 1rem;
            font-weight: 750;
            color: white;
            margin-bottom: 0.2rem;
            letter-spacing: 0.01em;
        }
        .glass-control {
            color: rgba(255,255,255,0.65);
            font-size: 0.78rem;
            margin-bottom: 0.35rem;
        }
        [data-testid="stVerticalBlock"] > div:has([data-testid="stChatMessage"]) {
            overflow-y: auto !important;
        }
        .chat-scroll {
            height: 360px;
            overflow-y: auto;
            padding: 0.75rem;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
            background:
                radial-gradient(circle at 18% 0%, rgba(255,255,255,0.07), transparent 32%),
                rgba(8, 10, 14, 0.22);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            margin: 0.8rem 0 0.75rem 0;
        }
        .chat-row {
            display: flex;
            width: 100%;
            margin: 0.42rem 0;
        }
        .chat-row-user {justify-content: flex-end;}
        .chat-row-assistant {justify-content: flex-start;}
        .chat-bubble {
            max-width: 82%;
            padding: 0.68rem 0.82rem;
            border-radius: 18px;
            line-height: 1.38;
            font-size: 0.92rem;
            word-break: break-word;
            white-space: normal;
            box-shadow: 0 12px 32px rgba(0,0,0,0.20);
        }
        .chat-bubble-user {
            color: #12161c;
            background: linear-gradient(135deg, rgba(248,250,255,0.95), rgba(215,222,235,0.88));
            border-bottom-right-radius: 7px;
        }
        .chat-bubble-assistant {
            color: rgba(255,255,255,0.93);
            background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
            border: 1px solid rgba(255,255,255,0.10);
            border-bottom-left-radius: 7px;
            backdrop-filter: blur(12px);
        }
        div[data-testid="stChatMessage"] {
            background: transparent;
            border: 0;
            padding: 0.18rem 0 !important;
        }
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            width: fit-content;
            max-width: 82%;
            padding: 0.62rem 0.78rem;
            border-radius: 18px;
            line-height: 1.35;
            box-shadow: 0 10px 32px rgba(0,0,0,0.18);
        }
        div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
            margin-left: auto;
            background: rgba(237, 242, 255, 0.88);
            color: #081018;
            border-bottom-right-radius: 6px;
        }
        div[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) [data-testid="stMarkdownContainer"] {
            background: rgba(255,255,255,0.09);
            color: rgba(255,255,255,0.92);
            border: 1px solid rgba(255,255,255,0.08);
            border-bottom-left-radius: 6px;
        }
        [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
            display: none !important;
        }
        [data-testid="stChatInput"] {
            background: rgba(11, 15, 20, 0.45);
            backdrop-filter: blur(18px);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 18px;
        }
        @media (max-width: 900px) {
            html, body, .stApp, [data-testid="stAppViewContainer"], .block-container {overflow: auto !important; height: auto !important;}
            [data-testid="stHorizontalBlock"] {display: block !important;}
            [data-testid="stHorizontalBlock"] > div:nth-of-type(1), [data-testid="stHorizontalBlock"] > div:nth-of-type(2) {
                min-width: 0 !important;
                max-width: none !important;
                width: 100% !important;
            }
            [data-testid="stHorizontalBlock"] > div:nth-of-type(2) > [data-testid="stVerticalBlock"] {margin-top: 0.75rem; max-height: none;}
            .surgical-video-shell img {height: 58vh;}
            .surgical-video-shell {height: auto; padding: 4.5rem 1rem 1rem 1rem;}
            .video-topbar {right: 1rem;}
            .overlay-key-card {max-width: calc(100vw - 2rem);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load API keys quietly
    cfg = load_app_config(str(resolve_repo_path("config/app_config.yaml")))
    groq_key = cfg.get("vlm", {}).get("api_key", "")
    roboflow_key = cfg.get("roboflow_laparoscopy", {}).get("api_key", "")

    initial_video = cfg["replayer"]["basename"].split("/")[0] if "/" in cfg["replayer"]["basename"] else cfg["replayer"]["basename"]

    state, server = start_video_server("config/app_config.yaml", groq_key, roboflow_key, initial_video)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_reported_overlay_generation" not in st.session_state:
        st.session_state.last_reported_overlay_generation = 0
    if "last_reported_chat_generation" not in st.session_state:
        st.session_state.last_reported_chat_generation = 0

    converted_dir = resolve_repo_path(cfg["replayer"]["directory"])
    video_options = [d.name for d in converted_dir.iterdir() if d.is_dir() and (d / "frames").exists()]
    if not video_options:
        video_options = [initial_video]
    if state.target_video != initial_video:
        state.target_video = initial_video
    selected_video = initial_video
    clear_overlay_label = st.query_params.get("clear_overlay")
    if clear_overlay_label:
        label_to_clear = clear_overlay_label[0] if isinstance(clear_overlay_label, list) else clear_overlay_label
        state.remove_overlay_labels([str(label_to_clear)])
        state.append_log_event(
            "overlay_cleared",
            frame_idx=state.last_source_frame_idx,
            prompt="overlay key toggle",
            removed_targets=[str(label_to_clear)],
        )
        del st.query_params["clear_overlay"]
        st.rerun()

    # Layout: video-first workspace with a translucent copilot rail.
    col1, col2 = st.columns([7, 3], gap="small")

    with col1:
        _, pipeline_status, pipeline_error = state.get_pipeline_snapshot()
        active_overlay_query, active_target_labels, overlay_status, overlay_error, overlay_job_running, _, _, _ = state.get_overlay_snapshot()
        pipeline, _, _ = state.get_pipeline_snapshot()
        key_items = state.get_overlay_key_items()
        key_rows = []
        for item in key_items:
            rgba = overlay_color_for_label(item["label"], cfg, pipeline)
            key_rows.append(
                "<div class='overlay-key-row'>"
                f"<span class='overlay-swatch' style='background:{rgba_to_css(rgba)}'></span>"
                f"<span>{item['name']}</span>"
                f"<a class='overlay-toggle' href='?clear_overlay={quote(item['label'])}'>on</a>"
                "</div>"
            )
        key_html = "".join(key_rows) if key_rows else "<div class='overlay-mode'>No active highlights</div>"
        status_text = "Pipeline initializing"
        if pipeline_status == "initializing":
            status_text = "AR pipeline initializing; raw video continues"
        elif pipeline_status == "degraded":
            status_text = f"Video active; AR degraded: {pipeline_error}"
        else:
            status_text = "Live AR pipeline ready"
        if active_overlay_query:
            if overlay_status == "running" or overlay_job_running:
                status_text = f"Overlay updating: {', '.join(humanize_overlay_label(label) for label in active_target_labels) or active_overlay_query}"
            elif overlay_status == "active":
                status_text = f"Overlay active: {', '.join(humanize_overlay_label(label) for label in active_target_labels) or active_overlay_query}"
            elif overlay_status == "ungrounded":
                status_text = f"Overlay not grounded yet: {active_overlay_query}"
            elif overlay_status == "error":
                status_text = f"Overlay error for '{active_overlay_query}': {overlay_error}"

        video_src = f"http://127.0.0.1:8503/video_feed?video={selected_video}"
        st.markdown(
            "<div class='surgical-video-shell'>"
            "<div class='video-topbar'>"
            f"<span>{selected_video}</span>"
            f"<span>{status_text}</span>"
            "</div>"
            f'<img src="{video_src}">'
            "<div class='overlay-key-card'>"
            "<div class='overlay-key-title'>Highlight Key</div>"
            f"{key_html}"
            "<div class='overlay-mode'>Click an on toggle to hide that highlight.</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("<div class='copilot-title'>Surgery Copilot</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='glass-control'>Live feed: {humanize_overlay_label(selected_video)}</div>",
            unsafe_allow_html=True,
        )

        active_overlay_query, active_target_labels, overlay_status, overlay_error, _, overlay_generation, overlay_completed_generation, _ = state.get_overlay_snapshot()
        if (
            overlay_completed_generation > st.session_state.last_reported_overlay_generation
            and overlay_completed_generation == overlay_generation
        ):
            if overlay_status == "ungrounded":
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": "I cannot confidently ground that highlight yet.",
                    }
                )
            elif overlay_status == "error":
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"Highlight failed: {overlay_error}",
                    }
                )
            st.session_state.last_reported_overlay_generation = overlay_completed_generation
            if overlay_status in {"ungrounded", "error"}:
                st.rerun()

        chat_status, chat_active_query, chat_error, chat_response, _, chat_result_generation = state.get_chat_snapshot()
        if chat_result_generation > st.session_state.last_reported_chat_generation:
            if chat_status == "done":
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": chat_response or "Scene analysis completed.",
                    }
                )
            elif chat_status == "error":
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": f"Scene analysis failed for **'{chat_active_query}'**: {chat_error}",
                    }
                )
            st.session_state.last_reported_chat_generation = chat_result_generation
            st.rerun()

        # Keep the full structured scene readout available, but collapsed during live use.
        if state.latest_scene_analysis:
            with st.expander("Scene Details", expanded=False):
                st.markdown(state.latest_scene_analysis.get("scene_summary", ""))
                st.caption(f"Status: {state.latest_scene_analysis.get('workflow_phase', '').replace('_', ' ')}")

        surgery_log, log_path, event_count = state.get_surgery_log_snapshot()
        report_status, report_error, report_text, report_path, _ = state.get_report_snapshot()
        with st.expander("Surgery Session Log", expanded=False):
            if surgery_log is None:
                st.caption("Surgery logging is disabled or failed to initialize.")
            else:
                st.caption(f"Session: {surgery_log.session_id}")
                st.caption(f"Events logged: {event_count}")
                st.caption(f"Log file: {log_path}")
            if st.button("End Surgery & Generate Report", disabled=report_status == "running"):
                start_report_worker(state=state, cfg=cfg)
                st.rerun()
            if report_status == "running":
                st.caption("Generating draft surgery report from the session log and keyframes...")
            elif report_status == "error":
                st.error(f"Report generation failed: {report_error}")
            elif report_status == "done":
                st.caption(f"Draft report saved: {report_path}")
                st.markdown(report_text)

        st.markdown(render_chat_history_html(st.session_state.chat_history), unsafe_allow_html=True)

        prompt = st.chat_input("Command the AR (e.g. 'Highlight the liver')")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            active_overlay_query, active_target_labels, _, _, _, _, _, _ = state.get_overlay_snapshot()
            overlay_command = parse_overlay_command(prompt, cfg, active_target_labels)
            source_frame, source_frame_idx = state.get_last_source_frame_snapshot()
            state.append_log_event(
                "user_question",
                frame_idx=source_frame_idx,
                user_text=prompt,
                active_overlay_targets=active_target_labels,
                requested_overlay_targets=overlay_command.target_labels,
                requested_overlay_removals=overlay_command.remove_labels,
            )
            state.save_keyframe(
                source_frame,
                frame_idx=source_frame_idx,
                reason="user_question",
                metadata={"user_text": prompt},
            )
            if overlay_command.clear_all:
                state.clear_overlay()
                state.append_log_event(
                    "overlay_cleared",
                    frame_idx=source_frame_idx,
                    prompt=prompt,
                    removed_targets=overlay_command.remove_labels or active_target_labels,
                )
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": "Overlay cleared.",
                    }
                )
            elif is_overlay_update_query(prompt):
                target_labels = overlay_command.target_labels
                if not target_labels:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "I need a visible structure or region to highlight.",
                        }
                    )
                else:
                    state.start_overlay_query(
                        prompt,
                        target_labels,
                        remove_labels=overlay_command.remove_labels,
                        mode="persistent",
                    )
                    state.append_log_event(
                        "overlay_requested",
                        frame_idx=source_frame_idx,
                        prompt=prompt,
                        overlay_targets=target_labels,
                        removed_targets=overlay_command.remove_labels,
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"Highlighting {', '.join(humanize_overlay_label(label) for label in target_labels)}.",
                        }
                    )
            else:
                pipeline, pipeline_status, pipeline_error = state.get_pipeline_snapshot()
                if pipeline is None or pipeline_status == "initializing" or source_frame is None:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "Scene analysis is still starting.",
                        }
                    )
                elif pipeline.scene_copilot is None:
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "Scene analysis is unavailable.",
                        }
                    )
                else:
                    start_scene_query_worker(
                        state=state,
                        pipeline=pipeline,
                        frame_idx=source_frame_idx,
                        query=prompt,
                        source_bgr=source_frame,
                        conversation_history=st.session_state.chat_history[-12:],
                        temporary_highlight_seconds=float(get_live_overlay_cfg(cfg).get("short_term_seconds", 6.0)),
                    )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": "Looking now.",
                        }
                    )
            st.rerun()

    report_status, _, _, _, _ = state.get_report_snapshot()
    if overlay_job_running or chat_status == "running" or report_status == "running":
        time.sleep(0.25)
        st.rerun()

if __name__ == "__main__":
    main()
