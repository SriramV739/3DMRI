"""Append-only surgery session logging for the live Streamlit demo."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _json_safe(value: Any) -> Any:
    """Convert common runtime values into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class SurgerySessionLog:
    """Manage one append-only JSONL log and keyframe folder for a surgery session."""

    def __init__(self, root_dir: str | Path, video_name: str):
        self.root_dir = Path(root_dir)
        self.video_name = video_name
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_video = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in video_name)
        self.session_id = f"{stamp}_{safe_video}"
        self.session_dir = self.root_dir / self.session_id
        self.keyframe_dir = self.session_dir / "keyframes"
        self.log_path = self.session_dir / "events.jsonl"
        self.event_count = 0

        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.append_event(
            "session_started",
            video_name=video_name,
            session_dir=str(self.session_dir),
        )

    def append_event(self, event_type: str, **payload: Any) -> Dict[str, Any]:
        event = {
            "event_id": self.event_count + 1,
            "event_type": event_type,
            "session_id": self.session_id,
            "video_name": self.video_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time": time.time(),
            **_json_safe(payload),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        self.event_count += 1
        return event

    def save_keyframe(
        self,
        frame_bgr,
        *,
        frame_idx: int,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if frame_bgr is None:
            return None
        try:
            import cv2

            frame_path = self.keyframe_dir / f"frame_{frame_idx:06d}_{reason}.jpg"
            ok = cv2.imwrite(str(frame_path), frame_bgr)
            if not ok:
                return None
            self.append_event(
                "keyframe_saved",
                frame_idx=frame_idx,
                reason=reason,
                frame_path=str(frame_path),
                metadata=metadata or {},
            )
            return str(frame_path)
        except Exception as exc:
            self.append_event(
                "keyframe_save_failed",
                frame_idx=frame_idx,
                reason=reason,
                error=str(exc),
            )
            return None

    def read_events(self) -> list[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        events: list[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def keyframes(self) -> list[str]:
        if not self.keyframe_dir.exists():
            return []
        return [str(path) for path in sorted(self.keyframe_dir.glob("*.jpg"))]

    def append_frame_observation(
        self,
        *,
        frame_idx: int,
        active_overlay_targets: Iterable[str],
        visible_structures: Iterable[str] = (),
        visible_tools: Iterable[str] = (),
        workflow_phase: str = "",
        uncertainties: Iterable[str] = (),
    ) -> Dict[str, Any]:
        return self.append_event(
            "frame_observation",
            frame_idx=frame_idx,
            active_overlay_targets=list(active_overlay_targets),
            visible_structures=list(visible_structures),
            visible_tools=list(visible_tools),
            workflow_phase=workflow_phase,
            uncertainties=list(uncertainties),
        )
