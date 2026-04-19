from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from session import SurgeryReportGenerator, SurgerySessionLog


def test_surgery_session_log_appends_events_and_keyframes(tmp_path):
    log = SurgerySessionLog(tmp_path, "clip_002")
    event = log.append_event(
        "user_question",
        frame_idx=12,
        user_text="What is visible?",
        visible_structures=["liver"],
    )

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    keyframe_path = log.save_keyframe(frame, frame_idx=12, reason="user_question")
    events = log.read_events()

    assert event["event_type"] == "user_question"
    assert log.log_path.exists()
    assert keyframe_path is not None
    assert len(events) == 3
    assert events[0]["event_type"] == "session_started"
    assert events[1]["visible_structures"] == ["liver"]
    assert events[2]["event_type"] == "keyframe_saved"


def test_rule_based_report_summarizes_session_events(tmp_path):
    log = SurgerySessionLog(tmp_path, "clip_002")
    log.append_event(
        "frame_observation",
        frame_idx=5,
        visible_structures=["liver", "gallbladder"],
        visible_tools=["grasper"],
        workflow_phase="exposure_and_retraction",
        active_overlay_targets=["liver"],
        uncertainties=["The cystic duct is not confidently visible."],
    )
    log.append_event("user_question", frame_idx=6, user_text="Is this the gallbladder?")

    report = SurgeryReportGenerator(provider="rule_based").generate(
        events=log.read_events(),
        keyframe_paths=[],
        metadata={"video_name": "clip_002"},
    )

    assert "AI-Generated Draft Surgery Documentation" in report
    assert "liver" in report
    assert "gallbladder" in report
    assert "grasper" in report
    assert "Is this the gallbladder?" in report


def test_vlm_report_falls_back_when_request_fails(tmp_path, monkeypatch):
    log = SurgerySessionLog(tmp_path, "clip_002")
    log.append_event("frame_observation", frame_idx=1, visible_structures=["liver"])

    def failing_post(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("session.surgery_report.requests.post", failing_post)
    report = SurgeryReportGenerator(
        provider="openai_compatible",
        api_url="https://example.test",
        api_key="key",
        model="model",
    ).generate(events=log.read_events(), keyframe_paths=[], metadata={"video_name": "clip_002"})

    assert "structured fallback" in report
    assert "network down" in report


def test_vlm_report_retries_text_only_after_vision_400(tmp_path, monkeypatch):
    log = SurgerySessionLog(tmp_path, "clip_002")
    log.append_event("frame_observation", frame_idx=1, visible_structures=["liver"])
    frame_path = tmp_path / "keyframe.jpg"
    cv2.imwrite(str(frame_path), np.zeros((32, 32, 3), dtype=np.uint8))

    calls = []

    class FakeResponse:
        def __init__(self, *, status_code=200, reason="OK", body=None):
            self.status_code = status_code
            self.reason = reason
            self.text = body or ""

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError("bad request", response=self)

        def json(self):
            return {"choices": [{"message": {"content": "Text-only VLM report"}}]}

    def fake_post(*_args, **kwargs):
        calls.append(kwargs["json"]["messages"][1]["content"])
        if len(calls) == 1:
            return FakeResponse(status_code=400, reason="Bad Request", body='{"error":"vision payload rejected"}')
        return FakeResponse()

    monkeypatch.setattr("session.surgery_report.requests.post", fake_post)
    report = SurgeryReportGenerator(
        provider="openai_compatible",
        api_url="https://example.test",
        api_key="key",
        model="model",
    ).generate(events=log.read_events(), keyframe_paths=[str(frame_path)], metadata={"video_name": "clip_002"})

    assert report == "Text-only VLM report"
    assert len(calls) == 2
    assert isinstance(calls[0], list)
    assert isinstance(calls[1], str)
