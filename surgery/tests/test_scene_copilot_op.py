import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.scene_copilot_op import SurgicalSceneCopilot
from operators.yolo_detection_op import Detection


def _frame():
    return torch.zeros(64, 64, 3, dtype=torch.uint8)


def _analyze(query, detections, masks=None):
    copilot = SurgicalSceneCopilot(
        enabled=True,
        provider="rule_based",
        user_query=query,
        refresh_every_n_frames=1,
        max_history_frames=10,
        conservative_mode=True,
    )
    return copilot.analyze(_frame(), detections=detections, masks=masks or {}, frame_idx=0)


def test_scene_copilot_generates_mode_grounded_workflow_analysis():
    analysis = _analyze(
        "What is happening here?",
        [
            Detection("gallbladder", [0, 0, 10, 10], 0.9),
            Detection("liver", [10, 10, 20, 20], 0.8),
            Detection("grasper", [20, 20, 30, 30], 0.85),
        ],
        masks={"gallbladder": torch.ones(64, 64), "liver": torch.ones(64, 64)},
    )

    assert analysis.assistant_mode == "workflow_status"
    assert "Operative Impression" in analysis.surgeon_response
    assert "workflow_status" not in analysis.surgeon_response
    assert "gallbladder" in analysis.visible_structures
    assert "liver" in analysis.visible_structures
    assert "grasper" in analysis.visible_tools
    assert analysis.workflow_phase in {
        "exposure_and_retraction",
        "calot_dissection",
        "clipping_or_cutting_preparation",
        "uncertain",
    }


def test_scene_copilot_uses_temporal_stability_for_structure_visibility():
    copilot = SurgicalSceneCopilot(
        enabled=True,
        provider="rule_based",
        user_query="Summarize the scene.",
        refresh_every_n_frames=1,
        max_history_frames=5,
    )

    for frame_idx in range(3):
        detections = [Detection("gallbladder", [0, 0, 10, 10], 0.9)]
        copilot.analyze(_frame(), detections=detections, masks={}, frame_idx=frame_idx)

    analysis = copilot.analyze(_frame(), detections=[], masks={}, frame_idx=3)

    assert "gallbladder" in analysis.visible_structures


def test_scene_copilot_returns_cached_analysis_between_refreshes():
    copilot = SurgicalSceneCopilot(
        enabled=True,
        provider="rule_based",
        user_query="Summarize the scene.",
        refresh_every_n_frames=3,
        max_history_frames=10,
    )
    detections = [Detection("hook", [0, 0, 10, 10], 0.88)]

    first = copilot.analyze(_frame(), detections=detections, masks={}, frame_idx=0)
    second = copilot.analyze(_frame(), detections=detections, masks={}, frame_idx=1)

    assert first.refreshed is True
    assert second.refreshed is False
    assert second.scene_summary == first.scene_summary
    assert second.surgeon_response == first.surgeon_response
    assert second.assistant_mode == first.assistant_mode


def test_scene_copilot_tool_question_routes_to_instrument_activity_mode():
    analysis = _analyze(
        "Explain what the surgery tool is and what it is doing.",
        [
            Detection("grasper", [20, 20, 30, 30], 0.85),
            Detection("gallbladder", [0, 0, 10, 10], 0.9),
        ],
    )

    assert analysis.assistant_mode == "instrument_activity"
    assert "Instrument Impression" in analysis.surgeon_response
    assert "grasper" in analysis.surgeon_response.lower()
    assert "cystic duct" not in analysis.surgeon_response.lower()


def test_scene_copilot_cvs_question_fails_closed_when_grounding_is_incomplete():
    analysis = _analyze(
        "Is the critical view of safety established here?",
        [
            Detection("gallbladder", [0, 0, 10, 10], 0.9),
            Detection("hepatocystic_triangle", [10, 10, 20, 20], 0.8),
        ],
    )

    assert analysis.assistant_mode == "uncertainty"
    assert "Safety check cannot be concluded confidently" in analysis.surgeon_response


def test_scene_copilot_structure_question_returns_indeterminate_without_substitution():
    analysis = _analyze(
        "Can you assess the liver in this frame?",
        [
            Detection("gallbladder", [0, 0, 10, 10], 0.9),
        ],
    )

    assert analysis.assistant_mode == "anatomy_verification"
    assert "liver" in analysis.surgeon_response.lower()
    assert "gallbladder" not in analysis.surgeon_response.lower()
    assert analysis.mode_payload["verification_status"] == "indeterminate"


def test_scene_copilot_structure_report_mode_stays_evidence_limited():
    analysis = _analyze(
        "Give a full report on the gallbladder.",
        [
            Detection("gallbladder", [0, 0, 10, 10], 0.9),
            Detection("hook", [10, 10, 20, 20], 0.8),
        ],
    )

    assert analysis.assistant_mode == "structure_report"
    assert "Requested Structure" in analysis.surgeon_response
    assert "gallbladder" in analysis.surgeon_response.lower()
    assert "cystic duct" not in analysis.surgeon_response.lower()
    assert "cystic artery" not in analysis.surgeon_response.lower()
    assert "full diagnosis cannot be made" in analysis.surgeon_response.lower()


def test_scene_copilot_overlay_mode_reports_only_grounded_mask_targets():
    analysis = _analyze(
        "Highlight the liver.",
        [
            Detection("liver", [0, 0, 10, 10], 0.9),
        ],
        masks={"liver": torch.ones(64, 64)},
    )

    assert analysis.assistant_mode == "overlay_update"
    assert analysis.mode_payload["updated_targets"] == ["liver"]
    assert "Overlay Update" in analysis.surgeon_response


def test_scene_copilot_general_scene_mode_remains_distinct_from_structure_report():
    workflow_analysis = _analyze(
        "What is happening here?",
        [
            Detection("gallbladder", [0, 0, 10, 10], 0.9),
            Detection("hook", [10, 10, 20, 20], 0.8),
        ],
    )
    report_analysis = _analyze(
        "Give a full report on the gallbladder.",
        [
            Detection("gallbladder", [0, 0, 10, 10], 0.9),
            Detection("hook", [10, 10, 20, 20], 0.8),
        ],
    )

    assert workflow_analysis.assistant_mode == "workflow_status"
    assert report_analysis.assistant_mode == "structure_report"
    assert workflow_analysis.surgeon_response != report_analysis.surgeon_response


def test_scene_copilot_openai_provider_uses_vlm_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"surgeon_response":"The liver edge and a grasper are visible. '
                                'I cannot confirm critical view structures from this frame alone.",'
                                '"clinical_log_note":"AI log: liver and grasper visible; CVS not confirmed.",'
                                '"scene_summary":"Liver and grasper visible.",'
                                '"observed_risks":[],"uncertainties":["CVS not confirmed"],'
                                '"recommended_attention_targets":["cystic_duct"],"confidence":0.62}'
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr("operators.scene_copilot_op.requests.post", lambda *_args, **_kwargs: FakeResponse())
    copilot = SurgicalSceneCopilot(
        enabled=True,
        provider="openai_compatible",
        user_query="What am I looking at?",
        api_url="https://example.test",
        api_key="key",
        model="model",
        refresh_every_n_frames=1,
    )

    analysis = copilot.analyze(
        _frame(),
        detections=[
            Detection("liver", [0, 0, 10, 10], 0.9),
            Detection("grasper", [5, 5, 20, 20], 0.8),
        ],
        masks={},
        frame_idx=0,
    )

    assert analysis.provider == "openai_compatible"
    assert "liver edge" in analysis.surgeon_response
    assert "AI log" in analysis.reasoning_summary
    assert analysis.recommended_attention_targets == ["cystic_duct"]
