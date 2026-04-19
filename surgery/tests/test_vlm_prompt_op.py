import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from operators.vlm_prompt_op import AnatomyVLMGuide
from operators.yolo_detection_op import Detection


def test_rule_based_query_selects_specific_anatomy():
    guide = AnatomyVLMGuide(
        enabled=True,
        provider="rule_based",
        user_query="Focus on the gallbladder and cystic duct only.",
        candidate_labels=["gallbladder", "cystic_duct", "liver"],
        anatomy_aliases={
            "gallbladder": ["gall bladder"],
            "cystic_duct": ["duct", "cystic duct"],
            "liver": ["liver"],
        },
    )

    frame = torch.zeros(64, 64, 3, dtype=torch.uint8)
    detections = [
        Detection("gallbladder", [0, 0, 10, 10], 0.9),
        Detection("cystic_duct", [10, 10, 20, 20], 0.8),
        Detection("liver", [20, 20, 30, 30], 0.7),
    ]

    selection = guide.select_prompts(frame, detections, frame_idx=0)
    assert selection.target_labels == ["gallbladder", "cystic_duct"]
    assert [det.class_name for det in selection.filtered_detections] == ["gallbladder", "cystic_duct"]


def test_rule_based_falls_back_to_detected_anatomy_when_query_is_generic():
    guide = AnatomyVLMGuide(
        enabled=True,
        provider="rule_based",
        user_query="Help with the anatomy.",
        candidate_labels=["gallbladder", "cystic_duct", "liver"],
        anatomy_aliases={"gallbladder": ["gallbladder"]},
    )
    frame = torch.zeros(64, 64, 3, dtype=torch.uint8)
    detections = [
        Detection("liver", [0, 0, 10, 10], 0.95),
        Detection("grasper", [5, 5, 15, 15], 0.8),
    ]

    selection = guide.select_prompts(frame, detections, frame_idx=0)
    assert selection.target_labels == ["liver"]
    assert [det.class_name for det in selection.filtered_detections] == ["liver"]


def test_set_query_refreshes_prompt_selection():
    guide = AnatomyVLMGuide(
        enabled=True,
        provider="rule_based",
        user_query="Highlight the liver.",
        candidate_labels=["gallbladder", "liver", "grasper"],
        anatomy_aliases={
            "gallbladder": ["gallbladder"],
            "liver": ["liver"],
            "grasper": ["grasper", "tool"],
        },
        prompt_every_n_frames=10,
    )
    frame = torch.zeros(64, 64, 3, dtype=torch.uint8)
    detections = [
        Detection("liver", [0, 0, 10, 10], 0.95),
        Detection("grasper", [5, 5, 15, 15], 0.8),
    ]

    first = guide.select_prompts(frame, detections, frame_idx=0)
    assert first.target_labels == ["liver"]

    guide.set_query("Wait, highlight tools instead.")
    second = guide.select_prompts(frame, detections, frame_idx=1)

    assert second.target_labels == ["grasper"]
    assert [det.class_name for det in second.filtered_detections] == ["grasper"]


def test_explicit_query_does_not_substitute_other_detected_structure():
    guide = AnatomyVLMGuide(
        enabled=True,
        provider="rule_based",
        user_query="Highlight the liver.",
        candidate_labels=["gallbladder", "liver"],
        anatomy_aliases={
            "gallbladder": ["gallbladder"],
            "liver": ["liver"],
        },
    )
    frame = torch.zeros(64, 64, 3, dtype=torch.uint8)
    detections = [
        Detection("gallbladder", [0, 0, 10, 10], 0.95),
    ]

    selection = guide.select_prompts(frame, detections, frame_idx=0)

    assert selection.target_labels == []
    assert selection.filtered_detections == []
    assert "could not be grounded" in selection.rationale.lower()


def test_explicit_query_uses_requested_label_when_detected():
    guide = AnatomyVLMGuide(
        enabled=True,
        provider="rule_based",
        user_query="Highlight the liver.",
        candidate_labels=["gallbladder", "liver"],
        anatomy_aliases={
            "gallbladder": ["gallbladder"],
            "liver": ["liver"],
        },
    )
    frame = torch.zeros(64, 64, 3, dtype=torch.uint8)
    detections = [
        Detection("gallbladder", [0, 0, 10, 10], 0.95),
        Detection("liver", [10, 10, 20, 20], 0.91),
    ]

    selection = guide.select_prompts(frame, detections, frame_idx=0)

    assert selection.target_labels == ["liver"]
    assert [det.class_name for det in selection.filtered_detections] == ["liver"]
