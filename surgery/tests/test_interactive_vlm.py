import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.interactive_vlm import (
    build_visible_response_text,
    is_overlay_update_query,
    parse_overlay_command,
    resolve_overlay_target_labels,
)


def _cfg():
    return {
        "vlm": {
            "candidate_labels": [
                "gallbladder",
                "cystic_duct",
                "liver",
                "grasper",
                "hook",
                "clipper",
            ],
            "anatomy_aliases": {
                "gallbladder": ["gall bladder"],
                "cystic_duct": ["duct", "cystic duct"],
                "liver": ["liver"],
                "grasper": ["grasper", "tool", "instrument"],
                "hook": ["hook", "cautery"],
                "clipper": ["clipper"],
            },
        }
    }


def test_non_overlay_question_does_not_mark_overlay_targets():
    result = {
        "selected_labels": ["gallbladder"],
        "overlay_updated": False,
    }

    response = build_visible_response_text(result, scene_analysis=None)

    assert "Overlay targets" not in response
    assert "unchanged for this question" in response


def test_overlay_query_detection_is_strict_to_highlight_intent():
    assert is_overlay_update_query("Highlight the liver") is True
    assert is_overlay_update_query("Can you give a diagnostic on the liver?") is False


def test_overlay_switch_prompt_removes_old_label_and_targets_new_label():
    command = parse_overlay_command(
        "Perfect, now turn off the liver segmentation and highlight the gallbladder",
        _cfg(),
        active_labels=["liver"],
    )

    assert command.remove_labels == ["liver"]
    assert command.target_labels == ["gallbladder"]
    assert command.clear_all is False


def test_overlay_switch_prompt_handles_gallbladder_typo():
    command = parse_overlay_command(
        "Perfect, now turn off the liver segmentation and highlight the galbladder",
        _cfg(),
        active_labels=["liver"],
    )

    assert command.remove_labels == ["liver"]
    assert command.target_labels == ["gallbladder"]


def test_overlay_switch_from_to_targets_destination_only():
    assert resolve_overlay_target_labels("switch from liver to gallbladder", _cfg()) == ["gallbladder"]


def test_overlay_remove_only_clears_last_active_target():
    command = parse_overlay_command("remove liver overlay", _cfg(), active_labels=["liver"])

    assert command.remove_labels == ["liver"]
    assert command.target_labels == []
    assert command.clear_all is True


def test_overlay_remove_only_keeps_remaining_active_targets():
    command = parse_overlay_command("turn off liver", _cfg(), active_labels=["liver", "grasper"])

    assert command.remove_labels == ["liver"]
    assert command.target_labels == ["grasper"]
    assert command.clear_all is False


def test_overlay_tool_alias_targets_grasper_for_demo():
    assert resolve_overlay_target_labels("highlight the surgical tool", _cfg()) == ["grasper"]


def test_overlay_freeform_target_is_allowed_for_vlm_prompt_box():
    command = parse_overlay_command("highlight the bleeding tissue edge", _cfg())

    assert command.target_labels == ["bleeding_tissue_edge"]
