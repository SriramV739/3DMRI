"""
test_groq_vlm.py - Smoke tests for the Groq Llama 4 Scout VLM integration.

Run with:
    pytest tests/test_groq_vlm.py -v -s

For live API tests, set GROQ_API_KEY in the environment.
Tests that require the API key are skipped when it is absent.
"""

from __future__ import annotations

import json
import os
import textwrap
from unittest import mock

import numpy as np
import pytest

from operators.vlm_prompt_op import AnatomyVLMGuide, PromptSelection
from operators.yolo_detection_op import Detection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

CANDIDATE_LABELS = [
    "gallbladder",
    "cystic_duct",
    "cystic_artery",
    "cystic_plate",
    "hepatocystic_triangle",
    "liver",
]


def _make_guide(provider: str = "openai_compatible", **overrides) -> AnatomyVLMGuide:
    defaults = dict(
        enabled=True,
        provider=provider,
        user_query="Focus on the gallbladder, cystic duct, and hepatocystic triangle.",
        candidate_labels=CANDIDATE_LABELS,
        anatomy_aliases={
            "gallbladder": ["gb", "gall bladder"],
            "cystic_duct": ["duct", "cystic duct"],
            "hepatocystic_triangle": ["calot", "calots triangle"],
        },
        prompt_every_n_frames=1,
        max_image_size=512,
        api_url=GROQ_API_URL,
        api_key=GROQ_API_KEY,
        api_key_env="GROQ_API_KEY",
        model=GROQ_MODEL,
    )
    defaults.update(overrides)
    return AnatomyVLMGuide(**defaults)


def _make_frame(h: int = 480, w: int = 854) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_detections() -> list[Detection]:
    return [
        Detection(class_name="gallbladder", bbox=[100, 80, 250, 200], confidence=0.9, class_id=0),
        Detection(class_name="cystic_duct", bbox=[260, 120, 350, 190], confidence=0.7, class_id=1),
        Detection(class_name="grasper", bbox=[400, 300, 500, 420], confidence=0.85, class_id=2),
    ]


# ---------------------------------------------------------------------------
# Unit tests (no API key required)
# ---------------------------------------------------------------------------

class TestGroqConfigValidation:
    """Ensure the VLM raises clear errors when Groq config is incomplete."""

    def test_missing_api_key_gives_helpful_message(self):
        guide = _make_guide(api_key="", api_url=GROQ_API_URL, model=GROQ_MODEL)
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            guide._infer_with_openai_compatible(frame, _make_detections(), frame_idx=0)

    def test_missing_model_gives_helpful_message(self):
        guide = _make_guide(api_key="test-key", model="")
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="model"):
            guide._infer_with_openai_compatible(frame, _make_detections(), frame_idx=0)

    def test_missing_api_url_gives_helpful_message(self):
        guide = _make_guide(api_key="test-key", api_url="")
        frame = _make_frame()
        with pytest.raises(RuntimeError, match="api_url"):
            guide._infer_with_openai_compatible(frame, _make_detections(), frame_idx=0)


class TestGroqResponseParsing:
    """Test JSON response parsing, including edge cases from LLMs."""

    def _mock_groq_response(self, content: str, status_code: int = 200):
        """Create a mock requests.post response for Groq."""
        mock_resp = mock.MagicMock()
        mock_resp.status_code = status_code
        mock_resp.headers = {}
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_resp.raise_for_status = mock.MagicMock()
        return mock_resp

    @mock.patch("operators.vlm_prompt_op.requests.post")
    def test_clean_json_response(self, mock_post):
        content = json.dumps({
            "target_labels": ["gallbladder", "cystic_duct"],
            "rationale": "Query asks for gallbladder and cystic duct."
        })
        mock_post.return_value = self._mock_groq_response(content)

        guide = _make_guide(api_key="test-key")
        labels, rationale = guide._infer_with_openai_compatible(
            _make_frame(), _make_detections(), frame_idx=0
        )

        assert "gallbladder" in labels
        assert "cystic_duct" in labels
        assert "hepatocystic_triangle" in labels
        assert "gallbladder" in rationale.lower() or len(rationale) > 0

    @mock.patch("operators.vlm_prompt_op.requests.post")
    def test_markdown_wrapped_json_response(self, mock_post):
        """Llama models sometimes wrap JSON in markdown code fences."""
        content = textwrap.dedent("""\
            ```json
            {
              "target_labels": ["gallbladder", "hepatocystic_triangle"],
              "rationale": "Surgeon wants Calot's triangle and gallbladder."
            }
            ```""")
        mock_post.return_value = self._mock_groq_response(content)

        guide = _make_guide(api_key="test-key")
        labels, rationale = guide._infer_with_openai_compatible(
            _make_frame(), _make_detections(), frame_idx=0
        )

        assert "gallbladder" in labels
        assert "hepatocystic_triangle" in labels

    @mock.patch("operators.vlm_prompt_op.requests.post")
    def test_invalid_labels_filtered_out(self, mock_post):
        """Labels not in candidate_labels should be silently dropped."""
        content = json.dumps({
            "target_labels": ["gallbladder", "some_fake_structure", "cystic_duct"],
            "rationale": "test"
        })
        mock_post.return_value = self._mock_groq_response(content)

        guide = _make_guide(api_key="test-key")
        labels, _ = guide._infer_with_openai_compatible(
            _make_frame(), _make_detections(), frame_idx=0
        )

        assert "gallbladder" in labels
        assert "cystic_duct" in labels
        assert "some_fake_structure" not in labels

    @mock.patch("operators.vlm_prompt_op.requests.post")
    def test_localize_prompt_boxes_accepts_freeform_label(self, mock_post):
        """VLM box localization can ground labels outside the detector ontology."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "boxes": [
                                    {
                                        "label": "bleeding_tissue_edge",
                                        "bbox": [10, 20, 120, 160],
                                        "confidence": 0.55,
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        guide = _make_guide(api_key="key")

        detections = guide.localize_prompt_boxes(
            _make_frame(),
            ["bleeding_tissue_edge"],
            existing_detections=[],
            frame_idx=3,
        )

        assert len(detections) == 1
        assert detections[0].class_name == "bleeding_tissue_edge"
        assert detections[0].source_model == "vlm_prompt_box"

    @mock.patch("operators.vlm_prompt_op.requests.post")
    def test_malformed_json_returns_empty(self, mock_post):
        """Graceful degradation when VLM returns garbage."""
        mock_post.return_value = self._mock_groq_response("not json at all {{{")

        guide = _make_guide(api_key="test-key")
        labels, rationale = guide._infer_with_openai_compatible(
            _make_frame(), _make_detections(), frame_idx=0
        )

        assert labels == []
        assert "parse error" in rationale.lower()


class TestGroqRetryLogic:
    """Test rate-limit retry behavior."""

    @mock.patch("operators.vlm_prompt_op.time.sleep")
    @mock.patch("operators.vlm_prompt_op.requests.post")
    def test_429_retry_then_success(self, mock_post, mock_sleep):
        rate_limited = mock.MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"retry-after": "0.1"}

        success_content = json.dumps({
            "target_labels": ["gallbladder"],
            "rationale": "Retried successfully."
        })
        success = mock.MagicMock()
        success.status_code = 200
        success.headers = {}
        success.json.return_value = {"choices": [{"message": {"content": success_content}}]}
        success.raise_for_status = mock.MagicMock()

        mock_post.side_effect = [rate_limited, success]

        guide = _make_guide(api_key="test-key")
        labels, _ = guide._infer_with_openai_compatible(
            _make_frame(), _make_detections(), frame_idx=0
        )

        assert "gallbladder" in labels
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once()

    @mock.patch("operators.vlm_prompt_op.time.sleep")
    @mock.patch("operators.vlm_prompt_op.requests.post")
    def test_all_retries_exhausted_raises(self, mock_post, mock_sleep):
        rate_limited = mock.MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {}

        mock_post.return_value = rate_limited

        guide = _make_guide(api_key="test-key")

        with pytest.raises(RuntimeError, match="failed after"):
            guide._infer_with_openai_compatible(
                _make_frame(), _make_detections(), frame_idx=0
            )


class TestRuleBasedFallback:
    """Verify the rule_based provider still works as an offline fallback."""

    def test_rule_based_selects_from_query(self):
        guide = _make_guide(provider="rule_based")
        selection = guide.select_prompts(_make_frame(), _make_detections(), frame_idx=0)

        assert isinstance(selection, PromptSelection)
        assert "gallbladder" in selection.target_labels
        assert "cystic_duct" in selection.target_labels
        assert "hepatocystic_triangle" in selection.target_labels
        # The grasper detection should be filtered out
        assert all(d.class_name != "grasper" for d in selection.filtered_detections)


class TestBase64SizeValidation:
    """Ensure the frame encoder respects image size limits."""

    def test_small_frame_stays_under_4mb(self):
        guide = _make_guide(api_key="test-key")
        frame = _make_frame(h=512, w=512)
        b64 = guide._frame_to_base64(frame)
        size_mb = len(b64) / (1024 * 1024)
        assert size_mb < 4.0, f"Base64 image is {size_mb:.1f} MB, exceeds Groq 4 MB limit"

    def test_large_frame_gets_resized(self):
        guide = _make_guide(api_key="test-key", max_image_size=256)
        frame = _make_frame(h=1080, w=1920)
        b64 = guide._frame_to_base64(frame)
        size_mb = len(b64) / (1024 * 1024)
        assert size_mb < 1.0, f"Resized image should be well under 1 MB, got {size_mb:.1f} MB"


# ---------------------------------------------------------------------------
# Live API test (requires GROQ_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
class TestGroqLiveAPI:
    """Live integration test against the real Groq API. Requires GROQ_API_KEY."""

    def test_live_groq_llama4_scout(self):
        guide = _make_guide()
        frame = _make_frame()
        detections = _make_detections()

        labels, rationale = guide._infer_with_openai_compatible(
            frame, detections, frame_idx=0
        )

        print(f"\n  Groq response labels: {labels}")
        print(f"  Groq rationale: {rationale}")

        # The model should select at least one anatomy label
        assert len(labels) > 0, "Groq returned no target labels"
        assert all(label in CANDIDATE_LABELS for label in labels)

    def test_live_full_select_prompts_flow(self):
        guide = _make_guide()
        frame = _make_frame()
        detections = _make_detections()

        selection = guide.select_prompts(frame, detections, frame_idx=0)

        print(f"\n  Full flow labels: {selection.target_labels}")
        print(f"  Full flow rationale: {selection.rationale}")
        print(f"  Filtered detections: {[d.class_name for d in selection.filtered_detections]}")

        assert isinstance(selection, PromptSelection)
        assert selection.provider == "openai_compatible"
"""
Description: Comprehensive test suite for the Groq Llama 4 Scout VLM integration.
Covers config validation, response parsing (including markdown fences), retry logic,
base64 size limits, rule_based fallback, and live API tests (when GROQ_API_KEY is set).
"""
