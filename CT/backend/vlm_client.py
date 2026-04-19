from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from groq import Groq
from google.genai import errors as genai_errors
from google import genai
from PIL import Image, ImageStat

from backend import settings
from backend.imaging.dicom_io import ensure_slice_preview_png


SNAPSHOT_SECTION_HEADINGS = [
    "Direct Answer",
    "Visible Anatomy",
    "Spatial Observations",
    "Uncertainty and Limitations",
    "Suggested Next View or Slice to Inspect",
    "Anatomy Visible",
    "Key Observations",
    "Potential Abnormalities Or Issues",
    "Limitations",
    "Suggested Follow-Up Questions For A Clinician",
]

SNAPSHOT_DETAIL_LABELS = [
    "Airway",
    "Arteries",
    "Arterial System",
    "Veins",
    "Venous System",
    "Lungs",
    "Cardiac Structures",
    "Bones",
    "Vascular Architecture",
    "Venous Convergence",
    "Thoracic Volume",
    "Mediastinal Centering",
    "Anatomical Integration",
    "Scale",
    "Visual Assessment",
    "Pixel-Level Confirmation",
    "Diagnostic Constraints",
    "Anatomical Scope",
    "Non-Diagnostic",
    "Axial Source Slices",
    "Coronal Reformation",
    "Coronal Multiplanar Reconstruction",
    "Radiologist Review",
]


def load_vlm_config() -> dict[str, Any]:
    with settings.VLM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_markdown_sections(text: str) -> str:
    normalized = text.strip()
    for heading in SNAPSHOT_SECTION_HEADINGS:
        pattern = re.compile(rf"(^|(?<=[.!?])\s*|\n+)(?:#+\s*)?{re.escape(heading)}:?\s*", re.IGNORECASE)
        normalized = pattern.sub(lambda match, title=heading: f"\n\n## {title}\n\n", normalized)

        start_pattern = re.compile(rf"^(?:#+\s*)?{re.escape(heading)}:?\s*", re.IGNORECASE)
        normalized = start_pattern.sub(f"## {heading}\n\n", normalized)

    for label in SNAPSHOT_DETAIL_LABELS:
        pattern = re.compile(rf"(^|(?<=[.!?])\s*|\n+){re.escape(label)}:\s*", re.IGNORECASE)
        normalized = pattern.sub(lambda match, title=label: f"\n\n- **{title}:** ", normalized)

    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def _polish_analysis_text(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return "No analysis text was returned."

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _normalize_markdown_sections(text)

    def titleize(value: str) -> str:
        return value.replace("_", " ").strip().title()

    def render_value(value: Any, indent: str = "") -> list[str]:
        if isinstance(value, dict):
            lines: list[str] = []
            for key, child in value.items():
                rendered = render_value(child, indent)
                if len(rendered) == 1 and not rendered[0].startswith("- "):
                    lines.append(f"{indent}- **{titleize(str(key))}:** {rendered[0]}")
                else:
                    lines.append(f"{indent}- **{titleize(str(key))}:**")
                    lines.extend(f"{indent}  {line}" for line in rendered)
            return lines
        if isinstance(value, list):
            if not value:
                return [f"{indent}- None noted."]
            lines = []
            for item in value:
                rendered = render_value(item, indent)
                if len(rendered) == 1:
                    lines.append(f"{indent}- {rendered[0].removeprefix('- ')}")
                else:
                    lines.extend(rendered)
            return lines
        return [str(value)]

    if isinstance(parsed, dict):
        sections: list[str] = []
        for key, value in parsed.items():
            sections.append(f"## {titleize(str(key))}")
            rendered = render_value(value)
            sections.extend(rendered)
            sections.append("")
        return "\n".join(sections).strip()

    if isinstance(parsed, list):
        return "\n".join(render_value(parsed))

    return str(parsed)


def _image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_from_base64(image_base64: str) -> Image.Image:
    payload = image_base64.strip()
    if "," in payload and payload.split(",", 1)[0].startswith("data:image"):
        payload = payload.split(",", 1)[1]
    data = base64.b64decode(payload)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _downscale_for_vlm(image: Image.Image, max_side: int = 960) -> Image.Image:
    width, height = image.size
    largest = max(width, height)
    if largest <= max_side:
        return image
    scale = max_side / float(largest)
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)


def _snapshot_signal(image: Image.Image) -> dict[str, Any]:
    sample = image.convert("L").resize((96, 96), Image.Resampling.BILINEAR)
    stat = ImageStat.Stat(sample)
    mean = float(stat.mean[0])
    stddev = float(stat.stddev[0])
    histogram = sample.histogram()
    total = max(1, sum(histogram))
    very_dark_ratio = sum(histogram[:18]) / total
    very_bright_ratio = sum(histogram[238:]) / total
    uninformative = (mean < 18 and stddev < 18) or very_dark_ratio > 0.92 or very_bright_ratio > 0.92
    return {
        "mean_luma": round(mean, 2),
        "stddev_luma": round(stddev, 2),
        "very_dark_ratio": round(very_dark_ratio, 4),
        "very_bright_ratio": round(very_bright_ratio, 4),
        "uninformative": uninformative,
    }


def _label_group(label: str) -> str:
    lower = label.lower()
    if "artery" in lower or "aorta" in lower or "trunk" in lower:
        return "arteries"
    if "vein" in lower or "vena" in lower:
        return "veins"
    if lower.startswith("lung_"):
        return "lungs"
    if "heart" in lower or "atrial" in lower or "ventricle" in lower:
        return "heart"
    if lower.startswith(("rib_", "vertebrae_")) or lower in {"sternum", "clavicula_left", "clavicula_right", "scapula_left", "scapula_right"}:
        return "bones"
    if lower in {"trachea", "esophagus"}:
        return "airway"
    if lower in {"liver", "spleen", "stomach", "kidney_left", "kidney_right", "pancreas"}:
        return "abdomen"
    return "other"


def _readable_label(label: str) -> str:
    return label.replace("_", " ").title()


def _visible_label_summary(labels: list[str]) -> str:
    if not labels:
        return "No app-visible segmentation labels were provided."
    grouped: dict[str, list[str]] = {}
    for label in labels:
        grouped.setdefault(_label_group(label), []).append(_readable_label(label))
    sections = []
    for group in sorted(grouped):
        values = ", ".join(sorted(grouped[group])[:24])
        extra = len(grouped[group]) - min(len(grouped[group]), 24)
        if extra > 0:
            values += f", plus {extra} more"
        sections.append(f"{group}: {values}")
    return "; ".join(sections)


def ensure_slice_png(slice_id: str) -> Path:
    return ensure_slice_preview_png(slice_id)


def build_vlm_payload(slice_ids: list[str], user_note: str | None = None) -> dict[str, Any]:
    config = load_vlm_config()
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": config["user_prompt_template"].format(
                slice_count=len(slice_ids),
                user_note=user_note or "No additional clinical context provided.",
            ),
        }
    ]
    for slice_id in slice_ids[: int(config.get("max_images", 5))]:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_data_url(ensure_slice_png(slice_id)),
                },
            }
        )

    return {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": content},
        ],
        "temperature": config.get("temperature", 0.2),
        "max_completion_tokens": config.get("max_completion_tokens", 1024),
    }


def analyze_slices(slice_ids: list[str], user_note: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    payload = build_vlm_payload(slice_ids=slice_ids, user_note=user_note)
    payload_preview = {
        **payload,
        "messages": [
            payload["messages"][0],
            {
                "role": "user",
                "content": [
                    item if item["type"] == "text" else {"type": "image_url", "image_url": {"url": "data:image/png;base64,<redacted>"}}
                    for item in payload["messages"][1]["content"]
                ],
            },
        ],
    }

    if dry_run:
        return {
            "mode": "dry_run",
            "slice_ids": slice_ids,
            "model": payload["model"],
            "payload_preview": payload_preview,
            "analysis": "Dry run only. Add GROQ_API_KEY to .env and call with dry_run=false to execute.",
        }

    if not settings.GROQ_API_KEY:
        return {
            "mode": "missing_api_key",
            "slice_ids": slice_ids,
            "model": payload["model"],
            "payload_preview": payload_preview,
            "analysis": "GROQ_API_KEY is not set. Add it to CT/.env to enable live Groq analysis.",
        }

    client = Groq(api_key=settings.GROQ_API_KEY)
    completion = client.chat.completions.create(**payload)
    message = completion.choices[0].message
    return {
        "mode": "live",
        "slice_ids": slice_ids,
        "model": payload["model"],
        "payload_preview": payload_preview,
        "analysis": message.content,
    }


def analyze_slices_gemini(slice_ids: list[str], user_note: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    config = load_vlm_config()
    model = config.get("gemini_model", "gemini-3.1-flash-lite-preview")
    fallback_models = [item for item in config.get("gemini_fallback_models", []) if isinstance(item, str) and item != model]
    prompt = f"{config['system_prompt']}\n\n" + config["user_prompt_template"].format(
        slice_count=len(slice_ids),
        user_note=user_note or "No additional clinical context provided.",
    )
    image_paths = [ensure_slice_png(slice_id) for slice_id in slice_ids[: int(config.get("max_images", 5))]]
    payload_preview = {
        "model": model,
        "contents": [
            {"type": "text", "text": prompt},
            *[{"type": "image", "path": str(path.relative_to(settings.BASE_DIR))} for path in image_paths],
        ],
    }

    if dry_run:
        return {
            "mode": "dry_run",
            "provider": "gemini",
            "slice_ids": slice_ids,
            "model": model,
            "payload_preview": payload_preview,
            "analysis": "Dry run only. GEMINI_API_KEY payload construction is ready.",
        }

    if not settings.GEMINI_API_KEY:
        return {
            "mode": "missing_api_key",
            "provider": "gemini",
            "slice_ids": slice_ids,
            "model": model,
            "payload_preview": payload_preview,
            "analysis": "GEMINI_API_KEY is not set. Add it to CT/.env to enable live Gemini analysis.",
        }

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    contents: list[Any] = [prompt]
    opened_images = [Image.open(path).convert("RGB") for path in image_paths]
    contents.extend(opened_images)
    attempted_models: list[str] = []
    provider_errors: list[str] = []
    analysis_text = ""
    used_model = model
    try:
        for candidate_model in [model, *fallback_models]:
            attempted_models.append(candidate_model)
            try:
                response = client.models.generate_content(model=candidate_model, contents=contents)
                analysis_text = _polish_analysis_text(response.text or "")
                used_model = candidate_model
                break
            except genai_errors.APIError as exc:
                status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                provider_errors.append(f"{candidate_model}: {exc}")
                if status_code not in {429, 500, 502, 503, 504}:
                    break
    except Exception as exc:
        return {
            "mode": "provider_error",
            "provider": "gemini",
            "slice_ids": slice_ids,
            "model": model,
            "attempted_models": attempted_models,
            "payload_preview": payload_preview,
            "analysis": f"Gemini inference error: {type(exc).__name__}: {exc}",
        }
    if not analysis_text:
        return {
            "mode": "provider_error",
            "provider": "gemini",
            "slice_ids": slice_ids,
            "model": model,
            "attempted_models": attempted_models,
            "payload_preview": payload_preview,
            "analysis": "Gemini did not return analysis. " + " | ".join(provider_errors),
        }
    return {
        "mode": "live" if used_model == model else "live_fallback",
        "provider": "gemini",
        "slice_ids": slice_ids,
        "model": used_model,
        "requested_model": model,
        "attempted_models": attempted_models,
        "payload_preview": payload_preview,
        "analysis": analysis_text,
    }


def analyze_snapshot_gemini(
    image_base64: str,
    user_note: str | None = None,
    asset_id: str | None = None,
    visible_labels: list[str] | None = None,
    rotation_x: float | None = None,
    rotation_y: float | None = None,
    scale: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_vlm_config()
    model = config.get("gemini_model", "gemini-3.1-flash-lite-preview")
    fallback_models = [item for item in config.get("gemini_fallback_models", []) if isinstance(item, str) and item != model]
    visible = visible_labels or []
    visible_summary = _visible_label_summary(visible)
    snapshot_prompt_template = config.get(
        "snapshot_user_prompt_template",
        (
            "Analyze this 3D medical anatomy viewport snapshot. User question: {user_note}. "
            "Visible anatomy labels: {visible_labels}. View state: rotation_x={rotation_x}, "
            "rotation_y={rotation_y}, scale={scale}. Return concise Markdown sections."
        ),
    )
    prompt = f"{config['system_prompt']}\n\n" + snapshot_prompt_template.format(
        user_note=user_note or "No specific question provided.",
        asset_id=asset_id or "unknown asset",
        visible_labels=", ".join(visible[:80]) if visible else "No labels provided.",
        visible_label_summary=visible_summary,
        rotation_x="unknown" if rotation_x is None else f"{rotation_x:.3f}",
        rotation_y="unknown" if rotation_y is None else f"{rotation_y:.3f}",
        scale="unknown" if scale is None else f"{scale:.3f}",
    )
    snapshot_image = _downscale_for_vlm(_image_from_base64(image_base64))
    snapshot_signal = _snapshot_signal(snapshot_image)
    if snapshot_signal["uninformative"]:
        prompt += (
            "\n\nAutomatic snapshot quality check: the captured image appears dark, blank, or otherwise "
            "uninformative. This is a known visionOS capture limitation for RealityKit layers. Do not "
            "state that the viewport is black, empty, failed, or unresponsive. Answer from the app-state "
            "visible anatomy labels, view state, and the user's question. If visual confirmation is limited, "
            "say that the capture is not suitable for pixel-level assessment, but still summarize the selected "
            "visible anatomy from app state."
        )

    payload_preview = {
        "model": model,
        "asset_id": asset_id,
        "visible_label_count": len(visible),
        "visible_label_summary": visible_summary,
        "snapshot_signal": snapshot_signal,
        "view_state": {
            "rotation_x": rotation_x,
            "rotation_y": rotation_y,
            "scale": scale,
        },
        "prompt": prompt,
        "image": "data:image/png;base64,<redacted>",
    }

    if dry_run:
        return {
            "mode": "dry_run",
            "provider": "gemini",
            "model": model,
            "payload_preview": payload_preview,
            "analysis": "Dry run only. 3D snapshot payload construction is ready.",
        }

    if not settings.GEMINI_API_KEY:
        return {
            "mode": "missing_api_key",
            "provider": "gemini",
            "model": model,
            "payload_preview": payload_preview,
            "analysis": "GEMINI_API_KEY is not set. Add it to CT/.env to enable live Gemini analysis.",
        }

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    attempted_models: list[str] = []
    provider_errors: list[str] = []
    analysis_text = ""
    used_model = model
    for candidate_model in [model, *fallback_models]:
        attempted_models.append(candidate_model)
        try:
            contents: list[Any] = [prompt] if snapshot_signal["uninformative"] else [prompt, snapshot_image]
            response = client.models.generate_content(model=candidate_model, contents=contents)
            analysis_text = _polish_analysis_text(response.text or "")
            used_model = candidate_model
            break
        except genai_errors.APIError as exc:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            provider_errors.append(f"{candidate_model}: {exc}")
            if status_code not in {429, 500, 502, 503, 504}:
                break

    if not analysis_text:
        return {
            "mode": "provider_error",
            "provider": "gemini",
            "model": model,
            "attempted_models": attempted_models,
            "payload_preview": payload_preview,
            "analysis": "Gemini did not return analysis. " + " | ".join(provider_errors),
        }

    return {
        "mode": "live" if used_model == model else "live_fallback",
        "provider": "gemini",
        "model": used_model,
        "requested_model": model,
        "attempted_models": attempted_models,
        "payload_preview": payload_preview,
        "analysis": analysis_text,
    }


def test_gemini_key() -> dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        return {"ok": False, "provider": "gemini", "error": "GEMINI_API_KEY is not set"}
    config = load_vlm_config()
    model = config.get("gemini_model", "gemini-3.1-flash-lite-preview")
    fallback_models = [item for item in config.get("gemini_fallback_models", []) if isinstance(item, str) and item != model]
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    errors: list[str] = []
    for candidate_model in [model, *fallback_models]:
        try:
            response = client.models.generate_content(model=candidate_model, contents="Reply with exactly: GEMINI_OK")
            text = (response.text or "").strip()
            return {
                "ok": "GEMINI_OK" in text,
                "provider": "gemini",
                "model": candidate_model,
                "requested_model": model,
                "mode": "live" if candidate_model == model else "live_fallback",
                "response": text[:80],
            }
        except genai_errors.APIError as exc:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            errors.append(f"{candidate_model}: {exc}")
            if status_code not in {429, 500, 502, 503, 504}:
                break
    return {"ok": False, "provider": "gemini", "model": model, "mode": "provider_error", "error": " | ".join(errors)}
