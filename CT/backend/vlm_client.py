from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from groq import Groq
from google.genai import errors as genai_errors
from google import genai
from PIL import Image

from backend import settings
from backend.imaging.dicom_io import ensure_slice_preview_png


def load_vlm_config() -> dict[str, Any]:
    with settings.VLM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        return text

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
        rotation_x="unknown" if rotation_x is None else f"{rotation_x:.3f}",
        rotation_y="unknown" if rotation_y is None else f"{rotation_y:.3f}",
        scale="unknown" if scale is None else f"{scale:.3f}",
    )
    payload_preview = {
        "model": model,
        "asset_id": asset_id,
        "visible_label_count": len(visible),
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

    snapshot_image = _image_from_base64(image_base64)
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    attempted_models: list[str] = []
    provider_errors: list[str] = []
    analysis_text = ""
    used_model = model
    for candidate_model in [model, *fallback_models]:
        attempted_models.append(candidate_model)
        try:
            response = client.models.generate_content(model=candidate_model, contents=[prompt, snapshot_image])
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
