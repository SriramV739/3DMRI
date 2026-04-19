from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from groq import Groq
from google import genai
from PIL import Image

from backend import settings
from backend.imaging.dicom_io import ensure_slice_preview_png


def load_vlm_config() -> dict[str, Any]:
    with settings.VLM_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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
    model = config.get("gemini_model", "gemini-2.5-flash")
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
    contents.extend(Image.open(path) for path in image_paths)
    response = client.models.generate_content(model=model, contents=contents)
    return {
        "mode": "live",
        "provider": "gemini",
        "slice_ids": slice_ids,
        "model": model,
        "payload_preview": payload_preview,
        "analysis": response.text,
    }


def test_gemini_key() -> dict[str, Any]:
    if not settings.GEMINI_API_KEY:
        return {"ok": False, "provider": "gemini", "error": "GEMINI_API_KEY is not set"}
    model = load_vlm_config().get("gemini_model", "gemini-2.5-flash")
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(model=model, contents="Reply with exactly: GEMINI_OK")
    text = (response.text or "").strip()
    return {"ok": "GEMINI_OK" in text, "provider": "gemini", "model": model, "response": text[:80]}
