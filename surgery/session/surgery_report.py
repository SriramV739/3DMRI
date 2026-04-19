"""Draft surgical report generation from a session event log and keyframes."""

from __future__ import annotations

import base64
import json
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import cv2
import requests


class SurgeryReportGenerator:
    """Generate a surgeon-review draft report from logged events."""

    def __init__(
        self,
        *,
        provider: str = "rule_based",
        api_url: str = "",
        api_key: str = "",
        api_key_env: str = "GROQ_API_KEY",
        model: str = "",
        max_keyframes: int = 6,
    ):
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.api_key_env = api_key_env
        self.model = model
        self.max_keyframes = max(1, max_keyframes)

    def generate(
        self,
        *,
        events: Sequence[Dict[str, Any]],
        keyframe_paths: Sequence[str],
        metadata: Dict[str, Any],
    ) -> str:
        if self.provider == "openai_compatible" and self.api_url and self.api_key and self.model:
            try:
                return self._generate_with_vlm(events=events, keyframe_paths=keyframe_paths, metadata=metadata)
            except Exception as exc:
                fallback = self._generate_rule_based(events=events, metadata=metadata)
                return (
                    fallback
                    + "\n\n## Generation Note\n"
                    + f"VLM report generation failed and a structured fallback was used: {exc}"
                )
        return self._generate_rule_based(events=events, metadata=metadata)

    @staticmethod
    def to_pdf_bytes(report_text: str, *, title: str = "Surgery Report") -> bytes:
        plain_text = SurgeryReportGenerator._markdown_to_plain_text(report_text)
        lines = SurgeryReportGenerator._paginate_pdf_lines(title=title, report_text=plain_text)
        return SurgeryReportGenerator._build_simple_pdf(lines)

    def _generate_with_vlm(
        self,
        *,
        events: Sequence[Dict[str, Any]],
        keyframe_paths: Sequence[str],
        metadata: Dict[str, Any],
    ) -> str:
        try:
            return self._request_vlm_report(
                events=events,
                keyframe_paths=keyframe_paths,
                metadata=metadata,
                include_images=True,
            )
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 400 or not keyframe_paths:
                raise RuntimeError(self._format_http_error(exc)) from exc
            try:
                return self._request_vlm_report(
                    events=events,
                    keyframe_paths=[],
                    metadata=metadata,
                    include_images=False,
                )
            except Exception as retry_exc:
                raise RuntimeError(
                    "Vision report request was rejected; text-only retry also failed. "
                    f"Vision error: {self._format_http_error(exc)}; text retry: {retry_exc}"
                ) from retry_exc

    def _request_vlm_report(
        self,
        *,
        events: Sequence[Dict[str, Any]],
        keyframe_paths: Sequence[str],
        metadata: Dict[str, Any],
        include_images: bool,
    ) -> str:
        event_summary = self._summarize_events(events)
        report_input = json.dumps(
            {
                "task": "Generate a surgeon-review draft operative video documentation summary.",
                "metadata": metadata,
                "event_summary": event_summary,
                "events": list(events)[-80:],
                "requirements": [
                    "Use cautious medical documentation language.",
                    "Do not claim certainty beyond visual evidence.",
                    "Include observed anatomy, instruments, workflow, risks, uncertainties, and AI overlay actions.",
                    "State that this is an AI-generated draft requiring surgeon review.",
                ],
                "image_context": "Keyframes are attached." if include_images else "No images are attached; use the structured event log only.",
            },
            ensure_ascii=True,
        )

        user_content: str | list[Dict[str, Any]]
        if include_images:
            content: list[Dict[str, Any]] = [{"type": "text", "text": report_input}]
            for path in list(keyframe_paths)[-self.max_keyframes :]:
                encoded = self._encode_image(path)
                if encoded:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
            user_content = content
        else:
            user_content = report_input

        payload = {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 1800,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are assisting with retrospective laparoscopic cholecystectomy video documentation. "
                        "Produce a concise but clinically useful draft report for surgeon review. "
                        "Never present the output as a finalized medical record."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
        }
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _encode_image(self, path: str) -> str:
        image_path = Path(path)
        if not image_path.exists():
            return ""
        image = cv2.imread(str(image_path))
        if image is None:
            return ""
        height, width = image.shape[:2]
        max_dim = 512
        scale = min(1.0, max_dim / float(max(height, width)))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _format_http_error(self, exc: requests.HTTPError) -> str:
        response = exc.response
        if response is None:
            return str(exc)
        body = (response.text or "").strip()
        if len(body) > 600:
            body = body[:600] + "..."
        return f"{response.status_code} {response.reason}: {body or str(exc)}"

    def _summarize_events(self, events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        visible_structures: Counter[str] = Counter()
        visible_tools: Counter[str] = Counter()
        overlay_targets: Counter[str] = Counter()
        workflow_phases: Counter[str] = Counter()
        risks: list[str] = []
        uncertainties: list[str] = []
        user_questions: list[str] = []

        for event in events:
            for label in event.get("visible_structures", []) or []:
                visible_structures[label] += 1
            for label in event.get("visible_tools", []) or []:
                visible_tools[label] += 1
            for label in event.get("overlay_targets", []) or event.get("active_overlay_targets", []) or []:
                overlay_targets[label] += 1
            phase = event.get("workflow_phase")
            if phase:
                workflow_phases[str(phase)] += 1
            risks.extend(event.get("observed_risks", []) or [])
            uncertainties.extend(event.get("uncertainties", []) or [])
            if event.get("event_type") == "user_question" and event.get("user_text"):
                user_questions.append(str(event["user_text"]))

        return {
            "event_count": len(events),
            "visible_structures": visible_structures.most_common(),
            "visible_tools": visible_tools.most_common(),
            "overlay_targets": overlay_targets.most_common(),
            "workflow_phases": workflow_phases.most_common(),
            "observed_risks": list(dict.fromkeys(risks))[:12],
            "uncertainties": list(dict.fromkeys(uncertainties))[:12],
            "user_questions": user_questions[-20:],
        }

    def _generate_rule_based(self, *, events: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        summary = self._summarize_events(events)
        structures = ", ".join(label for label, _ in summary["visible_structures"]) or "not consistently logged"
        tools = ", ".join(label for label, _ in summary["visible_tools"]) or "not consistently logged"
        overlays = ", ".join(label for label, _ in summary["overlay_targets"]) or "none recorded"
        phases = ", ".join(phase.replace("_", " ") for phase, _ in summary["workflow_phases"]) or "not determined"
        risks = summary["observed_risks"] or ["No specific risk events were logged by the AI assistant."]
        uncertainties = summary["uncertainties"] or ["No structured uncertainty notes were logged."]

        return (
            "# AI-Generated Draft Surgery Documentation\n\n"
            "This is an AI-generated draft from the live video assistant log and selected frames. "
            "It is not a finalized medical record and requires surgeon review, correction, and sign-off.\n\n"
            "## Procedure Context\n"
            f"- Video/session: {metadata.get('video_name', 'unknown')}\n"
            f"- Logged events reviewed: {summary['event_count']}\n\n"
            "## Observed Workflow Timeline\n"
            f"- Dominant logged workflow states: {phases}.\n\n"
            "## Anatomy and Instruments Visualized\n"
            f"- Anatomy: {structures}.\n"
            f"- Instruments/tools: {tools}.\n\n"
            "## AI Overlay and Assistance Timeline\n"
            f"- Overlay targets requested or active during the session: {overlays}.\n\n"
            "## Notable Surgeon Queries\n"
            + "\n".join(f"- {question}" for question in summary["user_questions"][-10:])
            + ("\n" if summary["user_questions"] else "- No surgeon questions were logged.\n")
            + "\n## Risk and Uncertainty Observations\n"
            + "\n".join(f"- {risk}" for risk in risks)
            + "\n\n## Documentation Limitations\n"
            + "\n".join(f"- {uncertainty}" for uncertainty in uncertainties)
        )

    @staticmethod
    def _markdown_to_plain_text(text: str) -> str:
        plain = text.replace("\r\n", "\n").replace("\r", "\n")
        plain = re.sub(r"^#{1,6}\s*", "", plain, flags=re.MULTILINE)
        plain = re.sub(r"^\s*[-*]\s+", "- ", plain, flags=re.MULTILINE)
        plain = re.sub(r"[*_`]+", "", plain)
        plain = re.sub(r"\n{3,}", "\n\n", plain)
        return plain.strip()

    @staticmethod
    def _paginate_pdf_lines(*, title: str, report_text: str) -> list[list[str]]:
        width = 92
        page_line_limit = 48
        wrapped_lines: list[str] = [title, ""]
        for paragraph in report_text.split("\n"):
            stripped = paragraph.strip()
            if not stripped:
                wrapped_lines.append("")
                continue
            if stripped.startswith("- "):
                bullet = stripped[:2]
                body = stripped[2:].strip()
                bullet_lines = textwrap.wrap(body, width=max(12, width - len(bullet)))
                if not bullet_lines:
                    wrapped_lines.append("-")
                    continue
                wrapped_lines.append(bullet + bullet_lines[0])
                for continuation in bullet_lines[1:]:
                    wrapped_lines.append("  " + continuation)
                continue
            wrapped_lines.extend(textwrap.wrap(stripped, width=width) or [""])

        pages: list[list[str]] = []
        current_page: list[str] = []
        for line in wrapped_lines:
            current_page.append(line)
            if len(current_page) >= page_line_limit:
                pages.append(current_page)
                current_page = []
        if current_page or not pages:
            pages.append(current_page)
        return pages

    @staticmethod
    def _pdf_escape(text: str) -> str:
        normalized = text.encode("latin-1", "replace").decode("latin-1")
        return normalized.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    @staticmethod
    def _build_simple_pdf(pages: Sequence[Sequence[str]]) -> bytes:
        objects: list[bytes] = []

        def add_object(payload: bytes) -> int:
            objects.append(payload)
            return len(objects)

        font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        page_ids: list[int] = []
        content_ids: list[int] = []
        page_size = (612, 792)
        left = 54
        top = 738
        line_height = 14

        for page_lines in pages:
            stream_lines = [
                "BT",
                "/F1 11 Tf",
                f"{left} {top} Td",
                f"{line_height} TL",
            ]
            for idx, line in enumerate(page_lines):
                escaped = SurgeryReportGenerator._pdf_escape(line)
                if idx == 0:
                    stream_lines.append(f"({escaped}) Tj")
                else:
                    stream_lines.append(f"T* ({escaped}) Tj")
            stream_lines.append("ET")
            stream_bytes = "\n".join(stream_lines).encode("latin-1", "replace")
            content_id = add_object(
                b"<< /Length " + str(len(stream_bytes)).encode("ascii") + b" >>\nstream\n" + stream_bytes + b"\nendstream"
            )
            content_ids.append(content_id)
            page_id = add_object(b"")
            page_ids.append(page_id)

        kids = " ".join(f"{page_id} 0 R" for page_id in page_ids).encode("ascii")
        pages_id = add_object(b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_ids)).encode("ascii") + b" >>")
        catalog_id = add_object(b"<< /Type /Catalog /Pages " + str(pages_id).encode("ascii") + b" 0 R >>")

        for page_id, content_id in zip(page_ids, content_ids):
            objects[page_id - 1] = (
                b"<< /Type /Page /Parent "
                + str(pages_id).encode("ascii")
                + b" 0 R /MediaBox [0 0 "
                + str(page_size[0]).encode("ascii")
                + b" "
                + str(page_size[1]).encode("ascii")
                + b"] /Resources << /Font << /F1 "
                + str(font_id).encode("ascii")
                + b" 0 R >> >> /Contents "
                + str(content_id).encode("ascii")
                + b" 0 R >>"
            )

        pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for idx, payload in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
            pdf.extend(payload)
            pdf.extend(b"\nendobj\n")

        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf.extend(
            (
                "trailer\n"
                f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
                "startxref\n"
                f"{xref_offset}\n"
                "%%EOF\n"
            ).encode("ascii")
        )
        return bytes(pdf)
