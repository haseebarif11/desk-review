"""Desk Review — Gradio UI for Hugging Face Spaces."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import gradio as gr
import httpx
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from pydantic import ValidationError

from core import AnalyzeRequest, Config, analyze_resume, configure_logging

logger = logging.getLogger(__name__)

load_dotenv()

config: Config | None = None
try:
    config = Config.from_env()
    configure_logging(config.log_level)
except ValueError:
    logger.warning("GEMINI_API_KEY not set; UI will load but analysis is disabled.")

_rate_limit_buckets: dict[str, list[float]] = {}


def _parse_rate_limit_per_minute(rate_limit: str) -> int:
    """Parse 'N per minute' into an integer N."""
    match = re.match(r"^\s*(\d+)\s+per\s+minute\s*$", rate_limit, re.IGNORECASE)
    if not match:
        return 10
    return int(match.group(1))


def _check_rate_limit(session_key: str) -> str | None:
    """Return a user-facing error if the session exceeded the per-minute limit."""
    if config is None:
        return None

    limit = _parse_rate_limit_per_minute(config.rate_limit_default)
    now = time.time()
    window_seconds = 60.0

    timestamps = [
        ts
        for ts in _rate_limit_buckets.get(session_key, [])
        if now - ts < window_seconds
    ]
    if len(timestamps) >= limit:
        _rate_limit_buckets[session_key] = timestamps
        return "Rate limit exceeded. Please wait a minute and try again."

    timestamps.append(now)
    _rate_limit_buckets[session_key] = timestamps
    return None


def _format_validation_errors(exc: ValidationError) -> str:
    """Turn pydantic validation errors into a readable message."""
    messages: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        msg = error.get("msg", "Invalid value")
        if loc:
            messages.append(f"{loc}: {msg}")
        else:
            messages.append(msg)
    return "Validation failed: " + "; ".join(messages)


def _format_result_as_markdown(result: dict[str, Any]) -> str:
    """Render AnalyzeResponse fields as readable Markdown sections."""
    lines = [f"## Score: {result['score']}/100", ""]

    lines.append("## Strengths")
    lines.extend(f"- {item}" for item in result.get("strengths", []))
    lines.append("")

    lines.append("## Weaknesses")
    lines.extend(f"- {item}" for item in result.get("weaknesses", []))
    lines.append("")

    lines.append("## Missing keywords")
    missing = result.get("missing_keywords", [])
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- None identified")
    lines.append("")

    lines.append("## Bullet rewrites")
    for rewrite in result.get("bullet_rewrites", []):
        lines.append(f"**Original:** {rewrite['original']}")
        lines.append(f"**Improved:** {rewrite['improved']}")
        lines.append("")
    if not result.get("bullet_rewrites"):
        lines.append("- No rewrites suggested")
        lines.append("")

    lines.append("## Next steps")
    lines.extend(f"- {item}" for item in result.get("next_steps", []))

    return "\n".join(lines).strip()


def analyze_handler(
    resume: str,
    target_role: str,
    job_description: str,
    request: gr.Request,
) -> str:
    """Validate input, enforce rate limits, and return formatted analysis."""
    session_key = request.session_hash or "unknown"

    rate_error = _check_rate_limit(session_key)
    if rate_error:
        return rate_error

    if config is None:
        return (
            "GEMINI_API_KEY is not configured. "
            "Set it in Space secrets or your .env file."
        )

    try:
        analyze_request = AnalyzeRequest.model_validate(
            {
                "resume": resume,
                "target_role": target_role,
                "job_description": job_description or None,
            }
        )
    except ValidationError as exc:
        return _format_validation_errors(exc)

    try:
        result = analyze_resume(analyze_request, config)
        return _format_result_as_markdown(result)
    except genai_errors.ClientError as exc:
        if exc.code == 429:
            return "AI service is busy. Please try again shortly."
        return "AI service request failed."
    except genai_errors.ServerError:
        return "AI service temporarily unavailable."
    except httpx.TimeoutException:
        return "AI service timed out. Please try again."
    except genai_errors.APIError:
        return "AI service error."
    except json.JSONDecodeError:
        return "AI returned malformed data."
    except ValidationError:
        return "AI returned malformed data."
    except ValueError as exc:
        return str(exc)


with gr.Blocks(title="Desk Review") as demo:
    gr.Markdown(
        "# Desk Review\n"
        "Paste your resume and target role for structured AI feedback."
    )
    resume_input = gr.Textbox(
        label="Resume",
        lines=12,
        placeholder="Paste your resume text here...",
    )
    target_role_input = gr.Textbox(
        label="Target role",
        lines=1,
        placeholder="e.g. Software Engineer",
    )
    job_description_input = gr.Textbox(
        label="Job description (optional)",
        lines=6,
        placeholder="Paste a job description for tighter matching...",
    )
    analyze_button = gr.Button("Analyze", variant="primary")
    result_output = gr.Markdown(label="Results")

    analyze_button.click(
        analyze_handler,
        inputs=[resume_input, target_role_input, job_description_input],
        outputs=result_output,
    )


@demo.app.get("/healthz")
def healthz() -> dict[str, str]:
    """Lightweight liveness probe for platform health checks."""
    return {"status": "ok"}


if __name__ == "__main__":
    demo.launch()
