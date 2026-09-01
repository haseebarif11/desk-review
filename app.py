"""Desk Review — Gradio UI for Hugging Face Spaces."""

from __future__ import annotations

import html
import json
import logging
import re
import time
from typing import Any

import gradio as gr
import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

try:
    import spaces
except ImportError:

    class _SpacesShim:
        @staticmethod
        def GPU():
            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesShim()  # type: ignore[assignment,misc]

from core import (
    AnalyzeRequest,
    Config,
    GeminiAPIError,
    GeminiClientError,
    GeminiServerError,
    analyze_resume,
    configure_logging,
)

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


def _format_result_as_html(result: dict[str, Any]) -> str:
    """Render analysis as an editorial manuscript report."""
    score = html.escape(str(result["score"]))

    score_html = (
        "<div class='desk-review-stamp'>"
        "<div class='stamp-header'>Manuscript Evaluation</div>"
        f"<div class='score'>Score: {score}/100</div>"
        "<div class='stamp-subtitle'>Editorial Readiness</div>"
        "</div>"
    )

    strengths = result.get("strengths", [])
    if strengths:
        s_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in strengths)
        s_body = f"<ul>{s_rows}</ul>"
    else:
        s_body = "<p class='muted'>None identified</p>"

    weaknesses = result.get("weaknesses", [])
    if weaknesses:
        w_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in weaknesses)
        w_body = f"<ul>{w_rows}</ul>"
    else:
        w_body = "<p class='muted'>None identified</p>"

    comparison_grid = (
        "<div class='editorial-grid'>"
        f"<section class='grid-col strengths-col'><h3>Strengths</h3>{s_body}</section>"
        "<section class='grid-col weaknesses-col'>"
        f"<h3>Weaknesses</h3>{w_body}</section>"
        "</div>"
    )

    missing = result.get("missing_keywords", [])
    if missing:
        tags_html = "".join(
            f"<span class='keyword-tag'>{html.escape(str(kw))}</span>" for kw in missing
        )
        missing_body = f"<div class='keyword-tags-wrapper'>{tags_html}</div>"
    else:
        missing_body = "<p class='muted'>None identified</p>"

    missing_section = (
        f"<section class='missing-keywords-section'>"
        f"<h3>Missing keywords</h3>{missing_body}</section>"
    )

    rewrites = result.get("bullet_rewrites", [])
    if rewrites:
        rewrite_blocks = []
        for rewrite in rewrites:
            original = html.escape(str(rewrite["original"]))
            improved = html.escape(str(rewrite["improved"]))
            rewrite_blocks.append(
                "<div class='rewrite-card'>"
                "<div class='rewrite-original'>"
                "<span class='rewrite-label'>Original Draft</span>"
                f"<p>{original}</p>"
                "</div>"
                "<div class='rewrite-improved'>"
                "<span class='rewrite-label'>Editorial Recommendation</span>"
                f"<p>{improved}</p>"
                "</div>"
                "</div>"
            )
        rewrites_html = "".join(rewrite_blocks)
    else:
        rewrites_html = "<p class='muted'>No rewrites suggested.</p>"

    rewrites_section = (
        f"<section class='rewrites-section'>"
        f"<h3>Bullet rewrites</h3>{rewrites_html}</section>"
    )

    next_steps = result.get("next_steps", [])
    if next_steps:
        ns_rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in next_steps)
        ns_body = f"<ol class='editorial-steps'>{ns_rows}</ol>"
    else:
        ns_body = "<p class='muted'>None identified</p>"

    next_steps_section = (
        f"<section class='next-steps-section'>"
        f"<h3>Next steps</h3>{ns_body}</section>"
    )

    return (
        "<div class='desk-review-results'>"
        f"{score_html}"
        f"{comparison_grid}"
        f"{missing_section}"
        f"{rewrites_section}"
        f"{next_steps_section}"
        "</div>"
    )


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


@spaces.GPU(duration=1)
def _zerogpu_startup_stub() -> None:
    """Satisfy ZeroGPU startup checks without routing analysis through the GPU queue."""
    return None


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
        return _format_result_as_html(result)
    except GeminiClientError as exc:
        if exc.code == 429:
            return "AI service is busy. Please try again shortly."
        return "AI service request failed."
    except GeminiServerError:
        return "AI service temporarily unavailable."
    except httpx.TimeoutException:
        return "AI service timed out. Please try again."
    except GeminiAPIError:
        return "AI service error."
    except json.JSONDecodeError:
        return "AI returned malformed data."
    except ValidationError:
        return "AI returned malformed data."
    except ValueError as exc:
        return str(exc)


EDITORIAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400..700;1,9..144,400..700&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* Focus and accessibility baseline */
*:focus-visible {
    outline: 2px solid #C89B3C !important;
    outline-offset: 2px !important;
}

/* Overall container styling */
.gradio-container {
    background-color: #F7F5F0 !important;
    font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem !important;
}

/* Hero Section */
.desk-review-hero {
    margin-bottom: 2rem;
    border-bottom: 2px solid #E2DDD5;
    padding-bottom: 1.5rem;
}
.desk-review-hero h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 2.75rem !important;
    font-weight: 700 !important;
    color: #1B2333 !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}
.desk-review-hero p {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1.05rem !important;
    color: #5A6474 !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

/* Review Button Styling */
#review-btn {
    background-color: #1B2333 !important;
    color: #F7F5F0 !important;
    border: 1px solid #1B2333 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    transition: background-color 0.2s ease, border-color 0.2s ease !important;
    box-shadow: none !important;
}
#review-btn:hover {
    background-color: #28344B !important;
    border-color: #28344B !important;
}

/* Results Section Container */
.desk-review-results {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    color: #1B2333 !important;
    background: #FFFFFF !important;
    padding: 2rem;
    border: 1px solid #E2DDD5;
    border-radius: 6px;
    max-width: 100%;
    overflow-wrap: anywhere;
    margin-top: 1.5rem;
    animation: editorialReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes editorialReveal {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (prefers-reduced-motion: reduce) {
    .desk-review-results {
        animation: none !important;
    }
}

/* Editorial Score Stamp */
.desk-review-stamp {
    border: 2px double #C89B3C;
    background: #FDFBF7;
    padding: 1rem 1.5rem;
    margin: 0 auto 2rem auto;
    max-width: 280px;
    text-align: center;
    border-radius: 6px;
}
.desk-review-stamp .stamp-header {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.6875rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #8C6212;
    text-transform: uppercase;
}
.desk-review-stamp .score {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    color: #1B2333 !important;
    margin: 0.35rem 0 !important;
    line-height: 1.1 !important;
}
.desk-review-stamp .stamp-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #5A6474;
    text-transform: uppercase;
}

/* Section Titles */
.desk-review-results h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1B2333 !important;
    margin: 0 0 0.75rem 0 !important;
    letter-spacing: -0.01em;
}

/* Editorial Comparison Grid (Strengths / Weaknesses) */
.editorial-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 1.75rem;
}
@media (max-width: 640px) {
    .editorial-grid {
        grid-template-columns: 1fr;
    }
}
.grid-col {
    background: #FAF8F5;
    border: 1px solid #E2DDD5;
    padding: 1.25rem;
    border-radius: 6px;
}
.strengths-col {
    border-left: 4px solid #15803D;
}
.weaknesses-col {
    border-left: 4px solid #C89B3C;
}
.grid-col ul {
    margin: 0;
    padding-left: 1.2rem;
}
.grid-col li {
    margin-bottom: 0.5rem;
    color: #1E2430 !important;
}

/* Missing Keywords Chips */
.missing-keywords-section {
    margin-bottom: 1.75rem;
}
.keyword-tags-wrapper {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.keyword-tag {
    background: #F4EFE6;
    border: 1px solid #D9D1C3;
    color: #1B2333 !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.3rem 0.75rem;
    border-radius: 4px;
    display: inline-block;
}

/* Bullet Rewrites Redline Motif */
.rewrites-section {
    margin-bottom: 1.75rem;
}
.rewrite-card {
    background: #FAF8F5;
    border: 1px solid #E2DDD5;
    border-radius: 6px;
    margin-bottom: 1rem;
    overflow: hidden;
}
.rewrite-original {
    background: #FDF2F2;
    border-left: 4px solid #991B1B;
    padding: 0.85rem 1.1rem;
}
.rewrite-original .rewrite-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.725rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #991B1B;
    display: block;
    text-transform: uppercase;
}
.rewrite-original p {
    color: #7F1D1D !important;
    margin: 0.25rem 0 0 0 !important;
    text-decoration: line-through;
    text-decoration-color: #991B1B;
}
.rewrite-improved {
    background: #F0F7F4;
    border-left: 4px solid #15803D;
    border-top: 1px solid #E2DDD5;
    padding: 0.85rem 1.1rem;
}
.rewrite-improved .rewrite-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.725rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #15803D;
    display: block;
    text-transform: uppercase;
}
.rewrite-improved p {
    color: #14532D !important;
    font-weight: 500;
    margin: 0.25rem 0 0 0 !important;
}

/* Next Steps Numbered List */
.next-steps-section {
    margin-bottom: 0.5rem;
}
.editorial-steps {
    margin: 0;
    padding-left: 1.4rem;
}
.editorial-steps li {
    margin-bottom: 0.6rem;
    color: #1E2430 !important;
    line-height: 1.5;
}
.editorial-steps li::marker {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 700;
    color: #C89B3C;
}

.desk-review-results .muted {
    color: #5A6474 !important;
    margin: 0;
}
"""

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#F7F5F0",
    body_text_color="#1B2333",
    background_fill_primary="#F7F5F0",
    background_fill_secondary="#FAF8F5",
    border_color_primary="#E2DDD5",
    button_primary_background_fill="#1B2333",
    button_primary_background_fill_hover="#28344B",
    button_primary_text_color="#F7F5F0",
    button_primary_border_color="#1B2333",
)

with gr.Blocks(title="Desk Review", theme=theme, css=EDITORIAL_CSS) as demo:
    gr.HTML(
        "<header class='desk-review-hero'>"
        "<h1>Desk Review</h1>"
        "<p>An editor's pre-publication manuscript evaluation for your resume.</p>"
        "</header>"
    )
    target_role_input = gr.Textbox(
        label="Target role",
        lines=1,
        placeholder="e.g. Software Engineer",
    )
    resume_input = gr.Textbox(
        label="Resume",
        lines=12,
        placeholder="Paste your resume text here...",
    )
    job_description_input = gr.Textbox(
        label="Job description (optional)",
        lines=6,
        placeholder="Paste a job description for tighter matching...",
    )
    analyze_button = gr.Button("Review", variant="primary", elem_id="review-btn")
    result_output = gr.HTML(label="Results")

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
