"""Desk Review — config, schemas, and analysis service (framework-agnostic)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from keywords import get_keywords_for_role

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiAPIError(Exception):
    """Base error for Gemini HTTP API failures."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(message)


class GeminiClientError(GeminiAPIError):
    """Non-retryable Gemini client error (4xx)."""


class GeminiServerError(GeminiAPIError):
    """Retryable Gemini server error (5xx)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Central configuration loaded from environment variables."""

    gemini_api_key: str
    gemini_model: str
    max_tokens: int
    api_timeout_seconds: float
    api_max_retries: int
    api_retry_base_delay_seconds: float
    rate_limit_default: str
    max_resume_length: int
    max_target_role_length: int
    max_job_description_length: int
    min_resume_length: int
    flask_secret_key: str
    cors_origins: str
    log_level: str

    @classmethod
    def from_env(cls) -> Config:
        """Build configuration from environment variables with sensible defaults."""
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        return cls(
            gemini_api_key=api_key,
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            max_tokens=int(os.environ.get("GEMINI_MAX_TOKENS", "2048")),
            api_timeout_seconds=float(os.environ.get("API_TIMEOUT_SECONDS", "60")),
            api_max_retries=int(os.environ.get("API_MAX_RETRIES", "3")),
            api_retry_base_delay_seconds=float(
                os.environ.get("API_RETRY_BASE_DELAY_SECONDS", "1.0")
            ),
            rate_limit_default=os.environ.get("RATE_LIMIT_DEFAULT", "10 per minute"),
            max_resume_length=int(os.environ.get("MAX_RESUME_LENGTH", "15000")),
            max_target_role_length=int(os.environ.get("MAX_TARGET_ROLE_LENGTH", "200")),
            max_job_description_length=int(
                os.environ.get("MAX_JOB_DESCRIPTION_LENGTH", "5000")
            ),
            min_resume_length=int(os.environ.get("MIN_RESUME_LENGTH", "50")),
            flask_secret_key=os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me"),
            cors_origins=os.environ.get("CORS_ORIGINS", ""),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Validated payload for the /api/analyze endpoint."""

    resume: str = Field(..., min_length=1)
    target_role: str = Field(..., min_length=1)
    job_description: str | None = None

    @field_validator("resume")
    @classmethod
    def validate_resume(cls, value: str) -> str:
        """Ensure resume text is substantive and not placeholder content."""
        stripped = value.strip()
        if len(stripped) < 50:
            raise ValueError("Resume must be at least 50 characters")
        if len(stripped) > 15000:
            raise ValueError("Resume must not exceed 15,000 characters")
        alpha_count = sum(1 for char in stripped if char.isalpha())
        if alpha_count < 30:
            raise ValueError("Resume must contain meaningful text content")
        if re.fullmatch(r"[\s\W\d]+", stripped):
            raise ValueError("Resume must contain readable text")
        return stripped

    @field_validator("target_role")
    @classmethod
    def validate_target_role(cls, value: str) -> str:
        """Normalize and validate the target job role."""
        stripped = value.strip()
        if len(stripped) < 2:
            raise ValueError("Target role must be at least 2 characters")
        if len(stripped) > 200:
            raise ValueError("Target role must not exceed 200 characters")
        if not re.search(r"[A-Za-z]", stripped):
            raise ValueError("Target role must contain letters")
        return stripped

    @field_validator("job_description")
    @classmethod
    def validate_job_description(cls, value: str | None) -> str | None:
        """Validate optional job description length."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if len(stripped) > 5000:
            raise ValueError("Job description must not exceed 5,000 characters")
        return stripped


class BulletRewrite(BaseModel):
    """A single before/after bullet point suggestion."""

    original: str = Field(..., min_length=1)
    improved: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    """Strict schema for model JSON output returned to the frontend."""

    score: int = Field(..., ge=0, le=100)
    strengths: list[str] = Field(..., min_length=1)
    weaknesses: list[str] = Field(..., min_length=1)
    missing_keywords: list[str]
    bullet_rewrites: list[BulletRewrite]
    next_steps: list[str] = Field(..., min_length=1)

    @field_validator(
        "strengths", "weaknesses", "missing_keywords", "next_steps", mode="before"
    )
    @classmethod
    def ensure_string_lists(cls, value: Any) -> list[str]:
        """Coerce list items to non-empty strings where required."""
        if not isinstance(value, list):
            raise TypeError("Expected a list")
        return [str(item).strip() for item in value if str(item).strip()]

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for API responses."""
        return self.model_dump()


SYSTEM_PROMPT = (
    "You are an expert resume coach. Analyze the resume against the target role "
    "and return ONLY valid JSON with this exact structure "
    "(no markdown, no commentary):\n\n"
    "{\n"
    '  "score": <integer 0-100>,\n'
    '  "strengths": [<string>, ...],\n'
    '  "weaknesses": [<string>, ...],\n'
    '  "missing_keywords": [<string>, ...],\n'
    '  "bullet_rewrites": [{"original": "<string>", "improved": "<string>"}, ...],\n'
    '  "next_steps": [<string>, ...]\n'
    "}\n\n"
    "Rules:\n"
    "- score reflects fit for the target role (0=very poor, 100=excellent)\n"
    "- strengths and weaknesses: 2-5 items each, specific and actionable\n"
    "- missing_keywords: terms from the provided keyword list that are absent "
    "or weak in the resume\n"
    "- bullet_rewrites: 2-4 weak bullets rewritten with stronger action verbs "
    "and metrics where possible\n"
    "- next_steps: 3-5 prioritized actions the candidate should take\n"
    "- Output JSON only. No preamble or explanation."
)


def configure_logging(level: str) -> None:
    """Configure structured application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_user_prompt(analyze_request: AnalyzeRequest) -> str:
    """Construct the user message with resume, role, and grounded keywords."""
    keywords = get_keywords_for_role(analyze_request.target_role)
    keyword_block = ", ".join(keywords)

    parts = [
        f"Target role: {analyze_request.target_role}",
        f"Role keywords to check: {keyword_block}",
    ]

    if analyze_request.job_description:
        parts.append(f"Job description:\n{analyze_request.job_description}")

    parts.append(f"Resume:\n{analyze_request.resume}")
    return "\n\n".join(parts)


def _extract_json_text(raw_text: str) -> str:
    """Strip optional markdown fences from model output."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _parse_generate_content_response(payload: dict[str, Any]) -> AnalyzeResponse:
    """Read structured output from a Gemini generateContent response."""
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("Empty response from AI service")

    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if part.get("text")]
    if not text_parts:
        raise ValueError("Empty response from AI service")

    parsed = json.loads(_extract_json_text("".join(text_parts)))
    return AnalyzeResponse.model_validate(parsed)


def _call_gemini_with_retry(
    config: Config,
    prompt: str,
    http_client: httpx.Client | None = None,
) -> AnalyzeResponse:
    """Call Gemini with timeout and exponential backoff on retryable errors."""
    client = http_client or httpx.Client(timeout=config.api_timeout_seconds)
    owns_client = http_client is None
    url = f"{GEMINI_API_BASE}/models/{config.gemini_model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": AnalyzeResponse.model_json_schema(),
            "maxOutputTokens": config.max_tokens,
        },
    }
    last_error: Exception | None = None

    try:
        for attempt in range(config.api_max_retries):
            try:
                response = client.post(
                    url,
                    params={"key": config.gemini_api_key},
                    json=payload,
                )
                if response.status_code >= 500:
                    raise GeminiServerError(
                        response.status_code,
                        response.text or "Gemini server error",
                    )
                if response.status_code >= 400:
                    raise GeminiClientError(
                        response.status_code,
                        response.text or "Gemini client error",
                    )
                return _parse_generate_content_response(response.json())
            except GeminiServerError as exc:
                last_error = exc
                if attempt < config.api_max_retries - 1:
                    delay = config.api_retry_base_delay_seconds * (2**attempt)
                    logger.warning(
                        "gemini 5xx attempt=%d code=%d delay=%.1fs",
                        attempt + 1,
                        exc.code,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < config.api_max_retries - 1:
                    delay = config.api_retry_base_delay_seconds * (2**attempt)
                    logger.warning(
                        "gemini timeout attempt=%d delay=%.1fs",
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise
            except GeminiClientError:
                raise
    finally:
        if owns_client:
            client.close()

    if last_error:
        raise last_error
    raise RuntimeError("Gemini call failed without a captured error")


def analyze_resume(
    analyze_request: AnalyzeRequest,
    config: Config | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """
    Run resume analysis via Google Gemini and validate structured output.

    The API key is only used server-side via the Gemini client and is never
    logged or returned to callers.
    """
    app_config = config or Config.from_env()
    prompt = f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(analyze_request)}"
    validated = _call_gemini_with_retry(app_config, prompt, http_client=http_client)
    return validated.to_api_dict()
