"""Pytest suite for Desk Review (mocked — no real API calls)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app import analyze_handler
from core import (
    AnalyzeRequest,
    AnalyzeResponse,
    Config,
    analyze_resume,
)

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


@dataclass
class FakeClient:
    host: str


@dataclass
class FakeRequest:
    client: FakeClient
    session_hash: str = "test-session-1"


@pytest.fixture
def test_config() -> Config:
    """Config for unit tests (no real API key needed)."""
    return Config(
        gemini_api_key="test-key-not-real",
        gemini_model="gemini-2.5-flash",
        max_tokens=1024,
        api_timeout_seconds=5.0,
        api_max_retries=2,
        api_retry_base_delay_seconds=0.01,
        rate_limit_default="1000 per minute",
        max_resume_length=15000,
        max_target_role_length=200,
        max_job_description_length=5000,
        min_resume_length=50,
        flask_secret_key="test-secret",
        cors_origins="",
        log_level="WARNING",
    )


@pytest.fixture
def valid_analyze_payload() -> dict[str, str]:
    """Minimal valid request payload for handler tests."""
    return {
        "resume": (
            "Jane Doe\nSoftware Engineer\n\nEXPERIENCE\n"
            "Built REST APIs with Python and Flask at Acme Corp for 3 years. "
            "Improved deployment pipeline with CI/CD and wrote unit tests. "
            "Collaborated with cross-functional teams on agile sprints."
        ),
        "target_role": "Software Engineer",
        "job_description": (
            "Looking for a backend engineer with Python and API experience."
        ),
    }


@pytest.fixture
def valid_model_response() -> dict:
    """Valid structured response matching AnalyzeResponse schema."""
    return {
        "score": 72,
        "strengths": [
            "Clear Python and Flask experience",
            "Mentions CI/CD and testing practices",
        ],
        "weaknesses": [
            "Limited quantified impact metrics",
            "Missing cloud platform experience",
        ],
        "missing_keywords": ["kubernetes", "docker"],
        "bullet_rewrites": [
            {
                "original": "Built REST APIs with Python and Flask.",
                "improved": (
                    "Designed and shipped 5+ REST APIs in Python/Flask, "
                    "serving 10k daily requests with 99.9% uptime."
                ),
            }
        ],
        "next_steps": [
            "Add metrics to each bullet point",
            "Include cloud and container keywords if applicable",
            "Tailor summary to target role",
        ],
    }


def _mock_client(
    response_dict: dict, *, parsed: AnalyzeResponse | None = None
) -> MagicMock:
    """Build a mock Gemini client returning structured output."""
    client = MagicMock()
    response = MagicMock()
    response.parsed = parsed
    response.text = json.dumps(response_dict)
    client.models.generate_content.return_value = response
    return client


def test_analyze_resume_returns_valid_schema(
    test_config, valid_analyze_payload, valid_model_response
):
    """Mocked analysis returns all required keys with score in range."""
    request = AnalyzeRequest.model_validate(valid_analyze_payload)
    parsed = AnalyzeResponse.model_validate(valid_model_response)
    client = _mock_client(valid_model_response, parsed=parsed)

    result = analyze_resume(request, config=test_config, client=client)

    validated = AnalyzeResponse.model_validate(result)
    assert 0 <= validated.score <= 100
    assert validated.strengths
    assert validated.weaknesses
    assert validated.next_steps


def test_analyze_resume_rejects_malformed_model_output(
    test_config, valid_analyze_payload
):
    """Malformed AI JSON should raise ValidationError."""
    request = AnalyzeRequest.model_validate(valid_analyze_payload)
    client = _mock_client({"score": 150, "strengths": []})

    with pytest.raises(ValidationError):
        analyze_resume(request, config=test_config, client=client)


def test_analyze_resume_sample_files(test_config, valid_model_response):
    """Each sample resume file can be parsed and analyzed with a mock."""
    parsed = AnalyzeResponse.model_validate(valid_model_response)
    for sample_path in SAMPLES_DIR.glob("*.txt"):
        request = AnalyzeRequest(
            resume=sample_path.read_text(encoding="utf-8"),
            target_role="Software Engineer",
        )
        result = analyze_resume(
            request,
            config=test_config,
            client=_mock_client(valid_model_response, parsed=parsed),
        )
        assert 0 <= result["score"] <= 100


def test_keywords_grounding_in_prompt(
    test_config, valid_analyze_payload, valid_model_response
):
    """The Gemini call should include role keywords from keywords.py."""
    request = AnalyzeRequest.model_validate(valid_analyze_payload)
    parsed = AnalyzeResponse.model_validate(valid_model_response)
    client = _mock_client(valid_model_response, parsed=parsed)

    analyze_resume(request, config=test_config, client=client)

    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert "python" in prompt.lower()
    assert "Target role:" in prompt


def test_analyze_handler_returns_formatted_markdown(
    monkeypatch, test_config, valid_analyze_payload, valid_model_response
):
    """Gradio handler should return Markdown sections for a successful analysis."""
    import app as gradio_app

    monkeypatch.setattr(gradio_app, "config", test_config)
    gradio_app._rate_limit_buckets.clear()

    with patch("app.analyze_resume", return_value=valid_model_response):
        result = analyze_handler(
            valid_analyze_payload["resume"],
            valid_analyze_payload["target_role"],
            valid_analyze_payload["job_description"],
            FakeRequest(client=FakeClient(host="127.0.0.1")),
        )

    assert "## Score: 72/100" in result
    assert "## Strengths" in result
    assert "## Weaknesses" in result
    assert "## Missing keywords" in result
    assert "## Bullet rewrites" in result
    assert "## Next steps" in result
    assert "Clear Python and Flask experience" in result


@pytest.fixture
def strict_rate_limit_config() -> Config:
    """Config with a tight rate limit for handler testing."""
    return Config(
        gemini_api_key="test-key-not-real",
        gemini_model="gemini-2.5-flash",
        max_tokens=1024,
        api_timeout_seconds=5.0,
        api_max_retries=2,
        api_retry_base_delay_seconds=0.01,
        rate_limit_default="1 per minute",
        max_resume_length=15000,
        max_target_role_length=200,
        max_job_description_length=5000,
        min_resume_length=50,
        flask_secret_key="test-secret",
        cors_origins="",
        log_level="WARNING",
    )


def test_analyze_handler_rate_limit_message(
    monkeypatch,
    strict_rate_limit_config,
    valid_analyze_payload,
    valid_model_response,
):
    """A second call from the same session should return the rate-limit message."""
    import app as gradio_app

    monkeypatch.setattr(gradio_app, "config", strict_rate_limit_config)
    gradio_app._rate_limit_buckets.clear()
    request = FakeRequest(client=FakeClient(host="10.0.0.99"))

    with patch("app.analyze_resume", return_value=valid_model_response):
        first = analyze_handler(
            valid_analyze_payload["resume"],
            valid_analyze_payload["target_role"],
            valid_analyze_payload["job_description"],
            request,
        )
        second = analyze_handler(
            valid_analyze_payload["resume"],
            valid_analyze_payload["target_role"],
            valid_analyze_payload["job_description"],
            request,
        )

    assert "## Score: 72/100" in first
    assert "Rate limit exceeded" in second
