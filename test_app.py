"""Unit and integration test suite for Desk Review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import app as gradio_app
from core import (
    AnalyzeRequest,
    AnalyzeResponse,
    Config,
    GeminiClientError,
    GeminiServerError,
    _call_gemini_with_retry,
    _extract_json_text,
    analyze_resume,
)
from keywords import get_keywords_for_role

SAMPLES_DIR = Path(__file__).parent / "samples"


class FakeClient:
    """Mock Gradio client connection object."""

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.host = host


class FakeRequest:
    """Mock Gradio request context object."""

    def __init__(
        self,
        client: FakeClient | None = None,
        session_hash: str | None = "test-session",
    ) -> None:
        self.client = client or FakeClient()
        self.session_hash = session_hash


@pytest.fixture
def test_config() -> Config:
    """Standard test configuration with mock parameters."""
    return Config(
        gemini_api_key="test-api-key",
        gemini_model="gemini-3.6-flash",
        max_tokens=2048,
        api_timeout_seconds=5.0,
        api_max_retries=3,
        api_retry_base_delay_seconds=0.01,
        rate_limit_default="10 per minute",
        max_resume_length=15000,
        max_target_role_length=200,
        max_job_description_length=5000,
        min_resume_length=50,
        flask_secret_key="dev-secret",
        cors_origins="",
        log_level="WARNING",
    )


@pytest.fixture
def valid_analyze_payload() -> dict[str, str]:
    """Minimal valid request payload for handler and core tests."""
    return {
        "resume": (
            "Jane Doe\nSoftware Engineer | jane.doe@email.com\n\nEXPERIENCE\n"
            "Developed RESTful APIs and backend services using Python and FastAPI. "
            "Automated deployment workflows using Docker and CI/CD pipelines. "
            "Collaborated with product and design teams in agile sprints."
        ),
        "target_role": "Backend Engineer",
        "job_description": (
            "Seeking backend engineer skilled in Python, APIs, and microservices."
        ),
    }


@pytest.fixture
def valid_model_response() -> dict[str, Any]:
    """Valid structured response dict matching AnalyzeResponse schema."""
    return {
        "score": 85,
        "strengths": [
            "Strong background in Python API development",
            "Demonstrated CI/CD automation experience",
        ],
        "weaknesses": [
            "Missing quantitative impact metrics on bullet points",
            "No explicit cloud provider (AWS/GCP) mentions",
        ],
        "missing_keywords": ["aws", "redis"],
        "bullet_rewrites": [
            {
                "original": "Developed RESTful APIs and backend services.",
                "improved": (
                    "Architected and deployed 6 RESTful APIs in Python/FastAPI, "
                    "reducing latency by [X]% across microservices."
                ),
            }
        ],
        "next_steps": [
            "Quantify API performance achievements with concrete metrics",
            "Add AWS or GCP cloud environment experience if applicable",
            "Align summary statement with target Backend Engineer role",
        ],
    }


def _mock_http_client(
    response_dict: dict[str, Any], *, status_code: int = 200
) -> MagicMock:
    """Build a mock httpx client returning a Gemini REST payload."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = status_code
    response.text = json.dumps(response_dict)
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": json.dumps(response_dict)}],
                }
            }
        ]
    }
    client.post.return_value = response
    client.close = MagicMock()
    return client


# ---------------------------------------------------------------------------
# Keywords Unit Tests
# ---------------------------------------------------------------------------


def test_keywords_exact_and_partial_match():
    """Verify exact, partial, and case-insensitive role lookup."""
    # Exact match
    se_keywords = get_keywords_for_role("software engineer")
    assert "python" in se_keywords
    assert "git" in se_keywords

    # Substring match
    be_keywords = get_keywords_for_role("Senior Backend Engineer")
    assert "postgresql" in be_keywords
    assert "microservices" in be_keywords

    # Case normalization
    fe_keywords = get_keywords_for_role("  FRONTEND ENGINEER  ")
    assert "react" in fe_keywords
    assert "typescript" in fe_keywords


def test_keywords_fallback():
    """Verify unknown roles fall back to general professional keywords."""
    fallback = get_keywords_for_role("Quantum Astronomer")
    assert "communication" in fallback
    assert "problem solving" in fallback


# ---------------------------------------------------------------------------
# Request Schema Validation Tests
# ---------------------------------------------------------------------------


def test_analyze_request_valid(valid_analyze_payload):
    """Valid payload creates a validated AnalyzeRequest instance."""
    req = AnalyzeRequest.model_validate(valid_analyze_payload)
    assert req.target_role == "Backend Engineer"
    assert req.job_description is not None


def test_analyze_request_short_resume(valid_analyze_payload):
    """Resume shorter than 50 characters fails validation."""
    payload = {**valid_analyze_payload, "resume": "Short resume text."}
    with pytest.raises(ValidationError, match="at least 50 characters"):
        AnalyzeRequest.model_validate(payload)


def test_analyze_request_non_meaningful_resume(valid_analyze_payload):
    """Resume with insufficient alphabetic characters fails validation."""
    payload = {
        **valid_analyze_payload,
        "resume": "1234567890 1234567890 1234567890 1234567890 1234567890",
    }
    with pytest.raises(ValidationError, match="contain meaningful text"):
        AnalyzeRequest.model_validate(payload)


def test_analyze_request_invalid_target_role(valid_analyze_payload):
    """Target role missing alphabetic characters fails validation."""
    payload = {**valid_analyze_payload, "target_role": "12345"}
    with pytest.raises(ValidationError, match="must contain letters"):
        AnalyzeRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# Response Schema Validation Tests
# ---------------------------------------------------------------------------


def test_analyze_response_valid(valid_model_response):
    """Valid dictionary parses into AnalyzeResponse."""
    resp = AnalyzeResponse.model_validate(valid_model_response)
    assert resp.score == 85
    assert len(resp.strengths) == 2
    assert resp.to_api_dict()["score"] == 85


def test_analyze_response_invalid_score(valid_model_response):
    """Score out of 0-100 range raises ValidationError."""
    payload = {**valid_model_response, "score": 150}
    with pytest.raises(ValidationError):
        AnalyzeResponse.model_validate(payload)


# ---------------------------------------------------------------------------
# Rate Limiting Tests
# ---------------------------------------------------------------------------


def test_rate_limit_parsing():
    """_parse_rate_limit_per_minute converts valid rate strings."""
    assert gradio_app._parse_rate_limit_per_minute("10 per minute") == 10
    assert gradio_app._parse_rate_limit_per_minute(" 5 per minute ") == 5
    assert gradio_app._parse_rate_limit_per_minute("invalid") == 10


def test_rate_limit_enforcement(monkeypatch, test_config):
    """_check_rate_limit blocks calls when threshold is hit."""
    strict_config = Config(
        gemini_api_key="test-key",
        gemini_model="gemini-3.6-flash",
        max_tokens=1024,
        api_timeout_seconds=5.0,
        api_max_retries=2,
        api_retry_base_delay_seconds=0.01,
        rate_limit_default="2 per minute",
        max_resume_length=15000,
        max_target_role_length=200,
        max_job_description_length=5000,
        min_resume_length=50,
        flask_secret_key="secret",
        cors_origins="",
        log_level="WARNING",
    )
    monkeypatch.setattr(gradio_app, "config", strict_config)
    gradio_app._rate_limit_buckets.clear()

    session = "user-session-123"
    assert gradio_app._check_rate_limit(session) is None
    assert gradio_app._check_rate_limit(session) is None
    err = gradio_app._check_rate_limit(session)
    assert err is not None
    assert "Rate limit exceeded" in err


# ---------------------------------------------------------------------------
# Core Service & Gemini Client Tests
# ---------------------------------------------------------------------------


def test_extract_json_text():
    """Markdown code fences are correctly stripped from raw output."""
    raw_fenced = '```json\n{"score": 90}\n```'
    assert _extract_json_text(raw_fenced) == '{"score": 90}'
    raw_plain = '{"score": 90}'
    assert _extract_json_text(raw_plain) == '{"score": 90}'


def test_analyze_resume_success(
    test_config, valid_analyze_payload, valid_model_response
):
    """analyze_resume returns structured dict when client succeeds."""
    request = AnalyzeRequest.model_validate(valid_analyze_payload)
    client = _mock_http_client(valid_model_response)

    res = analyze_resume(request, config=test_config, http_client=client)
    assert res["score"] == 85
    assert "bullet_rewrites" in res


def test_call_gemini_with_retry_client_error(test_config):
    """4xx client errors raise GeminiClientError without retrying."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 400
    response.text = "Bad Request"
    client.post.return_value = response

    with pytest.raises(GeminiClientError):
        _call_gemini_with_retry(test_config, "test prompt", http_client=client)

    assert client.post.call_count == 1


def test_call_gemini_with_retry_server_error(test_config, valid_model_response):
    """5xx server errors retry up to max_retries before raising."""
    client = MagicMock()
    response = MagicMock()
    response.status_code = 500
    response.text = "Internal Server Error"
    client.post.return_value = response

    with pytest.raises(GeminiServerError):
        _call_gemini_with_retry(test_config, "test prompt", http_client=client)

    assert client.post.call_count == test_config.api_max_retries


# ---------------------------------------------------------------------------
# HTML Output & Gradio Handler Tests
# ---------------------------------------------------------------------------


def test_format_result_as_html(valid_model_response):
    """_format_result_as_html generates expected editorial markup."""
    html_out = gradio_app._format_result_as_html(valid_model_response)
    assert "desk-review-stamp" in html_out
    assert "Score: 85/100" in html_out
    assert "Strengths" in html_out
    assert "Weaknesses" in html_out
    assert "keyword-tag" in html_out
    assert "rewrite-card" in html_out


def test_analyze_handler_success(
    monkeypatch, test_config, valid_analyze_payload, valid_model_response
):
    """analyze_handler returns HTML report for a valid submission."""
    monkeypatch.setattr(gradio_app, "config", test_config)
    gradio_app._rate_limit_buckets.clear()

    with patch("app.analyze_resume", return_value=valid_model_response):
        result = gradio_app.analyze_handler(
            valid_analyze_payload["resume"],
            valid_analyze_payload["target_role"],
            valid_analyze_payload["job_description"],
            FakeRequest(),
        )

    assert "desk-review-stamp" in result
    assert "Score: 85/100" in result


def test_analyze_handler_validation_error(monkeypatch, test_config):
    """analyze_handler returns validation failure message for invalid input."""
    monkeypatch.setattr(gradio_app, "config", test_config)
    gradio_app._rate_limit_buckets.clear()

    result = gradio_app.analyze_handler(
        "Too short",
        "Software Engineer",
        "",
        FakeRequest(),
    )

    assert "Validation failed" in result


# ---------------------------------------------------------------------------
# Sample Resumes Verification Test
# ---------------------------------------------------------------------------


def test_sample_resume_files_validation():
    """All sample files in samples/ directory pass request schema validation."""
    sample_files = list(SAMPLES_DIR.glob("*.txt"))
    assert len(sample_files) > 0, "No sample resume files found"

    for sample_file in sample_files:
        text = sample_file.read_text(encoding="utf-8")
        req = AnalyzeRequest(
            resume=text,
            target_role="Software Engineer",
        )
        assert len(req.resume) >= 50
