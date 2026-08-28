"""Pytest suite for Desk Review (mocked — no real API calls)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app import create_app
from core import (
    AnalyzeRequest,
    AnalyzeResponse,
    Config,
    analyze_resume,
)

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


@pytest.fixture
def test_config() -> Config:
    """Config for unit tests (no real API key needed)."""
    return Config(
        anthropic_api_key="test-key-not-real",
        anthropic_model="claude-sonnet-5",
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
    """Minimal valid request payload for route tests."""
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


def _mock_client(response_dict: dict) -> MagicMock:
    """Build a mock Anthropic client returning JSON text."""
    client = MagicMock()
    block = MagicMock()
    block.text = json.dumps(response_dict)
    message = MagicMock()
    message.content = [block]
    client.messages.create.return_value = message
    return client


def test_analyze_resume_returns_valid_schema(
    test_config, valid_analyze_payload, valid_model_response
):
    """Mocked analysis returns all required keys with score in range."""
    request = AnalyzeRequest.model_validate(valid_analyze_payload)
    client = _mock_client(valid_model_response)

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
    for sample_path in SAMPLES_DIR.glob("*.txt"):
        request = AnalyzeRequest(
            resume=sample_path.read_text(encoding="utf-8"),
            target_role="Software Engineer",
        )
        result = analyze_resume(
            request,
            config=test_config,
            client=_mock_client(valid_model_response),
        )
        assert 0 <= result["score"] <= 100


def test_keywords_grounding_in_prompt(
    test_config, valid_analyze_payload, valid_model_response
):
    """The Anthropic call should include role keywords from keywords.py."""
    request = AnalyzeRequest.model_validate(valid_analyze_payload)
    client = _mock_client(valid_model_response)

    analyze_resume(request, config=test_config, client=client)

    user_content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "python" in user_content.lower()
    assert "Target role:" in user_content


@pytest.fixture
def client(test_config, valid_analyze_payload, valid_model_response):
    """Flask test client with Anthropic mocked."""
    app = create_app(test_config)
    with patch("app.analyze_resume") as mock_analyze:
        mock_analyze.return_value = valid_model_response
        with app.test_client() as test_client:
            yield test_client, mock_analyze, valid_analyze_payload


def test_analyze_missing_fields_returns_400(client):
    """POST without required fields should return 400."""
    test_client, _, _ = client
    response = test_client.post(
        "/api/analyze",
        data=json.dumps({"target_role": "Engineer"}),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Validation failed"


def test_analyze_valid_input_returns_200(client):
    """Valid payload should return 200 with structured feedback."""
    test_client, mock_analyze, payload = client
    response = test_client.post(
        "/api/analyze",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.get_json()["score"] == 72
    mock_analyze.assert_called_once()


def test_analyze_malformed_model_output_returns_502(test_config, valid_analyze_payload):
    """Route should surface 502 when model output fails schema validation."""
    app = create_app(test_config)
    with patch(
        "app.analyze_resume",
        side_effect=ValidationError.from_exception_data(
            "AnalyzeResponse",
            [
                {
                    "type": "greater_than_equal",
                    "loc": ("score",),
                    "input": 200,
                    "ctx": {"ge": 0},
                }
            ],
        ),
    ):
        with app.test_client() as test_client:
            response = test_client.post(
                "/api/analyze",
                data=json.dumps(valid_analyze_payload),
                content_type="application/json",
            )
    assert response.status_code == 502
    assert response.get_json()["code"] == "INVALID_MODEL_OUTPUT"


def test_analyze_rejects_non_json(test_config):
    """Non-JSON requests should return 400."""
    app = create_app(test_config)
    with app.test_client() as test_client:
        response = test_client.post(
            "/api/analyze",
            data="not json",
            content_type="text/plain",
        )
    assert response.status_code == 400


@pytest.fixture
def strict_rate_limit_config() -> Config:
    """Config with a tight rate limit for 429 testing."""
    return Config(
        anthropic_api_key="test-key-not-real",
        anthropic_model="claude-sonnet-5",
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


def test_healthz_returns_ok(test_config):
    """Health check endpoint should return 200 without rendering the UI."""
    app = create_app(test_config)
    with app.test_client() as test_client:
        response = test_client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_analyze_rate_limit_returns_429(
    strict_rate_limit_config, valid_analyze_payload, valid_model_response
):
    """Second request within the window should return 429."""
    app = create_app(strict_rate_limit_config)
    payload = json.dumps(valid_analyze_payload)
    with patch("app.analyze_resume", return_value=valid_model_response):
        with app.test_client() as test_client:
            first = test_client.post(
                "/api/analyze",
                data=payload,
                content_type="application/json",
            )
            second = test_client.post(
                "/api/analyze",
                data=payload,
                content_type="application/json",
            )
    assert first.status_code == 200
    assert second.status_code == 429
