"""Desk Review — Flask application."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import ValidationError

from core import AnalyzeRequest, Config, analyze_resume, configure_logging

logger = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> Flask:
    """Application factory for Flask and tests."""
    app_config = config or Config.from_env()
    configure_logging(app_config.log_level)

    app = Flask(__name__)
    app.config["DESK_REVIEW_CONFIG"] = app_config
    app.secret_key = app_config.flask_secret_key

    if app_config.cors_origins:
        origins = [origin.strip() for origin in app_config.cors_origins.split(",")]
        CORS(app, resources={r"/api/*": {"origins": origins}})
        logger.info("CORS enabled for origins: %s", origins)

    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=[app_config.rate_limit_default],
        storage_uri="memory://",
    )

    @app.route("/healthz")
    @limiter.exempt
    def healthz() -> tuple[Any, int]:
        """Lightweight liveness probe for platform health checks."""
        return jsonify({"status": "ok"}), 200

    @app.route("/")
    def index() -> str:
        """Serve the single-page frontend."""
        return render_template("index.html")

    @app.route("/api/analyze", methods=["POST"])
    @limiter.limit(app_config.rate_limit_default)
    def analyze() -> tuple[Any, int]:
        """Analyze a resume and return structured feedback."""
        started = time.perf_counter()
        client_ip = get_remote_address()

        if not request.is_json:
            logger.warning("analyze rejected: non-json request ip=%s", client_ip)
            return jsonify({"error": "Content-Type must be application/json"}), 400

        payload = request.get_json(silent=True)
        if payload is None:
            logger.warning("analyze rejected: invalid json ip=%s", client_ip)
            return jsonify({"error": "Invalid JSON body"}), 400

        try:
            analyze_request = AnalyzeRequest.model_validate(payload)
        except ValidationError as exc:
            logger.warning(
                "analyze rejected: validation error ip=%s errors=%s",
                client_ip,
                exc.error_count(),
            )
            return jsonify({"error": "Validation failed", "details": exc.errors()}), 400

        resume_chars = len(analyze_request.resume)
        role = analyze_request.target_role
        logger.info(
            "analyze started ip=%s role=%r resume_chars=%d has_job_desc=%s",
            client_ip,
            role,
            resume_chars,
            bool(analyze_request.job_description),
        )

        try:
            result = analyze_resume(analyze_request, app_config)
        except anthropic.RateLimitError:
            logger.error("analyze failed: anthropic rate limit ip=%s", client_ip)
            return (
                jsonify({"error": "AI service is busy. Please try again shortly."}),
                503,
            )
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                logger.error(
                    "analyze failed: anthropic server error ip=%s status=%s",
                    client_ip,
                    exc.status_code,
                )
                return jsonify({"error": "AI service temporarily unavailable."}), 503
            logger.error(
                "analyze failed: anthropic client error ip=%s status=%s",
                client_ip,
                exc.status_code,
            )
            return jsonify({"error": "AI service request failed."}), 502
        except anthropic.APITimeoutError:
            logger.error("analyze failed: anthropic timeout ip=%s", client_ip)
            return jsonify({"error": "AI service timed out. Please try again."}), 504
        except anthropic.APIError as exc:
            logger.error(
                "analyze failed: anthropic api error ip=%s msg=%s", client_ip, exc
            )
            return jsonify({"error": "AI service error."}), 502
        except json.JSONDecodeError:
            logger.error("analyze failed: unparseable model json ip=%s", client_ip)
            return (
                jsonify(
                    {
                        "error": "AI returned malformed data",
                        "code": "INVALID_MODEL_OUTPUT",
                    }
                ),
                502,
            )
        except ValidationError:
            logger.error(
                "analyze failed: model output schema mismatch ip=%s", client_ip
            )
            return (
                jsonify(
                    {
                        "error": "AI returned malformed data",
                        "code": "INVALID_MODEL_OUTPUT",
                    }
                ),
                502,
            )
        except ValueError as exc:
            logger.error("analyze failed: %s ip=%s", exc, client_ip)
            return jsonify({"error": str(exc)}), 502

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "analyze completed ip=%s role=%r score=%s latency_ms=%.1f",
            client_ip,
            role,
            result.get("score"),
            elapsed_ms,
        )
        return jsonify(result), 200

    return app


# Gunicorn entrypoint (Docker / Render)
try:
    load_dotenv()
    application = create_app()
except ValueError:
    application = None  # type: ignore[misc, assignment]


if __name__ == "__main__":
    load_dotenv()
    create_app().run(debug=False, host="0.0.0.0", port=5000)
