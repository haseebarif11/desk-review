# Desk Review

AI-powered resume feedback web app. Paste your resume and target role; Desk Review returns a structured score, strengths, weaknesses, missing keywords, bullet rewrites, and next steps.

Built with **Flask**, the **Anthropic API**, and a vanilla HTML/CSS/JS frontend.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # add ANTHROPIC_API_KEY
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

### Docker

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up --build
```

Open [http://localhost:7860](http://localhost:7860).

### Tests

```bash
set ANTHROPIC_API_KEY=ci-test-key-not-used
pytest -v
```

## Project structure

```
app.py              Flask routes, rate limiting, health check
core.py             Config, schemas, Anthropic integration
keywords.py         Role → keyword lookup (retrieval / grounding layer)
test_app.py         Pytest suite (mocked)
templates/          HTML
static/             CSS + JS
samples/            Example resumes
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for request flow, design decisions, and interview talking points.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Server-side API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Model ID |
| `RATE_LIMIT_DEFAULT` | `10 per minute` | Per-IP limit |
| `CORS_ORIGINS` | *(empty)* | Comma-separated origins if frontend is separate |

See `.env.example` for all options.

## Deploy on Render

1. Push to GitHub → **New Web Service** on [Render](https://render.com).
2. Connect the repo and select **Docker** as the environment.
3. Set `ANTHROPIC_API_KEY` in Render's **Environment** tab (mark as secret).
4. Set the health check path to `/healthz`.
5. Deploy.

**Before connecting Render:** set a spend limit on your `ANTHROPIC_API_KEY` in the [Anthropic Console](https://console.anthropic.com/). This will be a live public endpoint.

## Known limitations

- Hand-curated keywords (limited role coverage)
- In-memory rate limiter (`storage_uri="memory://"`) — not shared across multiple replicas. Use Redis-backed Flask-Limiter storage if you scale horizontally.
- No user accounts or resume storage (by design)
