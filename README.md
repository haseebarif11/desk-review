# Desk Review

AI-powered resume feedback web app. Paste your resume and target role; Desk Review returns a structured score, strengths, weaknesses, missing keywords, bullet rewrites, and next steps.

Built with **Gradio**, the **Anthropic API**, and a keyword-grounding layer (`keywords.py`).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # add ANTHROPIC_API_KEY
python app.py
```

Open the local URL Gradio prints (default [http://localhost:7860](http://localhost:7860)).

### Docker (legacy)

Docker files remain for optional self-hosting; the primary UI is now Gradio (`python app.py`).

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up --build
```

### Tests

```bash
set ANTHROPIC_API_KEY=ci-test-key-not-used
pytest -v
```

## Project structure

```
app.py              Gradio UI, rate limiting, health check
core.py             Config, schemas, Anthropic integration
keywords.py         Role → keyword lookup (retrieval / grounding layer)
test_app.py         Pytest suite (mocked)
samples/            Example resumes
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for request flow, design decisions, and interview talking points.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Server-side API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Model ID |
| `RATE_LIMIT_DEFAULT` | `10 per minute` | Per-IP limit (`3 per minute` recommended for a public HF Space) |
| `CORS_ORIGINS` | *(empty)* | Legacy Flask option (unused by Gradio UI) |

See `.env.example` for all options.

## Deploy on Hugging Face Spaces

1. Push this repo to GitHub first — that stays your primary repo.
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space) → SDK: **Gradio** → Visibility: **Public**. This creates an empty Space repo with its own auto-generated README.
3. **Don't** push your GitHub repo directly over the Space — that overwrites HF's config header. Instead:
   - Clone the empty Space repo separately
   - Copy your project files into it
   - Prepend this block to the top of that copy's README.md:

   ```yaml
   ---
   title: Desk Review
   emoji: 📄
   colorFrom: blue
   colorTo: purple
   sdk: gradio
   sdk_version: 5.0.0
   app_file: app.py
   pinned: false
   ---
   ```

4. Commit and push to the Space repo.

   **Or use the deploy script** (after creating the empty Space once):

   ```bash
   set HF_TOKEN=hf_...          # write token from huggingface.co/settings/tokens
   python scripts/deploy_hf.py
   ```

5. In the Space's **Settings → Hardware**, select **ZeroGPU** so the free tier applies (required even though this app calls the Anthropic API and does no local GPU inference).
6. In **Settings → Variables and secrets**, add:
   - `ANTHROPIC_API_KEY` — secret
   - `RATE_LIMIT_DEFAULT` — `3 per minute` (recommended for a public demo)
7. **Before going public:** set a spend limit on that key in the [Anthropic Console](https://console.anthropic.com/). A public, always-on demo calling a paid API without a spend cap is the real risk here.

Live URL once deployed: [huggingface.co/spaces/HaseebArif11/desk-review](https://huggingface.co/spaces/HaseebArif11/desk-review)

## Known limitations

- Hand-curated keywords (limited role coverage)
- In-memory per-IP rate limiter — not shared across multiple replicas
- No user accounts or resume storage (by design)
