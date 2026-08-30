# Desk Review

AI-powered resume feedback web app. Paste your resume and target role; Desk Review returns a structured score, strengths, weaknesses, missing keywords, bullet rewrites, and next steps.

Built with **Gradio**, the **Google Gemini API**, and a keyword-grounding layer (`keywords.py`).

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # add GEMINI_API_KEY from Google AI Studio
python app.py
```

Open the local URL Gradio prints (default [http://localhost:7860](http://localhost:7860)).

### Tests

```bash
set GEMINI_API_KEY=ci-test-key-not-used
pytest -v
```

## Project structure

```
app.py              Gradio UI, rate limiting, health check
core.py             Config, schemas, Gemini integration
keywords.py         Role → keyword lookup (retrieval / grounding layer)
test_app.py         Pytest suite (mocked)
samples/            Example resumes
scripts/deploy_hf.py  Deploy to Hugging Face Spaces
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for request flow, design decisions, and interview talking points.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | *(required)* | Server-side API key from [Google AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Model ID |
| `RATE_LIMIT_DEFAULT` | `10 per minute` | Per-session limit (`3 per minute` recommended for a public HF Space) |

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
   pip install -r requirements-dev.txt
   set HF_TOKEN=hf_...          # write token from huggingface.co/settings/tokens
   python scripts/deploy_hf.py
   ```

5. In the Space's **Settings → Hardware**, select **ZeroGPU** (free tier).
6. In **Settings → Variables and secrets**, add:
   - `GEMINI_API_KEY` — secret
   - `RATE_LIMIT_DEFAULT` — `3 per minute` (recommended for a public demo)
7. **Before going public:** review [Google AI Studio usage limits](https://ai.google.dev/gemini-api/docs/rate-limits) for your API key.

Live URL once deployed: [huggingface.co/spaces/HaseebArif11/desk-review](https://huggingface.co/spaces/HaseebArif11/desk-review)

## Known limitations

- Hand-curated keywords (limited role coverage)
- In-memory per-session rate limiter — not shared across multiple replicas
- No user accounts or resume storage (by design)
- Hugging Face ZeroGPU Spaces have a free daily GPU quota (~3.5–5 minutes/day). Resume analysis calls the Gemini API only and does **not** need GPU; `analyze_handler` is intentionally **not** decorated with `@spaces.GPU()` so that quota is not wasted on API-only work
- On Gemini's free tier, submitted resume text may be used by Google to improve their models
