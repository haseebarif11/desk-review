# Desk Review — Architecture

Short reference for how the app works and why it is built this way.

## Request flow

```mermaid
sequenceDiagram
    participant User
    participant Gradio
    participant Core
    participant Keywords
    participant Gemini

    User->>Gradio: Paste resume + target role
    Gradio->>Gradio: Pydantic input validation
    Gradio->>Gradio: Per-session rate limit check
    Gradio->>Core: analyze_resume(...)
    Core->>Keywords: get_keywords_for_role(role)
    Keywords-->>Core: Curated keyword list
    Core->>Gemini: Prompt with resume + keywords
    Gemini-->>Core: JSON feedback
    Core->>Core: Parse + schema validate output
    Core-->>Gradio: Structured dict
    Gradio->>User: Render Markdown feedback
```

## Components

### Gradio UI (`app.py`)

- Gradio Blocks form for resume, target role, and optional job description
- `GET /healthz` — liveness probe on Gradio's underlying FastAPI app
- In-memory per-session rate limiting keyed by Gradio's `session_hash`

Responsibilities: validation, rate limiting, calling `core.analyze_resume`, and rendering results as Markdown.

**ZeroGPU note:** The Space can run on Hugging Face ZeroGPU hardware, but `analyze_handler` is **not** wrapped in `@spaces.GPU()`. Analysis is a remote Gemini API call with no local GPU work; decorating it would burn the free daily ZeroGPU quota (~3.5–5 min/day) for nothing.

### Analysis core (`core.py`)

Config, Pydantic schemas, Gemini client with retries, and `analyze_resume` — shared business logic with no web-framework imports.

### Keyword retrieval (`keywords.py`)

**Why it exists:** Large language models can hallucinate “missing skills” that are irrelevant to a role. A small, hand-curated role → keyword map gives the model a **grounded checklist** of terms to evaluate against the resume.

This is intentionally lightweight retrieval — not RAG over documents:

| Approach | Pros | Cons |
|----------|------|------|
| Hand-curated keywords (current) | Fast, no infra, easy to explain in interviews | Limited roles, manual upkeep |
| Vector DB over job postings | Broader coverage | More complexity, embedding cost |
| No retrieval | Simpler | Less consistent keyword suggestions |

**Lookup logic:** Normalize the role string → exact match → partial substring match → generic fallback keywords.

## Error handling

| User-facing message | When |
|-------|------|
| Validation failed | Invalid input (Pydantic) |
| Rate limit exceeded | Per-session limit in Gradio handler |
| AI returned malformed data | Unparseable or schema-invalid AI output |
| AI service is busy / unavailable / timed out | Gemini API errors |

## Security model

- `GEMINI_API_KEY` is read only on the server — never logged or sent to the browser

## Interview talking points

1. **Defense in depth:** validate on input (Pydantic), enforce rate limits, and validate AI output again before display.
2. **Retrieval without over-engineering:** `keywords.py` shows grounding without jumping to a vector DB on day one.
3. **Operational hygiene:** rate limits, retries, timeouts, and no PII in logs.
4. **Right-sized infrastructure:** Gradio on HF Spaces for a free public demo; no `@spaces.GPU()` on API-only handlers.
