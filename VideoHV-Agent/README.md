# VideoHV

Structured **hypothesis generation → distinction judging → tool-based verification → answer selection** for video question answering.

## Layout

- `video_hv/config.py` — data locations and API endpoints (overridable via environment variables).
- `video_hv/media.py` — frame sampling and caption dict helpers.
- `video_hv/vision_tools.py` — caption tool + OpenAI function schemas.
- `video_hv/verifier.py` — `external_model_answer` loop with optional caption tool calls.
- `video_hv/pipelines/egoschema_openai/` — OpenAI structured-parse API for the same stages (split into `schemas`, `prompts`, `openai_stages`, `runner`, `cli`).
- `load_data/` — dataset JSON assets (unchanged structure from the original project).

## Setup

```bash
cd VideoHV
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `VIDEOHV_LLM_API_KEY` | API key for that LLM (default `EMPTY`). |
| `VIDEOHV_CAPTION_BASE_URL` | OpenAI-compatible endpoint for frame captioning. |
| `VIDEOHV_CAPTION_API_KEY` | Key for captioning (falls back to `OPENAI_API_KEY`). |
| `VIDEOHV_CAPTION_MODEL` | Caption model name (default `gpt-4o-2024-11-20`). |
| `VIDEOHV_STRUCTURED_LLM_BASE_URL` | Endpoint for `egoschema_openai` (default `https://api.openai.com/v1`). |
| `VIDEOHV_STRUCTURED_LLM_API_KEY` | Key for structured parse API (falls back to `OPENAI_API_KEY`). |

Frame image roots inside the pipelines match the originals (`egoschema_openai`); override paths in those modules if your filesystem layout differs.

## Run

```bash
video-hv-egoschema-openai
```

Results are written to `model_exp/egoschema_subset.json`; run logs to `output_ego.log` under the `VideoHV` directory.
