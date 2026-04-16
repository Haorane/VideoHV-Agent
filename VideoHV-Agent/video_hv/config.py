"""Paths and environment-driven settings for VideoHV."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "load_data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_EXP = PROJECT_ROOT / "model_exp"


def _optional_env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


# Local LLM (Qwen-compatible OpenAI API)
LLM_BASE_URL = _optional_env("VIDEOHV_LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = _optional_env("VIDEOHV_LLM_API_KEY", "EMPTY")

# Caption / vision tool backend (OpenAI-compatible)
CAPTION_BASE_URL = _optional_env("VIDEOHV_CAPTION_BASE_URL", "https://api.openai.com/v1")
CAPTION_API_KEY = _optional_env("VIDEOHV_CAPTION_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
CAPTION_MODEL = _optional_env("VIDEOHV_CAPTION_MODEL", "gpt-4o-2024-11-20")

# Hypothesis/judge/answer via OpenAI structured parse (egoschema_openai pipeline)
STRUCTURED_LLM_BASE_URL = _optional_env("VIDEOHV_STRUCTURED_LLM_BASE_URL", "https://api.openai.com/v1")
STRUCTURED_LLM_API_KEY = os.environ.get("VIDEOHV_STRUCTURED_LLM_API_KEY") or os.environ.get(
    "OPENAI_API_KEY", ""
)
