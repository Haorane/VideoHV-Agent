"""Pydantic models for OpenAI `beta.chat.completions.parse` response shapes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HypothesisCandidate(BaseModel):
    option: str
    hypothesis: str


class HypothesisGenerationResult(BaseModel):
    """Structured output bundle; API schema keeps legacy key ``All_Hypothesis``."""

    model_config = ConfigDict(populate_by_name=True)

    hypotheses: list[HypothesisCandidate] = Field(alias="All_Hypothesis")


class AnswerSelectionResult(BaseModel):
    reasoning_summary: str
    conflict_resolution: str
    final_answer: int
    explanation: str


class DistinctnessJudgmentResult(BaseModel):
    score: float
    reasons: str
    clue: str


def hypotheses_to_row_dicts(items: list[HypothesisCandidate]) -> list[dict[str, str]]:
    return [h.model_dump() for h in items]
