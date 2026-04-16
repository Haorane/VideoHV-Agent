"""OpenAI structured-parse calls for hypothesis, distinctness, and answer selection."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from video_hv.media import read_clips

from .constants import DEFAULT_STRUCTURED_MODEL
from .prompts import (
    GENERATE_HYPOTHESES,
    JUDGE_DISTINCTNESS,
    REGENERATE_AFTER_LOW_DISTINCTION,
    REGENERATE_AFTER_VERIFICATION,
    SELECT_ANSWER,
)
from .schemas import (
    AnswerSelectionResult,
    DistinctnessJudgmentResult,
    HypothesisGenerationResult,
    hypotheses_to_row_dicts,
)


def _message_content_text(completion: Any) -> str:
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("Structured parse returned empty message content.")
    return content.strip()


def generate_initial_hypotheses(
    client: OpenAI,
    question: str,
    action_caption_summaries: list,
    _object_detection_summaries: list,
    formatted_option_lines: list[str],
    model: str = DEFAULT_STRUCTURED_MODEL,
) -> list[dict[str, str]]:
    action_clip_context = read_clips(action_caption_summaries)
    prompt = GENERATE_HYPOTHESES.format(
        question=question,
        options=formatted_option_lines,
        action_clip_context=action_clip_context,
    )
    messages = [
        {"role": "system", "content": "You are a hypothesis generator."},
        {"role": "user", "content": prompt},
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=HypothesisGenerationResult,
        temperature=0.2,
    )
    parsed = HypothesisGenerationResult.model_validate_json(_message_content_text(completion))
    return hypotheses_to_row_dicts(parsed.hypotheses)


def regenerate_hypotheses_after_failed_verification(
    client: OpenAI,
    question: str,
    action_caption_summaries: list,
    _object_detection_summaries: list,
    formatted_option_lines: list[str],
    verification_feedback: str,
    previous_hypotheses_summary: list[str],
    model: str = DEFAULT_STRUCTURED_MODEL,
) -> list[dict[str, str]]:
    action_clip_context = read_clips(action_caption_summaries)
    prompt = REGENERATE_AFTER_VERIFICATION.format(
        question=question,
        options=formatted_option_lines,
        previous_hypotheses=previous_hypotheses_summary,
        verification_feedback=verification_feedback,
        action_clip_context=action_clip_context,
    )
    messages = [
        {"role": "system", "content": "You are a hypothesis generator."},
        {"role": "user", "content": prompt},
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=HypothesisGenerationResult,
        temperature=0.2,
    )
    parsed = HypothesisGenerationResult.model_validate_json(_message_content_text(completion))
    return hypotheses_to_row_dicts(parsed.hypotheses)


def regenerate_hypotheses_after_low_distinction(
    client: OpenAI,
    question: str,
    action_caption_summaries: list,
    _object_detection_summaries: list,
    option_labels: list[str],
    distinction_feedback: str,
    prior_hypothesis_blocks: list[str],
    model: str = DEFAULT_STRUCTURED_MODEL,
) -> list[dict[str, str]]:
    action_clip_context = read_clips(action_caption_summaries)
    prompt = REGENERATE_AFTER_LOW_DISTINCTION.format(
        question=question,
        options=option_labels,
        previous_hypotheses=prior_hypothesis_blocks,
        verification_feedback=distinction_feedback,
        action_clip_context=action_clip_context,
    )
    messages = [
        {"role": "system", "content": "You are a hypothesis generator."},
        {"role": "user", "content": prompt},
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=HypothesisGenerationResult,
        temperature=0.2,
    )
    parsed = HypothesisGenerationResult.model_validate_json(_message_content_text(completion))
    return hypotheses_to_row_dicts(parsed.hypotheses)


def judge_hypothesis_distinctness(
    client: OpenAI,
    question: str,
    hypotheses_block: str,
    action_caption_summaries: list,
    model: str = DEFAULT_STRUCTURED_MODEL,
) -> tuple[float, str, str]:
    action_clip_context = read_clips(action_caption_summaries)
    prompt = JUDGE_DISTINCTNESS.format(
        question=question,
        hypotheses_block=hypotheses_block,
        action_clip_context=action_clip_context,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=DistinctnessJudgmentResult,
        temperature=0.2,
    )
    judgment = DistinctnessJudgmentResult.model_validate_json(_message_content_text(completion))
    return judgment.score, judgment.clue, judgment.reasons


def select_answer_from_verification(
    client: OpenAI,
    question: str,
    option_labels: list[str],
    hypotheses_block: str,
    verification_results: str,
    action_caption_summaries: list,
    _object_detection_summaries: list,
    model: str = DEFAULT_STRUCTURED_MODEL,
) -> tuple[int, str, str]:
    action_clip_context = read_clips(action_caption_summaries)
    prompt = SELECT_ANSWER.format(
        question=question,
        options=option_labels,
        verification_results=verification_results,
        action_clip_context=action_clip_context,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=AnswerSelectionResult,
        temperature=0.2,
    )
    selection = AnswerSelectionResult.model_validate_json(_message_content_text(completion))
    return selection.final_answer, selection.reasoning_summary, selection.explanation


__all__ = [
    "generate_initial_hypotheses",
    "judge_hypothesis_distinctness",
    "regenerate_hypotheses_after_failed_verification",
    "regenerate_hypotheses_after_low_distinction",
    "select_answer_from_verification",
]
