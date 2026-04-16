"""Per-video execution loop: hypotheses → distinctness → verification → answer."""

from __future__ import annotations

import time
from random import randint

from openai import OpenAI

from video_hv.media import get_image_path, read_cap, sample_frames
from video_hv.verifier import external_model_answer

from .constants import (
    DEFAULT_STRUCTURED_MODEL,
    DISTINCTION_SCORE_THRESHOLD,
    FRAME_IMAGE_ROOT,
    MAX_REFINEMENT_ROUNDS,
    NUM_CHOICE_OPTIONS,
    NUM_FRAME_SAMPLES,
)
from .openai_stages import (
    generate_initial_hypotheses,
    judge_hypothesis_distinctness,
    regenerate_hypotheses_after_failed_verification,
    regenerate_hypotheses_after_low_distinction,
    select_answer_from_verification,
)


def run_single_video_question(
    video_id: str,
    annotation: dict,
    per_frame_captions: list,
    video_summary_bundle: dict,
    shared_run_log: dict,
    *,
    structured_client: OpenAI,
    frame_image_root: str = FRAME_IMAGE_ROOT,
    structured_model: str = DEFAULT_STRUCTURED_MODEL,
) -> None:
    frame_indices = sample_frames(len(per_frame_captions), NUM_FRAME_SAMPLES)
    image_indices = sample_frames(len(per_frame_captions), NUM_FRAME_SAMPLES)
    frame_paths = get_image_path(image_indices, f"{frame_image_root}{video_id}")
    sampled_frame_captions = read_cap(per_frame_captions, frame_indices)

    action_caption_summaries = video_summary_bundle["action_caption_summaries"]
    object_detection_summaries = video_summary_bundle["object_detections_summaries"]
    _clip_boundaries = video_summary_bundle["clip_boundaries"]

    question_text = annotation["question"]
    choice_texts = [annotation[f"option {i}"] for i in range(NUM_CHOICE_OPTIONS)]
    formatted_option_lines = [f"{i}. {text}" for i, text in enumerate(choice_texts)]

    verification_trace_text = ""
    prior_hypothesis_lines: list[str] = []

    elapsed_hypothesis_generation_s = 0.0
    elapsed_distinctness_judge_s = 0.0
    elapsed_verification_s = 0.0
    caption_time = 0.0
    answer_phase_start_s = 0.0
    answer_phase_end_s = 0.0

    hypothesis_generation_log: list[str] = []
    distinctness_log: list[str] = []
    verification_log: list[str] = []
    answer_selection_log: list[str] = []

    final_answer = 0
    distinguishing_clue = ""

    for refinement_round in range(MAX_REFINEMENT_ROUNDS):
        hypothesis_texts: list[str] = []
        option_labels: list[str] = []
        try:
            if refinement_round == 0:
                t0 = time.perf_counter()
                hypothesis_rows = generate_initial_hypotheses(
                    structured_client,
                    question_text,
                    action_caption_summaries,
                    object_detection_summaries,
                    formatted_option_lines,
                    model=structured_model,
                )
                t1 = time.perf_counter()
                hypothesis_generation_log.append(f"iterator: {refinement_round}, hypothesis: {hypothesis_rows}")
                elapsed_hypothesis_generation_s += t1 - t0
            else:
                t0 = time.perf_counter()
                hypothesis_rows = regenerate_hypotheses_after_failed_verification(
                    structured_client,
                    question_text,
                    action_caption_summaries,
                    object_detection_summaries,
                    formatted_option_lines,
                    verification_trace_text,
                    prior_hypothesis_lines,
                    model=structured_model,
                )
                t1 = time.perf_counter()
                hypothesis_generation_log.append(f"iterator: {refinement_round}, hypothesis: {hypothesis_rows}")
                elapsed_hypothesis_generation_s += t1 - t0

            for row in hypothesis_rows:
                hypothesis_texts.append(row["hypothesis"])
                option_labels.append(row["option"])

            if len(hypothesis_texts) != 1:
                prior_hypothesis_lines = [f"{idx + 1} Hypothesis: {text}\n" for idx, text in enumerate(hypothesis_texts)]
                hypotheses_block = "".join(prior_hypothesis_lines)

                judge_start = time.perf_counter()
                distinction_score, distinguishing_clue, distinction_reasons = judge_hypothesis_distinctness(
                    structured_client,
                    question_text,
                    hypotheses_block,
                    action_caption_summaries,
                    model=structured_model,
                )
                judge_end = time.perf_counter()
                distinctness_log.append(
                    f"score: {distinction_score}, clue: {distinguishing_clue}, reasons: {distinction_reasons}"
                )
                elapsed_distinctness_judge_s += judge_end - judge_start

                if distinction_score < DISTINCTION_SCORE_THRESHOLD:
                    generation_hypothesis_start = time.perf_counter()
                    hypothesis_rows = regenerate_hypotheses_after_low_distinction(
                        structured_client,
                        question_text,
                        action_caption_summaries,
                        object_detection_summaries,
                        option_labels,
                        distinction_reasons,
                        prior_hypothesis_lines,
                        model=structured_model,
                    )
                    generation_hypothesis_end = time.perf_counter()
                    hypothesis_generation_log.append(
                        f"score: {distinction_score}, regenerate_hypothesis: {hypothesis_rows}"
                    )
                    elapsed_hypothesis_generation_s += generation_hypothesis_end - generation_hypothesis_start
                    hypothesis_texts.clear()
                    option_labels.clear()
                    for row in hypothesis_rows:
                        hypothesis_texts.append(row["hypothesis"])
                        option_labels.append(row["option"])
                    prior_hypothesis_lines = [
                        f"{idx + 1} Hypothesis: {text}\n" for idx, text in enumerate(hypothesis_texts)
                    ]
                    hypotheses_block = "".join(prior_hypothesis_lines)

                verify_start = time.perf_counter()
                verification_trace_text, detection_time, caption_time, tracking_time, verifier_steps = (
                    external_model_answer(
                        question_text,
                        hypotheses_block,
                        video_id,
                        distinguishing_clue,
                        sampled_frame_captions,
                        frame_paths,
                    )
                )
                verification_log.append(f"iterator {refinement_round}, verify_process: {verifier_steps}")
                verify_end = time.perf_counter()
                elapsed_verification_s += verify_end - verify_start

                detection_time += detection_time
                caption_time += caption_time
                tracking_time += tracking_time

                answer_phase_start_s = time.perf_counter()
                final_answer, reasoning_summary, explanation = select_answer_from_verification(
                    structured_client,
                    question_text,
                    option_labels,
                    hypotheses_block,
                    verification_trace_text,
                    action_caption_summaries,
                    object_detection_summaries,
                    model=structured_model,
                )
                answer_phase_end_s = time.perf_counter()
                answer_selection_log.append(
                    f"final_answer: {final_answer}, reasoning_summary: {reasoning_summary}, explanation: {explanation}"
                )
                if "not_verified" in verification_trace_text:
                    continue
                break
            else:
                final_answer = option_labels[0][0]
                break
        except Exception as exc:  # noqa: BLE001
            print("发生错误", exc)
            final_answer = randint(0, NUM_CHOICE_OPTIONS - 1)

    label = int(annotation["truth"])
    corr = 1 if label == final_answer else 0
    print("The answer is correct!" if corr else "The answer is incorrect!")

    shared_run_log[video_id] = {
        "video_id": video_id,
        "question": question_text,
        "options": formatted_option_lines,
        "answer": final_answer,
        "label": label,
        "corr": corr,
        "generation_hypothesis_time": elapsed_hypothesis_generation_s,
        "judge_time": elapsed_distinctness_judge_s,
        "verify_time": elapsed_verification_s,
        "caption_time": caption_time,
        "answer_time": answer_phase_end_s - answer_phase_start_s,
        "generation_process": hypothesis_generation_log,
        "judge_process": distinctness_log,
        "verify_process": verification_log,
        "answer_process": answer_selection_log,
    }
