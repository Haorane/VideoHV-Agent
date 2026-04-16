"""Prompt templates for hypothesis, distinctness, and answer stages."""

GENERATE_HYPOTHESES = """
[Instruction]
You are a reasoning planner. Given a video-related question and several answer options.
Pay special attention to, if the option is clearly inappropriate for the context, do not generate a corresponding hypothesis.
Rewrite remaining option into a testable hypothesis that can be verified from the video evidence.
Each hypothesis must specify:
- The key entities or objects involved,
- The main action or event,
- The temporal or causal relation among them.

[Inputs]
Question: {question}
Options: {options}
Context summary (optional): {action_clip_context}

[Output format]
Return option (It must be a complete option, not just numbers) and hypothesis.
"""

REGENERATE_AFTER_VERIFICATION = """
[Instruction]
You are a reasoning planner improving video-based hypotheses for a multiple-choice question.
Each hypothesis must correspond directly to one answer option.
The previous verification phase failed to confirm the clues, so you need to regenerate hypotheses that are:
    Still faithful to their respective answer options,
    More concretely testable from the video evidence,
    Better aligned with the available context.
Follow these reasoning steps:
1. Analyze Verification Feedback
    Identify why the previous verification failed (e.g., missing actions, unclear subject, no visible timing, or too abstract).
    Determine what type of visual or contextual evidence is available or missing.
2. Regenerate Option-based Hypotheses
    For each option, rewrite its hypothesis to make it more specific, testable, and observable within the context.
    Preserve each option's core meaning (do not alter its logical claim).
    Avoid unverifiable statements (e.g., emotions, intentions, unseen causes).
    If an option is clearly impossible to verify from context, note that no valid hypothesis can be formed.

[Inputs]
Question: {question}
Options: {options}
Previous hypotheses: {previous_hypotheses}
Verification feedback: {verification_feedback}
Context summary (optional): {action_clip_context}

[Output format]
Return option (It must be a complete option, not just numbers) and hypothesis.
"""

REGENERATE_AFTER_LOW_DISTINCTION = """
[Instruction]
You are a reasoning planner for video question answering.
The previous round of hypotheses had low distinction, meaning the hypotheses were too similar or not easily distinguishable based on video evidence.
Your task is to regenerate a new set of hypotheses—each still grounded in its respective option, but now rewritten to maximize their semantic and evidential differences so that they can be clearly tested and compared from the video.
Follow these reasoning steps:
1. Analyze Distinction Feedback
    Identify why the distinction score was low (e.g., overlapping actions, vague entities, lack of temporal or causal contrast).
    Determine what types of features (action, timing, spatial layout, emotion, interaction) can be emphasized to make them more distinct.
2. Regenerate Option-based Hypotheses
    Keep each hypothesis aligned with its original option's meaning.
    Rewrite it to highlight unique, observable, and discriminative aspects of the event.
    Ensure that each hypothesis involves different entities, actions, outcomes, or temporal relations whenever possible.

[Inputs]
Question: {question}
Options: {options}
Previous hypotheses: {previous_hypotheses}
Verification feedback: {verification_feedback}
Context summary (optional): {action_clip_context}

[Output format]
Return option (It must be a complete option, not just numbers) and hypothesis.
"""

JUDGE_DISTINCTNESS = """
[Instruction]
You are a reasoning analyst.
Given several rewritten hypotheses corresponding to different answer options of a video question, analyze how distinct they are in terms of testable evidence in the video.
Your task has three parts:

Compare Hypotheses:
Identify the core difference in entities, actions, events, causal/temporal relations, or visual evidence type (e.g., spatial layout, sequence of actions, emotional expression).

Generate a Distinguishing Clue:
Produce a concise description of what kind of video evidence could distinguish between these hypotheses.
For example: "Check whether the person hands the object before or after speaking," or "Verify if the dog appears indoors or outdoors."

Assign a Distinction Score (0–1):
Give a numeric score representing how distinguishable the hypotheses are based on likely video evidence, And provide the reasons for the score.:

0.0–0.3: Hypotheses are too similar or overlapping
0.4–0.6: Moderate distinction but may require nuanced understanding
0.7–1.0: Strongly distinct, clearly testable difference
If the distinction score < 0.5, subsequent hypotheses need to be regenerated.

[Inputs]
Question: {question}
Hypotheses: {hypotheses_block}
Context summary (optional): {action_clip_context}
"""

SELECT_ANSWER = """
[Instruction]
You are an answer reasoning agent.
Given a video question, and the verification results of the distinguishing clues, infer which option is best supported by the evidence.
You must:
1. Integrate Verification Results
2. Resolve Conflicts
    If multiple hypotheses are partially verified, reason which one is more strongly aligned with the overall context and clues.
    If all clues are unverified, indicate uncertainty and suggest that additional evidence is needed.
3. Generate a Transparent Reasoning Chain
    Summarize what was tested, what was found, and how it leads to your conclusion.
    Avoid just "guessing." Show cause–effect reasoning.
4. Output a Final Answer
    Specify the chosen option (0/1/2/...) and briefly justify it based on evidence.
5. Note that you should only refer to the Verification Results, not accept them wholesale.
[Inputs]
Question: {question}
Option: {options}
Clue Verification Results: {verification_results}
Context summary (optional): {action_clip_context}
[Output format]
"reasoning_summary": "Summarize the verification results.",
"conflict_resolution": "If any conflicting evidence exists, explain how it was resolved.",
"final_answer": "Choose from the options to return only numbers.",
"explanation": "Give a concise, human-readable rationale connecting evidence to conclusion."
"""
