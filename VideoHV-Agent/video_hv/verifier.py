"""Tool-calling verifier that resolves distinguishing clues against video evidence."""

from __future__ import annotations

import json
import os
import time

from openai import OpenAI

from video_hv.config import LLM_API_KEY, LLM_BASE_URL
from video_hv.vision_tools import caption, create_tool_schema, encode_image, parse_frame_range

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_available_tools():
    return {"caption": caption}


AVAILABLE_TOOLS = get_available_tools()

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def execute_caption_call(function_name: str, frame_range, encode_img, Caption):
    return caption(encode_img, frame_range)


def execute_detection_call(function_name: str, frame_range, object, video_path):
    pass


def execute_tracking_call(function_name: str, video_path, video_range, frame_range_str, object):
    pass


def external_model_answer(
    question,
    hypotheses,
    video_id,
    clue,
    caption,
    video,
    model="",
):
    images_encoded = ["data:image/jpg;base64," + encode_image(p) for p in video]
    verify_process = []
    video_path = os.path.dirname(video[0])
    tools = []
    for func_name, func in AVAILABLE_TOOLS.items():
        tool_schema = create_tool_schema(func_name, func)
        if tool_schema:
            tools.append(tool_schema)

    detection_time = 0
    caption_time = 0
    tracking_time = 0
    W_prompt = f"""
            [Instruction]
            You are a reasoning verifier.
            You will verify whether a distinguishing clue derived from multiple hypotheses can be supported or refuted by the provided video context information.
            Follow this reasoning plan:
            1. Clue Understanding
                Reinterpret the clue in plain terms: what needs to be verified?
                Identify what kind of evidence (objects, actions, relations, timing, spatial layout) would support or contradict it.

            2. Contextual Search
                Examine the given video summary, transcript, or extracted description.
                Find sentences, events, or visual cues that are relevant to the clue.

            3. Reasoning Trace
                Step by step, explain how the found evidence relates to the clue.
                Explicitly note whether the evidence supports or contradicts each hypothesis.
                Maintain logical transparency: show what was observed, what inference was drawn, and what conclusion followed.

            4. Final Output
                Summarize whether the clue was verified, partially verified, or not found.
                Provide a short reasoning trace and the relevant evidence snippet.

            [Inputs]
            Question: {question}
            Clue: {clue}
            Context summary (video caption / description): {caption}

            [Output format]
            "clue_understanding": "Describe what is being tested.",
            "evidence_found": "Summarize key details from the context or from tool retrieval.",
            "reasoning_trace": [
              "Step 1: Identify action/event ...",
              "Step 2: Compare with clue condition ...",
              "Step 3: Draw inference ..."
            ],
            "verification_result": "verified / partially_verified / not_verified",
            If you cannot verify clues well and which frames might be helpful for verification, describe what additional evidence is needed.
            Call only one tool at a time and frame_range no more than 5 frames.
            If the verification result is not_verified, it means the clue is not good and needs to be regenerated.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant with access to vision analysis tools."}],
        },
        {"role": "user", "content": [{"type": "text", "text": W_prompt}]},
    ]

    max_iterations = 2
    iteration = 0
    frame_ranges = []

    while iteration < max_iterations:
        iteration += 1
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )

        response_message = completion.choices[0].message
        verify_process.append(response_message.content)
        if response_message.tool_calls:
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_response = ""
                try:
                    if function_name.lower() == "detection":
                        detection_start = time.perf_counter()
                        function_args = json.loads(tool_call.function.arguments)
                        frame_range = function_args.get("frame_range", "0-0")
                        frame_ranges = parse_frame_range(frame_range)
                        object = function_args.get("object", "person")
                        function_response = execute_detection_call("detection", frame_range, object, video)
                        detection_end = time.perf_counter()
                        detection_time = detection_end - detection_start
                    elif function_name.lower() == "caption":
                        caption_start = time.perf_counter()
                        function_args = json.loads(tool_call.function.arguments)
                        frame_range = function_args.get("frame_range", "0-0")
                        frame_ranges = parse_frame_range(frame_range)
                        function_response = execute_caption_call("caption", frame_range, images_encoded, caption)
                        caption_end = time.perf_counter()
                        verify_process.append(f"tool: caption, frame_range: {frame_range}, response: {function_response}")
                        caption_time = caption_end - caption_start
                    elif function_name.lower() == "tracking":
                        tracking_start = time.perf_counter()
                        function_args = json.loads(tool_call.function.arguments)
                        frame_range = function_args.get("frame_range", "0-0")
                        frame_ranges = parse_frame_range(frame_range)
                        object = function_args.get("object", "person")
                        function_response = execute_tracking_call("tracking", video_path, video, frame_range, object)
                        tracking_end = time.perf_counter()
                        tracking_time = tracking_end - tracking_start
                    else:
                        function_response = f"Tool '{function_name}' not implemented."
                except Exception as e:
                    function_response = f"Error executing {function_name}: {str(e)}"
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                    }
                )
        else:
            final_response = response_message.content
            verify_process.append(final_response)
            return final_response, detection_time, caption_time, tracking_time, verify_process

    user_prompt = [{"type": "text", "text": "Answer directly without using tools."}]
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.2,
    )
    response_message = completion.choices[0].message
    final_response = response_message.content
    verify_process.append(final_response)
    return final_response, detection_time, caption_time, tracking_time, verify_process
