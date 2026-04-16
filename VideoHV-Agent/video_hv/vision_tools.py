"""Vision tool implementations and OpenAI function schemas for verification."""

from __future__ import annotations

import base64
import json
import os
import warnings

from openai import OpenAI

from video_hv.config import CAPTION_API_KEY, CAPTION_BASE_URL, CAPTION_MODEL

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
warnings.filterwarnings("ignore", message=".*requires_grad.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_client_caption = OpenAI(api_key=CAPTION_API_KEY, base_url=CAPTION_BASE_URL)


def parse_frame_range(frame_range_str: str) -> list[int]:
    if "-" in frame_range_str:
        start, end = map(int, frame_range_str.split("-"))
        return list(range(start, end + 1))
    return [int(frame_range_str)]


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def caption(img_encoded: list[str], frame_range: str) -> str:
    result = []
    frame_index = parse_frame_range(frame_range)
    image_contents = []
    for i in frame_index:
        if i >= len(img_encoded):
            continue
        img = img_encoded[i]
        if isinstance(img, str) and (
            img.startswith("data:image/jpg;base64,") or img.startswith("data:image/jpeg;base64,")
        ):
            image_contents.append({"type": "image_url", "image_url": {"url": img}})
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional image description assistant. For each image provided, "
                "generate a detailed description including people, objects, actions, and scenes. "
                "Output a numbered list, each item corresponding to the order of the images."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please describe each images in detail. For each image provided, generate a "
                        "detailed description including people, objects, actions, and scenes and output "
                        "a numbered list. Each number corresponds to the order of the images provided."
                    ),
                }
            ]
            + image_contents,
        },
    ]
    response = _client_caption.chat.completions.create(
        model=CAPTION_MODEL,
        messages=messages,
        temperature=0.3,
    )
    descriptions = response.choices[0].message.content.strip().split("\n")
    for idx, i in enumerate(frame_index):
        desc = descriptions[idx].strip() if idx < len(descriptions) else ""
        desc = desc.split(".", 1)[-1].strip() if "." in desc else desc
        result.append({"frame_id": i, "description": desc})
    return json.dumps(result, ensure_ascii=False)


def create_tool_schema(func_name: str, _func=None) -> dict | None:
    if func_name == "detection":
        return {
            "type": "function",
            "function": {
                "name": func_name,
                "description": (
                    "GroundingDINO zero-shot object detection. Detects arbitrary objects based on natural "
                    "language descriptions. Use when you need to identify specific objects, people, or "
                    "visual elements in the image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object": {
                            "type": "string",
                            "description": (
                                "Natural language description of objects to detect. Separate multiple "
                                "objects with dots, e.g., 'person . car . microwave oven . cup'"
                            ),
                        },
                        "frame_range": {
                            "type": "string",
                            "description": (
                                "The index of a single frame (e.g., '15') or a frame range in the format "
                                "'start-end' (e.g., '10-20'). No more than 20 frames"
                            ),
                        },
                        "box_threshold": {
                            "type": "number",
                            "description": (
                                "Bounding box confidence threshold (0-1), default 0.35. Lower values "
                                "detect more objects but may include false positives."
                            ),
                            "default": 0.35,
                        },
                        "text_threshold": {
                            "type": "number",
                            "description": (
                                "Text matching confidence threshold (0-1), default 0.25. Lower values "
                                "allow more flexible text matching."
                            ),
                            "default": 0.25,
                        },
                    },
                    "required": ["object", "frame_range"],
                },
            },
        }
    if func_name == "caption":
        return {
            "type": "function",
            "function": {
                "name": func_name,
                "description": (
                    "Generate detailed natural language description of the image content. Use when you "
                    "need to understand what's happening in the image, identify scenes, people, objects, "
                    "and actions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "frame_range": {
                            "type": "string",
                            "description": (
                                "The index of a single frame (e.g., '15') or a frame range in the format "
                                "'start-end' (e.g., '10-20'). No more than 20 frames"
                            ),
                        },
                    },
                    "required": ["frame_range"],
                },
            },
        }
    if func_name == "tracking":
        return {
            "type": "function",
            "function": {
                "name": func_name,
                "description": (
                    "Multi-frame object tracking. Tracks specified objects across a sequence of frames "
                    "and returns the tracking results for each frame."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object": {
                            "type": "string",
                            "description": (
                                "Natural language description of objects to track. Separate multiple "
                                "objects with dots, e.g., 'person . red cup'"
                            ),
                        },
                        "frame_range": {
                            "type": "string",
                            "description": (
                                "The index of a single frame (e.g., '15') or a frame range in the format "
                                "'start-end' (e.g., '10-20'). No more than 20 frames"
                            ),
                        },
                    },
                    "required": ["object", "frame_range"],
                },
            },
        }
    return None
