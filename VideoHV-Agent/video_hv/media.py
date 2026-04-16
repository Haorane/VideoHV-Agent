"""Frame sampling and caption indexing helpers."""

from __future__ import annotations

import os

import numpy as np


def sample_frames(total_frames: int, num_samples: int) -> list[int]:
    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    return frame_indices.tolist()


def get_image_path(frame_indices: list[int], video_path: str) -> list[str]:
    all_files = sorted(os.listdir(video_path))
    selected_filenames = [
        f"{video_path}/{all_files[i]}" for i in frame_indices if i < len(all_files)
    ]
    return selected_filenames


def read_clips(clips: list) -> dict:
    return {f"clip {idx}": clip for idx, clip in enumerate(clips)}


def read_cap(cap: list, frame_indices: list[int]) -> dict:
    return {f"frame {idx}": cap[idx] for idx in frame_indices}
