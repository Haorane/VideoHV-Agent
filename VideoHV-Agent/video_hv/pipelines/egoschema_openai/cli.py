"""CLI entry: load EgoSchema assets, run pool, write JSONL."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from video_hv.config import DATA_DIR, MODEL_EXP, PROJECT_ROOT, STRUCTURED_LLM_API_KEY, STRUCTURED_LLM_BASE_URL

from .constants import VERIFIER_WORKERS
from .runner import run_single_video_question


def main() -> None:
    annotation_path = DATA_DIR / "egoschema" / "egoschema_subset_anno.json"
    summary_path = DATA_DIR / "egoschema" / "summaries_egoschema_gpt-3.5-turbo-1106_dpcknnsplit_4clips_A6000.json"
    per_frame_caption_path = DATA_DIR / "egoschema" / "action_captions_egoschema_lavila_llovi.json"

    MODEL_EXP.mkdir(parents=True, exist_ok=True)
    output_json_path = MODEL_EXP / "egoschema_subset.json"

    annotations = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    per_frame_captions_by_video = json.loads(Path(per_frame_caption_path).read_text(encoding="utf-8"))
    video_summaries_by_video = json.loads(Path(summary_path).read_text(encoding="utf-8"))

    shared_run_log: dict[str, dict] = {}
    logging.basicConfig(
        filename=str(PROJECT_ROOT / "output_ego.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    structured_client = OpenAI(
        api_key=STRUCTURED_LLM_API_KEY,
        base_url=STRUCTURED_LLM_BASE_URL,
    )

    video_ids = list(annotations.keys())
    tasks = [
        (
            video_id,
            annotations[video_id],
            per_frame_captions_by_video[video_id],
            video_summaries_by_video[video_id],
            shared_run_log,
            structured_client,
        )
        for video_id in video_ids
    ]

    def _run(task: tuple) -> None:
        vid, ann, caps, summary, log, client = task
        run_single_video_question(
            vid,
            ann,
            caps,
            summary,
            log,
            structured_client=client,
        )

    with ThreadPoolExecutor(max_workers=VERIFIER_WORKERS) as executor:
        for _ in tqdm(executor.map(_run, tasks), total=len(tasks)):
            pass

    with open(output_json_path, "w", encoding="utf-8") as handle:
        for row in shared_run_log.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
