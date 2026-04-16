"""Defaults for the EgoSchema × OpenAI structured-output pipeline."""

DEFAULT_STRUCTURED_MODEL = "gpt-4o-2024-11-20"
MAX_REFINEMENT_ROUNDS = 3
DISTINCTION_SCORE_THRESHOLD = 0.5
NUM_FRAME_SAMPLES = 180
NUM_CHOICE_OPTIONS = 5
VERIFIER_WORKERS = 3
# Historical default; override via video_hv.config if you centralize paths later.
FRAME_IMAGE_ROOT = "/hy-tmp/subset_image/"
