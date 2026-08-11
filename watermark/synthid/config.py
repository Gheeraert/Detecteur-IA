"""Default SynthID watermarking configuration.

Matches google-deepmind/synthid-text's DEFAULT_WATERMARKING_CONFIG
(src/synthid_text/synthid_mixin.py) — the g-value computation only agrees
with text watermarked using these exact keys and ngram length.
"""

import torch

DEFAULT_NGRAM_LEN = 5  # H=4 context window size in the paper.

DEFAULT_KEYS = [
    654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468,
    990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960,
]

DEFAULT_CONTEXT_HISTORY_SIZE = 1024

# Heuristic, not a statistically calibrated threshold (see README of
# google-deepmind/synthid-text, Appendix A.3.1): weighted-mean scores for
# unwatermarked text center around 0.5, so this splits the difference. If you
# need a specific false-positive rate, calibrate empirically for your text
# lengths rather than trusting this default.
DEFAULT_THRESHOLD = 0.6


def default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
