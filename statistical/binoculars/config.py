"""Calibrated Binoculars decision thresholds, keyed by model pair.

Binoculars score = perplexity / cross-perplexity between an "observer" and a
"performer" model. The decision threshold is a property of *that specific
pair* (and, more weakly, of `use_bfloat16`) — it does not transfer to a
different pair. The Falcon-7B / Falcon-7B-Instruct entry below is the
threshold from the original paper (https://arxiv.org/abs/2401.12070),
optimized on their eval set.

If you swap in a different pair (e.g. for better French-language results),
run `scripts/calibrate_binoculars_threshold.py` against a labeled corpus and
add the resulting entry here — do not reuse the Falcon threshold for another
pair, it will silently misclassify.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibratedThresholds:
    accuracy: float  # optimized for F1
    low_fpr: float   # optimized for low false-positive rate (~0.01%)


CALIBRATED_THRESHOLDS: dict[tuple[str, str], CalibratedThresholds] = {
    ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"): CalibratedThresholds(
        accuracy=0.9015310749276843,
        low_fpr=0.8536432310785527,
    ),
}


def get_thresholds(observer_name_or_path: str, performer_name_or_path: str) -> CalibratedThresholds:
    key = (observer_name_or_path, performer_name_or_path)
    if key not in CALIBRATED_THRESHOLDS:
        raise ValueError(
            f"No calibrated threshold for model pair {key}. Binoculars' threshold "
            "is specific to the observer/performer pair it was calibrated on — "
            "run scripts/calibrate_binoculars_threshold.py against a labeled "
            "corpus for this pair and register the result in "
            "CALIBRATED_THRESHOLDS before using it for classification."
        )
    return CALIBRATED_THRESHOLDS[key]
