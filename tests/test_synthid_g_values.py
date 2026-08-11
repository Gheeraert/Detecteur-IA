"""SynthID g-value computation and scoring, tested without downloading a
tokenizer — synthetic token id sequences are enough."""

import torch

from watermark.synthid import config
from watermark.synthid.g_values import SynthIDGValueComputer
from watermark.synthid.scoring import weighted_mean_score


def _make_computer() -> SynthIDGValueComputer:
    return SynthIDGValueComputer(
        ngram_len=config.DEFAULT_NGRAM_LEN,
        keys=config.DEFAULT_KEYS,
        context_history_size=config.DEFAULT_CONTEXT_HISTORY_SIZE,
        device=torch.device("cpu"),
    )


def test_g_values_are_binary_and_deterministic():
    computer = _make_computer()
    input_ids = torch.randint(0, 50000, (1, 30), dtype=torch.long)

    g_values_1 = computer.compute_g_values(input_ids)
    g_values_2 = computer.compute_g_values(input_ids)

    assert torch.equal(g_values_1, g_values_2)
    assert set(g_values_1.unique().tolist()).issubset({0, 1})
    assert g_values_1.shape == (1, 30 - (config.DEFAULT_NGRAM_LEN - 1), len(config.DEFAULT_KEYS))


def test_context_repetition_mask_flags_repeated_context():
    computer = _make_computer()
    # Repeat the same 4-token context (ngram_len - 1) twice in a row.
    context = [1, 2, 3, 4]
    input_ids = torch.tensor([context + [5] + context + [6]], dtype=torch.long)

    mask = computer.compute_context_repetition_mask(input_ids)
    # Second occurrence of the identical context should be masked out (0).
    assert mask[0, -1].item() == 0


def test_weighted_mean_score_of_all_ones_is_one():
    g_values = torch.ones((1, 10, 3))
    mask = torch.ones((1, 10))
    assert weighted_mean_score(g_values, mask) == 1.0


def test_weighted_mean_score_of_all_zeros_is_zero():
    g_values = torch.zeros((1, 10, 3))
    mask = torch.ones((1, 10))
    assert weighted_mean_score(g_values, mask) == 0.0
