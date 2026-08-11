"""Perplexity/entropy math, exercised on tiny synthetic logits — no model
weights needed."""

import pytest
import torch

from statistical.binoculars.metrics import entropy, perplexity


class _FakeEncoding:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


def test_perplexity_is_zero_when_logits_perfectly_predict_next_token():
    vocab_size = 5
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    encoding = _FakeEncoding(input_ids, attention_mask)

    # Logits with a huge spike on the correct next token at every position.
    logits = torch.full((1, 4, vocab_size), -1e4)
    for t, next_token in enumerate(input_ids[0, 1:].tolist()):
        logits[0, t, next_token] = 1e4

    ppl = perplexity(encoding, logits)
    assert ppl[0] == pytest.approx(0.0, abs=1e-3)


def test_entropy_is_zero_when_p_and_q_agree_perfectly():
    vocab_size = 5
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    encoding = _FakeEncoding(input_ids, attention_mask)
    pad_token_id = -1  # none of the tokens are padding

    logits = torch.randn(1, 4, vocab_size)
    ce = entropy(logits, logits, encoding, pad_token_id)
    assert ce[0] >= 0.0
