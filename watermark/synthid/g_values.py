"""G-value computation for SynthID watermark detection.

Trimmed from google-deepmind/synthid-text's `SynthIDLogitsProcessor`
(src/synthid_text/logits_processing.py): this tool only ever *detects* text,
it never generates watermarked text, so everything related to warping
generation-time logits (temperature/top_k scaling, tournament score updates,
the stateful `watermarked_call` path) is dropped. What's left is the g-value
and mask computation the Weighted Mean detector needs.
"""

import hashlib
from collections.abc import Sequence

import torch

from watermark.synthid import hashing_function


class SynthIDState:
    """Tracks previously seen ngram contexts, to mask out repeated ones."""

    def __init__(self, batch_size: int, ngram_len: int, context_history_size: int, device: torch.device):
        self.context_history = torch.zeros(
            (batch_size, context_history_size), dtype=torch.int64, device=device
        )


class SynthIDGValueComputer:
    """Computes g-values and validity masks for a sequence of token ids."""

    def __init__(
        self,
        *,
        ngram_len: int,
        keys: Sequence[int],
        context_history_size: int,
        device: torch.device,
    ):
        self.ngram_len = ngram_len
        self.keys = torch.tensor(keys, device=device)
        self.context_history_size = context_history_size
        self.device = device

        self.hash_iv = hashlib.sha256(self.keys.to(torch.long).numpy().tobytes()).digest()
        torch_long_max = torch.iinfo(torch.int64).max
        self.hash_iv = int.from_bytes(self.hash_iv, byteorder="big") % torch_long_max

    def _check_input_ids_shape(self, input_ids: torch.LongTensor) -> None:
        if len(input_ids.shape) != 2:
            raise ValueError(
                f"Input ids should be of shape (batch_size, input_len), but is {input_ids.shape}"
            )

    def _get_gvals(self, ngram_keys: torch.LongTensor, num_apply_hash: int = 12) -> torch.LongTensor:
        shift = 64 // num_apply_hash
        for _ in range(num_apply_hash):
            ngram_keys = hashing_function.accumulate_hash(ngram_keys, torch.LongTensor([1])) >> shift
        return (ngram_keys >> 30) % 2

    def _compute_ngram_keys(self, ngrams: torch.LongTensor) -> torch.LongTensor:
        if ngrams.shape[2] != self.ngram_len:
            raise ValueError(
                "Ngrams should be of shape (batch_size, num_ngrams, ngram_len), where"
                f" ngram_len is {self.ngram_len}, but is {ngrams.shape}"
            )
        batch_size, _, _ = ngrams.shape
        hash_result = torch.full((batch_size,), self.hash_iv, dtype=torch.long, device=self.device)
        hash_result = torch.vmap(
            hashing_function.accumulate_hash, in_dims=(None, 1), out_dims=1
        )(hash_result, ngrams)

        keys = self.keys[None, None, :, None]
        hash_result = torch.vmap(
            hashing_function.accumulate_hash, in_dims=(None, 2), out_dims=2
        )(hash_result, keys)
        return hash_result

    def compute_g_values(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """G-values (batch_size, input_len - (ngram_len - 1), depth)."""
        self._check_input_ids_shape(input_ids)
        ngrams = input_ids.unfold(dimension=1, size=self.ngram_len, step=1)
        ngram_keys = self._compute_ngram_keys(ngrams)
        return self._get_gvals(ngram_keys)

    def compute_context_repetition_mask(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """1 for ngrams whose (ngram_len - 1)-token context is not a repeat, 0 otherwise."""
        self._check_input_ids_shape(input_ids)
        batch_size, _ = input_ids.shape
        state = SynthIDState(
            batch_size=batch_size,
            ngram_len=self.ngram_len,
            context_history_size=self.context_history_size,
            device=self.device,
        )
        contexts = input_ids[:, :-1].unfold(dimension=1, size=self.ngram_len - 1, step=1)
        _, num_contexts, _ = contexts.shape

        are_repeated_contexts = []
        for i in range(num_contexts):
            context = contexts[:, i, :]
            hash_result = torch.full((batch_size,), self.hash_iv, dtype=torch.long, device=self.device)
            context_hash = hashing_function.accumulate_hash(hash_result, context)[:, None]
            is_repeated_context = (state.context_history == context_hash).any(dim=1, keepdim=True)
            are_repeated_contexts.append(is_repeated_context)
            state.context_history = torch.concat(
                (context_hash, state.context_history), dim=1
            )[:, :-1]
        are_repeated_contexts = torch.concat(are_repeated_contexts, dim=1)
        return torch.logical_not(are_repeated_contexts)

    def compute_eos_token_mask(self, input_ids: torch.LongTensor, eos_token_id: int) -> torch.LongTensor:
        """1 for positions before the first EOS token, 0 from EOS onward."""
        self._check_input_ids_shape(input_ids)
        noneos_masks = []
        all_eos_equated = input_ids == eos_token_id
        for eos_equated in all_eos_equated:
            nonzero_idx = torch.nonzero(eos_equated)
            noneos_mask = torch.ones_like(eos_equated)
            if nonzero_idx.shape[0] != 0:
                noneos_mask[nonzero_idx[0][0]:] = 0
            noneos_masks.append(noneos_mask)
        return torch.stack(noneos_masks, dim=0)
