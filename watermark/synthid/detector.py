"""SynthID watermark detector (Weighted Mean scoring)."""

import torch
import transformers

from common.types import DetectionResult
from watermark.base import WatermarkDetector
from watermark.synthid import config
from watermark.synthid.g_values import SynthIDGValueComputer
from watermark.synthid.scoring import weighted_mean_score


class SynthIDDetector(WatermarkDetector):
    """Detects the SynthID watermark applied with the default reference keys.

    Only recognizes the watermark from google-deepmind/synthid-text's
    reference implementation with `config.DEFAULT_KEYS` — it cannot detect
    watermarks applied with different keys, nor other watermarking schemes.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        threshold: float = config.DEFAULT_THRESHOLD,
        ngram_len: int = config.DEFAULT_NGRAM_LEN,
        keys: list[int] | None = None,
        context_history_size: int = config.DEFAULT_CONTEXT_HISTORY_SIZE,
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.device = device or config.default_device()

        self._tokenizer: transformers.PreTrainedTokenizer | None = None
        self._computer = SynthIDGValueComputer(
            ngram_len=ngram_len,
            keys=keys or list(config.DEFAULT_KEYS),
            context_history_size=context_history_size,
            device=self.device,
        )

    def _tokenizer_or_load(self) -> transformers.PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def analyze(self, text: str) -> DetectionResult:
        tokenizer = self._tokenizer_or_load()
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        ngram_len = self._computer.ngram_len
        if input_ids.shape[1] < ngram_len + 1:
            raise ValueError(
                f"Text too short to analyze: needs at least {ngram_len + 1} tokens, "
                f"got {input_ids.shape[1]}."
            )

        g_values = self._computer.compute_g_values(input_ids)
        context_repetition_mask = self._computer.compute_context_repetition_mask(input_ids)
        combined_mask = context_repetition_mask
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            eos_mask = self._computer.compute_eos_token_mask(input_ids, eos_token_id)[:, ngram_len - 1:]
            combined_mask = combined_mask * eos_mask

        num_scored_tokens = int(combined_mask.sum().item())
        if num_scored_tokens == 0:
            raise ValueError(
                "No usable tokens after filtering (repeated contexts / end of "
                "sequence). Try a longer or more varied text."
            )

        score = weighted_mean_score(g_values.float(), combined_mask.float())
        is_watermarked = score >= self.threshold
        # Distance from the 0.5 unwatermarked baseline, scaled to [0, 1].
        confidence = min(1.0, abs(score - 0.5) * 2)

        return DetectionResult(
            label="watermarked" if is_watermarked else "unknown",
            score=score,
            confidence=confidence,
            method_name="synthid-weighted-mean",
            details={
                "num_scored_tokens": num_scored_tokens,
                "threshold": self.threshold,
                "ngram_len": ngram_len,
                "model_name": self.model_name,
            },
        )
