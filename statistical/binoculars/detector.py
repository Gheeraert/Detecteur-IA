"""Binoculars statistical detector.

Ported from ahans30/Binoculars (binoculars/detector.py), adapted to the
common `analyze() -> DetectionResult` interface. Core scoring logic
(perplexity / cross-perplexity ratio) is unchanged.
"""

import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.types import DetectionResult
from statistical.binoculars import config
from statistical.binoculars.metrics import entropy, perplexity
from statistical.binoculars.utils import assert_tokenizer_consistency

torch.set_grad_enabled(False)

huggingface_config = {
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class BinocularsDetector:
    def __init__(
        self,
        observer_name_or_path: str = "tiiuae/falcon-7b",
        performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
        use_bfloat16: bool = True,
        max_token_observed: int = 512,
        mode: str = "low-fpr",
    ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.observer_name_or_path = observer_name_or_path
        self.performer_name_or_path = performer_name_or_path
        self.mode = mode
        # Threshold is resolved lazily (see `threshold` property) so this
        # class stays usable purely for `compute_score` — e.g. from
        # scripts/calibrate_binoculars_threshold.py — against model pairs
        # that don't have a calibrated threshold yet.
        self._threshold: float | None = None

        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        )
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"],
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            thresholds = config.get_thresholds(self.observer_name_or_path, self.performer_name_or_path)
            if self.mode == "low-fpr":
                self._threshold = thresholds.low_fpr
            elif self.mode == "accuracy":
                self._threshold = thresholds.accuracy
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        return self._threshold

    def change_mode(self, mode: str) -> None:
        self.mode = mode
        self._threshold = None

    def _tokenize(self, text: str) -> transformers.BatchEncoding:
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False,
        ).to(self.observer_model.device)

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, text: str) -> float:
        encodings = self._tokenize(text)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(
            observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
            encodings.to(DEVICE_1), self.tokenizer.pad_token_id,
        )
        return float((ppl / x_ppl)[0])

    def analyze(self, text: str) -> DetectionResult:
        score = self.compute_score(text)
        is_ai_generated = score < self.threshold
        # Distance from the decision threshold, scaled to [0, 1] — a crude
        # but explicit stand-in for a real calibrated confidence curve.
        confidence = min(1.0, abs(score - self.threshold) / self.threshold)

        return DetectionResult(
            label="ai-generated" if is_ai_generated else "human",
            score=score,
            confidence=confidence,
            method_name="binoculars",
            details={
                "threshold": self.threshold,
                "mode": self.mode,
                "observer_model": self.observer_name_or_path,
                "performer_model": self.performer_name_or_path,
            },
        )
