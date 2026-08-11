"""Priority-based aggregation of watermark and statistical detectors.

Not a weighted average: a positive watermark verdict is strong, near-direct
evidence (a cryptographic signal was found or it wasn't), while a statistical
detector like Binoculars is a heuristic that can be fooled by paraphrasing,
translation, or unusual writing styles. Averaging the two would let a
confident-but-wrong statistical score erode a genuine watermark hit, or vice
versa. So instead: watermark positive wins outright; absent that, fall back to
the statistical verdict but discount its confidence to reflect that it's a
weaker signal.
"""

from dataclasses import dataclass, field

from common.types import DetectionResult, Detector, Label
from watermark.base import WatermarkDetector


@dataclass(frozen=True)
class AggregatedResult:
    label: Label
    confidence: float
    method_name: str = "aggregator"
    watermark_results: list[DetectionResult] = field(default_factory=list)
    statistical_result: DetectionResult | None = None
    reasoning: str = ""


class ForensicsAggregator:
    def __init__(
        self,
        watermark_detectors: list[WatermarkDetector],
        statistical_detector: Detector | None,
        statistical_fallback_confidence_discount: float = 0.7,
    ) -> None:
        """`statistical_detector` may be None if the caller disabled it (e.g. a
        GUI checkbox) — the aggregator then relies on watermark detectors alone."""
        self.watermark_detectors = watermark_detectors
        self.statistical_detector = statistical_detector
        self.statistical_fallback_confidence_discount = statistical_fallback_confidence_discount

    def analyze(self, text: str) -> AggregatedResult:
        watermark_results = [detector.analyze(text) for detector in self.watermark_detectors]
        statistical_result = (
            self.statistical_detector.analyze(text) if self.statistical_detector is not None else None
        )

        positive_watermark = next(
            (result for result in watermark_results if result.label == "watermarked"),
            None,
        )
        if positive_watermark is not None:
            return AggregatedResult(
                label="watermarked",
                confidence=positive_watermark.confidence,
                watermark_results=watermark_results,
                statistical_result=statistical_result,
                reasoning=(
                    f"{positive_watermark.method_name} detected a watermark — "
                    "this takes priority over the statistical score."
                ),
            )

        if statistical_result is None:
            return AggregatedResult(
                label="unknown",
                confidence=0.0,
                watermark_results=watermark_results,
                statistical_result=None,
                reasoning="No watermark detected, and no statistical detector was enabled to fall back on.",
            )

        return AggregatedResult(
            label=statistical_result.label,
            confidence=statistical_result.confidence * self.statistical_fallback_confidence_discount,
            watermark_results=watermark_results,
            statistical_result=statistical_result,
            reasoning=(
                "No watermark detected — falling back to the statistical "
                f"verdict from {statistical_result.method_name}, with reduced "
                "confidence since it is a heuristic, not a positive signal."
            ),
        )
