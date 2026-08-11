"""Aggregator priority logic, exercised with fake detectors so it needs no
model downloads and stays independent of the real SynthID/Binoculars code."""

from common.types import DetectionResult
from aggregator.aggregator import ForensicsAggregator
from watermark.base import WatermarkDetector


class FakeWatermarkDetector(WatermarkDetector):
    def __init__(self, label: str, confidence: float = 0.9):
        self._label = label
        self._confidence = confidence

    def analyze(self, text: str) -> DetectionResult:
        return DetectionResult(
            label=self._label,
            score=1.0 if self._label == "watermarked" else 0.5,
            confidence=self._confidence,
            method_name="fake-watermark",
        )


class FakeStatisticalDetector:
    def __init__(self, label: str, confidence: float = 0.8):
        self._label = label
        self._confidence = confidence

    def analyze(self, text: str) -> DetectionResult:
        return DetectionResult(
            label=self._label,
            score=0.5,
            confidence=self._confidence,
            method_name="fake-statistical",
        )


def test_watermark_positive_wins_regardless_of_statistical_verdict():
    aggregator = ForensicsAggregator(
        watermark_detectors=[FakeWatermarkDetector("watermarked", confidence=0.95)],
        statistical_detector=FakeStatisticalDetector("human", confidence=0.99),
    )
    result = aggregator.analyze("some text")
    assert result.label == "watermarked"
    assert result.confidence == 0.95
    assert result.statistical_result is not None  # kept for traceability
    assert result.statistical_result.label == "human"


def test_falls_back_to_statistical_with_discounted_confidence_when_no_watermark():
    aggregator = ForensicsAggregator(
        watermark_detectors=[FakeWatermarkDetector("unknown", confidence=0.1)],
        statistical_detector=FakeStatisticalDetector("ai-generated", confidence=0.8),
        statistical_fallback_confidence_discount=0.7,
    )
    result = aggregator.analyze("some text")
    assert result.label == "ai-generated"
    assert result.confidence == 0.8 * 0.7


def test_multiple_watermark_detectors_first_positive_wins():
    aggregator = ForensicsAggregator(
        watermark_detectors=[
            FakeWatermarkDetector("unknown"),
            FakeWatermarkDetector("watermarked", confidence=0.6),
        ],
        statistical_detector=FakeStatisticalDetector("human"),
    )
    result = aggregator.analyze("some text")
    assert result.label == "watermarked"
    assert len(result.watermark_results) == 2
