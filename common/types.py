"""Common interface every detector in this tool implements."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Label = Literal["watermarked", "ai-generated", "human", "unknown"]


@dataclass(frozen=True)
class DetectionResult:
    """Result of a single detector's analysis of a piece of text.

    `score` is method-specific and not comparable across detectors (e.g. a
    Binoculars ratio and a SynthID weighted-mean live on different scales).
    `details` carries whatever the method needs for traceability (thresholds
    used, token counts, etc.) so the raw evidence survives past the verdict.
    """

    label: Label
    score: float
    confidence: float
    method_name: str
    details: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    def analyze(self, text: str) -> DetectionResult: ...
