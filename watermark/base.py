"""Base class for watermark detectors.

A watermark detector only ever asserts "watermarked" or "unknown" — it has no
basis for a "human"/"ai-generated" verdict, that's the statistical detectors'
job. Concrete detectors (SynthID today; an Anthropic or KGW/MarkLLM detector
tomorrow) live in their own subpackage and only need to implement `analyze`.
"""

from abc import ABC, abstractmethod

from common.types import DetectionResult


class WatermarkDetector(ABC):
    @abstractmethod
    def analyze(self, text: str) -> DetectionResult: ...
