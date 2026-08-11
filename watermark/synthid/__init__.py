from .detector import SynthIDDetector
from watermark.registry import WatermarkDetectorSpec, register

register(
    WatermarkDetectorSpec(
        id="synthid",
        display_name="SynthID (Weighted Mean, GPT-2 keys)",
        factory=SynthIDDetector,
    )
)

__all__ = ["SynthIDDetector"]
