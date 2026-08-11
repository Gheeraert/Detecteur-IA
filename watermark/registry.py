"""Registry of available watermark detectors.

Adding a new watermarking system (e.g. an Anthropic watermark, or a generic
KGW/MarkLLM detector) means: create `watermark/<name>/`, implement
`WatermarkDetector`, and call `register()` with a spec in that subpackage's
`__init__.py`. Nothing else in this file, the aggregator, or the GUI needs to
change — the GUI discovers detectors by importing the subpackages listed in
`demo/app.py`'s `KNOWN_WATERMARK_MODULES` and reading `all_specs()`.
"""

from dataclasses import dataclass
from typing import Callable

from watermark.base import WatermarkDetector


@dataclass(frozen=True)
class WatermarkDetectorSpec:
    id: str
    display_name: str
    factory: Callable[[], WatermarkDetector]


_registry: dict[str, WatermarkDetectorSpec] = {}


def register(spec: WatermarkDetectorSpec) -> None:
    _registry[spec.id] = spec


def all_specs() -> list[WatermarkDetectorSpec]:
    return list(_registry.values())


def get_spec(detector_id: str) -> WatermarkDetectorSpec:
    return _registry[detector_id]
