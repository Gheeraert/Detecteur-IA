# llm-text-forensics

Unified research tool for detecting LLM-generated / watermarked text. Merges two
detection strategies behind one interface:

- **`watermark/`** — cryptographic watermark detectors. Currently `SynthIDDetector`
  (Weighted Mean scoring, ported from
  [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text)).
  Only recognizes the SynthID watermark applied with the reference implementation's
  default keys — extend this package (e.g. `watermark/anthropic/`, `watermark/kgw/`)
  to add other schemes without touching the rest of the tool.
- **`statistical/`** — heuristic zero-shot detectors. Currently `BinocularsDetector`,
  ported from [ahans30/Binoculars](https://github.com/ahans30/Binoculars).

Every detector implements the same interface:

```python
def analyze(self, text: str) -> DetectionResult:
    ...  # label, score, confidence, method_name, details
```

## Aggregation

`aggregator.ForensicsAggregator` combines both families with a **priority rule**,
not a weighted average: a positive watermark verdict is treated as strong,
near-direct evidence and wins outright. Absent a watermark hit, it falls back
to the statistical (Binoculars) verdict, with confidence discounted to reflect
that it's a heuristic rather than a positive signal. Both raw
`DetectionResult`s are always kept on the `AggregatedResult` for traceability.

```python
from aggregator import ForensicsAggregator
from watermark.synthid import SynthIDDetector
from statistical import BinocularsDetector

aggregator = ForensicsAggregator(
    watermark_detectors=[SynthIDDetector()],
    statistical_detector=BinocularsDetector(),
)
result = aggregator.analyze(some_text)
```

## Binoculars threshold calibration

Binoculars' decision threshold (`statistical/binoculars/config.py`) is specific
to the observer/performer model pair it was calibrated on — the shipped value
is only valid for Falcon-7B / Falcon-7B-Instruct. If you switch model pairs
(e.g. for better results on non-English text), recalibrate first:

```
python scripts/calibrate_binoculars_threshold.py \
    --observer <model> --performer <model> \
    --human-texts-dir <dir> --ai-texts-dir <dir>
```

Using an uncalibrated pair raises `ValueError` rather than silently reusing the
Falcon threshold.

## GUI

A Gradio interface (`demo/app.py`) provides: text paste or `.txt` file upload,
a checkbox per available detector (watermark detectors are discovered from
`watermark/registry.py`; Binoculars is a separate toggle), a color-coded risk
gauge (`demo/gauge.py`, plain inline SVG, no plotting dependency), and a
results panel that always shows the aggregated verdict plus the raw
score/confidence/details from every method that ran.

```
pip install -e .[demo]
python -m demo.app
```

To add a new watermarking system to the GUI: implement a `WatermarkDetector`
under `watermark/<name>/`, call `register()` on a `WatermarkDetectorSpec` in
that subpackage's `__init__.py`, then add `import watermark.<name>` to
`demo/app.py`. It appears as a new checkbox automatically — no other code
changes needed.

## Install

```
pip install -e .[dev]
```

Add `.[calibration]` for the calibration script's scikit-learn dependency, or
`.[demo]` for the Gradio UI dependency set.

## Tests

```
pytest
```

All tests run against synthetic tensors / fake detectors — no model downloads
required, so each detector and the aggregator are testable independently.
