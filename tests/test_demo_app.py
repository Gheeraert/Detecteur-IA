"""Pure-logic pieces of the demo app: input resolution and result formatting.
Does not launch Gradio (no browser/server needed for these)."""

from pathlib import Path

from aggregator.aggregator import AggregatedResult
from common.types import DetectionResult
from demo.app import _format_details, _format_verdict, _resolve_input_text


def test_uploaded_file_takes_priority_over_pasted_text(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("from file", encoding="utf-8")
    assert _resolve_input_text("from textbox", str(file_path)) == "from file"


def test_pasted_text_used_when_no_file():
    assert _resolve_input_text("from textbox", None) == "from textbox"


def test_format_verdict_includes_label_and_confidence():
    result = AggregatedResult(label="watermarked", confidence=0.87, reasoning="because reasons")
    text = _format_verdict(result)
    assert "WATERMARKÉ" in text
    assert "87%" in text
    assert "because reasons" in text


def test_format_details_lists_each_method():
    result = AggregatedResult(
        label="human",
        confidence=0.5,
        watermark_results=[
            DetectionResult(label="unknown", score=0.5, confidence=0.1, method_name="synthid-weighted-mean")
        ],
        statistical_result=DetectionResult(
            label="human", score=1.1, confidence=0.6, method_name="binoculars"
        ),
    )
    text = _format_details(result)
    assert "synthid-weighted-mean" in text
    assert "binoculars" in text


def test_format_details_handles_no_detectors_enabled():
    result = AggregatedResult(label="unknown", confidence=0.0)
    text = _format_details(result)
    assert "Aucun détecteur activé" in text
