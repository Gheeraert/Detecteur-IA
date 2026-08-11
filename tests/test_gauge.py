from aggregator.aggregator import AggregatedResult
from demo.gauge import _risk_score, build_gauge_svg


def test_risk_score_human_is_below_half():
    result = AggregatedResult(label="human", confidence=0.8)
    assert _risk_score(result) < 0.5


def test_risk_score_ai_generated_is_above_half():
    result = AggregatedResult(label="ai-generated", confidence=0.8)
    assert _risk_score(result) > 0.5


def test_risk_score_watermarked_is_above_half():
    result = AggregatedResult(label="watermarked", confidence=0.9)
    assert _risk_score(result) > 0.5


def test_risk_score_unknown_is_exactly_half():
    result = AggregatedResult(label="unknown", confidence=0.0)
    assert _risk_score(result) == 0.5


def test_build_gauge_svg_contains_marker_and_percentage():
    result = AggregatedResult(label="watermarked", confidence=1.0)
    svg = build_gauge_svg(result)
    assert "<svg" in svg
    assert "100%" in svg
