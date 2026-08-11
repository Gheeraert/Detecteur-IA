"""Renders an aggregated result as a red-yellow-green risk gauge (inline SVG).

No plotting dependency (matplotlib/plotly would be superfluous for a single
bar) — this is ~40 lines of SVG built from f-strings.
"""

import colorsys

from aggregator.aggregator import AggregatedResult

_WIDTH = 640
_HEIGHT = 90
_MARGIN = 16
_BAR_Y0, _BAR_Y1 = 30, 55


def _risk_score(result: AggregatedResult) -> float:
    """Maps an AggregatedResult onto [0, 1]: 0 = confidently human/clean,
    1 = confidently AI-generated/watermarked, 0.5 = no signal either way."""
    if result.label in ("ai-generated", "watermarked"):
        return 0.5 + result.confidence / 2
    if result.label == "human":
        return 0.5 - result.confidence / 2
    return 0.5


def _color_for(value: float) -> str:
    """0 -> green, 0.5 -> yellow, 1 -> red."""
    value = max(0.0, min(1.0, value))
    hue = (1.0 - value) * 120.0 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.85)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def build_gauge_svg(result: AggregatedResult) -> str:
    risk = _risk_score(result)
    bar_x0, bar_x1 = _MARGIN, _WIDTH - _MARGIN

    gradient_stops = "".join(
        f'<stop offset="{i / 10:.0%}" stop-color="{_color_for(i / 10)}"/>' for i in range(11)
    )
    marker_x = bar_x0 + (bar_x1 - bar_x0) * risk

    return f"""
<svg width="{_WIDTH}" height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Jauge de risque {risk:.0%}">
  <defs>
    <linearGradient id="riskGradient" x1="0" y1="0" x2="1" y2="0">{gradient_stops}</linearGradient>
  </defs>
  <rect x="{bar_x0}" y="{_BAR_Y0}" width="{bar_x1 - bar_x0}" height="{_BAR_Y1 - _BAR_Y0}"
        fill="url(#riskGradient)" stroke="#888" rx="4"/>
  <text x="{bar_x0}" y="{_BAR_Y1 + 16}" font-size="11" fill="#666">Humain / propre</text>
  <text x="{bar_x1}" y="{_BAR_Y1 + 16}" font-size="11" fill="#666" text-anchor="end">IA / watermark</text>
  <polygon points="{marker_x - 8},{_BAR_Y0 - 4} {marker_x + 8},{_BAR_Y0 - 4} {marker_x},{_BAR_Y0 + 8}" fill="#111"/>
  <text x="{marker_x}" y="{_BAR_Y0 - 8}" font-size="13" font-weight="bold" fill="#111" text-anchor="middle">{risk:.0%}</text>
</svg>
""".strip()
