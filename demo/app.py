"""Gradio interface for llm-text-forensics.

Run with:
    pip install -e .[demo]
    python -m demo.app
"""

from pathlib import Path

import gradio as gr

from aggregator.aggregator import AggregatedResult, ForensicsAggregator
from common.types import DetectionResult
from demo.gauge import build_gauge_svg
from statistical.binoculars.detector import BinocularsDetector
from watermark.registry import all_specs, get_spec

# Registers each watermark detector as a side effect of import. To add a new
# watermarking system: implement it under watermark/<name>/, register() it in
# that subpackage's __init__.py, then add the import here.
import watermark.synthid  # noqa: F401,E402

# Detector instances are expensive to build (they load tokenizers/models), so
# they're built once on first use and cached for the lifetime of the process.
_watermark_detector_cache: dict[str, object] = {}
_binoculars_cache: BinocularsDetector | None = None


def _get_watermark_detector(detector_id: str):
    if detector_id not in _watermark_detector_cache:
        _watermark_detector_cache[detector_id] = get_spec(detector_id).factory()
    return _watermark_detector_cache[detector_id]


def _get_binoculars() -> BinocularsDetector:
    global _binoculars_cache
    if _binoculars_cache is None:
        _binoculars_cache = BinocularsDetector()
    return _binoculars_cache


def _resolve_input_text(pasted_text: str, uploaded_file: str | None) -> str:
    """A freshly uploaded file takes priority over stale pasted text."""
    if uploaded_file:
        return Path(uploaded_file).read_text(encoding="utf-8", errors="replace")
    return pasted_text or ""


def _format_detail(result: DetectionResult) -> str:
    detail_items = ", ".join(f"{k}={v}" for k, v in result.details.items())
    return (
        f"- **{result.method_name}** — label: `{result.label}`, "
        f"score: `{result.score:.4f}`, confidence: `{result.confidence:.2f}`"
        + (f"  \n  _{detail_items}_" if detail_items else "")
    )


def _format_verdict(result: AggregatedResult) -> str:
    label_fr = {
        "watermarked": "WATERMARKÉ",
        "ai-generated": "PROBABLEMENT GÉNÉRÉ PAR IA",
        "human": "PROBABLEMENT HUMAIN",
        "unknown": "INDÉTERMINÉ",
    }[result.label]
    return (
        f"## Verdict : {label_fr}\n"
        f"**Confiance : {result.confidence:.0%}**\n\n"
        f"{result.reasoning}"
    )


def _format_details(result: AggregatedResult) -> str:
    lines = ["### Détail par méthode"]
    if not result.watermark_results and result.statistical_result is None:
        lines.append("_Aucun détecteur activé._")
        return "\n".join(lines)

    if result.watermark_results:
        lines.append("**Watermark**")
        lines.extend(_format_detail(r) for r in result.watermark_results)
    if result.statistical_result is not None:
        lines.append("**Statistique**")
        lines.append(_format_detail(result.statistical_result))
    return "\n".join(lines)


def analyze(
    pasted_text: str,
    uploaded_file: str | None,
    enabled_watermark_ids: list[str],
    use_binoculars: bool,
):
    text = _resolve_input_text(pasted_text, uploaded_file)
    if not text.strip():
        return "Veuillez coller du texte ou importer un fichier.", "", _empty_gauge_svg()

    if not enabled_watermark_ids and not use_binoculars:
        return "Activez au moins un détecteur.", "", _empty_gauge_svg()

    try:
        watermark_detectors = [_get_watermark_detector(i) for i in enabled_watermark_ids]
        statistical_detector = _get_binoculars() if use_binoculars else None
        aggregator = ForensicsAggregator(
            watermark_detectors=watermark_detectors,
            statistical_detector=statistical_detector,
        )
        result = aggregator.analyze(text)
    except Exception as e:  # noqa: BLE001 — surfaced to the user, not swallowed
        return f"**Erreur d'analyse :** {e}", "", _empty_gauge_svg()

    return _format_verdict(result), _format_details(result), build_gauge_svg(result)


def _empty_gauge_svg() -> str:
    return ""


def build_app() -> gr.Blocks:
    watermark_choices = [(spec.display_name, spec.id) for spec in all_specs()]

    with gr.Blocks(title="LLM Text Forensics") as app:
        gr.Markdown("# LLM Text Forensics\nOutil de recherche : détection de watermark et détection statistique, avec traçabilité complète par méthode.")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Texte à analyser (coller ici)",
                    lines=14,
                    placeholder="Collez le texte à analyser…",
                )
                file_input = gr.File(label="…ou importer un fichier .txt", file_types=[".txt"], type="filepath")

            with gr.Column(scale=1):
                gr.Markdown("### Détecteurs")
                watermark_checkboxes = gr.CheckboxGroup(
                    choices=watermark_choices,
                    value=[spec.id for spec in all_specs()],
                    label="Watermark",
                )
                binoculars_checkbox = gr.Checkbox(value=True, label="Binoculars (statistique)")
                analyze_button = gr.Button("Analyser", variant="primary")

        gr.Markdown("### Jauge de risque")
        gauge_output = gr.HTML(_empty_gauge_svg())

        verdict_output = gr.Markdown()
        details_output = gr.Markdown()

        analyze_button.click(
            fn=analyze,
            inputs=[text_input, file_input, watermark_checkboxes, binoculars_checkbox],
            outputs=[verdict_output, details_output, gauge_output],
        )

    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
