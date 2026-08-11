"""Weighted Mean scoring, reimplemented on torch (as in synthid-text's
tools/detector_gui.py) so the watermark module only depends on
torch/transformers — no jax/flax/optax for this scoring method.
"""

import torch


def weighted_mean_score(g_values: torch.Tensor, mask: torch.Tensor) -> float:
    """Weighted mean of g-values, weighted 10..1 across watermarking depth.

    Args:
        g_values: g-values of shape [1, seq_len, depth].
        mask: binary validity mask of shape [1, seq_len].

    Returns:
        Score in [0, 1]. ~0.5 for unwatermarked text, pulled higher by a
        SynthID watermark applied with the matching keys.
    """
    depth = g_values.shape[-1]
    weights = torch.linspace(10, 1, depth, dtype=g_values.dtype)
    weights *= depth / weights.sum()

    weighted = g_values * weights.view(1, 1, depth)
    num_unmasked = mask.sum(dim=1)
    score = (weighted * mask.unsqueeze(2)).sum(dim=(1, 2)) / (depth * num_unmasked)
    return float(score.item())
