"""Magnitude (unstructured) pruning.

Zero out the smallest-magnitude weights globally or per-layer. Unstructured
sparsity yields no latency win on dense GPU kernels but is a strong
*compression* tool (good for storage and as a regularizer during fine-tuning).
"""

from __future__ import annotations

import torch
import torch.nn.utils.prune as prune_utils
from torch import nn


def prune_magnitude(
    model: nn.Module,
    *,
    sparsity: float = 0.5,
    scope: str = "global",
    target_layers: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    make_permanent: bool = True,
    **_: object,
) -> nn.Module:
    """Zero out the bottom ``sparsity`` fraction of weights by absolute value.

    Parameters
    ----------
    sparsity:
        Fraction of weights to zero, in ``[0, 1)``.
    scope:
        ``"global"`` ranks all eligible weights together (recommended).
        ``"layerwise"`` applies the ratio to each layer independently.
    make_permanent:
        If ``True``, removes the pruning reparametrization and bakes zeros into
        the weight tensor. Set to ``False`` if you want to fine-tune with masks.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")

    params_to_prune = [
        (m, "weight") for m in model.modules() if isinstance(m, target_layers)
    ]
    if not params_to_prune:
        return model

    if scope == "global":
        prune_utils.global_unstructured(
            params_to_prune,
            pruning_method=prune_utils.L1Unstructured,
            amount=sparsity,
        )
    elif scope == "layerwise":
        for module, name in params_to_prune:
            prune_utils.l1_unstructured(module, name=name, amount=sparsity)
    else:
        raise ValueError(f"Unknown scope '{scope}'")

    if make_permanent:
        for module, name in params_to_prune:
            try:
                prune_utils.remove(module, name)
            except ValueError:
                pass  # nothing to remove

    return model


@torch.no_grad()
def measure_layer_sparsity(model: nn.Module) -> dict[str, float]:
    """Return per-layer fraction of zero weights for inspection."""
    out: dict[str, float] = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            w = module.weight
            if w.is_floating_point():
                out[name] = ((w == 0).sum() / w.numel()).item()
    return out
