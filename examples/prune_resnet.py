"""L1 structured channel pruning on ResNet-50, then export to ONNX.

Run:
    python examples/prune_resnet.py --sparsity 0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50

from turboquant import prune
from turboquant.export import export_onnx
from turboquant.pruning import sparsity as measure_sparsity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparsity", type=float, default=0.30)
    parser.add_argument("--strategy", default="l1-channel", choices=["l1-channel", "l2-channel", "magnitude"])
    parser.add_argument("--out", type=Path, default=Path("outputs/resnet50_pruned"))
    args = parser.parse_args()

    print(f"Loading ResNet-50 ImageNet weights ...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

    print(f"Applying {args.strategy} pruning at {args.sparsity:.0%} sparsity ...")
    pruned = prune(model, strategy=args.strategy, sparsity=args.sparsity)
    print(f"  achieved sparsity: {measure_sparsity(pruned):.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(pruned.state_dict(), args.out / "weights.pt")

    sample = torch.randn(1, 3, 224, 224)
    onnx_path = export_onnx(
        pruned,
        sample,
        args.out / "model.onnx",
        opset=17,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"✓ ONNX saved to {onnx_path}")


if __name__ == "__main__":
    main()
