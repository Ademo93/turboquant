# Pruning

```python
from turboquant import prune
pruned = prune(model, strategy="l1-channel", sparsity=0.3)
```

## Unstructured vs structured

**Unstructured** sparsity (`magnitude`) zeros individual weights anywhere in
the tensor. On dense GPU kernels this gives no speedup — it pays off only with
sparse runtimes or as a regularizer during fine-tuning.

**Structured** sparsity (`l1-channel`, `l2-channel`, `random-channel`) removes
whole output channels / filters. The weight tensor genuinely shrinks, FLOPs
drop, and ONNX shape inference picks it up.

**N:M sparsity** (`nm-sparsity`) keeps N out of every M consecutive weights.
The 2:4 pattern is accelerated by NVIDIA Ampere/Hopper sparse Tensor Cores.

## Strategies

### `magnitude`

```python
prune(model, strategy="magnitude", sparsity=0.5, scope="global")
```

`scope="global"` ranks all eligible weights together (recommended).
`scope="layerwise"` applies the ratio independently per layer.

### `l1-channel` / `l2-channel`

Score each output channel by the Lₙ norm of its weights, drop the lowest.

```python
prune(model, strategy="l1-channel", sparsity=0.3)
```

### `nm-sparsity`

```python
prune(model, strategy="nm-sparsity", n=2, m=4)
```

## Typical recipe

1. **Train** the dense model.
2. **Prune** to target sparsity.
3. **Fine-tune** (1–10% of original schedule) to recover accuracy.
4. **Quantize** the pruned model — pruning + INT8 stacks well.

The "prune → finetune → quantize" sequence usually outperforms doing
quantization first.
