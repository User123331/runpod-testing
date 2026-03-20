# Speed Optimizations: Triton Kernels & Libraries

**Goal**: More training steps in the same wallclock = better bpb

---

## Priority 1: FlashAttention 3 (~5% step time reduction)

**What**: H100-optimized attention using Hopper async TMA + warp specialization
**Speedup**: 1.5-2x over FA2 in attention forward, ~5% overall step time
**Integration**: Drop-in replacement
```python
from flash_attn_interface import flash_attn_func as flash_attn_3_func
```
**Status**: Proven — PRs #198 and #164 use this. Only external library in top submissions.
**Install**: `pip install flash-attn --no-build-isolation` (from hopper branch)

---

## Priority 2: Fused Linear+ReLU^2 Triton Kernel (~5-15% MLP speedup)

**What**: Fuses CastedLinear + relu().square() into one Triton kernel
**Source**: modded-nanogpt `triton_kernels.FusedLinearReLUSquareFunction`
**Why it helps**: Eliminates intermediate tensor materialization in MLP (which is 3x expanded)
**Integration**: Copy Triton kernel, replace MLP forward pass
**Status**: Used in modded-nanogpt speedrun, not yet in any Parameter Golf PR

---

## Priority 3: Fused Softcapped Cross-Entropy (~2-5% loss speedup)

**What**: Fuses logit_softcap + cross_entropy into one Triton kernel
**Source**: modded-nanogpt `triton_kernels.FusedSoftcappedCrossEntropy`
**Why it helps**: Avoids materializing softcapped logits tensor
**Integration**: Copy Triton kernel, replace loss computation
**Note**: Only applies to non-MoS path (MoS uses nll_loss on log-probs)
**Status**: Used in modded-nanogpt speedrun, not yet in any Parameter Golf PR

---

## Priority 4: torch.compile Tuning (0-5% overall)

```python
# Current
torch.compile(model, dynamic=False, fullgraph=True)

# Try
torch.compile(model, dynamic=False, fullgraph=True, mode="max-autotune")
```

Also set env var:
```bash
export PYTORCH_ALLOC_CONF="expandable_segments:True"
```

---

## Priority 5: Gradient Checkpointing (enables larger batch/seq)

**What**: Recompute activations in backward pass instead of storing them
**Benefit**: 50-70% activation memory reduction, enables seq=2048 or larger batch on 1xH100
**Cost**: ~20-33% more compute (5-10% wall-clock in practice)
**When to use**: If moving to seq=2048+ on 1xH100

---

## Priority 6: Custom Triton MoS Kernel (if MoS proves useful)

**What**: Fuse log_softmax over K components + logsumexp mixture into one kernel
**Expected**: Reduce MoS overhead from ~5ms to ~2-3ms per step
**Effort**: ~50-100 lines of Triton, based on fused softmax tutorial
**Note**: The bigger bottleneck is the K einsum matmuls, not the softmax

---

## NOT Worth It at Our Scale

| Technique | Why Skip |
|-----------|----------|
| FP8 training (torchao) | dim=512 matrices too small, overhead > benefit |
| Fused RMSNorm | torch.compile already fuses it |
| Apex FusedAdam | Already using fused=True, marginal gain |
| Liger FusedCE | Logit tensor tiny at vocab=1024 |
| bitsandbytes 8-bit optimizer | Model too small to benefit |

---

## Impact Estimate

| Optimization | Step Time Reduction | Extra Steps in 10min | bpb Impact |
|-------------|--------------------|--------------------|------------|
| FA3 | ~5% | +1000 steps | ~0.005 bpb |
| Fused MLP | ~10% | +2000 steps | ~0.008 bpb |
| Fused CE | ~3% | +600 steps | ~0.002 bpb |
| max-autotune | ~2% | +400 steps | ~0.001 bpb |
| **Combined** | **~20%** | **+4000 steps** | **~0.015 bpb** |

At current ~500ms/step, 20% reduction = 400ms/step = ~1500 steps in 10min → ~1875 steps.
On 8xH100 at ~27ms/step, 20% = ~22ms/step = ~27,300 steps vs ~22,200.
