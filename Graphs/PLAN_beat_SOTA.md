# Plan: Beat SOTA (1.1428 bpb)

**Date**: 2026-03-21
**Current SOTA**: 1.1428 (thwu1, PR #180)
**Emerging**: 1.1303 (PR #254), 1.1307 (PR #265) — not yet on leaderboard
**Our target**: < 1.13 bpb

---

## Strategy: Combine proven techniques nobody has stacked together yet

The key insight from analyzing all PRs: **no single submission combines ALL the best techniques**. Each top entry uses a subset. We stack them all.

---

## The Stack

### Layer 1: Base Architecture (from thwu1 #180)
- 10-11 layers, dim=512, 8 heads, 4 KV heads (GQA)
- MLP 3x (hidden=1536), ReLU-squared
- U-Net skip connections
- Tied embeddings (FP16 passthrough)
- Logit softcap=30

### Layer 2: Quantization (from thwu1 #180)
- Int5 for MLP weights (saves ~1.86MB for extra layer/features)
- Int6 for attention weights
- zstd-22 compression
- 3% magnitude pruning post-training (better compression)
- WD=0.04 for quantization robustness

### Layer 3: Input Augmentation (from thwu1 #180 + #265)
- BigramHash(10240) buckets, dim=128, projected to 512
- SmearGate (proven compatible, +0.005-0.008)

### Layer 4: Training Optimization (best of all PRs)
- Muon: lr=0.02, WD=0.04, momentum warmup 0.92→0.99 over 1500 steps (from #265)
- SWA: start_frac=0.4, every=50 steps (from thwu1)
- OrthoInit + muP scaling
- Warmdown=3000, warmup=20, grad_clip=0.3
- Seq2048, batch=524K tokens (from #236 — more gradient updates)

### Layer 5: Speed (from #265 + modded-nanogpt)
- FlashAttention 3 (Hopper native) — ~5% faster steps
- Fused Linear+ReLU^2 Triton kernel — ~10% MLP speedup
- torch.compile mode="max-autotune"

### Layer 6: Eval-Time (from #265 + #267)
- Sliding window eval (stride=64)
- Partial XSA on last 3 layers (from #265, +0.002 bpb, only 2ms/step)
- Causal TTT: SGD on val chunks after scoring (from #267, +0.003 bpb)

### Layer 7: Free Training Signal
- MTP auxiliary head (predict t+2, t+3) — discarded at save, zero artifact cost
- From PR #88 — provides gradient enrichment during training

---

## Expected Impact Breakdown

| Technique | bpb gain over baseline | Source |
|-----------|----------------------|--------|
| Int5/6 + MLP3x + 10L | ~0.08 | thwu1 baseline |
| BigramHash(10240) | ~0.01 | thwu1 |
| SmearGate | ~0.006 | PR #162 |
| SWA | ~0.005 | thwu1 |
| OrthoInit + muP | ~0.004 | PR #198 |
| Sliding Window | ~0.03 | All top PRs |
| Seq2048 | ~0.015 | PR #198 |
| Smaller batch (524K) | ~0.003 | PR #236 |
| FA3 + fused kernels (more steps) | ~0.005 | PR #265 |
| Partial XSA (last 3 layers) | ~0.002 | PR #265 |
| Causal TTT | ~0.003 | PR #267 |
| MTP auxiliary | ~0.002 | PR #88 |
| **Total from 1.2244 baseline** | **~0.165** | |
| **Projected bpb** | **~1.06-1.10** | |

Conservative estimate: **1.10-1.12 bpb** (not everything stacks perfectly).

---

## Implementation Phases

### Phase 1: Fork SOTA code (~2 hours)
- Take thwu1's train_gpt.py from PR #180 as base
- Verify it reproduces 1.1428 on 8xH100 (10 min run, ~$3)
- This becomes our baseline to improve upon

### Phase 2: Add proven extras (~3 hours)
- Add SmearGate (if not already in thwu1's code)
- Add Muon momentum warmup (0.92→0.99)
- Switch to batch=524K
- Add FlashAttention 3
- Test on 1xH100 for quick validation

### Phase 3: Add novel techniques (~4 hours)
- Implement Partial XSA on last 3 layers (from PR #265)
- Add MTP auxiliary head (from PR #88)
- Add fused Triton kernels (Linear+ReLU^2, softcapped CE)
- Test on 1xH100

### Phase 4: Eval-time optimization (~2 hours)
- Implement Causal TTT (SGD, 3 epochs per chunk)
- Tune TTT hyperparameters (lr, momentum, epochs)
- Test on 1xH100

### Phase 5: Record attempt (~$20)
- Full run on 8xH100, 10 min
- Submit to record track
- If < 1.13 → PR to openai/parameter-golf

---

## Compute Budget

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Phase 1 | 8xH100 | 15 min | ~$5 |
| Phase 2 | 1xH100 | 30 min | ~$2 |
| Phase 3 | 1xH100 | 1 hour | ~$4 |
| Phase 4 | 1xH100 | 30 min | ~$2 |
| Phase 5 | 8xH100 | 15 min | ~$5 |
| Buffer | — | — | ~$5 |
| **Total** | | | **~$23** |

---

## What Makes This Novel

Nobody has combined ALL of these:
1. Int5/Int6 mixed quant + 10-11L (thwu1)
2. + Partial XSA (PR #265, brand new technique)
3. + MTP auxiliary training (PR #88, free signal)
4. + Causal TTT (PR #267)
5. + FA3 + fused Triton kernels (modded-nanogpt)
6. + Optimized batch size (PR #236)

Each top PR uses 3-4 of these. We use all 6+.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Techniques don't stack as expected | Phase-by-phase testing on 1xH100 |
| XSA + TTT conflict | Test independently first |
| Int5 fragile with new techniques | Fall back to Int6 if quant degrades |
| Compute budget overrun | 1xH100 validation before 8xH100 record |
| FA3 install issues on RunPod | FA3 may already be in the template; fall back to FA2 |

---

## Immediate Next Step

Pull thwu1's code from PR #180 and start Phase 1.
