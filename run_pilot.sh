#!/bin/bash
# Quick start script for 1x H100 MoS pilot
# Run from the parameter-golf repo root directory

set -e

echo "=== Parameter Golf MoS Pilot ==="
echo "Date: $(date)"
echo "GPU: 1x H100 SXM"
echo ""

# Configuration (all via env vars — train_gpt.py has no argparse)
ITERATIONS=2000
SEED=42
MOS_K=2
MOS_RANK=64  # Low-rank to fit in 16MB budget (~100KB vs ~500KB full-rank)

echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Seed: $SEED"
echo "  MoS K: $MOS_K"
echo "  MoS Rank: $MOS_RANK (0=full-rank)"
echo ""

# Baseline run
echo "=== Running Baseline ==="
ITERATIONS=$ITERATIONS SEED=$SEED MAX_WALLCLOCK_SECONDS=99999 \
    python3 train_gpt.py 2>&1 | tee baseline_log.txt

echo ""
echo "=== Running MoS K=$MOS_K rank=$MOS_RANK ==="
ITERATIONS=$ITERATIONS SEED=$SEED MAX_WALLCLOCK_SECONDS=99999 \
    USE_MOS=1 MOS_K=$MOS_K MOS_RANK=$MOS_RANK \
    python3 train_gpt.py 2>&1 | tee mos_k${MOS_K}_r${MOS_RANK}_log.txt

echo ""
echo "=== Done ==="
echo "Compare results:"
echo "  grep 'val_bpb' baseline_log.txt"
echo "  grep 'val_bpb' mos_k${MOS_K}_r${MOS_RANK}_log.txt"
echo "  grep 'bytes' baseline_log.txt"
echo "  grep 'bytes' mos_k${MOS_K}_r${MOS_RANK}_log.txt"