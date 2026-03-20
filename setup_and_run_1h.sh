#!/bin/bash
# === Parameter Golf: MoS 1-Hour Validation on 1x H100 ===
# Paste this into your RunPod terminal.
# Total time: ~18 min download + 60 min MoS run = ~78 min
# Target to beat: val_bpb 1.2540 (PR#111 vanilla baseline, 1hr/1xH100)

set -e

echo "=== Step 1: Download dataset ==="
cd /workspace/parameter-golf

# HF token for faster downloads
export HF_TOKEN="${HF_TOKEN:-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}"

# Download full dataset (~18 min)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify dataset
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: $TRAIN_COUNT  Val shards: $VAL_COUNT"
if [ "$TRAIN_COUNT" -lt 1 ]; then
    echo "ERROR: No training shards found. Dataset download failed."
    exit 1
fi

echo ""
echo "=== Step 2: Run MoS K=2 R=64 (1 HOUR, 1x H100) ==="
echo "Start time: $(date)"
echo "Target: beat PR#111 baseline val_bpb=1.2540"
echo ""

RUN_ID=mos_k2_r64_1h \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
USE_MOS=1 \
MOS_K=2 \
MOS_RANK=64 \
WARMDOWN_ITERS=100 \
MAX_WALLCLOCK_SECONDS=3600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee /workspace/mos_1h_log.txt

echo ""
echo "=== RESULTS ==="
echo ""
grep -E 'val_bpb|val_loss|bytes|param|model_params|stopping' /workspace/mos_1h_log.txt | tail -20
echo ""
echo "=== COMPARISON ==="
echo "Target (PR#111 vanilla 1hr): val_bpb=1.2540"
echo "Our 10-min MoS pilot:        val_bpb=1.3932"
echo "PR#111 10-min baseline:       val_bpb=1.3486"
echo ""
echo "Done at: $(date)"
