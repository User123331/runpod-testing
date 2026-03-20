#!/bin/bash
# === Parameter Golf: MoS Pilot on 1x H100 ===
# Paste this entire script into your RunPod terminal.
# Uses the pre-built runpod/parameter-golf:latest template.
# Total time: ~18 min download + 10 min baseline + 10 min MoS = ~40 min

set -e

echo "=== Step 1: Clone fork and download dataset ==="
cd /workspace
git clone https://github.com/User123331/parameter-golf.git
cd parameter-golf

# Download full dataset (takes ~18 min, needs all 80 shards for proper training)
python3 data/cached_challenge_fineweb.py --variant sp1024 &
DOWNLOAD_PID=$!

echo "Dataset downloading in background (PID: $DOWNLOAD_PID)..."
echo "Waiting for download to complete..."
wait $DOWNLOAD_PID
echo "Dataset download complete!"

# Verify dataset
ls -la data/datasets/fineweb10B_sp1024/ | head -5
echo "Train shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)"
echo "Val shards: $(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)"

echo ""
echo "=== Step 2: Run Baseline (10 min, 1x H100) ==="
echo "Start time: $(date)"

RUN_ID=baseline_pilot \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee /workspace/baseline_log.txt

echo ""
echo "Baseline done at: $(date)"
echo ""

# Save baseline artifact
cp final_model.int8.ptz /workspace/baseline_model.int8.ptz 2>/dev/null || true

echo "=== Step 3: Run MoS K=2 rank=64 (10 min, 1x H100) ==="
echo "Start time: $(date)"

RUN_ID=mos_k2_r64_pilot \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=42 \
USE_MOS=1 \
MOS_K=2 \
MOS_RANK=64 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee /workspace/mos_log.txt

echo ""
echo "MoS done at: $(date)"

# Save MoS artifact
cp final_model.int8.ptz /workspace/mos_model.int8.ptz 2>/dev/null || true

echo ""
echo "============================================"
echo "=== RESULTS COMPARISON ==="
echo "============================================"
echo ""
echo "--- Baseline ---"
grep -E 'val_bpb|val_loss|bytes|param' /workspace/baseline_log.txt | tail -10
echo ""
echo "--- MoS K=2 rank=64 ---"
grep -E 'val_bpb|val_loss|bytes|param' /workspace/mos_log.txt | tail -10
echo ""
echo "--- Artifact Sizes ---"
ls -la /workspace/baseline_model.int8.ptz /workspace/mos_model.int8.ptz 2>/dev/null
echo ""
echo "Done! Copy these results back."
