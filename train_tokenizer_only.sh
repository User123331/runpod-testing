#!/bin/bash
# =============================================================================
# Parameter Golf: Train Custom Tokenizer Only
# =============================================================================
#
# This script:
#   1. Downloads docs_selected.jsonl (~45GB)
#   2. Trains unigram tokenizer
#   3. Evaluates against baseline
#
# Output: data/tokenizers_custom/spm_unigram_1024.model (~1MB)
#
# Usage on RunPod:
#   git clone https://github.com/User123331/parameter-golf.git
#   cd parameter-golf
#   bash train_tokenizer_only.sh
# =============================================================================

set -e

VOCAB_SIZE=1024
MODEL_TYPE=unigram
MAX_TRAIN_DOCS=200000
EVAL_DOCS=10000

DATA_DIR="./data/datasets"
TOKENIZER_DIR="./data/tokenizers_custom"
DOCS_JSONL="${DATA_DIR}/docs_selected.jsonl"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }

echo ""
echo "============================================================"
echo "  Custom Tokenizer Training"
echo "  Vocab: ${VOCAB_SIZE}, Type: ${MODEL_TYPE}"
echo "============================================================"
echo ""

# Check disk
AVAIL_GB=$(df -BG . 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || echo "?")
log "Available disk: ${AVAIL_GB} GB (need ~50GB)"

# =============================================================================
# Step 1: Download docs_selected.jsonl
# =============================================================================
if [ -f "${DOCS_JSONL}" ]; then
    log "Docs already exist: $(du -h "${DOCS_JSONL}" | cut -f1)"
else
    log "Downloading docs_selected.jsonl (~45GB, 10-30 min)..."

    mkdir -p "${DATA_DIR}"
    pip install --quiet huggingface_hub

    python3 -c "
from huggingface_hub import hf_hub_download
import shutil, os

cached = hf_hub_download(
    repo_id='willdepueoai/parameter-golf',
    filename='docs_selected.jsonl',
    subfolder='datasets',
    repo_type='dataset',
)
src = os.path.realpath(cached)
dst = '${DOCS_JSONL}'
print(f'Copying {src} -> {dst}')
try:
    os.link(src, dst)
    print('Hard-linked (no extra disk)')
except OSError:
    shutil.copy2(src, dst)
    print('Copied')
"
    log "Download complete: $(du -h "${DOCS_JSONL}" | cut -f1)"
fi

# =============================================================================
# Step 2: Train unigram tokenizer
# =============================================================================
log "Training ${MODEL_TYPE} tokenizer (vocab=${VOCAB_SIZE})..."

mkdir -p "${TOKENIZER_DIR}"
pip install --quiet sentencepiece numpy

python3 data/train_tokenizer.py \
    --vocab-size ${VOCAB_SIZE} \
    --model-type ${MODEL_TYPE} \
    --docs-path "${DOCS_JSONL}" \
    --max-docs ${MAX_TRAIN_DOCS} \
    --eval-docs ${EVAL_DOCS} \
    --character-coverage 0.995

# =============================================================================
# Step 3: Evaluate baseline for comparison
# =============================================================================
log ""
log "Evaluating baseline tokenizer for comparison..."

BASELINE_MODEL="./data/tokenizers/fineweb_1024_bpe.model"
if [ -f "${BASELINE_MODEL}" ]; then
    python3 data/train_tokenizer.py \
        --evaluate "${BASELINE_MODEL}" \
        --docs-path "${DOCS_JSONL}" \
        --eval-docs ${EVAL_DOCS}
else
    log "Baseline not found at ${BASELINE_MODEL}"
    log "Download it with: python data/cached_challenge_fineweb.py --variant sp1024"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
log "TOKENIZER TRAINED: ${TOKENIZER_DIR}/spm_${MODEL_TYPE}_${VOCAB_SIZE}.model"
log ""
log "To download from RunPod:"
log "  scp root@<runpod-ip>:/workspace/parameter-golf/${TOKENIZER_DIR}/spm_${MODEL_TYPE}_${VOCAB_SIZE}.model ."
echo "============================================================"