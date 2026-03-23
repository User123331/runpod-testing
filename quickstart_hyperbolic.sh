#!/usr/bin/env bash
# === Hyperbolic.ai Quick Start ===
# Run from ~/runpod-testing directory

set -euo pipefail
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
(while true; do sleep 60; nvidia-smi > /dev/null 2>&1; done) &
trap "kill $! 2>/dev/null" EXIT

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
log "Detected ${GPU_COUNT} GPUs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Clone parameter-golf if needed
if [ ! -d "$HOME/parameter-golf" ]; then
    log "Cloning parameter-golf..."
    git clone https://github.com/openai/parameter-golf.git "$HOME/parameter-golf"
fi

# Build FA3 selectively (~5 min on H100)
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    log "Building Flash Attention 3 (selective, ~5 min)..."

    if [ ! -d "$HOME/flash-attention" ]; then
        git clone https://github.com/Dao-AILab/flash-attention.git "$HOME/flash-attention"
    fi

    cd "$HOME/flash-attention/hopper"
    rm -rf build/
    mkdir -p flash_attn_3

    # Only build bf16 hdim64 SM90 - skip everything else
    export FLASH_ATTENTION_DISABLE_FP16=TRUE
    export FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM128=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    export FLASH_ATTENTION_DISABLE_SM80=TRUE
    export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
    export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
    export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
    export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
    export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
    export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
    export FLASH_ATTENTION_DISABLE_CLUSTER=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE

    pip install --no-build-isolation --break-system-packages -e .

    # Symlink config to torch (fixes torch.compile backward crash)
    SITE_PACKAGES=$(python3 -c "import site; print(site.getusersitepackages())")
    TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
    ln -sf "${SITE_PACKAGES}/flash_attn_3/flash_attn_config.py" "${TORCH_PATH}/flash_attn_config.py" 2>/dev/null || true

    python3 -c "from flash_attn_interface import flash_attn_func; print('FA3: OK')"
fi

# Download dataset
cd "$HOME/parameter-golf"
log "Downloading FineWeb dataset (8B tokens)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Symlink data to runpod-testing
cd "${SCRIPT_DIR}"
mkdir -p data/datasets data/tokenizers
[ ! -L "data/datasets/fineweb10B_sp1024" ] && \
    ln -s "$HOME/parameter-golf/data/datasets/fineweb10B_sp1024" data/datasets/
[ ! -L "data/tokenizers/fineweb_1024_bpe.model" ] && \
    ln -s "$HOME/parameter-golf/data/tokenizers/fineweb_1024_bpe.model" data/tokenizers/

log ""
log "=== Setup Complete ==="
log "GPUs: ${GPU_COUNT}"
log "FA3: $(python3 -c 'from flash_attn_interface import flash_attn_func; print("OK")' 2>/dev/null || echo 'FAILED')"
log "Dataset: $(ls -1 data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l) train shards"
log ""
log "Ready! Run:"
log "  MODE=mos bash run_mos_sota.sh"