#!/usr/bin/env bash
# === Hyperbolic.ai Quick Start ===
# Run from home directory

set -euo pipefail
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
(while true; do sleep 60; nvidia-smi > /dev/null 2>&1; done) &
trap "kill $! 2>/dev/null" EXIT

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')
log "Detected ${GPU_COUNT} GPUs"

WORK_DIR="$HOME/golf"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Clone repos
[ ! -d "parameter-golf" ] && git clone https://github.com/openai/parameter-golf.git
[ ! -d "runpod-testing" ] && git clone https://github.com/User123331/runpod-testing.git

# Install FA3 using pre-compiled .so + cloned Python interface
if ! python3 -c "from flash_attn_interface import flash_attn_func" 2>/dev/null; then
    log "Installing FA3 (pre-compiled .so + Python interface)..."

    # Clone flash-attention repo for Python interface files
    [ ! -d "flash-attention" ] && git clone https://github.com/Dao-AILab/flash-attention.git

    # Copy pre-compiled .so into place
    cd "${WORK_DIR}/runpod-testing/compiled FA3"
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    mkdir -p "${SITE_PACKAGES}/flash_attn_3"
    cp _C.abi3.so "${SITE_PACKAGES}/flash_attn_3/"
    cp flash_attn_config.py "${SITE_PACKAGES}/flash_attn_3/"

    # Copy Python interface from flash-attention/hopper/flash_attn_3
    cd "${WORK_DIR}/flash-attention/hopper"
    cp -r flash_attn_3/*.py "${SITE_PACKAGES}/flash_attn_3/" 2>/dev/null || true

    # Install the interface package
    pip install -e . --no-build-isolation 2>/dev/null || {
        cp flash_attn_interface.py "${SITE_PACKAGES}/" 2>/dev/null || true
    }

    # Symlink flash_attn_config.py to torch path (fixes torch.compile backward crash)
    TORCH_PATH=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
    ln -sf "${SITE_PACKAGES}/flash_attn_3/flash_attn_config.py" "${TORCH_PATH}/flash_attn_config.py" 2>/dev/null || true

    log "FA3 installed"
    python3 -c "from flash_attn_interface import flash_attn_func; print('FA3: OK')" || {
        log "WARNING: FA3 interface check failed, will need selective build"
    }
fi

# Download dataset
cd "${WORK_DIR}/parameter-golf"
log "Downloading FineWeb dataset (8B tokens)..."
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Symlink data from parameter-golf to runpod-testing
cd "${WORK_DIR}/runpod-testing"
mkdir -p data/datasets data/tokenizers
[ ! -L "data/datasets/fineweb10B_sp1024" ] && \
    ln -s "${WORK_DIR}/parameter-golf/data/datasets/fineweb10B_sp1024" data/datasets/
[ ! -L "data/tokenizers/fineweb_1024_bpe.model" ] && \
    ln -s "${WORK_DIR}/parameter-golf/data/tokenizers/fineweb_1024_bpe.model" data/tokenizers/

log ""
log "=== Setup Complete ==="
log "Work dir: ${WORK_DIR}"
log "GPUs: ${GPU_COUNT}"
log "FA3: $(python3 -c 'from flash_attn_interface import flash_attn_func; print("OK")' 2>/dev/null || echo 'FAILED')"
log "Dataset: $(ls -1 data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l) train shards"
log ""
log "Ready! Run experiments with:"
log "  cd ${WORK_DIR}/runpod-testing"
log "  MODE=mos bash run_mos_sota.sh"