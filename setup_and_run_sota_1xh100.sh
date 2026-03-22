#!/usr/bin/env bash
# === Parameter Golf: SOTA comparison on 1x H100 (RunPod) ===
# Default target: PR #198 (1.1318 bpb on 8xH100, current best open result in local notes)
#
# Usage on RunPod:
#   git clone https://github.com/User123331/runpod-testing.git
#   cd runpod-testing
#   HF_TOKEN=hf_xxx bash setup_and_run_sota_1xh100.sh
#
# Optional:
#   TARGET_PR=180 bash setup_and_run_sota_1xh100.sh   # thwu1 merged record
#   SEED=42 bash setup_and_run_sota_1xh100.sh
#   TRAIN_SHARDS=1 bash setup_and_run_sota_1xh100.sh  # smoke download only

set -euo pipefail

log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_clean_checkout() {
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "Existing checkout at ${SRC_DIR} has uncommitted changes. Use a fresh SRC_DIR."
    fi
}

discover_legacy_hf_token() {
    python3 - "$@" <<'PY'
from pathlib import Path
import re
import sys

pattern = re.compile(r'export\s+HF_TOKEN="\$\{HF_TOKEN:-([^"}]+)\}"')

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = pattern.search(text)
    if match:
        print(f"{match.group(1)}\t{path}")
        sys.exit(0)

sys.exit(1)
PY
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PR="${TARGET_PR:-198}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
SRC_DIR="${SRC_DIR:-${WORKSPACE_DIR}/parameter-golf-pr${TARGET_PR}}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/logs}"
TRAIN_SHARDS="${TRAIN_SHARDS:-80}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-786432}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-2048}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
HF_TOKEN_SOURCE=""

case "${TARGET_PR}" in
    198)
        TARGET_SHA="${TARGET_SHA:-372bddea57f465c7217c5e26af2252a803221518}"
        TRAIN_SCRIPT_REL="records/track_10min_16mb/2026-03-20_11L_Int6_MLP3x_WD04_SmearBigram2k_1.1318/train_gpt.py"
        NUM_LAYERS="${NUM_LAYERS:-11}"
        BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-2048}"
        MATRIX_LR="${MATRIX_LR:-0.025}"
        SCALAR_LR="${SCALAR_LR:-0.025}"
        TIED_EMBED_LR="${TIED_EMBED_LR:-0.035}"
        MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
        MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
        MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
        WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
        ITERATIONS="${ITERATIONS:-9000}"
        EVAL_STRIDE="${EVAL_STRIDE:-64}"
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
        TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
        SWA_EVERY="${SWA_EVERY:-200}"
        MUON_WD="${MUON_WD:-0.04}"
        ADAM_WD="${ADAM_WD:-0.04}"
        SEED="${SEED:-1337}"
        REQUIRED_PY_MODULES="flash_attn_interface"
        ;;
    180)
        TARGET_SHA="${TARGET_SHA:-1a8be36c17e20b1fb53dbf4975e1d67f5b8a63e9}"
        TRAIN_SCRIPT_REL="records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py"
        NUM_LAYERS="${NUM_LAYERS:-10}"
        BIGRAM_VOCAB_SIZE="${BIGRAM_VOCAB_SIZE:-10240}"
        MATRIX_LR="${MATRIX_LR:-0.02}"
        SCALAR_LR="${SCALAR_LR:-0.02}"
        TIED_EMBED_LR="${TIED_EMBED_LR:-0.03}"
        MUON_MOMENTUM="${MUON_MOMENTUM:-0.99}"
        MUON_MOMENTUM_WARMUP_START="${MUON_MOMENTUM_WARMUP_START:-0.92}"
        MUON_MOMENTUM_WARMUP_STEPS="${MUON_MOMENTUM_WARMUP_STEPS:-1500}"
        WARMDOWN_ITERS="${WARMDOWN_ITERS:-3000}"
        ITERATIONS="${ITERATIONS:-9000}"
        EVAL_STRIDE="${EVAL_STRIDE:-64}"
        VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-500}"
        TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
        SWA_START_FRAC="${SWA_START_FRAC:-0.4}"
        SWA_EVERY="${SWA_EVERY:-50}"
        WEIGHT_DECAY="${WEIGHT_DECAY:-0.04}"
        SEED="${SEED:-42}"
        REQUIRED_PY_MODULES=""
        ;;
    *)
        die "Unsupported TARGET_PR=${TARGET_PR}. Use 198 (default) or 180."
        ;;
esac

RUN_ID="${RUN_ID:-pr${TARGET_PR}_1xh100_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

require_cmd git
require_cmd python3
require_cmd torchrun
require_cmd nvidia-smi

GPU_COUNT="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
[ "${GPU_COUNT}" -ge 1 ] || die "No GPUs detected."

log "Detected ${GPU_COUNT} GPU(s). This script uses exactly 1 GPU."
if [ -z "${HF_TOKEN}" ]; then
    if TOKEN_RECORD="$(
        discover_legacy_hf_token \
            "${SCRIPT_DIR}/setup_and_run.sh" \
            "${SCRIPT_DIR}/setup_and_run_1h.sh" \
            "${SCRIPT_DIR}/run_custom_tokenizer_pipeline.sh" \
            "${SCRIPT_DIR}/../parameter-golf/setup_and_run.sh" \
            "${SCRIPT_DIR}/../parameter-golf/setup_and_run_1h.sh" \
            "${SCRIPT_DIR}/../trainer-tokenizer/setup_runpod.sh"
    )"; then
        IFS=$'\t' read -r HF_TOKEN HF_TOKEN_SOURCE <<< "${TOKEN_RECORD}"
    fi
fi

if [ -n "${HF_TOKEN}" ]; then
    export HF_TOKEN
    if [ -n "${HF_TOKEN_SOURCE}" ]; then
        warn "Using HF token found in existing local script: ${HF_TOKEN_SOURCE}"
    else
        log "HF token detected in environment; authenticated downloads enabled."
    fi
else
    warn "HF_TOKEN/HUGGINGFACE_TOKEN not set. Public downloads may still work, but auth is recommended."
fi

python3 - <<'PY'
import importlib.util
import subprocess
import sys

missing = [pkg for pkg in ("huggingface_hub", "zstandard") if importlib.util.find_spec(pkg) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
PY

mkdir -p "${LOG_DIR}"

if [ ! -d "${SRC_DIR}/.git" ]; then
    log "Cloning openai/parameter-golf into ${SRC_DIR}"
    git clone https://github.com/openai/parameter-golf.git "${SRC_DIR}"
fi

cd "${SRC_DIR}"
require_clean_checkout

log "Fetching PR #${TARGET_PR}"
git fetch origin "pull/${TARGET_PR}/head:runpod-pr-${TARGET_PR}" --force
git checkout --detach "${TARGET_SHA}"

CURRENT_SHA="$(git rev-parse HEAD)"
[ "${CURRENT_SHA}" = "${TARGET_SHA}" ] || die "Checked out ${CURRENT_SHA}, expected ${TARGET_SHA}"
log "Checked out PR #${TARGET_PR} commit ${CURRENT_SHA}"
[ -f "${TRAIN_SCRIPT_REL}" ] || die "Target training script not found: ${TRAIN_SCRIPT_REL}"

python3 - "${TRAIN_SCRIPT_REL}" "${REQUIRED_PY_MODULES}" <<'PY'
from pathlib import Path
import importlib.util
import sys

train_script = Path(sys.argv[1])
required_modules = [m for m in sys.argv[2].split(",") if m]

source = train_script.read_text(encoding="utf-8")
compile(source, str(train_script), "exec")

missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing required Python modules for {train_script}: {', '.join(missing)}")
PY
log "Preflight compile check passed for ${TRAIN_SCRIPT_REL}"

DATASET_DIR="data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH="data/tokenizers/fineweb_1024_bpe.model"

if [ ! -f "${DATASET_DIR}/fineweb_train_000000.bin" ] || [ ! -f "${TOKENIZER_PATH}" ]; then
    log "Downloading FineWeb cached dataset/tokenizer (train_shards=${TRAIN_SHARDS})"
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "${TRAIN_SHARDS}"
else
    log "Dataset/tokenizer already present; skipping download."
fi

TRAIN_COUNT="$(find "${DATASET_DIR}" -maxdepth 1 -name 'fineweb_train_*.bin' 2>/dev/null | wc -l | tr -d ' ')"
VAL_COUNT="$(find "${DATASET_DIR}" -maxdepth 1 -name 'fineweb_val_*.bin' 2>/dev/null | wc -l | tr -d ' ')"
log "Dataset ready: train_shards=${TRAIN_COUNT} val_shards=${VAL_COUNT}"

export PYTHONUNBUFFERED=1
export RUN_ID
export DATA_PATH="./${DATASET_DIR}"
export TOKENIZER_PATH="./${TOKENIZER_PATH}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export NUM_LAYERS
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-3.0}"
export TIE_EMBEDDINGS="${TIE_EMBEDDINGS:-1}"
export TRAIN_BATCH_TOKENS
export TRAIN_SEQ_LEN
export BIGRAM_VOCAB_SIZE
export BIGRAM_DIM="${BIGRAM_DIM:-128}"
export MATRIX_LR
export SCALAR_LR
export TIED_EMBED_LR
export MUON_MOMENTUM
export MUON_MOMENTUM_WARMUP_START
export MUON_MOMENTUM_WARMUP_STEPS
export SWA_ENABLED="${SWA_ENABLED:-1}"
export EVAL_STRIDE
export EVAL_BATCH_SEQS="${EVAL_BATCH_SEQS:-32}"
export ITERATIONS
export WARMDOWN_ITERS
export MAX_WALLCLOCK_SECONDS
export VAL_LOSS_EVERY
export TRAIN_LOG_EVERY
export SEED

if [ "${TARGET_PR}" = "198" ]; then
    export MUON_WD
    export ADAM_WD
    export SWA_EVERY
else
    export WEIGHT_DECAY
    export SWA_START_FRAC
    export SWA_EVERY
fi

log "Starting 1xH100 run for PR #${TARGET_PR}"
log "Run ID: ${RUN_ID}"
log "Log file: ${LOG_PATH}"

torchrun --standalone --nproc_per_node=1 "${TRAIN_SCRIPT_REL}" 2>&1 | tee "${LOG_PATH}"

log "Run completed. Final metrics:"
grep -E 'val_bpb|val_loss|artifact|bytes|final_int|submission|model_params|swa:' "${LOG_PATH}" | tail -20 || true

log "Done."
