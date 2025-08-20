#!/bin/bash
#SBATCH --job-name=vlm_parallel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j.out


set -euo pipefail
mkdir -p logs

echo "[INFO] SLURM_NODELIST=$SLURM_NODELIST"
echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] SLURM_TMPDIR=$SLURM_TMPDIR"

ENV_PATH=~/envs/activevision

source $ENV_PATH/bin/activate

export HF_HOME=$HOME/scratch/hf_cache
mkdir -p "$HF_HOME"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3

# === Pre-download model weights ===
# huggingface-cli login
# pip install -U "huggingface_hub[cli]"
# export HF_HUB_ENABLE_HF_TRANSFER=1
# huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$HF_HOME/models/Qwen2_5_VL_7B"

# === Disable flash-attn ===
# export TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa

# === NCCL settings ===
export NCCL_DEBUG=warn
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# === Configurable options (can be overridden with export before submission) ===
: ${LOCAL_MODELS:="$HF_HOME/models/Qwen2_5_VL_7B"}
: ${VLM_MODELS:="Qwen/Qwen2.5-VL-7B-Instruct"}
: ${IMG_DIR:="$HOME/scratch/images"}
: ${DTYPE:="bfloat16"}
: ${MAX_NEW:=128}
: ${BATCH:=4}
: ${QUANT:="none"}
: ${USE_TMP_COPY:=1}


if [ -n "$LOCAL_MODELS" ]; then
  if [ "$USE_TMP_COPY" -eq 1 ] && [ -n "$SLURM_TMPDIR" ]; then
    TMP_MODELS_DIR="$SLURM_TMPDIR/models"
    mkdir -p "$TMP_MODELS_DIR"

    MODELS_ARG=""
    IFS=',' read -ra LMODELS <<< "$LOCAL_MODELS"
    for m in "${LMODELS[@]}"; do
      name=$(basename "$m")
      dst="$TMP_MODELS_DIR/$name"
      echo "[INFO] Copying $m -> $dst"
      rsync -ah --info=progress2 "$m/" "$dst/"
      MODELS_ARG+="$dst,"
    done
    MODELS_ARG="${MODELS_ARG%,}"
  else
    MODELS_ARG="$LOCAL_MODELS"
  fi
else
  MODELS_ARG="$VLM_MODELS"
fi

echo "[INFO] GPUs on node:"
nvidia-smi || true
echo "[INFO] MODELS=$MODELS_ARG"
echo "[INFO] IMG_DIR=$IMG_DIR  DTYPE=$DTYPE  MAX_NEW=$MAX_NEW  BATCH=$BATCH  QUANT=$QUANT"

python vlm_parallel.py --models "$MODELS_ARG" --img_dir "$IMG_DIR" --dtype "$DTYPE" --max_new $MAX_NEW --batch $BATCH --quant $QUANT

echo "[DONE] Outputs: captions_*.jsonl  Logs: logs/${SLURM_JOB_ID}.out"
