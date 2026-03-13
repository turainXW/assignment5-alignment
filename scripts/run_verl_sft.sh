#!/usr/bin/env bash
# veRL FSDP SFT training — 4×A10 (23GB each), Qwen2.5-3B
# Usage:
#   bash scripts/run_verl_sft.sh
#   WANDB_MODE=offline bash scripts/run_verl_sft.sh   # no wandb upload
#
# Prerequisites:
#   1. pip install verl qwen-vl-utils
#   2. python scripts/prepare_verl_sft_data.py
#   3. Model in data/models/Qwen2.5-3B/

set -euo pipefail

cd "$(dirname "$0")/.."      # always run from repo root

source .venv/bin/activate

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
TRAIN_DATA="data/verl_sft/train.parquet"
VAL_DATA="data/verl_sft/val.parquet"
MODEL_PATH="data/models/Qwen2.5-3B"
OUTPUT_DIR="data/verl_sft_output"

# --------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------
for f in "$TRAIN_DATA" "$VAL_DATA" "$MODEL_PATH"; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: required path not found: $f"
        echo "  Run prepare_verl_sft_data.py first."
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"

echo "========================================================"
echo " veRL SFT — Qwen2.5-3B — 4×A10"
echo " Train:  $TRAIN_DATA"
echo " Val:    $VAL_DATA"
echo " Model:  $MODEL_PATH"
echo " Output: $OUTPUT_DIR"
echo "========================================================"

python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=512 \
    data.truncation=right \
    model.partial_pretrain="$MODEL_PATH" \
    model.fsdp_config.model_dtype=bfloat16 \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.project_name=cs336_sft \
    trainer.experiment_name=qwen25_3b_ultrachat_safety \
    trainer.total_epochs=1 \
    "trainer.logger=[console,wandb]" \
    trainer.save_freq=7123 \
    trainer.test_freq=500 \
    trainer.max_ckpt_to_keep=1 \
    "trainer.checkpoint.save_contents=[model]" \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.03 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=8

echo "Training complete. Model saved to $OUTPUT_DIR"
