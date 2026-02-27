#!/usr/bin/env bash
# Launch R1-format SFT training using TRL SFTTrainer with DeepSpeed.
#
# Usage:
#   ./scripts/run_r1_sft.sh
#   ./scripts/run_r1_sft.sh --learning_rate 1e-5 --num_train_epochs 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${SCRIPT_DIR}/.venv"
ACCELERATE="${VENV}/bin/accelerate"

# wandb API key
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# Defaults
MODEL_PATH="${SCRIPT_DIR}/data/a5-alignment/models/Qwen2.5-Math-1.5B"
TRAIN_DATA="${SCRIPT_DIR}/data/MATH/math_train_r1.jsonl"
VAL_DATA="${SCRIPT_DIR}/data/MATH/math_test_r1.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/data/sft_r1_format"
DS_CONFIG="${SCRIPT_DIR}/cs336_alignment/ds_zero2_trl.json"

NUM_GPUS=4

echo "============================================"
echo "R1-format SFT Training (TRL SFTTrainer)"
echo "  GPUs:    ${NUM_GPUS}"
echo "  Model:   ${MODEL_PATH}"
echo "  Output:  ${OUTPUT_DIR}"
echo "============================================"

# Step 1: Prepare R1-format data (if not already done)
if [ ! -f "$TRAIN_DATA" ]; then
    echo "[Data] Converting MATH data to R1 format..."
    python3 -m cs336_alignment.prepare_r1_sft_data \
        --train_input "${SCRIPT_DIR}/data/MATH/math_train.jsonl" \
        --test_input "${SCRIPT_DIR}/data/MATH/math_test.jsonl" \
        --train_output "$TRAIN_DATA" \
        --test_output "$VAL_DATA"
    echo "[Data] Done."
else
    echo "[Data] R1-format data already exists, skipping conversion."
fi

# Step 2: Launch SFT training
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "[Train] Launching on ${NUM_GPUS} GPUs..."
$ACCELERATE launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MASTER_PORT" \
    -m cs336_alignment.trl_sft_train \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DS_CONFIG" \
    --gradient_checkpointing \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    "$@"

echo "R1 SFT training complete! Model saved to ${OUTPUT_DIR}"
