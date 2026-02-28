#!/usr/bin/env bash
# Launch script for TRL GRPO training with vLLM colocate mode.
#
# Architecture: 4 GPUs training + vLLM colocated (shared GPUs)
#
# Usage (all args on ONE line):
#   ./scripts/run_trl_grpo.sh --learning_rate 1e-5 --run_name lr_1e-5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${SCRIPT_DIR}/.venv"
PYTHON="${VENV}/bin/python"

# wandb API key
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# Defaults
MODEL_PATH="${SCRIPT_DIR}/data/sft_r1_format"
TRAIN_DATA="${SCRIPT_DIR}/data/MATH/math_train_r1.jsonl"
VAL_DATA="${SCRIPT_DIR}/data/MATH/math_test_r1.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/data/trl_grpo_output"
DS_CONFIG="${SCRIPT_DIR}/cs336_alignment/ds_zero2_trl.json"
NUM_GPUS=4

echo "============================================"
echo "TRL GRPO Training (vLLM colocate mode)"
echo "  GPUs:   ${NUM_GPUS}"
echo "  Model:  ${MODEL_PATH}"
echo "  Batch:  per_device=2 x ${NUM_GPUS} GPUs x gas=12 = $((2 * NUM_GPUS * 12))"
echo "============================================"

# Find a free port for distributed training
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

$PYTHON -m accelerate.commands.launch \
    --num_processes "$NUM_GPUS" \
    --main_process_port "$MASTER_PORT" \
    -m cs336_alignment.trl_grpo_train \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DS_CONFIG" \
    --gradient_checkpointing \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 12 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --learning_rate 1e-6 \
    --max_steps 200 \
    --eval_strategy no \
    --save_steps 50 \
    "$@"

echo "Training complete!"
