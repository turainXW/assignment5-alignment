#!/usr/bin/env bash
# Launch script for TRL GRPO training with vLLM inference server.
#
# Architecture: 3 GPUs training (0,1,2) + 1 GPU vLLM (3)
#
# Usage (all args on ONE line):
#   ./scripts/run_trl_grpo.sh --learning_rate 1e-5 --run_name lr_1e-5
#
# LR sweep:
#   for LR in 5e-6 1e-5 2e-5 5e-5; do ./scripts/run_trl_grpo.sh --learning_rate $LR --run_name lr_sweep_${LR}; done

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${SCRIPT_DIR}/.venv"
ACCELERATE="${VENV}/bin/accelerate"
TRL="${VENV}/bin/trl"

# wandb API key (new format, set as env var to bypass old client validation)
export WANDB_API_KEY="${WANDB_API_KEY:-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}"

# Defaults
MODEL_PATH="${SCRIPT_DIR}/data/sft_r1_format"
TRAIN_DATA="${SCRIPT_DIR}/data/MATH/math_train.jsonl"
VAL_DATA="${SCRIPT_DIR}/data/MATH/math_test.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/data/trl_grpo_output"
DS_CONFIG="${SCRIPT_DIR}/cs336_alignment/ds_zero2_trl.json"

# GPU allocation
VLLM_GPU=3
TRAIN_GPUS="0,1,2"
NUM_TRAIN_GPUS=3
VLLM_PORT=8000

echo "============================================"
echo "TRL GRPO Training (vLLM accelerated)"
echo "  Training GPUs: ${TRAIN_GPUS} (${NUM_TRAIN_GPUS} GPUs)"
echo "  vLLM GPU:      ${VLLM_GPU}"
echo "  Model:         ${MODEL_PATH}"
echo "  Batch:         per_device=8 x ${NUM_TRAIN_GPUS} GPUs x gas=16 = $((8 * NUM_TRAIN_GPUS * 16))"
echo "============================================"

# --- Start vLLM server on dedicated GPU ---
echo "[vLLM] Starting server on GPU ${VLLM_GPU}, port ${VLLM_PORT}..."
CUDA_VISIBLE_DEVICES=${VLLM_GPU} $TRL vllm-serve \
    --model "$MODEL_PATH" \
    --port "$VLLM_PORT" \
    --gpu_memory_utilization 0.85 \
    --dtype bfloat16 \
    --max_model_len 1536 \
    &
VLLM_PID=$!

# Cleanup vLLM on exit
cleanup() {
    echo "[cleanup] Stopping vLLM server (PID ${VLLM_PID})..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup EXIT

# Wait for vLLM server to be ready
echo "[vLLM] Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[vLLM] Server ready after ${i}s!"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[vLLM] ERROR: Server process died."
        exit 1
    fi
    sleep 1
done

# Verify server is actually up
if ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "[vLLM] ERROR: Server not ready after 120s."
    exit 1
fi

# --- Launch training on remaining GPUs ---
# Find a free port for distributed training
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "[Train] Launching on GPUs ${TRAIN_GPUS} with ${NUM_TRAIN_GPUS} processes..."
CUDA_VISIBLE_DEVICES=${TRAIN_GPUS} $ACCELERATE launch \
    --num_processes "$NUM_TRAIN_GPUS" \
    --main_process_port "$MASTER_PORT" \
    -m cs336_alignment.trl_grpo_train \
    --model_path "$MODEL_PATH" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed "$DS_CONFIG" \
    --gradient_checkpointing \
    --use_vllm \
    --vllm_server_port "$VLLM_PORT" \
    "$@"

echo "Training complete!"
