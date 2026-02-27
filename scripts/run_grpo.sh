#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

deepspeed --num_gpus 4 \
    --module cs336_alignment.grpo \
    --model_path "${SCRIPT_DIR}/data/sft_r1_format" \
    --train_dataset_path "${SCRIPT_DIR}/data/MATH/math_train_r1.jsonl" \
    --val_dataset_path "${SCRIPT_DIR}/data/MATH/math_test_r1.jsonl" \
    --output_dir "${SCRIPT_DIR}/data/grpo_output" \
    --ds_config_path "${SCRIPT_DIR}/cs336_alignment/ds_config.json" \
    --n_epochs 100 \
    --learning_rate 1e-6 \
    --rollout_batch_size 128 \
    --group_size 4 \
    --sampling_temperature 0.7 \
    --sampling_max_tokens 512 \
    --loss_type reinforce_with_baseline
