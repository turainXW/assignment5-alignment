#!/bin/bash
# 使用 DeepSpeed 启动 SFT 训练
# Usage: ./run_sft_deepspeed.sh <dataset_size> [epochs] [learning_rate] [batch_size] [grad_accum]
# Example: ./run_sft_deepspeed.sh 512 10 5e-5 2 8
#          ./run_sft_deepspeed.sh full 5

set -e

SIZE=${1:-128}
EPOCHS=${2:-${EPOCHS:-3}}
LR=${3:-${LR:-5e-5}}
BS=${4:-${BS:-1}}  # 降低默认batch size从2到1，避免OOM
GRAD_ACCUM=${5:-${GRAD_ACCUM:-16}}  # 增加梯度累积以保持有效batch size

echo "=========================================="
echo "SFT Training with DeepSpeed"
echo "=========================================="
echo "Dataset size: $SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Batch size per GPU: $BS"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Training GPUs: cuda:0, cuda:1, cuda:2, cuda:3 (4 GPUs)"
echo ""
echo "Usage: ./run_sft_deepspeed.sh <size> [epochs] [lr] [bs] [grad_accum]"
echo "Examples:"
echo "  ./run_sft_deepspeed.sh 512 10        # 512 samples, 10 epochs"
echo "  ./run_sft_deepspeed.sh full 5        # full dataset, 5 epochs"
echo "  ./run_sft_deepspeed.sh 1024 15 1e-4  # custom lr"
echo "=========================================="
echo ""

cd /root/assignment5-alignment
source .venv/bin/activate

# 使用 DeepSpeed 启动器 - 4个GPU并行训练
deepspeed --num_gpus=4 \
    --master_port=29500 \
    cs336_alignment/sft_trainer_deepspeed.py \
    --dataset_size $SIZE \
    --learning_rate $LR \
    --batch_size $BS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $EPOCHS \
    --wandb_project "math-sft-deepspeed"

echo ""
echo "✓ Training complete!"
