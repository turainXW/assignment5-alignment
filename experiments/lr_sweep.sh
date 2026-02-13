#!/bin/bash
# 学习率扫描实验 - 节省时间版本（50步）

# 基础配置
MODEL_PATH="data/sft_ds_size_full_lr5e-05_bs1x16"
TRAIN_DATA="data/MATH/math_train.jsonl"
VAL_DATA="data/MATH/math_test.jsonl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 实验超参数（减少训练步数以节约时间）
N_GRPO_STEPS=50  # 从200降到50
TRAIN_BATCH_SIZE=128
GRADIENT_ACCUMULATION_STEPS=64
ROLLOUT_BATCH_SIZE=128
GROUP_SIZE=8
SAMPLING_TEMPERATURE=1.0
SAMPLING_MIN_TOKENS=4
SAMPLING_MAX_TOKENS=512
VAL_INTERVAL=5
SAVE_INTERVAL=25

# 学习率列表（对数尺度扫描）
LEARNING_RATES=(
    "5e-6"   # 较小
    "1e-5"   # 推荐值
    "2e-5"   # 较大
    "5e-5"   # 更大
)

echo "======================================"
echo "学习率扫描实验"
echo "训练步数: $N_GRPO_STEPS (减少以节约时间)"
echo "学习率范围: ${LEARNING_RATES[@]}"
echo "======================================"
echo ""

# 依次运行每个学习率实验
for LR in "${LEARNING_RATES[@]}"; do
    OUTPUT_DIR="results/lr_sweep_${TIMESTAMP}/lr_${LR}"

    echo "======================================"
    echo "实验: 学习率 = $LR"
    echo "输出目录: $OUTPUT_DIR"
    echo "======================================"

    deepspeed --num_gpus=4 cs336_alignment/grpo.py \
        --model_path $MODEL_PATH \
        --train_dataset_path $TRAIN_DATA \
        --val_dataset_path $VAL_DATA \
        --output_dir $OUTPUT_DIR \
        --n_grpo_steps $N_GRPO_STEPS \
        --learning_rate $LR \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --rollout_batch_size $ROLLOUT_BATCH_SIZE \
        --group_size $GROUP_SIZE \
        --sampling_temperature $SAMPLING_TEMPERATURE \
        --sampling_min_tokens $SAMPLING_MIN_TOKENS \
        --sampling_max_tokens $SAMPLING_MAX_TOKENS \
        --val_interval $VAL_INTERVAL \
        --save_interval $SAVE_INTERVAL \
        --ds_config_path cs336_alignment/ds_config.json \
        --use_std_normalization \
        --loss_type reinforce_with_baseline \
        --advantage_eps 1e-6 \
        --cliprange 0.2 \
        --epochs_per_rollout_batch 1

    echo ""
    echo "实验完成: $LR"
    echo "======================================"
    echo ""
done

echo "======================================"
echo "所有实验完成!"
echo "生成对比图表..."
python experiments/compare_lr_results.py results/lr_sweep_${TIMESTAMP}
echo "======================================"
