# GRPO训练说明

## 概述

本项目实现了GRPO (Group Relative Policy Optimization)算法，用于对SFT模型进行强化学习后训练。

## GPU配置

- **硬件**: 4块NVIDIA A10 GPU (每块23GB显存)
- **模型**: Qwen2.5-Math-1.5B
- **SFT模型路径**: `data/sft_ds_size_full_lr5e-05_bs1x16/`

## 超参数配置

根据A10 GPU的显存限制，训练超参数已优化为:

```bash
n_grpo_steps: 200
learning_rate: 1e-5
train_batch_size: 128
gradient_accumulation_steps: 64  # micro_batch_size = 2
rollout_batch_size: 128
group_size: 8
sampling_temperature: 1.0
sampling_min_tokens: 4
sampling_max_tokens: 512
```

## 快速开始

### 1. 快速测试（推荐先运行）

运行3个步骤的快速测试，验证代码是否正常工作：

```bash
./train_grpo_test.sh
```

### 2. 完整训练

运行完整的200步训练：

```bash
./train_grpo.sh
```

### 3. 自定义训练

如果需要自定义超参数：

```bash
deepspeed --num_gpus=4 cs336_alignment/grpo.py \
    --model_path data/sft_ds_size_full_lr5e-05_bs1x16 \
    --train_dataset_path data/MATH/math_train.jsonl \
    --val_dataset_path data/MATH/math_test.jsonl \
    --output_dir results/grpo_custom \
    --n_grpo_steps 200 \
    --learning_rate 1e-5 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 64 \
    --rollout_batch_size 128 \
    --group_size 8 \
    --sampling_temperature 1.0 \
    --sampling_max_tokens 512 \
    --val_interval 5 \
    --save_interval 20 \
    --use_std_normalization
```

## 可视化训练日志

训练完成后，使用以下命令生成训练曲线：

```bash
python plot_grpo_logs.py results/grpo_output_XXXXXX/training_log.json --output training_curves.png
```

或者直接显示图表：

```bash
python plot_grpo_logs.py results/grpo_output_XXXXXX/training_log.json
```

## 输出文件

训练会生成以下文件：

```
results/grpo_output_YYYYMMDD_HHMMSS/
├── checkpoint_step_20/          # 第20步的checkpoint
├── checkpoint_step_40/          # 第40步的checkpoint
├── ...
├── final_model/                 # 最终模型
├── training_log.json            # 完整训练日志
├── training_log_step_20.json    # 第20步的日志快照
└── ...
```

## 监控训练

### 查看实时日志

训练过程中会输出详细的日志信息：

- Loss
- Gradient Norm
- Train Rewards (total, format, answer)
- Validation Rewards (每5步)
- Token Entropy
- 样例Rollout (每10步)

### 训练日志包含

1. **Loss曲线** - 策略梯度损失
2. **Train Rewards** - 训练集奖励（总奖励、格式奖励、答案奖励）
3. **Validation Rewards** - 验证集准确率
4. **Gradient Norm** - 梯度范数（用于监控训练稳定性）
5. **Token Entropy** - Token熵（监控模型的探索程度）

## 调优建议

### 如果遇到OOM (显存不足):

1. 降低 `rollout_batch_size` (例如: 64)
2. 降低 `train_batch_size` (例如: 64)
3. 降低 `sampling_max_tokens` (例如: 256)
4. 降低 `group_size` (例如: 4)

### 如果训练不稳定:

1. 降低 `learning_rate` (例如: 5e-6)
2. 增加 `gradient_accumulation_steps`
3. 检查 `grad_norm` 是否过大

### 如果想加速训练:

1. 增加 `rollout_batch_size` (如果显存允许)
2. 减少 `val_interval` (验证更少)
3. 增加 `group_size` (更多采样)

## 实现细节

### 关键组件

1. **compute_group_normalized_rewards** - 计算组归一化的优势函数
2. **compute_policy_gradient_loss** - 计算策略梯度损失（支持3种loss类型）
3. **grpo_train_loop** - 完整的GRPO训练循环

### Loss类型

- `no_baseline`: REINFORCE without baseline
- `reinforce_with_baseline`: REINFORCE with group-normalized baseline (推荐)
- `grpo_clip`: PPO-style clipped loss (仅用于off-policy)

### DeepSpeed ZeRO-2

使用DeepSpeed ZeRO-2进行分布式训练：

- 自动梯度累积
- 自动梯度裁剪 (clip_value=1.0)
- Optimizer状态分片
- 梯度通信overlap

## 故障排除

### 1. vLLM初始化失败

如果遇到vLLM初始化错误，尝试：
- 降低 `gpu_memory_utilization` (例如: 0.75)
- 设置 `max_model_len` 为更小的值

### 2. DeepSpeed错误

确保环境变量正确设置：
```bash
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 3. 数据加载错误

确保数据文件路径正确：
- 训练数据: `data/MATH/math_train.jsonl`
- 验证数据: `data/MATH/math_test.jsonl`

## 预期结果

使用默认超参数，你应该看到：

1. **初始阶段 (0-50步)**:
   - Train reward: 0.1-0.3
   - Loss逐渐下降
   - Val reward开始上升

2. **中期 (50-150步)**:
   - Train reward: 0.3-0.5
   - Val reward稳步提升
   - Grad norm稳定

3. **后期 (150-200步)**:
   - Train reward: 0.5-0.7
   - Val reward趋于收敛
   - Loss稳定在较低水平

## 参考资料

- GRPO算法: Assignment 5 - Section 7.1
- DeepSpeed文档: https://www.deepspeed.ai/
- vLLM文档: https://docs.vllm.ai/
