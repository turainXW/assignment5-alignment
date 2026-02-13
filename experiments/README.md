# 学习率调优实验指南

## 快速开始

### 1. 单个学习率快速测试（推荐先运行）

测试学习率 1e-5（推荐值），运行10步验证代码：

```bash
cd /mnt/cs336/assignment5-alignment

deepspeed --num_gpus=4 cs336_alignment/grpo.py \
    --model_path data/sft_ds_size_full_lr5e-05_bs1x16 \
    --train_dataset_path data/MATH/math_train.jsonl \
    --val_dataset_path data/MATH/math_test.jsonl \
    --output_dir results/quick_test_lr1e-5 \
    --n_grpo_steps 10 \
    --learning_rate 1e-5 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 64 \
    --rollout_batch_size 128 \
    --group_size 8 \
    --sampling_temperature 1.0 \
    --sampling_min_tokens 4 \
    --sampling_max_tokens 512 \
    --val_interval 5 \
    --use_std_normalization
```

### 2. 完整学习率扫描（50步，节省时间版）

运行多个学习率实验（5e-6, 1e-5, 2e-5, 5e-5）：

```bash
cd /mnt/cs336/assignment5-alignment
./experiments/lr_sweep.sh
```

**预计时间**:
- 每个学习率实验: 约 2.5-4 小时（50步）
- 总共4个实验: 约 10-16 小时

### 3. 单个学习率完整训练（达到25%目标）

如果想单独训练一个学习率到更多步数：

```bash
deepspeed --num_gpus=4 cs336_alignment/grpo.py \
    --model_path data/sft_ds_size_full_lr5e-05_bs1x16 \
    --train_dataset_path data/MATH/math_train.jsonl \
    --val_dataset_path data/MATH/math_test.jsonl \
    --output_dir results/grpo_lr1e-5_full \
    --n_grpo_steps 100 \
    --learning_rate 1e-5 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 64 \
    --rollout_batch_size 128 \
    --group_size 8 \
    --sampling_temperature 1.0 \
    --sampling_min_tokens 4 \
    --sampling_max_tokens 512 \
    --val_interval 5 \
    --save_interval 20 \
    --use_std_normalization
```

## 分析结果

### 查看单个实验结果

```bash
# 可视化训练曲线
python plot_grpo_logs.py results/lr_sweep_*/lr_1e-5/training_log.json

# 或保存为图片
python plot_grpo_logs.py results/lr_sweep_*/lr_1e-5/training_log.json --output lr1e5_curves.png
```

### 对比所有学习率实验

```bash
# 自动对比所有学习率，生成对比图和总结
python experiments/compare_lr_results.py results/lr_sweep_YYYYMMDD_HHMMSS/
```

这会生成：
- `lr_sweep_comparison.png` - 6个子图的对比图表
- `summary.txt` - 文本总结

## 实验配置说明

### 当前配置（节省时间版）

- **训练步数**: 50步（原始200步的1/4）
- **验证间隔**: 每5步
- **学习率范围**: [5e-6, 1e-5, 2e-5, 5e-5]
- **Batch sizes**:
  - rollout_batch_size: 128
  - train_batch_size: 128
  - micro_batch_size: 2 (128/64)

### 如果需要更快的实验

修改 `experiments/lr_sweep.sh`：

```bash
N_GRPO_STEPS=30        # 减少到30步
ROLLOUT_BATCH_SIZE=64  # 减少rollout size
VAL_INTERVAL=10        # 减少验证频率
```

### 如果需要达到25%目标

根据50步实验选出最佳学习率后，单独训练100-150步：

```bash
# 使用最佳学习率
BEST_LR="1e-5"  # 根据实验结果修改

deepspeed --num_gpus=4 cs336_alignment/grpo.py \
    --model_path data/sft_ds_size_full_lr5e-05_bs1x16 \
    --train_dataset_path data/MATH/math_train.jsonl \
    --val_dataset_path data/MATH/math_test.jsonl \
    --output_dir results/grpo_best_lr_full \
    --n_grpo_steps 150 \
    --learning_rate $BEST_LR \
    --train_batch_size 128 \
    --gradient_accumulation_steps 64 \
    --rollout_batch_size 128 \
    --group_size 8 \
    --sampling_temperature 1.0 \
    --sampling_min_tokens 4 \
    --sampling_max_tokens 512 \
    --val_interval 5 \
    --save_interval 25 \
    --use_std_normalization
```

## 监控训练

### 实时查看日志

```bash
# 如果使用nohup后台运行
tail -f nohup.out

# 或者直接在终端查看（会实时显示）
```

### 检查GPU使用

```bash
watch -n 1 nvidia-smi
```

### 中途停止训练

```bash
# 找到进程
ps aux | grep grpo.py

# 优雅停止（保存当前checkpoint）
kill -SIGTERM <PID>

# 强制停止
kill -9 <PID>
```

## 预期结果

### 50步实验后

你应该能看到：

1. **学习率过小（5e-6）**:
   - 训练缓慢，validation reward增长较慢
   - Loss下降缓慢
   - 可能需要更多步数

2. **学习率适中（1e-5, 2e-5）**:
   - 稳定的训练曲线
   - Validation reward稳步提升
   - 最有可能达到25%目标

3. **学习率过大（5e-5）**:
   - 训练可能不稳定
   - Gradient norm较大
   - Validation reward可能震荡

### 100-150步完整训练后

使用最佳学习率应该能达到：
- **Validation Accuracy**: > 25% (目标)
- **Training Reward**: 0.5-0.7
- **Loss**: 稳定收敛

## 故障排除

### OOM错误

```bash
# 减小batch size
--rollout_batch_size 64 \
--train_batch_size 64 \
--sampling_max_tokens 256
```

### 训练发散（loss=nan）

```bash
# 降低学习率
--learning_rate 5e-6

# 或增加梯度累积
--gradient_accumulation_steps 128
```

### vLLM初始化失败

检查GPU显存：
```bash
nvidia-smi
# 如果显存不足，先清理：
pkill -f python
```

## 交付物

实验完成后，准备以下内容：

1. **Validation reward curves** - 使用 `compare_lr_results.py` 生成
2. **达到25%的模型** - 在 `results/grpo_best_lr_full/final_model/`
3. **2句话分析** - 基于对比实验的观察

示例分析：
```
"较高的学习率(5e-5)导致训练初期梯度范数较大且validation reward震荡，而较小的学习率(5e-6)收敛速度较慢。
最优学习率(1e-5)在稳定性和收敛速度之间取得了平衡，token entropy的适度下降表明模型在保持探索性的同时逐渐优化策略。"
```
