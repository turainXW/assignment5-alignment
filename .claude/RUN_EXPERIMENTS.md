# SFT实验运行指南

## 环境配置
- 4个GPU：cuda:0, cuda:1, cuda:2, cuda:3
- 训练：4个GPU并行（DeepSpeed ZeRO Stage 2）
- 推理：单个GPU

## 训练命令

### 1. 不同数据集大小的训练

```bash
# 激活虚拟环境
source .venv/bin/activate

# 训练 128 samples
./cs336_alignment/run_sft_deepspeed.sh 128

# 训练 256 samples
./cs336_alignment/run_sft_deepspeed.sh 256

# 训练 512 samples
./cs336_alignment/run_sft_deepspeed.sh 512

# 训练 1024 samples
./cs336_alignment/run_sft_deepspeed.sh 1024

# 训练 full dataset (7500 samples)
./cs336_alignment/run_sft_deepspeed.sh full
```

### 2. 自定义训练参数

```bash
# 使用环境变量自定义参数
LR=1e-4 BS=4 GRAD_ACCUM=4 EPOCHS=5 ./cs336_alignment/run_sft_deepspeed.sh 256
```

## 评估命令

训练完成后，使用单GPU进行评估：

```bash
# 评估训练好的模型
python cs336_alignment/evaluate_sft.py \
    --model_path ./sft_outputs/sft_ds_size_128_lr5e-05_bs2x8 \
    --max_samples 500 \
    --device cuda:0 \
    --gpu_memory_utilization 0.9

# 保存评估结果到文件
python cs336_alignment/evaluate_sft.py \
    --model_path ./sft_outputs/sft_ds_size_128_lr5e-05_bs2x8 \
    --max_samples 500 \
    --device cuda:0 \
    --output_file ./results/eval_128.json
```

## 模型保存位置

训练后的模型保存在：
```
./sft_outputs/sft_ds_size_{SIZE}_lr{LR}_bs{BS}x{GRAD_ACCUM}/
```

例如：
- `./sft_outputs/sft_ds_size_128_lr5e-05_bs2x8/`
- `./sft_outputs/sft_ds_size_256_lr5e-05_bs2x8/`

模型格式：
- ✅ safetensors格式（安全且兼容vLLM）
- ✅ transformers标准格式
- ✅ 包含完整的tokenizer和config

## WandB监控

所有训练run可以在WandB查看：
- Project: math-sft-deepspeed
- URL: https://wandb.ai/912868332-peking-university/math-sft-deepspeed

## 当前完成状态

✅ 128 samples训练完成
- Loss: Epoch 1: 0.0936 → Epoch 2: 0.4230 → Epoch 3: 0.3556
- 模型已保存为safetensors格式

## 下一步任务

1. [ ] 训练256, 512, 1024, full数据集
2. [ ] 评估所有训练好的模型
3. [ ] 生成validation accuracy曲线
4. [ ] 筛选正确样本重新训练
5. [ ] 达到≥15%准确率目标
