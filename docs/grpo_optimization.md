# TRL GRPO 训练优化方案

## 当前问题

4 卡全部用于训练（无 vLLM），HF `model.generate()` 生成极慢：
- 每步生成 256 条 × 1024 tokens，耗时 ~30 分钟
- 200 步 × 30 分钟 ≈ **100 小时/次**，4 个 LR sweep ≈ 400 小时，不可接受

## 硬件环境

- 4× NVIDIA A10 (22GB each)
- 模型: Qwen2.5-1.5B (bf16)
- 当前显存占用: ~18.9GB/卡 (4 卡 ZeRO-2)

## 优化方案（按收益排序）

### 1. 启用 vLLM 推理服务器（最大收益，5-10x）

| | 当前 (HF generate) | 优化后 (vLLM) |
|---|---|---|
| 生成方式 | 每卡独立 model.generate() | 专用 vLLM 服务器，PagedAttention + CUDA graphs |
| 生成速度 | ~30 分钟/步 | ~30-60 秒/步 |
| GPU 分配 | 4 卡全训练 | 3 卡训练 + 1 卡 vLLM |

### 2. 降低 max_completion_length: 1024 → 512（~2x）

- 数学题的推理过程大多在 512 tokens 内完成
- 生成长度减半，速度翻倍
- 也减少训练阶段的序列长度，降低显存

### 3. 关闭 torch.compile（省首步编译时间）

- 在 GRPOConfig 中加 `torch_compile=False`
- 避免第一步长达数分钟的编译开销

### 4. 测试时关闭 eval（省 eval 生成时间）

- `--eval_steps 999` 或 `--report_to none`
- 快速验证 pipeline 时不需要跑 eval

### 5. 关闭 gradient_checkpointing（训练更快但更费显存）

- 减少训练阶段的重复计算
- 需要在显存允许的前提下使用

## 推荐最终配置

```
GPU 分配:    3 卡训练 (GPU 0,1,2) + 1 卡 vLLM (GPU 3)
DeepSpeed:   ZeRO-2 (3 卡分片)
batch:       per_device=4 × 3 GPUs × gas=32 = 384 effective batch
num_gen:     8 (group_size)
completion:  512 tokens
temperature: 1.0
optimizer:   AdamW(lr=sweep, betas=(0.9,0.95), wd=0.0)
grad_clip:   1.0
```

### 显存估算 (3 卡 ZeRO-2, per_device=4)

| 组件 | 每卡显存 |
|------|---------|
| 模型权重 (bf16) | 3 GB |
| 优化器 (ZeRO-2, /3) | 4 GB |
| 梯度 (ZeRO-2, /3) | 1 GB |
| 激活值 (batch=4, seq≤1024, grad_ckpt) | 2-3 GB |
| **合计** | **~10-11 GB** |
| 剩余 (22GB) | ~11 GB ✅ |

### 预估训练时间

```
每步: vLLM 生成 ~40秒 + 训练 ~30秒 ≈ 1.5 分钟
200 步: 200 × 1.5 ≈ 5 小时/次
4 个 LR sweep: ~20 小时 ✅
```

## LR Sweep 命令

```bash
for LR in 5e-6 1e-5 2e-5 5e-5; do ./scripts/run_trl_grpo.sh --learning_rate $LR --run_name lr_sweep_${LR}; done
```
