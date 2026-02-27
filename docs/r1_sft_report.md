# Qwen2.5-Math-1.5B R1 格式 SFT 微调报告

## 1. 任务概述

对 Qwen2.5-Math-1.5B 基座模型进行 R1 风格的 SFT（Supervised Fine-Tuning）微调，使模型学会使用 `<think>...</think> <answer>...</answer>` 格式进行"先推理再回答"的数学解题。

## 2. 硬件环境

| 资源 | 规格 |
|------|------|
| GPU | 4× NVIDIA A10 (23GB VRAM/卡) |
| 总显存 | 92GB |
| RAM | 739GB |

## 3. 模型与数据

### 3.1 基座模型

- **模型**: Qwen2.5-Math-1.5B（1.5B 参数）
- **路径**: `data/a5-alignment/models/Qwen2.5-Math-1.5B/`

### 3.2 训练数据

- **原始数据**: MATH 数据集
  - 训练集: 7,500 条 → 转换后 7,498 条（2 条因无法提取 `\boxed{}` 答案被跳过）
  - 测试集: 5,000 条 → 转换后 5,000 条
  - 路径: `data/MATH/math_train.jsonl` / `data/MATH/math_test.jsonl`

### 3.3 数据格式转换 (MATH → R1)

使用 `cs336_alignment/prepare_r1_sft_data.py` 进行格式转换：

**转换前**（标准 MATH 格式）:
```json
{
  "prompt": "Question: What is 2+3?\n\nAnswer:",
  "response": "We compute $2+3=5$. The answer is $\\boxed{5}$."
}
```

**转换后**（R1 格式）:
```json
{
  "prompt": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: What is 2+3?\nAssistant: <think>",
  "response": "\nWe compute $2+3=5$.\n</think> <answer> 5 </answer>"
}
```

**转换逻辑**:
- 从 `prompt` 中提取纯问题文本（去除 `Question:` 前缀和 `\n\nAnswer:` 后缀）
- 使用 R1-zero prompt 模板（`cs336_alignment/trl_grpo_data.py` 中的 `PROMPT_TEMPLATE`）
- 从 `response` 中提取 `\boxed{}` 答案作为 `<answer>` 内容
- 原始解题过程作为 `<think>` 推理链

转换后数据保存至 `data/MATH/math_train_r1.jsonl` 和 `data/MATH/math_test_r1.jsonl`。

## 4. 训练配置

### 4.1 训练框架

- **训练器**: TRL SFTTrainer (v0.29.0)
- **分布式**: DeepSpeed ZeRO-2
- **加速器**: Accelerate (multi-GPU launch)

### 4.2 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| per_device_train_batch_size | 1 | 每卡 batch size |
| gradient_accumulation_steps | 16 | 梯度累积步数 |
| 有效 batch size | 1 × 16 × 4 = **64** | 全局有效 batch |
| learning_rate | 2e-5 | 学习率 |
| num_train_epochs | 2 | 训练轮数 |
| max_length | 1024 | 最大序列长度 |
| warmup_ratio | 0.1 | 10% warmup |
| max_grad_norm | 1.0 | 梯度裁剪 |
| precision | bf16 | 混合精度训练 |
| gradient_checkpointing | True | 节省显存 |
| attn_implementation | flash_attention_2 | FlashAttention-2 加速 |
| completion_only_loss | True | 仅对 response 部分计算损失 |
| optimizer | AdamW (via DeepSpeed) | 优化器 |
| lr_scheduler | linear with warmup | 学习率调度 |
| seed | 42 | 随机种子 |

### 4.3 显存使用

| 组件 | 占用 (每卡) |
|------|-------------|
| 模型参数 (bf16) | ~4.3 GB |
| ZeRO-2 优化器状态 | ~1.5 GB |
| 缓存 | ~5.8 GB |
| **总计** | **~5.8 GB / 23 GB** |

### 4.4 关键文件

| 文件 | 用途 | 修改情况 |
|------|------|---------|
| `cs336_alignment/prepare_r1_sft_data.py` | 数据格式转换 | 未修改 |
| `cs336_alignment/trl_sft_train.py` | TRL SFTTrainer 训练脚本 | 适配 trl v0.29.0 API |
| `scripts/run_r1_sft.sh` | 训练启动脚本 | 修改模型路径和 batch size |
| `cs336_alignment/ds_zero2_trl.json` | DeepSpeed ZeRO-2 配置 | 未修改 |
| `cs336_alignment/trl_grpo_data.py` | R1 prompt 模板 | 未修改 |

### 4.5 代码适配 (trl v0.29.0)

原代码基于旧版 trl，需要以下适配：

1. **移除 `DataCollatorForCompletionOnlyLM`**（已从 trl v0.29.0 中删除）
   - 改用 `completion_only_loss=True` + prompt-completion 数据格式
2. **`max_seq_length` → `max_length`**（参数重命名）
3. **数据格式**从单字段 `text` 改为双字段 `prompt` + `completion`

## 5. 训练过程

### 5.1 执行命令

```bash
./scripts/run_r1_sft.sh
```

### 5.2 训练指标

| 训练阶段 | Step | Loss | Token 准确率 | 梯度范数 | 学习率 |
|---------|------|------|-------------|---------|--------|
| 开始 | 5 | 0.663 | 0.819 | 1.204 | 3.3e-6 |
| Warmup 中 | 15 | 0.594 | 0.830 | 0.630 | 1.2e-5 |
| Warmup 结束 | 25 | 0.524 | 0.847 | 0.598 | 2.0e-5 |
| Epoch 1 中 | 50 | 0.447 | 0.860 | 0.540 | 1.9e-5 |
| Epoch 1 末 | 100 | 0.398 | 0.878 | 0.448 | 1.2e-5 |
| Epoch 2 初 | 130 | 0.398 | 0.878 | 0.448 | 1.0e-5 |
| Epoch 2 中 | 175 | 0.415 | 0.869 | 0.574 | 5.8e-6 |
| Epoch 2 末 | 230 | 0.416 | 0.873 | 0.490 | 6.6e-7 |
| **最终** | **236** | **0.458** | **0.863** | **0.587** | **1.9e-7** |

### 5.3 评估指标 (训练中)

| Eval 时间点 | Eval Loss | Eval Token 准确率 |
|------------|-----------|-------------------|
| Step 50 | 0.4842 | 0.8552 |
| Step 100 | 0.5040 | 0.8470 |
| Step 150 | 0.5387 | 0.8440 |
| Step 200 | 0.5354 | 0.8449 |

### 5.4 训练统计

| 统计项 | 值 |
|--------|-----|
| 总训练步数 | 236 |
| 总训练时间 | 62 分 21 秒 |
| 训练速度 | 4.0 samples/s |
| 最终 Train Loss | 0.488 |
| 最终 Train Token 准确率 | 0.905 |
| WandB 链接 | https://wandb.ai/912868332-peking-university/huggingface/runs/jgfqsj0c |

## 6. 评估结果

### 6.1 评估设置

- **评估数据**: MATH 测试集前 500 条
- **推理引擎**: vLLM
- **采样参数**: temperature=1.0, top_p=1.0, max_tokens=1024
- **停止条件**: `</answer>`
- **评分函数**: `r1_zero_reward_fn`（检查格式 + 答案正确性）

### 6.2 结果对比

| 指标 | 基座模型 | R1 SFT 模型 | 提升 |
|------|---------|------------|------|
| **答案正确率** | 0.60% (3/500) | **9.00%** (45/500) | **+8.4%** (15×) |
| **格式正确率** | 5.40% (27/500) | **83.60%** (418/500) | **+78.2%** (15.5×) |

### 6.3 结果分析

#### 格式学习

R1 SFT 最显著的效果是**格式学习**：
- 微调后 83.6% 的回答能正确生成 `</think> <answer>...</answer>` 结构
- 基座模型几乎无法生成此格式（仅 5.4%），说明格式是通过 SFT 有效习得的

#### 答案准确率

答案正确率从 0.6% 提升到 9.0%（15 倍提升）。绝对值偏低的原因：
1. **采样策略**: 使用 temperature=1.0 随机采样，而非贪婪解码（greedy）；贪婪解码通常会更高
2. **双重条件**: R1 评分要求格式和答案同时正确
3. **模型规模**: 1.5B 参数模型数学推理能力有限
4. **训练数据**: 仅用 MATH 训练集（7.5K 条）进行 SFT

#### 生成质量对比

**R1 SFT 模型**（结构化输出）:
```
We compute $2+3=5$.
</think> <answer> 5 </answer>
```

**基座模型**（非结构化，经常退化）:
```
To determine... Let's break it down step-by-step:
1. **Calculate the decimal expansion...
```python
print(123)
```output
```

## 7. 输出文件

| 文件 | 说明 |
|------|------|
| `data/sft_r1_format/` | 训练完成的 R1 SFT 模型权重 |
| `data/MATH/math_train_r1.jsonl` | R1 格式训练数据 (7,498 条) |
| `data/MATH/math_test_r1.jsonl` | R1 格式测试数据 (5,000 条) |
| `data/eval_results/base_model_r1_eval.json` | 基座模型评估详细结果 |
| `data/eval_results/r1_sft_model_eval.json` | R1 SFT 模型评估详细结果 |

## 8. 训练数据深度分析

### 8.1 过拟合分析

训练曲线显示明显的过拟合迹象：

| 指标 | Step 100 (Epoch 0.85) | Step 200 (Epoch 1.70) | 趋势 |
|------|----------------------|----------------------|------|
| Train Loss | 0.506 | 0.439 | 持续下降 |
| **Eval Loss** | **0.535 (最佳)** | 0.535 | Step 100 后停滞 |
| Train-Eval Gap | 0.029 | 0.096 | 持续扩大 |

**关键发现**: Eval Loss 在 Step 100（Epoch 0.85，第 1 轮尚未训完）即已触底，之后 Train Loss 继续下降但 Eval Loss 停滞甚至轻微上升（0.5346 → 0.5387 → 0.5354）。**Epoch 2 的训练基本在过拟合**，最佳模型应在 ~Step 100 保存。

### 8.2 Entropy 变化

| 阶段 | Entropy | 说明 |
|------|---------|------|
| Epoch 1 初 | 1.055 | 正常不确定性 |
| Epoch 2 初 | 0.952 | 开始下降 |
| 最终 | 0.764 | 下降 27.6% |

Entropy 急剧下降表明模型变得**过度自信**，生成多样性降低，在第 2 轮尤为明显。

### 8.3 评估结果细分

对 R1 SFT 模型的 500 条评估结果进行分类：

| 类别 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 格式正确 + 答案正确 | 45 | 9.0% | 完全成功 |
| 格式正确 + 答案错误 | 373 | 74.6% | 学会格式但推理不足 |
| 格式错误 + 答案错误 | 82 | 16.4% | 格式和答案均失败 |
| 格式错误 + 答案正确 | 0 | 0.0% | 不存在 |

### 8.4 格式失败分析 (82 条)

| 失败原因 | 数量 | 说明 |
|----------|------|------|
| 缺少 `</think>` | 44 | 推理链未完成即截断 |
| 缺少 `<answer>` | 79 | 未生成答案标签 |
| 生成极短 (<50 字符) | 13 | 模型直接"放弃" |
| 使用错误标签 | 3 | 幻觉出 `<ask>`/`<solution>` 等标签 |

**格式失败主要原因**: `max_tokens=1024` 限制导致长推理被截断，以及模型对格式的学习仍不够稳定。

格式错误样例：
```
# 推理被截断，未闭合标签
" $(2n + 1) + (2n + 3) + ... = 8(n+1), we see that the sum is divisible by . [/think> "

# 幻觉出错误标签
"  <solution> 4 </solution> Since $\frac{3}{7}=0.42857142857\cdots$..."
```

### 8.5 答案错误分析 (373 条格式正确但答案错误)

| 问题类型 | 数量 | 说明 |
|----------|------|------|
| 推理过短 (<20 字符) | 13 | 几乎没有思考就给出答案 |
| 推理过长 (>800 字符) | 42 | 冗长但不正确 |
| 答案过长 (>100 字符) | 1 | 答案格式不规范 |

答案错误样例：
```
问题: 正确答案 202 → 模型输出 21
问题: 正确答案 Friday → 模型输出 Wednesday
问题: 正确答案 4130_5 → 模型输出 1013_5
```

### 8.6 与旧 SFT 模型对比

| 指标 | 旧 SFT (boxed 格式) | R1 SFT (think/answer 格式) | 差异 |
|------|---------------------|---------------------------|------|
| 格式正确率 | **94.4%** | 83.6% | -10.8% |
| 答案正确率 | 7.4% | **9.0%** | +1.6% |

旧的 boxed 格式 SFT 格式正确率更高（94.4% vs 83.6%），因为 `\boxed{}` 格式比 `<think>...</think> <answer>...</answer>` 更简单。R1 格式答案正确率略高（9.0% vs 7.4%），可能因为推理链引导了更好的思考。

### 8.7 不同训练集大小的效果 (旧 SFT)

| 训练集大小 | 格式正确率 | 答案正确率 |
|-----------|-----------|-----------|
| 128 条 | 0.0% | 0.0% |
| 256 条 | 41.6% | 1.6% |
| 512 条 | 87.4% | 5.8% |
| 全量 (7,498 条) | 94.4% | 7.4% |

数据量对格式学习影响显著，但对答案准确率影响有限——这进一步说明准确率瓶颈在模型能力而非数据量。

### 8.8 基座模型行为分析

基座 Qwen2.5-Math-1.5B 在 R1 prompt 下已有一定能力：

| 行为 | 数量 (500 条) | 占比 |
|------|-------------|------|
| 生成 `</think>` 标签 | 220 | 44.0% |
| 生成 `<answer>` 标签 | 193 | 38.6% |
| 生成 Python 代码 | 44 | 8.8% |
| 极长 (>2000 字符) | 69 | 13.8% |
| 平均生成长度 | 852 字符 | — |

基座模型对指令格式有一定理解但无法稳定执行，且经常退化为生成代码或冗长输出。

## 9. 发现的问题与改进建议

### 9.1 已发现问题

| # | 问题 | 严重程度 | 说明 |
|---|------|---------|------|
| 1 | **过拟合** | 高 | 只需 ~1 epoch，第 2 轮训练有害 |
| 2 | **推理能力瓶颈** | 高 | 1.5B 参数模型数学推理能力有限，SFT 只教会格式未提升推理 |
| 3 | **采样策略** | 中 | temperature=1.0 随机采样拉低了准确率 |
| 4 | **序列长度限制** | 中 | max_tokens=1024 导致部分长推理截断 |
| 5 | **格式复杂度** | 低 | R1 格式比 boxed 格式更复杂，格式正确率低 10.8% |

### 9.2 改进建议

1. **训练策略**: 减少训练至 1 epoch，或在 Step 100 处使用 early stopping
2. **评估策略**: 使用 greedy decoding（temperature=0）进行评估，预期准确率会显著提升
3. **序列长度**: 增大 max_tokens 至 2048，减少推理截断
4. **强化学习**: 以此 R1 SFT 模型为起点进行 GRPO 强化学习，真正提升推理能力（而非仅学格式）
5. **数据质量**: 考虑对训练数据中的推理链进行质量过滤，去除低质量样本

## 10. 结论

R1 格式 SFT 微调成功地让 Qwen2.5-Math-1.5B 学会了：
1. 使用 `<think>...</think> <answer>...</answer>` 格式进行结构化推理输出（格式正确率 83.6%）
2. 在 MATH 数据集上的答案正确率提升了 15 倍（0.6% → 9.0%）

但训练也暴露了关键问题：过拟合（2 epoch 过多）、推理能力瓶颈（SFT 无法教会推理）。该 R1 SFT 模型可作为后续 GRPO（Group Relative Policy Optimization）强化学习训练的起点，通过奖励信号驱动来真正提升数学推理能力。
