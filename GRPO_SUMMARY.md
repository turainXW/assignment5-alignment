# GRPO训练实现总结

## 完成的工作

### 1. 修复和优化训练循环

**主要问题修复：**
- ✅ **修复梯度累积逻辑** - 原实现在每个microbatch后都调用`model_engine.step()`，现已修复为正确的梯度累积
- ✅ **添加梯度范数监控** - 使用`model_engine.get_global_grad_norm()`记录梯度范数
- ✅ **添加Token Entropy监控** - 监控模型的探索程度
- ✅ **优化验证流程** - 添加`enforce_eager=True`和`max_model_len=2048`避免OOM
- ✅ **改进日志系统** - 记录format reward、answer reward等详细信息
- ✅ **添加定期保存** - 每20步保存checkpoint

**性能优化：**
- ✅ 针对4块A10 GPU（23GB显存）优化超参数
- ✅ 降低batch size以适应显存限制
- ✅ 优化vLLM配置以减少显存使用
- ✅ 添加样例输出显示（每10步）

### 2. 创建的文件

```
/mnt/cs336/assignment5-alignment/
├── cs336_alignment/grpo.py          # 修复后的GRPO训练代码
├── train_grpo.sh                    # 完整训练脚本（200步）
├── train_grpo_test.sh               # 快速测试脚本（3步）
├── plot_grpo_logs.py                # 训练日志可视化工具
├── verify_grpo.py                   # 核心函数验证脚本
└── GRPO_README.md                   # 详细使用文档
```

## 关键改进

### 训练循环优化

**Before (有问题):**
```python
for mb_start in range(0, len(prompts), micro_batch_per_gpu):
    # ... forward pass ...
    model_engine.backward(loss)
    model_engine.step()  # ❌ 每个microbatch都step，破坏梯度累积
```

**After (正确):**
```python
for accum_step in range(gradient_accumulation_steps):
    # ... forward pass ...
    loss = loss / gradient_accumulation_steps  # 正确缩放
    model_engine.backward(loss)  # 累积梯度

grad_norm = model_engine.get_global_grad_norm()
model_engine.step()  # ✅ 所有梯度累积完成后才step
```

### 超参数优化（针对A10 GPU）

| 参数 | 原始值 | 优化值 | 说明 |
|------|-------|--------|------|
| train_batch_size | 256 | 128 | 减少显存使用 |
| gradient_accumulation_steps | 128 | 64 | micro_batch=2 |
| rollout_batch_size | 256 | 128 | 减少vLLM显存 |
| sampling_max_tokens | 1024 | 512 | 减少生成长度 |
| group_size | 8 | 8 | 保持不变 |

## 使用方法

### 快速测试（推荐先运行）

```bash
cd /mnt/cs336/assignment5-alignment
./train_grpo_test.sh
```

这将运行3个步骤的快速测试，验证：
- 模型加载
- vLLM生成
- 奖励计算
- 梯度计算
- 验证流程

**预期输出：**
```
Step   0 | loss: 0.xxxx | grad_norm: xx.xx | reward: 0.xxx | entropy: xx.xx
Step   2 | [Validation] reward: 0.xxx
```

### 完整训练

```bash
./train_grpo.sh
```

这将运行200步完整训练，大约需要：
- 每步约 3-5分钟（包括rollout和训练）
- 总时间约 10-16小时

### 监控训练

训练过程中可以实时查看：

1. **终端输出** - 实时显示loss、reward、grad_norm等
2. **样例rollout** - 每10步显示一个生成样例
3. **验证结果** - 每5步在1024个验证样本上评估

### 可视化结果

训练完成后：

```bash
python plot_grpo_logs.py results/grpo_output_*/training_log.json --output curves.png
```

这将生成包含6个子图的训练曲线：
1. Training Loss
2. Training Rewards (total, format, answer)
3. Validation Accuracy
4. Gradient Norm
5. Token Entropy
6. Train vs Val Rewards对比

## 预期结果

根据论文和实验经验，使用默认超参数应该看到：

### 初始阶段 (Steps 0-50)
- **Train Reward**: 0.1 → 0.3
- **Val Reward**: 0.05 → 0.15
- **Loss**: 快速下降
- **Grad Norm**: 可能较大，逐渐稳定

### 中期 (Steps 50-150)
- **Train Reward**: 0.3 → 0.5
- **Val Reward**: 0.15 → 0.35
- **Loss**: 继续下降
- **Grad Norm**: 稳定在合理范围

### 后期 (Steps 150-200)
- **Train Reward**: 0.5 → 0.7
- **Val Reward**: 0.35 → 0.45
- **Loss**: 趋于稳定
- **Token Entropy**: 轻微下降（正常）

## 故障排除

### OOM (显存不足)

如果遇到显存错误，尝试以下调整：

```bash
# 选项1：减小rollout batch
--rollout_batch_size 64 --train_batch_size 64

# 选项2：减小生成长度
--sampling_max_tokens 256

# 选项3：减小group size
--group_size 4

# 选项4：降低vLLM显存使用
修改grpo.py中的gpu_memory_utilization=0.7
```

### 训练不稳定

如果loss震荡或梯度爆炸：

```bash
# 选项1：降低学习率
--learning_rate 5e-6

# 选项2：增加梯度累积
--gradient_accumulation_steps 128

# 选项3：检查梯度裁剪
# 已在ds_config.json中设置为1.0
```

### vLLM初始化失败

如果vLLM启动失败：

```bash
# 清理GPU缓存
export CUDA_VISIBLE_DEVICES=0,1,2,3
pkill -f python

# 或修改grpo.py中的LLM初始化
llm = LLM(model=tmp_path,
         gpu_memory_utilization=0.7,  # 降低
         max_model_len=1024,           # 降低
         enforce_eager=True)
```

## 实现细节

### 关键算法

**Group Normalized Rewards (GRPO核心):**
```python
# 对每个prompt的group_size个响应：
# 1. 计算原始rewards
# 2. 计算group内的mean和std
# 3. 归一化：advantage = (reward - mean) / (std + eps)
```

**Loss Types:**
- `no_baseline`: -reward * log_prob
- `reinforce_with_baseline`: -advantage * log_prob (推荐)
- `grpo_clip`: PPO-style clipping (off-policy only)

**DeepSpeed集成:**
- ZeRO-2优化器状态分片
- 自动梯度累积和同步
- 自动梯度裁剪
- BF16混合精度

### 代码结构

```
grpo_train_loop():
  for step in range(n_grpo_steps):
    1. 采样prompts（从训练集）
    2. vLLM rollout（生成responses）
    3. 计算rewards和advantages
    4. 计算old log probs
    5. 梯度累积训练循环
       for accum_step:
         - Forward pass
         - 计算loss
         - Backward（累积梯度）
       - Step（应用累积的梯度）
    6. 验证（每val_interval步）
    7. 保存checkpoint（每save_interval步）
```

## 检查清单

在开始训练前，确保：

- [x] GPU可用: `nvidia-smi`
- [x] SFT模型存在: `data/sft_ds_size_full_lr5e-05_bs1x16/`
- [x] 数据文件存在: `data/MATH/math_train.jsonl`, `math_test.jsonl`
- [x] DeepSpeed配置: `cs336_alignment/ds_config.json`
- [x] 输出目录有写权限: `results/`

## 下一步

1. **快速测试**: `./train_grpo_test.sh`
2. **检查输出**: 确认训练正常运行
3. **完整训练**: `./train_grpo.sh`
4. **监控进度**: 观察loss、reward、grad_norm
5. **可视化结果**: `python plot_grpo_logs.py ...`
6. **调优超参**: 根据结果调整超参数

## 参考

- Assignment PDF: Section 7.1 (GRPO Algorithm)
- DeepSpeed: https://www.deepspeed.ai/
- vLLM: https://docs.vllm.ai/
