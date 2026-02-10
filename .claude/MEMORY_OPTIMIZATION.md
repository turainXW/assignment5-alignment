# 内存优化记录

## 问题
训练full数据集时遇到OOM（Out of Memory）错误
- GPU显存: 23GB
- 使用量: 21.97GB
- 可用: 仅89MB

## 已应用的优化

### 1. FlashAttention2 ✅
- 已启用: `attn_implementation="flash_attention_2"`
- 效果: 降低注意力机制的显存使用

### 2. DeepSpeed ZeRO-2 ✅
- 已启用: `stage: 2`
- 效果: 将优化器状态分片到4个GPU

### 3. 混合精度训练 ✅
- 已启用: bfloat16
- 效果: 减半显存使用

### 4. 梯度检查点 ⚡ NEW
**刚刚添加**:
```python
model.gradient_checkpointing_enable()
```
- 效果: 节省约30-50%的激活值显存
- 代价: 训练速度降低约20%（值得的！）

### 5. 减小Batch Size ⚡ NEW
**刚刚修改**:
- Batch size: 2 → 1
- Gradient accumulation: 8 → 16
- 有效batch size保持不变: 64

## 预期效果

### 显存使用（每个GPU）
**之前（OOM）:**
- 模型: ~3GB
- 优化器状态: ~6GB (ZeRO-2分片后)
- 梯度: ~3GB
- 激活值: ~10GB
- **总计: ~22GB (超过23GB限制)**

**优化后:**
- 模型: ~3GB
- 优化器状态: ~6GB
- 梯度: ~3GB
- 激活值: ~5GB (梯度检查点减半)
- **总计: ~17GB ✅ 有6GB余量**

## 训练时间影响

### Full数据集（7500样本，5 epochs）
**优化前估算**: ~8小时
**优化后估算**: ~10小时（梯度检查点增加20%时间）

### 1024样本（10 epochs）
**预计时间**: ~2小时

## 建议的训练顺序

### 选项1: 先测试小数据集（推荐）
```bash
# 1. 测试1024样本（验证内存优化有效）
./cs336_alignment/run_sft_deepspeed.sh 1024 10

# 2. 如果成功，再训练full
./cs336_alignment/run_sft_deepspeed.sh full 5
```

### 选项2: 直接训练full（冒险）
```bash
# 直接训练完整数据集
./cs336_alignment/run_sft_deepspeed.sh full 5
```

### 选项3: 进一步降低显存（如果还OOM）
```bash
# 更激进的设置
export BS=1
export GRAD_ACCUM=32  # 进一步增加梯度累积
./cs336_alignment/run_sft_deepspeed.sh full 5 5e-5 1 32
```

## 其他可能的优化（如果还不够）

1. **CPU Offload优化器**
   - 修改DeepSpeed config: `"offload_optimizer": {"device": "cpu"}`
   - 效果: 节省~6GB显存
   - 代价: 训练速度降低50%

2. **ZeRO-3**
   - 更激进的分片策略
   - 将模型参数也分片
   - 效果: 节省更多显存
   - 代价: 通信开销增加

3. **减少序列长度**
   - 如果数据允许，可以截断更短
   - 效果: 显存使用与序列长度平方成正比

## 监控命令

训练期间监控GPU:
```bash
watch -n 1 nvidia-smi
```

检查是否OOM:
```bash
tail -f wandb/latest-run/logs/debug.log
```
