# TRL GRPO 训练调试记录

## 目标

使用 TRL 框架的 GRPOTrainer 对 R1 SFT 模型 (`data/sft_r1_format/`) 进行 GRPO 强化学习训练，提升数学推理正确率（SFT 后格式正确率 83.6%，答案正确率仅 9.0%）。

## 环境问题与解决

### 1. TRL 依赖版本不兼容

**原始环境**: vLLM 0.7.2, PyTorch 2.5.1, transformers 5.2.0, TRL 0.29.0

**问题 1**: `FSDPModule` 导入错误
- TRL 0.29.0 的 `trl/models/utils.py` 导入 `from torch.distributed.fsdp import FSDPModule`
- FSDPModule 在 PyTorch 2.6+ 才可用，当前 2.5.1 缺失
- 根因: TRL 0.29.0 依赖 PyTorch >= 2.6

**问题 2**: `Qwen2Tokenizer.all_special_tokens_extended` 缺失
- vLLM 0.7.2 访问 `tokenizer.all_special_tokens_extended`，但 transformers 5.2.0 已移除该属性

**解决方案**: 升级 vLLM
```bash
pip install "vllm>=0.10.2"
# 结果: vLLM 0.16.0 → PyTorch 2.9.1 → transformers 4.57.6
# 但后续发现 vLLM 0.16.0 与 TRL 0.29.0 的 NCCL 通信不兼容
# 最终使用: vLLM 0.12.0 (TRL 官方支持 0.10.2-0.12.0)
pip install "vllm==0.12.0"
# 最终版本: vLLM 0.12.0, PyTorch 2.9.0, transformers 4.57.6
```

### 2. flash-attn 二进制不兼容

升级 PyTorch 后，旧版 flash-attn 2.7.4 的 CUDA 二进制与 PyTorch 2.9 不兼容：
```
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs
```

**解决方案**: 卸载 flash-attn，vLLM 使用内置注意力实现
```bash
pip uninstall flash-attn -y
```
同时在 `trl_grpo_train.py` 中移除 `"attn_implementation": "flash_attention_2"`。

### 3. tokenizer_config.json 格式不兼容

transformers 4.57.6 要求 `extra_special_tokens` 为 dict，但模型保存的是 list：
```json
// 修改前 (错误):
"extra_special_tokens": ["<|im_start|>", "<|im_end|>", ...]

// 修改后 (正确):
"extra_special_tokens": {
    "im_start_token": "<|im_start|>",
    "im_end_token": "<|im_end|>",
    ...
}
```

### 4. GRPOConfig API 变更

TRL 0.29.0 的 GRPOConfig 移除了 `max_prompt_length` 参数，需要从代码中删除。

### 5. 批次大小约束

```
ValueError: The global eval batch size (2 * 3) must be divisible by the number of generations (4).
```
全局批次 = per_device_batch × num_gpus，必须能被 num_generations 整除。

### 6. vLLM server 模式 NCCL 通信失败（关键问题）

**架构**: 3 GPUs 训练 (0,1,2) + 1 GPU vLLM server (3)

TRL 的 server 模式使用 NCCL 在训练进程和 vLLM 服务器之间同步权重。但 `CUDA_VISIBLE_DEVICES` 隔离导致 NCCL 无法跨进程通信：
- vLLM 进程 (`CUDA_VISIBLE_DEVICES=3`) 只能看到 1 个 GPU
- 训练进程需要通过 NCCL 与 vLLM 通信，但 GPU 3 对训练进程不可见（或反之）
- 尝试 `CUDA_VISIBLE_DEVICES=3,0,1,2` 也无法解决

**解决方案**: 改用 `colocate` 模式

### 7. colocate 模式 OOM

4 GPUs 全部用于训练 + vLLM 共享，每卡需要同时容纳训练参数、优化器状态和 vLLM 推理模型。

**解决方案**: 减小 per_device_train_batch_size 和 vllm_gpu_memory_utilization。

---

## 最终正确配置

### 文件修改清单

1. **`data/sft_r1_format/tokenizer_config.json`**: `extra_special_tokens` 改为 dict 格式
2. **`cs336_alignment/trl_grpo_train.py`**: 移除 `max_prompt_length`、`flash_attention_2`，添加 `vllm_mode`/`vllm_gpu_memory_utilization` 参数
3. **`cs336_alignment/trl_grpo_data.py`**: 支持 R1 格式数据集（自动检测 `<answer>` 标签）
4. **`scripts/run_trl_grpo.sh`**: colocate 模式启动脚本

### 启动命令

```bash
# 确保 GPU 空闲
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

# 启动训练
cd /mnt/cs336/assignment5-alignment
bash scripts/run_trl_grpo.sh

# 带自定义参数
bash scripts/run_trl_grpo.sh --learning_rate 1e-5 --run_name lr_1e-5
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| vllm_mode | colocate | vLLM 与训练共享 GPU，避免 NCCL 通信问题 |
| vllm_gpu_memory_utilization | 0.2 | 降低 vLLM 内存占用，避免 OOM |
| per_device_train_batch_size | 2 | 减小以避免 OOM |
| gradient_accumulation_steps | 12 | 保持有效批次 2×4×12=96 |
| num_generations | 4 | 每个 prompt 生成 4 个回答 |
| max_completion_length | 1024 | 避免回答被截断 |
| learning_rate | 1e-6 | RL 训练使用较低学习率 |
| max_steps | 200 | 训练步数 |
| deepspeed | ZeRO-2 | 分片优化器状态 |
| gradient_checkpointing | true | 节省显存 |
| beta | 0.0 | 无 KL 惩罚 |
| temperature | 1.0 | 采样温度 |

### 当前环境版本

```
vLLM:         0.12.0  (TRL 支持 0.10.2-0.12.0)
PyTorch:      2.9.0+cu128
transformers: 4.57.6
TRL:          0.29.0
DeepSpeed:    已安装
flash-attn:   已卸载（不兼容 PyTorch 2.9）
```

### 如果 OOM 仍然出现

1. 进一步降低 `vllm_gpu_memory_utilization` 到 0.15
2. 减小 `per_device_train_batch_size` 到 1（需要 `num_generations` 也改为 1 或 2）
3. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少碎片
4. 考虑回退到 server 模式（需要解决 NCCL 问题或使用全部 GPU 可见方案）

### 待解决

- colocate 模式下 OOM 需要进一步调优内存配置
- 如要使用 server 模式：需要所有进程能看到所有 GPU，不能用 CUDA_VISIBLE_DEVICES 隔离
