# SFT Training Scripts for MATH Dataset

完成 Assignment 5 Task 4.3: SFT Experiment

使用 **3 GPUs 分布式训练 + 1 GPU 推理**

## 文件说明

- `sft_trainer_ddp.py`: 分布式训练脚本（3 GPUs训练 + 1 GPU推理）
- `filter_correct_examples.py`: 过滤训练数据，只保留能生成正确答案的样本
- `run_all_sft_experiments_ddp.sh`: 运行所有实验的批处理脚本
- `run_single_experiment_ddp.sh`: 运行单个实验的脚本
- `QUICK_START.md`: 快速启动指南

## 依赖模块

所有脚本都使用 cs336_alignment 中已实现的函数：
- `cs336_alignment.tokenize.tokenize_prompt_and_output`: 数据预处理
- `cs336_alignment.task.get_response_log_probs`: 获取对数概率
- `cs336_alignment.task.sft_microbatch_train_step`: SFT 训练步骤
- `cs336_alignment.drgrpo_grader.r1_zero_reward_fn`: 评估函数

## 系统配置

### 分布式训练配置
- **GPU**: 4x NVIDIA A10 (23GB each)
- **训练**: 3 GPUs (cuda:0, cuda:1, cuda:2) - 使用 PyTorch DDP
- **推理**: 1 GPU (cuda:3) - 使用 vLLM
- **Model**: Qwen 2.5 Math 1.5B Base
- **有效 batch size**: batch_size × gradient_accumulation × world_size

### 为什么需要分布式训练？
- Qwen 1.5B 模型 + 梯度 + 优化器状态 ≈ 15-20GB
- 单GPU (23GB) 会 OOM
- 使用 3 GPUs 分布式训练可以分摊模型和梯度的内存

## 快速开始

### 运行所有实验

```bash
cd /root/assignment5-alignment
./cs336_alignment/run_all_sft_experiments_ddp.sh
```

这将依次运行：
1. 不同数据集大小的实验 (128, 256, 512, 1024, full)
2. 过滤正确样本并训练

### 运行单个实验

```bash
# 使用 128 个样本训练
./cs336_alignment/run_single_experiment_ddp.sh 128

# 使用完整数据集训练
./cs336_alignment/run_single_experiment_ddp.sh full

# 过滤并训练
./cs336_alignment/run_single_experiment_ddp.sh filtered
```

### 自定义超参数

```bash
# 设置环境变量来自定义超参数
export LR=1e-4
export BS=8  # Per GPU batch size
export GRAD_ACCUM=2
export EPOCHS=5
export WORLD_SIZE=3  # Number of training GPUs

./cs336_alignment/run_single_experiment_ddp.sh 512
```

## 直接使用 Python 脚本

### 分布式训练

```bash
python -m cs336_alignment.sft_trainer_ddp \
    --dataset_size 1024 \
    --learning_rate 5e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --eval_steps 100 \
    --world_size 3 \
    --vllm_device cuda:3 \
    --wandb_project "math-sft-experiment"
```

### 过滤数据集

```bash
python -m cs336_alignment.filter_correct_examples \
    --model_path "/root/assignment5-alignment/data/a5-alignment/models/Qwen2.5-Math-1.5B" \
    --input_data "/root/assignment5-alignment/data/MATH/math_train.jsonl" \
    --output_data "/root/assignment5-alignment/data/MATH/math_train_filtered.jsonl" \
    --device cuda:3 \
    --batch_size 32
```

### 在过滤后的数据集上训练

```bash
python -m cs336_alignment.sft_trainer_ddp \
    --dataset_size full \
    --train_data_path "/root/assignment5-alignment/data/MATH/math_train_filtered.jsonl" \
    --learning_rate 5e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --eval_steps 100 \
    --world_size 3 \
    --vllm_device cuda:3 \
    --wandb_project "math-sft-experiment" \
    --wandb_run_name "sft_filtered_correct"
```

## 实验任务

### Task 1: 不同数据集大小

训练 SFT 模型，数据集大小为：128, 256, 512, 1024, full (7500 examples)

目标：达到至少 15% 的验证准确率（使用完整数据集）

**Deliverable**: 不同数据集大小对应的验证准确率曲线

### Task 2: 过滤正确样本

1. 过滤训练集，只保留能产生正确答案的样本
2. 在过滤后的完整数据集上训练
3. 报告过滤后的数据集大小和验证准确率

**Deliverable**:
- 过滤后的数据集大小
- 验证准确率曲线
- 与原始数据集的对比

## 超参数调优建议

需要调优以达到 >15% 验证准确率：
- **Learning rate**: 尝试 [1e-5, 5e-5, 1e-4]
- **Batch size per GPU**: 尝试 [2, 4, 8]
- **Gradient accumulation**: 调整以获得合适的有效批量大小
- **Epochs**: 根据验证曲线调整

**注意**：使用分布式训练时，有效批量大小 = batch_size × gradient_accumulation × world_size

例如：
- batch_size=4, gradient_accumulation=4, world_size=3
- 有效批量大小 = 4 × 4 × 3 = 48

## 输出结果

- 训练好的模型: `./sft_outputs/`
- WandB 日志: 项目名 `math-sft-experiment`
- 过滤后的数据集: `/root/assignment5-alignment/data/MATH/math_train_filtered.jsonl`

## 关键实现细节

### 1. 分布式训练 (DDP)

```python
# 使用 PyTorch DistributedDataParallel
setup_distributed(rank, world_size)
model = DDP(model, device_ids=[rank])
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
```

### 2. GPU 分配

- **cuda:0, cuda:1, cuda:2**: 分布式训练，每个GPU运行一个进程
- **cuda:3**: vLLM 推理评估，只在 rank 0 进程中使用

### 3. 数据格式

MATH 数据集格式：
```json
{"prompt": "Question: ...\n\nAnswer:", "response": "solution..."}
```

### 4. 评估格式

使用 R1-Zero 格式进行评估，模型输出需要包含：
- `</think> <answer>` 标签
- `</answer>` 标签

### 5. Gradient Clipping

使用 clip value 1.0 (按照作业要求)

### 6. Wandb Metrics

- 训练指标使用 `train_step` 作为 x 轴
- 评估指标使用 `eval_step` 作为 x 轴
- 只在 rank 0 进程中记录日志

### 7. vLLM 权重加载

每次评估前需要将 policy 权重加载到 vLLM 实例中：
```python
load_policy_into_vllm_instance(policy, vllm_model)
```

注意：需要处理 DDP wrapper，使用 `policy.module.state_dict()`

## 故障排查

### OOM (Out of Memory)

如果仍然遇到内存不足：
- 减小 `batch_size` (per GPU)
- 减小 `max_eval_samples` (默认 500)
- 调整 `gpu_memory_utilization` for vLLM (默认 0.85)
- 减少训练 GPU 数量 (修改 `world_size`)

### 分布式训练初始化失败

确保：
- 所有训练 GPU (cuda:0-2) 可用
- MASTER_ADDR 和 MASTER_PORT 未被占用
- 使用 `torch.multiprocessing.spawn` 启动

检查命令：
```bash
nvidia-smi
```

### vLLM 在 cuda:3 上初始化失败

确保 cuda:3 没有被其他进程占用：
```bash
nvidia-smi
fuser -v /dev/nvidia3
```

### 进程卡住

检查是否所有 `dist.barrier()` 调用都被所有进程执行到。如果某个进程异常退出，其他进程会一直等待。

### WandB 登录

首次使用需要登录：
```bash
wandb login
```

## 性能优化建议

1. **调整 batch size**: 根据GPU内存使用情况增大 per-GPU batch size
2. **减少 eval_steps**: 如果评估太慢，可以降低评估频率 (例如从 100 改为 200)
3. **使用梯度检查点**: 如果内存不够，可以启用 `gradient_checkpointing`
4. **混合精度训练**: 已使用 bfloat16

## 监控训练

### 查看 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi dmon
```

### 查看 WandB

在浏览器中访问：https://wandb.ai/

项目名：`math-sft-experiment`

### 查看日志

训练日志会实时输出到终端（只从 rank 0 进程）

## 完整工作流程

### 1. 准备环境

```bash
cd /root/assignment5-alignment
nvidia-smi  # 确认 4 个 GPU 可用
```

### 2. 运行小规模测试

```bash
# 快速测试，确保代码能运行
./cs336_alignment/run_single_experiment_ddp.sh 128
```

### 3. 运行完整实验

```bash
# 运行所有数据集大小的实验
./cs336_alignment/run_all_sft_experiments_ddp.sh
```

### 4. 查看结果

- 检查 WandB 中的曲线
- 查看 `./sft_outputs/` 中保存的模型
- 分析不同数据集大小对准确率的影响

### 5. 调优（如果需要）

如果验证准确率 < 15%，调整超参数：

```bash
# 例如：提高学习率
export LR=1e-4
export EPOCHS=5
./cs336_alignment/run_single_experiment_ddp.sh full
```

## 参考资料

- **Assignment 5 文档**: Task 4.3 SFT Experiment
- **PyTorch DDP**: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- **vLLM**: https://docs.vllm.ai/
- **Transformers**: https://huggingface.co/docs/transformers/
- **WandB**: https://docs.wandb.ai/
