# DeepSpeed ZeRO-2 模型保存说明

## 问题：只保存rank 0能否保存完整模型？

**答案：可以！** 在DeepSpeed ZeRO-2下，只在rank 0保存模型是**正确且标准的做法**。

## DeepSpeed ZeRO阶段对比

### ZeRO Stage 0/1
- ✅ 模型参数：完整（每个rank都有）
- ⚠️ 优化器状态：分片

### ZeRO Stage 2 (我们使用的)
- ✅ **模型参数：完整（每个rank都有）**
- ⚠️ 优化器状态：分片
- ⚠️ 梯度：分片

### ZeRO Stage 3
- ⚠️ 模型参数：分片
- ⚠️ 优化器状态：分片
- ⚠️ 梯度：分片

## 为什么ZeRO-2可以只保存rank 0？

在ZeRO-2中：

1. **模型参数不分片** - 每个GPU都保存完整的模型参数副本
2. **前向传播** - 使用完整参数
3. **反向传播** - 使用完整参数计算梯度
4. **梯度和优化器状态才分片** - 节省显存

因此，**任意rank都可以保存完整模型**，通常选择rank 0保存。

## 我们的实现

### 1. vLLM使用的临时保存（每步）

```python
def save_ds_model_for_vllm(model_engine, tokenizer, path):
    """
    ZeRO-2 下保存完整模型权重供 vLLM 加载
    """
    rank = dist.get_rank()

    # 只在rank 0上获取state_dict并保存
    if rank == 0:
        # 在ZeRO-2中，model_engine.module包含完整的模型参数
        state_dict = model_engine.module.state_dict()
        model_engine.module.save_pretrained(path, state_dict=state_dict)
        tokenizer.save_pretrained(path)

    # 等待rank 0完成保存
    dist.barrier()
```

**为什么要barrier？**
- 确保rank 0完成保存后，其他rank才继续
- vLLM需要从保存的路径加载完整模型

### 2. 最终模型保存

```python
# main函数中
rank = int(os.environ.get("RANK", 0))
if rank == 0:
    save_path = os.path.join(args.output_dir, "final_model")
    trained_model.save_pretrained(save_path)  # 保存完整模型
    tokenizer.save_pretrained(save_path)

# 等待所有rank完成
if dist.is_initialized():
    dist.barrier()
```

## 验证保存的模型

使用我们提供的验证脚本：

```bash
python verify_saved_model.py results/grpo_output_*/final_model/
```

验证脚本会检查：
1. ✅ 所有必需文件是否存在
2. ✅ 模型是否可以加载
3. ✅ 参数是否包含NaN/Inf
4. ✅ 推理是否正常工作

## 如果使用ZeRO-3怎么办？

如果使用ZeRO-3（模型参数也分片），需要不同的保存方式：

```python
# ZeRO-3需要收集分片参数
if rank == 0:
    # 方法1：使用DeepSpeed的save_checkpoint
    model_engine.save_checkpoint(save_dir)

    # 方法2：使用save_16bit_model
    from deepspeed.utils.zero_to_fp32 import save_16bit_model
    save_16bit_model(model_engine, save_path)
```

**但我们使用ZeRO-2，不需要这些！**

## 最佳实践总结

✅ **推荐做法（我们的实现）：**
```python
if rank == 0:
    state_dict = model_engine.module.state_dict()
    model.save_pretrained(path, state_dict=state_dict)
dist.barrier()
```

⚠️ **不推荐（浪费内存）：**
```python
# 所有rank都获取state_dict（浪费！）
state_dict = model_engine.module.state_dict()
if rank == 0:
    model.save_pretrained(path, state_dict=state_dict)
```

❌ **错误做法：**
```python
# 没有barrier，其他rank可能在rank 0保存完成前就继续执行
if rank == 0:
    model.save_pretrained(path)
# 缺少 dist.barrier()
```

## 常见问题

### Q1: 为什么不用`model_engine.save_checkpoint()`？

A: `save_checkpoint()`会保存：
- 模型参数
- **优化器状态（分片的）**
- 学习率调度器
- 训练步数等

我们只需要模型参数给vLLM用，不需要优化器状态。使用`save_pretrained()`更简洁。

### Q2: 保存的模型能在单GPU上加载吗？

A: **可以！** 保存的是标准HuggingFace格式，可以在任何设备上加载：

```python
from transformers import AutoModelForCausalLM

# 可以在单GPU或多GPU上加载
model = AutoModelForCausalLM.from_pretrained("path/to/saved/model")
```

### Q3: 如何确认模型保存成功？

A: 使用验证脚本：

```bash
python verify_saved_model.py results/grpo_output_*/final_model/
```

或手动检查：
```bash
ls -lh results/grpo_output_*/final_model/
# 应该看到：
# - pytorch_model.bin (约3GB，对于1.5B模型)
# - config.json
# - tokenizer files
```

### Q4: 训练中断了，能恢复吗？

A: 我们保存的checkpoint可以用于恢复：

```bash
# 使用checkpoint继续训练
python cs336_alignment/grpo.py \
    --model_path results/grpo_output_*/checkpoint_step_20/ \
    --n_grpo_steps 200 \
    # ... 其他参数
```

## 参考资源

- [DeepSpeed ZeRO文档](https://www.deepspeed.ai/tutorials/zero/)
- [HuggingFace模型保存](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained)
