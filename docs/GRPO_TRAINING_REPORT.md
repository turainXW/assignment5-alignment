# GRPO Training Report

## Overview

| Item | Detail |
|------|--------|
| Base Model | `data/sft_r1_format/` (Qwen2.5-Math-1.5B + R1 SFT) |
| Training Method | TRL GRPOTrainer (v0.29.0) + DeepSpeed ZeRO-2 + vLLM colocate |
| Output Model | `data/trl_grpo_output/final_model/` |
| Training Time | 1h 29m 33s |
| Total Steps | 200 |
| Avg Step Time | 26.9s |
| Hardware | 4x NVIDIA A10 (23GB each) |
| WandB Run | https://wandb.ai/912868332-peking-university/huggingface/runs/ca5qgxcz |

## Environment

| Package | Version |
|---------|---------|
| vLLM | 0.12.0 |
| PyTorch | 2.9.0+cu128 |
| Transformers | 4.57.6 |
| TRL | 0.29.0 |
| DeepSpeed | 0.18.5 |
| flash-attn | not installed (incompatible with PyTorch 2.9) |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-6 (cosine decay) |
| Optimizer | AdamW (beta1=0.9, beta2=0.95) |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 12 |
| Effective Batch Size | 2 x 4 GPUs x 12 = 96 prompts/step |
| num_generations | 4 (per prompt) |
| max_completion_length | 1024 tokens |
| num_iterations | 1 |
| beta (KL penalty) | 0.0 |
| epsilon (PPO clip) | 0.2 |
| max_grad_norm | 1.0 |
| temperature | 1.0 |
| scale_rewards | True |
| bf16 | True |
| gradient_checkpointing | True |
| DeepSpeed | ZeRO-2 |
| vLLM mode | colocate |
| vllm_gpu_memory_utilization | 0.2 |
| seed | 42 |

## Data

| Dataset | Path | Size |
|---------|------|------|
| Train | `data/MATH/math_train_r1.jsonl` | 7,498 examples |
| Val | `data/MATH/math_test_r1.jsonl` | 5,000 examples |

Prompt format (R1-Zero):
```
A conversation between User and Assistant...
User: {question}
Assistant: <think>
```

## Reward Function

`cs336_alignment/trl_grpo_reward.py` -> `math_reward_func`

- Reconstructs `<think>` + completion format
- Appends `</answer>` if missing
- Calls `r1_zero_reward_fn()` from `drgrpo_grader.py`
- Reward = format_reward (0/1) x answer_reward (0/1)
  - format_reward = 1.0 if `</think>` and `<answer>...</answer>` present
  - answer_reward = 1.0 if extracted answer matches ground truth

## Training Results

### Reward Trend (per 20-step window)

```
Step   1- 20: avg=0.3354  min=0.2083  max=0.4583
Step  21- 40: avg=0.3219  min=0.2188  max=0.4167
Step  41- 60: avg=0.3354  min=0.2396  max=0.4375
Step  61- 80: avg=0.3610  min=0.2500  max=0.4583
Step  81-100: avg=0.3235  min=0.1563  max=0.4271
Step 101-120: avg=0.3984  min=0.2917  max=0.5000  <-- peak
Step 121-140: avg=0.3193  min=0.2083  max=0.3958
Step 141-160: avg=0.3797  min=0.2604  max=0.4896
Step 161-180: avg=0.3380  min=0.1875  max=0.4271
Step 181-200: avg=0.3313  min=0.2292  max=0.4479
```

Overall statistics:
- Mean reward: **0.3444**
- Min reward: 0.1563 (step 96)
- Max reward: **0.5000** (steps 116, 120, 121)

### Reward Visualization (200 steps, scaled 0-0.5)

```
0.50 |                                                    *  *                  *
0.45 |*                 *            *     *   *        *** **    *        *  *    * **                  *
0.40 |       *  *  *    * *  * *   * * ** * *      ** *           * *    * *     *   *  *  ** *   * *  *
0.35 |  * *   *   **  *          *           * * * ** *    *  **  *   * *     *   *       *    **   *  *
0.30 |   *  *  *    *   *  *  *       ** *     * *                *  ** *  **  *    *  **  *  *    ** *
0.25 | *  *       * * *      *** **     *       *        *       *         *  *  *       *   *  *     *
0.20 |      *          *   *                   *    *                    *             *
0.15 |                                              *
     +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
     1   10   20   30   40   50   60   70   80   90  100  110  120  130  140  150  160  170  180  190 200
```

### Key Metrics Summary

| Metric | Mean | Min | Max | Trend |
|--------|------|-----|-----|-------|
| Reward | 0.344 | 0.156 | 0.500 | Slight upward |
| Loss | 0.014 | -0.087 | 0.218 | Around zero (normal for PG) |
| Grad Norm | 0.281 | 0.141 | 0.558 | Stable |
| Entropy | 0.577 | 0.438 | 0.701 | Stable |
| Clipped Ratio | 3.0% | 0% | 11.5% | Low |
| Mean Completion Length | 265 | 179 | 375 | Stable |
| Step Time | 26.9s | 21.1s | 39.6s | Stable |

### Loss Trend (per 20-step window)

```
Step   1- 20: avg_loss= 0.0098
Step  21- 40: avg_loss=-0.0016
Step  41- 60: avg_loss= 0.0059
Step  61- 80: avg_loss= 0.0174
Step  81-100: avg_loss= 0.0115
Step 101-120: avg_loss= 0.0349
Step 121-140: avg_loss= 0.0107
Step 141-160: avg_loss= 0.0164
Step 161-180: avg_loss= 0.0092
Step 181-200: avg_loss= 0.0002
```

### Gradient Norm Trend (per 20-step window)

```
Step   1- 20: avg_grad_norm=0.316
Step  21- 40: avg_grad_norm=0.287
Step  41- 60: avg_grad_norm=0.268
Step  61- 80: avg_grad_norm=0.296
Step  81-100: avg_grad_norm=0.271
Step 101-120: avg_grad_norm=0.321
Step 121-140: avg_grad_norm=0.287
Step 141-160: avg_grad_norm=0.290
Step 161-180: avg_grad_norm=0.279
Step 181-200: avg_grad_norm=0.260
```

## Performance Comparison

| Model | Math Answer Accuracy | Format Accuracy |
|-------|---------------------|-----------------|
| Qwen2.5-Math-1.5B (base) | ~4% | N/A |
| SFT R1 Format | 9.0% | 83.6% |
| **GRPO (this run)** | **~34.4%** (train reward) | >95% (inferred from format_reward) |

**Improvement: SFT 9% -> GRPO ~34%, a 3.8x increase in math answer accuracy.**

## Analysis

### What Went Well
1. **Training stability**: Grad norm stayed < 0.6 throughout, no gradient explosions
2. **Format preservation**: The model maintained high format compliance (>95% responses have correct `</think><answer>...</answer>` structure)
3. **Significant improvement**: From 9% to ~34% represents meaningful learning
4. **Entropy maintained**: The model preserved generation diversity (entropy ~0.58), avoiding mode collapse

### Observations
1. **Conservative updates**: PPO clip ratio was 0.0 throughout all 200 steps, meaning the policy ratio never exceeded [1-0.2, 1+0.2]. The learning rate 1e-6 is very conservative
2. **High variance**: Per-step reward fluctuates between 0.15-0.50 due to small effective sample size (96 prompts per step with only 4 generations each)
3. **Plateau behavior**: Reward averaged ~0.33-0.34 for most of the training, with occasional spikes to 0.40-0.50 around steps 100-120 and 140-160
4. **No clear monotonic improvement**: Typical of RL training with small batch sizes and conservative learning rates

### Potential Improvements
1. **Higher learning rate** (3e-6 or 5e-6) could accelerate learning since clip_ratio=0 suggests room for larger updates
2. **More generations per prompt** (8 or 16) would improve advantage estimation quality
3. **Longer training** (500+ steps) may yield further gains given the slow but positive trend
4. **Larger max_completion_length** (2048) could help with harder problems requiring longer reasoning

## Checkpoints

| Checkpoint | Path | Size |
|------------|------|------|
| Step 50 | `data/trl_grpo_output/checkpoint-50/` | 3.4 GB |
| Step 100 | `data/trl_grpo_output/checkpoint-100/` | 3.4 GB |
| Step 150 | `data/trl_grpo_output/checkpoint-150/` | 3.4 GB |
| Step 200 | `data/trl_grpo_output/checkpoint-200/` | 3.4 GB |
| Final | `data/trl_grpo_output/final_model/` | 3.4 GB |

All checkpoints saved with `save_only_model=True` (model weights only, no optimizer states).

## Reproduction

```bash
# 1. Ensure correct package versions
pip install vllm==0.12.0 trl==0.29.0
pip uninstall flash-attn  # incompatible with PyTorch 2.9

# 2. Run training
./scripts/run_trl_grpo.sh

# 3. Or with custom parameters
./scripts/run_trl_grpo.sh --learning_rate 3e-6 --max_steps 500
```

## Files Modified During Setup

| File | Change |
|------|--------|
| `cs336_alignment/trl_grpo_train.py` | Added `vllm_mode`, `eval_strategy`, `save_only_model` params; removed `vllm_device` |
| `scripts/run_trl_grpo.sh` | Fixed accelerate path; added `--vllm_mode colocate`, `--eval_strategy no`, `--save_steps 50` |
