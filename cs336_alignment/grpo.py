import torch
from typing import Callable, Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from cs336_alignment.sft_tokenize import tokenize_prompt_and_output

import deepspeed
import torch.distributed as dist
import os
import pickle
import argparse

def compute_group_normalized_rewards(
        reward_fn:Callable[[str,str],dict[str,float]],
        rollout_responses:list[str],
        repeated_ground_truths:list[str],
        group_size:int,
        advantage_eps:float,
        normalize_by_std:bool=True,
)->tuple[torch.Tensor,torch.Tensor,dict[str,float]]:
    
    rewards=[]
    for response,truth in zip(rollout_responses,repeated_ground_truths):
        reward=reward_fn(response,truth)["reward"]
        rewards.append(reward)
    rewards_tensor=torch.tensor(rewards,dtype=torch.float32)
    
    rewards_group=rewards_tensor.reshape(-1,group_size)
    rewards_mean=rewards_group.mean(dim=1,keepdim=True)
    rewards_std=rewards_group.std(dim=1,keepdim=True)
    rewards_normalized=rewards_group-rewards_mean
    if normalize_by_std:
        rewards_normalized/=(rewards_std+advantage_eps)
    
    rewards_normalized_flat=rewards_normalized.flatten()

    reward_stats={ 
                'mean_reward': rewards_mean.mean().item(),
                'std_reward': rewards_std.mean().item(),
            
    }
    return rewards_normalized_flat, rewards_tensor, reward_stats

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages:torch.Tensor,
        policy_log_probs:torch.Tensor,
)->torch.Tensor:
    loss=-(raw_rewards_or_advantages*policy_log_probs)
    return loss

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratios=torch.exp(policy_log_probs-old_log_probs)
    clipped_ratios=torch.clamp(ratios,1-cliprange,1+cliprange)
    loss=-torch.minimum(advantages*clipped_ratios,advantages*ratios)
    stats={
        'mean_ratio': ratios.mean().item(),
        'mean_clipped_ratio': clipped_ratios.mean().item(),
    }
    return loss, stats


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type=="no_baseline":
        loss=compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        stats={}
    elif loss_type=="reinforce_with_baseline":
        loss=compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        stats={}
    elif loss_type=="grpo_clip":
        loss, stats=compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    return loss, stats
    
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    masked_tensor=tensor*mask
    sum_mask=mask.sum(dim=dim)
    mean=masked_tensor.sum(dim=dim)/sum_mask.clamp(min=1e-8)
    zero_mask=(sum_mask==0)
    mean=mean.masked_fill(zero_mask,float('nan'))
    return mean

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss,stats=compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    masked_mean_loss=masked_mean(per_token_loss,response_mask,dim=1)
    loss=masked_mean_loss.mean()
    microbatch_loss=loss/gradient_accumulation_steps
    microbatch_loss.backward()
    return microbatch_loss, stats


def get_log_probs_from_model(
    model,
    tokenizer,
    prompts:list[str],
    responses:list[str],
    device,
):
    from cs336_alignment.sft_tokenize import tokenize_prompt_and_output
    # Process one sample at a time to avoid OOM with 151K vocab
    all_log_probs = []
    all_masks = []
    max_len = 0

    for i in range(len(prompts)):
        tokenized = tokenize_prompt_and_output([prompts[i]], [responses[i]], tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        response_mask = tokenized["response_mask"].to(device)

        logits = model(input_ids=input_ids).logits  # Don't pass labels
        log_probs = torch.log_softmax(logits, dim=-1)
        safe_labels = labels.clamp(min=0)
        lp_at_labels = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

        all_log_probs.append(lp_at_labels.cpu())
        all_masks.append(response_mask.cpu())
        max_len = max(max_len, lp_at_labels.shape[-1])

        del logits, log_probs
        torch.cuda.empty_cache()

    # Pad to same length and stack
    padded_lp, padded_m = [], []
    for lp, m in zip(all_log_probs, all_masks):
        pad = max_len - lp.shape[-1]
        if pad > 0:
            padded_lp.append(torch.cat([lp, torch.zeros(1, pad)], dim=-1))
            padded_m.append(torch.cat([m, torch.zeros(1, pad)], dim=-1))
        else:
            padded_lp.append(lp)
            padded_m.append(m)

    return torch.cat(padded_lp, dim=0).to(device), torch.cat(padded_m, dim=0).to(device)

def format_r1_zero_prompt(
    prompt:str,
) -> str:
    return (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind "
        "and then provides the user with the answer. The reasoning process "
        "and answer are enclosed within <think> </think>and"
        "  <answer> </answer> tags, respectively.\n\n"
        f"User: {prompt}\n\n"
        f"Assistant: <think>\n"
    )


def broadcast_object(obj, src=0):
    """将 Python 对象从 src rank 广播到所有 rank"""
    rank = dist.get_rank()
    if rank == src:
        data = pickle.dumps(obj)
        size = torch.tensor(len(data), dtype=torch.long, device="cuda")
    else:
        size = torch.tensor(0, dtype=torch.long, device="cuda")
    dist.broadcast(size, src=src)
    if rank == src:
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    else:
        tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda")
    dist.broadcast(tensor, src=src)
    if rank != src:
        obj = pickle.loads(tensor.cpu().numpy().tobytes())
    return obj


def save_ds_model_for_vllm(model_engine, tokenizer, path):
    """ZeRO-2 下保存完整模型权重供 vLLM 加载"""
    state_dict = model_engine.module.state_dict()
    if dist.get_rank() == 0:
        model_engine.module.save_pretrained(path, state_dict=state_dict)
        tokenizer.save_pretrained(path)
    dist.barrier()


def grpo_train_loop(
    model_path: str,
    train_dataset_path: str,
    val_dataset_path: str,
    n_epochs: int,
    learning_rate: float,
    advantage_eps: float,
    rollout_batch_size: int,
    group_size: int,
    sampling_temperature: float,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    ds_config_path: str = "ds_config_zero2.json",
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    cliprange: float = 0.2,
):
    # ========== 初始化分布式 ==========
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = (rank == 0)

    # ========== 加载模型 ==========
    policy = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========== DeepSpeed 配置 ==========
    with open(ds_config_path, "r") as f:
        ds_config = json.load(f)

    # Compute batch params from rollout_batch_size
    micro_batch_per_gpu = 1
    grad_accum = rollout_batch_size // (micro_batch_per_gpu * world_size)
    effective_batch = micro_batch_per_gpu * grad_accum * world_size
    ds_config["train_batch_size"] = effective_batch
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch_per_gpu
    ds_config["gradient_accumulation_steps"] = grad_accum
    # Remove scheduler and optimizer from config since we pass our own optimizer
    ds_config.pop("scheduler", None)
    ds_config.pop("optimizer", None)

    # ========== DeepSpeed 初始化 ==========
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95),
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=policy, optimizer=optimizer, config=ds_config,
    )

    # ========== 加载数据 ==========
    with open(train_dataset_path, "r") as f:
        train_dataset = [json.loads(line) for line in f]
    with open(val_dataset_path, "r") as f:
        val_dataset = [json.loads(line) for line in f]

    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout = rollout_batch_size // group_size
    n_microbatches = rollout_batch_size // micro_batch_per_gpu

    log = {"steps": [], "losses": [], "grad_norms": [], "train_rewards": [], "val_rewards": []}
    data_idx = 0
    tmp_path = "/tmp/grpo_policy_ds"
    sampling_params = SamplingParams(
        temperature=sampling_temperature, max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens, stop=["</answer>"],
    )

    def _extract_question(prompt_field):
        return prompt_field.replace("Question: ", "").replace("\n\nAnswer:", "").strip()

    def _extract_gt(response_field):
        boxed = extract_boxed_answer(response_field)
        return boxed if boxed else response_field

    def _make_prompt(question):
        return (
            "A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind "
            "and then provides the user with the answer. The reasoning process "
            "and answer are enclosed within <think> </think>and "
            "<answer> </answer> tags, respectively.\n\n"
            f"User: {question}\n\nAssistant: <think>\n"
        )

    # ========== 主循环 ==========
    for step in range(n_epochs):
        # ------ 构造 prompts ------
        prompts, ground_truths = [], []
        for i in range(n_prompts_per_rollout):
            example = train_dataset[data_idx % len(train_dataset)]
            data_idx += 1
            question = _extract_question(example["prompt"])
            gt = _extract_gt(example["response"])
            p = _make_prompt(question)
            for _ in range(group_size):
                prompts.append(p)
                ground_truths.append(gt)

        # ------ vLLM Rollout（仅 rank 0 生成，广播给所有 rank）------
        save_ds_model_for_vllm(model_engine, tokenizer, tmp_path)

        responses = None
        if is_main:
            llm = LLM(model=tmp_path, gpu_memory_utilization=0.5,
                       dtype="bfloat16", enforce_eager=True, max_model_len=2048)
            vllm_outputs = llm.generate(prompts, sampling_params)
            responses = []
            for output in vllm_outputs:
                text = output.outputs[0].text
                if not text.endswith("</answer>"):
                    text += "</answer>"
                # Prepend <think>\n since prompt ends with it
                responses.append("<think>\n" + text)
            del llm
            torch.cuda.empty_cache()

        responses = broadcast_object(responses, src=0)

        # ------ 计算 Rewards ------
        raw_rewards = []
        for resp, gt in zip(responses, ground_truths):
            raw_rewards.append(r1_zero_reward_fn(resp, gt)["reward"])
        raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
        avg_train_reward = raw_rewards_tensor.mean().item()

        if is_main:
            print(f"Step {step}: train_reward={avg_train_reward:.3f}")

        # ------ 计算 Advantages ------
        if loss_type == "no_baseline":
            rewards_for_loss = raw_rewards_tensor.unsqueeze(1)
            advantages_tensor = None
        else:
            advantages_flat, _, _ = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn, rollout_responses=responses,
                repeated_ground_truths=ground_truths, group_size=group_size,
                advantage_eps=advantage_eps, normalize_by_std=True,
            )
            advantages_tensor = advantages_flat.to(device).unsqueeze(1)
            rewards_for_loss = None

        # ------ 计算 Old Log Probs ------
        model_engine.eval()
        with torch.no_grad():
            old_log_probs, response_mask = get_log_probs_from_model(
                model_engine.module, tokenizer, prompts, responses, device
            )
            old_log_probs = old_log_probs.detach()
        model_engine.train()

        # ------ 训练（DeepSpeed 自动处理梯度累积和通信）------
        for epoch in range(epochs_per_rollout_batch):
            total_loss = 0.0

            # 将数据按每个 GPU 的 micro_batch 分块
            indices = list(range(len(prompts)))
            # 每个 rank 处理不同的 micro batch（数据并行）
            for mb_start in range(0, len(prompts), micro_batch_per_gpu):
                mb_end = min(mb_start + micro_batch_per_gpu, len(prompts))

                mb_prompts = prompts[mb_start:mb_end]
                mb_responses = responses[mb_start:mb_end]
                mb_adv = advantages_tensor[mb_start:mb_end] if advantages_tensor is not None else None
                mb_raw = rewards_for_loss[mb_start:mb_end] if rewards_for_loss is not None else None

                # Forward
                tokenized = tokenize_prompt_and_output(mb_prompts, mb_responses, tokenizer)
                input_ids = tokenized["input_ids"].to(device)
                labels = tokenized["labels"].to(device)
                mb_mask = tokenized["response_mask"].to(device)

                logits = model_engine(input_ids=input_ids).logits
                lp = torch.log_softmax(logits, dim=-1)
                safe_labels = labels.clamp(min=0)
                mb_log_probs = lp.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

                # Trim old_log_probs to match current tokenization length
                cur_len = mb_log_probs.shape[-1]
                mb_old = old_log_probs[mb_start:mb_end, :cur_len] if loss_type == "grpo_clip" else None

                # 计算 loss
                per_token_loss, stats = compute_policy_gradient_loss(
                    mb_log_probs, loss_type,
                    raw_rewards=mb_raw, advantages=mb_adv,
                    old_log_probs=mb_old,
                    cliprange=cliprange if loss_type == "grpo_clip" else None,
                )
                masked_loss = masked_mean(per_token_loss, mb_mask, dim=1)
                loss = masked_loss.mean()

                # DeepSpeed backward + step（自动处理梯度累积）
                model_engine.backward(loss)
                model_engine.step()

                total_loss += loss.item()

        # ------ 日志 ------
        log["steps"].append(step)
        log["losses"].append(total_loss / max(n_microbatches, 1))
        log["train_rewards"].append(avg_train_reward)

        # ------ 验证（仅 rank 0）------
        if step % 5 == 0:
            save_ds_model_for_vllm(model_engine, tokenizer, tmp_path)
            if is_main:
                val_llm = LLM(model=tmp_path, gpu_memory_utilization=0.5,
                              dtype="bfloat16", enforce_eager=True, max_model_len=2048)
                n_val = min(1024, len(val_dataset))
                val_prompts = [_make_prompt(_extract_question(ex["prompt"]))
                               for ex in val_dataset[:n_val]]
                val_gts = [_extract_gt(ex["response"]) for ex in val_dataset[:n_val]]
                val_outputs = val_llm.generate(val_prompts, sampling_params)
                correct = sum(
                    r1_zero_reward_fn(
                        "<think>\n" + vo.outputs[0].text + ("" if vo.outputs[0].text.endswith("</answer>") else "</answer>"),
                        vg
                    )["reward"]
                    for vo, vg in zip(val_outputs, val_gts)
                )
                val_reward = correct / n_val
                log["val_rewards"].append(val_reward)
                print(f"  val_reward={val_reward:.3f}")
                del val_llm
                torch.cuda.empty_cache()
            dist.barrier()

    return model_engine.module, log

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training with DeepSpeed ZeRO-2")

    # 模型和数据
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练模型路径")
    parser.add_argument("--train_dataset_path", type=str, required=True,
                        help="训练数据 jsonl 路径")
    parser.add_argument("--val_dataset_path", type=str, required=True,
                        help="验证数据 jsonl 路径")
    parser.add_argument("--output_dir", type=str, default="./grpo_output",
                        help="模型和日志保存路径")

    # 训练超参
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline",
                        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"])

    # GRPO 采样
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--sampling_temperature", type=float, default=0.7)
    parser.add_argument("--sampling_min_tokens", type=int, default=1)
    parser.add_argument("--sampling_max_tokens", type=int, default=512)

    # DeepSpeed
    parser.add_argument("--ds_config_path", type=str, default="cs336_alignment/ds_config.json")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="DeepSpeed 自动传入，不需要手动设置")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 创建输出目录
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("=" * 60)
        print("GRPO Training Config:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")
        print("=" * 60)

    trained_model, log = grpo_train_loop(
        model_path=args.model_path,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        ds_config_path=args.ds_config_path,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        loss_type=args.loss_type,
        cliprange=args.cliprange,
    )

    # 保存最终模型和日志（仅 rank 0）
    if int(os.environ.get("RANK", 0)) == 0:
        save_path = os.path.join(args.output_dir, "final_model")
        trained_model.save_pretrained(save_path)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")

        log_path = os.path.join(args.output_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"训练日志已保存到: {log_path}")

        # 打印最终统计
        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"  最终 train_reward: {log['train_rewards'][-1]:.4f}")
        if log['val_rewards']:
            print(f"  最终 val_reward:   {log['val_rewards'][-1]:.4f}")
        print(f"  最终 loss:         {log['losses'][-1]:.6f}")
        print("=" * 60)


if __name__ == "__main__":
    main()