#!/usr/bin/env python3
"""
SFT Training with DeepSpeed - Fixed vLLM timeout issue
延迟 vLLM 初始化，只在第一次评估时才初始化
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import wandb
import deepspeed

# Import from cs336_alignment
from cs336_alignment.sft_tokenize import tokenize_prompt_and_output
from cs336_alignment.task import get_response_log_probs


class MATHDataset(Dataset):
    """Dataset for MATH SFT training"""

    def __init__(self, data_path: str, tokenizer, max_samples: Optional[int] = None, seed: int = 42):
        self.tokenizer = tokenizer
        self.examples = []

        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        if max_samples is not None and max_samples < len(self.examples):
            random.seed(seed)
            random.shuffle(self.examples)
            self.examples = self.examples[:max_samples]

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer):
    """Collate function for dataloader"""
    prompts = [ex["prompt"] for ex in batch]
    responses = [ex["response"] for ex in batch]

    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)

    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
        "response_mask": tokenized["response_mask"].bool(),
    }


def init_vllm_lazy(model_path: str, device: str, seed: int, tokenizer, gpu_memory_utilization: float = 0.85):
    """延迟初始化 vLLM - 只在第一次调用时初始化"""
    from vllm import LLM, SamplingParams
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    from unittest.mock import patch

    print(f"Initializing vLLM on {device}...")
    vllm_set_random_seed(seed)

    # 修复tokenizer缺失的属性
    if not hasattr(tokenizer, 'all_special_tokens_extended'):
        # 为tokenizer添加缺失的属性
        tokenizer.all_special_tokens_extended = tokenizer.all_special_tokens

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        llm = LLM(
            model=model_path,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    print("vLLM initialization complete!")
    return llm


def load_policy_into_vllm_instance(policy, llm):
    """Load policy weights into vLLM instance"""
    # Handle DeepSpeed wrapped model
    if hasattr(policy, 'module'):
        state_dict = policy.module.state_dict()
    else:
        state_dict = policy.state_dict()

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def format_r1_zero_prompt(problem: str) -> str:
    """Format problem for R1-Zero style evaluation"""
    return (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <thought> and <answer> tags, respectively.\n"
        f"User: {problem}\n"
        "Assistant: <thought>\n"
    )


def evaluate_model(llm, eval_examples, max_tokens=1024):
    """Evaluate model on MATH validation set"""
    from vllm import SamplingParams

    problems = []
    for ex in eval_examples:
        prompt = ex["prompt"]
        problem = prompt.replace("Question: ", "").replace("\n\nAnswer:", "")
        problems.append(problem)

    prompts = [format_r1_zero_prompt(p) for p in problems]
    gold_solutions = [ex["response"] for ex in eval_examples]

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    format_correct = 0

    for output, gold in zip(outputs, gold_solutions):
        generated_text = output.outputs[0].text
        score_dict = r1_zero_reward_fn(generated_text, gold)

        if score_dict.get('format_reward', 0) == 1:
            format_correct += 1
        if score_dict.get('answer_reward', 0) == 1:
            correct += 1

    total = len(outputs)
    accuracy = correct / total if total > 0 else 0
    format_accuracy = format_correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "correct": correct,
        "format_correct": format_correct,
        "total": total,
    }


def train(args):
    """Main training function with DeepSpeed"""

    # DeepSpeed initialization
    deepspeed.init_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Set random seeds
    random.seed(args.seed + local_rank)
    torch.manual_seed(args.seed + local_rank)

    # Determine dataset size
    if args.dataset_size == "full":
        dataset_size = None
        size_str = "full"
    else:
        dataset_size = int(args.dataset_size)
        size_str = str(dataset_size)

    # Define run name (all ranks need this for checkpoint saving)
    run_name = f"sft_ds_size_{size_str}_lr{args.learning_rate}_bs{args.batch_size}x{args.gradient_accumulation_steps}"
    if args.wandb_run_name:
        run_name = args.wandb_run_name

    # Initialize wandb (只在 rank 0)
    if local_rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # Load tokenizer
    if local_rank == 0:
        print(f"\nLoading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if local_rank == 0:
        print(f"Loading datasets...")
    train_dataset = MATHDataset(
        args.train_data_path,
        tokenizer,
        max_samples=dataset_size,
        seed=args.seed,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=0,
    )

    # Load policy model
    if local_rank == 0:
        print(f"Loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 使用FlashAttention2加速
    )

    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    print("✓ Gradient checkpointing enabled")

    # DeepSpeed config
    ds_config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps * torch.distributed.get_world_size(),
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.max_grad_norm,
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "none"},
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": len(train_loader) * args.num_epochs // args.gradient_accumulation_steps,
                "warmup_num_steps": int(len(train_loader) * args.num_epochs // args.gradient_accumulation_steps * args.warmup_ratio)
            }
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    if local_rank == 0:
        total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
        print(f"Total training steps: {total_steps}")
        print(f"\nStarting training...")
        print("Note: In-training evaluation disabled. Use evaluate_sft.py after training.\n")

    # Training loop
    global_step = 0
    train_step_logged = 0

    model_engine.train()

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)

        if local_rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print('='*60)

        epoch_loss = 0

        if local_rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        else:
            progress_bar = train_loader

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)
            response_mask = batch["response_mask"].to(model_engine.local_rank)

            # Get log probabilities
            log_probs_dict = get_response_log_probs(
                model=model_engine,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )

            policy_log_probs = log_probs_dict["log_probs"]

            # Compute loss manually (don't use sft_microbatch_train_step to avoid double backward)
            normalize_constant = response_mask.float().sum(dim=1)
            masked_log_probs = (-policy_log_probs).masked_fill(~response_mask, 0)
            raw_loss = masked_log_probs.sum(dim=1) / normalize_constant
            loss = raw_loss.mean() / args.gradient_accumulation_steps

            metadata = {
                "loss": raw_loss.detach()
            }

            # DeepSpeed backward and step
            model_engine.backward(loss)
            model_engine.step()

            epoch_loss += loss.item()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                if local_rank == 0:
                    avg_batch_loss = metadata["loss"].mean().item()
                    wandb.log({
                        "train/loss": avg_batch_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train_step": train_step_logged,
                    })

                    progress_bar.set_postfix({
                        "loss": f"{avg_batch_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                train_step_logged += 1

            # Periodic evaluation removed - vLLM initialization too slow
            # Use separate evaluate_sft.py script after training completes

            global_step += 1

        if local_rank == 0:
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"\nEpoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    # Save final model as complete transformers format (vLLM compatible)
    if local_rank == 0:
        print("\n" + "="*60)
        print("Saving model checkpoint...")
        print("="*60)

    # Gather full model parameters on rank 0
    # For ZeRO stage 2, we need to gather optimizer-partitioned parameters
    output_dir = Path(args.output_dir) / run_name
    if local_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Use DeepSpeed's save_16bit_model to save complete model on rank 0
    # This automatically gathers all ZeRO-sharded parameters
    model_engine.save_16bit_model(output_dir, save_filename="pytorch_model.bin")

    # Only rank 0 saves tokenizer and config
    if local_rank == 0:
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)

        # Save config
        model_engine.module.config.save_pretrained(output_dir)

        print(f"\nModel saved to {output_dir}")
        print(f"Model format: transformers (vLLM compatible)")

        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"To evaluate the model, run:")
        print(f"  python cs336_alignment/evaluate_sft.py --model_path {output_dir}")
        print("="*60)

        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="SFT Training with DeepSpeed")

    parser.add_argument("--model_path", type=str,
                        default="/root/assignment5-alignment/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str,
                        default="/root/assignment5-alignment/data/MATH/math_train.jsonl")
    parser.add_argument("--test_data_path", type=str,
                        default="/root/assignment5-alignment/data/MATH/math_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./sft_outputs")

    parser.add_argument("--dataset_size", type=str, default="full")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=1024)

    parser.add_argument("--vllm_device", type=str, default="cuda:3")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)

    parser.add_argument("--wandb_project", type=str, default="math-sft-experiment")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
