"""Main GRPO training script using TRL's GRPOTrainer."""

import argparse
import os

from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from cs336_alignment.trl_grpo_data import load_math_dataset
from cs336_alignment.trl_grpo_reward import math_reward_func


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training with TRL")

    # Model and data
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, default="data/MATH/math_train.jsonl")
    parser.add_argument("--val_data", type=str, default="data/MATH/math_test.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/trl_grpo_output")
    parser.add_argument("--run_name", type=str, default="trl_grpo")

    # GRPO hyperparameters
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # DeepSpeed / memory
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON (e.g. cs336_alignment/ds_zero2_trl.json)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # vLLM
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["server", "colocate"])
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_server_timeout", type=float, default=300.0)

    # Logging / eval
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--log_completions", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load datasets
    train_dataset = load_math_dataset(args.train_data)
    eval_dataset = load_math_dataset(args.val_data)

    # GRPOConfig — pass model as string so TRL handles init (needed for DeepSpeed)
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        num_iterations=args.num_iterations,
        beta=args.beta,
        scale_rewards=True,
        epsilon=args.epsilon,
        max_grad_norm=args.max_grad_norm,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        log_completions=args.log_completions,
        seed=42,
        # Let TRL handle model loading
        model_init_kwargs={
            "torch_dtype": "bfloat16",
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build trainer — pass model as string path for DeepSpeed compatibility
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=math_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
