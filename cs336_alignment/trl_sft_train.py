#!/usr/bin/env python3
"""R1-format SFT training using TRL SFTTrainer with DeepSpeed."""

import argparse
import json

from datasets import Dataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer


def load_r1_dataset(path: str) -> Dataset:
    """Load R1-format JSONL and return HF Dataset with 'text' column."""
    records = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            # Concatenate prompt + response into single text field
            records.append({"text": ex["prompt"] + ex["response"]})
    return Dataset.from_list(records)


def parse_args():
    parser = argparse.ArgumentParser(description="R1-format SFT with TRL")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, default="data/MATH/math_train_r1.jsonl")
    parser.add_argument("--val_data", type=str, default="data/MATH/math_test_r1.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/sft_r1_format")
    parser.add_argument("--run_name", type=str, default="sft_r1_format")

    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="wandb")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    train_dataset = load_r1_dataset(args.train_data)
    eval_dataset = load_r1_dataset(args.val_data)
    print(f"Train: {len(train_dataset)} examples, Eval: {len(eval_dataset)} examples")

    # Data collator: only compute loss on completion (after "Assistant: <think>")
    response_template = "Assistant: <think>"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        deepspeed=args.deepspeed,
        dataset_text_field="text",
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=42,
        model_init_kwargs={
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        },
    )

    # Build trainer
    trainer = SFTTrainer(
        model=args.model_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
