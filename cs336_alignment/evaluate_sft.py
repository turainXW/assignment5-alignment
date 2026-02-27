#!/usr/bin/env python3
"""
独立的SFT模型评估脚本 - 使用vLLM进行推理评估
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Monkey-patch for Qwen2Tokenizer compatibility with vLLM
from transformers import Qwen2Tokenizer
if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
    Qwen2Tokenizer.all_special_tokens_extended = property(lambda self: [])

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.trl_grpo_data import PROMPT_TEMPLATE


def format_r1_zero_prompt(problem: str) -> str:
    """Format problem for R1-Zero style evaluation using the same template as training."""
    return PROMPT_TEMPLATE.format(question=problem)


def load_eval_data(data_path: str, max_samples: int = None):
    """Load evaluation data from JSONL file"""
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if max_samples is not None and max_samples < len(examples):
        examples = examples[:max_samples]

    print(f"Loaded {len(examples)} evaluation examples")
    return examples


def evaluate_model(model_path: str, eval_data_path: str,
                   max_samples: int = 500, max_tokens: int = 1024,
                   device: str = "cuda:0", gpu_memory_utilization: float = 0.9):
    """
    Evaluate SFT model using vLLM

    Args:
        model_path: Path to the trained model checkpoint
        eval_data_path: Path to evaluation data (JSONL)
        max_samples: Maximum number of samples to evaluate
        max_tokens: Maximum tokens to generate per example
        device: CUDA device to use
        gpu_memory_utilization: GPU memory utilization for vLLM
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    print(f"{'='*60}\n")

    # Load evaluation data
    eval_examples = load_eval_data(eval_data_path, max_samples)

    # Extract problems and gold solutions
    problems = []
    for ex in eval_examples:
        prompt = ex["prompt"]
        # Remove "Question: " and "\n\nAnswer:" from prompt
        problem = prompt.replace("Question: ", "").replace("\n\nAnswer:", "")
        problems.append(problem)

    prompts = [format_r1_zero_prompt(p) for p in problems]
    gold_solutions = [ex["response"] for ex in eval_examples]

    # Initialize vLLM
    print(f"Initializing vLLM on {device}...")
    llm = LLM(
        model=model_path,
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    print("vLLM initialization complete!\n")

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Generate responses
    print(f"Generating responses for {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params)
    print("Generation complete!\n")

    # Evaluate responses
    print("Evaluating responses...")
    correct = 0
    format_correct = 0
    results = []

    for i, (output, gold) in enumerate(zip(outputs, gold_solutions)):
        generated_text = output.outputs[0].text
        score_dict = r1_zero_reward_fn(generated_text, gold)

        format_reward = score_dict.get('format_reward', 0)
        answer_reward = score_dict.get('answer_reward', 0)

        if format_reward == 1:
            format_correct += 1
        if answer_reward == 1:
            correct += 1

        results.append({
            'problem': problems[i],
            'generated': generated_text,
            'gold': gold,
            'format_reward': format_reward,
            'answer_reward': answer_reward,
        })

        # Print progress every 100 examples
        if (i + 1) % 100 == 0:
            curr_acc = correct / (i + 1)
            curr_format_acc = format_correct / (i + 1)
            print(f"  Progress: {i+1}/{len(outputs)} | "
                  f"Accuracy: {curr_acc:.4f} | "
                  f"Format: {curr_format_acc:.4f}")

    # Print final results
    total = len(outputs)
    accuracy = correct / total if total > 0 else 0
    format_accuracy = format_correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"Answer correct: {correct} ({accuracy:.2%})")
    print(f"Format correct: {format_correct} ({format_accuracy:.2%})")
    print(f"{'='*60}\n")

    return {
        'accuracy': accuracy,
        'format_accuracy': format_accuracy,
        'correct': correct,
        'format_correct': format_correct,
        'total': total,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model with vLLM")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--eval_data_path", type=str,
                        default="/root/assignment5-alignment/data/MATH/math_test.jsonl",
                        help="Path to evaluation data")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum tokens to generate per example")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device to use for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save detailed results (JSON)")

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        eval_data_path=args.eval_data_path,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Save detailed results if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
