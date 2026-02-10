#!/usr/bin/env python3
"""
Filter MATH training dataset to only include examples that produce correct answers
使用 cs336_alignment 中已有的实现
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from tqdm import tqdm

# Import from cs336_alignment
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Initialize vLLM for inference"""
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )


def format_r1_zero_prompt(problem: str) -> str:
    """Format problem for R1-Zero style evaluation"""
    return (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <thought> and <answer> tags, respectively.\n"
        f"User: {problem}\n"
        "Assistant: <thought>\n"
    )


def filter_correct_examples(
    data_path: str,
    model_path: str,
    output_path: str,
    device: str = "cuda:0",
    batch_size: int = 32,
    max_tokens: int = 1024,
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
):
    """Filter training examples to only include those that produce correct answers"""

    print(f"\nFiltering dataset to include only correct examples...")
    print(f"Input: {data_path}")
    print(f"Output: {output_path}")

    # Load all examples
    all_examples = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

    print(f"Total examples before filtering: {len(all_examples)}")

    # Initialize vLLM for evaluation
    print(f"Initializing vLLM on {device}...")
    llm = init_vllm(
        model_id=model_path,
        device=device,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Evaluate each example in batches
    filtered_examples = []

    for i in tqdm(range(0, len(all_examples), batch_size), desc="Filtering examples"):
        batch = all_examples[i:i + batch_size]

        # Extract problems
        problems = []
        for ex in batch:
            prompt = ex["prompt"]
            # Format: "Question: {problem}\n\nAnswer:"
            problem = prompt.replace("Question: ", "").replace("\n\nAnswer:", "")
            problems.append(problem)

        prompts = [format_r1_zero_prompt(p) for p in problems]
        gold_solutions = [ex["response"] for ex in batch]

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Check correctness using cs336_alignment.drgrpo_grader
        for j, (output, gold, example) in enumerate(zip(outputs, gold_solutions, batch)):
            generated_text = output.outputs[0].text
            score_dict = r1_zero_reward_fn(generated_text, gold)

            # Only keep if answer is correct
            if score_dict.get('answer_reward', 0) == 1:
                filtered_examples.append(example)

    print(f"\nExamples after filtering: {len(filtered_examples)}")
    print(f"Filter rate: {len(filtered_examples) / len(all_examples) * 100:.2f}%")

    # Save filtered dataset
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for ex in filtered_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"Filtered dataset saved to {output_path}")

    return len(filtered_examples)


def main():
    parser = argparse.ArgumentParser(description="Filter MATH dataset to correct examples only")

    parser.add_argument("--model_path", type=str,
                        default="/root/assignment5-alignment/data/a5-alignment/models/Qwen2.5-Math-1.5B",
                        help="Path to the base model")
    parser.add_argument("--input_data", type=str,
                        default="/root/assignment5-alignment/data/MATH/math_train.jsonl",
                        help="Input training data path")
    parser.add_argument("--output_data", type=str,
                        default="/root/assignment5-alignment/data/MATH/math_train_filtered.jsonl",
                        help="Output filtered data path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for vLLM")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for filtering")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Max tokens for generation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="GPU memory utilization for vLLM")

    args = parser.parse_args()

    print("="*60)
    print("Filtering Configuration")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_data}")
    print(f"Output: {args.output_data}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print("="*60 + "\n")

    filtered_size = filter_correct_examples(
        data_path=args.input_data,
        model_path=args.model_path,
        output_path=args.output_data,
        device=args.device,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"\n✓ Filtering complete!")
    print(f"  Filtered dataset size: {filtered_size}")


if __name__ == "__main__":
    main()
