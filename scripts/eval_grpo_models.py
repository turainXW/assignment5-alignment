#!/usr/bin/env python3
"""Evaluate SFT base model and all GRPO checkpoints on MATH test set."""

import argparse
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.trl_grpo_data import PROMPT_TEMPLATE


def load_test_data(path: str, max_samples: int = None):
    """Load R1-format JSONL test data."""
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
    if max_samples and max_samples < len(examples):
        examples = examples[:max_samples]
    return examples


def extract_question(prompt_field: str) -> str:
    """Extract question from prompt."""
    if "User:" in prompt_field:
        parts = prompt_field.split("User:")
        q = parts[-1].strip()
        if q.endswith("Assistant: <think>"):
            q = q[: -len("Assistant: <think>")].strip()
        return q
    q = prompt_field.strip()
    if q.startswith("Question: "):
        q = q[len("Question: "):]
    if q.endswith("\n\nAnswer:"):
        q = q[: -len("\n\nAnswer:")]
    return q.strip()


def extract_ground_truth(response_field: str) -> str:
    """Extract answer from <answer> tags or \\boxed{}."""
    if "<answer>" in response_field and "</answer>" in response_field:
        return response_field.split("<answer>")[-1].replace("</answer>", "").strip()
    from cs336_alignment.drgrpo_grader import extract_boxed_answer
    boxed = extract_boxed_answer(response_field)
    return boxed if boxed else response_field


def evaluate_single_model(model_path: str, test_data: list, max_tokens: int = 1024):
    """Evaluate a single model on test data. Returns accuracy metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")

    # Build prompts
    prompts = []
    ground_truths = []
    for ex in test_data:
        question = extract_question(ex["prompt"])
        prompt = PROMPT_TEMPLATE.format(question=question)
        prompts.append(prompt)
        ground_truths.append(extract_ground_truth(ex["response"]))

    # Init vLLM
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Generate
    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Score
    answer_correct = 0
    format_correct = 0
    total = len(outputs)

    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        scores = r1_zero_reward_fn(text, ground_truths[i])
        if scores.get("format_reward", 0) == 1.0:
            format_correct += 1
        if scores.get("answer_reward", 0) == 1.0 and scores.get("format_reward", 0) == 1.0:
            answer_correct += 1

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] answer_acc={answer_correct/(i+1):.4f} format_acc={format_correct/(i+1):.4f}")

    answer_acc = answer_correct / total
    format_acc = format_correct / total

    print(f"  Answer Accuracy: {answer_correct}/{total} = {answer_acc:.4f} ({answer_acc:.2%})")
    print(f"  Format Accuracy: {format_correct}/{total} = {format_acc:.4f} ({format_acc:.2%})")

    # Cleanup GPU memory
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_path": model_path,
        "total": total,
        "answer_correct": answer_correct,
        "format_correct": format_correct,
        "answer_accuracy": answer_acc,
        "format_accuracy": format_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="data/MATH/math_test_r1.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--output", type=str, default="eval_results/grpo_eval_results.json")
    args = parser.parse_args()

    # Models to evaluate
    models = [
        ("SFT R1 Base", "data/sft_r1_format"),
        ("GRPO Step 50", "data/trl_grpo_output/checkpoint-50"),
        ("GRPO Step 100", "data/trl_grpo_output/checkpoint-100"),
        ("GRPO Step 150", "data/trl_grpo_output/checkpoint-150"),
        ("GRPO Step 200", "data/trl_grpo_output/checkpoint-200"),
    ]

    # Filter existing models
    models = [(name, path) for name, path in models if Path(path).exists()]
    print(f"Found {len(models)} models to evaluate:")
    for name, path in models:
        print(f"  {name}: {path}")

    # Load test data
    test_data = load_test_data(args.test_data, args.max_samples)
    print(f"\nLoaded {len(test_data)} test examples from {args.test_data}")

    # Evaluate each model
    all_results = []
    for name, path in models:
        result = evaluate_single_model(path, test_data, args.max_tokens)
        result["name"] = name
        all_results.append(result)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Model':<20} {'Answer Acc':>12} {'Format Acc':>12} {'Correct':>10}")
    print(f"{'='*70}")
    for r in all_results:
        print(f"{r['name']:<20} {r['answer_accuracy']:>11.2%} {r['format_accuracy']:>11.2%} {r['answer_correct']:>5}/{r['total']}")
    print(f"{'='*70}")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
