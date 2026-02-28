#!/usr/bin/env python3
"""Evaluate all models and save 50 detailed examples per model."""

import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.trl_grpo_data import PROMPT_TEMPLATE


def load_test_data(path: str, max_samples: int = 50):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples[:max_samples]


def extract_question(prompt_field: str) -> str:
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
    if "<answer>" in response_field and "</answer>" in response_field:
        return response_field.split("<answer>")[-1].replace("</answer>", "").strip()
    from cs336_alignment.drgrpo_grader import extract_boxed_answer
    boxed = extract_boxed_answer(response_field)
    return boxed if boxed else response_field


def evaluate_model(model_path: str, test_data: list, max_tokens: int = 1024):
    """Evaluate model and return detailed per-example results."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")

    prompts = []
    questions = []
    ground_truths = []
    for ex in test_data:
        q = extract_question(ex["prompt"])
        questions.append(q)
        prompts.append(PROMPT_TEMPLATE.format(question=q))
        ground_truths.append(extract_ground_truth(ex["response"]))

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

    outputs = llm.generate(prompts, sampling_params)

    results = []
    correct = 0
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        scores = r1_zero_reward_fn(text, ground_truths[i])
        is_correct = scores.get("answer_reward", 0) == 1.0 and scores.get("format_reward", 0) == 1.0
        if is_correct:
            correct += 1

        # Truncate very long outputs for readability
        display_text = text if len(text) <= 1500 else text[:750] + "\n...[truncated]...\n" + text[-750:]

        results.append({
            "idx": i + 1,
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "model_output": display_text,
            "format_correct": scores.get("format_reward", 0) == 1.0,
            "answer_correct": is_correct,
        })

    acc = correct / len(outputs)
    print(f"  Accuracy: {correct}/{len(outputs)} = {acc:.2%}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return results, acc


def main():
    test_data = load_test_data("data/MATH/math_test_r1.jsonl", 50)
    print(f"Loaded {len(test_data)} test examples")

    models = [
        ("SFT_R1_Base", "data/sft_r1_format"),
        ("GRPO_Step50", "data/trl_grpo_output/checkpoint-50"),
        ("GRPO_Step100", "data/trl_grpo_output/checkpoint-100"),
        ("GRPO_Step150", "data/trl_grpo_output/checkpoint-150"),
        ("GRPO_Step200", "data/trl_grpo_output/checkpoint-200"),
    ]

    models = [(n, p) for n, p in models if Path(p).exists()]
    all_results = {}

    for name, path in models:
        results, acc = evaluate_model(path, test_data)
        all_results[name] = {"accuracy": acc, "examples": results}

    # Save full JSON
    out_path = "eval_results/grpo_50_examples.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_path}")

    # Print comparison summary
    print(f"\n{'='*70}")
    print(f"{'Model':<18} {'Accuracy':>10} {'Correct':>10}")
    print(f"{'='*70}")
    for name in all_results:
        r = all_results[name]
        n_correct = sum(1 for e in r["examples"] if e["answer_correct"])
        print(f"{name:<18} {r['accuracy']:>9.2%} {n_correct:>5}/50")

    # Print side-by-side example comparison
    print(f"\n{'='*70}")
    print("Side-by-side comparison (first 10 examples)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Ground Truth':<20}", end="")
    for name in all_results:
        print(f" {name:<8}", end="")
    print()
    print("-" * (24 + 9 * len(all_results)))

    for i in range(min(10, 50)):
        gt = list(all_results.values())[0]["examples"][i]["ground_truth"]
        gt_display = gt[:18] if len(gt) <= 18 else gt[:15] + "..."
        print(f"{i+1:<4} {gt_display:<20}", end="")
        for name in all_results:
            ok = all_results[name]["examples"][i]["answer_correct"]
            print(f" {'  OK  ' if ok else ' FAIL ':>8}", end="")
        print()


if __name__ == "__main__":
    main()
