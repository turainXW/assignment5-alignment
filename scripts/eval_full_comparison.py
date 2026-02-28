#!/usr/bin/env python3
"""Evaluate Base(native prompt), SFT, GRPO-150, GRPO-Final. 500 samples, 100 detailed."""

import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, extract_boxed_answer, grade
from cs336_alignment.trl_grpo_data import PROMPT_TEMPLATE

# Native prompt for base Qwen2.5-Math (no R1 format)
BASE_PROMPT = "Question: {question}\n\nAnswer: Let me solve this step by step.\n"


def load_test_data(path: str, max_samples: int):
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
    boxed = extract_boxed_answer(response_field)
    return boxed if boxed else response_field


def grade_base_output(text: str, ground_truth: str) -> bool:
    """Grade base model output: extract \\boxed{} or last expression, compare with ground truth."""
    # Try boxed answer first
    boxed = extract_boxed_answer(text)
    if boxed:
        return grade(boxed, ground_truth)
    # Try <answer> tag (unlikely for base)
    if "<answer>" in text and "</answer>" in text:
        ans = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return grade(ans, ground_truth)
    # Try common patterns
    import re
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'[Tt]he answer is[:\s]*\$?([^$\n.]+)',
        r'=\s*\$?\\?([^$\n,.]+?)\s*$',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            candidate = matches[-1].strip()
            if grade(candidate, ground_truth):
                return True
    return False


def evaluate_model(model_path: str, prompts: list, ground_truths: list,
                   is_base: bool = False, max_tokens: int = 1024):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path} {'(native prompt)' if is_base else '(R1 prompt)'}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
    )

    if is_base:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_tokens,
        )
    else:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_tokens,
            stop=["</answer>"], include_stop_str_in_output=True,
        )

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    answer_correct = 0
    format_correct = 0
    total = len(outputs)

    for i, output in enumerate(outputs):
        text = output.outputs[0].text

        if is_base:
            ans_ok = grade_base_output(text, ground_truths[i])
            fmt_ok = False  # base model has no R1 format
        else:
            scores = r1_zero_reward_fn(text, ground_truths[i])
            fmt_ok = scores.get("format_reward", 0) == 1.0
            ans_ok = scores.get("answer_reward", 0) == 1.0 and fmt_ok

        if fmt_ok:
            format_correct += 1
        if ans_ok:
            answer_correct += 1

        results.append({
            "output": text,
            "format_correct": fmt_ok,
            "answer_correct": ans_ok,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] ans_acc={answer_correct/(i+1):.4f}")

    acc = answer_correct / total
    fmt_acc = format_correct / total
    print(f"  Answer Accuracy: {answer_correct}/{total} = {acc:.2%}")
    if not is_base:
        print(f"  Format Accuracy: {format_correct}/{total} = {fmt_acc:.2%}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return results, answer_correct, format_correct


def main():
    test_data = load_test_data("data/MATH/math_test_r1.jsonl", 500)
    print(f"Loaded {len(test_data)} test examples")

    questions = []
    r1_prompts = []
    base_prompts = []
    ground_truths = []
    for ex in test_data:
        q = extract_question(ex["prompt"])
        questions.append(q)
        r1_prompts.append(PROMPT_TEMPLATE.format(question=q))
        base_prompts.append(BASE_PROMPT.format(question=q))
        ground_truths.append(extract_ground_truth(ex["response"]))

    models = [
        ("Base_Qwen2.5-Math-1.5B", "data/a5-alignment/models/Qwen2.5-Math-1.5B", True, base_prompts),
        ("SFT_R1", "data/sft_r1_format", False, r1_prompts),
        ("GRPO_Step150", "data/trl_grpo_output/checkpoint-150", False, r1_prompts),
        ("GRPO_Final", "data/trl_grpo_output/final_model", False, r1_prompts),
    ]

    models = [(n, p, b, pr) for n, p, b, pr in models if Path(p).exists()]

    all_results = {}
    for name, path, is_base, prm in models:
        results, ans_c, fmt_c = evaluate_model(path, prm, ground_truths, is_base=is_base)
        all_results[name] = {
            "total": len(results),
            "answer_correct": ans_c,
            "format_correct": fmt_c,
            "answer_accuracy": ans_c / len(results),
            "format_accuracy": fmt_c / len(results),
            "is_base": is_base,
            "per_example": results,
        }

    # Build output
    record_n = 100
    output = {"summary": {}, "detailed_examples": []}

    print(f"\n{'='*75}")
    print(f"{'Model':<28} {'Answer Acc':>12} {'Format Acc':>12} {'Correct':>10}")
    print(f"{'='*75}")
    for name in all_results:
        r = all_results[name]
        fmt_str = f"{r['format_accuracy']:>11.2%}" if not r.get("is_base") else "      N/A  "
        output["summary"][name] = {
            "answer_accuracy": r["answer_accuracy"],
            "format_accuracy": r["format_accuracy"],
            "answer_correct": r["answer_correct"],
            "format_correct": r["format_correct"],
            "total": r["total"],
        }
        print(f"{name:<28} {r['answer_accuracy']:>11.2%} {fmt_str} {r['answer_correct']:>5}/{r['total']}")
    print(f"{'='*75}")

    # Detailed examples
    for i in range(min(record_n, len(questions))):
        example = {
            "idx": i + 1,
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "models": {},
        }
        for name in all_results:
            r = all_results[name]["per_example"][i]
            text = r["output"]
            if len(text) > 1500:
                text = text[:700] + "\n...[truncated]...\n" + text[-700:]
            example["models"][name] = {
                "output": text,
                "format_correct": r["format_correct"],
                "answer_correct": r["answer_correct"],
            }
        output["detailed_examples"].append(example)

    # Side-by-side
    print(f"\nSide-by-side (first 20):")
    print(f"{'#':<4} {'Ground Truth':<22}", end="")
    for name in all_results:
        print(f" {name[:10]:>10}", end="")
    print()
    print("-" * (26 + 11 * len(all_results)))
    for i in range(min(20, len(questions))):
        gt = ground_truths[i]
        gt_d = gt[:20] if len(gt) <= 20 else gt[:17] + "..."
        print(f"{i+1:<4} {gt_d:<22}", end="")
        for name in all_results:
            ok = all_results[name]["per_example"][i]["answer_correct"]
            print(f" {'    OK' if ok else '  FAIL':>10}", end="")
        print()

    out_path = "eval_results/full_comparison_500.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
