#!/usr/bin/env python3
"""
Evaluate base model and SFT model on:
  - MMLU (multiple choice accuracy)
  - GSM8K (math reasoning accuracy)
  - AlpacaEval (generate outputs for later winrate evaluation)
  - SimpleSafetyTests (generate outputs for safety evaluation)

Usage:
  # Evaluate base model
  python scripts/evaluate_benchmarks.py --model_path data/models/Qwen2.5-3B --model_name base --prompt_style zero_shot

  # Evaluate SFT model
  python scripts/evaluate_benchmarks.py --model_path data/verl_sft_output/hf_model --model_name sft --prompt_style alpaca
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

# -----------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------

ZERO_SHOT_SYSTEM_PROMPT = open(
    Path(__file__).parent.parent / "cs336_alignment/prompts/zero_shot_system_prompt.prompt"
).read()

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

MMLU_QUESTION_TEMPLATE = (
    "Answer the following multiple choice question about {subject}. "
    "Respond with a single sentence of the form \"The correct answer is _\", "
    "filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n\n"
    "Question: {question}\n"
    "A. {A}\nB. {B}\nC. {C}\nD. {D}\n\nAnswer:"
)


def format_prompt(instruction: str, prompt_style: str) -> str:
    if prompt_style == "zero_shot":
        return ZERO_SHOT_SYSTEM_PROMPT.format(instruction=instruction)
    elif prompt_style == "alpaca":
        return ALPACA_TEMPLATE.format(instruction=instruction)
    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}")


# -----------------------------------------------------------------------
# Parse helpers
# -----------------------------------------------------------------------

def parse_mmlu_response(text: str):
    """Extract A/B/C/D from model response. Returns None if unparseable."""
    # Look for "The correct answer is X" pattern
    m = re.search(r"correct answer is\s+([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: look for standalone letter at start or after newline
    m = re.search(r"(?:^|\n)\s*([A-D])[.\):\s]", text)
    if m:
        return m.group(1).upper()
    # Last resort: first single capital letter A-D
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()
    return None


def parse_gsm8k_response(text: str):
    """Extract the final number from model response. Returns None if unparseable."""
    # Find all numbers (int or float, possibly with commas)
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if not numbers:
        return None
    # Take the last number, remove commas
    last = numbers[-1].replace(",", "")
    try:
        return float(last)
    except ValueError:
        return None


def parse_gsm8k_gold(answer: str):
    """Extract numeric answer from GSM8K gold answer (after ####)."""
    m = re.search(r"####\s*([\d,\-\.]+)", answer)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    # fallback: last number
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


# -----------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------

def load_mmlu(data_dir: str = "data/mmlu/test"):
    """Load all MMLU test CSV files."""
    examples = []
    for csv_path in sorted(Path(data_dir).glob("*_test.csv")):
        subject = csv_path.stem.replace("_test", "").replace("_", " ")
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                question, A, B, C, D, answer = row[0], row[1], row[2], row[3], row[4], row[5]
                examples.append({
                    "subject": subject,
                    "question": question,
                    "A": A, "B": B, "C": C, "D": D,
                    "answer": answer.strip().upper(),
                })
    return examples


def load_gsm8k(data_path: str = "data/gsm8k/test.jsonl"):
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_alpaca_eval(data_path: str = "data/alpaca_eval/alpaca_eval.jsonl"):
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_sst(data_path: str = "data/simple_safety_tests/simple_safety_tests.csv"):
    examples = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(row)
    return examples


# -----------------------------------------------------------------------
# Evaluation tasks
# -----------------------------------------------------------------------

def evaluate_mmlu(llm, prompt_style: str, output_dir: Path):
    print("\n" + "="*60)
    print("MMLU Evaluation")
    print("="*60)

    examples = load_mmlu()
    print(f"Loaded {len(examples)} examples")

    prompts = []
    for ex in examples:
        q = MMLU_QUESTION_TEMPLATE.format(
            subject=ex["subject"],
            question=ex["question"],
            A=ex["A"], B=ex["B"], C=ex["C"], D=ex["D"],
        )
        prompts.append(format_prompt(q, prompt_style))

    stop_tokens = ["# Query:"] if prompt_style == "zero_shot" else ["###"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, stop=stop_tokens)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(prompts) / elapsed

    correct = 0
    failed_parse = 0
    results = []
    for ex, output in zip(examples, outputs):
        generated = output.outputs[0].text
        predicted = parse_mmlu_response(generated)
        gold = ex["answer"]
        is_correct = (predicted == gold) if predicted is not None else False
        if predicted is None:
            failed_parse += 1
        if is_correct:
            correct += 1
        results.append({**ex, "generated": generated, "predicted": predicted, "correct": is_correct})

    accuracy = correct / len(examples)
    print(f"Total: {len(examples)}, Correct: {correct}, Accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print(f"Failed to parse: {failed_parse}")
    print(f"Throughput: {throughput:.2f} examples/sec (elapsed {elapsed:.1f}s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "mmlu_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(examples),
            "failed_parse": failed_parse,
            "throughput_examples_per_sec": throughput,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {output_dir}/mmlu_results.json")
    return accuracy


def evaluate_gsm8k(llm, prompt_style: str, output_dir: Path):
    print("\n" + "="*60)
    print("GSM8K Evaluation")
    print("="*60)

    examples = load_gsm8k()
    print(f"Loaded {len(examples)} examples")

    prompts = []
    for ex in examples:
        instruction = ex["question"] + "\nAnswer:"
        prompts.append(format_prompt(instruction, prompt_style))

    stop_tokens = ["# Query:"] if prompt_style == "zero_shot" else ["###"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512, stop=stop_tokens)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(prompts) / elapsed

    correct = 0
    failed_parse = 0
    results = []
    for ex, output in zip(examples, outputs):
        generated = output.outputs[0].text
        predicted = parse_gsm8k_response(generated)
        gold = parse_gsm8k_gold(ex["answer"])
        is_correct = (predicted is not None and gold is not None and abs(predicted - gold) < 1e-3)
        if predicted is None:
            failed_parse += 1
        if is_correct:
            correct += 1
        results.append({
            "question": ex["question"],
            "gold_answer": ex["answer"],
            "gold_num": gold,
            "generated": generated,
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(examples)
    print(f"Total: {len(examples)}, Correct: {correct}, Accuracy: {accuracy:.4f} ({accuracy:.1%})")
    print(f"Failed to parse: {failed_parse}")
    print(f"Throughput: {throughput:.2f} examples/sec (elapsed {elapsed:.1f}s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gsm8k_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(examples),
            "failed_parse": failed_parse,
            "throughput_examples_per_sec": throughput,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {output_dir}/gsm8k_results.json")
    return accuracy


def evaluate_alpaca_eval(llm, prompt_style: str, model_name: str, output_dir: Path):
    print("\n" + "="*60)
    print("AlpacaEval Generation")
    print("="*60)

    examples = load_alpaca_eval()
    print(f"Loaded {len(examples)} examples")

    prompts = [format_prompt(ex["instruction"], prompt_style) for ex in examples]
    stop_tokens = ["# Query:"] if prompt_style == "zero_shot" else ["###"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=stop_tokens)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(prompts) / elapsed
    print(f"Throughput: {throughput:.2f} examples/sec (elapsed {elapsed:.1f}s)")

    results = []
    for ex, output in zip(examples, outputs):
        results.append({
            "instruction": ex["instruction"],
            "output": output.outputs[0].text.strip(),
            "generator": model_name,
            "dataset": ex.get("dataset", ""),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "alpaca_eval_outputs.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Outputs saved to {out_path}")
    print(f"Run winrate with:")
    print(f"  uv run alpaca_eval --model_outputs {out_path} --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn --base-dir .")
    return throughput


def evaluate_sst(llm, prompt_style: str, output_dir: Path):
    print("\n" + "="*60)
    print("SimpleSafetyTests Generation")
    print("="*60)

    examples = load_sst()
    print(f"Loaded {len(examples)} examples")

    prompts = [format_prompt(ex["prompts_final"], prompt_style) for ex in examples]
    stop_tokens = ["# Query:"] if prompt_style == "zero_shot" else ["###"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512, stop=stop_tokens)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    throughput = len(prompts) / elapsed
    print(f"Throughput: {throughput:.2f} examples/sec (elapsed {elapsed:.1f}s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sst_outputs.jsonl"
    with open(out_path, "w") as f:
        for ex, output in zip(examples, outputs):
            record = {**ex, "output": output.outputs[0].text.strip()}
            f.write(json.dumps(record) + "\n")
    print(f"Outputs saved to {out_path}")
    print(f"Run safety eval with:")
    print(f"  uv run python scripts/evaluate_safety.py --input-path {out_path} --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct --num-gpus 2 --output-path {output_dir}/sst_annotations.jsonl")
    return throughput


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU, GSM8K, AlpacaEval, SimpleSafetyTests")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--model_name", required=True, help="Name for outputs (e.g. 'base' or 'sft')")
    parser.add_argument("--prompt_style", choices=["zero_shot", "alpaca"], required=True,
                        help="zero_shot for base model, alpaca for SFT model")
    parser.add_argument("--tasks", nargs="+",
                        default=["mmlu", "gsm8k", "alpaca_eval", "sst"],
                        help="Which tasks to run")
    parser.add_argument("--output_dir", default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model_path}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Output dir: {output_dir}")
    print(f"Tasks: {args.tasks}")

    print("\nLoading model with vLLM...")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )

    summary = {}

    if "mmlu" in args.tasks:
        acc = evaluate_mmlu(llm, args.prompt_style, output_dir)
        summary["mmlu_accuracy"] = acc

    if "gsm8k" in args.tasks:
        acc = evaluate_gsm8k(llm, args.prompt_style, output_dir)
        summary["gsm8k_accuracy"] = acc

    if "alpaca_eval" in args.tasks:
        tp = evaluate_alpaca_eval(llm, args.prompt_style, args.model_name, output_dir)
        summary["alpaca_eval_throughput"] = tp

    if "sst" in args.tasks:
        tp = evaluate_sst(llm, args.prompt_style, output_dir)
        summary["sst_throughput"] = tp

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir}/summary.json")


if __name__ == "__main__":
    main()
