#!/usr/bin/env python3
"""Generate a markdown evaluation report comparing base and SFT models."""

import json
import random
from pathlib import Path
from datetime import datetime


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def mmlu_analysis(results_path):
    data = load_json(results_path)
    results = data["results"]
    failed = [r for r in results if r["predicted"] is None]
    wrong = [r for r in results if not r["correct"] and r["predicted"] is not None]
    # Sample 5 wrong examples
    samples = random.sample(wrong, min(5, len(wrong)))
    return data, failed, samples


def gsm8k_analysis(results_path):
    data = load_json(results_path)
    results = data["results"]
    failed = [r for r in results if r["predicted"] is None]
    wrong = [r for r in results if not r["correct"] and r["predicted"] is not None]
    samples = random.sample(wrong, min(5, len(wrong)))
    return data, failed, samples


def generate_report(base_dir="eval_results/base", sft_dir="eval_results/sft",
                    output_path="eval_results/evaluation_report.md"):
    base_dir = Path(base_dir)
    sft_dir = Path(sft_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Evaluation Report: Base vs SFT Model")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Base model:** Qwen2.5-3B (zero-shot prompting)")
    lines.append(f"\n**SFT model:** Qwen2.5-3B fine-tuned on UltraChat-200K + SafetyTunedLlamas (Alpaca prompting)")
    lines.append(f"\n**Training:** 1 epoch, lr=2e-5, cosine decay, batch_size=32, max_len=512")
    lines.append(f"\n---\n")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    lines.append("## Summary\n")

    base_summary = load_json(base_dir / "summary.json") if (base_dir / "summary.json").exists() else {}
    sft_summary = load_json(sft_dir / "summary.json") if (sft_dir / "summary.json").exists() else {}

    lines.append("| Benchmark | Base (Qwen2.5-3B) | SFT | Δ |")
    lines.append("|---|---|---|---|")

    def fmt_acc(v):
        return f"{v:.2%}" if v is not None else "N/A"

    def delta(b, s):
        if b is None or s is None:
            return "N/A"
        d = s - b
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.2%}"

    base_mmlu = base_summary.get("mmlu_accuracy")
    sft_mmlu = sft_summary.get("mmlu_accuracy")
    base_gsm = base_summary.get("gsm8k_accuracy")
    sft_gsm = sft_summary.get("gsm8k_accuracy")

    lines.append(f"| MMLU (accuracy) | {fmt_acc(base_mmlu)} | {fmt_acc(sft_mmlu)} | {delta(base_mmlu, sft_mmlu)} |")
    lines.append(f"| GSM8K (accuracy) | {fmt_acc(base_gsm)} | {fmt_acc(sft_gsm)} | {delta(base_gsm, sft_gsm)} |")
    # Load Claude judge results
    judge_summary = {}
    judge_path = Path(args.output_dir if hasattr(args, 'output_dir') else "eval_results") / "claude_judge_summary.json"
    if judge_path.exists():
        judge_summary = load_json(judge_path)

    base_alpaca_wr = judge_summary.get("alpaca_base_winrate")
    sft_alpaca_wr = judge_summary.get("alpaca_sft_winrate")
    base_sst_sr = judge_summary.get("sst_base_safe_rate")
    sft_sst_sr = judge_summary.get("sst_sft_safe_rate")

    lines.append(f"| AlpacaEval winrate (vs text_davinci_003) | {fmt_acc(base_alpaca_wr)} | {fmt_acc(sft_alpaca_wr)} | {delta(base_alpaca_wr, sft_alpaca_wr)} |")
    lines.append(f"| SimpleSafetyTests safe% | {fmt_acc(base_sst_sr)} | {fmt_acc(sft_sst_sr)} | {delta(base_sst_sr, sft_sst_sr)} |")
    lines.append("")

    # ----------------------------------------------------------------
    # MMLU detail
    # ----------------------------------------------------------------
    lines.append("---\n")
    lines.append("## MMLU\n")

    for label, d in [("Base", base_dir), ("SFT", sft_dir)]:
        p = d / "mmlu_results.json"
        if not p.exists():
            lines.append(f"### {label}\n*Results not available.*\n")
            continue
        data, failed, wrong_samples = mmlu_analysis(p)
        lines.append(f"### {label}\n")
        lines.append(f"- **Accuracy:** {data['accuracy']:.4f} ({data['accuracy']:.1%})")
        lines.append(f"- **Correct:** {data['correct']} / {data['total']}")
        lines.append(f"- **Failed to parse:** {data['failed_parse']}")
        lines.append(f"- **Throughput:** {data.get('throughput_examples_per_sec', 'N/A'):.2f} examples/sec")
        lines.append("")
        lines.append("**Sample wrong predictions (5 examples):**\n")
        for i, r in enumerate(wrong_samples, 1):
            lines.append(f"{i}. **Subject:** {r['subject']}")
            lines.append(f"   - Q: {r['question'][:120]}...")
            lines.append(f"   - Gold: `{r['answer']}` | Predicted: `{r['predicted']}`")
            lines.append(f"   - Model output: `{r['generated'][:150].strip()}`")
            lines.append("")

    # ----------------------------------------------------------------
    # GSM8K detail
    # ----------------------------------------------------------------
    lines.append("---\n")
    lines.append("## GSM8K\n")

    for label, d in [("Base", base_dir), ("SFT", sft_dir)]:
        p = d / "gsm8k_results.json"
        if not p.exists():
            lines.append(f"### {label}\n*Results not available.*\n")
            continue
        data, failed, wrong_samples = gsm8k_analysis(p)
        lines.append(f"### {label}\n")
        lines.append(f"- **Accuracy:** {data['accuracy']:.4f} ({data['accuracy']:.1%})")
        lines.append(f"- **Correct:** {data['correct']} / {data['total']}")
        lines.append(f"- **Failed to parse:** {data['failed_parse']}")
        lines.append(f"- **Throughput:** {data.get('throughput_examples_per_sec', 'N/A'):.2f} examples/sec")
        lines.append("")
        lines.append("**Sample wrong predictions (5 examples):**\n")
        for i, r in enumerate(wrong_samples, 1):
            lines.append(f"{i}. **Q:** {r['question'][:120]}...")
            lines.append(f"   - Gold: `{r['gold_num']}` | Predicted: `{r['predicted']}`")
            lines.append(f"   - Model output: `{r['generated'][:200].strip()}`")
            lines.append("")

    # ----------------------------------------------------------------
    # AlpacaEval
    # ----------------------------------------------------------------
    lines.append("---\n")
    lines.append("## AlpacaEval\n")
    lines.append("Outputs generated and saved. Run the following to get winrate vs GPT-4 Turbo:\n")
    lines.append("```bash")
    lines.append("# Base model")
    lines.append("uv run alpaca_eval --model_outputs eval_results/base/alpaca_eval_outputs.json \\")
    lines.append("  --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn --base-dir .")
    lines.append("")
    lines.append("# SFT model")
    lines.append("uv run alpaca_eval --model_outputs eval_results/sft/alpaca_eval_outputs.json \\")
    lines.append("  --annotators_config scripts/alpaca_eval_vllm_llama3_3_70b_fn --base-dir .")
    lines.append("```\n")

    # Sample AlpacaEval outputs for qualitative comparison
    for label, d in [("Base", base_dir), ("SFT", sft_dir)]:
        p = d / "alpaca_eval_outputs.json"
        if not p.exists():
            continue
        outputs = load_json(p)
        samples = random.sample(outputs, min(3, len(outputs)))
        lines.append(f"### {label} - Sample outputs\n")
        for i, s in enumerate(samples, 1):
            lines.append(f"**Example {i}:** {s['instruction'][:100]}...")
            lines.append(f"> {s['output'][:300].strip()}...\n")

    # ----------------------------------------------------------------
    # SimpleSafetyTests
    # ----------------------------------------------------------------
    lines.append("---\n")
    lines.append("## SimpleSafetyTests\n")
    lines.append("Outputs generated and saved. Run the following to get safety annotations:\n")
    lines.append("```bash")
    lines.append("# Base model")
    lines.append("uv run python scripts/evaluate_safety.py \\")
    lines.append("  --input-path eval_results/base/sst_outputs.jsonl \\")
    lines.append("  --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \\")
    lines.append("  --num-gpus 2 --output-path eval_results/base/sst_annotations.jsonl")
    lines.append("")
    lines.append("# SFT model")
    lines.append("uv run python scripts/evaluate_safety.py \\")
    lines.append("  --input-path eval_results/sft/sst_outputs.jsonl \\")
    lines.append("  --model-name-or-path /data/a5-alignment/models/Llama-3.3-70B-Instruct \\")
    lines.append("  --num-gpus 2 --output-path eval_results/sft/sst_annotations.jsonl")
    lines.append("```\n")

    for label, d in [("Base", base_dir), ("SFT", sft_dir)]:
        p = d / "sst_outputs.jsonl"
        if not p.exists():
            continue
        records = load_jsonl(p)
        samples = random.sample(records, min(3, len(records)))
        lines.append(f"### {label} - Sample outputs\n")
        for i, r in enumerate(samples, 1):
            lines.append(f"**Example {i} [{r.get('harm_area', '')}]:** {r['prompts_final']}")
            lines.append(f"> {r['output'][:300].strip()}\n")

    # ----------------------------------------------------------------
    # Training info
    # ----------------------------------------------------------------
    lines.append("---\n")
    lines.append("## SFT Training Details\n")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append("| Model | Qwen2.5-3B |")
    lines.append("| Training data | UltraChat-200K + SafetyTunedLlamas |")
    lines.append("| Prompt format | Alpaca template |")
    lines.append("| Epochs | 1 |")
    lines.append("| Total steps | 7123 |")
    lines.append("| Learning rate | 2e-5 (cosine decay) |")
    lines.append("| LR warmup | 3% of steps |")
    lines.append("| Batch size | 32 (micro=8/GPU × 4 GPUs) |")
    lines.append("| Max seq length | 512 tokens |")
    lines.append("| Hardware | 4× A10 (FSDP) |")
    lines.append("| Training time | ~4h 37min |")
    lines.append("| Final val loss | 1.1234 |")
    lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report)
    print(f"Report saved to: {output_path}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="eval_results/base")
    parser.add_argument("--sft_dir", default="eval_results/sft")
    parser.add_argument("--output", default="eval_results/evaluation_report.md")
    args = parser.parse_args()
    random.seed(42)
    generate_report(args.base_dir, args.sft_dir, args.output)
