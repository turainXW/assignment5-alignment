#!/usr/bin/env python3
"""Generate a detailed qualitative comparison report with sampled outputs."""

import json
import random
from pathlib import Path
from datetime import datetime

random.seed(42)

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

def trunc(s, n=400):
    s = str(s).strip()
    return s[:n] + ("…" if len(s) > n else "")

def verdict_emoji(v):
    return {"win": "✅ Win", "loss": "❌ Loss", "tie": "🟡 Tie"}.get(v, v)

def safe_emoji(v):
    return "✅ SAFE" if v == "safe" else "❌ UNSAFE"

BASE = Path("eval_results/base")
SFT  = Path("eval_results/sft")

lines = []

lines += [
    "# Qualitative Output Comparison Report",
    f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "\n**Models compared:**",
    "- **Base**: Qwen2.5-3B, zero-shot prompting",
    "- **SFT**: Qwen2.5-3B fine-tuned on UltraChat-200K + SafetyTunedLlamas, Alpaca prompting",
    "\n**Judge**: Claude Sonnet 4.6 (AlpacaEval & SST); rule-based (MMLU & GSM8K)",
    "\n---\n",
]

# ====================================================================
# 1. METRIC SUMMARY
# ====================================================================
lines += [
    "## 1. Overall Metrics\n",
    "| Benchmark | Base | SFT | Δ |",
    "|---|---|---|---|",
    "| MMLU accuracy | 58.78% | 61.95% | +3.17% ✅ |",
    "| GSM8K accuracy | 65.43% | 13.50% | −51.93% ❌ |",
    "| AlpacaEval winrate | 44.72% | 17.76% | −26.96% ❌ |",
    "| SST safe rate | 74.00% | 36.00% | −38.00% ❌ |",
    "",
]

# ====================================================================
# 2. MMLU SAMPLES
# ====================================================================
lines += ["---\n", "## 2. MMLU — Sample Comparisons\n",
          "> Task: multiple-choice knowledge Q&A. Format: answer `A/B/C/D`.\n"]

base_mmlu = load_json(BASE / "mmlu_results.json")["results"]
sft_mmlu  = load_json(SFT  / "mmlu_results.json")["results"]

# Build lookup by question
sft_mmlu_map = {r["question"]: r for r in sft_mmlu}

# Case 1: both correct
both_correct = [r for r in base_mmlu if r["correct"] and sft_mmlu_map.get(r["question"], {}).get("correct")]
# Case 2: base correct, SFT wrong
base_only = [r for r in base_mmlu if r["correct"] and not sft_mmlu_map.get(r["question"], {}).get("correct")]
# Case 3: SFT correct, base wrong
sft_only  = [r for r in base_mmlu if not r["correct"] and sft_mmlu_map.get(r["question"], {}).get("correct")]
# Case 4: both wrong
both_wrong = [r for r in base_mmlu if not r["correct"] and not sft_mmlu_map.get(r["question"], {}).get("correct")]

for title, pool, n in [
    ("Both correct", both_correct, 2),
    ("Base correct, SFT wrong", base_only, 3),
    ("SFT correct, Base wrong", sft_only, 3),
    ("Both wrong", both_wrong, 2),
]:
    samples = random.sample(pool, min(n, len(pool)))
    lines.append(f"### 2.{['Both correct','Base correct, SFT wrong','SFT correct, Base wrong','Both wrong'].index(title)+1} {title} ({len(pool)} cases)\n")
    for i, r in enumerate(samples, 1):
        q = r["question"]
        sr = sft_mmlu_map.get(q, {})
        lines += [
            f"**Example {i}** — Subject: *{r['subject']}*",
            f"> Q: {trunc(q, 200)}",
            f"> Options: A. {r['A']}  B. {r['B']}  C. {r['C']}  D. {r['D']}",
            f"> **Gold answer: {r['answer']}**",
            "",
            f"| | Predicted | Output |",
            f"|---|---|---|",
            f"| Base | `{r.get('predicted','?')}` {'✅' if r['correct'] else '❌'} | {trunc(r['generated'], 120)} |",
            f"| SFT  | `{sr.get('predicted','?')}` {'✅' if sr.get('correct') else '❌'} | {trunc(sr.get('generated',''), 120)} |",
            "",
        ]

# ====================================================================
# 3. GSM8K SAMPLES
# ====================================================================
lines += ["---\n", "## 3. GSM8K — Sample Comparisons\n",
          "> Task: math word problems. Parse last number as answer.\n"]

base_gsm = load_json(BASE / "gsm8k_results.json")["results"]
sft_gsm  = load_json(SFT  / "gsm8k_results.json")["results"]
sft_gsm_map = {r["question"]: r for r in sft_gsm}

base_gsm_correct = [r for r in base_gsm if r["correct"]]
base_gsm_wrong   = [r for r in base_gsm if not r["correct"]]
sft_failed_parse = [r for r in sft_gsm if r["predicted"] is None]

# Base correct, SFT format fail
base_ok_sft_fail = [r for r in base_gsm if r["correct"] and sft_gsm_map.get(r["question"], {}).get("predicted") is None]
# Both correct
both_gsm_correct = [r for r in base_gsm if r["correct"] and sft_gsm_map.get(r["question"], {}).get("correct")]
# Both wrong
both_gsm_wrong = [r for r in base_gsm if not r["correct"] and not sft_gsm_map.get(r["question"], {}).get("correct")]

for title, pool, n in [
    ("Base correct, SFT failed to parse (format regression)", base_ok_sft_fail, 3),
    ("Both correct", both_gsm_correct, 2),
    ("Both wrong", both_gsm_wrong, 2),
]:
    samples = random.sample(pool, min(n, len(pool)))
    lines.append(f"### {title} ({len(pool)} cases)\n")
    for i, r in enumerate(samples, 1):
        q = r["question"]
        sr = sft_gsm_map.get(q, {})
        lines += [
            f"**Example {i}**",
            f"> Q: {trunc(q, 200)}",
            f"> **Gold: `{r['gold_num']}`**",
            "",
            f"| | Predicted | Full output |",
            f"|---|---|---|",
            f"| Base | `{r.get('predicted','?')}` {'✅' if r['correct'] else '❌'} | {trunc(r['generated'], 200)} |",
            f"| SFT  | `{sr.get('predicted','?')}` {'✅' if sr.get('correct') else '❌'} | {trunc(sr.get('generated',''), 200)} |",
            "",
        ]

# ====================================================================
# 4. ALPACAEVAL SAMPLES
# ====================================================================
lines += ["---\n", "## 4. AlpacaEval — Sample Comparisons\n",
          "> Judged by Claude Sonnet 4.6 vs text_davinci_003 reference.\n"]

base_alpaca = load_json(BASE / "alpaca_eval_claude_judge.json")["results"]
sft_alpaca  = load_json(SFT  / "alpaca_eval_claude_judge.json")["results"]

# Build lookup by instruction
base_alpaca_map = {r["instruction"]: r for r in base_alpaca}
sft_alpaca_map  = {r["instruction"]: r for r in sft_alpaca}

all_instructions = list(base_alpaca_map.keys())

# Categories
base_win_sft_win   = [i for i in all_instructions if base_alpaca_map[i]["verdict"]=="win" and sft_alpaca_map.get(i,{}).get("verdict")=="win"]
base_win_sft_loss  = [i for i in all_instructions if base_alpaca_map[i]["verdict"]=="win" and sft_alpaca_map.get(i,{}).get("verdict")=="loss"]
base_loss_sft_win  = [i for i in all_instructions if base_alpaca_map[i]["verdict"]=="loss" and sft_alpaca_map.get(i,{}).get("verdict")=="win"]
base_loss_sft_loss = [i for i in all_instructions if base_alpaca_map[i]["verdict"]=="loss" and sft_alpaca_map.get(i,{}).get("verdict")=="loss"]

for title, pool, n in [
    ("Base wins, SFT also wins (both beat reference)", base_win_sft_win, 2),
    ("Base wins, SFT loses (SFT regression)", base_win_sft_loss, 3),
    ("Base loses, SFT wins (SFT improvement)", base_loss_sft_win, 3),
    ("Both lose to reference", base_loss_sft_loss, 2),
]:
    samples = random.sample(pool, min(n, len(pool)))
    lines.append(f"### {title} ({len(pool)} cases)\n")
    for i, instr in enumerate(samples, 1):
        br = base_alpaca_map[instr]
        sr = sft_alpaca_map.get(instr, {})
        lines += [
            f"**Example {i}**",
            f"> Instruction: *{trunc(instr, 150)}*",
            "",
            f"**Reference (text_davinci_003):**",
            f"> {trunc(br['reference_output'], 250)}",
            "",
            f"**Base model** — {verdict_emoji(br['verdict'])}:",
            f"> {trunc(br['model_output'], 250)}",
            "",
            f"**SFT model** — {verdict_emoji(sr.get('verdict','?'))}:",
            f"> {trunc(sr.get('model_output','N/A'), 250)}",
            "",
        ]

# ====================================================================
# 5. SIMPLESAFETYTESTS SAMPLES
# ====================================================================
lines += ["---\n", "## 5. SimpleSafetyTests — Sample Comparisons\n",
          "> Judged by Claude Sonnet 4.6 as SAFE/UNSAFE.\n"]

base_sst = load_json(BASE / "sst_claude_judge.json")["results"]
sft_sst  = load_json(SFT  / "sst_claude_judge.json")["results"]
sft_sst_map = {r["prompts_final"]: r for r in sft_sst}

# Categories
base_safe_sft_unsafe = [r for r in base_sst if r["safety_label"]=="safe" and sft_sst_map.get(r["prompts_final"],{}).get("safety_label")=="unsafe"]
base_unsafe_sft_safe = [r for r in base_sst if r["safety_label"]=="unsafe" and sft_sst_map.get(r["prompts_final"],{}).get("safety_label")=="safe"]
both_safe   = [r for r in base_sst if r["safety_label"]=="safe" and sft_sst_map.get(r["prompts_final"],{}).get("safety_label")=="safe"]
both_unsafe = [r for r in base_sst if r["safety_label"]=="unsafe" and sft_sst_map.get(r["prompts_final"],{}).get("safety_label")=="unsafe"]

for title, pool, n in [
    ("Base SAFE, SFT UNSAFE — SFT safety regression", base_safe_sft_unsafe, 4),
    ("Base UNSAFE, SFT SAFE — SFT improvement", base_unsafe_sft_safe, 2),
    ("Both SAFE", both_safe, 2),
    ("Both UNSAFE", both_unsafe, 2),
]:
    samples = random.sample(pool, min(n, len(pool)))
    lines.append(f"### {title} ({len(pool)} cases)\n")
    for i, r in enumerate(samples, 1):
        p = r["prompts_final"]
        sr = sft_sst_map.get(p, {})
        lines += [
            f"**Example {i}** — Harm area: *{r.get('harm_area','')}*",
            f"> Prompt: **{p}**",
            "",
            f"**Base model** — {safe_emoji(r['safety_label'])}:",
            f"> {trunc(r['output'], 300)}",
            f"> *Judge: {r['safety_verdict_raw']}*",
            "",
            f"**SFT model** — {safe_emoji(sr.get('safety_label','?'))}:",
            f"> {trunc(sr.get('output','N/A'), 300)}",
            f"> *Judge: {sr.get('safety_verdict_raw','')}*",
            "",
        ]

# ====================================================================
# 6. KEY FINDINGS
# ====================================================================
lines += [
    "---\n",
    "## 6. Key Findings\n",
    "### MMLU",
    "- SFT 提升 +3.17%，格式遵循更好（parse 失败 21→6）",
    "- SFT 错误模式：会给出更长的解释，但有时过度推理导致答案错误",
    "",
    "### GSM8K",
    "- SFT 严重退化（65.4%→13.5%），根本原因：**565/1319 条无法解析**",
    "- SFT 模型学会了「解释推理过程」但不给出简洁数字结果",
    "- 实际数学推理能力未必完全丧失，主要是输出格式问题",
    "- 修复方向：在 GSM8K 格式数据上继续微调，或改变 parse 策略",
    "",
    "### AlpacaEval",
    "- 基模 winrate 44.7%（尚可），SFT 后骤降至 17.8%",
    "- SFT 后 Alpaca 模板限制了输出风格，模型倾向于结构化列表而非自然对话",
    "- SFT 模型在事实类问题上偶有明显错误（如 AK47 类型错误）",
    "- 修复方向：使用更多高质量对话数据，DPO 调整偏好",
    "",
    "### SimpleSafetyTests",
    "- 基模 safe rate 74%，SFT 后降至 36%",
    "- 最典型问题：SFT 模型会「先给出危险信息，再加免责声明」",
    "- SafetyTunedLlamas 数据量不足，覆盖场景有限",
    "- **这正是需要 DPO/RLHF 的核心原因**：通过偏好对齐直接惩罚不安全输出",
    "",
]

report = "\n".join(lines)
out = Path("eval_results/qualitative_comparison_report.md")
out.write_text(report)
print(f"Report saved: {out}")
print(f"Lines: {len(lines)}, Size: {len(report)/1024:.1f} KB")
