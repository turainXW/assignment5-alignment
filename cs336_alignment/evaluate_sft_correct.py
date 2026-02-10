#!/usr/bin/env python3
"""
SFT模型评估脚本 - 使用标准MATH格式
根据Assignment 5 Task 4.3要求
"""

import argparse
import json
import re
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def extract_boxed_answer(text: str) -> str:
    """从文本中提取\\boxed{}中的答案"""
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return ""


def normalize_answer(answer: str) -> str:
    """标准化答案用于比较"""
    # 移除空格、转小写
    answer = answer.lower().replace(" ", "").replace("\\,", "").replace("\\:", "")
    # 移除常见的LaTeX命令
    answer = answer.replace("\\text", "").replace("{", "").replace("}", "")
    return answer


def compare_answers(predicted: str, gold: str) -> bool:
    """比较预测答案和标准答案"""
    pred_answer = extract_boxed_answer(predicted)
    gold_answer = extract_boxed_answer(gold)

    if not pred_answer:
        return False

    pred_norm = normalize_answer(pred_answer)
    gold_norm = normalize_answer(gold_answer)

    return pred_norm == gold_norm


def load_eval_data(data_path: str, max_samples: int = None):
    """加载评估数据"""
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
                   max_samples: int = 500, max_tokens: int = 512,
                   device: str = "cuda:0", gpu_memory_utilization: float = 0.9):
    """
    评估SFT模型

    Args:
        model_path: 模型检查点路径
        eval_data_path: 评估数据路径（JSONL）
        max_samples: 最大评估样本数
        max_tokens: 每个样本最大生成token数
        device: CUDA设备
        gpu_memory_utilization: GPU显存利用率
    """
    print(f"\n{'='*70}")
    print(f"Evaluating model: {model_path}")
    print(f"{'='*70}\n")

    # 加载评估数据
    eval_examples = load_eval_data(eval_data_path, max_samples)

    # 提取prompts和标准答案
    prompts = [ex["prompt"] for ex in eval_examples]
    gold_solutions = [ex["response"] for ex in eval_examples]

    # 初始化vLLM
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

    # 设置采样参数（贪婪解码）
    sampling_params = SamplingParams(
        temperature=0.0,  # 贪婪解码，确保确定性结果
        top_p=1.0,
        max_tokens=max_tokens,
    )

    # 生成回答
    print(f"Generating responses for {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params)
    print("Generation complete!\n")

    # 评估回答
    print("Evaluating responses...")
    correct = 0
    has_boxed = 0
    results = []

    for i, (output, gold) in enumerate(zip(outputs, gold_solutions)):
        generated_text = output.outputs[0].text
        is_correct = compare_answers(generated_text, gold)

        pred_answer = extract_boxed_answer(generated_text)
        gold_answer = extract_boxed_answer(gold)

        if pred_answer:
            has_boxed += 1

        if is_correct:
            correct += 1

        results.append({
            'prompt': prompts[i],
            'generated': generated_text,
            'gold': gold,
            'predicted_answer': pred_answer,
            'gold_answer': gold_answer,
            'correct': is_correct,
        })

        # 每100个样本打印进度
        if (i + 1) % 100 == 0:
            curr_acc = correct / (i + 1)
            curr_format_acc = has_boxed / (i + 1)
            print(f"  Progress: {i+1}/{len(outputs)} | "
                  f"Accuracy: {curr_acc:.4f} ({correct}/{i+1}) | "
                  f"Has Answer: {curr_format_acc:.4f} ({has_boxed}/{i+1})")

    # 打印最终结果
    total = len(outputs)
    accuracy = correct / total if total > 0 else 0
    format_accuracy = has_boxed / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"Evaluation Results")
    print(f"{'='*70}")
    print(f"Total examples: {total}")
    print(f"Correct answers: {correct} ({accuracy:.2%})")
    print(f"Has boxed answer: {has_boxed} ({format_accuracy:.2%})")
    print(f"{'='*70}\n")

    return {
        'accuracy': accuracy,
        'format_accuracy': format_accuracy,
        'correct': correct,
        'has_boxed': has_boxed,
        'total': total,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model on MATH dataset")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--eval_data_path", type=str,
                        default="/root/assignment5-alignment/data/MATH/math_test.jsonl",
                        help="Path to evaluation data (JSONL)")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum tokens to generate per example")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="CUDA device to use for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save detailed results (JSON)")

    args = parser.parse_args()

    # 运行评估
    results = evaluate_model(
        model_path=args.model_path,
        eval_data_path=args.eval_data_path,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 保存详细结果
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
