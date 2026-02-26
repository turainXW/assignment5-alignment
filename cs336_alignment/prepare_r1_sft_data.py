#!/usr/bin/env python3
"""Convert MATH JSONL data to R1-Zero format for SFT training.

Input format (existing):
  {"prompt": "Question: ...\n\nAnswer:", "response": "... $\\boxed{5}$ ..."}

Output format (R1):
  {"prompt": "A conversation between User and Assistant...User: ...\nAssistant: <think>",
   "response": "\n...\n</think> <answer> 5 </answer>"}
"""

import argparse
import json
import re
from pathlib import Path

from cs336_alignment.drgrpo_grader import extract_boxed_answer
from cs336_alignment.trl_grpo_data import PROMPT_TEMPLATE


def _extract_question(prompt_field: str) -> str:
    """Strip 'Question: ' prefix and '\\n\\nAnswer:' suffix."""
    q = prompt_field.strip()
    if q.startswith("Question: "):
        q = q[len("Question: "):]
    if q.endswith("\n\nAnswer:"):
        q = q[: -len("\n\nAnswer:")]
    return q.strip()


def _clean_reasoning(response: str) -> str:
    """Remove the \\boxed{...} from the reasoning text and clean up."""
    # Remove \boxed{...} patterns (including nested braces)
    cleaned = re.sub(r"\$?\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\$?", "", response)
    # Clean trailing punctuation artifacts
    cleaned = cleaned.strip()
    # Remove trailing "The answer is" type phrases left over
    cleaned = re.sub(r"\s*The answer is\s*\.?\s*$", "", cleaned)
    return cleaned.strip()


def convert_example(example: dict) -> dict | None:
    """Convert a single MATH example to R1-Zero format."""
    question = _extract_question(example["prompt"])
    answer = extract_boxed_answer(example["response"])
    if answer is None:
        return None

    prompt = PROMPT_TEMPLATE.format(question=question)
    # The response starts right after <think> in the prompt
    # Include the original reasoning, then close think and give answer
    reasoning = _clean_reasoning(example["response"])
    response = f"\n{reasoning}\n</think> <answer> {answer} </answer>"

    return {"prompt": prompt, "response": response}


def convert_file(input_path: str, output_path: str) -> tuple[int, int]:
    """Convert an entire JSONL file. Returns (converted, skipped) counts."""
    converted = 0
    skipped = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            example = json.loads(line)
            result = convert_example(example)
            if result is None:
                skipped += 1
                continue
            fout.write(json.dumps(result) + "\n")
            converted += 1
    return converted, skipped


def main():
    parser = argparse.ArgumentParser(description="Convert MATH data to R1-Zero SFT format")
    parser.add_argument("--train_input", type=str, default="data/MATH/math_train.jsonl")
    parser.add_argument("--test_input", type=str, default="data/MATH/math_test.jsonl")
    parser.add_argument("--train_output", type=str, default="data/MATH/math_train_r1.jsonl")
    parser.add_argument("--test_output", type=str, default="data/MATH/math_test_r1.jsonl")
    args = parser.parse_args()

    for input_path, output_path in [
        (args.train_input, args.train_output),
        (args.test_input, args.test_output),
    ]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        converted, skipped = convert_file(input_path, output_path)
        print(f"{input_path} -> {output_path}: {converted} converted, {skipped} skipped")

    # Print a sample for verification
    print("\n--- Sample output ---")
    with open(args.train_output) as f:
        sample = json.loads(f.readline())
    print(f"Prompt (last 100 chars): ...{sample['prompt'][-100:]}")
    print(f"Response (first 200 chars): {sample['response'][:200]}")
    print(f"Response (last 100 chars): ...{sample['response'][-100:]}")


if __name__ == "__main__":
    main()
