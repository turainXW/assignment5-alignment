"""Dataset loader for TRL GRPO training on MATH data."""

import json

from datasets import Dataset

from cs336_alignment.drgrpo_grader import extract_boxed_answer


PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with "
    "the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within "
    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>.\n"
    "User: {question}\n"
    "Assistant: <think>"
)


def _extract_question(prompt_field: str) -> str:
    """Strip 'Question: ' prefix and '\\n\\nAnswer:' suffix from the raw prompt."""
    q = prompt_field.strip()
    if q.startswith("Question: "):
        q = q[len("Question: "):]
    if q.endswith("\n\nAnswer:"):
        q = q[: -len("\n\nAnswer:")]
    return q.strip()


def _extract_ground_truth(response_field: str) -> str:
    """Extract answer from <answer> tags (R1 format) or \\boxed{} (standard format)."""
    if "<answer>" in response_field and "</answer>" in response_field:
        return response_field.split("<answer>")[-1].replace("</answer>", "").strip()
    boxed = extract_boxed_answer(response_field)
    return boxed if boxed else response_field


def _is_r1_format(prompt_field: str) -> bool:
    """Check if the prompt is already in R1 format (contains 'Assistant: <think>')."""
    return "Assistant: <think>" in prompt_field


def load_math_dataset(path: str) -> Dataset:
    """Load a MATH JSONL file and return an HF Dataset for TRL GRPO.

    Returns Dataset with columns: ["prompt", "ground_truth"]
    - prompt: formatted R1-Zero prompt string
    - ground_truth: extracted answer for reward computation
    """
    records = []
    with open(path) as f:
        for line in f:
            example = json.loads(line)
            ground_truth = _extract_ground_truth(example["response"])
            if _is_r1_format(example["prompt"]):
                # R1 format: prompt already includes system instruction + "Assistant: <think>"
                prompt = example["prompt"]
            else:
                # Standard format: wrap with PROMPT_TEMPLATE
                question = _extract_question(example["prompt"])
                prompt = PROMPT_TEMPLATE.format(question=question)
            records.append({"prompt": prompt, "ground_truth": ground_truth})

    return Dataset.from_list(records)
