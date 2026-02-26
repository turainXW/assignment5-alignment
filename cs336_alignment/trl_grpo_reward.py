"""Reward function wrappers for TRL GRPO training."""

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def math_reward_func(prompts: list[str], completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """Reward function compatible with TRL GRPOTrainer.

    TRL calls this with:
      - prompts: list of prompt strings
      - completions: list of generated completion strings
      - ground_truth: extra dataset column passed as kwarg

    The prompt ends with '<think>', so the completion starts right after.
    We reconstruct the full response as '<think>' + completion and append
    '</answer>' if the model didn't generate it.
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # Reconstruct the full response that r1_zero_reward_fn expects
        response = "<think>" + completion
        if not response.rstrip().endswith("</answer>"):
            response = response.rstrip() + "</answer>"
        result = r1_zero_reward_fn(response, gt)
        rewards.append(result["reward"])
    return rewards
