import torch
from typing import Callable

def compute_group_normalized_rewards(
        reward_fn:Callable[[str,str],dict[str,float]],
        rollout_responses:list[str],
        repeated_ground_truths:list[str],
        group_size:int,
        advantage_eps:float,
        normalize_by_std:bool=True,
)->tuple[torch.Tensor,torch.Tensor,dict[str,float]]:
    
    rewards=[]
    for response,truth in zip(rollout_responses,repeated_ground_truths):
        reward=reward_fn(response,truth)["reward"]
        rewards.append(reward)
    rewards_tensor=torch.tensor(rewards,dtype=torch.float32)
    
    rewards_group=rewards_tensor.reshape(-1,group_size)
    rewards_mean=rewards_group.mean(dim=1,keepdim=True)
    rewards_std=rewards_group.std(dim=1,keepdim=True)
    rewards_normalized=rewards_group-rewards_mean
    if normalize_by_std:
        rewards_normalized/=(rewards_std+advantage_eps)
    
    rewards_normalized_flat=rewards_normalized.flatten()

    reward_stats={ 
                'mean_reward': rewards_mean.mean().item(),
                'std_reward': rewards_std.mean().item(),
            
    }
    return rewards_normalized_flat, rewards_tensor, reward_stats

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages:torch.Tensor,
        policy_log_probs:torch.Tensor,
)->torch.Tensor:
    loss=-(raw_rewards_or_advantages*policy_log_probs)
    return loss.mean()

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratios=torch.exp(policy_log_probs-old_log_probs)
    clipped_ratios=torch.clamp(ratios,1-cliprange,1+cliprange)
    loss=-torch.minimum(advantages*clipped_ratios,advantages*ratios)
    stats={
        'mean_ratio': ratios.mean().item(),
        'mean_clipped_ratio': clipped_ratios.mean().item(),
    }
    return loss, stats