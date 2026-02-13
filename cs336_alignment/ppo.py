import torch
from torch import nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super(RewardModel, self).__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        reward_scores = self.reward_head(last_hidden_state[:, -1, :])  # 取最后一个 token 的 hidden state
        return reward_scores.squeeze(-1)

@staticmethod
def compute_loss(reward_chose,reward_reject):
    # 计算 loss，使用 hinge loss
    margin = 1.0
    loss = torch.mean(torch.clamp(margin - reward_chose + reward_reject, min=0))
    return loss

class ValueModel(nn.Module):
    def __init__(self,base_model):
        super.__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self,input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        value_scores = self.value_head(last_hidden_state[:, -1, :])  # 取最后一个 token 的 hidden state
        return value_scores.squeeze(-1)

def compute_gae(
    rewards: torch.Tensor,       # (B, T) 每个 token 的 reward
    values: torch.Tensor,        # (B, T) Value Model 的预测
    response_mask: torch.Tensor, # (B, T)
    gamma: float = 1.0,          # 折扣因子（LLM中通常=1）
    lam: float = 0.95,           # GAE 参数
):
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    
    # 从后往前递推: A_t = δ_t + γλ·A_{t+1}
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0  # 序列结束后没有未来了
        else:
            next_value = values[:, t + 1]
        
        # δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        
        # A_t = δ_t + γλ·A_{t+1}
        last_gae = delta + gamma * lam * last_gae
        last_gae = last_gae * response_mask[:, t]  # prompt 部分不算
        
        advantages[:, t] = last_gae
    
    # Returns = A + V（Value Model 的训练目标）
    returns = advantages + values
    
    return advantages, returns

def construct_rewards(
    rm_scores,
    policy_log_probs,
    ref_log_probs,
    response_mask,
    kl_coeff=0.1
):
    """
    r_t = -β·(log π - log π_ref)           for t < T
    r_T = R_RM - β·(log π - log π_ref)     for t = T
    """
    B, T = policy_log_probs.shape

    kl_penalty = kl_coeff * (policy_log_probs - ref_log_probs)  # (B, T)
    rewards = -kl_penalty * response_mask
    last_token_indices = response_mask.sum(dim=1).long() - 1  # (B,)
    rewards[torch.arange(B), last_token_indices] += rm_scores  # 只有最后   
    return rewards

def ppo_loss(
        new_log_probs:torch.Tensor,  # (B, T)
        old_log_probs:torch.Tensor,  # (B, T)
        advantages:torch.Tensor,     # (B, T)
        response_mask:torch.Tensor,  # (B, T)
        clip_eps:float=0.2
):
    ratios=torch.exp(new_log_probs - old_log_probs)  # (B, T)
    clipped_ratios=torch.clamp(ratios,1-clip_eps,1+clip_eps)  # (B, T)
    loss_terms=torch.min(ratios*advantages,clipped_ratios*advantages)  # (B, T)
    masked_loss=loss_terms*response_mask  # 只计算 response 部分的 loss
    sum_masked_loss=masked_loss.sum()  # (B,)
    return -sum_masked_loss / response_mask.sum()  # 平均每个 token 的 loss

